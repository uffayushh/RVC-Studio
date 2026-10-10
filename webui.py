YES = "Yes"
NO = "No"

import sys
if sys.platform == "win32":
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
        _orig_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost

        def _silenced_call_connection_lost(self, exc):
            try:
                _orig_call_connection_lost(self, exc)
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
                pass

        _ProactorBasePipeTransport._call_connection_lost = _silenced_call_connection_lost
    except Exception:
        pass

import os
import shutil
import html
import re

# Offline WebUI keeps the CUDA Graph implementation available, but remains
# eager by default. Set RVC_OFFLINE_CUDA_GRAPH=1 to opt in for benchmarking or
# controlled deployments.
_offline_cuda_graph = os.environ.get("RVC_OFFLINE_CUDA_GRAPH", "0") == "1"
os.environ["RVC_CUDA_GRAPH"] = "1" if _offline_cuda_graph else "0"

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("no_proxy", "localhost, 127.0.0.1, ::1")
os.environ.setdefault("weight_root", "assets/weights")
os.environ.setdefault("weight_pymss_root", "assets/pymss_weights")
os.environ.setdefault("index_root", "logs")
os.environ.setdefault("outside_index_root", "assets/indices")
os.environ.setdefault("rmvpe_root", "assets/rmvpe")

now_dir = os.getcwd()
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
for name in os.listdir(tmp):
    if name == "jieba.cache":
        continue
    path = os.path.join(tmp, name)
    delete = (
        os.remove if os.path.isfile(path) or os.path.islink(path) else shutil.rmtree
    )
    try:
        delete(path)
    except Exception as error:
        print(str(error))

from configs.config import Config, GPU_INDEX, GPU_INFOS, GPU_MEMORY, IS_GPU
from infer.vc.modules import VC
from tools.pymss_webui import (
    PYMSS_MODEL_CHOICES,
    get_model_info,
    pymss_separate as _pymss_separate_core,
    stop_pymss_separation as _stop_pymss_separation_core,
)
from tools.file_io import read_text
from tools.process_utils import kill_process_tree
from train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    show_info,
)
import torch, platform
import numpy as np
import gradio as gr
from urllib.parse import quote
import pathlib
import json
from time import sleep
from subprocess import Popen
from random import shuffle
import warnings
import traceback
import threading
import logging
import socket
import subprocess
import time


logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def find_available_port(start_port, host="0.0.0.0"):
    """Return the first bindable TCP port at or above ``start_port``."""
    if not 1 <= start_port <= 65535:
        raise ValueError(f"Port must be between 1 and 65535, got {start_port}.")

    for port in range(start_port, 65536):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((host, port))
            return port
        except OSError:
            continue

    raise OSError(
        f"No available TCP port from {start_port} through 65535; WebUI was not started."
    )


def is_gradio_port_in_use_error(error, port):
    """Recognize Gradio's explicit-port conflict without hiding other launch errors."""
    return str(error).startswith(f"Port {port} is in use.")


def launch_webui_with_port_fallback(app, config):
    """Launch Gradio, increasing the requested port until startup succeeds."""
    next_port = config.listen_port
    queued_app = app.queue(concurrency_count=511, max_size=1022)
    while True:
        config.listen_port = find_available_port(next_port)
        if config.listen_port != next_port:
            logger.warning(
                "Port %s is occupied; trying port %s instead.",
                next_port,
                config.listen_port,
            )
        try:
            queued_app.launch(
                server_name="0.0.0.0",
                inbrowser=not config.noautoopen,
                server_port=config.listen_port,
                quiet=True,
            )
            return config.listen_port
        except OSError as error:
            if not is_gradio_port_in_use_error(error, config.listen_port):
                raise
            if config.listen_port == 65535:
                raise OSError(
                    "No available TCP port through 65535; WebUI was not started."
                ) from error
            logger.warning(
                "Port %s became occupied while Gradio was starting; trying the next port.",
                config.listen_port,
            )
            next_port = config.listen_port + 1

runtime_dirs = (
    os.path.join(now_dir, "logs"),
    os.environ["weight_root"],
    os.environ["weight_pymss_root"],
    os.environ["index_root"],
    os.environ["outside_index_root"],
    os.environ["rmvpe_root"],
    os.path.join(now_dir, "assets", "hubert_base"),
    os.path.join(now_dir, "assets", "pretrained"),
    os.path.join(now_dir, "assets", "pretrained_v2"),
)
for runtime_dir in runtime_dirs:
    os.makedirs(runtime_dir, exist_ok=True)
warnings.filterwarnings("ignore")
torch.manual_seed(114514)


config = Config()
vc = VC(config)


print(
    "Current device: %s | Inference precision: %s" % (config.device, config.dtype),
    flush=True,
)
# GPU filtering and precision rules are shared with inference/extraction/training.
gpu_infos = list(GPU_INFOS)
gpu_indices = sorted(GPU_INDEX)
if_gpu_ok = IS_GPU
if if_gpu_ok:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = max(1, int(min(GPU_MEMORY[i] for i in gpu_indices)) // 2)
else:
    gpu_info = "Unfortunately, no compatible GPU was detected for training."
    default_batch_size = 1
gpus = "-".join(str(i) for i in gpu_indices)


weight_root = os.getenv("weight_root")
weight_pymss_root = os.getenv("weight_pymss_root")
outside_index_root = os.getenv("outside_index_root")

def weight_names():
    return sorted(
        name for name in os.listdir(weight_root) if name.endswith(".pth")
    )


def refresh_weight_choices(previous_names=None, force=False):
    current_names = tuple(weight_names())
    if force or current_names != previous_names:
        return current_names, change_choices()
    return current_names, {"__type__": "update"}


names = weight_names()
pymss_names = PYMSS_MODEL_CHOICES


def change_choices():
    return {"choices": weight_names(), "__type__": "update"}


def clean():
    return {"value": "", "__type__": "update"}

def download_yt_audio(url):
    url = (url or "").strip()
    if not url:
        return None, "Please provide a valid YouTube / audio URL."
    os.makedirs("opt", exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(now_dir, "opt", "%(title).50s-%(id)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "flac",
            "preferredquality": "0",
        }],
        "quiet": True,
        "ffmpeg_location": now_dir,
    }
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            flac_path = os.path.splitext(filename)[0] + ".flac"
            mp3_path = os.path.splitext(filename)[0] + ".mp3"
            if os.path.exists(flac_path):
                return flac_path, "Downloaded successfully: %s" % os.path.basename(flac_path)
            elif os.path.exists(mp3_path):
                return mp3_path, "Downloaded successfully: %s" % os.path.basename(mp3_path)
            elif os.path.exists(filename):
                return filename, "Downloaded successfully: %s" % os.path.basename(filename)
            else:
                return None, "Download finished but target file could not be located."
    except Exception as e:
        return None, "Download failed: %s" % str(e)


sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


TRAIN_TASK_LOCK = threading.Lock()
TRAIN_TASK = None


def button_update(value=None, variant=None, visible=None):
    update = {"__type__": "update"}
    if value is not None:
        update["value"] = value
    if variant is not None:
        update["variant"] = variant
    if visible is not None:
        update["visible"] = visible
    return update


def render_pymss_progress(percent=0, label="Waiting to start", state="idle"):
    percent = max(0.0, min(100.0, float(percent or 0)))
    colors = {
        "idle": "#64748b",
        "running": "#2563eb",
        "done": "#15803d",
        "stopped": "#b45309",
        "failed": "#b91c1c",
    }
    color = colors.get(state, colors["running"])
    safe_label = html.escape(str(label or ""))
    return (
        '<div style="min-height:52px;padding:6px 0;">'
        '<div style="display:flex;justify-content:space-between;gap:12px;'
        'align-items:center;margin-bottom:7px;font-size:14px;line-height:20px;">'
        '<span style="overflow-wrap:anywhere;">%s</span>'
        '<strong style="flex:0 0 auto;color:%s;">%.1f%%</strong>'
        "</div>"
        '<div role="progressbar" aria-valuemin="0" aria-valuemax="100" '
        'aria-valuenow="%.1f" style="height:10px;width:100%%;overflow:hidden;'
        'border-radius:2px;background:#e2e8f0;">'
        '<div style="height:100%%;width:%.3f%%;background:%s;"></div>'
        "</div></div>"
    ) % (safe_label, color, percent, percent, percent, color)


def run_pymss_separation(
    model_name,
    inp_root,
    save_root_vocal,
    paths,
    save_root_ins,
    format0,
):
    progress_state = {
        "percent": 0.0,
        "label": "Preparing separation task",
        "state": "running",
    }
    busy = False

    def update_progress(event):
        nonlocal busy
        event_type = event.get("event")
        file_count = max(1, int(event.get("file_count") or 1))
        file_index = max(1, int(event.get("file_index") or 1))
        message = str(event.get("message") or "")

        if event_type == "progress":
            done = max(0.0, float(event.get("done") or 0))
            total = max(1.0, float(event.get("total") or 1))
            file_fraction = min(1.0, done / total)
            progress_state["percent"] = (
                (file_index - 1 + file_fraction) / file_count * 100
            )
            progress_state["label"] = "File %s/%s: %s (%.0f of %.0f s)" % (
                file_index,
                file_count,
                message or "Processing audio",
                done,
                total,
            )
            progress_state["state"] = "running"
        elif event_type == "file_start":
            progress_state["percent"] = (file_index - 1) / file_count * 100
            progress_state["label"] = message
            progress_state["state"] = "running"
        elif event_type == "file":
            progress_state["percent"] = file_index / file_count * 100
            progress_state["label"] = (
                message.splitlines()[0] if message else "File finished"
            )
            progress_state["state"] = "running" if event.get("ok") else "failed"
        elif event_type == "done":
            successful = int(event.get("successful") or 0)
            failed = int(event.get("failed") or 0)
            progress_state["percent"] = 100.0
            progress_state["label"] = "Separation finished: %s succeeded, %s failed" % (
                successful,
                failed,
            )
            progress_state["state"] = "done" if failed == 0 else "failed"
        elif event_type == "retry_fp32":
            progress_state["percent"] = 0.0
            progress_state["label"] = message
            progress_state["state"] = "running"
        elif event_type == "cancelled":
            progress_state["label"] = message or "Separation stopped"
            progress_state["state"] = "stopped"
        elif event_type in {"fatal", "busy"}:
            progress_state["label"] = (
                message.splitlines()[0] if message else "Separation failed"
            )
            progress_state["state"] = "failed"
            busy = event_type == "busy"
        elif event_type in {"preparing", "precision_attempt", "status"}:
            progress_state["label"] = message
            progress_state["state"] = "running"

    last_info = ""
    try:
        for info in _pymss_separate_core(
            model_name,
            inp_root,
            save_root_vocal,
            paths,
            save_root_ins,
            format0,
            event_callback=update_progress,
        ):
            last_info = info
            start_button = button_update()
            stop_button = button_update()
            if not busy:
                start_button = button_update(visible=False)
                stop_button = button_update(visible=True)
            yield (
                info,
                render_pymss_progress(**progress_state),
                start_button,
                stop_button,
            )
    except Exception:
        last_info = "Failed\n%s" % traceback.format_exc()
        progress_state.update(
            {"label": "Separation failed", "state": "failed"}
        )
        logger.exception("PyMSS WebUI task failed")

    start_button = button_update()
    stop_button = button_update()
    if not busy:
        start_button = button_update(visible=True)
        stop_button = button_update(visible=False)
    yield (last_info, render_pymss_progress(**progress_state), start_button, stop_button)


def stop_pymss_webui():
    return (
        _stop_pymss_separation_core(),
        render_pymss_progress(0, "Separation stopped", "stopped"),
        button_update(visible=True),
        button_update(visible=False),
    )


def format_status(title, state, detail=""):
    lines = ["[%s]" % title, "Status: %s" % state]
    if detail:
        lines.extend(["", detail.strip()])
    return "\n".join(lines)


def format_workflow_status(step, detail="", completed_steps=None, state="Running"):
    completed_steps = completed_steps or []
    detail = str(detail).strip()
    lines = []
    if completed_steps:
        lines.append("Completed stages:")
        lines.extend(
            "[x] %s: Success" % completed_step
            for completed_step in completed_steps
        )
    if step:
        if lines:
            lines.append("")
        lines.append("Current stage: %s" % step)
    if detail:
        lines.extend(["", detail])
    return format_status(
        "One-click Training",
        state,
        "\n".join(lines),
    )


def read_log(path, max_lines=40):
    try:
        lines = [line.rstrip() for line in read_text(path, errors="ignore").splitlines()]
        lines = [line for line in lines if line.strip()]
        if len(lines) > max_lines:
            tail_count = max(0, max_lines - 1)
            omitted = len(lines) - tail_count
            tail = lines[-tail_count:] if tail_count else []
            lines = ["...omitted earlier %s lines, displaying latest output" % omitted]
            lines.extend(tail)
        return "\n".join(lines)
    except FileNotFoundError:
        return ""


def artifact_names(directory, suffix):
    if not os.path.isdir(directory):
        return set()
    return {
        name.split(".")[0]
        for name in os.listdir(directory)
        if name.lower().endswith(suffix)
    }


def validate_preprocess_outputs(exp_dir):
    exp_path = os.path.join(now_dir, "logs", exp_dir)
    gt_names = artifact_names(os.path.join(exp_path, "0_gt_wavs"), ".wav")
    wav16_names = artifact_names(os.path.join(exp_path, "1_16k_wavs"), ".wav")
    if not gt_names:
        raise RuntimeError("Dataset slicing did not produce valid training audio. Please check dataset and logs.")
    if not wav16_names:
        raise RuntimeError("Dataset slicing did not generate 16kHz audio. Subsequent extraction stopped.")
    if not gt_names & wav16_names:
        raise RuntimeError("Dataset slicing output files do not match. Subsequent extraction stopped.")


def validate_feature_outputs(exp_dir, version, if_f0):
    exp_path = os.path.join(now_dir, "logs", exp_dir)
    wav16_names = artifact_names(os.path.join(exp_path, "1_16k_wavs"), ".wav")
    feature_name = "3_feature256" if version == "v1" else "3_feature768"
    feature_names = artifact_names(os.path.join(exp_path, feature_name), ".npy")
    matched = wav16_names & feature_names
    if not feature_names or not matched:
        raise RuntimeError("HuBERT feature extraction did not generate valid outputs. Training stopped.")
    if if_f0:
        f0_names = artifact_names(os.path.join(exp_path, "2a_f0"), ".npy")
        f0nsf_names = artifact_names(os.path.join(exp_path, "2b-f0nsf"), ".npy")
        matched &= f0_names & f0nsf_names
        if not f0_names or not f0nsf_names or not matched:
            raise RuntimeError("F0 pitch extraction did not generate valid outputs. Training stopped.")
    return matched


def begin_train_task(name):
    global TRAIN_TASK
    with TRAIN_TASK_LOCK:
        if TRAIN_TASK is None:
            state = {
                "name": name,
                "processes": [],
                "stop_requested": False,
            }
            TRAIN_TASK = state
            return "start", state
        return "busy", TRAIN_TASK


def stop_train_task(name):
    with TRAIN_TASK_LOCK:
        if TRAIN_TASK is None:
            return (
                format_status(name, "Not running"),
                button_update(visible=True),
                button_update(visible=False),
            )
        if TRAIN_TASK["name"] != name:
            return (
                format_status(
                    name,
                    "Cannot stop",
                    "%s is running, please stop it first" % TRAIN_TASK["name"],
                ),
                button_update(),
                button_update(),
            )
        state = TRAIN_TASK
        state["stop_requested"] = True
        processes = list(state["processes"])
    for process in processes:
        kill_process_tree(process, name, logger)
    return (
        format_status(name, "Stopped"),
        button_update(visible=True),
        button_update(visible=False),
    )


def finish_train_task(state):
    global TRAIN_TASK
    with TRAIN_TASK_LOCK:
        if TRAIN_TASK is state:
            TRAIN_TASK = None


def train_task_stopped(state):
    with TRAIN_TASK_LOCK:
        return state["stop_requested"]


def start_train_process(state, cmd):
    kwargs = {"shell": True, "cwd": now_dir}
    if "train/train.py" in cmd.replace("\\", "/"):
        training_env = os.environ.copy()
        training_env["RVC_CUDA_GRAPH"] = "0"
        kwargs["env"] = training_env
    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    logger.info("Executing command: %s", cmd)
    process = Popen(cmd, **kwargs)
    with TRAIN_TASK_LOCK:
        state["processes"].append(process)
        stopped = state["stop_requested"]
    if stopped:
        kill_process_tree(process, state["name"], logger)
    return process


def wait_train_processes(
    state,
    processes,
    log_path=None,
    title="Task",
    format_output=True,
    watch_weights=False,
):
    last_snapshot = None
    last_emit_time = 0
    last_weight_names = tuple(weight_names()) if watch_weights else ()
    while any(process.poll() is None for process in processes):
        if train_task_stopped(state):
            for process in processes:
                kill_process_tree(process, state["name"], logger)
            break
        if log_path:
            snapshot = read_log(log_path)
            current_time = time.monotonic()
            current_weight_names = tuple(weight_names()) if watch_weights else ()
            weights_changed = watch_weights and current_weight_names != last_weight_names
            if (
                snapshot != last_snapshot
                or weights_changed
                or current_time - last_emit_time >= 5
            ):
                yield (
                    format_status(title, "Running", snapshot)
                    if format_output
                    else snapshot
                )
                last_snapshot = snapshot
                last_emit_time = current_time
                last_weight_names = current_weight_names
        sleep(1)
    with TRAIN_TASK_LOCK:
        for process in processes:
            if process in state["processes"]:
                state["processes"].remove(process)
    if log_path:
        final_state = "Stopped" if train_task_stopped(state) else "Finishing"
        snapshot = read_log(log_path)
        yield (
            format_status(title, final_state, snapshot)
            if format_output
            else snapshot
        )
    if not train_task_stopped(state):
        failed = [process.returncode for process in processes if process.returncode != 0]
        if failed:
            raise RuntimeError("Subprocess execution failed with exit code: %s" % failed)


def run_preprocess_dataset(trainset_dir, exp_dir, sr, n_p, state, format_output=True):
    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    log_path = "%s/logs/%s/preprocess.log" % (now_dir, exp_dir)
    with open(log_path, "w", encoding="utf8"):
        pass
    cmd = '"%s" train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f' % (
        config.python_cmd,
        trainset_dir,
        sr,
        n_p,
        now_dir,
        exp_dir,
        config.noparallel,
        config.preprocess_per,
    )
    extract_start_time = time.time()
    requested_workers = max(int(n_p), 1)
    actual_workers = 1 if config.noparallel else requested_workers
    print(
        "Data extraction started: start_time=%.6f, requested_parallel=%s, actual_parallel_limit=%s"
        % (extract_start_time, requested_workers, actual_workers),
        flush=True,
    )
    try:
        process = start_train_process(state, cmd)
        yield from wait_train_processes(
            state, [process], log_path, "Dataset Slicing", format_output
        )
    finally:
        extract_end_time = time.time()
        print(
            "Data extraction ended: end_time=%.6f, total_elapsed=%.3fs"
            % (extract_end_time, extract_end_time - extract_start_time),
            flush=True,
        )
    if not train_task_stopped(state):
        validate_preprocess_outputs(exp_dir)


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    action, state = begin_train_task("Dataset Slicing")
    if action == "busy":
        yield (
            format_status(
                "Dataset Slicing",
                "Waiting",
                "%s is running, please stop it first" % state["name"],
            ),
            button_update(),
            button_update(),
        )
        return
    final_info = None
    try:
        yield (
            format_status("Dataset Slicing", "Starting"),
            button_update(visible=False),
            button_update(visible=True),
        )
        for info in run_preprocess_dataset(trainset_dir, exp_dir, sr, n_p, state):
            yield info, button_update(visible=False), button_update(visible=True)
        if train_task_stopped(state):
            final_info = format_status("Dataset Slicing", "Stopped")
    except Exception:
        final_info = format_status("Dataset Slicing", "Failed", traceback.format_exc())
    finally:
        finish_train_task(state)
    if final_info is None:
        final_info = format_status("Dataset Slicing", "Completed")
    yield final_info, button_update(visible=True), button_update(visible=False)


def stop_preprocess_dataset():
    return stop_train_task("Dataset Slicing")


# but2.click(extract_f0,[gpus6,np7,f0method8,if_f0_3,trainset_dir4],[info2])
def run_extract_f0_feature(
    gpus,
    n_p,
    f0method,
    if_f0,
    exp_dir,
    version19,
    gpus_rmvpe,
    state,
    format_output=True,
):
    if f0method != "rmvpe":
        raise ValueError("Only rmvpe pitch extraction algorithm is supported")
    log_path = "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir)
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    validate_preprocess_outputs(exp_dir)
    with open(log_path, "w", encoding="utf8"):
        pass

    if if_f0:
        processes = []
        rmvpe_devices = [gpu for gpu in gpus_rmvpe.split("-") if gpu != ""]
        if (
            not rmvpe_devices and not config.dml
        ):
            cmd = (
                '"%s" train/dataset/extract_f0.py cpu "%s/logs/%s" %s %s'
                % (config.python_cmd, now_dir, exp_dir, n_p, f0method)
            )
            processes.append(start_train_process(state, cmd))
        elif rmvpe_devices:
            count = len(rmvpe_devices)
            for index, gpu in enumerate(rmvpe_devices):
                cmd = (
                    '"%s" train/dataset/extract_f0.py cuda %s %s %s "%s/logs/%s" %s'
                    % (
                        config.python_cmd,
                        count,
                        index,
                        gpu,
                        now_dir,
                        exp_dir,
                        config.is_half,
                    )
                )
                processes.append(start_train_process(state, cmd))
        else:
            cmd = (
                '"%s" train/dataset/extract_f0.py dml "%s/logs/%s"'
                % (config.python_cmd, now_dir, exp_dir)
            )
            processes.append(start_train_process(state, cmd))
        yield from wait_train_processes(
            state, processes, log_path, "F0 Extraction", format_output
        )
        if train_task_stopped(state):
            return

        with open(log_path, "w", encoding="utf8"):
            pass

    feature_gpus = [gpu for gpu in gpus.split("-") if gpu != ""]
    processes = []
    if feature_gpus:
        count = len(feature_gpus)
        for index, gpu in enumerate(feature_gpus):
            cmd = (
                '"%s" train/dataset/extract_hubert_feature.py %s %s %s %s "%s/logs/%s" %s %s'
                % (
                    config.python_cmd,
                    config.device,
                    count,
                    index,
                    gpu,
                    now_dir,
                    exp_dir,
                    version19,
                    config.is_half,
                )
            )
            processes.append(start_train_process(state, cmd))
    else:
        cmd = (
            '"%s" train/dataset/extract_hubert_feature.py %s 1 0 "%s/logs/%s" %s %s'
            % (
                config.python_cmd,
                config.device,
                now_dir,
                exp_dir,
                version19,
                config.is_half,
            )
        )
        processes.append(start_train_process(state, cmd))
    yield from wait_train_processes(
        state, processes, log_path, "HuBERT Features", format_output
    )
    if not train_task_stopped(state):
        validate_feature_outputs(exp_dir, version19, if_f0)


def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe):
    action, state = begin_train_task("Feature Extraction")
    if action == "busy":
        yield (
            format_status(
                "Feature Extraction",
                "Waiting",
                "%s is running, please stop it first" % state["name"],
            ),
            button_update(),
            button_update(),
        )
        return
    final_info = None
    try:
        yield (
            format_status("Feature Extraction", "Starting"),
            button_update(visible=False),
            button_update(visible=True),
        )
        for info in run_extract_f0_feature(
            gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe, state
        ):
            yield info, button_update(visible=False), button_update(visible=True)
        if train_task_stopped(state):
            final_info = format_status("Feature Extraction", "Stopped")
    except Exception:
        final_info = format_status("Feature Extraction", "Failed", traceback.format_exc())
    finally:
        finish_train_task(state)
    if final_info is None:
        final_info = format_status("Feature Extraction", "Completed")
    yield final_info, button_update(visible=True), button_update(visible=False)


def stop_extract_f0_feature():
    return stop_train_task("Feature Extraction")

def get_pretrained_models(path_str, f0_str, sr2):
    if_pretrained_generator_exist = os.access(
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        logger.warning(
            "Pretrained generator model not found, skipping: assets/pretrained%s/%sG%s.pth",
            path_str,
            f0_str,
            sr2,
        )
    if not if_pretrained_discriminator_exist:
        logger.warning(
            "Pretrained discriminator model not found, skipping: assets/pretrained%s/%sD%s.pth",
            path_str,
            f0_str,
            sr2,
        )
    return (
        (
            "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)
            if if_pretrained_generator_exist
            else ""
        ),
        (
            "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)
            if if_pretrained_discriminator_exist
            else ""
        ),
    )


def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)


def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    to_return_sr2 = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["40k", "48k", "32k"], "__type__": "update", "value": sr2}
    )
    f0_str = "f0" if if_f0_3 else ""
    return (
        *get_pretrained_models(path_str, f0_str, sr2),
        to_return_sr2,
    )


def change_f0(if_f0_3, sr2, version19):  # f0method8,pretrained_G14,pretrained_D15
    path_str = "" if version19 == "v1" else "_v2"
    return (
        {"visible": if_f0_3, "__type__": "update"},
        {"visible": if_f0_3, "__type__": "update"},
        *get_pretrained_models(path_str, "f0" if if_f0_3 == True else "", sr2),
    )


# but3.click(click_train,[exp_dir1,sr2,if_f0_3,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16])
def run_train_model(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    state,
    format_output=True,
):
    # Generate filelist
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    if not names:
        raise RuntimeError("No valid audio available for training. Please complete dataset slicing and feature extraction first.")
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w", encoding="utf8") as f:
        f.write("\n".join(opt))
    logger.debug("Training file list written successfully")
    # Generate config (not needed)
    # cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 4 -g 0 -te 10 -se 5 -pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth -l 1 -c 0"
    logger.info("Using GPU: %s", str(gpus16))
    if pretrained_G14 == "":
        logger.info("Pretrained generator model not used")
    if pretrained_D15 == "":
        logger.info("Pretrained discriminator model not used")
    if version19 == "v1" or sr2 == "40k":
        config_path = "v1/%s.json" % sr2
    else:
        config_path = "v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf8") as f:
            json.dump(
                config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    if gpus16:
        cmd = (
            '"%s" train/train.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
            % (
                config.python_cmd,
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 in (YES, "1", 1, True) else 0,
                1 if if_cache_gpu17 in (YES, "1", 1, True) else 0,
                1 if if_save_every_weights18 in (YES, "1", 1, True) else 0,
                version19,
            )
        )
    else:
        cmd = (
            '"%s" train/train.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
            % (
                config.python_cmd,
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 in (YES, "1", 1, True) else 0,
                1 if if_cache_gpu17 in (YES, "1", 1, True) else 0,
                1 if if_save_every_weights18 in (YES, "1", 1, True) else 0,
                version19,
            )
        )
    logger.info("Executing command: %s", cmd)
    process = start_train_process(state, cmd)
    yield from wait_train_processes(
        state,
        [process],
        os.path.join(exp_dir, "train.log"),
        "Model Training",
        format_output,
        True,
    )


def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    known_models = tuple(weight_names())
    action, state = begin_train_task("Model Training")
    if action == "busy":
        yield (
            format_status(
                "Model Training",
                "Waiting",
                "%s is running, please stop it first" % state["name"],
            ),
            button_update(),
            button_update(),
            button_update(),
        )
        return
    final_info = None
    try:
        yield (
            format_status("Model Training", "Starting"),
            button_update(visible=False),
            button_update(visible=True),
            button_update(),
        )
        for info in run_train_model(
            exp_dir1,
            sr2,
            if_f0_3,
            spk_id5,
            save_epoch10,
            total_epoch11,
            batch_size12,
            if_save_latest13,
            pretrained_G14,
            pretrained_D15,
            gpus16,
            if_cache_gpu17,
            if_save_every_weights18,
            version19,
            state,
        ):
            known_models, model_update = refresh_weight_choices(known_models)
            yield (
                info,
                button_update(visible=False),
                button_update(visible=True),
                model_update,
            )
        if train_task_stopped(state):
            final_info = format_status("Model Training", "Stopped")
    except Exception:
        final_info = format_status("Model Training", "Failed", traceback.format_exc())
    finally:
        finish_train_task(state)
    if final_info is None:
        final_info = format_status("Model Training", "Completed")
    model_update = change_choices()
    yield (
        final_info,
        button_update(visible=True),
        button_update(visible=False),
        model_update,
    )


def stop_train_model():
    return stop_train_task("Model Training")


# but4.click(train_index, [exp_dir1], info3)
def run_train_index(exp_dir1, version19, state, format_output=True):
    exp_dir = os.path.join(now_dir, "logs", exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    log_path = os.path.join(exp_dir, "train_index.log")
    with open(log_path, "w", encoding="utf8"):
        pass
    cmd = (
        '"%s" train/train_index.py "%s" %s "%s" %s'
        % (
            config.python_cmd,
            exp_dir1,
            version19,
            outside_index_root,
            config.n_cpu,
        )
    )
    process = start_train_process(state, cmd)
    yield from wait_train_processes(
        state, [process], log_path, "Index Training", format_output
    )


def train_index(exp_dir1, version19):
    action, state = begin_train_task("Index Training")
    if action == "busy":
        yield (
            format_status(
                "Index Training",
                "Waiting",
                "%s is running, please stop it first" % state["name"],
            ),
            button_update(),
            button_update(),
        )
        return
    final_info = None
    try:
        yield (
            format_status("Index Training", "Starting"),
            button_update(visible=False),
            button_update(visible=True),
        )
        for info in run_train_index(exp_dir1, version19, state):
            yield info, button_update(visible=False), button_update(visible=True)
        if train_task_stopped(state):
            final_info = format_status("Index Training", "Stopped")
    except Exception:
        final_info = format_status("Index Training", "Failed", traceback.format_exc())
    finally:
        finish_train_task(state)
    if final_info is None:
        final_info = format_status("Index Training", "Completed")
    yield final_info, button_update(visible=True), button_update(visible=False)


def stop_train_index():
    return stop_train_task("Index Training")

# but5.click(train1key, [exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, gpus6, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17], info3)
def train1key(
    exp_dir1,
    sr2,
    if_f0_3,
    trainset_dir4,
    spk_id5,
    np7,
    f0method8,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    gpus_rmvpe,
):
    known_models = tuple(weight_names())
    action, state = begin_train_task("One-click Training")
    if action == "busy":
        yield (
            format_status(
                "One-click Training",
                "Waiting",
                "%s is running, please stop it first" % state["name"],
            ),
            button_update(),
            button_update(),
            button_update(),
        )
        return

    completed_steps = []
    step = ""
    final_info = None
    running = True
    start_button = button_update(visible=False)
    stop_button = button_update(visible=True)
    try:
        yield (
            format_status("One-click Training", "Starting"),
            start_button,
            stop_button,
            button_update(),
        )

        step = "Dataset Slicing"
        yield (
            format_workflow_status(step, completed_steps=completed_steps),
            start_button,
            stop_button,
            button_update(),
        )
        for info in run_preprocess_dataset(
            trainset_dir4, exp_dir1, sr2, np7, state, False
        ):
            yield (
                format_workflow_status(step, info, completed_steps),
                start_button,
                stop_button,
                button_update(),
            )
        running = not train_task_stopped(state)
        if running:
            completed_steps.append(step)

        if running:
            step = "F0 and HuBERT Feature Extraction"
            yield (
                format_workflow_status(step, completed_steps=completed_steps),
                start_button,
                stop_button,
                button_update(),
            )
            for info in run_extract_f0_feature(
                gpus16,
                np7,
                f0method8,
                if_f0_3,
                exp_dir1,
                version19,
                gpus_rmvpe,
                state,
                False,
            ):
                yield (
                    format_workflow_status(step, info, completed_steps),
                    start_button,
                    stop_button,
                    button_update(),
                )
            running = not train_task_stopped(state)
            if running:
                completed_steps.append(step)

        if running:
            step = "Model Training"
            yield (
                format_workflow_status(step, completed_steps=completed_steps),
                start_button,
                stop_button,
                button_update(),
            )
            for info in run_train_model(
                exp_dir1,
                sr2,
                if_f0_3,
                spk_id5,
                save_epoch10,
                total_epoch11,
                batch_size12,
                if_save_latest13,
                pretrained_G14,
                pretrained_D15,
                gpus16,
                if_cache_gpu17,
                if_save_every_weights18,
                version19,
                state,
                False,
            ):
                known_models, model_update = refresh_weight_choices(known_models)
                yield (
                    format_workflow_status(step, info, completed_steps),
                    start_button,
                    stop_button,
                    model_update,
                )
            if not train_task_stopped(state):
                yield (
                    format_workflow_status(step, completed_steps=completed_steps),
                    start_button,
                    stop_button,
                    change_choices(),
                )
            running = not train_task_stopped(state)
            if running:
                completed_steps.append(step)

        if running:
            step = "Index Training"
            yield (
                format_workflow_status(step, completed_steps=completed_steps),
                start_button,
                stop_button,
                button_update(),
            )
            for info in run_train_index(exp_dir1, version19, state, False):
                yield (
                    format_workflow_status(step, info, completed_steps),
                    start_button,
                    stop_button,
                    button_update(),
                )
            running = not train_task_stopped(state)
            if running:
                completed_steps.append(step)

        if not running:
            final_info = format_workflow_status(
                step, completed_steps=completed_steps, state="Stopped"
            )
        else:
            final_info = format_workflow_status(
                "", completed_steps=completed_steps, state="Completed"
            )
    except Exception:
        final_info = format_workflow_status(
            step,
            traceback.format_exc(),
            completed_steps,
            "Failed",
        )
    finally:
        finish_train_task(state)
    model_update = change_choices()
    yield (
        final_info,
        button_update(visible=True),
        button_update(visible=False),
        model_update,
    )


def stop_train1key():
    return stop_train_task("One-click Training")

#                    ckpt_path2.change(change_info_,[ckpt_path2],[sr__,if_f0__])
def change_info_(ckpt_path):
    if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path), "train.log")):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        info = eval(
            read_text(
                ckpt_path.replace(os.path.basename(ckpt_path), "train.log")
            )
            .strip("\n")
            .split("\n")[0]
            .split("\t")[-1]
        )
        sr, f0 = info["sample_rate"], info["if_f0"]
        version = "v2" if ("version" in info and info["version"] == "v2") else "v1"
        return sr, str(f0), version
    except Exception:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}


F0GPUVisible = IS_GPU

# ---------------------------------------------------------------------------
# RVC Studio interface
# ---------------------------------------------------------------------------

STUDIO_NAME = "RVC Studio 2026"


def _icon_url(inner):
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
        'stroke-linejoin="round">%s</svg>' % inner
    )
    return 'url("data:image/svg+xml;utf8,%s")' % quote(svg)


TAB_ICONS = [
    # Voice conversion: smooth waveform
    '<path d="M3 12h2c1.5 0 2-5 3.5-5s2 10 3.5 10 2-8 3.5-8 1.5 3 3 3h2.5"/>',
    # Vocal extraction: layered leaves
    '<path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 12l10 5 10-5"/><path d="M2 17l10 5 10-5"/>',
    # Voice training: rounded chip
    '<rect x="4" y="4" width="16" height="16" rx="3"/><rect x="9" y="9" width="6" height="6" rx="1"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3"/>',
    # Training monitor: chart with dots
    '<path d="M3 3v18h18"/><path d="M7 14l4-4 3 3 6-7"/><circle cx="7" cy="14" r="1.5" fill="currentColor"/><circle cx="11" cy="10" r="1.5" fill="currentColor"/><circle cx="14" cy="13" r="1.5" fill="currentColor"/><circle cx="20" cy="6" r="1.5" fill="currentColor"/>',


    # API: rounded code brackets
    '<path d="M8 5L3 12l5 7"/><path d="M16 5l5 7-5 7"/>',
    # About: info circle
    '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
]

STUDIO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

:root, .dark {
    --rs-bg: #0c0d14;
    --rs-surface: #131520;
    --rs-surface-2: #181b29;
    --rs-surface-3: #202436;
    --rs-ink: #f0f2fa;
    --rs-muted: #8b92b0;
    --rs-line: #25283c;
    --rs-line-bright: #3c4263;
    --rs-accent-purple: #6366f1;
    --rs-accent-indigo: #4f46e5;
    --rs-accent-cyan: #38bdf8;
    --rs-accent-yellow: #ffd33d;
    --rs-accent-red: #ef4444;
    --rs-pixel-font: 'Press Start 2P', monospace;
    --rs-body-font: 'Inter', system-ui, -apple-system, sans-serif;
    --rs-code-font: 'JetBrains Mono', 'Cascadia Code', monospace;

    --body-background-fill: var(--rs-bg);
    --body-text-color: var(--rs-ink);
    --body-text-color-subdued: var(--rs-muted);
    --background-fill-primary: var(--rs-surface);
    --background-fill-secondary: var(--rs-surface-2);
    --border-color-primary: var(--rs-line);
    --block-background-fill: var(--rs-surface);
    --block-border-color: var(--rs-line);
    --block-border-width: 1px;
    --block-shadow: 3px 3px 0px #06070a;
    --block-label-background-fill: var(--rs-surface-2);
    --block-label-text-color: var(--rs-muted);
    --block-title-text-color: var(--rs-ink);
    --block-radius: 4px;
    --container-radius: 4px;
    --input-radius: 4px;
    --input-background-fill: #0d0f17;
    --input-border-color: var(--rs-line);
    --input-shadow: none;
    --button-large-radius: 4px;
    --button-small-radius: 4px;
    --checkbox-border-radius: 2px;
    --checkbox-label-border-radius: 4px;
    --checkbox-label-background-fill: var(--rs-surface);
    --checkbox-label-background-fill-hover: var(--rs-surface-2);
    --checkbox-label-background-fill-selected: var(--rs-surface-3);
    --checkbox-label-border-color: var(--rs-line);
    --checkbox-label-border-color-selected: var(--rs-accent-yellow);
    --checkbox-label-text-color: var(--rs-ink);
    --checkbox-label-text-color-selected: var(--rs-accent-yellow);
    --slider-color: var(--rs-accent-yellow);
    --color-accent: var(--rs-accent-purple);
}

body, .gradio-container {
    font-family: var(--rs-body-font) !important;
    background-color: #0c0d14 !important;
    background-image: 
        radial-gradient(circle at 50% 240px, rgba(99, 102, 241, 0.16) 0%, rgba(79, 70, 229, 0.08) 36%, transparent 68%),
        radial-gradient(#22263b 1.2px, transparent 1.2px) !important;
    background-size: 100% 700px, 16px 16px !important;
    background-repeat: no-repeat, repeat !important;
    color: var(--rs-ink) !important;
}

.gradio-container {
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding: 0 0 32px 0 !important;
}
footer { display: none !important; }

/* -- Retro Pixel Top Header Bar -- */
.rs-top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 24px;
    background: #0f111a;
    border-bottom: 1px solid rgba(255, 255, 255, 0.12);
    font-family: var(--rs-pixel-font);
    box-sizing: border-box;
}
.rs-top-brand {
    font-size: 11px;
    color: #ffffff;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.rs-top-center {
    display: flex;
    align-items: center;
    gap: 16px;
}
.rs-score-label {
    font-size: 11px;
    color: #ffffff;
    letter-spacing: 1px;
}
.rs-hearts {
    display: flex;
    align-items: center;
    gap: 6px;
}
.pixel-heart {
    font-size: 12px;
    filter: drop-shadow(0 0 4px rgba(255, 211, 61, 0.5));
}
.rs-top-right {
    display: flex;
    align-items: center;
}
.rs-menu-lines {
    display: flex;
    flex-direction: column;
    gap: 4px;
    cursor: pointer;
    padding: 4px;
}
.menu-line {
    display: block;
    width: 24px;
    height: 2px;
    background: #ffffff;
}

/* -- Pixel Art Showcase Hero Section -- */
.rs-pixel-hero {
    position: relative;
    padding: 38px 32px 24px 32px;
    margin: 0 0 20px 0;
    background: 
        radial-gradient(ellipse at 50% 50%, rgba(99, 102, 241, 0.18) 0%, rgba(46, 36, 92, 0.28) 45%, transparent 72%),
        radial-gradient(#272c44 1.4px, transparent 1.4px);
    background-size: 100% 100%, 14px 14px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 4px;
    overflow: hidden;
}

.rs-hero-main-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: relative;
    z-index: 2;
    gap: 24px;
}

.rs-hero-left {
    position: relative;
    flex: 1;
}

/* Transform Box around LEVEL UP */
.pixel-transform-box {
    display: inline-block;
    position: relative;
    padding: 10px 18px;
    border: 1px dashed rgba(255, 255, 255, 0.65);
    background: rgba(15, 17, 26, 0.7);
    margin-bottom: 8px;
}
.pixel-box-content {
    font-family: var(--rs-pixel-font);
    font-size: 26px;
    color: #ffffff;
    letter-spacing: 2px;
    text-shadow: 2px 2px 0px #000000;
}
.t-handle {
    position: absolute;
    width: 7px;
    height: 7px;
    background: #ffffff;
    border: 1px solid #000000;
}
.th-tl { top: -4px; left: -4px; }
.th-tc { top: -4px; left: calc(50% - 3.5px); }
.th-tr { top: -4px; right: -4px; }
.th-ml { top: calc(50% - 3.5px); left: -4px; }
.th-mr { top: calc(50% - 3.5px); right: -4px; }
.th-bl { bottom: -4px; left: -4px; }
.th-bc { bottom: -4px; left: calc(50% - 3.5px); }
.th-br { bottom: -4px; right: -4px; }

/* Floating Pixel Coin */
.pixel-coin-float {
    position: absolute;
    left: -32px;
    top: 60px;
    animation: coinBounce 2.5s ease-in-out infinite alternate;
}
@keyframes coinBounce {
    0% { transform: translateY(0); }
    100% { transform: translateY(-8px); }
}

/* Huge Bold Headline */
.pixel-huge-headline {
    font-family: var(--rs-body-font);
    font-size: 46px;
    font-weight: 900;
    line-height: 1.05;
    color: #ffffff;
    letter-spacing: -1.5px;
    margin-top: 6px;
    text-transform: uppercase;
}

/* 3D Isometric Pixel Cursor */
.rs-hero-center-cursor {
    flex: 0 0 auto;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 0 20px;
}
.pixel-3d-cursor {
    display: block;
    filter: drop-shadow(0 6px 16px rgba(0, 0, 0, 0.6));
}

/* Hero Right Side Info */
.rs-hero-right-info {
    flex: 0 0 280px;
    font-family: var(--rs-pixel-font);
    font-size: 10px;
    line-height: 1.8;
    color: #c0c6e0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    background: rgba(17, 20, 31, 0.8);
    border: 1px solid #282c42;
    padding: 14px;
    border-radius: 2px;
}

/* Hero Footer Row */
.rs-hero-footer-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 28px;
    padding-top: 14px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    font-family: var(--rs-pixel-font);
}
.rs-hero-scroll {
    font-size: 10px;
    color: #8b92b0;
    letter-spacing: 1px;
}
.rs-hero-file-badge {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 10px;
    color: #ffffff;
    letter-spacing: 1px;
}

/* Studio Stats Row */
.rs-stats-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}
.rs-stat-box {
    background: #141624;
    border: 1px solid #25283c;
    border-radius: 4px;
    padding: 12px 16px;
    box-shadow: 2px 2px 0px #06070a;
}
.stat-lbl {
    display: block;
    font-family: var(--rs-pixel-font);
    font-size: 9px;
    color: #8b92b0;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.stat-val {
    display: block;
    font-family: var(--rs-pixel-font);
    font-size: 12px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.5px;
}

/* -- Tabs: Retro Pixel Arcade Navigation -- */
#main-tabs { margin-top: 0; }
#main-tabs > .tab-nav {
    background: #0f111a !important;
    border-bottom: 2px solid #25283c !important;
    padding: 0 12px !important;
    gap: 6px !important;
}
.tab-nav button {
    font-family: var(--rs-pixel-font) !important;
    font-size: 10px !important;
    letter-spacing: 0.5px !important;
    color: #8b92b0 !important;
    border: 1px solid transparent !important;
    border-bottom: 0 !important;
    background: #141624 !important;
    padding: 12px 16px !important;
    border-radius: 4px 4px 0 0 !important;
    transition: all 0.15s ease !important;
}
.tab-nav button:hover {
    color: #ffffff !important;
    background: #1a1d2e !important;
}
.tab-nav button.selected {
    color: var(--rs-accent-yellow) !important;
    background: #1e2238 !important;
    border-color: #383e5c !important;
    border-bottom: 3px solid var(--rs-accent-yellow) !important;
    box-shadow: 0 -2px 10px rgba(255, 211, 61, 0.12) !important;
}
#main-tabs > .tab-nav button::before {
    content: "";
    display: inline-block;
    width: 14px;
    height: 14px;
    margin-right: 8px;
    vertical-align: -2px;
    background-color: currentColor;
    -webkit-mask: var(--ic) center / contain no-repeat;
    mask: var(--ic) center / contain no-repeat;
}
__TAB_ICON_RULES__
.tabitem { background: transparent !important; border: 0 !important; padding: 20px 0 0 0 !important; }

/* -- Cards: Dark Slate with Pixel Accent Lines -- */
.rs-card {
    background: #141624 !important;
    border: 1px solid #25283c !important;
    border-radius: 4px !important;
    padding: 18px 20px !important;
    gap: 14px !important;
    box-shadow: 3px 3px 0px #06070a !important;
    position: relative !important;
}
.rs-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: #393e60;
}
.rs-card.accent-amber::before { background: #ffd33d; }
.rs-card.accent-teal::before  { background: #38bdf8; }
.rs-card.accent-blue::before  { background: #6366f1; }
.rs-card.accent-red::before   { background: #ef4444; }
.rs-card.accent-navy::before  { background: #818cf8; }

.rs-sec { display: flex; gap: 14px; align-items: flex-start; }
.rs-sec h3 {
    margin: 0 0 4px 0;
    font-family: var(--rs-body-font);
    font-size: 18px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.3px;
}
.rs-sec p {
    margin: 0;
    font-size: 13px;
    line-height: 1.5;
    color: #8f95b3;
    max-width: 80ch;
}
.rs-step {
    flex: 0 0 auto;
    font-family: var(--rs-pixel-font);
    font-size: 10px;
    padding: 4px 8px;
    background: #1f2238;
    color: #ffd33d;
    border: 1px solid #ffd33d;
    border-radius: 2px;
    box-shadow: 2px 2px 0px #000000;
}

/* -- Buttons: 3D Pixel Art Styling -- */
button, .gr-button {
    font-family: var(--rs-pixel-font) !important;
    font-size: 11px !important;
    letter-spacing: 0.5px !important;
    border-radius: 4px !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    transition: transform 0.08s ease, box-shadow 0.08s ease !important;
}
button.primary, .gr-button-primary {
    background: #5b5fc7 !important;
    border: 2px solid #ffffff !important;
    color: #ffffff !important;
    box-shadow: 4px 4px 0px #000000 !important;
}
button.primary:hover, .gr-button-primary:hover {
    background: #6c71e0 !important;
    transform: translate(2px, 2px) !important;
    box-shadow: 2px 2px 0px #000000 !important;
}
button.primary:active, .gr-button-primary:active {
    transform: translate(4px, 4px) !important;
    box-shadow: 0px 0px 0px #000000 !important;
}
button.secondary, .gr-button-secondary {
    background: #161826 !important;
    border: 2px solid #383e5c !important;
    color: #d1d6ef !important;
    box-shadow: 3px 3px 0px #000000 !important;
    font-size: 10px !important;
}
button.secondary:hover, .gr-button-secondary:hover {
    background: #1f2235 !important;
    border-color: #ffd33d !important;
    color: #ffd33d !important;
    transform: translate(2px, 2px) !important;
    box-shadow: 1px 1px 0px #000000 !important;
}
button.stop, .gr-button-stop {
    background: #dc2626 !important;
    border: 2px solid #fca5a5 !important;
    color: #ffffff !important;
    box-shadow: 3px 3px 0px #000000 !important;
    font-size: 10px !important;
}
button.stop:hover, .gr-button-stop:hover {
    background: #ef4444 !important;
    transform: translate(2px, 2px) !important;
    box-shadow: 1px 1px 0px #000000 !important;
}

/* -- Form Inputs & Chips -- */
input[type="text"], input[type="number"], textarea, select {
    background: #0d0f17 !important;
    border: 1px solid #272b42 !important;
    color: #f0f2fb !important;
    border-radius: 4px !important;
    font-family: var(--rs-body-font) !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
input:focus, textarea:focus, select:focus {
    border-color: #ffd33d !important;
    box-shadow: 0 0 0 2px rgba(255, 211, 61, 0.25) !important;
    outline: none !important;
}
.gradio-container .wrap > label {
    border-radius: 4px !important;
    background: #0d0f17 !important;
    border: 1px solid #272b42 !important;
    color: #8b92b0 !important;
}
.gradio-container .wrap > label:hover {
    border-color: #4b5278 !important;
    color: #ffffff !important;
}
.gradio-container .wrap > label.selected {
    background: #1f233a !important;
    border-color: #ffd33d !important;
    color: #ffd33d !important;
}
input[type="range"] {
    accent-color: #ffd33d;
}

/* -- Training Log Boxes -- */
#training-info-step2a textarea,
#training-info-step2b textarea,
#training-info-step3 textarea {
    height: 12rem !important;
    min-height: 12rem !important;
    max-height: 12rem !important;
    overflow-y: auto !important;
    resize: none !important;
    background: #08090e !important;
    color: #4ade80 !important;
    border: 1px solid #22263a !important;
    border-radius: 4px !important;
    font-family: var(--rs-code-font) !important;
    font-size: 12px !important;
    box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.6) !important;
}

/* -- Training Monitor -- */
.rs-empty {
    padding: 28px;
    border: 1px dashed var(--rs-line);
    border-radius: 4px;
    background: #141624;
    color: var(--rs-muted);
    font-size: 14px;
    font-family: var(--rs-pixel-font);
}
.rs-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 14px; }
.rs-metric { background: #141624; border: 1px solid var(--rs-line); border-radius: 4px; padding: 12px 14px; }
.rs-metric span { display: block; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--rs-muted); font-family: var(--rs-pixel-font); }
.rs-metric b { display: block; margin-top: 4px; font-size: 20px; font-family: var(--rs-code-font); color: #ffd33d; }
.rs-charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }
.rs-chart { background: #141624; border: 1px solid var(--rs-line); border-top: 3px solid #6366f1; border-radius: 4px; padding: 14px 16px; }
.rs-chart h4 { margin: 0 0 2px 0; font-size: 14px; font-family: var(--rs-pixel-font); color: #ffffff; }
.rs-chart small { display: block; margin-bottom: 8px; color: var(--rs-muted); font-size: 12px; }
.rs-chart svg { display: block; width: 100%; height: auto; background: #0b0c13; border-radius: 4px; }
.rs-files { width: 100%; border-collapse: collapse; margin-top: 14px; font-size: 13px; }
.rs-files th, .rs-files td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--rs-line); }
.rs-files th { color: #ffd33d; font-family: var(--rs-pixel-font); font-size: 10px; background: #1a1d2e; }

/* -- Flow & Team Table -- */
.rs-flow { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }
.rs-flow div {
    background: #141624 !important;
    border: 1px solid #272b42 !important;
    border-top: 3px solid #6366f1 !important;
    border-radius: 4px !important;
    padding: 14px 16px !important;
    color: #8b92b0 !important;
    box-shadow: 2px 2px 0px #000000 !important;
    font-size: 13px;
    line-height: 1.5;
}
.rs-flow b {
    display: block;
    margin-bottom: 4px;
    color: #ffffff !important;
    font-family: var(--rs-body-font) !important;
    font-size: 15px;
}
.rs-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.rs-table th, .rs-table td { text-align: left; padding: 10px 12px; border-bottom: 1px solid #222538; vertical-align: top; }
.rs-table th { background: #1a1d2e; color: #ffd33d; font-family: var(--rs-pixel-font); font-size: 10px; }
.rs-table code { font-family: var(--rs-code-font); font-size: 12px; color: #38bdf8; }
.rs-footer {
    margin: 28px 16px 0 16px;
    padding-top: 14px;
    border-top: 1px solid #262a40;
    font-family: var(--rs-pixel-font);
    font-size: 10px;
    letter-spacing: 0.5px;
    color: #8b92b0;
    text-align: center;
}

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0c0d14; border-radius: 2px; }
::-webkit-scrollbar-thumb { background: #282c42; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #3e4466; }

@media (max-width: 768px) {
    .rs-hero-main-row { flex-direction: column; text-align: center; }
    .rs-top-bar { flex-direction: column; gap: 8px; }
    .pixel-coin-float { display: none; }
    .pixel-huge-headline { font-size: 32px; }
}
"""

STUDIO_CSS = STUDIO_CSS.replace(
    "__TAB_ICON_RULES__",
    "\n".join(
        "#main-tabs > .tab-nav button:nth-of-type(%d)::before { --ic: %s; }"
        % (i + 1, _icon_url(inner))
        for i, inner in enumerate(TAB_ICONS)
    ),
)

try:
    STUDIO_THEME = gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.violet,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_sm,
        font=["Inter", "Press Start 2P", "system-ui", "sans-serif"],
    )
except Exception:
    STUDIO_THEME = None


def render_header():
    stats = [
        ("Device", str(config.device)),
        ("GPUs", str(len(gpu_indices)) if if_gpu_ok else "None"),
        ("Voice models", str(len(names))),
        ("Status", "ONLINE [●]"),
    ]
    stat_boxes = "".join(
        "<div class='rs-stat-box'><span class='stat-lbl'>%s</span><span class='stat-val'>%s</span></div>"
        % (html.escape(label), html.escape(value))
        for label, value in stats
    )

    top_bar = (
        "<div class='rs-top-bar'>"
        "<div class='rs-top-brand'>MAKEREIGN* ACADEMY</div>"
        "<div class='rs-top-center'>"
        "<span class='rs-score-label'>SCORE : 0</span>"
        "<div class='rs-hearts'>"
        "<span class='pixel-heart'>💛</span>"
        "<span class='pixel-heart'>💛</span>"
        "<span class='pixel-heart'>💛</span>"
        "</div></div>"
        "<div class='rs-top-right'><div class='rs-menu-lines'><span class='menu-line'></span><span class='menu-line'></span></div></div>"
        "</div>"
    )

    coin_svg = (
        '<svg width="28" height="34" viewBox="0 0 16 19" fill="none">'
        '<rect x="4" y="0" width="8" height="2" fill="#000000"/>'
        '<rect x="2" y="2" width="2" height="2" fill="#000000"/>'
        '<rect x="12" y="2" width="2" height="2" fill="#000000"/>'
        '<rect x="0" y="4" width="2" height="11" fill="#000000"/>'
        '<rect x="14" y="4" width="2" height="11" fill="#000000"/>'
        '<rect x="2" y="15" width="2" height="2" fill="#000000"/>'
        '<rect x="12" y="15" width="2" height="2" fill="#000000"/>'
        '<rect x="4" y="17" width="8" height="2" fill="#000000"/>'
        '<rect x="4" y="2" width="8" height="2" fill="#ffe066"/>'
        '<rect x="2" y="4" width="12" height="11" fill="#ffd33d"/>'
        '<rect x="4" y="15" width="8" height="2" fill="#d49b00"/>'
        '<rect x="4" y="4" width="4" height="4" fill="#fff3a8"/>'
        '<rect x="10" y="6" width="2" height="7" fill="#d49b00"/>'
        '<rect x="6" y="7" width="4" height="5" fill="#f5b300"/>'
        '</svg>'
    )

    cursor_svg = (
        '<svg class="pixel-3d-cursor" width="110" height="140" viewBox="0 0 44 56" fill="none">'
        '<path d="M12 2v40l9-9 7 15 6-3-7-15 13-1L12 2z" fill="#000000" transform="translate(3, 4)"/>'
        '<path d="M12 2v40l9-9 7 15 6-3-7-15 13-1L12 2z" fill="#222433" transform="translate(2, 2)"/>'
        '<path d="M12 2v40l9-9 7 15 6-3-7-15 13-1L12 2z" fill="#4d5169" transform="translate(1, 1)"/>'
        '<path d="M12 2v40l9-9 7 15 6-3-7-15 13-1L12 2z" fill="#888fa8"/>'
        '<path d="M13 5v34l7-7 7 15 3-1-7-15 11-1L13 5z" fill="#c3c8dc"/>'
        '<path d="M14 8v28l5-5 7 15 1-0.5-7-15 9-1L14 8z" fill="#ffffff"/>'
        '</svg>'
    )

    file_badge_svg = (
        '<svg width="20" height="24" viewBox="0 0 14 18" fill="none">'
        '<path d="M0 0h9l5 5v13H0V0z" fill="#ffffff"/>'
        '<path d="M9 0v5h5" fill="#a0a5c4"/>'
        '<path d="M7 7v6M4 10l3 3 3-3" stroke="#12131a" stroke-width="1.8" stroke-linecap="square"/>'
        '</svg>'
    )

    hero = (
        "<div class='rs-pixel-hero'>"
        "<div class='rs-hero-main-row'>"
        "<div class='rs-hero-left'>"
        "<div class='pixel-transform-box'>"
        "<span class='t-handle th-tl'></span><span class='t-handle th-tc'></span><span class='t-handle th-tr'></span>"
        "<span class='t-handle th-ml'></span><span class='t-handle th-mr'></span>"
        "<span class='t-handle th-bl'></span><span class='t-handle th-bc'></span><span class='t-handle th-br'></span>"
        "<div class='pixel-box-content'>LEVEL UP</div>"
        "</div>"
        "<div class='pixel-coin-float'>%s</div>"
        "<div class='pixel-huge-headline'>YOUR AUDIO<br>SKILLZ.</div>"
        "</div>"
        "<div class='rs-hero-center-cursor'>%s</div>"
        "<div class='rs-hero-right-info'>"
        "** NEXT-GEN NEURAL VOICE CLONING, STEM SEPARATION & MODEL TRAINING STUDIO."
        "</div>"
        "</div>"
        "<div class='rs-hero-footer-bar'>"
        "<div class='rs-hero-scroll'>SCROLL ▼</div>"
        "<div class='rs-hero-file-badge'>%s<span class='pixel-badge-text'>AUDIO_OUTPUT_V2.FLAC</span></div>"
        "</div>"
        "</div>"
        "<div class='rs-stats-container'>%s</div>"
    ) % (coin_svg, cursor_svg, file_badge_svg, stat_boxes)

    return top_bar + hero


def section_head(title, text="", step=None):
    marker = "<div class='rs-step'>[%02d]</div>" % int(step) if step else ""
    body = "<p>%s</p>" % text if text else ""
    return gr.HTML(
        "<div class='rs-sec'>%s<div><h3>%s</h3>%s</div></div>"
        % (marker, html.escape(title), body)
    )


# --- Training monitor ------------------------------------------------------

LOSS_PATTERN = re.compile(r"(loss_[a-z_]+)\s*=\s*(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", re.I)
EPOCH_PATTERN = re.compile(r"Train Epoch:\s*(\d+)")
CHART_COLORS = ["#2d6a4f", "#52b788", "#74c69d", "#e63946", "#1b4332", "#95d5b2"]


def list_experiments():
    root = os.path.join(now_dir, "logs")
    if not os.path.isdir(root):
        return []
    return sorted(
        name
        for name in os.listdir(root)
        if os.path.isfile(os.path.join(root, name, "train.log"))
    )


def refresh_experiments():
    return {"choices": list_experiments(), "__type__": "update"}


def parse_train_log(path, max_points=400):
    series = {}
    epoch = None
    for line in read_text(path, errors="ignore").splitlines():
        match = EPOCH_PATTERN.search(line)
        if match:
            epoch = int(match.group(1))
        for key, value in LOSS_PATTERN.findall(line):
            series.setdefault(key.lower(), []).append(float(value))
    for key, values in series.items():
        if len(values) > max_points:
            step = -(-len(values) // max_points)
            sampled = values[::step]
            if sampled[-1] != values[-1]:
                sampled.append(values[-1])
            series[key] = sampled
    return series, epoch


def render_loss_chart(name, values, color):
    width, height, pad = 360, 120, 8
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    count = len(values)
    points = []
    for index, value in enumerate(values):
        x = pad + (width - 2 * pad) * (index / (count - 1) if count > 1 else 0.5)
        y = pad + (height - 2 * pad) * (1 - (value - low) / span)
        points.append("%.1f,%.1f" % (x, y))
    if count > 1:
        shape = '<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>' % (
            color,
            " ".join(points),
        )
    else:
        x, y = points[0].split(",")
        shape = '<circle cx="%s" cy="%s" r="3" fill="%s"/>' % (x, y, color)
    label = name.replace("loss_", "").replace("_", " ")
    return (
        "<div class='rs-chart' style='border-top-color:%s'>"
        "<h4>%s loss</h4>"
        "<small>latest %.3f, min %.3f, max %.3f</small>"
        '<svg viewBox="0 0 %d %d" role="img" aria-label="%s loss over training">%s</svg>'
        "</div>"
    ) % (
        color,
        html.escape(label),
        values[-1],
        low,
        high,
        width,
        height,
        html.escape(label),
        shape,
    )


def render_monitor(exp_name):
    if not exp_name:
        return (
            "<div class='rs-empty'>Choose an experiment to see its training curves. "
            "Experiments appear here once training has written a train.log file.</div>"
        )
    exp_name = os.path.basename(str(exp_name))
    exp_dir = os.path.join(now_dir, "logs", exp_name)
    log_path = os.path.join(exp_dir, "train.log")
    if not os.path.isfile(log_path):
        return "<div class='rs-empty'>No train.log found for %s.</div>" % html.escape(
            exp_name
        )
    try:
        series, epoch = parse_train_log(log_path)
    except Exception:
        logger.exception("Could not read training log")
        return "<div class='rs-empty'>The training log could not be read.</div>"

    updated = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(log_path)))
    samples = max((len(v) for v in series.values()), default=0)
    metrics = (
        "<div class='rs-metrics'>"
        "<div class='rs-metric'><span>Experiment</span><b>%s</b></div>"
        "<div class='rs-metric'><span>Latest epoch</span><b>%s</b></div>"
        "<div class='rs-metric'><span>Loss samples</span><b>%d</b></div>"
        "<div class='rs-metric'><span>Log updated</span><b style='font-size:16px'>%s</b></div>"
        "</div>"
    ) % (
        html.escape(exp_name),
        epoch if epoch is not None else "n/a",
        samples,
        updated,
    )

    if series:
        charts = "<div class='rs-charts'>%s</div>" % "".join(
            render_loss_chart(name, values, CHART_COLORS[i % len(CHART_COLORS)])
            for i, (name, values) in enumerate(sorted(series.items()))
        )
    else:
        charts = (
            "<div class='rs-empty'>No loss values found in this log yet. "
            "They appear after the first logging interval of training.</div>"
        )

    checkpoints = []
    if os.path.isdir(exp_dir):
        for fname in os.listdir(exp_dir):
            if fname.lower().endswith(".pth"):
                fpath = os.path.join(exp_dir, fname)
                checkpoints.append((os.path.getmtime(fpath), fname, os.path.getsize(fpath)))
    checkpoints.sort(reverse=True)
    rows = "".join(
        "<tr><td>%s</td><td>%.1f MB</td><td>%s</td></tr>"
        % (
            html.escape(fname),
            size / 1048576.0,
            time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)),
        )
        for mtime, fname, size in checkpoints[:6]
    )
    files = (
        "<table class='rs-files'><tr><th>Latest checkpoints</th><th>Size</th><th>Saved</th></tr>%s</table>"
        % rows
        if rows
        else ""
    )
    return metrics + charts + files


# --- API reference ---------------------------------------------------------

API_ENDPOINTS = [
    ("/yt_download", "Download audio from YouTube / URL via yt-dlp"),
    ("/infer_change_voice", "Load a voice model and match its feature index"),
    ("/infer_convert", "Convert one audio file"),
    ("/infer_convert_batch", "Convert a folder or a set of uploaded files"),
    ("/infer_refresh", "Refresh the list of voice models"),
    ("/infer_clean", "Unload the current voice model"),
    ("/pymss_separate", "Separate vocals, instrumentals and reverb"),
    ("/train_preprocess", "Slice and normalise the training audio"),
    ("/train_extract_f0_feature", "Extract pitch and HuBERT features"),
    ("/train_start", "Train the voice model"),
    ("/train_start_all", "Run preprocessing, extraction, training and indexing"),
    ("/monitor_show", "Render the training monitor for an experiment"),




]


def render_api_table():
    rows = "".join(
        "<tr><td><code>%s</code></td><td>%s</td></tr>" % (html.escape(path), html.escape(text))
        for path, text in API_ENDPOINTS
    )
    return "<table class='rs-table'><tr><th>Endpoint</th><th>What it does</th></tr>%s</table>" % rows


def render_api_example():
    return (
        "```python\n"
        "from gradio_client import Client\n\n"
        'client = Client("http://127.0.0.1:%s/")\n'
        "client.view_api()  # prints the exact parameters of every endpoint\n\n"
        "# 1. choose the voice model, 2. convert a file\n"
        'client.predict("my-voice.pth", 0.33, 0.33, api_name="/infer_change_voice")\n'
        'client.predict(0, "input.flac", 0, "rmvpe", "", 0.75, 0, 0.25, 0.33,\n'
        '               api_name="/infer_convert")\n'
        "```" % config.listen_port
    )


ABOUT_FLOW = [
    ("Input", "Source audio: any voice, any content."),
    ("Encode", "HuBERT turns the speech into feature embeddings."),
    ("Retrieve", "FAISS finds the closest features of the target voice."),
    ("Decode", "The vocoder synthesises audio in the target voice."),
]

ABOUT_TEAM = [
    ("Ayush Pandey", "25BCE10198"),
    ("Punya Prasoon", "25BCE10774"),
    ("Shyam Singh", "25BCE10775"),
    ("Harshit Shrimali", "25BCE10714"),
    ("Arnob Ghosh", "25BCE10254"),
]


def change_f0_method(f0method8):
    if f0method8 == "rmvpe":
        visible = F0GPUVisible
    else:
        visible = False
    return {"visible": visible, "__type__": "update"}


blocks_kwargs = {"theme": STUDIO_THEME} if STUDIO_THEME is not None else {}

with gr.Blocks(title=STUDIO_NAME, css=STUDIO_CSS, **blocks_kwargs) as app:
    gr.HTML(render_header())
    with gr.Tabs(elem_id="main-tabs"):
        # ------------------------------------------------------------------
        # Voice conversion
        # ------------------------------------------------------------------
        with gr.TabItem("Voice Conversion"):
            with gr.Column(elem_classes=["rs-card", "accent-blue"]):
                section_head(
                    "Voice model",
                    "Pick the model that will speak. Selecting a model also fills in its feature index.",
                )
                with gr.Row():
                    sid0 = gr.Dropdown(label="Voice model", choices=sorted(names))
                    with gr.Column():
                        refresh_button = gr.Button("Refresh models", variant="secondary")
                        clean_button = gr.Button(
                            "Unload model to free VRAM", variant="secondary"
                        )
                    spk_item = gr.Slider(
                        minimum=0,
                        maximum=2333,
                        step=1,
                        label="Speaker ID",
                        value=0,
                        visible=False,
                        interactive=True,
                    )
                    clean_button.click(
                        fn=clean, inputs=[], outputs=[sid0], api_name="infer_clean"
                    )
            with gr.Tabs():
                with gr.TabItem("Single file"):
                    with gr.Row():
                        with gr.Column(elem_classes=["rs-card", "accent-blue"]):
                            section_head("Source audio")
                            input_audio0 = gr.Audio(
                                label="Source audio (drop a file or click to upload)",
                                source="upload",
                                type="filepath",
                                interactive=True,
                            )
                            with gr.Accordion("Or download from YouTube / URL (yt-dlp)", open=False):
                                with gr.Row():
                                    yt_url_input = gr.Textbox(
                                        label="YouTube / audio URL",
                                        placeholder="https://www.youtube.com/watch?v=...",
                                    )
                                    yt_dl_btn = gr.Button("Fetch Audio", variant="secondary")
                                yt_status = gr.Textbox(label="Download status", interactive=False)
                                yt_dl_btn.click(
                                    download_yt_audio,
                                    [yt_url_input],
                                    [input_audio0, yt_status],
                                    api_name="yt_download",
                                )
                            vc_transform0 = gr.Number(
                                label="Pitch shift in semitones (+12 is one octave up, -12 one octave down)",
                                value=0, visible=False,
                            )
                            f0method0 = gr.Radio(
                                label="Pitch extraction method",
                                choices=["rmvpe"],
                                value="rmvpe",
                                interactive=True,
                            )
                        with gr.Column(elem_classes=["rs-card", "accent-teal"]):
                            section_head("Conversion settings")
                            index_rate1 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label="Index influence (how much retrieved target-voice detail to use)",
                                value=0.75,
                                interactive=True,
                            )
                            file_index1 = gr.Textbox(
                                label="Feature index path (filled in when a model is selected; editable)",
                                placeholder="assets/indices/my-voice.index",
                                interactive=True,
                            )
                            with gr.Accordion("Advanced settings", open=False):
                                resample_sr0 = gr.Slider(
                                    minimum=0,
                                    maximum=48000,
                                    label="Output sample rate (0 keeps the model's native rate)",
                                    value=0,
                                    step=1,
                                    interactive=True,
                                )
                                rms_mix_rate0 = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    label="Volume envelope mix (closer to 1 keeps the converted loudness, lower follows the source)",
                                    value=0.25,
                                    interactive=True,
                                )
                                protect0 = gr.Slider(
                                    minimum=0,
                                    maximum=0.5,
                                    label="Consonant and breath protection (0.5 turns it off; lower protects more but weakens the index effect)",
                                    value=0.33,
                                    step=0.01,
                                    interactive=True,
                                )
                            refresh_button.click(
                                fn=change_choices,
                                inputs=[],
                                outputs=sid0,
                                api_name="infer_refresh",
                            )
                    with gr.Column(elem_classes=["rs-card", "accent-amber"]):
                        section_head("Result")
                        but0 = gr.Button("Convert voice", variant="primary")
                        with gr.Row():
                            vc_output1 = gr.Textbox(label="Status")
                            vc_output2 = gr.Audio(
                                label="Converted audio (use the menu in the corner to download)"
                            )

                        but0.click(
                            vc.vc_single,
                            [
                                spk_item,
                                input_audio0,
                                vc_transform0,
                                f0method0,
                                file_index1,
                                index_rate1,
                                resample_sr0,
                                rms_mix_rate0,
                                protect0,
                            ],
                            [vc_output1, vc_output2],
                            api_name="infer_convert",
                        )
                with gr.TabItem("Batch"):
                    with gr.Column(elem_classes=["rs-card", "accent-blue"]):
                        section_head(
                            "Batch conversion",
                            "Convert a folder of audio, or several uploaded files, and write the results to the output folder.",
                        )
                        with gr.Row():
                            dir_input = gr.Textbox(
                                label="Input folder path",
                                placeholder="datasets/songs",
                            )
                            inputs = gr.File(
                                file_count="multiple",
                                label="Or upload audio files (the folder is used if both are given)",
                            )
                    with gr.Row():
                        with gr.Column(elem_classes=["rs-card", "accent-teal"]):
                            section_head("Output")
                            vc_transform1 = gr.Number(
                                label="Pitch shift in semitones (+12 is one octave up, -12 one octave down)",
                                value=0, visible=False,
                            )
                            opt_input = gr.Textbox(label="Output folder", value="opt")
                            file_index3 = gr.Textbox(
                                label="Feature index path (filled in when a model is selected; editable)",
                                value="",
                                interactive=True,
                            )
                            f0method1 = gr.Radio(
                                label="Pitch extraction method",
                                choices=["rmvpe"],
                                value="rmvpe",
                                interactive=True,
                            )
                            format1 = gr.Radio(
                                label="Export format",
                                choices=["flac", "mp3"],
                                value="flac",
                                interactive=True,
                            )
                        with gr.Column(elem_classes=["rs-card", "accent-teal"]):
                            section_head("Quality")
                            index_rate2 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label="Index influence (how much retrieved target-voice detail to use)",
                                value=1,
                                interactive=True,
                            )
                            resample_sr1 = gr.Slider(
                                minimum=0,
                                maximum=48000,
                                label="Output sample rate (0 keeps the model's native rate)",
                                value=0,
                                step=1,
                                interactive=True,
                            )
                            rms_mix_rate1 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label="Volume envelope mix (closer to 1 keeps the converted loudness, lower follows the source)",
                                value=1,
                                interactive=True,
                            )
                            protect1 = gr.Slider(
                                minimum=0,
                                maximum=0.5,
                                label="Consonant and breath protection (0.5 turns it off; lower protects more but weakens the index effect)",
                                value=0.33,
                                step=0.01,
                                interactive=True,
                            )
                    with gr.Column(elem_classes=["rs-card", "accent-amber"]):
                        but1 = gr.Button("Convert batch", variant="primary")
                        vc_output3 = gr.Textbox(label="Status")

                        but1.click(
                            vc.vc_multi,
                            [
                                spk_item,
                                dir_input,
                                opt_input,
                                inputs,
                                vc_transform1,
                                f0method1,
                                file_index3,
                                index_rate2,
                                resample_sr1,
                                rms_mix_rate1,
                                protect1,
                                format1,
                            ],
                            [vc_output3],
                            api_name="infer_convert_batch",
                        )
                    sid0.change(
                        fn=vc.get_vc,
                        inputs=[sid0, protect0, protect1],
                        outputs=[spk_item, protect0, protect1, file_index1, file_index3],
                        api_name="infer_change_voice",
                    )

        # ------------------------------------------------------------------
        # Vocal extraction
        # ------------------------------------------------------------------
        with gr.TabItem("Vocal Extraction"):
            with gr.Row():
                with gr.Column(elem_classes=["rs-card", "accent-amber"]):
                    section_head(
                        "Source audio",
                        "Separate vocals from instrumentals and remove reverb in bulk with PyMSS/MSST models.",
                    )
                    dir_wav_input = gr.Textbox(
                        label="Input folder path",
                        placeholder="datasets/songs",
                    )
                    wav_inputs = gr.File(
                        file_count="multiple",
                        label="Or upload audio files (the folder is used if both are given)",
                    )
                with gr.Column(elem_classes=["rs-card", "accent-amber"]):
                    section_head("Separation settings")
                    model_choose = gr.Dropdown(
                        label="Separation model",
                        choices=pymss_names,
                        value=pymss_names[0],
                        interactive=True,
                    )
                    model_info = gr.Textbox(
                        label="Underlying model",
                        value=get_model_info(pymss_names[0]),
                        interactive=False,
                    )
                    model_choose.change(
                        get_model_info,
                        [model_choose],
                        [model_info],
                        queue=False,
                    )
                    opt_vocal_root = gr.Textbox(
                        label="Primary output folder", value="opt"
                    )
                    opt_ins_root = gr.Textbox(
                        label="Residual output folder", value="opt"
                    )
                    format0 = gr.Radio(
                        label="Export format",
                        choices=["flac", "mp3"],
                        value="flac",
                        interactive=True,
                    )
            with gr.Column(elem_classes=["rs-card", "accent-blue"]):
                section_head("Progress")
                with gr.Row():
                    but2 = gr.Button("Start separation", variant="primary")
                    stop_pymss_button = gr.Button(
                        "Stop separation", variant="stop", visible=False
                    )
                pymss_progress = gr.HTML(
                    value=render_pymss_progress(0, "Waiting to start", "idle")
                )
                vc_output4 = gr.Textbox(label="Status")
                but2.click(
                    run_pymss_separation,
                    [
                        model_choose,
                        dir_wav_input,
                        opt_vocal_root,
                        wav_inputs,
                        opt_ins_root,
                        format0,
                    ],
                    [vc_output4, pymss_progress, but2, stop_pymss_button],
                    api_name="pymss_separate",
                )
                stop_pymss_button.click(
                    stop_pymss_webui,
                    [],
                    [vc_output4, pymss_progress, but2, stop_pymss_button],
                    queue=False,
                )

        # ------------------------------------------------------------------
        # Voice training
        # ------------------------------------------------------------------
        with gr.TabItem("Voice Training"):
            with gr.Column(elem_classes=["rs-card", "accent-teal"]):
                section_head(
                    "Experiment setup",
                    "Each experiment gets its own folder under logs, holding its configuration, logs and trained model files.",
                    step=1,
                )
                with gr.Row():
                    exp_dir1 = gr.Textbox(label="Experiment name", value="my-voice")
                    sr2 = gr.Radio(
                        label="Target sample rate",
                        choices=["40k", "48k"],
                        value="40k",
                        interactive=True,
                    )
                    if_f0_3 = gr.Radio(
                        label="Pitch guidance (required for singing, optional for speech)",
                        choices=[True, False],
                        value=True, visible=False,
                        interactive=True,
                    )
                    version19 = gr.Radio(
                        label="Model version",
                        choices=["v2"],
                        value="v2",
                        interactive=True,
                        visible=False,
                    )
                np7 = gr.Slider(
                    minimum=0,
                    maximum=config.n_cpu,
                    step=1,
                    label="CPU processes for pitch extraction and data processing",
                    value=int(np.ceil(config.n_cpu / 1.5)),
                    interactive=True,
                )
            with gr.Column(elem_classes=["rs-card", "accent-teal"]):
                section_head(
                    "Process the dataset",
                    "Every decodable audio file in the training folder is sliced and normalised into the experiment folder. Single-speaker training only.",
                    step=2,
                )
                with gr.Row():
                    trainset_dir4 = gr.Textbox(
                        label="Training audio folder",
                        value="",
                        placeholder="datasets/my-voice",
                    )
                    spk_id5 = gr.Slider(
                        minimum=0,
                        maximum=4,
                        step=1,
                        label="Speaker ID",
                        value=0, visible=False,
                        interactive=True,
                    )
                with gr.Row():
                    but1 = gr.Button("Process data", variant="primary")
                    stop_but1 = gr.Button(
                        "Stop processing", variant="stop", visible=False
                    )
                info1 = gr.Textbox(
                    label="Status",
                    value="",
                    lines=8,
                    max_lines=8,
                    elem_id="training-info-step2a",
                )
                but1.click(
                    preprocess_dataset,
                    [trainset_dir4, exp_dir1, sr2, np7],
                    [info1, but1, stop_but1],
                    api_name="train_preprocess",
                )
                stop_but1.click(
                    stop_preprocess_dataset,
                    [],
                    [info1, but1, stop_but1],
                    queue=False,
                )
            with gr.Column(elem_classes=["rs-card", "accent-teal"]):
                section_head(
                    "Extract pitch and features",
                    "Pitch is extracted on the CPU when the model uses pitch guidance. HuBERT features are extracted on the GPU you choose.",
                    step=3,
                )
                with gr.Row():
                    with gr.Column():
                        gpus6 = gr.Textbox(
                            label="GPU IDs separated by - (0-1-2 uses GPUs 0, 1 and 2)",
                            value=gpus,
                            interactive=True,
                            visible=F0GPUVisible,
                        )
                        gpu_info9 = gr.Textbox(
                            label="GPU information", value=gpu_info, visible=F0GPUVisible
                        )
                    with gr.Column():
                        f0method8 = gr.Radio(
                            label="Pitch extraction method",
                            choices=["rmvpe"],
                            value="rmvpe",
                            interactive=True,
                        )
                        gpus_rmvpe = gr.Textbox(
                            label="RMVPE GPU assignment: one entry per process, separated by - (0-0-1 runs two processes on GPU 0 and one on GPU 1)",
                            value="%s-%s" % (gpus, gpus),
                            interactive=True,
                            visible=F0GPUVisible,
                        )
                with gr.Row():
                    but2 = gr.Button("Extract features", variant="primary")
                    stop_but2 = gr.Button(
                        "Stop extraction", variant="stop", visible=False
                    )
                info2 = gr.Textbox(
                    label="Status",
                    value="",
                    lines=8,
                    max_lines=8,
                    elem_id="training-info-step2b",
                )
                f0method8.change(
                    fn=change_f0_method,
                    inputs=[f0method8],
                    outputs=[gpus_rmvpe],
                )
                but2.click(
                    extract_f0_feature,
                    [
                        gpus6,
                        np7,
                        f0method8,
                        if_f0_3,
                        exp_dir1,
                        version19,
                        gpus_rmvpe,
                    ],
                    [info2, but2, stop_but2],
                    api_name="train_extract_f0_feature",
                )
                stop_but2.click(
                    stop_extract_f0_feature,
                    [],
                    [info2, but2, stop_but2],
                    queue=False,
                )
            with gr.Column(elem_classes=["rs-card", "accent-amber"]):
                section_head(
                    "Train the model and index",
                    "Set the training options, then train the model and the feature index, or run the whole pipeline in one go.",
                    step=4,
                )
                with gr.Row():
                    save_epoch10 = gr.Slider(
                        minimum=1,
                        maximum=50,
                        step=1,
                        label="Save every N epochs",
                        value=5,
                        interactive=True,
                    )
                    total_epoch11 = gr.Slider(
                        minimum=2,
                        maximum=1200,
                        step=1,
                        label="Total epochs",
                        value=20,
                        interactive=True,
                    )
                    batch_size12 = gr.Slider(
                        minimum=1,
                        maximum=40,
                        step=1,
                        label="Batch size per GPU",
                        value=default_batch_size,
                        interactive=True,
                    )
                with gr.Row():
                    if_save_latest13 = gr.Radio(
                        label="Keep only the latest checkpoint (saves disk space)",
                        choices=[YES, NO],
                        value=NO, visible=False,
                        interactive=True,
                    )
                    if_cache_gpu17 = gr.Radio(
                        label="Cache the dataset in VRAM (worth it only for under 10 minutes of audio)",
                        choices=[YES, NO],
                        value=NO,
                        interactive=True,
                    )
                    if_save_every_weights18 = gr.Radio(
                        label="Save a final small model to the weights folder at every save point",
                        choices=[YES, NO],
                        value=NO,
                        interactive=True,
                    )
                with gr.Row():
                    pretrained_G14 = gr.Textbox(
                        label="Pretrained generator (G) path",
                        value="assets/pretrained_v2/f0G40k.pth", visible=False,
                        interactive=True,
                    )
                    pretrained_D15 = gr.Textbox(
                        label="Pretrained discriminator (D) path",
                        value="assets/pretrained_v2/f0D40k.pth", visible=False,
                        interactive=True,
                    )
                sr2.change(
                    fn=change_sr2,
                    inputs=[sr2, if_f0_3, version19],
                    outputs=[pretrained_G14, pretrained_D15],
                )
                # Default v2 pretrained weights are updated dynamically

                gpus16 = gr.Textbox(
                    label="GPU IDs separated by - (0-1-2 uses GPUs 0, 1 and 2)",
                    value=gpus,
                    interactive=True,
                )
                with gr.Row():
                    but3 = gr.Button("Train model", variant="primary")
                    stop_but3 = gr.Button(
                        "Stop training", variant="stop", visible=False
                    )
                    but4 = gr.Button("Train feature index", variant="primary")
                    stop_but4 = gr.Button(
                        "Stop index training", variant="stop", visible=False
                    )
                    but5 = gr.Button("Run full pipeline", variant="primary")
                    stop_but5 = gr.Button(
                        "Stop pipeline", variant="stop", visible=False
                    )
                info3 = gr.Textbox(
                    label="Status",
                    value="",
                    lines=8,
                    max_lines=8,
                    elem_id="training-info-step3",
                )
                but3.click(
                    click_train,
                    [
                        exp_dir1,
                        sr2,
                        if_f0_3,
                        spk_id5,
                        save_epoch10,
                        total_epoch11,
                        batch_size12,
                        if_save_latest13,
                        pretrained_G14,
                        pretrained_D15,
                        gpus16,
                        if_cache_gpu17,
                        if_save_every_weights18,
                        version19,
                    ],
                    [info3, but3, stop_but3, sid0],
                    api_name="train_start",
                )
                stop_but3.click(
                    stop_train_model,
                    [],
                    [info3, but3, stop_but3],
                    queue=False,
                )
                but4.click(
                    train_index,
                    [exp_dir1, version19],
                    [info3, but4, stop_but4],
                )
                stop_but4.click(
                    stop_train_index,
                    [],
                    [info3, but4, stop_but4],
                    queue=False,
                )
                but5.click(
                    train1key,
                    [
                        exp_dir1,
                        sr2,
                        if_f0_3,
                        trainset_dir4,
                        spk_id5,
                        np7,
                        f0method8,
                        save_epoch10,
                        total_epoch11,
                        batch_size12,
                        if_save_latest13,
                        pretrained_G14,
                        pretrained_D15,
                        gpus16,
                        if_cache_gpu17,
                        if_save_every_weights18,
                        version19,
                        gpus_rmvpe,
                    ],
                    [info3, but5, stop_but5, sid0],
                    api_name="train_start_all",
                )
                stop_but5.click(
                    stop_train1key,
                    [],
                    [info3, but5, stop_but5],
                    queue=False,
                )

        # ------------------------------------------------------------------
        # Training monitor
        # ------------------------------------------------------------------
        with gr.TabItem("Training Monitor"):
            with gr.Column(elem_classes=["rs-card", "accent-red"]):
                section_head(
                    "Training curves",
                    "Reads the experiment's train.log and plots each loss value it finds. Press Refresh view while training runs to pull in new points.",
                )
                with gr.Row():
                    monitor_exp = gr.Dropdown(
                        label="Experiment",
                        choices=list_experiments(),
                        value=None,
                        interactive=True,
                    )
                    with gr.Column():
                        monitor_list_button = gr.Button(
                            "Refresh experiments", variant="secondary"
                        )
                        monitor_button = gr.Button("Refresh view", variant="primary")
                monitor_view = gr.HTML(value=render_monitor(None))
                monitor_list_button.click(
                    refresh_experiments, [], [monitor_exp], queue=False
                )
                monitor_button.click(
                    render_monitor,
                    [monitor_exp],
                    [monitor_view],
                    api_name="monitor_show",
                )
                monitor_exp.change(
                    render_monitor, [monitor_exp], [monitor_view], queue=False
                )

        # ------------------------------------------------------------------
        # API
        # ------------------------------------------------------------------
        with gr.TabItem("API"):
            with gr.Column(elem_classes=["rs-card", "accent-teal"]):
                section_head(
                    "Use RVC Studio from code",
                    "Every action in this interface is also available over HTTP, so other programs can run conversions, separation and training.",
                )
                gr.HTML(render_api_table())
            with gr.Column(elem_classes=["rs-card", "accent-teal"]):
                section_head("Example")
                gr.Markdown(render_api_example())

        # ------------------------------------------------------------------
        # About
        # ------------------------------------------------------------------
        with gr.TabItem("About"):
            with gr.Column(elem_classes=["rs-card", "accent-navy"]):
                section_head(
                    "How the conversion works",
                    "RVC changes who is speaking, not what is said. It learns the acoustic character of a target voice and, at conversion time, borrows matching features from that voice.",
                )
                gr.HTML(
                    "<div class='rs-flow'>%s</div>"
                    % "".join(
                        "<div><b>%d. %s</b>%s</div>" % (i + 1, html.escape(t), html.escape(d))
                        for i, (t, d) in enumerate(ABOUT_FLOW)
                    )
                )
            with gr.Column(elem_classes=["rs-card", "accent-navy"]):
                section_head("Project team", "Group project, Exhibition 2026.")
                gr.HTML(
                    "<table class='rs-table'><tr><th>Name</th><th>Registration number</th></tr>%s</table>"
                    % "".join(
                        "<tr><td>%s</td><td>%s</td></tr>" % (html.escape(n), html.escape(r))
                        for n, r in ABOUT_TEAM
                    )
                )
    gr.HTML(
        "<div class='rs-footer'>%s — Exhibition 2026 Group Project.</div>" % html.escape(STUDIO_NAME)
    )

    if config.iscolab:
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        launch_webui_with_port_fallback(app, config)
