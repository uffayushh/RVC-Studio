<div align="center">

# 🎙️ RVC Studio

**A powerful, all-in-one Real-time Voice Conversion Studio**  
Built with Gradio · Powered by RVC v2 · Featuring the Obama Voice Module

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-WebUI-orange?style=for-the-badge&logo=gradio)](https://gradio.app)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

</div>

---

## ✨ Features

- 🎤 **Real-time Voice Conversion** using RVC v2 models
- 🎵 **Music Source Separation** with PyMSS (Mel-Band Roformer)
- 🧠 **RMVPE Pitch Extraction** for accurate F0 estimation
- 🏋️ **Model Training** pipeline built-in
- 🔊 **Audio de-reverb & karaoke** separation
- 🚀 **CUDA Graph acceleration** support
- 🌐 **Gradio WebUI** — easy to use in your browser
- 🎭 Includes **Obama Voice Module** (custom-trained RVC model)

---

## 🚀 Quick Start

### Prerequisites
- Windows 10/11 (64-bit)
- NVIDIA GPU with CUDA 11.8 support (recommended)
- [Git](https://git-scm.com/)

### 1. Clone the Repository
```bash
git clone https://github.com/uffayushh/RVC-Studio.git
cd RVC-Studio
```

### 2. Download Required Files

> ⚠️ **The following large files are NOT included in this repo** due to GitHub's size limits.  
> You need to download them separately and place them in the correct folders.

| Component | Destination Folder | What to Download |
|---|---|---|
| 🐍 Python Runtime | `runtime/` | Embedded Python 3.12 for Windows |
| 🎵 FFmpeg | Root folder | `ffmpeg.exe` + `ffprobe.exe` |
| 🧠 HuBERT Base | `assets/hubert_base/` | `pytorch_model.bin` |
| 🎯 RMVPE Model | `assets/rmvpe/` | `rmvpe.pt` + `rmvpe.onnx` |
| 🏋️ Pretrained V1 | `assets/pretrained/` | `D32k/40k/48k.pth`, `G32k/40k/48k.pth` |
| 🏋️ Pretrained V2 | `assets/pretrained_v2/` | `f0D/G32k/40k/48k.pth` |
| 🎭 PyMSS Weights | `assets/pymss_weights/` | Mel-Band Roformer `.ckpt` files |
| 🎙️ Voice Models | `assets/weights/` | Your `.pth` voice model files |
| 📑 Vector Indexes | `assets/indices/` | Your `.index` files |

### 3. Run the WebUI
Double-click `go-webui.bat` or run:
```bash
runtime\python.exe webui.py --port 7897
```
Then open your browser at: **http://localhost:7897**

---

## 📁 Project Structure

```
RVC-Studio/
├── webui.py               # Main Gradio WebUI (all tabs)
├── go-webui.bat           # Windows launcher
├── requirments_cu118_py312.txt  # Python dependencies
│
├── infer/                 # Inference engine
│   ├── audio.py           # Audio processing
│   ├── hubert.py          # HuBERT feature extraction
│   ├── rmvpe.py           # RMVPE pitch extraction
│   ├── rtrvc.py           # Real-time RVC pipeline
│   ├── fcpe.py            # FCPE pitch estimation
│   └── vc/                # Voice conversion modules
│
├── train/                 # Training pipeline
├── tools/                 # Utilities (PyMSS, process utils)
├── configs/               # Model & app configuration
│
└── assets/                # Model weights (download separately)
    ├── weights/           # Voice model .pth files
    ├── indices/           # FAISS index files
    ├── pretrained/        # RVC v1 pretrained weights
    ├── pretrained_v2/     # RVC v2 pretrained weights
    ├── hubert_base/       # HuBERT base model
    ├── rmvpe/             # RMVPE pitch model
    └── pymss_weights/     # Source separation models
```

---

## 🎭 Obama Voice Module

This build includes a custom-trained **Barack Obama** RVC voice model.

To use it:
1. Place `obama.pth` → `assets/weights/`
2. Place `obama_added_IVF*.index` → `assets/indices/`
3. Launch the WebUI and select the Obama model from the dropdown

---

## 🛠️ Manual Installation (Advanced)

If you want to use your own Python environment:

```bash
# Install CUDA 11.8 compatible dependencies
pip install -r requirments_cu118_py312.txt
```

Then run:
```bash
python webui.py --pycmd python --port 7897
```

---

## 📋 Requirements

See [`requirments_cu118_py312.txt`](requirments_cu118_py312.txt) for the full dependency list.

Key dependencies:
- `torch` (CUDA 11.8)
- `gradio`
- `librosa`
- `faiss-gpu`
- `onnxruntime-gpu`

---

## 🙏 Credits

- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) — Core voice conversion engine
- [RMVPE](https://github.com/Dream-High/RMVPE) — Pitch extraction
- [Mel-Band Roformer](https://github.com/lucidrains/mel-band-roformer) — Source separation

---

<div align="center">

Made with ❤️ by **[uffayushh](https://github.com/uffayushh)**

</div>
