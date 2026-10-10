import math


def should_report(index, total, max_updates=12):
    if total <= 0:
        return False
    if total <= max_updates:
        return True
    interval = max(1, math.ceil(total / max_updates))
    return index == 0 or index + 1 == total or (index + 1) % interval == 0


def batch_status(title, current, total, success, failed, latest="", failures=None):
    if total <= 0:
        state = "Waiting for input"
    elif current >= total:
        state = "Completed"
    else:
        state = "Processing"
    lines = [
        "[%s]" % title,
        "Status: %s" % state,
        "Progress: %s/%s | Success: %s | Failed: %s"
        % (
            current,
            total,
            success,
            failed,
        ),
    ]
    if latest:
        lines.append("Current: %s" % latest)
    if failures:
        lines.append("Failed entries:")
        lines.extend(failures[-10:])
        if len(failures) > 10:
            lines.append("...showing only the last 10 failed entries")
    return "\n".join(lines)
