"""Best-effort GPU busy check before starting a model-loading job."""
import os
import shutil
import subprocess
import sys


def gpu_status():
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    line = out.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 4:
        return None
    try:
        used, total, util = int(parts[1]), int(parts[2]), int(parts[3])
    except ValueError:
        return None
    return {"name": parts[0], "used_mb": used, "total_mb": total,
            "utilization": util, "free_mb": max(0, total - used)}


def gpu_busy(status=None) -> bool:
    status = status or gpu_status()
    if not status:
        return False
    used_threshold = int(os.environ.get("GPU_BUSY_MB", "1500"))
    util_threshold = int(os.environ.get("GPU_BUSY_UTIL", "25"))
    return status["used_mb"] >= used_threshold or status["utilization"] >= util_threshold


def warning_text(status=None) -> str:
    status = status or gpu_status()
    if not status or not gpu_busy(status):
        return ""
    return (f"WARNING: GPU looks busy ({status['name']}: "
            f"{status['used_mb']}/{status['total_mb']} MB used, "
            f"{status['utilization']}% util). A model-loading job may be slow or OOM.")


def warn_if_busy() -> str:
    text = warning_text()
    if text:
        print(text, file=sys.stderr, flush=True)
    return text
