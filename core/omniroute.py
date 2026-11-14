"""Start the local OmniRoute server when a pipeline step needs the model pool."""
import shutil
import subprocess
import time

import requests

BASE_URL = "http://localhost:20128"
MODELS_URL = BASE_URL + "/v1/models"


def is_up(timeout: int = 2) -> bool:
    """Any HTTP answer means the server is listening (401 without a key is fine)."""
    try:
        return requests.get(MODELS_URL, timeout=timeout).status_code < 500
    except requests.RequestException:
        return False


def ensure_running(wait: int = 120) -> tuple:
    """Return (ready, note). Starts `omniroute serve --daemon` if needed."""
    if is_up():
        return True, "already running"
    exe = shutil.which("omniroute")
    if not exe:
        return False, "omniroute CLI not found"
    try:
        subprocess.run(
            [exe, "serve", "--daemon", "--no-open", "--headless"],
            capture_output=True, text=True, timeout=90,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"could not start omniroute: {exc}"
    deadline = time.time() + wait
    while time.time() < deadline:
        if is_up():
            return True, "started"
        time.sleep(2)
    return False, f"omniroute did not become ready in {wait}s"
