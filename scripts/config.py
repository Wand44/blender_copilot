"""Single source of truth for the project: Blender version, paths, task types."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load .env without python-dotenv (not installed on the system python).
if (ROOT / ".env").exists():
    for _line in (ROOT / ".env").read_text().splitlines():
        _k, _, _v = _line.partition("=")
        if _k.strip() and _v.strip():
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

DATA = ROOT / "data"
RAW = DATA / "raw"          # hand-collected / extracted sources
GENERATED = DATA / "generated"  # pipeline outputs
FINAL = DATA / "final"      # training-ready output
LOGS = ROOT / "logs"

# Pin the target Blender version here. Docs, API cache and execution all use
# the same binary.
TARGET_BLENDER = os.environ.get("TARGET_BLENDER", "5.1.1")
BLENDER_BIN = os.environ.get("BLENDER_BIN", "blender")

# Raw sources
DOC_FILE = RAW / "doc_data.jsonl"
ADDONS_FILE = RAW / "addons_data.jsonl"
GENERAL_DATA_FILE = RAW / "general_data.jsonl"
GENERAL_PYTHON_FILE = RAW / "general_python.jsonl"
DOC_DIR_DEFAULT = RAW / f"blender_python_reference_{TARGET_BLENDER.replace('.', '_')}"

# Generated sources
CACHE_PATH = GENERATED / "bpy_api_cache.json"
SYNTHETIC_FILE = GENERATED / "synthetic_training_data.jsonl"
MODEL_CORRECTIONS_FILE = GENERATED / "model_corrections.jsonl"

# Training-ready output (question/answer only)
TRAIN_FILE = FINAL / "train.jsonl"
VAL_FILE = FINAL / "val.jsonl"
PREFERENCE_FILE = FINAL / "preference_pairs.jsonl"

LOG_FILE = LOGS / "generation.log"

# One taxonomy for every stage. Keys are the generator's numeric types.
TASK_TYPES = {
    3: "explanation",
    4: "explanation",
    5: "bugfix_explained",
    6: "fim",
    7: "bugfix",
    8: "refactor",
}
TASK_TYPE_NAMES = set(TASK_TYPES.values()) | {"addon"}


def task_type(strategy) -> str:
    """Normalize a generator type (int) or a task-type name (str)."""
    if isinstance(strategy, str) and strategy in TASK_TYPE_NAMES:
        return strategy
    try:
        return TASK_TYPES.get(int(strategy), "explanation")
    except (TypeError, ValueError):
        return "explanation"
