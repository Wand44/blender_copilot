"""Single source of truth for the Blender version used by the pipeline."""
import os

# Pin the target Blender version here. Docs, API cache and execution all use
# the same binary.
TARGET_BLENDER = os.environ.get("TARGET_BLENDER", "4.2")
BLENDER_BIN = os.environ.get("BLENDER_BIN", "blender")

# Default paths derived from the pinned version
DOC_DIR_DEFAULT = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw",
    f"blender_python_reference_{TARGET_BLENDER.replace('.', '_')}"
)
