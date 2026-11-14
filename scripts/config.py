"""Deprecated launcher: config lives in core/config.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import *  # noqa: F401,F403,E402
