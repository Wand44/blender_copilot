#!/usr/bin/env python3
"""Deprecated launcher: use `python3 tests/test_validator.py`."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "tests" / "test_validator.py"), run_name="__main__")
