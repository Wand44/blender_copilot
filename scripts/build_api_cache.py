#!/usr/bin/env python3
"""Deprecated launcher: use `blender -b --python core/build_api_cache.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.build_api_cache import main  # noqa: E402

if __name__ == "__main__":
    main()
