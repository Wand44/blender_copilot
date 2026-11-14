#!/usr/bin/env python3
"""Deprecated launcher: use `python3 -m pipeline.model_correction`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.model_correction import main  # noqa: E402

if __name__ == "__main__":
    main()
