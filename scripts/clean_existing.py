#!/usr/bin/env python3
"""Re-validate the existing synthetic file with the current validator rules.

Keeps only rows that pass both format and Blender-API validation, removes
duplicates, and writes the cleaned file back in place (old version saved as
synthetic_training_data.jsonl.bak).
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validator
from collections import Counter

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
PATH = os.path.join(DATA, "synthetic_training_data.jsonl")
BACKUP = PATH + ".bak"


def main():
    validator.load_cache()
    if not os.path.exists(PATH):
        print(f"No file to clean: {PATH}")
        return

    kept, reasons, types = [], Counter(), Counter()
    seen_hashes = set()
    with open(PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                reasons["json_error"] += 1
                continue
            q, a = r.get("question", ""), r.get("answer", "")
            t = int(r.get("type", 0))
            ok, reason = validator.validate_sample(q, a, t, seen_hashes)
            if ok:
                kept.append(r)
                types[str(t)] += 1
            else:
                reasons[reason] += 1

    print(f"Cleaned {PATH}: {len(kept)} kept / {sum(reasons.values())} rejected")
    print("  per type:", dict(types.most_common()))
    print("  top rejects:", dict(reasons.most_common(10)))

    shutil.move(PATH, BACKUP)
    with open(PATH, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Original moved to {BACKUP}")


if __name__ == "__main__":
    main()
