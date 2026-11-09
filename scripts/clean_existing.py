#!/usr/bin/env python3
"""Re-validate the existing synthetic file with the current validator rules.

Keeps only rows that pass format, Blender-API, and (optionally) headless
execution validation. Drops all old type-6 masked-completion rows, removes
duplicates, and writes the cleaned file back in place (the first run saves the
original as synthetic_training_data.jsonl.bak; later runs keep that backup).
"""
import argparse
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validator
import execution_validator
from collections import Counter

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
PATH = os.path.join(DATA, "synthetic_training_data.jsonl")
BACKUP = PATH + ".bak"
# Old generator's rigid two-section template; the pipeline now emits short prose.
LEGACY_HEADING_RE = re.compile(
    r"^[12]\) (?:Architecture Overview|The Code)[ \t]*\n(?:[ \t]*\n)?", re.MULTILINE)


def normalize_answer(answer: str) -> str:
    return LEGACY_HEADING_RE.sub("", answer).lstrip("\n")


def main():
    parser = argparse.ArgumentParser(description="Clean the synthetic dataset in place")
    parser.add_argument("--execute", action="store_true",
                        help="also run every code answer in headless Blender")
    parser.add_argument("--timeout", type=int, default=300,
                        help="timeout per execution chunk (seconds)")
    parser.add_argument("--chunk", type=int, default=25,
                        help="codes per headless Blender process")
    args = parser.parse_args()

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
            if str(r.get("type")) == "6":
                reasons["masked_completion"] += 1
                continue
            q, a = r.get("question", ""), normalize_answer(r.get("answer", ""))
            r["answer"] = a
            t = r.get("type", 0)
            ok, reason = validator.validate_sample(q, a, t, seen_hashes,
                                                   strict_format=True)
            if ok:
                kept.append(r)
                types[str(t)] += 1
            else:
                reasons[reason] += 1

    if args.execute:
        codes = []
        for r in kept:
            if str(r.get("type")) == "fim":
                codes.append(validator.fim_full_code(r["question"], r["answer"]) or "")
            else:
                blocks = validator.extract_fences(r["answer"])
                codes.append(blocks[0] if blocks else "")
        results = []
        for i in range(0, len(codes), args.chunk):
            results.extend(execution_validator.run_batch(
                codes[i:i + args.chunk], timeout=args.timeout))
        executed = []
        for r, res in zip(kept, results):
            if res["ok"]:
                executed.append(r)
            else:
                reasons["exec_" + str(res.get("error_type", "Error"))] += 1
        kept = executed
        types = Counter(str(r.get("type", 0)) for r in kept)

    print(f"Cleaned {PATH}: {len(kept)} kept / {sum(reasons.values())} rejected")
    print("  per type:", dict(types.most_common()))
    print("  top rejects:", dict(reasons.most_common(10)))

    if os.path.exists(BACKUP):
        print(f"Keeping existing backup: {BACKUP}")
    else:
        shutil.move(PATH, BACKUP)
        print(f"Original moved to {BACKUP}")
    with open(PATH, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
