"""Combine all data sources into the final train/validation dataset.

Reads the RAW sources (validated here, in memory - no intermediate files):
valid rows are kept, tagged with type/source, ratio-sampled, deduplicated
across sources, capped at 35% per type, and split 95/5 (seed 42).

Outputs exactly one file: data/final_dataset.jsonl (95/5 split is carried
as a "split" field on each row: "train" or "validation").
"""
import json
import os
import random
import re
import sys
from collections import Counter
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validator

DATA = "/home/mintlover/Documents/python/blender_copilot/data"
SEED = 42
TYPE_CAP = 0.35
VAL_SPLIT = 0.05

CONFIG = {
    "paths": {
        "synthetic": os.path.join(DATA, "synthetic_training_data.jsonl"),
        "raw_addons": os.path.join(DATA, "addons_data.jsonl"),
        "raw_general": os.path.join(DATA, "doc_data.jsonl"),
        "general_python": os.path.join(DATA, "general_python.jsonl"),
        "general_usage": os.path.join(DATA, "general_data.jsonl"),
    },
    "ratios": {
        "synthetic": 0.35,
        "raw_addons": 0.1,
        "raw_general": 0.35,
        "general_python": 0.1,
        "general_usage": 0.1,
    },
    "target_total": 10000,
    "output": os.path.join(DATA, "final_dataset.jsonl"),
}

random.seed(SEED)


def classify_strategy(q: str, a: str) -> int:
    """Best-effort type for rows that don't carry one (raw doc/addon sources).
    Only assign a synthetic strategy when the row actually follows that format;
    otherwise treat as 0 (unknown/prose) so the strict synthetic-format guard
    does not reject raw/API-reference rows."""
    if a.strip().startswith("1) Architecture Overview") and "2) The Code" in a:
        return 3
    if "<MASK>" in q:
        return 6
    if q.strip().startswith("Fix the bug in this Blender addon:"):
        return 7
    if q.strip().startswith("Refactor this Blender addon for clarity and performance:"):
        return 8
    if validator.extract_fences(a):
        return 0  # raw code-only answer: validate as unknown
    return 0  # prose / unknown


def load_validated(key: str, path: str) -> List[Dict]:
    """Load a raw source, validate every row, tag kept rows. No files written."""
    if not os.path.exists(path):
        print(f"[{key}] file not found, skipped: {path}")
        return []
    kept, reasons, types = [], Counter(), Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                reasons["json_error"] += 1
                continue
            q, a = r.get("question", ""), r.get("answer", "")
            # New v2 schema stores messages instead of question/answer
            msgs = r.get("messages")
            if (not q or not a) and isinstance(msgs, list):
                for m in msgs:
                    if m.get("role") == "user":
                        q = m.get("content", "")
                    elif m.get("role") == "assistant":
                        a = m.get("content", "")
            if not isinstance(q, str) or not isinstance(a, str) \
                    or not q.strip() or not a.strip():
                reasons["empty_or_non_string"] += 1
                continue
            # normalize raw code-only answers (e.g. addons files) into fences
            if "```python" not in a and validator.is_raw_code(a):
                a = f"```python\n{a.strip()}\n```"
            if "seed_id" in r or "task_type" in r:  # synthetic rows know their type
                strategy = int(r.get("type", 0))
            else:
                strategy = classify_strategy(q, a)
            ok, reason = validator.validate_sample(q, a, strategy, seen_hashes=None,
                                                   strict_format=False)
            if not ok:
                reasons[reason] += 1
                continue
            row_type = "reference" if reason == "ok_reference" else str(strategy)
            types[row_type] += 1
            # Keep Q/A pair plus chat-style messages. `_type`/`_source` are only
            # used internally for capping and are stripped before writing.
            kept.append({
                "question": q,
                "answer": a,
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
                "_type": row_type,
                "_source": key,
            })
    print(f"[{key}] {len(kept)} valid / {sum(reasons.values())} rejected "
          f"-> types {dict(types.most_common())}; top rejects: {dict(reasons.most_common(3))}")
    return kept


def norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", q.lower())).strip()


def dedup(rows: List[Dict]) -> List[Dict]:
    seen_hash, seen_q, seen_words = set(), set(), set()
    kept = []
    d_code = d_q = d_near = 0
    for r in rows:
        q, a = r.get("question", ""), r.get("answer", "")
        h = validator.code_hash(a)
        nq = norm_q(q)
        ws = set(nq.split())
        nw = " ".join(sorted(ws)) if len(ws) >= 4 else None
        if h in seen_hash:
            d_code += 1
            continue
        if nq in seen_q:
            d_q += 1
            continue
        if nw and nw in seen_words:
            d_near += 1
            continue
        seen_hash.add(h)
        seen_q.add(nq)
        if nw:
            seen_words.add(nw)
        kept.append(r)
    print(f"[dedup] -{d_code} dup code, -{d_q} dup question, -{d_near} near-dup -> {len(kept)}")
    return kept


def cap_types(rows: List[Dict]) -> List[Dict]:
    by_type = {}
    for r in rows:
        by_type.setdefault(r.get("_type", "?"), []).append(r)
    for v in by_type.values():
        random.shuffle(v)
    # Cap every type at 35% of the pre-cap total; keep smaller types untouched.
    total = sum(len(v) for v in by_type.values())
    cap = int(TYPE_CAP * total)
    final = [r for v in by_type.values() for r in v[:cap]]
    print(f"[cap {int(TYPE_CAP*100)}%] {len(rows)} -> {len(final)}; "
          f"per type: {dict(Counter(r.get('_type', '?') for r in final).most_common())}")
    return final


def main():
    validator.load_cache()
    all_rows = []
    for key, path in CONFIG["paths"].items():
        ratio = CONFIG["ratios"].get(key, 0)
        if ratio <= 0:
            continue
        valid = load_validated(key, path)
        target = int(CONFIG["target_total"] * ratio)
        if len(valid) > target:
            random.shuffle(valid)
            valid = valid[:target]
            print(f"[{key}] sampled down to {target} (ratio {ratio})")
        all_rows.extend(valid)

    if not all_rows:
        sys.exit("No valid data from any source, aborting.")
    rows = cap_types(dedup(all_rows))
    random.shuffle(rows)

    n_val = max(1, int(len(rows) * VAL_SPLIT))
    for i, r in enumerate(rows):
        r["split"] = "validation" if i < n_val else "train"

    # Write only the Q/A pair (chat format) and split; drop internal metadata.
    with open(CONFIG["output"], "w", encoding="utf-8") as f:
        for r in rows:
            out = {
                "question": r["question"],
                "answer": r["answer"],
                "messages": r["messages"],
                "split": r["split"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"[out] {CONFIG['output']} ({len(rows)} rows: "
          f"{len(rows) - n_val} train + {n_val} validation)")
    print("      per type:", dict(Counter(r.get('_type', '?') for r in rows).most_common()))


if __name__ == "__main__":
    main()
