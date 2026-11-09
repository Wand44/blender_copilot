"""Combine and clean all data sources into one training file.

Groups:
  blender_core    doc_data + synthetic + model_corrections (one 70% group)
  general_python  basic Python (10%, so general coding doesn't atrophy)
  addons          whole source (small)
  general_usage   whole source (small)

Cleaning happens here, before rows are merged: legacy headings stripped, raw
code fenced, leaked prompts and rows marked validated: false dropped, duplicate
code/questions removed, then a per-type cap. There is no separate cleaner.
The output is question/answer only: data/final/train.jsonl.
"""
import json
import os
import random
import re
import sys
from collections import Counter
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import validator

SEED = 42
TYPE_CAP = 0.35

CONFIG = {
    "groups": {
        "blender_core": {
            # Docs are the bulk; generated rows are kept in full and docs fill
            # the rest of the group's target.
            "paths": [
                {"source": "doc_data", "path": str(config.DOC_FILE), "fill": True},
                {"source": "synthetic", "path": str(config.SYNTHETIC_FILE), "keep": True},
                {"source": "model_corrections",
                 "path": str(config.MODEL_CORRECTIONS_FILE), "keep": True},
            ],
            "ratio": 0.70,
        },
        "general_python": {
            "paths": [{"source": "general_python",
                       "path": str(config.GENERAL_PYTHON_FILE), "fill": True}],
            "ratio": 0.10,
        },
        "addons": {
            "paths": [{"source": "addons",
                       "path": str(config.ADDONS_FILE)}],
            "ratio": None,  # whole source
        },
        "general_usage": {
            "paths": [{"source": "general_usage",
                       "path": str(config.GENERAL_DATA_FILE)}],
            "ratio": None,  # whole source
        },
    },
    "target_total": 10000,
    "output": str(config.TRAIN_FILE),
    "val": str(config.VAL_FILE),
    "preferences": str(config.PREFERENCE_FILE),
}

random.seed(SEED)


def classify_strategy(q: str, a: str) -> str:
    """Best-effort task type for raw rows that don't carry one."""
    if "<|fim_prefix|>" in q:
        return "fim"
    if q.strip().startswith("Fix the bug in this Blender addon:"):
        return "bugfix"
    if q.strip().startswith("Refactor this Blender addon for clarity and performance:"):
        return "refactor"
    return "explanation"


def load_clean(source: str, path: str, whole: bool = False) -> List[Dict]:
    """Load, clean, validate, and tag one file. No rows written."""
    if not os.path.exists(path):
        print(f"[{source}] file not found, skipped: {path}")
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
            if str(r.get("type")) == "6":
                reasons["masked_completion"] += 1
                continue
            # Generated rows must be explicitly verified; legacy rows without
            # a "validated" key are still allowed through.
            validated = r.get("validated")
            if validated is False or (
                    "validated" in r
                    and source in ("model_corrections", "synthetic")
                    and validated is not True):
                reasons["marked_invalid"] += 1
                continue
            a = validator.normalize_answer(a)
            # normalize raw code-only answers (e.g. addons files) into fences
            if "```python" not in a and validator.is_raw_code(a):
                a = f"```python\n{a.strip()}\n```"
            if "seed_id" in r or "task_type" in r or "type" in r:
                strategy = config.task_type(r.get("type", 0))
            else:
                strategy = classify_strategy(q, a)
            ok, reason = validator.validate_sample(q, a, strategy, seen_hashes=None,
                                                   strict_format=False)
            if not ok:
                reasons[reason] += 1
                continue
            row_type = "reference" if reason == "ok_reference" else strategy
            types[row_type] += 1
            kept.append({
                "question": q,
                "answer": a,
                "_type": row_type,
                "_source": source,
                "_whole": whole,
                "_seed_ids": r.get("seed_ids") or [],
                "_small_ok": r.get("small_ok"),
                "_small_answer": r.get("small_answer"),
                "_small_error": r.get("small_error"),
                "_validated": r.get("validated"),
            })
    print(f"[{source}] {len(kept)} valid / {sum(reasons.values())} rejected "
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
        # Task types are different training signals even when they share code:
        # a bugfix answer and a doc explanation of the same snippet can coexist.
        h = (r.get("_type"), validator.code_hash(a))
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


def split_validation(rows: List[Dict], ratio: float = 0.03):
    """Hold out rows before dedup/cap, keeping each seed group together."""
    groups = {}
    for i, r in enumerate(rows):
        seeds = r.get("_seed_ids") or []
        key = tuple(seeds) if seeds else ("row", i)
        groups.setdefault(key, []).append(r)
    keys = list(groups)
    random.shuffle(keys)
    target = max(1, int(len(rows) * ratio)) if rows else 0
    val = []
    for key in keys:
        if len(val) >= target:
            break
        val.extend(groups[key])
    val_ids = {id(r) for r in val}
    return [r for r in rows if id(r) not in val_ids], val


def write_qa(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            # final schema is exactly question + answer
            f.write(json.dumps({"question": r["question"],
                                "answer": r["answer"]},
                               ensure_ascii=False) + "\n")


def write_preferences(path: str, rows: List[Dict]) -> None:
    prefs = [{"prompt": r["question"], "chosen": r["answer"],
              "rejected": r["_small_answer"], "small_error": r["_small_error"]}
             for r in rows
             if r.get("_source") == "model_corrections"
             and r.get("_small_ok") is False
             and r.get("_validated") is True
             and r.get("_small_answer")]
    if not prefs:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in prefs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[out] {path} ({len(prefs)} rows)")


def main():
    validator.load_cache()
    all_rows = []
    for name, group in CONFIG["groups"].items():
        whole = group["ratio"] is None
        keep_rows, fill_rows = [], []
        for item in group["paths"]:
            rows = load_clean(item["source"], item["path"], whole)
            if item.get("keep"):
                for r in rows:
                    r["_keep"] = True
                keep_rows.extend(rows)
            else:
                fill_rows.extend(rows)
        ratio = group["ratio"]
        if ratio is not None:
            target = int(CONFIG["target_total"] * ratio)
            random.shuffle(fill_rows)
            fill = max(0, target - len(keep_rows))
            if len(fill_rows) > fill:
                print(f"[{name}] kept {len(keep_rows)} generated + "
                      f"sampled {fill}/{len(fill_rows)} docs (ratio {ratio})")
                fill_rows = fill_rows[:fill]
        valid = keep_rows + fill_rows
        for r in valid:
            r["_group"] = name
        print(f"[{name}] {len(valid)} rows")
        all_rows.extend(valid)

    if not all_rows:
        sys.exit("No valid data from any source, aborting.")

    # Split is reserved before dedup/cap so validation rows are untouched.
    core = [r for r in all_rows if r.get("_group") == "blender_core"]
    rest = [r for r in all_rows if r.get("_group") != "blender_core"]
    core_train, val_rows = split_validation(core)
    print(f"[val] {len(val_rows)} rows held out from blender_core")
    write_preferences(CONFIG["preferences"], core)

    rows = dedup(core_train + rest)
    # Small sources and generated keep rows are never ratio-sampled or
    # per-type-capped; everything else goes through the cap.
    exempt = [r for r in rows if r.get("_whole") or r.get("_keep")]
    rows = cap_types([r for r in rows if not (r.get("_whole") or r.get("_keep"))]) + exempt
    random.shuffle(rows)

    write_qa(CONFIG["output"], rows)
    print(f"[out] {CONFIG['output']} ({len(rows)} rows)")
    write_qa(CONFIG["val"], val_rows)
    print(f"[out] {CONFIG['val']} ({len(val_rows)} rows)")

    print("      per type:",
          dict(Counter(r.get("_type", "?") for r in rows).most_common()))


if __name__ == "__main__":
    main()
