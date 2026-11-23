"""Combine and clean every data source into train / val / preferred + manifest.

Cleaning happens here: legacy headings stripped, raw code fenced, leaked
prompts and unverified generated rows dropped, benchmark-prompt overlaps
dropped, duplicates removed, then a per-type cap.
"""
import json
import os
import random
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from core import config, validator
from core import code as code_mod
from core import dedupe as dedupe_mod
from core import rows as rows_mod

# Benchmark cases are read-only assets. They are never modified or used as
# training data; their prompts are only used to filter overlaps.
sys.path.insert(0, str(config.BENCH))
from cases import CASES  # noqa: E402

SEED = 42
TYPE_CAP = 0.35
# Decontamination thresholds: question vs benchmark prompt, answer code vs
# benchmark reference code.
PROMPT_MATCH_RATIO = 0.85
CODE_MATCH_RATIO = 0.9
RAW_SOURCES = {"docs", "addon", "general", "replay"}
SOURCE_MAP = {
    "doc_data": "docs",
    "synthetic": "synthetic",
    "model_corrections": "correction",
    "general_python": "general",
    "replay": "replay",
}

CONFIG = {
    "groups": {
        "blender_core": {
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
        "replay": {
            "paths": [{"source": "replay", "path": str(config.REPLAY_FILE)}],
            "ratio": None,
        },
    },
    "target_total": 1000,
    "output": str(config.TRAIN_FILE),
    "val": str(config.VAL_FILE),
    "preferences": str(config.PREFERENCE_FILE),
    "manifest": str(config.MANIFEST_FILE),
    "dropped": str(config.DROPPED_BENCHMARK_FILE),
}

random.seed(SEED)


def classify_strategy(question: str, answer: str) -> str:
    """Best-effort task type for raw rows that don't carry one."""
    if "<|fim_prefix|>" in question:
        return "fim"
    if question.strip().startswith("Fix the bug in this Blender addon:"):
        return "bugfix"
    if question.strip().startswith("Refactor this Blender addon for clarity and performance:"):
        return "refactor"
    return "explanation"


def _row_id(source_key: str, mapped: str, row: dict, line_no: int):
    # Always a string: mixed int/string ids break typed JSON readers.
    if row.get("id") not in (None, ""):
        return f"{mapped}:{row['id']}"
    return f"{mapped}:{line_no}"


def _producing_model(source_key: str, mapped: str, row: dict):
    model = row.get("producing_model") or row.get("final_model")
    if not model and row.get("small_ok") is True:
        model = row.get("small_model")
    if model:
        return model
    if source_key == "model_corrections":
        return "legacy"
    if mapped in RAW_SOURCES:
        return "raw"
    return "programmatic:fim" if str(row.get("type")) == "fim" else "unknown"


def load_clean(source: str, path: str, whole: bool = False):
    """Load, clean, validate, and tag one file. No rows written."""
    mapped = SOURCE_MAP.get(source, source)
    if not os.path.exists(path):
        print(f"[{source}] file not found, skipped: {path}")
        return []
    kept, reasons, types = [], Counter(), Counter()
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                reasons["json_error"] += 1
                continue
            question = row.get("question", "")
            answer = row.get("answer", "")
            messages = row.get("messages")
            if (not question or not answer) and isinstance(messages, list):
                for message in messages:
                    if message.get("role") == "user":
                        question = message.get("content", "")
                    elif message.get("role") == "assistant":
                        answer = message.get("content", "")
            if not isinstance(question, str) or not isinstance(answer, str) \
                    or not question.strip() or not answer.strip():
                reasons["empty_or_non_string"] += 1
                continue
            if str(row.get("type")) == "6":
                reasons["masked_completion"] += 1
                continue
            validated = row.get("validated")
            if validated is False or (
                    "validated" in row
                    and source in ("model_corrections", "synthetic")
                    and validated is not True):
                reasons["marked_invalid"] += 1
                continue
            answer = validator.normalize_answer(answer)
            if "```python" not in answer and validator.is_raw_code(answer):
                answer = f"```python\n{answer.strip()}\n```"
            if "seed_id" in row or "task_type" in row or "type" in row:
                strategy = config.task_type(row.get("type", 0))
            else:
                strategy = classify_strategy(question, answer)
            ok, reason = validator.validate_sample(question, answer, strategy,
                                                   seen_hashes=None,
                                                   strict_format=False)
            if not ok:
                reasons[reason] += 1
                continue
            row_type = "reference" if reason == "ok_reference" else strategy
            types[row_type] += 1
            kept.append({
                "id": _row_id(source, mapped, row, line_no),
                "question": question,
                "answer": answer,
                "source": mapped,
                "source_detail": source,
                "producing_model": _producing_model(source, mapped, row),
                "verified": bool(row.get("verified") is True or validated is True),
                "attempts": int(row.get("attempts") or 0),
                "timestamp": row.get("timestamp") or rows_mod.file_mtime_iso(path),
                "_type": row_type,
                "_source": source,
                "_whole": whole,
                "_seed_ids": row.get("seed_ids") or [],
                "_small_ok": row.get("small_ok"),
                "_small_answer": row.get("small_answer"),
                "_small_error": row.get("small_error"),
                "_validated": validated,
            })
    print(f"[{source}] {len(kept)} valid / {sum(reasons.values())} rejected "
          f"-> types {dict(types.most_common())}; top rejects: {dict(reasons.most_common(3))}")
    return kept


def norm_q(question: str) -> str:
    return dedupe_mod.norm_q(question)


FENCE_LABEL_RE = re.compile(r"```[A-Za-z]*")


def norm_question(text: str) -> str:
    """Lowercase, strip fences/punctuation, collapse whitespace."""
    return dedupe_mod.norm_q(FENCE_LABEL_RE.sub(" ", text or ""))


def norm_code(text: str) -> str:
    """Code only: fences unwrapped, comments and all whitespace removed."""
    blocks = code_mod.extract_fences(text)
    text = "\n".join(blocks) if blocks else (text or "")
    text = re.sub(r"#[^\n]*", "", text)
    return re.sub(r"\s+", "", text)


def _benchmark_cases():
    return [{
        "id": case["id"],
        "prompt": norm_question(case.get("prompt", "")),
        "code": norm_code(case.get("reference", "")),
    } for case in CASES]


def _match_benchmark(row, cases):
    """Return (case_id, rule, score) for the first decontamination hit."""
    question = norm_question(row.get("question", ""))
    for case in cases:
        if question and question == case["prompt"]:
            return case["id"], "question_exact", 1.0
    for case in cases:
        if not question or not case["prompt"]:
            continue
        ratio = SequenceMatcher(None, question, case["prompt"], autojunk=False).ratio()
        if ratio >= PROMPT_MATCH_RATIO:
            return case["id"], "question_similar", round(ratio, 3)
    code = norm_code(row.get("answer", ""))
    if code:
        for case in cases:
            if not case["code"]:
                continue
            if code == case["code"]:
                return case["id"], "code_exact", 1.0
            ratio = SequenceMatcher(None, code, case["code"], autojunk=False).ratio()
            if ratio >= CODE_MATCH_RATIO:
                return case["id"], "code_similar", round(ratio, 3)
    return None


def drop_benchmark_overlaps(rows, path: str = None):
    """Drop rows that copy a benchmark prompt, wording, or reference code."""
    path = path or CONFIG["dropped"]
    cases = _benchmark_cases()
    kept, dropped, rules = [], [], Counter()
    for row in rows:
        hit = _match_benchmark(row, cases)
        if hit:
            case_id, rule, score = hit
            rules[rule] += 1
            dropped.append({**row, "match_case": case_id,
                            "match_rule": rule, "score": score})
        else:
            kept.append(row)
    rows_mod.write_jsonl(path, dropped)
    per_rule = ", ".join(f"{name} {count}" for name, count in rules.most_common())
    print(f"[decontamination] rows before {len(rows)} -> after {len(kept)}; "
          f"dropped {len(dropped)} ({per_rule or 'none'}) -> {path}")
    return kept, dropped, rules


def dedup(rows):
    kept, stats = dedupe_mod.dedupe(rows)
    print(f"[dedup] -{stats['duplicate_code']} dup code, "
          f"-{stats['duplicate_question']} dup question, "
          f"-{stats['near_duplicate']} near-dup -> {len(kept)}")
    return kept


def cap_types(rows):
    by_type = {}
    for row in rows:
        by_type.setdefault(row.get("_type", "?"), []).append(row)
    for values in by_type.values():
        random.shuffle(values)
    total = sum(len(values) for values in by_type.values())
    cap = int(TYPE_CAP * total)
    final = [row for values in by_type.values() for row in values[:cap]]
    print(f"[cap {int(TYPE_CAP * 100)}%] {len(rows)} -> {len(final)}; "
          f"per type: {dict(Counter(row.get('_type', '?') for row in final).most_common())}")
    return final


def split_validation(rows, ratio: float = 0.03):
    """Hold out rows before dedup/cap, keeping each seed group together."""
    groups = {}
    for index, row in enumerate(rows):
        seeds = row.get("_seed_ids") or []
        key = tuple(seeds) if seeds else ("row", index)
        groups.setdefault(key, []).append(row)
    keys = list(groups)
    random.shuffle(keys)
    target = max(1, int(len(rows) * ratio)) if rows else 0
    val = []
    for key in keys:
        if len(val) >= target:
            break
        val.extend(groups[key])
    val_ids = {id(row) for row in val}
    return [row for row in rows if id(row) not in val_ids], val


def to_training_row(row: dict) -> dict:
    areas = rows_mod.api_areas(row["question"], row["answer"])
    return rows_mod.training_row(
        row["question"], row["answer"],
        id=row.get("id"), source=row["source"],
        producing_model=row.get("producing_model"),
        verified=bool(row.get("verified")),
        attempts=int(row.get("attempts") or 0),
        timestamp=row.get("timestamp"),
        areas=areas,
    )


def write_rows(path: str, rows) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_preferences(path: str, rows) -> list:
    prefs = []
    for row in rows:
        if row.get("_source") != "model_corrections":
            continue
        if row.get("_small_ok") is not False or row.get("_validated") is not True:
            continue
        if not row.get("_small_answer"):
            continue
        public = to_training_row(row)
        prefs.append({
            "id": public["id"],
            "source": "correction",
            "producing_model": public["producing_model"],
            "verified": public["verified"],
            "attempts": public["attempts"],
            "api_areas": public["api_areas"],
            "content_hash": public["content_hash"],
            "token_length": public["token_length"],
            "timestamp": public["timestamp"],
            "prompt": public["question"],
            "chosen": public["answer"],
            "rejected": row.get("_small_answer"),
            "small_error": row.get("_small_error"),
        })
    write_rows(path, prefs)
    print(f"[out] {path} ({len(prefs)} rows)")
    return prefs


def build_manifest(train_rows, val_rows, preferred_rows, dedupe_stats,
                   benchmark_dropped, benchmark_rules, rows_loaded) -> dict:
    sources = Counter(row["source"] for row in train_rows)
    verified = sum(1 for row in train_rows if row["verified"])
    raw_unverified = sum(1 for row in train_rows
                         if row["source"] in RAW_SOURCES and not row["verified"])
    api_area_counts = Counter()
    category_counts = Counter()
    for row in train_rows:
        api_area_counts.update(row["api_areas"])
        for category in rows_mod.api_categories(row["api_areas"], row["question"], row["answer"]):
            category_counts[category] += 1
    duplicate_total = sum(dedupe_stats.values())
    lengths = rows_mod.length_stats([row["token_length"] for row in train_rows])
    return {
        "generated_at": rows_mod.utc_now(),
        "counts": {"train": len(train_rows), "val": len(val_rows),
                   "preferred": len(preferred_rows), "rows_loaded": rows_loaded},
        "sources": dict(sources.most_common()),
        "verified": {
            "count": verified,
            "pct": round(100.0 * verified / len(train_rows), 1) if train_rows else 0.0,
            "not_execution_verified": len(train_rows) - verified,
            "raw_not_execution_verified": raw_unverified,
        },
        "duplicates_removed": {**dedupe_stats, "total": duplicate_total},
        "benchmark_overlap_dropped": len(benchmark_dropped),
        "benchmark_overlap_rules": dict(benchmark_rules),
        "length": lengths,
        "api_areas": dict(api_area_counts.most_common()),
        "api_categories": {name: category_counts.get(name, 0)
                           for name in rows_mod.API_AREA_GROUPS},
        "files": {name: path for name, path in (
            ("train", CONFIG["output"]), ("val", CONFIG["val"]),
            ("preferred", CONFIG["preferences"]),
            ("manifest", CONFIG["manifest"]),
            ("dropped_benchmark_overlap", CONFIG["dropped"]),
        )},
    }


def main():
    validator.load_cache()
    all_rows = []
    for name, group in CONFIG["groups"].items():
        whole = group["ratio"] is None
        keep_rows, fill_rows = [], []
        for item in group["paths"]:
            rows = load_clean(item["source"], item["path"], whole)
            if item.get("keep"):
                for row in rows:
                    row["_keep"] = True
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
        for row in valid:
            row["_group"] = name
        print(f"[{name}] {len(valid)} rows")
        all_rows.extend(valid)

    if not all_rows:
        sys.exit("No valid data from any source, aborting.")

    rows_loaded = len(all_rows)
    # Dedupe first, then decontaminate, so the split only ever sees clean rows.
    rows, dedupe_stats = dedupe_mod.dedupe(all_rows)
    print(f"[dedup] -{dedupe_stats['duplicate_code']} dup code, "
          f"-{dedupe_stats['duplicate_question']} dup question, "
          f"-{dedupe_stats['near_duplicate']} near-dup -> {len(rows)}")
    rows, benchmark_dropped, benchmark_rules = drop_benchmark_overlaps(rows)

    # Split is reserved before cap so validation rows are untouched.
    core = [row for row in rows if row.get("_group") == "blender_core"]
    rest = [row for row in rows if row.get("_group") != "blender_core"]
    core_train, val_rows = split_validation(core)
    print(f"[val] {len(val_rows)} rows held out from blender_core")
    preferred_rows = write_preferences(CONFIG["preferences"], core)

    train_pool = core_train + rest
    exempt = [row for row in train_pool if row.get("_whole") or row.get("_keep")]
    rows = cap_types([row for row in train_pool
                      if not (row.get("_whole") or row.get("_keep"))]) + exempt
    random.shuffle(rows)

    train_rows = [to_training_row(row) for row in rows]
    val_public = [to_training_row(row) for row in val_rows]
    write_rows(CONFIG["output"], train_rows)
    print(f"[out] {CONFIG['output']} ({len(train_rows)} rows)")
    write_rows(CONFIG["val"], val_public)
    print(f"[out] {CONFIG['val']} ({len(val_public)} rows)")

    manifest = build_manifest(train_rows, val_public, preferred_rows,
                              dedupe_stats, benchmark_dropped, benchmark_rules,
                              rows_loaded)
    Path(CONFIG["manifest"]).parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG["manifest"], "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"[out] {CONFIG['manifest']}")
    print("      per source:", manifest["sources"])
    print("      per type:",
          dict(Counter(row["source"] for row in train_rows).most_common()))


if __name__ == "__main__":
    main()
