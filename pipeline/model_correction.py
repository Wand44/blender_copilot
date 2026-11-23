#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""Small-model -> verification -> sequential big-model correction pipeline.

Per question, strictly in order:
  1. ask the small model;
  2. verify with the static validator + a fresh headless Blender;
  3. if it fails, ask the big model to fix that exact code with the error,
     up to 3 attempts on the same question;
  4. stop on the first pass, otherwise mark the row failed and move on.

Rows keep a stable integer id and are resumable. Every attempt is saved, so a
crash loses at most the attempt in flight.

Usage:
    python3 -m pipeline.model_correction --rows 20
    python3 -m pipeline.model_correction --generate 20   # add new questions
    python3 -m pipeline.model_correction --purge
    python3 -m pipeline.model_correction --revalidate
"""
import argparse
import os
import re
import shutil
import sys
import threading
from collections import Counter
from pathlib import Path

import requests

from core import config, execution_validator, validator
from core import verify as verify_mod
from core.gpu import warn_if_busy
from core.progress import emit
from core.code import content_hash
from core.rows import append_jsonl, read_jsonl, utc_now, write_jsonl
import pipeline.generate_synthetic as big

SEEDS = config.DOC_FILE
OUTPUT = config.MODEL_CORRECTIONS_FILE

_FILE_LOCK = threading.Lock()

SMALL_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "blender-copilot-v3.2")
MAX_ATTEMPTS = 3

SMALL_SYSTEM = (
    "You are a Blender 5.x Python coding assistant. Answer with a short explanation "
    "followed by exactly one ```python code block and nothing after it. The code must "
    "run standalone in a fresh headless Blender scene (an active default cube exists) "
    "and use only real Blender 5.x Python APIs."
)

CORRECTION_PROMPT = """Rewrite the CORRECT answer for this Blender 5.x question.

Question:
{question}

An answer failed. It was:
{attempt}

It failed the headless-Blender check with: {reason}

Respond in EXACTLY this delimiter format, no JSON, no prose outside the sections:
===ANSWER===
a short explanation, then exactly one ```python fenced block, and nothing after the block
===CHECK===
plain Python (no fences) that runs after the answer in the same namespace, sets up
any scene state it needs, calls the answer's function(s) with sample arguments, and
asserts read-back values (modifier type, property values, object count).

The code must run standalone in a fresh headless Blender scene (an active default cube
exists) and use only real Blender 5.x Python APIs. Do not reuse the failed code.
Requirements: if the question asks for a function, the answer must be functions with
explicit arguments (objects passed in) and no hardcoded scene lookups; the check must
call the answer's functions or inspect results, never reset the scene, and derive from
what the question asks. For class/register answers, the check must invoke the behaviour
(e.g. bpy.ops call) and assert results, not just registration. No invented properties;
hasattr guards to dodge missing APIs are not allowed."""

GEN_FORMAT = (
    "Respond in EXACTLY this delimiter format, no JSON, no prose outside the requested sections:\n"
    "===QUESTION===\n<question text>\n"
    "===ANSWER===\n<answer text>\n"
    "===CHECK===\n<short Python check>\n"
    "The answer must be a short explanation followed by exactly one ```python fenced block "
    "and end there. The check must be plain Python (no fences, no prose) that runs after the "
    "answer in the same namespace: create any scene state it needs, call the answer's "
    "function(s) with sample arguments, and assert read-back values (modifier type, property "
    "values, object count). Keep it short.\n"
    "Requirements for function questions: the question must name the function and its "
    "signature when it asks for a function (e.g. \"Write `get_camera_names() -> list[str]` that ...\"). "
    "The answer must be functions with explicit arguments (objects/trees passed in) and no "
    "hardcoded scene lookups such as bpy.data.objects['Cube']. For simple beginner one-liners "
    "that are scripts, the check must inspect the state the script left behind, without deleting "
    "or resetting the scene.\n"
    "The check must call the answer's functions or inspect their results; it must never "
    "reimplement the task itself, never reset or clear the scene first, and must be derived "
    "from what the question asks (return type, names, values), not from what the answer happens "
    "to do. The check may only use object/data names the question or the answer's code creates.\n"
    "The headless harness has no camera and exactly one default cube named \"Cube\". "
    "For class/register answers (operators, panels, render engines), the check must invoke "
    "the class behaviour (e.g. the bpy.ops call for an operator, or instantiate and call its "
    "methods) and assert results, not just assert registration.\n"
    "Every attribute used in the answer must exist in Blender 5.x; guarding with hasattr to "
    "dodge a missing API is not allowed."
)

DIFFICULTIES = [
    ("beginner", 1,
     "A single-step task a beginner meets first: add a primitive, move or "
     "scale an object, set one property. Example: 'How do I add a cube?'"),
    ("intermediate", 1,
     "A small multi-step task: materials, modifiers, collections, or a simple "
     "operator with a few settings and a read-back result."),
    ("advanced", 2,
     "Combine the two references into one focused tool or addon feature with "
     "several steps and a clear, checkable outcome."),
    ("expert", 3,
     "Combine all three references into a realistic mini-addon or pipeline "
     "with edge cases, cleanup, and several interacting properties."),
]

evaluate = verify_mod.evaluate
_empty_info = verify_mod._empty_info


def parse_answer(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"===CHECK===.*", "", text, flags=re.DOTALL)
    match = re.search(r"===ANSWER===\s*(.*)", text, re.DOTALL)
    return (match.group(1) if match else text).strip()


def parse_check(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"===CHECK===\s*(.*)", text, re.DOTALL)
    if not match:
        return ""
    check = match.group(1).strip()
    if check.startswith("```"):
        check = re.sub(r"^```[a-zA-Z]*\n?", "", check)
        check = re.sub(r"\n?```$", "", check)
    return check.strip()


def parse_generation(text: str) -> dict:
    if not text:
        return {}
    question = re.search(r"===QUESTION===\s*(.*?)\s*===ANSWER===", text, re.DOTALL)
    answer = parse_answer(text)
    if not question or not answer:
        return {}
    return {"question": question.group(1).strip(), "reference": answer,
            "reference_check": parse_check(text)}


def _next_id(rows) -> int:
    ids = [r.get("id") for r in rows if isinstance(r.get("id"), int)]
    return (max(ids) + 1) if ids else 1


def migrate_rows(rows, small_model: str = SMALL_MODEL) -> bool:
    """Assign stable ids and normalize legacy rows. Returns True if changed."""
    changed = False
    next_id = _next_id(rows)
    seen = set()
    for row in rows:
        if not isinstance(row.get("id"), int):
            row["id"] = next_id
            next_id += 1
            changed = True
        if row["id"] in seen:
            row["id"] = next_id
            next_id += 1
            changed = True
        seen.add(row["id"])
        if not row.get("producing_model"):
            row["producing_model"] = "legacy" if row.get("validated") is True else "unknown"
            changed = True
        if not row.get("status"):
            if row.get("validated") is True:
                row["status"] = "passed"
            elif row.get("validated") is False:
                row["status"] = "failed"
            else:
                row["status"] = "pending"
            changed = True
        if "verified" not in row:
            row["verified"] = bool(row.get("validated") is True)
            changed = True
        if "attempts" not in row or row.get("attempts") is None:
            row["attempts"] = 0
            changed = True
        if "error_history" not in row:
            row["error_history"] = []
            changed = True
        if "small_model" not in row:
            row["small_model"] = small_model if row.get("small_answer") else None
            changed = True
        if "small_ok" not in row:
            row["small_ok"] = None
            changed = True
        if not row.get("timestamp"):
            row["timestamp"] = utc_now()
            changed = True
    return changed


def append_row(path: Path, row: dict) -> None:
    append_jsonl(path, row)


def save_row(path: Path, question=None, row=None) -> None:
    """Replace the row with the same id (or question) and append if missing."""
    if row is None:
        row = question
        question = row.get("question")
    with _FILE_LOCK:
        rows = read_jsonl(path)
        for index, existing in enumerate(rows):
            same_id = row.get("id") is not None and existing.get("id") == row.get("id")
            # Only fall back to the question when the new row has no id; with ids
            # present, a duplicate question must not overwrite a different row.
            same_q = row.get("id") is None and existing.get("question") == (
                question or row.get("question"))
            if same_id or same_q:
                rows[index] = row
                break
        else:
            rows.append(row)
        write_jsonl(path, rows)


def remove_row(path: Path, question: str) -> None:
    with _FILE_LOCK:
        write_jsonl(path, [r for r in read_jsonl(path) if r.get("question") != question])


def _lock_write(path: Path, rows) -> None:
    with _FILE_LOCK:
        write_jsonl(path, rows)


def purge(path: Path = OUTPUT) -> None:
    """Back up, drop leaked/duplicate questions, tag the rest pending."""
    if not path.exists():
        raise SystemExit(f"No file to purge: {path}")
    rows = read_jsonl(path)
    before = len(rows)

    kept, leaked = [], 0
    for row in rows:
        question = row.get("question") or ""
        if "```" in question or "combine these into one" in question.lower():
            leaked += 1
            continue
        kept.append(row)

    by_question = {row.get("question"): row for row in kept}
    deduped = list(by_question.values())
    duplicates = len(kept) - len(deduped)

    for row in deduped:
        row["validated"] = "unverified"
        row["verified"] = False
        row["status"] = "pending"
        row["attempts"] = 0
        row["error_history"] = []

    shutil.copy2(path, str(path) + ".bak")
    _lock_write(path, deduped)
    print(f"Purged {path}: {before} -> {len(deduped)} rows "
          f"(-{leaked} leaked, -{duplicates} duplicate question); "
          f"backup: {path}.bak")


def revalidate(path: Path = OUTPUT) -> None:
    """Re-run the shared verification path over existing rows."""
    validator.load_cache()
    if not path.exists():
        raise SystemExit(f"No file to revalidate: {path}")
    rows = read_jsonl(path)
    migrate_rows(rows)
    passed = 0
    for index, row in enumerate(rows, 1):
        verify_mod.revalidate_row(row)
        if row.get("status") == "passed":
            passed += 1
        emit(index, len(rows), f"revalidating id {row.get('id')}")
    _lock_write(path, rows)
    print(f"Revalidated {len(rows)} rows: {passed} pass, {len(rows) - passed} fail")


def build_question(seeds, buckets, known, difficulty):
    """Ask the big LLM for one new question at the requested difficulty."""
    import random
    label, seed_count, guidance = difficulty
    indices = list(range(len(seeds)))
    random.shuffle(indices)
    for i in indices:
        seed = seeds[i]
        peers = [j for j in buckets[big.topic_key(seed)] if j != i]
        if len(peers) < seed_count - 1:
            continue
        chosen = random.sample(peers, seed_count - 1)
        refs = [seed] + [seeds[j] for j in chosen]
        prompt = (
            "Reference snippets (the only allowed API source):\n\n"
            + "\n\n".join(f"- {s['question']}: {s['answer']}" for s in refs)
            + f"\n\nDifficulty: {label}. {guidance}\n"
              "The question asks for the task; the answer provides the reference "
              "implementation.\n"
            + big.GROUNDING + "\n" + GEN_FORMAT)
        result = parse_generation(big.call_llm(prompt, big.ARCH_SYSTEM))
        if not result:
            continue
        question = result["question"]
        if "```" in question or "combine these into one" in question.lower():
            continue
        check = result.get("reference_check", "")
        if not check or not validator.parse_ok(check):
            continue
        ok, _ = validator.validate_sample(question, result["reference"], "explanation",
                                          strict_format=True)
        if not ok or question in known:
            continue
        ok, _, _ = evaluate(question, result["reference"], check, require_check=True)
        if not ok:
            continue
        return {"question": question, "reference": result["reference"],
                "reference_check": check, "seed_ids": [i] + chosen,
                "difficulty": label}
    return None


def ask_small(question: str, timeout: int = None, retries: int = None):
    timeout = timeout or int(os.environ.get("SMALL_MODEL_TIMEOUT", "300"))
    attempts = retries if retries is not None else int(os.environ.get("MODEL_RETRIES", "2"))
    payload = {
        "model": SMALL_MODEL,
        "think": False,
        "stream": False,
        "messages": [
            {"role": "system", "content": SMALL_SYSTEM},
            {"role": "user", "content": question},
        ],
        "options": {"temperature": 0.2, "num_predict": 1200},
    }
    for _attempt in range(max(1, attempts)):
        try:
            response = requests.post(SMALL_URL, json=payload, timeout=timeout)
            response.raise_for_status()
            return (response.json().get("message", {}).get("content") or "").strip(), SMALL_MODEL
        except Exception:
            continue
    return "", SMALL_MODEL


def correct(question: str, small_answer: str, reason: str,
            reference: str = "", reference_check: str = "",
            timeout: int = None, retries: int = None,
            max_attempts: int = MAX_ATTEMPTS, start_attempt: int = 1,
            last_candidate: str = "", error_history=None, status=None):
    """Sequential big-model fix attempts on the same question."""
    history = list(error_history or [])
    attempt_code = last_candidate or small_answer
    why = reason
    info = _empty_info()
    # The big model already wrote a reference answer with the question: try it
    # before spending another call (and an execution run) on a rewrite.
    if reference:
        ref_ok, _ref_why, ref_info = evaluate(
            question, reference, reference_check,
            require_check=bool(reference_check.strip()))
        if ref_ok:
            return reference, reference_check, ref_info, history, "reference", start_attempt
    for attempt in range(max(1, start_attempt), max_attempts + 1):
        if status:
            status("fixing with big model", attempt)
        prompt = CORRECTION_PROMPT.format(question=question,
                                          attempt=attempt_code[:4000],
                                          reason=why)
        text, model = big.call_llm_ex(prompt, big.ARCH_SYSTEM, timeout=timeout,
                                      max_attempts=retries)
        candidate, check = parse_answer(text), parse_check(text)
        entry = {"attempt": attempt, "model": model, "error": None,
                 "timestamp": utc_now()}
        if not candidate:
            entry["error"] = "empty_response"
            history.append(entry)
            why = "the big model returned no answer"
            continue
        use_check = check or reference_check
        ok, failure, info = evaluate(question, candidate, use_check,
                                     require_check=bool(use_check.strip()))
        entry["error"] = None if ok else failure
        history.append(entry)
        if ok:
            return candidate, use_check, info, history, model, attempt
        attempt_code, why = candidate, failure
    info["correction_error"] = why
    info["last_candidate"] = attempt_code
    return None, "", info, history, None, max_attempts


def process(item: dict, status=None, small_answer: str = None,
            timeout: int = None, retries: int = None,
            max_attempts: int = MAX_ATTEMPTS):
    """Ask, verify, then correct one question. Saves the row after each attempt."""
    question = item["question"]
    row_id = item.get("id")
    reference_check = item.get("reference_check") or item.get("check") or ""
    small_model = item.get("small_model") or SMALL_MODEL

    if small_answer is None:
        small_answer = item.get("small_answer")
    if small_answer is None:
        if status:
            status("asking small model")
        small_answer, small_model = ask_small(question, timeout=timeout, retries=retries)

    if status:
        status("verifying small answer")
    ok, reason, info = evaluate(question, small_answer, reference_check,
                                require_check=bool(reference_check.strip()))

    attempts = int(item.get("attempts") or 0)
    history = list(item.get("error_history") or [])
    corrected = False
    final = small_answer
    final_check = reference_check
    final_model = small_model if ok else None

    if not ok and attempts < max_attempts:
        if status:
            status(f"small model failed: {reason}")
        result, check, cinfo, chistory, cmodel, used = correct(
            question, small_answer, reason,
            item.get("reference", ""), reference_check,
            timeout=timeout, retries=retries, max_attempts=max_attempts,
            start_attempt=attempts + 1,
            last_candidate=item.get("last_candidate") or small_answer,
            error_history=history, status=status,
        )
        history = chistory
        attempts = used
        info = cinfo
        if result is not None:
            final, final_check = result, check
            final_model = cmodel
            corrected = True

    if ok or corrected:
        status_key = "passed" if ok else "corrected"
        validated = True
    else:
        status_key = "failed"
        validated = False
        final = info.get("last_candidate") or item.get("last_candidate") or final
        final_check = final_check or reference_check

    row = {
        "id": row_id,
        "question": question,
        "answer": final,
        "check": final_check,
        "status": status_key,
        "validated": validated,
        "verified": validated is True,
        "attempts": attempts,
        "error_history": history,
        "small_model": small_model,
        "small_answer": small_answer,
        "small_ok": ok,
        "small_error": None if ok else reason,
        "producing_model": final_model or item.get("producing_model") or "none",
        "final_model": final_model,
        "corrected": corrected,
        "coverage": round(float(info.get("coverage", 0.0)), 3),
        "exec_nonfatal": bool(info.get("exec_nonfatal")),
        "source": "correction",
        "difficulty": item.get("difficulty", ""),
        "seed_ids": item.get("seed_ids", []),
        "reference": item.get("reference", ""),
        "reference_check": reference_check,
        "correction_error": info.get("correction_error") if not validated else None,
        "last_candidate": final if not validated else None,
        "content_hash": content_hash(question, final or ""),
        "timestamp": item.get("timestamp") or utc_now(),
    }
    save_row(OUTPUT, row)
    return row, status_key


def _select_pending(rows, max_attempts):
    pending = []
    for row in rows:
        if row.get("validated") is True or row.get("status") == "passed":
            continue
        if row.get("status") == "failed" and int(row.get("attempts") or 0) >= max_attempts:
            continue
        pending.append(row)
    return pending


def main():
    parser = argparse.ArgumentParser(description="Sequential small->big correction pipeline")
    parser.add_argument("--rows", type=int, default=0,
                        help="process at most N unfinished rows (0 = all)")
    parser.add_argument("--generate", type=int, default=0,
                        help="generate N new questions before processing")
    parser.add_argument("--questions", type=int, default=0,
                        help="target unfinished questions this run; new questions "
                             "fill the gap (0 = process only)")
    parser.add_argument("--workers", type=int, default=1,
                        help="ignored: correction is strictly sequential")
    parser.add_argument("--timeout", type=int,
                        default=int(os.environ.get("CORRECTION_TIMEOUT", "180")),
                        help="seconds per big-model call")
    parser.add_argument("--small-timeout", type=int,
                        default=int(os.environ.get("SMALL_MODEL_TIMEOUT", "300")),
                        help="seconds per small-model call")
    parser.add_argument("--retries", type=int,
                        default=int(os.environ.get("MODEL_RETRIES", "2")),
                        help="provider retries per call")
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS,
                        help="big-model correction attempts per question")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--purge", action="store_true",
                        help="drop leaked/duplicate rows and tag the rest pending")
    action.add_argument("--revalidate", action="store_true",
                        help="re-run verification over existing rows")
    args = parser.parse_args()

    if args.purge:
        purge()
        return
    if args.revalidate:
        warn_if_busy()
        revalidate()
        return

    rows = read_jsonl(OUTPUT)
    if migrate_rows(rows):
        _lock_write(OUTPUT, rows)
    next_id = _next_id(rows)
    total = len(rows)

    generate_count = args.generate
    if not generate_count:
        pending = _select_pending(rows, args.max_attempts)
        generate_count = max(0, args.questions - len(pending))
    if generate_count:
        validator.load_cache()
        big.POOL.refresh(big._headers())
        warn_if_busy()
        seeds = read_jsonl(SEEDS)
        if not seeds:
            raise SystemExit(f"No seeds in {SEEDS}")
        buckets = {}
        for index, seed in enumerate(seeds):
            buckets.setdefault(big.topic_key(seed), []).append(index)
        known = {row.get("question") for row in rows}
        for slot in range(generate_count):
            difficulty = DIFFICULTIES[slot % len(DIFFICULTIES)]
            print(f"[gen {slot + 1}/{generate_count}] asking model ({difficulty[0]}) ...",
                  flush=True)
            item = build_question(seeds, buckets, known, difficulty)
            if not item:
                print("  could not generate another usable question", flush=True)
                continue
            row = {"id": next_id, "question": item["question"], "answer": "",
                   "reference": item.get("reference", ""),
                   "reference_check": item.get("reference_check", ""),
                   "check": "", "status": "pending", "validated": "unverified",
                   "verified": False, "attempts": 0, "error_history": [],
                   "small_model": None, "small_ok": None, "producing_model": None,
                   "final_model": None, "source": "correction",
                   "difficulty": item.get("difficulty", ""),
                   "seed_ids": item.get("seed_ids", []), "timestamp": utc_now()}
            append_row(OUTPUT, row)
            print(f"[gen {slot + 1}/{generate_count}] id {next_id}: "
                  f"{item['question'][:70]}", flush=True)
            emit(slot + 1, generate_count, f"generated id {next_id}",
                 id=next_id, phase="generate")
            known.add(item["question"])
            rows.append(row)
            next_id += 1
            total += 1

    pending = _select_pending(rows, args.max_attempts)
    if args.rows > 0:
        pending = pending[:args.rows]
    if not pending:
        print(f"Nothing to do: {total} rows on disk, all passed or exhausted.")
        return

    validator.load_cache()
    if not big.POOL.models:
        big.POOL.refresh(big._headers())
    warn_if_busy()
    print(f"Big model(s): {', '.join(big.POOL.models)}")
    print(f"Small model:  {SMALL_MODEL} @ {SMALL_URL}")
    print(f"Dataset: {total} rows; processing {len(pending)} unfinished row(s) sequentially")

    counts = Counter()
    for index, item in enumerate(pending, 1):
        row_id = item.get("id")
        def status(message, attempt=None, _row_id=row_id):
            suffix = f" attempt {attempt}/{args.max_attempts}" if attempt else ""
            print(f"[id {_row_id}/{total}] {message}{suffix}", flush=True)
        row, key = process(item, status, timeout=args.timeout, retries=args.retries,
                           max_attempts=args.max_attempts)
        counts[key] += 1
        print(f"[id {row_id}/{total}] {key} "
              f"(attempts {row.get('attempts')}/{args.max_attempts}, "
              f"producer {row.get('producing_model')})", flush=True)
        emit(index, len(pending), f"id {row_id}: {key}",
             id=row_id, status=key, dataset_total=total,
             attempts=row.get("attempts"), producing_model=row.get("producing_model"))

    print(f"Done: {counts['passed']} passed, {counts['corrected']} corrected, "
          f"{counts['failed']} failed -> {OUTPUT}")


if __name__ == "__main__":
    main()
