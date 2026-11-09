#!/usr/bin/env python3
"""Type-4 data pipeline: big LLM -> small model -> headless Blender -> big LLM fix.

1. OmniRoute (big LLM) writes a list of type-4 Blender questions (reusing the
   topic-bucketed generate_type_4 prompt from generate_synthetic.py) and a
   reference answer for each.
2. qwen3.5:4b via Ollama answers every question.
3. The answer's code block runs in a fresh headless Blender (execution_validator).
4. If the answer is wrong (format, invented API, or execution error), the big
   LLM rewrites the correct answer, with the failure fed back to it. The
   reference answer is tried first, so a valid one costs no extra call.

Everything lands in one file, data/generated/model_corrections.jsonl, which
combine_data.py cleans and merges into data/final/train.jsonl.
Questions are written one at a time, before they are answered, so a crash
loses at most the question in flight. Difficulty cycles beginner ->
intermediate -> advanced -> expert to keep a realistic spread.

Usage:
    python3 scripts/model_correction.py --questions 20
    python3 scripts/model_correction.py --purge        # clean the existing file
    python3 scripts/model_correction.py --revalidate   # re-run execution checks
"""
import argparse
import json
import os
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

import requests

import execution_validator
import generate_synthetic as big
import validator
import config

SEEDS = config.DOC_FILE
OUTPUT = config.MODEL_CORRECTIONS_FILE

SMALL_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "qwen3.5:4b")
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
exists) and use only real Blender 5.x Python APIs. Do not reuse the failed code."""

GEN_FORMAT = (
    "Respond in EXACTLY this delimiter format, no JSON, no prose outside the requested sections:\n"
    "===QUESTION===\n<question text>\n"
    "===ANSWER===\n<answer text>\n"
    "===CHECK===\n<short Python check>\n"
    "The answer must be a short explanation followed by exactly one ```python fenced block "
    "and end there. The check must be plain Python (no fences, no prose) that runs after the "
    "answer in the same namespace: create any scene state it needs, call the answer's "
    "function(s) with sample arguments, and assert read-back values (modifier type, property "
    "values, object count). Keep it short."
)

# Cycled per question so a run spans real-world difficulty, not just addons.
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

USE_COLOR = sys.stdout.isatty()


def c(text, code):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def status_line(n, total, key, row):
    icon, code = {"passed": ("✔", "32"), "corrected": ("✎", "33"),
                  "rejected": ("✘", "31")}[key]
    detail = "" if key == "passed" else f"  {row.get('small_error') or ''}"
    tqdm.write(f"  {c(icon, code)} {c(f'Q{n:02d}/{total}', '36')} "
               f"{c(row.get('difficulty') or '?', '35')}  "
               f"{c(key, code)}{detail[:80]}")


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def save_row(path: Path, question: str, row: dict) -> None:
    """Replace the question's placeholder row (or append if it is missing)."""
    # shortcut: rewrites the whole file per answer; fine for thousands of rows.
    rows = read_jsonl(path)
    for i, r in enumerate(rows):
        if r.get("question") == question:
            rows[i] = row
            break
    else:
        rows.append(row)
    write_jsonl(path, rows)


def purge(path: Path = OUTPUT) -> None:
    """Back up, drop leaked/duplicate questions, tag the rest unverified."""
    if not path.exists():
        raise SystemExit(f"No file to purge: {path}")
    rows = read_jsonl(path)
    before = len(rows)

    kept, leaked = [], 0
    for r in rows:
        q = r.get("question") or ""
        if "```" in q or "combine these into one" in q.lower():
            leaked += 1
            continue
        kept.append(r)

    by_question = {r.get("question"): r for r in kept}
    deduped = list(by_question.values())
    duplicates = len(kept) - len(deduped)

    for r in deduped:
        r["validated"] = "unverified"

    shutil.copy2(path, str(path) + ".bak")
    write_jsonl(path, deduped)
    print(f"Purged {path}: {before} -> {len(deduped)} rows "
          f"(-{leaked} leaked, -{duplicates} duplicate question); "
          f"backup: {path}.bak")


def revalidate(path: Path = OUTPUT) -> None:
    """Re-run coverage validation over old rows (empty check) and update them."""
    validator.load_cache()
    if not path.exists():
        raise SystemExit(f"No file to revalidate: {path}")
    rows = read_jsonl(path)

    codes = []
    for r in rows:
        blocks = validator.extract_fences(r.get("answer") or "")
        codes.append(blocks[0] if blocks else "")

    job_rows = [i for i, code in enumerate(codes) if code.strip()]
    results = execution_validator.run_batch([codes[i] for i in job_rows]) \
        if job_rows else []
    result_by_row = dict(zip(job_rows, results))

    passed = 0
    reasons = Counter()
    for i, r in enumerate(rows):
        res = result_by_row.get(i)
        if res is None:
            ok, reason, coverage, nonfatal = False, "no_code_block", 0.0, False
        else:
            coverage = float(res.get("coverage") or 0.0)
            class_reg = execution_validator.is_class_register(codes[i])
            nonfatal = bool(res.get("nonfatal"))
            if not res.get("ok"):
                if res.get("stage") == "check":
                    reason = "check_failed"
                else:
                    reason = "exec_" + str(res.get("error_type", "ExecutionError")).lower()
                ok = False
            elif coverage < execution_validator.COVERAGE_FLOOR and not class_reg:
                ok, reason = False, f"low_coverage:{int(round(coverage * 100))}"
            else:
                ok, reason = True, "ok"
        if ok:
            passed += 1
        else:
            reasons[reason] += 1
        r["validated"] = ok
        r["coverage"] = round(coverage, 3)
        r["exec_nonfatal"] = nonfatal
        r.setdefault("check", "")

    write_jsonl(path, rows)
    print(f"Revalidated {len(rows)} rows: {passed} pass, {len(rows) - passed} fail")
    print("Top failures:", dict(reasons.most_common(5)))


def build_question(seeds, buckets, known, difficulty):
    """Ask the big LLM for one new question at the requested difficulty."""
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
        res = parse_generation(big.call_llm(prompt, big.ARCH_SYSTEM))
        if not res:
            continue
        q = res["question"]
        # A question that contains code fences or prompt instructions means the
        # big model echoed the template instead of asking a question.
        if "```" in q or "combine these into one" in q.lower():
            continue
        check = res.get("reference_check", "")
        if not check or not validator.parse_ok(check):
            continue
        ok, _ = validator.validate_sample(q, res["reference"], "explanation",
                                          strict_format=True)
        if not ok or q in known:
            continue
        return {"question": q, "reference": res["reference"],
                "reference_check": check, "seed_ids": [i] + chosen,
                "difficulty": label}
    return None


def ask_small(question: str) -> str:
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
    for _ in range(3):
        try:
            r = requests.post(SMALL_URL, json=payload, timeout=900)
            r.raise_for_status()
            return (r.json().get("message", {}).get("content") or "").strip()
        except Exception as e:  # transient Ollama load/connection errors
            print(f"  small model error: {e}")
    return ""


def _empty_info() -> dict:
    return {"coverage": 0.0, "exec_nonfatal": False, "check_ok": None,
            "code_ok": False, "unverified": False}


def evaluate(question: str, answer: str, check: str = "",
             require_check: bool = False):
    """Static + coverage + hidden-check evaluation of one candidate answer."""
    info = _empty_info()
    ok, reason = validator.validate_sample(question, answer, "explanation",
                                           strict_format=True)
    if not ok:
        return False, reason, info
    code = validator.extract_fences(answer)[0]
    res = execution_validator.run_with_check(code, check)
    info.update({
        "coverage": float(res.get("coverage") or 0.0),
        "exec_nonfatal": bool(res.get("nonfatal")),
        "check_ok": res.get("check_ok"),
        "code_ok": bool(res.get("code_ok")),
    })
    class_reg = execution_validator.is_class_register(code)
    if not res.get("code_ok"):
        if res.get("stage") == "check":
            return False, "check_failed:" + (res.get("check_error") or "")[:60], info
        if res.get("nonfatal"):
            return False, f"nonfatal:{res.get('error_type', 'RuntimeError')}", info
        reason = res.get("error") or "execution failed"
        return False, f"exec_{str(res.get('error_type', 'ExecutionError')).lower()}:{reason[:60]}", info
    if not class_reg and info["coverage"] < execution_validator.COVERAGE_FLOOR:
        return False, f"low_coverage:{int(round(info['coverage'] * 100))}", info
    if require_check and not class_reg:
        if not check:
            info["unverified"] = True
            return False, "missing_check", info
        if res.get("check_ok") is not True:
            return False, "check_failed", info
    return True, "ok", info


def parse_answer(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"===CHECK===.*", "", text, flags=re.DOTALL)
    m = re.search(r"===ANSWER===\s*(.*)", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def parse_check(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"===CHECK===\s*(.*)", text, re.DOTALL)
    if not m:
        return ""
    check = m.group(1).strip()
    if check.startswith("```"):
        check = re.sub(r"^```[a-zA-Z]*\n?", "", check)
        check = re.sub(r"\n?```$", "", check)
    return check.strip()


def parse_generation(text: str) -> dict:
    if not text:
        return {}
    q = re.search(r"===QUESTION===\s*(.*?)\s*===ANSWER===", text, re.DOTALL)
    answer = parse_answer(text)
    if not q or not answer:
        return {}
    return {"question": q.group(1).strip(), "reference": answer,
            "reference_check": parse_check(text)}


def correct(question: str, small_answer: str, reason: str,
            reference: str = "", reference_check: str = ""):
    """Try the big model's reference answer, then up to 2 targeted rewrites."""
    info = _empty_info()
    if reference:
        ok, r, info = evaluate(question, reference, reference_check,
                               require_check=True)
        if ok:
            return reference, reference_check, info
    attempt, why = small_answer, reason
    for _ in range(2):
        prompt = CORRECTION_PROMPT.format(question=question,
                                          attempt=attempt[:4000],
                                          reason=why)
        text = big.call_llm(prompt, big.ARCH_SYSTEM)
        candidate, check = parse_answer(text), parse_check(text)
        if not candidate:
            break
        ok, r, info = evaluate(question, candidate, check, require_check=True)
        if ok:
            return candidate, check, info
        attempt, why = candidate, r
    return None, "", info


def process(item: dict, status) -> tuple:
    """Answer, validate, and (if needed) fix one question. Saves the row."""
    question = item["question"]
    reference_check = item.get("reference_check") or item.get("check") or ""
    status("asking small model")
    small_answer = ask_small(question)
    status("checking question + code")
    ok, reason, info = evaluate(question, small_answer, reference_check,
                                require_check=bool(reference_check))

    final, final_check = small_answer, reference_check if ok else ""
    corrected = False
    if not ok:
        status("fixing with big model")
        result, check, cinfo = correct(question, small_answer, reason,
                                       item.get("reference", ""),
                                       reference_check)
        if result is not None:
            final, final_check = result, check
            info = cinfo
            corrected = True

    if ok or corrected:
        validated = True
    elif info.get("unverified"):
        validated = "unverified"
    else:
        validated = False

    row = {
        "question": question,
        "answer": final,
        "check": final_check,
        "coverage": round(float(info.get("coverage", 0.0)), 3),
        "exec_nonfatal": bool(info.get("exec_nonfatal")),
        "type": "4",
        "task_type": config.task_type(4),
        "source": "model_corrections",
        "difficulty": item.get("difficulty", ""),
        "seed_ids": item.get("seed_ids", []),
        "small_model": SMALL_MODEL,
        "small_answer": small_answer,
        "small_ok": ok,
        "small_error": None if ok else reason,
        "corrected": corrected,
        "validated": validated,
    }
    save_row(OUTPUT, question, row)
    return row, ("passed" if ok else ("corrected" if corrected else "rejected"))


def main():
    parser = argparse.ArgumentParser(description="Type-4 big->small->Blender->big pipeline")
    parser.add_argument("--questions", type=int, default=10,
                        help="number of new questions to generate/process this run")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--purge", action="store_true",
                        help="drop leaked/duplicate rows and tag the rest unverified")
    action.add_argument("--revalidate", action="store_true",
                        help="re-run execution validation over existing rows")
    args = parser.parse_args()

    if args.purge:
        purge()
        return
    if args.revalidate:
        revalidate()
        return

    validator.load_cache()
    big.POOL.refresh(big._headers())
    print(f"Big model(s): {', '.join(big.POOL.models)}")
    print(f"Small model:  {SMALL_MODEL} @ {SMALL_URL}")

    seeds = read_jsonl(SEEDS)
    if not seeds:
        raise SystemExit(f"No seeds in {SEEDS}")
    buckets = {}
    for i, s in enumerate(seeds):
        buckets.setdefault(big.topic_key(s), []).append(i)

    output = read_jsonl(OUTPUT)
    known = {row.get("question") for row in output}
    # A row without an answer is an unfinished question: answer it this run.
    pending = [row for row in output if row.get("question") and not row.get("answer")]

    new_count = max(0, args.questions - len(pending))
    total = len(pending) + new_count
    if not total:
        print("Nothing to do (seed or question supply exhausted).")
        return

    counts = {"passed": 0, "corrected": 0, "rejected": 0}
    misses = 0
    with tqdm(total=total, desc="questions", unit="q", dynamic_ncols=True,
              file=sys.stdout) as bar:
        n = 0

        def run(item):
            nonlocal n
            n += 1
            bar.set_description(f"Q{n:02d}")
            tqdm.write(f"{c(f'Q{n:02d}', '36')} "
                       f"{c(item.get('difficulty') or 'resume', '35')}  "
                       f"{item['question'][:96]}")
            row, key = process(item, bar.set_postfix_str)
            counts[key] += 1
            status_line(n, total, key, row)
            bar.update(1)

        for item in pending:
            run(item)

        made = 0
        while made < new_count:
            difficulty = DIFFICULTIES[made % len(DIFFICULTIES)]
            bar.set_postfix_str(f"generating {difficulty[0]}")
            item = build_question(seeds, buckets, known, difficulty)
            if item is None:
                misses += 1
                if misses >= 3:
                    tqdm.write(c("no usable question from the big model; stopping", "31"))
                    break
                continue
            misses = 0
            # Persist the question before answering it, so a crash loses only
            # the question in flight and the next run resumes it.
            append_jsonl(OUTPUT, {
                "question": item["question"],
                "reference": item.get("reference", ""),
                "reference_check": item.get("reference_check", ""),
                "check": "",
                "type": "4",
                "task_type": config.task_type(4),
                "source": "model_corrections",
                "difficulty": item.get("difficulty", ""),
                "seed_ids": item.get("seed_ids", []),
            })
            known.add(item["question"])
            run(item)
            made += 1

    print(f"\n{c('Done', '1;32')}: {c(str(counts['passed']), '32')} passed, "
          f"{c(str(counts['corrected']), '33')} corrected, "
          f"{c(str(counts['rejected']), '31')} rejected -> {OUTPUT}")


if __name__ == "__main__":
    main()
