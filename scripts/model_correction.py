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
Questions are written as soon as they are generated; resume answers any row
that has no answer yet.

Usage:
    python3 scripts/model_correction.py --questions 20
"""
import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# system python may lack python-dotenv, which generate_synthetic imports softly
if not os.environ.get("OMNIRoute_API_KEY") and (ROOT / ".env").exists():
    for line in (ROOT / ".env").read_text().splitlines():
        k, _, v = line.partition("=")
        if k.strip() and v.strip():
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

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

Write the correct answer: a short explanation, then exactly one ```python fenced block,
and nothing after the block. The code must run standalone in a fresh headless Blender
scene (an active default cube exists) and use only real Blender 5.x Python APIs.
Do not reuse the failed code."""


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
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def build_questions(count: int, seeds, buckets, known: set, workers: int):
    """Ask the big LLM for `count` new questions, in parallel batches."""
    indices = list(range(len(seeds)))
    random.shuffle(indices)

    def make(i):
        seed = seeds[i]
        peers = [j for j in buckets[big.topic_key(seed)] if j != i]
        if len(peers) < 2:
            return None
        chosen = random.sample(peers, 2)
        res = big.generate_type_4([seed] + [seeds[j] for j in chosen])
        if not res:
            return None
        q, a = res.get("question", ""), res.get("answer", "")
        # A question that contains code fences or prompt instructions means the
        # big model echoed the template instead of asking a question.
        if "```" in q or "combine these into one" in q.lower():
            return None
        ok, _ = validator.validate_sample(q, a, "explanation", strict_format=True)
        if not ok:
            return None
        return {"question": q, "reference": a, "seed_ids": [i] + chosen}

    fresh = []
    cursor = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while cursor < len(indices) and len(fresh) < count:
            batch = indices[cursor:cursor + workers]
            cursor += workers
            for row in pool.map(make, batch):
                if not row or row["question"] in known:
                    continue
                known.add(row["question"])
                fresh.append(row)
                if len(fresh) >= count:
                    break
    return fresh


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


def evaluate(question: str, answer: str):
    ok, reason = validator.validate_sample(question, answer, "explanation",
                                           strict_format=True)
    if not ok:
        return False, reason
    return execution_validator.validate(validator.extract_fences(answer)[0])


def parse_answer(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"===ANSWER===\s*(.*)", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def correct(question: str, small_answer: str, reason: str, reference: str = ""):
    """Try the big model's reference answer, then up to 2 targeted rewrites."""
    if reference:
        ok, r = evaluate(question, reference)
        if ok:
            return reference, r
    attempt, why = small_answer, reason
    for _ in range(2):
        prompt = CORRECTION_PROMPT.format(question=question,
                                          attempt=attempt[:4000],
                                          reason=why)
        candidate = parse_answer(big.call_llm(prompt, big.ARCH_SYSTEM))
        if not candidate:
            break
        ok, r = evaluate(question, candidate)
        if ok:
            return candidate, r
        attempt, why = candidate, r
    return None, why


def main():
    parser = argparse.ArgumentParser(description="Type-4 big->small->Blender->big pipeline")
    parser.add_argument("--questions", type=int, default=10,
                        help="number of new questions to generate/process this run")
    parser.add_argument("--workers", type=int, default=6,
                        help="parallel big-LLM question-generation requests")
    args = parser.parse_args()

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
    if new_count:
        print(f"Asking the big LLM for {new_count} new question(s)...")
        fresh = build_questions(new_count, seeds, buckets, known, args.workers)
        print(f"Generated {len(fresh)} question(s).")
        # Persist questions before answering them, so a crash loses no work and
        # the next run can pick them up from the file.
        for item in fresh:
            append_jsonl(OUTPUT, {
                "question": item["question"],
                "reference": item.get("reference", ""),
                "type": "4",
                "task_type": config.task_type(4),
                "source": "model_corrections",
                "seed_ids": item.get("seed_ids", []),
            })
        pending.extend(fresh)

    if not pending:
        print("Nothing new to process (seed or question supply exhausted).")
        return
    print(f"Processing {len(pending)} question(s).")

    counts = {"passed": 0, "corrected": 0, "rejected": 0}
    for n, item in enumerate(pending, 1):
        question = item["question"]
        small_answer = ask_small(question)
        ok, reason = evaluate(question, small_answer)

        final, corrected = small_answer, False
        if not ok:
            final, _ = correct(question, small_answer, reason,
                               item.get("reference", ""))
            corrected = final is not None
            final = final or small_answer

        accepted = ok or corrected
        row = {
            "question": question,
            "answer": final,
            "type": "4",
            "task_type": config.task_type(4),
            "source": "model_corrections",
            "seed_ids": item.get("seed_ids", []),
            "small_model": SMALL_MODEL,
            "small_answer": small_answer,
            "small_ok": ok,
            "small_error": None if ok else reason,
            "corrected": corrected,
            "validated": accepted,
        }
        save_row(OUTPUT, question, row)

        key = "passed" if ok else ("corrected" if corrected else "rejected")
        counts[key] += 1
        print(f"[{n}/{len(pending)}] {key}"
              f"{'' if ok else ' (' + str(row['small_error']) + ')'}")

    print(f"\nDone: {counts['passed']} passed, {counts['corrected']} corrected, "
          f"{counts['rejected']} rejected -> {OUTPUT}")


if __name__ == "__main__":
    main()
