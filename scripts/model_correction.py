
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
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

import execution_validator
import generate_synthetic as big
import validator
import config

SEEDS = config.DOC_FILE
OUTPUT = config.MODEL_CORRECTIONS_FILE

# Worker threads save rows concurrently; every file write goes through this lock.
_FILE_LOCK = threading.Lock()
# The small model is served one request at a time; big-model calls run in parallel.
_SMALL_LOCK = threading.Lock()

SMALL_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "qwen3.5:4b")
# First run + one retry run; rows still failing after that are dropped.
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


class LiveStatus:
    """In-flight questions, redrawn in place on a TTY (plain lines otherwise)."""

    ICONS = {"passed": ("✔", "32"), "corrected": ("✎", "33"),
             "rejected": ("✘", "31")}

    def __init__(self, total):
        self.total = total
        self.done = 0
        self.counts = {"passed": 0, "corrected": 0, "rejected": 0, "dropped": 0}
        self.active = {}      # jid -> (stage, question)
        self.recent = []      # last few results, so failures stay visible
        self.lock = threading.Lock()
        self.drawn = 0
        self.tty = sys.stdout.isatty()

    def stage(self, jid, msg, question):
        line = question.replace("\n", " ")
        with self.lock:
            self.active[jid] = (msg, line)
            if self.tty:
                self._draw()
            else:
                print(f"  Q{jid:02d} {msg:<22} {line[:80]}")

    def finish(self, jid, key, row):
        with self.lock:
            self.active.pop(jid, None)
            self.done += 1
            self.counts[key] += 1
            icon, code = self.ICONS[key]
            detail = "" if key == "passed" else "  " + (
                row.get("small_error") or row.get("correction_error") or "")
            line = f"{c(icon, code)} {c(f'Q{jid:02d}', '36')} {c(key, code)}{detail[:72]}"
            self.recent = (self.recent + [line])[-3:]
            if self.tty:
                self._draw()
            else:
                print("  " + line)

    def drop(self, jid):
        with self.lock:
            self.counts["dropped"] += 1
            tag = f"Q{jid:02d}"
            self.recent = [r for r in self.recent if tag not in r]
            self.recent = (self.recent + [c(f"⌫ {tag} dropped", "31")])[-3:]
            if self.tty:
                self._draw()

    def _draw(self):
        n = self.counts
        lines = [f"  {c(f'{self.done}/{self.total} done', '1')}   "
                 f"{c('✔' + str(n['passed']), '32')} "
                 f"{c('✎' + str(n['corrected']), '33')} "
                 f"{c('✘' + str(n['rejected']), '31')}   "
                 f"{len(self.active)} active"]
        lines += ["  " + r for r in self.recent]
        lines.append("")
        for jid, (msg, question) in sorted(self.active.items()):
            lines.append(f"  {c(f'Q{jid:02d}', '36')} {c(f'{msg:<22}', '35')} "
                         f"{question[:72]}")
        out = sys.stdout
        if self.drawn:
            out.write(f"\x1b[{self.drawn}A")
        width = max(self.drawn, len(lines))
        for i in range(width):
            out.write("\x1b[2K" + (lines[i] if i < len(lines) else "") + "\n")
        self.drawn = width
        out.flush()


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
    with _FILE_LOCK, path.open("a", encoding="utf-8") as f:
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
    with _FILE_LOCK:
        rows = read_jsonl(path)
        for i, r in enumerate(rows):
            if r.get("question") == question:
                rows[i] = row
                break
        else:
            rows.append(row)
        write_jsonl(path, rows)


def remove_row(path: Path, question: str) -> None:
    with _FILE_LOCK:
        write_jsonl(path, [r for r in read_jsonl(path)
                           if r.get("question") != question])


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
    """Re-run execution validation over old rows (empty check) and update them."""
    validator.load_cache()
    if not path.exists():
        raise SystemExit(f"No file to revalidate: {path}")
    rows = read_jsonl(path)

    codes = []
    for r in rows:
        blocks = validator.extract_fences(r.get("answer") or "")
        codes.append(blocks[0] if blocks else "")

    passed = 0
    reasons = Counter()
    vacuous_checked = 0
    cot_checked = 0
    vacuous_fail = 0
    cot_fail = 0
    for i, r in enumerate(rows):
        check = r.get("check") or ""
        code = codes[i]
        res = execution_validator.run_with_check(code, check) if check.strip() \
            else execution_validator.run_code(code)
        coverage = float(res.get("coverage") or 0.0)
        class_reg = execution_validator.is_class_register(code)
        nonfatal = bool(res.get("nonfatal"))
        ok = bool(res.get("ok"))
        reason = "ok"
        if not ok:
            ok = False
            if res.get("stage") == "check":
                reason = "check_failed"
            else:
                reason = "exec_" + str(res.get("error_type", "ExecutionError")).lower()
        elif check.strip():
            vacuous_checked += 1
            if execution_validator.is_vacuous_check(check):
                ok, reason = False, "vacuous_check"
        elif class_reg or coverage < execution_validator.COVERAGE_FLOOR:
            ok, reason = False, "unverified"
        if check.strip() and ok:
            cot_checked += 1
            prose = (r.get("answer") or "").split("```", 1)[0]
            if any(m in prose.lower() for m in validator.COT_MARKERS):
                ok, reason = False, "chain_of_thought"
        if ok and not class_reg and coverage < execution_validator.COVERAGE_FLOOR:
            ok, reason = False, f"low_coverage:{int(round(coverage * 100))}"

        if ok:
            passed += 1
        else:
            reasons[reason] += 1
            if reason == "vacuous_check":
                vacuous_fail += 1
            if reason == "chain_of_thought":
                cot_fail += 1
        r["validated"] = ok
        r["coverage"] = round(coverage, 3)
        r["exec_nonfatal"] = nonfatal
        r.setdefault("check", "")

    write_jsonl(path, rows)
    print(f"Revalidated {len(rows)} rows: {passed} pass, {len(rows) - passed} fail")
    print("Top failures:", dict(reasons.most_common(5)))
    print(f"vacuous_check: {vacuous_checked} checked, {vacuous_fail} failed")
    print(f"COT prose: {cot_checked} checked, {cot_fail} failed")


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
        # The reference must pass its own check: otherwise no answer ever can,
        # and every correction call on this question is wasted.
        ok, _, _ = evaluate(q, res["reference"], check, require_check=True)
        if not ok:
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
            # Ollama serves one request at a time: keep small calls serial.
            with _SMALL_LOCK:
                r = requests.post(SMALL_URL, json=payload, timeout=900)
                r.raise_for_status()
                return (r.json().get("message", {}).get("content") or "").strip()
        except Exception:  # transient Ollama load/connection errors
            # No console output here: the live panel owns the terminal. A broken
            # Ollama call surfaces as a no_code_block rejection instead.
            pass
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
        if res.get("nonfatal"):
            return False, f"nonfatal:{res.get('error_type', 'RuntimeError')}", info
        reason = res.get("error") or "execution failed"
        return False, f"exec_{str(res.get('error_type', 'ExecutionError')).lower()}:{reason[:60]}", info
    # The check comes before coverage: a broken check is the actionable reason,
    # and low coverage would otherwise hide it from the correction prompt.
    if require_check:
        if not check:
            info["unverified"] = True
            return False, "missing_check", info
        if res.get("check_ok") is not True:
            detail = (res.get("check_error") or res.get("error") or "")[:120]
            return False, f"check_failed:{detail}", info
    if not class_reg and info["coverage"] < execution_validator.COVERAGE_FLOOR:
        return False, f"low_coverage:{int(round(info['coverage'] * 100))}", info
    # Vacuous check: does the check also pass when no answer code ran at all?
    if check and res.get("check_ok") is True and execution_validator.is_vacuous_check(check):
        return False, "vacuous_check", info
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
    info["correction_error"] = why
    return None, "", info


def process(item: dict, status, small_answer: str = None) -> tuple:
    """Answer, validate, and (if needed) fix one question. Saves the row."""
    question = item["question"]
    reference_check = item.get("reference_check") or item.get("check") or ""
    if small_answer is None:
        status("asking small model")
        small_answer = ask_small(question)
    status("checking question + code")
    ok, reason, info = evaluate(question, small_answer, reference_check,
                                require_check=bool(reference_check))

    final, final_check = small_answer, reference_check if ok else ""
    corrected = False
    answer_source = "small"
    if ok and item.get("reference"):
        prose = small_answer.split("```", 1)[0]
        bad_prose = len(prose) > 600 or any(m in prose.lower() for m in validator.COT_MARKERS)
        if bad_prose:
            ref_ok, _, ref_info = evaluate(question, item.get("reference") or "",
                                          item.get("reference_check") or "",
                                          require_check=True)
            if ref_ok:
                final, final_check = item.get("reference") or small_answer, \
                    item.get("reference_check") or ""
                info = ref_info
                answer_source = "reference"
    if not ok:
        status("fixing with big model")
        result, check, cinfo = correct(question, small_answer, reason,
                                       item.get("reference", ""),
                                       reference_check)
        if result is not None:
            final, final_check = result, check
            info = cinfo
            corrected = True
            answer_source = "reference"
        else:
            info["correction_error"] = cinfo.get("correction_error")

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
        # Kept so a retry can still try the big model's reference answer first.
        "reference": item.get("reference", ""),
        "reference_check": reference_check,
        "correction_error": info.get("correction_error"),
        "coverage": round(float(info.get("coverage", 0.0)), 3),
        "exec_nonfatal": bool(info.get("exec_nonfatal")),
        "source": "model_corrections",
        "difficulty": item.get("difficulty", ""),
        "answer_source": answer_source,
        "attempts": int(item.get("attempts") or 0) + 1,
        "seed_ids": item.get("seed_ids", []),
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
    parser.add_argument("--workers", type=int, default=10,
                        help="questions processed concurrently")
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
    # Failed / unverified rows get one more run (re-check stored answer + 2 rewrites).
    retry = [row for row in output
             if (row.get("validated") is False or row.get("validated") == "unverified")
             and int(row.get("attempts") or 0) < MAX_ATTEMPTS]

    new_count = max(0, args.questions - len(pending))
    total = len(pending) + len(retry) + new_count
    if not total:
        print("Nothing to do (seed or question supply exhausted).")
        return

    panel = LiveStatus(total)
    def status(jid, msg, question):
        panel.stage(jid, msg, question)

    def answer(jid, item, small_answer=None):
        """Answer, validate and save one question (worker thread)."""
        q = item["question"]
        return item, *process(item, lambda m: status(jid, m, q), small_answer)

    def make(jid, difficulty):
        """Generate one new question, then answer it (worker thread)."""
        for _ in range(3):
            status(jid, "generating question", f"({difficulty})")
            item = build_question(seeds, buckets, known, difficulty)
            if not item:
                continue
            with _FILE_LOCK:
                if item["question"] in known:
                    continue
                known.add(item["question"])
            # Persist the question before answering it, so a crash loses only
            # the question in flight and the next run resumes it.
            append_jsonl(OUTPUT, {
                "question": item["question"],
                "reference": item.get("reference", ""),
                "reference_check": item.get("reference_check", ""),
                "check": "",
                "source": "model_corrections",
                "difficulty": item.get("difficulty", ""),
                "seed_ids": item.get("seed_ids", []),
            })
            q = item["question"]
            return item, *process(item, lambda m: status(jid, m, q), None)
        return None, None, None

    jobs = [(answer, (item, None)) for item in pending]
    jobs += [(answer, (item, item.get("small_answer") or item.get("answer") or None))
             for item in retry]
    jobs += [(make, (DIFFICULTIES[i % len(DIFFICULTIES)],))
             for i in range(new_count)]

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(fn, jid, *a): jid
                   for jid, (fn, a) in enumerate(jobs, 1)}
        for fut in as_completed(futures):
            jid = futures[fut]
            try:
                item, row, key = fut.result()
            except Exception as e:
                panel.finish(jid, "rejected", {"small_error": str(e)[:80]})
                continue
            if item is None:
                panel.finish(jid, "rejected", {"small_error": "no usable question"})
                continue
            panel.finish(jid, key, row)
            if key == "rejected" and row.get("attempts", 0) >= MAX_ATTEMPTS:
                remove_row(OUTPUT, item["question"])
                known.discard(item["question"])
                panel.drop(jid)

    print(f"\n{c('Done', '1;32')}: {c(str(panel.counts['passed']), '32')} passed, "
          f"{c(str(panel.counts['corrected']), '33')} corrected, "
          f"{c(str(panel.counts['rejected']), '31')} rejected -> {OUTPUT}")


if __name__ == "__main__":
    main()
