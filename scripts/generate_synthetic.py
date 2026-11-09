"""Synthetic training-data generator for the Blender QLoRA model.

Output rows stay {"question", "answer"} (plain strings) for training-script
compatibility. All code answers are ```python-fenced and validated against
the real Blender API (see validator.py) before being written.
"""
import os
import ast
import logging
import json
import time
import random
import requests
import re
import threading
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # system python may lack the venv's python-dotenv
    pass
import validator
import execution_validator

logging.basicConfig(
    filename='/home/mintlover/Documents/python/blender_copilot/generation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CONFIG = {
    "omniroute_url": "http://localhost:20128/v1/chat/completions",
    "seed_file": "/home/mintlover/Documents/python/blender_copilot/data/doc_data.jsonl",
    "output_file": "/home/mintlover/Documents/python/blender_copilot/data/synthetic_training_data.jsonl",
    "models_url": "http://localhost:20128/v1/models",
    # "all" (default) = use every model OmniRoute exposes, round-robin.
    # Set OMNIRoute_MODEL=<model id> to force a single model instead.
    "model": os.environ.get("OMNIRoute_MODEL", "all"),
    # extra regex of model ids to skip (e.g. "gpt-oss-20b|small")
    "exclude": os.environ.get("OMNIRoute_EXCLUDE", ""),
    "max_llm_attempts": 6,  # each retry goes to a DIFFERENT model
    "temperature": 0.2,   # factual code: low temperature
    "max_workers": 10,
    "cap": 1000,
    "max_retries": 3,
}

# Grounding text appended to every generation prompt (STEP 4)
GROUNDING = ("Use ONLY APIs that appear in the provided reference snippet or that you "
             "are certain exist in Blender 5.x. If unsure, do not use it. Target Blender 5.x. "
             "All code MUST be inside ```python fenced blocks.")

FORMAT_SPEC = (
    "Respond in EXACTLY this delimiter format, no JSON, no prose outside the "
    "requested sections, no other text:\n"
    "===QUESTION===\n<question text>\n"
    "===ANSWER===\n<answer text>\n"
    "For explanation answers (types 3-5) the answer must be a short explanation "
    "followed by exactly one ```python ... ``` block and end there.\n"
    "For coder answers (types 7-8) the answer must be exactly one "
    "```python ... ``` block and nothing else."
)

ARCH_SYSTEM = ("You are a world-class Blender API Architect. Write for Blender 5.x only. "
               "The 'answer' MUST be a short explanation followed by exactly one "
               "```python fenced code block and nothing after it. "
               "Use ONLY APIs from the provided reference or ones you are certain exist. "
               "If unsure, do not use it.")

CODER_SYSTEM = ("You are a precise Blender addon coder. Write for Blender 5.x only. "
                "The 'answer' MUST be exactly one ```python fenced code block and nothing else. "
                "Use ONLY APIs from the provided reference or ones you are certain exist. "
                "If unsure, do not use it.")


# ==============================================================================
# MODEL POOL: round-robin over every model OmniRoute exposes (no combos)
# ==============================================================================
class ModelPool:
    # non-chat models that show up in /v1/models
    NON_CHAT = re.compile(r"embed|whisper|tts|speech|transcri|rerank|moderation|"
                          r"image|dall|sdxl|flux|stable-diffusion", re.I)

    def __init__(self):
        self.lock = threading.Lock()
        self.models: List[str] = []
        self.fails: Dict[str, int] = {}
        self.benched: Dict[str, float] = {}   # model -> time.time() until it is skipped
        self.i = 0

    def refresh(self, headers: Dict[str, str]) -> None:
        forced = CONFIG["model"]
        if forced and forced.lower() != "all":
            self.models = [forced]
            return
        r = requests.get(CONFIG["models_url"], headers=headers, timeout=60)
        r.raise_for_status()
        ids = [m["id"] for m in r.json().get("data", []) if m.get("id")]
        ids = [m for m in ids if not self.NON_CHAT.search(m)]
        if CONFIG["exclude"]:
            ids = [m for m in ids if not re.search(CONFIG["exclude"], m)]
        if not ids:
            raise RuntimeError("OmniRoute returned no usable models")
        self.models = sorted(set(ids))

    def next(self) -> str:
        with self.lock:
            now = time.time()
            live = [m for m in self.models if self.benched.get(m, 0) <= now]
            if not live:  # everything benched: give them all another chance
                self.benched.clear()
                live = list(self.models)
            m = live[self.i % len(live)]
            self.i += 1
            return m

    def ok(self, model: str) -> None:
        with self.lock:
            self.fails[model] = 0

    def bad(self, model: str, cooldown: int = 0, hard: bool = False) -> None:
        """hard=True (401/402/403/404): bench for an hour. Otherwise bench for
        `cooldown` seconds, or after 3 consecutive failures for 5 minutes."""
        with self.lock:
            self.fails[model] = self.fails.get(model, 0) + 1
            if hard:
                self.benched[model] = time.time() + 3600
            elif cooldown:
                self.benched[model] = time.time() + cooldown
            elif self.fails[model] >= 3:
                self.benched[model] = time.time() + 300
            else:
                return
        logger.warning(f"Benched model {model} (fails={self.fails[model]})")


POOL = ModelPool()


def _headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OMNIRoute_API_KEY', 'sk-no-key')}",
    }


# ==============================================================================
# LLM TRANSPORT (each retry rotates to the next model)
# ==============================================================================
def call_llm(prompt: str, system_prompt: str) -> str:
    for attempt in range(CONFIG["max_llm_attempts"]):
        model = POOL.next()
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": CONFIG["temperature"],
            "max_tokens": 2000,
        }
        try:
            response = requests.post(CONFIG["omniroute_url"], headers=_headers(),
                                     json=payload, timeout=120)
            status = response.status_code
            if status == 429:
                POOL.bad(model, cooldown=60)
                logger.warning(f"{model}: rate limited, rotating")
                continue
            if status in (401, 402, 403, 404):  # auth / out of credit / model gone
                POOL.bad(model, hard=True)
                logger.error(f"{model}: HTTP {status}, benched")
                continue
            if status >= 500:
                POOL.bad(model)
                logger.warning(f"{model}: server error {status}, rotating")
                continue
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"].strip()
            if not text:
                POOL.bad(model)
                continue
            POOL.ok(model)
            return text
        except Exception as e:
            POOL.bad(model)
            logger.warning(f"{model}: call failed ({e}), rotating")
            time.sleep(0.5)
    return ""


def parse_delimited(text: str) -> Dict[str, str]:
    """Parse the ===QUESTION=== / ===ANSWER=== format. No JSON anywhere, so
    embedded code needs no escaping and cannot be corrupted by un-escaping."""
    if not text:
        return {}
    text = text.strip()
    if text.startswith("```"):  # model wrapped the whole reply; unwrap
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    q = re.search(r"===QUESTION===\s*(.*?)\s*===ANSWER===", text, re.DOTALL)
    a = re.search(r"===ANSWER===\s*(.*)", text, re.DOTALL)
    if not q or not a:
        return {}
    return {"question": q.group(1).strip(), "answer": a.group(1).strip()}


def seed_code(seed: Dict) -> Optional[str]:
    """Extract the seed's first ```python block. Returns None when the seed
    has no usable fence (instead of crashing with IndexError)."""
    code = seed.get("answer", "")
    if not isinstance(code, str):
        return None
    m = re.search(r"```python\s*(.*?)```", code, re.DOTALL)
    if not m:
        return None
    return m.group(1).strip() or None


def topic_key(seed: Dict) -> str:
    """Bucket seeds by the first API area they mention so type 4 combines
    related seeds instead of 3 random unrelated ones."""
    text = seed.get("question", "") + " " + seed.get("answer", "")
    m = re.search(r"bpy\.(?:types|ops)\.(\w+)", text)
    if m:
        return "api:" + m.group(1)
    return "kw:" + " ".join(seed.get("question", "").lower().split()[:4])


# ==============================================================================
# STRATEGIES (3-8). All return Dict or {} on failure. All code is fenced.
# ==============================================================================
def generate_type_3(seed: Dict) -> Optional[Dict]:
    code = seed_code(seed)
    if not code:
        return None
    prompt = (f"Reference Blender Python snippet (the only allowed API source):\n\n{code}\n\n"
              "Create a training pair where the question asks for a concise explanation of "
              "this snippet and the answer gives a short technical explanation followed by "
              "exactly one ```python block.\n" + GROUNDING + "\n" + FORMAT_SPEC)
    return parse_delimited(call_llm(prompt, ARCH_SYSTEM))


def generate_type_4(seeds: List[Dict]) -> Optional[Dict]:
    prompt = ("Reference snippets (the only allowed API source):\n\n"
              + "\n\n".join(f"- {s['question']}: {s['answer']}" for s in seeds)
              + "\n\nCombine these into one advanced tool. The answer MUST be a short "
              "explanation followed by exactly one ```python block.\n"
              + GROUNDING + "\n" + FORMAT_SPEC)
    return parse_delimited(call_llm(prompt, ARCH_SYSTEM))


def generate_type_5(seed: Dict) -> Optional[Dict]:
    code = seed_code(seed)
    if not code:
        return None
    prompt = (f"Reference correct Blender code (the only allowed API source):\n\n{code}\n\n"
              "Create a training pair: the question shows a subtly buggy version of this code "
              "and asks to fix it; the answer explains the bug and provides the fix. The answer "
              "MUST be a short explanation followed by exactly one ```python block.\n"
              + GROUNDING + "\n" + FORMAT_SPEC)
    return parse_delimited(call_llm(prompt, ARCH_SYSTEM))


def split_fim(code: str):
    """Split validated code at a top-level statement boundary for real FIM.

    Returns (prefix, middle, suffix) or None when the snippet has no safe
    statement to remove.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    lines = code.splitlines()
    # Any statement (including one inside a function/class body) can be the
    # masked span; the full snippet is what gets validated.
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.stmt)
             and not isinstance(n, (ast.Import, ast.ImportFrom, ast.Pass))
             and getattr(n, "lineno", None) and getattr(n, "end_lineno", None)]
    nodes = [n for n in nodes
             if not (n.lineno == 1 and n.end_lineno == len(lines))]
    if not nodes:
        return None

    def start_line(n):
        # A decorated function's lineno points at `def`, so keep the decorators.
        return min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])

    mid = len(lines) / 2
    # Prefer a span with code on both sides (real prefix + suffix), near the middle.
    node = min(nodes, key=lambda n: (
        not (start_line(n) > 1 and n.end_lineno < len(lines)),
        abs((start_line(n) + n.end_lineno) / 2 - mid),
        n.end_lineno - start_line(n)))
    start = start_line(node)
    prefix = "\n".join(lines[:start - 1]).strip("\n")
    middle = "\n".join(lines[start - 1:node.end_lineno])
    suffix = "\n".join(lines[node.end_lineno:]).strip("\n")
    if not middle.strip() or not (prefix or suffix):
        return None
    return prefix, middle, suffix


def generate_type_7(seed: Dict) -> Optional[Dict]:
    code = seed_code(seed)
    if not code:
        return None
    prompt = (f"Reference correct Blender code (the only allowed API source):\n\n{code}\n\n"
              "Create a 'Fix this bug' pair. The question MUST start with the instruction "
              "'Fix the bug in this Blender addon:' followed by a ```python-fenced buggy "
              "version of the code (the bug must be logic or API misuse, never an invented "
              "API). The answer MUST be only one ```python-fenced corrected version, no prose.\n"
              + GROUNDING + "\n" + FORMAT_SPEC)
    return parse_delimited(call_llm(prompt, CODER_SYSTEM))


def generate_type_8(seed: Dict) -> Optional[Dict]:
    code = seed_code(seed)
    if not code:
        return None
    prompt = (f"Reference correct Blender code (the only allowed API source):\n\n{code}\n\n"
              "Create a 'Refactor' pair. The question MUST start with the instruction "
              "'Refactor this Blender addon for clarity and performance:' followed by a "
              "```python-fenced working-but-messy 'Junior' version. The answer MUST be only "
              "one ```python-fenced optimized 'Senior' version, no prose.\n"
              + GROUNDING + "\n" + FORMAT_SPEC)
    return parse_delimited(call_llm(prompt, CODER_SYSTEM))


# ==============================================================================
# MAIN
# ==============================================================================
class RejectionStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.counts: Dict[tuple, int] = {}

    def record(self, gen_type: int, reason: str) -> None:
        with self.lock:
            self.counts[(gen_type, reason)] = self.counts.get((gen_type, reason), 0) + 1

    def report(self) -> str:
        return "\n".join(f"type {t}: {r} = {n}"
                         for (t, r), n in sorted(self.counts.items()))


def append_fim_rows(path: str, seen_hashes: set, limit: int, execute: bool,
                    stats: "RejectionStats") -> int:
    """Build FIM rows from already-validated code instead of asking an LLM to
    mask characters (the old type 6 garbage)."""
    if limit <= 0 or not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    added = 0
    for r in rows:
        if added >= limit:
            break
        if str(r.get("type")) in ("6", "fim"):
            continue
        blocks = validator.extract_fences(r.get("answer", ""))
        if not blocks:
            continue
        parts = split_fim(blocks[0])
        if not parts:
            continue
        prefix, middle, suffix = parts
        row = {
            "question": f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>",
            "answer": f"```python\n{middle}\n```",
        }
        ok, reason = validator.validate_sample(row["question"], row["answer"], "fim",
                                               seen_hashes, strict_format=True)
        if ok and execute:
            ok, reason = execution_validator.validate(prefix + "\n" + middle + "\n" + suffix)
        if not ok:
            stats.record(6, reason)
            continue
        row.update({"type": "fim", "seed_id": r.get("seed_id"), "source": "synthetic"})
        with open(path, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        added += 1
    return added


def main():
    parser = argparse.ArgumentParser(description="Synthetic Blender training-data generator")
    parser.add_argument("--limit", type=int, default=0,
                        help="process at most N unprocessed seeds this run (0 = all)")
    parser.add_argument("--show", type=int, default=0,
                        help="print N random accepted samples at the end")
    parser.add_argument("--no-execute", action="store_true",
                        help="skip headless-Blender execution validation")
    args = parser.parse_args()
    execute = not args.no_execute
    print("--- SCRIPT STARTING ---")
    logger.info("Starting validated generation process")

    validator.load_cache()  # builds data/bpy_api_cache.json headless if missing
    POOL.refresh(_headers())
    print(f"Using {len(POOL.models)} models: {', '.join(POOL.models)}")
    logger.info(f"Models: {POOL.models}")

    with open(CONFIG["seed_file"], 'r', encoding='utf-8') as f:
        seeds = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(seeds)} seeds.")

    # dedup baseline + resume: seed_ids and code hashes already in the output
    seen_hashes = set()
    done = set()
    accepted = 0
    if os.path.exists(CONFIG['output_file']):
        with open(CONFIG['output_file'], 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if isinstance(r.get("answer"), str):
                        seen_hashes.add(validator.code_hash(r["answer"]))
                        accepted += 1
                    if isinstance(r.get("seed_id"), int):
                        done.add(r["seed_id"])
                except Exception:
                    pass
    print(f"Resuming: {accepted} samples on disk, {len(done)} seeds already processed.")

    if accepted >= CONFIG["cap"]:
        print(f"Already reached the cap of {CONFIG['cap']}. Nothing to do.")
        return

    # topic buckets for type 4 (related seeds only)
    buckets: Dict[str, List[int]] = {}
    for i, s in enumerate(seeds):
        buckets.setdefault(topic_key(s), []).append(i)

    stats = RejectionStats()
    accepted_rows = []  # for --show
    lock = threading.Lock()

    def run_one(i: int) -> None:
        seed = seeds[i]
        # Type 6 (masked completion) is gone; FIM rows are built from validated
        # code after generation (append_fim_rows).
        gen_type = (3, 4, 5, 7, 8)[i % 5]
        for attempt in range(CONFIG["max_retries"]):
            if gen_type == 4:
                same = [j for j in buckets[topic_key(seed)] if j != i]
                if len(same) < 2:
                    stats.record(gen_type, "no_related_seeds")
                    return
                res = generate_type_4([seed] + [seeds[j] for j in random.sample(same, 2)])
            else:
                res = {3: generate_type_3, 5: generate_type_5,
                       7: generate_type_7, 8: generate_type_8}[gen_type](seed)
            if res is None:
                stats.record(gen_type, "seed_unusable")
                return
            if not res:
                stats.record(gen_type, "llm_empty_or_unparsed")
                continue
            # !!! Why validate format before the Blender-API check?
            # Because a malformed Q&A pair is useless for supervised fine-tuning
            # even if the code is correct; the model learns from the exact
            # question/answer text, so shape and fences matter as much as API.
            ok, reason = validator.validate_sample(res["question"], res["answer"],
                                                   gen_type, seen_hashes,
                                                   strict_format=True)
            if ok and execute:
                blocks = validator.extract_fences(res["answer"])
                ok, reason = execution_validator.validate(blocks[0] if blocks else "")
            stats.record(gen_type, reason)
            if ok:
                row = {"question": res["question"], "answer": res["answer"],
                       "type": str(gen_type), "seed_id": i, "source": "synthetic"}
                with lock:
                    # another thread may have hit the cap while this one was validating
                    if state["accepted"] >= CONFIG["cap"]:
                        return
                    with open(CONFIG['output_file'], 'a', encoding='utf-8') as out_f:
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    state["accepted"] += 1
                    accepted_rows.append(row)
                return
        stats.record(gen_type, "retries_exhausted")

    # shortcut: failed seeds are retried on every restart (resume only knows
    # accepted seed_ids); fine while seeds are plentiful, add a skip-file if
    # rejected seeds start burning quota
    todo = [i for i in range(len(seeds)) if i not in done]
    if args.limit:
        todo = todo[:args.limit]
    state = {"accepted": accepted}
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as pool:
        futures = []
        for i in todo:
            with lock:
                if state["accepted"] >= CONFIG["cap"]:
                    break
            futures.append(pool.submit(run_one, i))
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            try:
                fut.result()
            except Exception as e:
                logger.error(f"Seed task failed: {e}")

    # FIM needs multi-statement code; seeds are one-line API references, so
    # build it from the code answers that just passed validation.
    fim_limit = min(CONFIG["cap"] - state["accepted"], max(1, state["accepted"] // 4))
    added = append_fim_rows(CONFIG["output_file"], seen_hashes, fim_limit,
                            execute, stats)
    print(f"Added {added} FIM rows.")

    print("\n--- RUN REPORT (per type / rejection reason) ---")
    print(stats.report())
    if args.show and accepted_rows:
        print(f"\n--- {min(args.show, len(accepted_rows))} RANDOM ACCEPTED SAMPLES ---")
        for row in random.sample(accepted_rows, min(args.show, len(accepted_rows))):
            print(f"\n{'='*70}\n[type={row['type']} seed={row['seed_id']}]\n"
                  f"QUESTION: {row['question']}\n\nANSWER: {row['answer']}")
    print("--- SCRIPT FINISHED ---")


if __name__ == "__main__":
    main()