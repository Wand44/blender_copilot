"""Synthetic training-data generator for the Blender QLoRA model.

Output rows stay {"question", "answer"} (plain strings) for training-script
compatibility. All code answers are ```python-fenced and validated against
the real Blender API (see validator.py) before being written.
"""
import os
import ast
import difflib
import logging
import json
import time
import random
import requests
import re
import sys
import tempfile
import textwrap
import threading
import subprocess
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Optional

# Allow `python3 pipeline/generate_synthetic.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config, execution_validator, validator
from core.gpu import warn_if_busy
from core.omniroute import ensure_running
from core.progress import emit
from core.rows import utc_now

config.LOGS.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(config.LOG_FILE),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CONFIG = {
    "omniroute_url": "http://localhost:20128/v1/chat/completions",
    "seed_file": str(config.DOC_FILE),
    "output_file": str(config.SYNTHETIC_FILE),
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
_LAST_MODEL = threading.local()


def _headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OMNIRoute_API_KEY', 'sk-no-key')}",
    }


# ==============================================================================
# LLM TRANSPORT (each retry rotates to the next model)
# ==============================================================================
def _content_text(content) -> str:
    """Some providers return content as a list of parts instead of a string."""
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return (content or "").strip()


def call_llm_ex(prompt: str, system_prompt: str,
                timeout: int = None, max_attempts: int = None):
    """Call the pool and return (text, model). Model is kept for row metadata."""
    attempts = max_attempts or CONFIG["max_llm_attempts"]
    timeout = timeout or 120
    for _attempt in range(attempts):
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
                                     json=payload, timeout=timeout)
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
            text = _content_text(response.json()["choices"][0]["message"]["content"])
            if not text:
                POOL.bad(model)
                continue
            POOL.ok(model)
            _LAST_MODEL.value = model
            return text, model
        except Exception as e:
            POOL.bad(model)
            logger.warning(f"{model}: call failed ({e}), rotating")
            time.sleep(0.5)
    _LAST_MODEL.value = ""
    return "", ""


def call_llm(prompt: str, system_prompt: str, timeout: int = None,
             max_attempts: int = None) -> str:
    return call_llm_ex(prompt, system_prompt, timeout=timeout,
                       max_attempts=max_attempts)[0]


def last_model() -> str:
    return getattr(_LAST_MODEL, "value", "")


def validate_candidate(question: str, answer: str, gen_type, seen_hashes: set,
                       execute: bool = True):
    """Single validation path for every generated row (LLM and targeted)."""
    ok, reason = validator.validate_sample(question, answer, gen_type,
                                           seen_hashes, strict_format=True)
    if ok and execute:
        blocks = validator.extract_fences(answer)
        ok, reason = execution_validator.validate(blocks[0] if blocks else "")
    return ok, reason


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
        row.update({"type": "fim", "seed_id": r.get("seed_id"), "source": "synthetic",
                    "producing_model": "programmatic:fim", "validated": True,
                    "attempts": 0, "timestamp": utc_now()})
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
    parser.add_argument("--batch", type=int, default=20,
                        help="targeted rows per area this run (0 = skip)")
    args = parser.parse_args()
    execute = not args.no_execute
    print("--- SCRIPT STARTING ---")
    logger.info("Starting validated generation process")
    warn_if_busy()

    validator.load_cache()  # builds data/generated/bpy_api_cache.json headless if missing
    Path(CONFIG["output_file"]).parent.mkdir(parents=True, exist_ok=True)
    print("Ensuring OmniRoute is running ...", flush=True)
    omniroute_ready, omniroute_note = ensure_running()
    if not omniroute_ready:
        raise SystemExit(
            f"OmniRoute is not available ({omniroute_note}). "
            "Run `omniroute serve` first, then retry."
        )
    print(f"OmniRoute: {omniroute_note}", flush=True)
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
            ok, reason = validate_candidate(res["question"], res["answer"],
                                            gen_type, seen_hashes, execute)
            stats.record(gen_type, reason)
            if ok:
                row = {"question": res["question"], "answer": res["answer"],
                       "type": str(gen_type), "seed_id": i, "source": "synthetic",
                       "producing_model": last_model() or "unknown",
                       "validated": True, "attempts": 0, "timestamp": utc_now()}
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
        completed = 0
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            try:
                fut.result()
            except Exception as e:
                logger.error(f"Seed task failed: {e}")
            completed += 1
            emit(completed, len(futures), "generating synthetic rows")

    # FIM needs multi-statement code; seeds are one-line API references, so
    # build it from the code answers that just passed validation.
    fim_limit = min(CONFIG["cap"] - state["accepted"], max(1, state["accepted"] // 4))
    added = append_fim_rows(CONFIG["output_file"], seen_hashes, fim_limit,
                            execute, stats)
    print(f"Added {added} FIM rows.")

    # Targeted (deterministic) rows: error-fix + topic coverage, validated
    # through the same validate_candidate path as the LLM rows above.
    remaining = CONFIG["cap"] - state["accepted"]
    added = run_targeted(CONFIG["output_file"], args.batch, seen_hashes,
                         execute, remaining)
    state["accepted"] += added

    print("\n--- RUN REPORT (per type / rejection reason) ---")
    print(stats.report())
    if args.show and accepted_rows:
        print(f"\n--- {min(args.show, len(accepted_rows))} RANDOM ACCEPTED SAMPLES ---")
        for row in random.sample(accepted_rows, min(args.show, len(accepted_rows))):
            print(f"\n{'='*70}\n[type={row['type']} seed={row['seed_id']}]\n"
                  f"QUESTION: {row['question']}\n\nANSWER: {row['answer']}")
    print("--- SCRIPT FINISHED ---")


# ==============================================================================
# TARGETED (DETERMINISTIC) ROWS: error-fix + topic coverage
# ==============================================================================
BLENDER = config.BLENDER_BIN


def run_blender_code(code, timeout=60):
    script = textwrap.dedent("""\
        import bpy, json, sys, traceback
        bpy.ops.wm.read_factory_settings(use_empty=True)
        if not bpy.data.scenes:
            bpy.data.scenes.new("Scene")
        bpy.ops.mesh.primitive_cube_add()
        ns = {"bpy": bpy, "__name__": "__main__"}
        try:
            exec(compile(sys.argv[-1], "<code>", "exec"), ns)
            print("###RES###" + json.dumps({"ok": True}))
        except Exception as e:
            tb = traceback.format_exc()
            print("###RES###" + json.dumps({
                "ok": False,
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": tb,
            }))
        bpy.ops.wm.quit_blender()
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        proc = subprocess.run(
            [BLENDER, "-b", "--factory-startup", "--python", path, "--", code],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "TimeoutError", f"exceeded {timeout}s", ""
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("###RES###"):
            try:
                out = json.loads(line[len("###RES###"):])
            except ValueError:
                continue
            return out.get("ok", False), out.get("error_type", ""), out.get("error", ""), out.get("traceback", "")
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
    return False, "Crash", " | ".join(tail), ""


def make_answer(opener, code):
    opener = (opener or "").strip()
    block = f"```python\n{code.strip()}\n```"
    return f"{opener}\n{block}" if opener else block


BASE_SCRIPTS = [
("monkey_subdiv", """import bpy
bpy.ops.mesh.primitive_monkey_add(location=(0, 0, 1))
obj = bpy.context.active_object
obj.name = "Suzanne"
mod = obj.modifiers.new(name="Subsurf", type='SUBSURF')
mod.levels = 2
bpy.ops.object.shade_smooth()
"""),
("camera_track", """import bpy
cam_data = bpy.data.cameras.new(name="Cam")
cam = bpy.data.objects.new("Cam", cam_data)
bpy.context.collection.objects.link(cam)
cam.location = (0, -6, 3)
target = bpy.context.active_object
if target:
    con = cam.constraints.new(type='TRACK_TO')
    con.target = target
    con.track_axis = 'TRACK_NEGATIVE_Z'
    con.up_axis = 'UP_Y'
"""),
("point_light_energy", """import bpy
light_data = bpy.data.lights.new(name="Fill", type='POINT')
light_data.energy = 500
light = bpy.data.objects.new("Fill", light_data)
bpy.context.collection.objects.link(light)
light.location = (3, -3, 4)
"""),
("material_assign_slots", """import bpy
obj = bpy.context.active_object
if obj and obj.type == 'MESH':
    mat = bpy.data.materials.new(name="BlueMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0, 0, 1, 1)
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
"""),
("render_output", """import bpy
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 64
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.filepath = "/tmp/render.png"
"""),
("keyframe_scale", """import bpy
obj = bpy.context.active_object
if obj:
    obj.scale = (1, 1, 1)
    obj.keyframe_insert(data_path="scale", frame=1)
    obj.scale = (2, 2, 2)
    obj.keyframe_insert(data_path="scale", frame=24)
"""),
("fcurve_linear", """import bpy
obj = bpy.context.active_object
if obj and obj.animation_data and obj.animation_data.action:
    action = obj.animation_data.action
    for fc in action.fcurves:
        if fc.data_path == "location":
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'
"""),
("bmesh_bevel", """import bpy
import bmesh
obj = bpy.context.active_object
if obj and obj.type == 'MESH':
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    edges = [e for e in bm.edges]
    bmesh.ops.bevel(bm, geom=edges, offset=0.05, segments=2, affect='EDGES')
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')
"""),
("from_pydata_pyramid", """import bpy
verts = [(1,1,0),(1,-1,0),(-1,-1,0),(-1,1,0),(0,0,2)]
faces = [(0,1,2,3),(0,1,4),(1,2,4),(2,3,4),(3,0,4)]
mesh = bpy.data.meshes.new("PyramidMesh")
mesh.from_pydata(verts, [], faces)
mesh.update()
obj = bpy.data.objects.new("Pyramid", mesh)
bpy.context.collection.objects.link(obj)
"""),
("collection_create_link", """import bpy
col = bpy.data.collections.new("Assets")
bpy.context.scene.collection.children.link(col)
obj = bpy.context.active_object
if obj and obj.name in bpy.context.scene.collection.objects:
    bpy.context.scene.collection.objects.unlink(obj)
    col.objects.link(obj)
"""),
("parent_objects", """import bpy
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
parent = bpy.context.active_object
bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 2))
child = bpy.context.active_object
child.parent = parent
child.matrix_parent_inverse = parent.matrix_world.inverted()
"""),
("shader_mix", """import bpy
mat = bpy.data.materials.new(name="MixMat")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()
out = nodes.new(type="ShaderNodeOutputMaterial")
bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
"""),
("rename_objects_loop", """import bpy
for i, obj in enumerate(bpy.data.objects):
    if obj.type == 'MESH':
        obj.name = f"Mesh_{i:02d}"
"""),
("hide_collection_view", """import bpy
col = bpy.context.scene.collection
for obj in col.objects:
    obj.hide_viewport = False
    obj.hide_render = False
"""),
("apply_modifier", """import bpy
obj = bpy.context.active_object
if obj and obj.type == 'MESH':
    mod = obj.modifiers.new(name="Bevel", type='BEVEL')
    mod.width = 0.1
    bpy.ops.object.modifier_apply(modifier=mod.name)
"""),
("cursor_to_selected", """import bpy
obj = bpy.context.active_object
if obj:
    bpy.context.scene.cursor.location = obj.location.copy()
"""),
("duplicate_object", """import bpy
obj = bpy.context.active_object
if obj:
    new_obj = obj.copy()
    new_obj.data = obj.data.copy()
    new_obj.location.x += 2.0
    bpy.context.collection.objects.link(new_obj)
"""),
("text_object", """import bpy
curve = bpy.data.curves.new(type="FONT", name="TextCurve")
curve.body = "Hello"
obj = bpy.data.objects.new(name="Text", object_data=curve)
bpy.context.collection.objects.link(obj)
obj.location = (0, 0, 1)
"""),
("set_origin", """import bpy
obj = bpy.context.active_object
if obj and obj.type == 'MESH':
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
"""),
("smooth_flat_switch", """import bpy
obj = bpy.context.active_object
if obj and obj.type == 'MESH':
    bpy.ops.object.shade_flat()
    bpy.ops.object.shade_smooth()
"""),
("world_background", """import bpy
world = bpy.context.scene.world
if world is None:
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg:
    bg.inputs[0].default_value = (0.1, 0.1, 0.1, 1)
"""),
("add_plane_scale", """import bpy
bpy.ops.mesh.primitive_plane_add(size=4)
obj = bpy.context.active_object
obj.scale = (2, 2, 1)
"""),
("select_by_name", """import bpy
bpy.ops.object.select_all(action='DESELECT')
obj = bpy.data.objects.get("Cube")
if obj:
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
"""),
("edge_split_modifier", """import bpy
obj = bpy.context.active_object
if obj and obj.type == 'MESH':
    mod = obj.modifiers.new(name="EdgeSplit", type='EDGE_SPLIT')
    mod.split_angle = 1.0472
"""),
("area_lamp", """import bpy
light_data = bpy.data.lights.new(name="Area", type='AREA')
light_data.energy = 200
obj = bpy.data.objects.new("Area", light_data)
bpy.context.collection.objects.link(obj)
obj.location = (0, 0, 5)
"""),
("camera_settings", """import bpy
cam_data = bpy.data.cameras.new(name="CamData")
cam_data.lens = 35
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj
"""),
("frame_range", """import bpy
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 120
scene.frame_current = 1
"""),
("vertex_group_assign", """import bpy
obj = bpy.context.active_object
if obj and obj.type == 'MESH':
    vg = obj.vertex_groups.new(name="Group")
    vg.add([v.index for v in obj.data.vertices], 1.0, 'REPLACE')
"""),
("join_meshes", """import bpy
bpy.ops.object.select_all(action='DESELECT')
meshes = [o for o in bpy.data.objects if o.type == 'MESH']
if meshes:
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
"""),
("extrude_bmesh", """import bpy
import bmesh
obj = bpy.context.active_object
if obj and obj.type == 'MESH':
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    top = [f for f in bm.faces if f.normal.z > 0.9]
    if top:
        bmesh.ops.extrude_face_region(bm, geom=top)
        bmesh.ops.translate(bm, verts=[v for v in bm.verts if v.select], vec=(0, 0, 1))
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')
"""),
("custom_prop", """import bpy
obj = bpy.context.active_object
if obj:
    obj["my_id"] = 42
    obj["tag"] = "asset"
"""),
("remove_unused_mats", """import bpy
obj = bpy.context.active_object
if obj and obj.type == 'MESH':
    obj.data.materials.clear()
"""),
("sun_light_rotate", """import bpy
import math
light_data = bpy.data.lights.new(name="Sun", type='SUN')
light_data.energy = 2.0
obj = bpy.data.objects.new("Sun", light_data)
bpy.context.collection.objects.link(obj)
obj.rotation_euler = (math.radians(45), 0, math.radians(30))
"""),
("cycles_device", """import bpy
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 32
"""),
("image_texture_node", """import bpy
mat = bpy.data.materials.new(name="TexMat")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
tex = nodes.new(type="ShaderNodeTexImage")
bsdf = nodes.get("Principled BSDF")
if bsdf:
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
"""),
("duplicate_linked", """import bpy
obj = bpy.context.active_object
if obj:
    dup = obj.copy()
    dup.location.y += 2.0
    bpy.context.collection.objects.link(dup)
"""),
("snap_cursor", """import bpy
bpy.ops.view3d.snap_cursor_to_center()
"""),
("array_modifier", """import bpy
obj = bpy.context.active_object
if obj and obj.type == 'MESH':
    mod = obj.modifiers.new(name="Array", type='ARRAY')
    mod.count = 5
    mod.relative_offset_displace = (1.2, 0, 0)
"""),
("boolean_modifier", """import bpy
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
a = bpy.context.active_object
bpy.ops.mesh.primitive_uv_sphere_add(location=(0.5, 0, 0))
b = bpy.context.active_object
mod = a.modifiers.new(name="Bool", type='BOOLEAN')
mod.operation = 'DIFFERENCE'
mod.object = b
"""),
("collection_instance", """import bpy
col = bpy.data.collections.get("Assets")
if col is None:
    col = bpy.data.collections.new("Assets")
    bpy.context.scene.collection.children.link(col)
inst = bpy.data.objects.new("AssetsInstance", None)
inst.instance_type = 'COLLECTION'
inst.instance_collection = col
bpy.context.collection.objects.link(inst)
"""),
("animation_action_name", """import bpy
obj = bpy.context.active_object
if obj and obj.animation_data and obj.animation_data.action:
    obj.animation_data.action.name = "WalkCycle"
"""),
("mesh_normals", """import bpy
obj = bpy.context.active_object
if obj and obj.type == 'MESH':
    mesh = obj.data
    mesh.normals_split_custom_set_from_vertices([v.normal for v in mesh.vertices])
"""),
]

# ensure >=40
assert len(BASE_SCRIPTS) >= 40, len(BASE_SCRIPTS)


def _replace_once(code, old, new):
    if old in code:
        return code.replace(old, new, 1)
    return None

def b_scene_link(code):
    return _replace_once(code, "bpy.context.collection.objects.link", "bpy.context.scene.objects.link")
def b_scene_active(code):
    return re.sub(r"bpy\.context\.view_layer\.objects\.active", "bpy.context.scene.objects.active", code) if "view_layer.objects.active" in code else None
def b_matmul_star(code):
    return re.sub(r"(\w+(?:\.\w+)*(?:\([^)]*\))?)\s*@\s*(\w+(?:\.\w+)*(?:\([^)]*\))?)", r"\1 * \2", code, count=1) if "@" in code else None
def b_float_location(code):
    m = re.search(r"(\w+)\.location\s*=\s*\([^\)]+\)", code)
    if m:
        return re.sub(rf"{re.escape(m.group(1))}\.location\s*=\s*\([^\)]+\)", f"{m.group(1)}.location = 1.0", code, count=1)
    return None
def b_float_scale(code):
    m = re.search(r"(\w+)\.scale\s*=\s*\([^\)]+\)", code)
    if m:
        return re.sub(rf"{re.escape(m.group(1))}\.scale\s*=\s*\([^\)]+\)", f"{m.group(1)}.scale = 2.0", code, count=1)
    return None
def b_wrong_op_cube(code):
    return _replace_once(code, "bpy.ops.mesh.primitive_cube_add", "bpy.ops.mesh.add_cube")
def b_wrong_op_sphere(code):
    return _replace_once(code, "bpy.ops.mesh.primitive_uv_sphere_add", "bpy.ops.mesh.primitive_sphere_add")
def b_wrong_op_smooth(code):
    return _replace_once(code, "bpy.ops.object.shade_smooth", "bpy.ops.object.smooth_shade")
def b_wrong_op_flat(code):
    return _replace_once(code, "bpy.ops.object.shade_flat", "bpy.ops.object.flat_shade")
def b_wrong_op_select_all(code):
    return _replace_once(code, "bpy.ops.object.select_all", "bpy.ops.object.select_everything")
def b_wrong_op_monkey(code):
    return _replace_once(code, "bpy.ops.mesh.primitive_monkey_add", "bpy.ops.mesh.monkey_add")
def b_wrong_attr_hide_viewport(code):
    return _replace_once(code, ".hide_viewport", ".hide")
def b_wrong_attr_use_nodes(code):
    return _replace_once(code, ".use_nodes", ".use_node")
def b_wrong_attr_render_engine(code):
    return _replace_once(code, "render.engine", "render.render_engine")
def b_wrong_attr_filepath(code):
    return _replace_once(code, "render.filepath", "render.output_path")
def b_wrong_attr_lens(code):
    return _replace_once(code, ".lens", ".focal_length")
def b_wrong_attr_energy(code):
    return _replace_once(code, ".energy", ".power")
def b_wrong_enum_engine(code):
    return _replace_once(code, "'BLENDER_EEVEE'", "'EEVEE'")
def b_wrong_enum_interp(code):
    return _replace_once(code, "'LINEAR'", "'LINE'")
def b_wrong_enum_track(code):
    return _replace_once(code, "'TRACK_NEGATIVE_Z'", "'TRACK_Z'")
def b_readonly_vertices(code):
    return _replace_once(code, "obj.data.vertices", "obj.vertices")
def b_readonly_materials(code):
    return _replace_once(code, "obj.data.materials", "obj.materials")
def b_wrong_collection_method(code):
    return _replace_once(code, ".objects.link(", ".objects.append(")
def b_wrong_collection_unlink(code):
    return _replace_once(code, ".objects.unlink(", ".objects.remove(")
def b_removed_import_bmesh(code):
    if "import bmesh" in code and "bmesh." in code:
        return code.replace("import bmesh\n", "", 1)
    return None
def b_removed_import_math(code):
    if "import math" in code and "math." in code:
        return code.replace("import math\n", "", 1)
    return None
def b_removed_import_mathutils(code):
    # not present but try
    if "import mathutils" in code and "mathutils." in code:
        return code.replace("import mathutils\n", "", 1)
    return None
def b_missing_mode_edit(code):
    if "bpy.ops.object.mode_set(mode='EDIT')" in code:
        return code.replace("bpy.ops.object.mode_set(mode='EDIT')\n", "", 1)
    return None
def b_missing_mode_object(code):
    if "bpy.ops.object.mode_set(mode='OBJECT')" in code and "bmesh.update_edit_mesh" in code:
        return code.replace("bpy.ops.object.mode_set(mode='OBJECT')\n", "", 1)
    return None
def b_wrong_context_active(code):
    # set wrong active before operator
    if "bpy.context.view_layer.objects.active = " in code:
        return code.replace("bpy.context.view_layer.objects.active = ", "bpy.context.scene.camera = ", 1)
    return None
def b_wrong_select_call(code):
    return _replace_once(code, "obj.select_set(True)", "obj.select = True")
def b_wrong_children_link(code):
    return _replace_once(code, ".children.link(", ".children.append(")
def b_wrong_modifier_apply(code):
    return _replace_once(code, "bpy.ops.object.modifier_apply", "bpy.ops.object.apply_modifier")
def b_wrong_keyframe(code):
    return _replace_once(code, "keyframe_insert", "insert_keyframe")

BREAKERS = [
 ("legacy_scene_link", b_scene_link, "In 2.8+ objects are linked to collections, not `scene.objects`."),
 ("legacy_scene_active", b_scene_active, "`scene.objects.active` was removed; use `view_layer.objects.active`."),
 ("legacy_matrix_mul", b_matmul_star, "Matrix multiplication uses `@` in Python 3 / Blender 2.8+, not `*`."),
 ("float_to_location", b_float_location, "Object location is a vector, not a single float."),
 ("float_to_scale", b_float_scale, "Object scale is a vector, not a single float."),
 ("wrong_op_cube", b_wrong_op_cube, "The correct operator is `bpy.ops.mesh.primitive_cube_add`."),
 ("wrong_op_sphere", b_wrong_op_sphere, "Use `bpy.ops.mesh.primitive_uv_sphere_add` for a UV sphere."),
 ("wrong_op_smooth", b_wrong_op_smooth, "Smooth shading uses `bpy.ops.object.shade_smooth`."),
 ("wrong_op_flat", b_wrong_op_flat, "Flat shading uses `bpy.ops.object.shade_flat`."),
 ("wrong_op_select_all", b_wrong_op_select_all, "The operator is `bpy.ops.object.select_all`."),
 ("wrong_op_monkey", b_wrong_op_monkey, "The operator is `bpy.ops.mesh.primitive_monkey_add`."),
 ("wrong_attr_hide_viewport", b_wrong_attr_hide_viewport, "Viewport visibility uses `hide_viewport` in 2.8+."),
 ("wrong_attr_use_nodes", b_wrong_attr_use_nodes, "Materials use the `use_nodes` property."),
 ("wrong_attr_render_engine", b_wrong_attr_render_engine, "The engine property is `scene.render.engine`."),
 ("wrong_attr_filepath", b_wrong_attr_filepath, "Output path is set via `scene.render.filepath`."),
 ("wrong_attr_lens", b_wrong_attr_lens, "Camera focal length is stored in `data.lens`."),
 ("wrong_attr_energy", b_wrong_attr_energy, "Light intensity uses the `energy` property."),
 ("wrong_enum_engine", b_wrong_enum_engine, "The Eevee engine enum is `BLENDER_EEVEE` in current Blender."),
 ("wrong_enum_interp", b_wrong_enum_interp, "Keyframe interpolation enum is `LINEAR`, not `LINE`."),
 ("wrong_enum_track", b_wrong_enum_track, "Track-to axis enum is `TRACK_NEGATIVE_Z`."),
 ("readonly_vertices", b_readonly_vertices, "Mesh vertices live on `obj.data.vertices`, not `obj.vertices`."),
 ("readonly_materials", b_readonly_materials, "Material slots live on `obj.data.materials`."),
 ("wrong_collection_link", b_wrong_collection_method, "Collections use `objects.link(obj)`, not `append`.") ,
 ("wrong_collection_unlink", b_wrong_collection_unlink, "Use `objects.unlink(obj)` to remove from a collection."),
 ("missing_import_bmesh", b_removed_import_bmesh, "`bmesh` must be imported before use."),
 ("missing_import_math", b_removed_import_math, "The `math` module must be imported before use."),
 ("missing_mode_edit", b_missing_mode_edit, "Edit-mode mesh operators require switching to Edit Mode first."),
 ("missing_mode_object", b_missing_mode_object, "Return to Object Mode after editing bmesh data."),
 ("wrong_select_api", b_wrong_select_call, "Selection uses `obj.select_set(True)`, not `obj.select`."),
 ("wrong_children_link", b_wrong_children_link, "Collection children are linked with `.children.link()`."),
 ("wrong_modifier_apply", b_wrong_modifier_apply, "Apply modifiers with `bpy.ops.object.modifier_apply(modifier=...)`."),
 ("wrong_keyframe_insert", b_wrong_keyframe, "Use `keyframe_insert` to insert keyframes."),
]
assert len(BREAKERS) >= 30, len(BREAKERS)


def trim_traceback(tb):
    lines = [l.rstrip() for l in (tb or "").splitlines() if l.strip()]
    # keep relevant tail
    keep = []
    for l in lines:
        if 'File "' in l or l.startswith('Traceback') or 'Error' in l or l.strip().startswith('bpy'):
            keep.append(l)
    if not keep:
        keep = lines[-6:]
    return "\n".join(keep[-8:])


def question_variants(broken, err_text, tb, rng):
    short = f"{err_text}"
    full_tb = trim_traceback(tb)
    variants = [
        f"This Blender script raises an error. How can I fix it?\n```python\n{broken}\n```\nError: {short}",
        f"Running this in Blender gives `{short}`. What's wrong and how should it be corrected?\n```python\n{broken}\n```",
        f"Can you fix this script?\n```python\n{broken}\n```\n```\n{full_tb}\n```",
        f"I get this traceback when I run my Blender code:\n```\n{full_tb}\n```\nCode:\n```python\n{broken}\n```\nPlease provide a working fix.",
        f"Why does this fail with {short}?\n```python\n{broken}\n```",
    ]
    return rng.choice(variants)


A_OPENERS = [
    lambda exp: exp,
    lambda exp: exp + " Here's the corrected script.",
    lambda exp: exp,
    lambda exp: "The fix is to update the API usage.\n" + exp,
    lambda exp: "",
    lambda exp: "Corrected code:",
]


def gen_area_a(n=20, rng_seed=100, seen_hashes=None, execute=True):
    """Break known-good scripts, capture the real error, pair with the fix."""
    rng = random.Random(rng_seed)
    kept = []
    stats = Counter()
    pool = []
    for sname, scode in BASE_SCRIPTS:
        for tag, fn, explanation in BREAKERS:
            broken = fn(scode)
            if broken and broken != scode:
                pool.append((sname, scode, broken, tag, explanation))
    rng.shuffle(pool)
    attempts = 0
    for sname, fixed, broken, tag, explanation in pool:
        if len(kept) >= n:
            break
        attempts += 1
        ok_broken, err_type, err_msg, tb = run_blender_code(broken)
        if ok_broken:
            stats["broken_ran_ok"] += 1
            continue
        err_text = f"{err_type}: {err_msg}".strip()
        question = question_variants(broken, err_text, tb or err_text, rng)
        opener = rng.choice(A_OPENERS)(explanation)
        answer = make_answer(opener, fixed)
        ok, reason = validate_candidate(question, answer, "bugfix_explained",
                                        seen_hashes, execute)
        if not ok:
            stats[f"verify_reject:{reason}"] += 1
            continue
        kept.append({
            "question": question,
            "answer": answer,
            "type": "bugfix_explained",
            "seed_id": None,
            "source": "synthetic",
            "producing_model": "programmatic:targeted",
            "validated": True,
            "verified": True,
            "attempts": 0,
            "topic_area": f"error_fix/{tag}",
            "timestamp": utc_now(),
        })
        stats[f"kept:{tag}"] += 1
        stats[f"base:{sname}"] += 1
    stats["candidates"] = attempts
    stats["kept_total"] = len(kept)
    return kept, stats


# -------------------- B: topic coverage --------------------
B_TOPICS = {
"render_cycles": [
("How do I switch the render engine to Cycles and set 128 samples?",
 "import bpy\nscene = bpy.context.scene\nscene.render.engine = 'CYCLES'\nscene.cycles.samples = 128\n"),
("Set scene.render.engine to CYCLES with 64 render samples in Python.",
 "import bpy\nscene = bpy.context.scene\nscene.render.engine = 'CYCLES'\nscene.cycles.samples = 64\n"),
("Configure Cycles preview samples to 32 for the current scene.",
 "import bpy\nscene = bpy.context.scene\nscene.render.engine = 'CYCLES'\nscene.cycles.preview_samples = 32\n"),
("Script to use Cycles on CPU with 16 samples.",
 "import bpy\nscene = bpy.context.scene\nscene.render.engine = 'CYCLES'\nscene.cycles.device = 'CPU'\nscene.cycles.samples = 16\n"),
("Change the active scene to Cycles and set both render and preview samples.",
 "import bpy\nscene = bpy.context.scene\nscene.render.engine = 'CYCLES'\nscene.cycles.samples = 48\nscene.cycles.preview_samples = 16\n"),
],
"modifiers_shade": [
("Add a Subdivision Surface modifier to the active mesh and smooth-shade it.",
 "import bpy\nobj = bpy.context.active_object\nif obj and obj.type == 'MESH':\n    mod = obj.modifiers.new(name='Subsurf', type='SUBSURF')\n    mod.levels = 2\n    bpy.ops.object.shade_smooth()\n"),
("Give the active mesh a Bevel modifier with width 0.1.",
 "import bpy\nobj = bpy.context.active_object\nif obj and obj.type == 'MESH':\n    bev = obj.modifiers.new(name='Bevel', type='BEVEL')\n    bev.width = 0.1\n"),
("How can I add a Solidify modifier and enable smooth shading in one script?",
 "import bpy\nobj = bpy.context.active_object\nif obj and obj.type == 'MESH':\n    bpy.ops.object.shade_smooth()\n    obj.modifiers.new(name='Solidify', type='SOLIDIFY')\n"),
("Apply flat shading then add an Array modifier with 3 copies.",
 "import bpy\nobj = bpy.context.active_object\nif obj and obj.type == 'MESH':\n    bpy.ops.object.shade_flat()\n    mod = obj.modifiers.new(name='Array', type='ARRAY')\n    mod.count = 3\n"),
("Add an Edge Split modifier to the active mesh.",
 "import bpy\nobj = bpy.context.active_object\nif obj and obj.type == 'MESH':\n    mod = obj.modifiers.new(name='EdgeSplit', type='EDGE_SPLIT')\n    mod.split_angle = 1.0472\n"),
],
"lights_cameras": [
("Create a point light via bpy.data and link it to the current collection.",
 "import bpy\nlight_data = bpy.data.lights.new(name='MyLight', type='POINT')\nlight_obj = bpy.data.objects.new(name='MyLight', object_data=light_data)\nbpy.context.collection.objects.link(light_obj)\nlight_obj.location = (2, -2, 3)\n"),
("Using bpy.data (not operators), add a sun light and set its energy.",
 "import bpy\nsun = bpy.data.lights.new(name='Sun', type='SUN')\nsun.energy = 3.0\nobj = bpy.data.objects.new(name='Sun', object_data=sun)\nbpy.context.collection.objects.link(obj)\n"),
("Create a camera datablock and object at (0, -5, 3) and set it active.",
 "import bpy\ncam_data = bpy.data.cameras.new(name='MyCam')\ncam_obj = bpy.data.objects.new(name='MyCam', object_data=cam_data)\nbpy.context.collection.objects.link(cam_obj)\ncam_obj.location = (0, -5, 3)\nbpy.context.scene.camera = cam_obj\n"),
("Add an area light with energy 200 using bpy.data APIs.",
 "import bpy\nlight_data = bpy.data.lights.new(name='Area', type='AREA')\nlight_data.energy = 200\nobj = bpy.data.objects.new(name='Area', light_data)\nbpy.context.collection.objects.link(obj)\nobj.location = (0, 0, 5)\n"),
("Make a camera with lens 35mm and link it to the scene.",
 "import bpy\ncam_data = bpy.data.cameras.new(name='Cam35')\ncam_data.lens = 35\ncam_obj = bpy.data.objects.new('Cam35', cam_data)\nbpy.context.collection.objects.link(cam_obj)\n"),
],
"animation_fcurves": [
("Insert location keyframes for the active object at frame 1 and frame 50.",
 "import bpy\nobj = bpy.context.active_object\nif obj:\n    obj.location = (0, 0, 0)\n    obj.keyframe_insert(data_path='location', frame=1)\n    obj.location = (5, 0, 0)\n    obj.keyframe_insert(data_path='location', frame=50)\n"),
("Keyframe the active object's Z rotation at frames 1 and 100.",
 "import bpy\nobj = bpy.context.active_object\nif obj:\n    obj.rotation_euler.z = 0.0\n    obj.keyframe_insert(data_path='rotation_euler', index=2, frame=1)\n    obj.rotation_euler.z = 3.14159\n    obj.keyframe_insert(data_path='rotation_euler', index=2, frame=100)\n"),
("Set all location fcurve keyframes on the active object to LINEAR interpolation.",
 "import bpy\nobj = bpy.context.active_object\nif obj and obj.animation_data and obj.animation_data.action:\n    for fc in obj.animation_data.action.fcurves:\n        if fc.data_path == 'location':\n            for kp in fc.keyframe_points:\n                kp.interpolation = 'LINEAR'\n"),
("Insert a scale keyframe at frame 1 and frame 24 for the active object.",
 "import bpy\nobj = bpy.context.active_object\nif obj:\n    obj.scale = (1, 1, 1)\n    obj.keyframe_insert(data_path='scale', frame=1)\n    obj.scale = (2, 2, 2)\n    obj.keyframe_insert(data_path='scale', frame=24)\n"),
("Change interpolation of every keyframe in the active action to BEZIER.",
 "import bpy\nobj = bpy.context.active_object\nif obj and obj.animation_data and obj.animation_data.action:\n    for fc in obj.animation_data.action.fcurves:\n        for kp in fc.keyframe_points:\n            kp.interpolation = 'BEZIER'\n"),
],
"materials_nodes": [
("Create a node material and set Principled BSDF base color to red.",
 "import bpy\nmat = bpy.data.materials.new(name='RedMat')\nmat.use_nodes = True\nbsdf = mat.node_tree.nodes.get('Principled BSDF')\nif bsdf:\n    bsdf.inputs['Base Color'].default_value = (1, 0, 0, 1)\n"),
("Build a material that links Principled BSDF to Material Output.",
 "import bpy\nmat = bpy.data.materials.new(name='NodeMat')\nmat.use_nodes = True\nnodes = mat.node_tree.nodes\nlinks = mat.node_tree.links\nbsdf = nodes.get('Principled BSDF')\noutput = nodes.get('Material Output')\nif bsdf and output:\n    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])\n"),
("Assign a new node material to the active mesh object.",
 "import bpy\nobj = bpy.context.active_object\nmat = bpy.data.materials.new(name='Assigned')\nmat.use_nodes = True\nif obj and obj.type == 'MESH':\n    if obj.data.materials:\n        obj.data.materials[0] = mat\n    else:\n        obj.data.materials.append(mat)\n"),
("Add an Image Texture node and connect its Color to Principled BSDF Base Color.",
 "import bpy\nmat = bpy.data.materials.new(name='TexMat')\nmat.use_nodes = True\nnodes = mat.node_tree.nodes\nlinks = mat.node_tree.links\ntex = nodes.new(type='ShaderNodeTexImage')\nbsdf = nodes.get('Principled BSDF')\nif bsdf:\n    links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])\n"),
("Create a node setup with a Principled BSDF and set its roughness to 0.2.",
 "import bpy\nmat = bpy.data.materials.new(name='RoughMat')\nmat.use_nodes = True\nbsdf = mat.node_tree.nodes.get('Principled BSDF')\nif bsdf:\n    bsdf.inputs['Roughness'].default_value = 0.2\n"),
],
"bmesh": [
("Use bmesh to subdivide all edges of the active mesh in edit mode.",
 "import bpy\nimport bmesh\nobj = bpy.context.active_object\nif obj and obj.type == 'MESH':\n    bpy.ops.object.mode_set(mode='EDIT')\n    bm = bmesh.from_edit_mesh(obj.data)\n    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=1, use_grid_fill=True)\n    bmesh.update_edit_mesh(obj.data)\n    bpy.ops.object.mode_set(mode='OBJECT')\n"),
("Translate all vertices of the active mesh up by 1.0 using bmesh.",
 "import bpy\nimport bmesh\nobj = bpy.context.active_object\nif obj and obj.type == 'MESH':\n    bm = bmesh.new()\n    bm.from_mesh(obj.data)\n    for v in bm.verts:\n        v.co.z += 1.0\n    bm.to_mesh(obj.data)\n    bm.free()\n"),
("Bevel all edges of the active mesh with bmesh operators.",
 "import bpy\nimport bmesh\nobj = bpy.context.active_object\nif obj and obj.type == 'MESH':\n    bpy.ops.object.mode_set(mode='EDIT')\n    bm = bmesh.from_edit_mesh(obj.data)\n    bmesh.ops.bevel(bm, geom=bm.edges[:], offset=0.05, segments=2, affect='EDGES')\n    bmesh.update_edit_mesh(obj.data)\n    bpy.ops.object.mode_set(mode='OBJECT')\n"),
("Create a bmesh, add a cube primitive, and write it to a new mesh.",
 "import bpy\nimport bmesh\nbm = bmesh.new()\nbmesh.ops.create_cube(bm, size=2.0)\nmesh = bpy.data.meshes.new('BMeshCube')\nbm.to_mesh(mesh)\nbm.free()\nobj = bpy.data.objects.new('BMeshCube', mesh)\nbpy.context.collection.objects.link(obj)\n"),
("Delete faces with normal pointing down using bmesh.",
 "import bpy\nimport bmesh\nobj = bpy.context.active_object\nif obj and obj.type == 'MESH':\n    bm = bmesh.new()\n    bm.from_mesh(obj.data)\n    faces = [f for f in bm.faces if f.normal.z < -0.5]\n    bmesh.ops.delete(bm, geom=faces, context='FACES')\n    bm.to_mesh(obj.data)\n    bm.free()\n"),
],
"from_pydata": [
("Create a quad mesh from vertices and faces with from_pydata.",
 "import bpy\nverts = [(0,0,0),(1,0,0),(1,1,0),(0,1,0)]\nfaces = [(0,1,2,3)]\nmesh = bpy.data.meshes.new('QuadMesh')\nmesh.from_pydata(verts, [], faces)\nmesh.update()\nobj = bpy.data.objects.new('Quad', mesh)\nbpy.context.collection.objects.link(obj)\n"),
("Build a triangle mesh using from_pydata and link it.",
 "import bpy\nverts = [(0,0,0),(2,0,0),(1,2,0)]\nfaces = [(0,1,2)]\nmesh = bpy.data.meshes.new('TriMesh')\nmesh.from_pydata(verts, [], faces)\nmesh.update()\nobj = bpy.data.objects.new('Triangle', mesh)\nbpy.context.collection.objects.link(obj)\n"),
("Construct a pyramid mesh from_pydata with 5 vertices.",
 "import bpy\nverts = [(1,1,0),(1,-1,0),(-1,-1,0),(-1,1,0),(0,0,2)]\nfaces = [(0,1,2,3),(0,1,4),(1,2,4),(2,3,4),(3,0,4)]\nmesh = bpy.data.meshes.new('PyramidMesh')\nmesh.from_pydata(verts, [], faces)\nmesh.update()\nobj = bpy.data.objects.new('Pyramid', mesh)\nbpy.context.collection.objects.link(obj)\n"),
("Make a line mesh from edges using from_pydata.",
 "import bpy\nverts = [(0,0,0),(1,0,0),(2,1,0),(3,1,0)]\nedges = [(0,1),(1,2),(2,3)]\nmesh = bpy.data.meshes.new('LineMesh')\nmesh.from_pydata(verts, edges, [])\nmesh.update()\nobj = bpy.data.objects.new('Line', mesh)\nbpy.context.collection.objects.link(obj)\n"),
("Create a custom two-triangle plane via from_pydata.",
 "import bpy\nverts = [(0,0,0),(2,0,0),(2,2,0),(0,2,0)]\nfaces = [(0,1,2),(0,2,3)]\nmesh = bpy.data.meshes.new('PlaneMesh')\nmesh.from_pydata(verts, [], faces)\nmesh.update()\nobj = bpy.data.objects.new('Plane2', mesh)\nbpy.context.collection.objects.link(obj)\n"),
],
"collections_parenting": [
("Create a new collection, link it to the scene, and move the active object into it.",
 "import bpy\ncol = bpy.data.collections.new('Assets')\nbpy.context.scene.collection.children.link(col)\nobj = bpy.context.active_object\nif obj and obj.name in bpy.context.scene.collection.objects:\n    bpy.context.scene.collection.objects.unlink(obj)\n    col.objects.link(obj)\n"),
("Parent one object to another and keep its world transform.",
 "import bpy\nbpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))\nparent = bpy.context.active_object\nbpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 2))\nchild = bpy.context.active_object\nchild.parent = parent\nchild.matrix_parent_inverse = parent.matrix_world.inverted()\n"),
("Make a nested collection hierarchy under the scene collection.",
 "import bpy\nparent_col = bpy.data.collections.new('ParentCol')\nchild_col = bpy.data.collections.new('ChildCol')\nbpy.context.scene.collection.children.link(parent_col)\nparent_col.children.link(child_col)\n"),
("Unlink an object from the scene collection and link it to a custom collection.",
 "import bpy\ncol = bpy.data.collections.new('Props')\nbpy.context.scene.collection.children.link(col)\nobj = bpy.context.active_object\nif obj and obj.name in bpy.context.scene.collection.objects:\n    bpy.context.scene.collection.objects.unlink(obj)\nif obj:\n    col.objects.link(obj)\n"),
("Create an empty as parent and parent the active mesh to it.",
 "import bpy\nempty = bpy.data.objects.new('ParentEmpty', None)\nbpy.context.collection.objects.link(empty)\nobj = bpy.context.active_object\nif obj and obj != empty:\n    obj.parent = empty\n    obj.matrix_parent_inverse = empty.matrix_world.inverted()\n"),
],
"camera_render_output": [
("Set the active camera and configure render resolution and output path.",
 "import bpy\ncam_data = bpy.data.cameras.new(name='Cam')\ncam_obj = bpy.data.objects.new('Camera', cam_data)\nbpy.context.collection.objects.link(cam_obj)\nbpy.context.scene.camera = cam_obj\nscene = bpy.context.scene\nscene.render.resolution_x = 1920\nscene.render.resolution_y = 1080\nscene.render.filepath = '/tmp/out.png'\n"),
("Change camera lens and set render percentage to 50%.",
 "import bpy\ncam_data = bpy.data.cameras.new(name='Cam')\ncam_data.lens = 50\ncam_obj = bpy.data.objects.new('Camera', cam_data)\nbpy.context.collection.objects.link(cam_obj)\nbpy.context.scene.camera = cam_obj\nbpy.context.scene.render.resolution_percentage = 50\n"),
("Set render output format to PNG and filepath for the scene.",
 "import bpy\nscene = bpy.context.scene\nscene.render.image_settings.file_format = 'PNG'\nscene.render.filepath = '/tmp/frame.png'\n"),
("Create a camera, position it, and set frame range for rendering.",
 "import bpy\ncam_data = bpy.data.cameras.new(name='Cam')\ncam_obj = bpy.data.objects.new('Camera', cam_data)\nbpy.context.collection.objects.link(cam_obj)\ncam_obj.location = (7, -7, 5)\nbpy.context.scene.camera = cam_obj\nscene = bpy.context.scene\nscene.frame_start = 1\nscene.frame_end = 250\n"),
("Configure render resolution 1280x720 and assign an existing camera if present.",
 "import bpy\nscene = bpy.context.scene\nscene.render.resolution_x = 1280\nscene.render.resolution_y = 720\nfor obj in bpy.data.objects:\n    if obj.type == 'CAMERA':\n        scene.camera = obj\n        break\n"),
],
"shader_node_links": [
("Create a Principled BSDF and link it to the Material Output surface input.",
 "import bpy\nmat = bpy.data.materials.new(name='LinkMat')\nmat.use_nodes = True\nnodes = mat.node_tree.nodes\nlinks = mat.node_tree.links\nbsdf = nodes.get('Principled BSDF')\noutput = nodes.get('Material Output')\nif bsdf and output:\n    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])\n"),
("Add a Noise Texture node and connect its Fac to Principled Roughness.",
 "import bpy\nmat = bpy.data.materials.new(name='NoiseMat')\nmat.use_nodes = True\nnodes = mat.node_tree.nodes\nlinks = mat.node_tree.links\nnoise = nodes.new(type='ShaderNodeTexNoise')\nbsdf = nodes.get('Principled BSDF')\nif bsdf:\n    links.new(noise.outputs['Fac'], bsdf.inputs['Roughness'])\n"),
("Connect an Image Texture Color output to Base Color of Principled BSDF.",
 "import bpy\nmat = bpy.data.materials.new(name='ImgMat')\nmat.use_nodes = True\nnodes = mat.node_tree.nodes\nlinks = mat.node_tree.links\ntex = nodes.new(type='ShaderNodeTexImage')\nbsdf = nodes.get('Principled BSDF')\nif bsdf:\n    links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])\n"),
("Wire a UV Map node into an Image Texture vector input.",
 "import bpy\nmat = bpy.data.materials.new(name='UVMat')\nmat.use_nodes = True\nnodes = mat.node_tree.nodes\nlinks = mat.node_tree.links\nuv = nodes.new(type='ShaderNodeUVMap')\ntex = nodes.new(type='ShaderNodeTexImage')\nlinks.new(uv.outputs['UV'], tex.inputs['Vector'])\n"),
("Add a Color Ramp between Noise Texture and Principled Base Color.",
 "import bpy\nmat = bpy.data.materials.new(name='RampMat')\nmat.use_nodes = True\nnodes = mat.node_tree.nodes\nlinks = mat.node_tree.links\nnoise = nodes.new(type='ShaderNodeTexNoise')\nramp = nodes.new(type='ShaderNodeValToRGB')\nbsdf = nodes.get('Principled BSDF')\nlinks.new(noise.outputs['Fac'], ramp.inputs['Fac'])\nif bsdf:\n    links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])\n"),
],
"addon_operator_panel": [
("Write a minimal add-on with an operator and a 3D View panel that shows a button.",
 "import bpy\n\nclass MY_OT_hello(bpy.types.Operator):\n    bl_idname = 'my.hello'\n    bl_label = 'Hello'\n    def execute(self, context):\n        self.report({'INFO'}, 'Hello')\n        return {'FINISHED'}\n\nclass MY_PT_panel(bpy.types.Panel):\n    bl_label = 'My Panel'\n    bl_idname = 'MY_PT_panel'\n    bl_space_type = 'VIEW_3D'\n    bl_region_type = 'UI'\n    bl_category = 'MyTab'\n    def draw(self, context):\n        self.layout.operator('my.hello')\n\nclasses = (MY_OT_hello, MY_PT_panel)\n\ndef register():\n    for c in classes:\n        bpy.utils.register_class(c)\n\ndef unregister():\n    for c in reversed(classes):\n        bpy.utils.unregister_class(c)\n\nif __name__ == '__main__':\n    register()\n"),
("Create an operator that adds a cube and a panel in the 3D View sidebar to run it.",
 "import bpy\n\nclass MY_OT_add_cube(bpy.types.Operator):\n    bl_idname = 'my.add_cube'\n    bl_label = 'Add Cube'\n    def execute(self, context):\n        bpy.ops.mesh.primitive_cube_add()\n        return {'FINISHED'}\n\nclass MY_PT_panel(bpy.types.Panel):\n    bl_label = 'Add Tools'\n    bl_idname = 'MY_PT_add_tools'\n    bl_space_type = 'VIEW_3D'\n    bl_region_type = 'UI'\n    bl_category = 'Create'\n    def draw(self, context):\n        self.layout.operator('my.add_cube')\n\nclasses = (MY_OT_add_cube, MY_PT_panel)\n\ndef register():\n    for c in classes:\n        bpy.utils.register_class(c)\n\ndef unregister():\n    for c in reversed(classes):\n        bpy.utils.unregister_class(c)\n\nif __name__ == '__main__':\n    register()\n"),
("Register a simple operator and panel that reports the active object name.",
 "import bpy\n\nclass MY_OT_report(bpy.types.Operator):\n    bl_idname = 'my.report_active'\n    bl_label = 'Report Active'\n    def execute(self, context):\n        name = context.active_object.name if context.active_object else 'None'\n        self.report({'INFO'}, name)\n        return {'FINISHED'}\n\nclass MY_PT_panel(bpy.types.Panel):\n    bl_label = 'Info'\n    bl_idname = 'MY_PT_info'\n    bl_space_type = 'VIEW_3D'\n    bl_region_type = 'UI'\n    bl_category = 'Info'\n    def draw(self, context):\n        self.layout.operator('my.report_active')\n\nclasses = (MY_OT_report, MY_PT_panel)\n\ndef register():\n    for c in classes:\n        bpy.utils.register_class(c)\n\ndef unregister():\n    for c in reversed(classes):\n        bpy.utils.unregister_class(c)\n\nif __name__ == '__main__':\n    register()\n"),
("Make an add-on panel with an operator that toggles viewport visibility of the active object.",
 "import bpy\n\nclass MY_OT_toggle(bpy.types.Operator):\n    bl_idname = 'my.toggle_hide'\n    bl_label = 'Toggle Hide'\n    def execute(self, context):\n        obj = context.active_object\n        if obj:\n            obj.hide_viewport = not obj.hide_viewport\n        return {'FINISHED'}\n\nclass MY_PT_panel(bpy.types.Panel):\n    bl_label = 'Visibility'\n    bl_idname = 'MY_PT_visibility'\n    bl_space_type = 'VIEW_3D'\n    bl_region_type = 'UI'\n    bl_category = 'View'\n    def draw(self, context):\n        self.layout.operator('my.toggle_hide')\n\nclasses = (MY_OT_toggle, MY_PT_panel)\n\ndef register():\n    for c in classes:\n        bpy.utils.register_class(c)\n\ndef unregister():\n    for c in reversed(classes):\n        bpy.utils.unregister_class(c)\n\nif __name__ == '__main__':\n    register()\n"),
("Provide a Blender add-on script defining an operator and a UI panel in the N-panel.",
 "import bpy\n\nclass MY_OT_select_meshes(bpy.types.Operator):\n    bl_idname = 'my.select_meshes'\n    bl_label = 'Select Meshes'\n    def execute(self, context):\n        bpy.ops.object.select_all(action='DESELECT')\n        for obj in context.scene.objects:\n            if obj.type == 'MESH':\n                obj.select_set(True)\n        return {'FINISHED'}\n\nclass MY_PT_panel(bpy.types.Panel):\n    bl_label = 'Select'\n    bl_idname = 'MY_PT_select'\n    bl_space_type = 'VIEW_3D'\n    bl_region_type = 'UI'\n    bl_category = 'Select'\n    def draw(self, context):\n        self.layout.operator('my.select_meshes')\n\nclasses = (MY_OT_select_meshes, MY_PT_panel)\n\ndef register():\n    for c in classes:\n        bpy.utils.register_class(c)\n\ndef unregister():\n    for c in reversed(classes):\n        bpy.utils.unregister_class(c)\n\nif __name__ == '__main__':\n    register()\n"),
],
}


B_OPENERS = [
    "",
    "You can do it like this:",
    "Here's one way:",
    "This script sets it up:",
    "Example:",
]


def gen_area_b(n=20, rng_seed=200, seen_hashes=None, execute=True):
    """Static topic-coverage pairs, verified the same way as LLM rows."""
    rng = random.Random(rng_seed)
    kept = []
    stats = Counter()
    pool = []
    for topic, items in B_TOPICS.items():
        for q, code in items:
            pool.append((topic, q, code))
    rng.shuffle(pool)
    attempts = 0
    for topic, question, code in pool:
        if len(kept) >= n:
            break
        attempts += 1
        opener = rng.choice(B_OPENERS)
        answer = make_answer(opener, code)
        ok, reason = validate_candidate(question, answer, "explanation",
                                        seen_hashes, execute)
        if not ok:
            stats[f"verify_reject:{reason}"] += 1
            continue
        kept.append({
            "question": question,
            "answer": answer,
            "type": "explanation",
            "seed_id": None,
            "source": "synthetic",
            "producing_model": "programmatic:targeted",
            "validated": True,
            "verified": True,
            "attempts": 0,
            "topic_area": topic,
            "timestamp": utc_now(),
        })
        stats[f"kept:{topic}"] += 1
    stats["candidates"] = attempts
    stats["kept_total"] = len(kept)
    return kept, stats


# -------------------- dedupe & overlap report --------------------
def near_dedupe(rows, ratio=0.9, keep=2):
    groups = []  # list of lists of indices
    used = set()
    for i, r in enumerate(rows):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in range(i + 1, len(rows)):
            if j in used:
                continue
            q1, q2 = rows[i]["question"], rows[j]["question"]
            if difflib.SequenceMatcher(None, q1, q2).ratio() >= ratio:
                group.append(j)
                used.add(j)
        groups.append(group)
    kept_idx = []
    removed = 0
    for g in groups:
        kept_idx.extend(g[:keep])
        removed += max(0, len(g) - keep)
    kept_idx = sorted(kept_idx)
    return [rows[i] for i in kept_idx], removed, sum(1 for g in groups if len(g) > 1)



def decontam_rows(rows):
    from pipeline.combine_data import drop_benchmark_overlaps
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        kept, dropped, _rules = drop_benchmark_overlaps(rows, path=path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return kept, dropped


def run_targeted(output_file, batch, seen_hashes, execute, remaining):
    """Generate targeted rows and append them to the single output file."""
    if batch <= 0 or remaining <= 0:
        return 0
    if not execute:
        print("Skipping targeted rows: they need execution validation.")
        return 0
    print(f"Generating targeted rows (batch {batch} per area) ...", flush=True)
    a_rows, a_stats = gen_area_a(batch, seen_hashes=seen_hashes, execute=execute)
    b_rows, b_stats = gen_area_b(batch, seen_hashes=seen_hashes, execute=execute)
    rows, removed, _groups = near_dedupe(a_rows + b_rows, ratio=0.9, keep=2)
    if removed:
        print(f"Near-duplicate dedupe: removed {removed}")
    rows, dropped = decontam_rows(rows)
    rows = rows[:remaining]
    with open(output_file, "a", encoding="utf-8") as out_f:
        for row in rows:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Added {len(rows)} targeted rows "
          f"(A={a_stats['kept_total']}, B={b_stats['kept_total']}, "
          f"dropped {len(dropped)} benchmark overlaps).")
    return len(rows)


if __name__ == "__main__":
    main()
