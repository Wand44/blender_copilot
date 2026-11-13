#!/usr/bin/env python3
"""Benchmark a local Ollama model on bench/cases.py.

    python3 bench/run_bench.py --validate             # sanity-check every reference
    python3 bench/run_bench.py --samples 5            # benchmark qwen3.5:4b

Each sample runs in a fresh headless Blender, so one bad script cannot affect
the next one.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from math import comb
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cases import CASES  # noqa: E402

BLENDER = os.environ.get("BLENDER_BIN", "blender")
RUNNER = HERE / "blender_runner.py"
OLLAMA = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:4b"
RUN_TIMEOUT = 60
FENCE = re.compile(r"```python[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(response):
    match = FENCE.search(response)
    return match.group(1) if match else response


def ask_ollama(prompt, temperature):
    try:
        resp = requests.post(
            OLLAMA,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=600,
        )
    except requests.RequestException as e:
        raise SystemExit(f"ollama request failed ({OLLAMA}): {e}")
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def run_blender(setup, code, check):
    cmd = [BLENDER, "-b", "--factory-startup", "--python", str(RUNNER), "--",
           json.dumps({"setup": setup, "code": code, "check": check})]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        return "timeout", f"exceeded {RUN_TIMEOUT}s"

    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                out = json.loads(line)
            except ValueError:
                continue
            if "status" in out:
                return out["status"], out.get("error")

    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
    return "crashed", f"blender exit {proc.returncode}: " + " | ".join(tail)


def run_many(payloads, jobs):
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(lambda p: run_blender(**p), payloads))


def pass_at_1(samples):
    return sum(s["status"] == "passed" for s in samples) / len(samples)


def pass_at_k(samples, k):
    """Unbiased pass@k over n iid samples per case, averaged over cases."""
    if not samples:
        return 0.0
    n = len(samples)
    c = sum(s["status"] == "passed" for s in samples)
    k = min(k, n)
    return 1.0 - comb(n - c, k) / comb(n, k)


def validate(jobs):
    payloads = [{"setup": c["setup"], "code": c["reference"], "check": c["check"]}
                for c in CASES]
    outcomes = run_many(payloads, jobs)
    failures = 0
    print(f"{'case':28} {'tier':8} status")
    print("-" * 55)
    for case, (status, error) in zip(CASES, outcomes):
        print(f"{case['id']:28} {case['tier']:8} {status}")
        if status != "passed":
            failures += 1
            print(f"    {error}")
    print("-" * 55)
    if failures:
        print(f"{failures}/{len(CASES)} reference(s) FAILED")
        return 1
    print(f"all {len(CASES)} references passed")
    return 0


def benchmark(samples_per_case, temperature, jobs):
    samples = []
    for case in CASES:
        print(f"querying {case['id']} x{samples_per_case} ...", flush=True)
        for _ in range(samples_per_case):
            response = ask_ollama(case["prompt"], temperature)
            samples.append({
                "id": case["id"],
                "tier": case["tier"],
                "prompt": case["prompt"],
                "response": response,
                "code": extract_code(response),
            })

    print(f"running {len(samples)} blender processes ...", flush=True)
    case_by_id = {c["id"]: c for c in CASES}
    outcomes = run_many(
        [{"setup": case_by_id[s["id"]]["setup"], "code": s["code"],
          "check": case_by_id[s["id"]]["check"]} for s in samples],
        jobs,
    )
    for sample, (status, error) in zip(samples, outcomes):
        sample["status"] = status
        sample["error"] = error

    summary = summarize(samples, samples_per_case)
    path = save(samples, summary, samples_per_case, temperature)
    print_summary(summary)
    print(f"\nwrote {path}")
    return 0


def summarize(samples, samples_per_case):
    tiers = {}
    for tier in dict.fromkeys(c["tier"] for c in CASES):
        tiers[tier] = []
    for case in CASES:
        own = [s for s in samples if s["id"] == case["id"]]
        tiers[case["tier"]].extend(own)
    counts = {s: sum(x["status"] == s for x in samples)
              for s in ("passed", "wrong", "crashed", "timeout")}
    k = samples_per_case
    return {
        "samples_per_case": samples_per_case,
        "pass_at_1": pass_at_1(samples),
        "pass_at_k": k,
        "pass_at_k_value": sum(pass_at_k([s for s in samples if s["id"] == c["id"]], k)
                               for c in CASES) / len(CASES),
        "counts": counts,
        "tiers": {t: {"n": len(v), "passed": sum(s["status"] == "passed" for s in v),
                      "rate": pass_at_1(v) if v else 0.0}
                  for t, v in tiers.items()},
    }


def save(samples, summary, samples_per_case, temperature):
    out_dir = HERE.parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{stamp}_{MODEL.replace(':', '-').replace('/', '-')}.json"
    path.write_text(json.dumps({
        "model": MODEL,
        "timestamp": stamp,
        "samples_per_case": samples_per_case,
        "temperature": temperature,
        "summary": summary,
        "samples": samples,
    }, indent=2))
    return path


def print_summary(summary):
    print(f"\n{'tier':10} {'samples':>8} {'passed':>7} {'rate':>7}")
    print("-" * 36)
    for tier, row in summary["tiers"].items():
        print(f"{tier:10} {row['n']:>8} {row['passed']:>7} {row['rate']:>7.2%}")
    print("-" * 36)
    c = summary["counts"]
    total = sum(c.values())
    print(f"overall    {total:>8} {c['passed']:>7} {summary['pass_at_1']:>7.2%}")
    print(f"passed {c['passed']}  wrong {c['wrong']}  "
          f"crashed {c['crashed']}  timeout {c['timeout']}")
    print(f"pass@1 {summary['pass_at_1']:.2%}   "
          f"pass@{summary['pass_at_k']} {summary['pass_at_k_value']:.2%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5,
                        help="samples per case (default 5)")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="sampling temperature (default 0.3)")
    parser.add_argument("--validate", action="store_true",
                        help="run every case's reference instead of the model")
    parser.add_argument("--jobs", type=int, default=4,
                        help="parallel Blender processes (default 4)")
    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if args.validate:
        return validate(args.jobs)
    return benchmark(args.samples, args.temperature, args.jobs)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
