#!/usr/bin/env python3
"""Execute candidate Blender Python code in a fresh headless Blender process.

Two modes:
- run_code(code): single code block.
- run_batch(codes): many code blocks in one Blender process, scene reset between.
- run_with_check(code, check): code followed by a hidden assertion snippet.

Error policy:
- AttributeError / TypeError / ValueError (invalid enum) -> fail.
- RuntimeError caused by operator poll / missing context -> non-fatal (ok=True,
  nonfatal=True) but still subject to the coverage floor.
"""
import ast
import json
import os
import subprocess
import tempfile
from typing import Dict, List, Tuple

import config

BLENDER_BIN = config.BLENDER_BIN
RUN_TIMEOUT = int(os.environ.get("BLENDER_EXEC_TIMEOUT", "120"))
RESULT_START = "###EXEC_RESULT###"
COVERAGE_FLOOR = 0.9


def _classify_error(ok: bool, msg: str, error_type: str) -> Dict:
    if ok:
        return {"ok": True, "nonfatal": False, "error": msg, "error_type": error_type}
    lower = (msg or "").lower()
    nonfatal = ("poll() failed" in lower or
                "context is incorrect" in lower or
                "context missing" in lower or
                "operator cannot run" in lower or
                "incorrect context" in lower or
                error_type == "ContextError")
    return {"ok": nonfatal, "nonfatal": nonfatal,
            "error": msg, "error_type": error_type}


def _statement_lines(src: str) -> set:
    """Line numbers of executable statements, excluding imports and class bodies."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    lines = set()

    def walk(node):
        if isinstance(node, ast.ClassDef):
            # Operator/panel methods cannot run headless; only the class line counts.
            lines.add(node.lineno)
            return
        if isinstance(node, ast.stmt):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return
            lines.add(node.lineno)
        for field in ("body", "orelse", "finalbody"):
            for child in getattr(node, field, []) or []:
                walk(child)
        for handler in getattr(node, "handlers", []) or []:
            walk(handler)
        for case in getattr(node, "cases", []) or []:
            for child in getattr(case, "body", []) or []:
                walk(child)

    for stmt in tree.body:
        walk(stmt)
    return lines


def is_class_register(code: str) -> bool:
    """True for a pure addon answer: class definitions plus register() under main."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    if not any(isinstance(n, ast.ClassDef) for n in tree.body):
        return False
    for node in tree.body:
        if not isinstance(node, ast.If) or "__main__" not in ast.unparse(node.test):
            continue
        for call in ast.walk(node):
            if isinstance(call, ast.Call):
                name = ast.unparse(call.func)
                if name == "register" or name.endswith(".register"):
                    return True
    return False


RUNNER_HELPERS = r"""
import ast
import bpy
import json
import sys
import traceback

COVERAGE_FILENAME = "<job>"

def _statement_lines(src):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    lines = set()
    def walk(node):
        if isinstance(node, ast.ClassDef):
            lines.add(node.lineno)
            return
        if isinstance(node, ast.stmt):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return
            lines.add(node.lineno)
        for field in ("body", "orelse", "finalbody"):
            for child in getattr(node, field, []) or []:
                walk(child)
        for handler in getattr(node, "handlers", []) or []:
            walk(handler)
        for case in getattr(node, "cases", []) or []:
            for child in getattr(case, "body", []) or []:
                walk(child)
    for stmt in tree.body:
        walk(stmt)
    return lines

def _reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if not bpy.data.scenes:
        bpy.data.scenes.new("Scene")
    # Most snippets assume an active mesh; give them one.
    bpy.ops.mesh.primitive_cube_add()

def _run_job(job):
    src = job.get("code", "")
    check = job.get("check") or ""
    res = {"id": job.get("id"), "ok": False, "code_ok": False,
           "error": None, "error_type": None, "coverage": 0.0,
           "check_ok": None, "check_error": None, "stage": None}
    wanted = _statement_lines(src)
    hit = set()

    def tracer(frame, event, arg):
        if event == "line" and frame.f_code.co_filename == COVERAGE_FILENAME:
            hit.add(frame.f_lineno)
        return tracer

    ns = {"bpy": bpy, "__name__": "__main__"}
    old = sys.gettrace()
    try:
        sys.settrace(tracer)
        try:
            exec(compile(src, COVERAGE_FILENAME, "exec"), ns)
            res["code_ok"] = True
            res["ok"] = True
        finally:
            sys.settrace(old)
        if check:
            try:
                sys.settrace(tracer)
                exec(compile(check, "<check>", "exec"), ns)
                res["check_ok"] = True
            except Exception as e:
                res["check_ok"] = False
                res["check_error"] = str(e)
                res["error"] = str(e)
                res["error_type"] = type(e).__name__
                res["stage"] = "check"
                res["ok"] = False
            finally:
                sys.settrace(old)
    except Exception as e:
        res["error"] = str(e)
        res["error_type"] = type(e).__name__
        res["traceback"] = traceback.format_exc()
        res["stage"] = "code"
    finally:
        sys.settrace(old)
    res["coverage"] = (len(wanted & hit) / len(wanted)) if wanted else 1.0
    return res
"""

BATCH_RUNNER = RUNNER_HELPERS + r"""
in_path = sys.argv[-2]
out_path = sys.argv[-1]
with open(in_path, "r") as f:
    jobs = json.load(f)

results = []
for job in jobs:
    _reset()
    results.append(_run_job(job))
    # Flush after every job so a crash/hang keeps the results already earned.
    with open(out_path, "w") as f:
        json.dump(results, f)

bpy.ops.wm.quit_blender()
"""


def _with_coverage(raw: Dict) -> Dict:
    out = _classify_error(raw.get("ok"), raw.get("error"), raw.get("error_type"))
    out["coverage"] = float(raw.get("coverage") or 0.0)
    out["code_ok"] = bool(raw.get("code_ok"))
    out["check_ok"] = raw.get("check_ok")
    out["check_error"] = raw.get("check_error")
    out["stage"] = raw.get("stage")
    return out


def _merge_partial(out_path: str, jobs: List[Dict],
                   missing_error: str, missing_type: str) -> List[Dict]:
    """Keep every result the crashed/hung batch already flushed to disk."""
    try:
        with open(out_path, "r") as f:
            partial = json.load(f)
    except (OSError, ValueError):
        partial = []
    by_id = {r.get("id"): r for r in partial}
    return [by_id.get(j["id"], {"id": j["id"], "ok": False,
                                "error": missing_error, "error_type": missing_type})
            for j in jobs]


def run_batch(codes: List[str], timeout: int = RUN_TIMEOUT,
              checks: List[str] = None) -> List[Dict]:
    """Run many code blocks in a single headless Blender process."""
    jobs = []
    for i, code in enumerate(codes):
        if not code.strip():
            continue
        jobs.append({"id": i, "code": code,
                     "check": checks[i] if checks and i < len(checks) else ""})
    if not jobs:
        return []

    in_fd, in_path = tempfile.mkstemp(suffix=".json")
    out_path = in_path + ".out"
    with os.fdopen(in_fd, "w") as f:
        json.dump(jobs, f)

    runner_fd, runner_path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(runner_fd, "w") as f:
        f.write(BATCH_RUNNER)

    try:
        proc = subprocess.run(
            [BLENDER_BIN, "-b", "--python", runner_path, "--", in_path, out_path],
            capture_output=True, text=True, timeout=timeout,
        )
        results = _merge_partial(out_path, jobs, "no result (batch crashed/hung)",
                                 "ExecutionError")
    except subprocess.TimeoutExpired:
        results = _merge_partial(out_path, jobs,
                                 f"batch timeout after {timeout}s", "Timeout")
    except Exception as e:
        results = _merge_partial(out_path, jobs,
                                 f"batch runner failed: {e}", type(e).__name__)
    finally:
        for p in (in_path, out_path, runner_path):
            try:
                os.unlink(p)
            except OSError:
                pass

    results = sorted(results, key=lambda x: x["id"])
    return [_with_coverage(r) for r in results]


def run_code(code: str) -> Dict:
    if not code.strip():
        return {"ok": False, "nonfatal": False, "error": "empty code",
                "error_type": "ValueError", "coverage": 0.0, "code_ok": False,
                "check_ok": None, "check_error": None, "stage": "code"}
    return run_batch([code])[0]


def run_with_check(code: str, check: str) -> Dict:
    """Run code, then check in the same namespace and fresh scene."""
    if not code.strip():
        return run_code(code)
    return run_batch([code], checks=[check])[0]


def vacuous_checks(checks: List[str]) -> List[bool]:
    """True where a check also passes with no answer code (it tests nothing).

    All stubs run in one Blender process; checks that call answer-defined names
    fail with NameError and are therefore not vacuous.
    """
    idx = [i for i, check in enumerate(checks) if check and check.strip()]
    out = [False] * len(checks)
    if not idx:
        return out
    results = run_batch(["import bpy"] * len(idx),
                        checks=[checks[i] for i in idx])
    for i, res in zip(idx, results):
        out[i] = res.get("check_ok") is True
    return out


def is_vacuous_check(check: str) -> bool:
    return bool(check and check.strip()) and vacuous_checks([check])[0]


def validate(code: str, check: str = "") -> Tuple[bool, str]:
    res = run_with_check(code, check) if check else run_code(code)
    if res["ok"]:
        if not is_class_register(code) and res.get("coverage", 0.0) < COVERAGE_FLOOR:
            return False, f"low_coverage:{int(round(res['coverage'] * 100))}"
        if res.get("nonfatal"):
            return True, f"ok_nonfatal:{res.get('error_type', 'RuntimeError')}"
        return True, "ok"
    if res.get("stage") == "check":
        msg = (res.get("check_error") or res.get("error") or "")[:80].replace("\n", " ")
        return False, f"check_failed:{msg}"
    et = res.get("error_type", "ExecutionError")
    msg = (res.get("error") or "")[:80].replace("\n", " ")
    return False, f"exec_{et.lower()}:{msg}"


if __name__ == "__main__":
    import sys
    code = sys.argv[1] if len(sys.argv) > 1 else "import bpy\nprint('hello')"
    print(json.dumps(run_code(code), indent=2))
