#!/usr/bin/env python3
"""Execute candidate Blender Python code in a fresh headless Blender process.

Two modes:
- run_code(code): single code block.
- run_batch(codes): many code blocks in one Blender process, scene reset between.

Error policy:
- AttributeError / TypeError / ValueError (invalid enum) -> fail.
- RuntimeError caused by operator poll / missing context -> non-fatal (ok=True).
"""
import json
import os
import re
import subprocess
import tempfile
import textwrap
from typing import Dict, List, Tuple

import blender_config

BLENDER_BIN = blender_config.BLENDER_BIN
RUN_TIMEOUT = int(os.environ.get("BLENDER_EXEC_TIMEOUT", "120"))
RESULT_START = "###EXEC_RESULT###"


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


def run_code(code: str) -> Dict:
    if not code.strip():
        return {"ok": False, "error": "empty code", "error_type": "ValueError"}

    script = textwrap.dedent(f"""
        import bpy
        import json
        import sys
        import traceback

        def _setup():
            bpy.ops.wm.read_factory_settings(use_empty=True)
            if not bpy.data.scenes:
                bpy.data.scenes.new("Scene")
            # Most snippets assume an active mesh; give them one.
            bpy.ops.mesh.primitive_cube_add()

        result = {{"ok": False, "error": None, "error_type": None}}
        try:
            _setup()
            exec({json.dumps(code)}, {{"bpy": bpy, "__name__": "__main__"}})
            result["ok"] = True
        except Exception as e:
            result["error"] = str(e)
            result["error_type"] = type(e).__name__
            result["traceback"] = traceback.format_exc()

        print({json.dumps(RESULT_START)} + json.dumps(result))
        bpy.ops.wm.quit_blender()
    """)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(script)
        proc = subprocess.run(
            [BLENDER_BIN, "-b", "--python", tmp_path],
            capture_output=True, text=True, timeout=RUN_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {RUN_TIMEOUT}s", "error_type": "Timeout"}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    out = proc.stdout or ""
    err = proc.stderr or ""
    for line in out.splitlines():
        if line.startswith(RESULT_START):
            try:
                payload = json.loads(line[len(RESULT_START):])
                return _classify_error(payload.get("ok"), payload.get("error"), payload.get("error_type"))
            except json.JSONDecodeError:
                break
    error = err.strip() or out.strip() or "unknown execution failure"
    return _classify_error(False, error[:500], "ExecutionError")


BATCH_RUNNER = r"""
import bpy
import json
import sys
import traceback

def _reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if not bpy.data.scenes:
        bpy.data.scenes.new("Scene")
    # Most snippets assume an active mesh; give them one.
    bpy.ops.mesh.primitive_cube_add()

in_path = sys.argv[-2]
out_path = sys.argv[-1]
with open(in_path, "r") as f:
    jobs = json.load(f)

results = []
for job in jobs:
    _reset()
    res = {"id": job["id"], "ok": False, "error": None, "error_type": None}
    try:
        exec(job["code"], {"bpy": bpy, "__name__": "__main__"})
        res["ok"] = True
    except Exception as e:
        res["error"] = str(e)
        res["error_type"] = type(e).__name__
        res["traceback"] = traceback.format_exc()
    results.append(res)
    # Flush after every job so a crash/hang keeps the results already earned.
    with open(out_path, "w") as f:
        json.dump(results, f)

bpy.ops.wm.quit_blender()
"""


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


def run_batch(codes: List[str], timeout: int = RUN_TIMEOUT) -> List[Dict]:
    """Run many code blocks in a single headless Blender process."""
    jobs = [{"id": i, "code": c} for i, c in enumerate(codes) if c.strip()]
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
    return [_classify_error(r.get("ok"), r.get("error"), r.get("error_type")) for r in results]


def validate(code: str) -> Tuple[bool, str]:
    res = run_code(code)
    if res["ok"]:
        return True, "ok"
    et = res.get("error_type", "ExecutionError")
    msg = res.get("error", "")[:80].replace("\n", " ")
    return False, f"exec_{et.lower()}:{msg}"


if __name__ == "__main__":
    import sys
    code = sys.argv[1] if len(sys.argv) > 1 else "import bpy\nprint('hello')"
    print(json.dumps(run_code(code), indent=2))
