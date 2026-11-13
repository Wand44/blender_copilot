"""Run one benchmark sample in a fresh Blender process.

Usage: blender -b --factory-startup --python bench/blender_runner.py -- '<json>'
Payload keys: setup, code, check. Prints one JSON line: {"status", "error"}.
"""

import json
import sys
import traceback

import bpy


def main():
    payload = json.loads(sys.argv[sys.argv.index("--") + 1])
    result = {"status": "crashed", "error": None}
    # One module-like namespace so nested functions see exec'd names.
    ns = {"__name__": "__main__", "__builtins__": __builtins__, "bpy": bpy}

    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        exec(payload["setup"], ns)
    except Exception:
        result["error"] = "setup: " + traceback.format_exc()
        return result

    try:
        exec(payload["code"], ns)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    try:
        exec(payload["check"], ns)
    except Exception as e:
        result["status"] = "wrong"
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    result["status"] = "passed"
    return result


if __name__ == "__main__":
    try:
        out = main()
    except Exception:
        out = {"status": "crashed", "error": "runner: " + traceback.format_exc()}
    print(json.dumps(out))
    sys.stdout.flush()
