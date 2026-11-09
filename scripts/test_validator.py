"""Smoke tests for the v2 static validator."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validator


def ok(result):
    return result[0]


# 1. cache loads
validator.load_cache()

# 2. good operator sample passes
q = "How do I add a cube?"
a = "Use primitive_cube_add.\n\n```python\nimport bpy\nbpy.ops.mesh.primitive_cube_add(size=2, align='CURSOR')\n```"
assert ok(validator.validate_static(q, a, "explanation")), "good sample rejected"

# 3. invented API rejected
bad = "```python\nimport bpy\nbpy.ops.fluid.bake_noize()\n```"
assert not ok(validator.validate_static("q", bad, "explanation"))

# 4. deprecated API rejected (use_auto_smooth)
dep = "```python\nimport bpy\nbpy.context.object.data.use_auto_smooth = True\n```"
r = validator.validate_static("q", dep, "explanation")
assert not ok(r) and "deprecated" in r[1], r

# 5. reference signature listing accepted
ref = "```python\nbpy.ops.constraint.childof_clear_inverse(*, constraint='', owner='OBJECT')\n```"
r = validator.validate_static("q", "a\n" + ref, "explanation", strict_format=False)
assert ok(r) and r[1] == "ok_reference", r

# 6. chain-of-thought rejected
cot = "Wait, let's look closer.\n\n```python\nimport bpy\n```"
assert not ok(validator.validate_static("q", cot, "explanation"))

# 7. non-string rejected
assert not ok(validator.validate_static("q", ["list"], "explanation"))

# 8. backward-compatible validate_sample still works
coder = "```python\nimport bpy\nbpy.ops.mesh.primitive_cube_add(size=2, align='CURSOR')\n```"
assert ok(validator.validate_sample("q", coder, 7, strict_format=True))

# 9. fake bugfix (answer == broken input) rejected
broken = "```python\nimport bpy\nobj.location.y = obj.location.y\n```"
q = "Fix the bug in this Blender addon:\n" + broken
assert not ok(validator.validate_static(q, broken, "bugfix"))

# 9b. same check applies to explained bugfixes (type 5)
assert not ok(validator.validate_static(q, "Explanation.\n\n" + broken,
                                        "bugfix_explained"))

# 10. truncated / unbalanced fence rejected
assert not ok(validator.validate_static("q", "```python\nimport bpy\n", "bugfix"))
assert not ok(validator.validate_static("q", "```python\n\n```", "bugfix"))

# 11. real FIM is rebuilt from prefix/suffix before validation
fim_q = ("<|fim_prefix|>import bpy\n<|fim_suffix|>obj.location.x = 1\n<|fim_middle|>")
fim_a = "```python\nobj = bpy.context.object\n```"
assert ok(validator.validate_static(fim_q, fim_a, "fim")), \
    validator.validate_static(fim_q, fim_a, "fim")
assert not ok(validator.validate_static("<|fim_prefix|>import bpy\n", fim_a, "fim"))

# 12. FIM full-code helper rebuilds the snippet for execution
assert validator.fim_full_code(fim_q, fim_a) == (
    "import bpy\nobj = bpy.context.object\nobj.location.x = 1")
assert validator.fim_full_code("<|fim_prefix|>import bpy\n", fim_a) is None

# 13. legacy two-section headings are stripped when merging old rows
legacy = ("1) Architecture Overview\n\nProse.\n\n2) The Code\n"
          "```python\nimport bpy\n```")
assert validator.normalize_answer(legacy) == (
    "Prose.\n\n```python\nimport bpy\n```")

# 14. one taxonomy: generator ints and task names normalize the same way
import config
assert config.task_type(4) == "explanation"
assert config.task_type(7) == "bugfix"
assert config.task_type("fim") == "fim"
assert config.task_type(99) == "explanation"

# 15. type-4 pipeline parses the big model's ===ANSWER=== wrapper
import model_correction
assert model_correction.parse_answer("noise\n===ANSWER===\nfinal") == "final"
assert model_correction.parse_answer("```python\nx = 1\n```") == "```python\nx = 1\n```"

# 16. questions are saved first; answering fills the same row instead of appending
import tempfile
from pathlib import Path
with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "corrections.jsonl"
    model_correction.append_jsonl(path, {"question": "q1"})
    model_correction.save_row(path, "q1", {"question": "q1", "answer": "a1"})
    rows = model_correction.read_jsonl(path)
    assert len(rows) == 1 and rows[0]["answer"] == "a1", rows
    assert not [r for r in rows if not r.get("answer")]

# 17. execution validation needs real Blender: coverage, hidden checks, class +
#     register exemption, and combine_data dropping "unverified" rows.
import json
import shutil
import combine_data
import execution_validator

CLASS_REGISTER = (
    "import bpy\n"
    "class MY_OT_thing(bpy.types.Operator):\n"
    "    bl_idname = 'object.thing'\n"
    "    bl_label = 'Thing'\n"
    "    def execute(self, context):\n"
    "        return {'FINISHED'}\n"
    "def register():\n"
    "    bpy.utils.register_class(MY_OT_thing)\n"
    "if __name__ == '__main__':\n"
    "    register()\n"
)
UNCALLED = "import bpy\ndef foo():\n    return 1\n"
GUARDED = "import bpy\nobj = None\nif obj:\n    obj.name = 'x'\n"
CALLED = "import bpy\ndef foo():\n    return 1\nfoo()\n"
ATTR_ERROR = "import bpy\nbpy.context.object.does_not_exist = 1\n"
EMPTY = "import bpy\n"


def check_execution():
    codes = [UNCALLED, GUARDED, CALLED, ATTR_ERROR, CLASS_REGISTER, EMPTY, EMPTY]
    checks = ["", "", "", "", "", "assert len(bpy.data.objects) == 1",
              "assert len(bpy.data.objects) == 0"]
    res = execution_validator.run_batch(codes, checks=checks)
    assert len(res) == len(codes), res

    assert res[0]["code_ok"] and res[0]["coverage"] < 0.8, res[0]
    assert res[1]["code_ok"] and res[1]["coverage"] < 0.8, res[1]
    assert res[2]["code_ok"] and res[2]["coverage"] == 1.0, res[2]

    assert not res[3]["ok"] and res[3]["error_type"] == "AttributeError", res[3]

    assert res[4]["code_ok"] and execution_validator.is_class_register(CLASS_REGISTER), res[4]

    assert res[5]["ok"] and res[5]["check_ok"] is True, res[5]
    assert not res[6]["ok"] and res[6]["check_ok"] is False, res[6]
    assert res[6]["stage"] == "check", res[6]

    # validate() turns sub-floor coverage into a failure...
    ok, reason = execution_validator.validate(UNCALLED)
    assert not ok and reason.startswith("low_coverage:"), (ok, reason)
    # ...but exempts a pure class + register() answer.
    ok, reason = execution_validator.validate(CLASS_REGISTER)
    assert ok, (ok, reason)

    good = "```python\nimport bpy\nbpy.ops.mesh.primitive_cube_add(size=2, align='CURSOR')\n```"
    rows = [
        {"question": "How do I add a cube?", "answer": good, "validated": True},
        {"question": "How do I add a cube twice?", "answer": good,
         "validated": "unverified"},
        # legacy row without the key must still pass
        {"question": "How do I add a cube three times?", "answer": good},
    ]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "rows.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        kept = combine_data.load_clean("model_corrections", str(path))
    questions = [r["question"] for r in kept]
    assert "How do I add a cube?" in questions, questions
    assert "How do I add a cube three times?" in questions, questions
    assert "How do I add a cube twice?" not in questions, questions


if shutil.which(config.BLENDER_BIN) is None:
    print(f"SKIPPED execution checks: Blender not found ({config.BLENDER_BIN})")
else:
    check_execution()
    print("EXECUTION CHECKS PASSED")

print("ALL CHECKS PASSED")
