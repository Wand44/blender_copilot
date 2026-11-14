"""Smoke tests for the shared validator, row schema, and pipeline steps."""
import json
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import config, execution_validator, validator  # noqa: E402
from core import rows as rows_mod  # noqa: E402
from core import verify as verify_mod  # noqa: E402
from pipeline import combine_data, generate_synthetic as gs, model_correction  # noqa: E402


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

# 6b. new COT markers rejected (case-insensitive), e.g. "Actually,"
cot2 = "Actually, let me just do it.\n\n```python\nimport bpy\n```"
assert not ok(validator.validate_static(q, cot2, "explanation"))

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
assert config.task_type(4) == "explanation"
assert config.task_type(7) == "bugfix"
assert config.task_type("fim") == "fim"
assert config.task_type(99) == "explanation"

# 15. type-4 pipeline parses the big model's ===ANSWER=== wrapper
assert model_correction.parse_answer("noise\n===ANSWER===\nfinal") == "final"
assert model_correction.parse_answer("```python\nx = 1\n```") == "```python\nx = 1\n```"

# 15b. provider content parts are flattened, not .strip()'d (was a crash)
assert gs._content_text([{"type": "text", "text": "a"},
                         {"type": "text", "text": " b "}]) == "a b"
assert gs._content_text(" x ") == "x" and gs._content_text(None) == ""

# 15c. AST API areas + categories + training-row metadata
area_code = (
    "```python\nimport bpy, bmesh\n"
    "bpy.ops.mesh.primitive_cube_add()\n"
    "bpy.context.scene.cycles.samples = 64\n"
    "mat = bpy.data.materials.new('M')\n"
    "mat.node_tree.nodes.new('ShaderNodeTexNoise')\n"
    "bm = bmesh.new()\n```"
)
areas = rows_mod.api_areas("q", area_code)
assert "bpy.ops.mesh" in areas, areas
assert "scene.cycles" in areas, areas
assert "bpy.data.materials" in areas, areas
assert "ShaderNodeTexNoise" in areas, areas
assert "bmesh" in areas, areas
categories = rows_mod.api_categories(areas, "q", area_code)
assert {"mesh", "bmesh", "materials/nodes", "render/cycles"}.issubset(set(categories)), categories
row = rows_mod.training_row("q", area_code, id=7, source="synthetic",
                            producing_model="test-model", verified=True, attempts=2)
for field in ("id", "source", "producing_model", "verified", "attempts",
              "api_areas", "content_hash", "token_length", "timestamp"):
    assert field in row, field
assert row["id"] == 7 and row["source"] == "synthetic" and row["token_length"] > 0

# 16. questions are saved first; answering fills the same row instead of appending
with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "corrections.jsonl"
    model_correction.append_jsonl(path, {"id": 1, "question": "q1"})
    model_correction.save_row(path, "q1", {"id": 1, "question": "q1", "answer": "a1"})
    rows = model_correction.read_jsonl(path)
    assert len(rows) == 1 and rows[0]["answer"] == "a1", rows
    assert not [row for row in rows if not row.get("answer")]

# 16b. concurrent workers don't lose rows (the file lock)
with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "concurrent.jsonl"
    for i in range(20):
        model_correction.append_jsonl(path, {"id": i, "question": f"q{i}"})
    threads = [threading.Thread(target=model_correction.save_row,
                                args=(path, f"q{i}",
                                      {"id": i, "question": f"q{i}", "answer": f"a{i}"}))
               for i in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    rows = model_correction.read_jsonl(path)
    assert len(rows) == 20, len(rows)
    assert all(row["answer"] == "a" + row["question"][1:] for row in rows), rows

# 16c. benchmark decontamination: exact copy, reworded copy, copied reference
# code are dropped; an unrelated row is kept.
with tempfile.TemporaryDirectory() as d:
    case = combine_data.CASES[0]
    exact = {"id": "docs:1", "source": "docs", "question": case["prompt"],
             "answer": "x"}
    reworded = {"id": "docs:2", "source": "docs",
                "question": "Make a Blender script that when run creates a cube "
                            "in the center of the scene",
                "answer": "x"}
    code_copy = {"id": "docs:3", "source": "docs",
                 "question": "How do I make a material blue?",
                 "answer": case["reference"]}
    unrelated = {"id": "docs:4", "source": "docs",
                 "question": "How do I make a material blue?", "answer": "x"}
    kept, dropped, rules = combine_data.drop_benchmark_overlaps(
        [exact, reworded, code_copy, unrelated], str(Path(d) / "dropped.jsonl"))
    assert kept == [unrelated], kept
    assert rules["question_exact"] == 1, rules
    assert rules["question_similar"] == 1, rules
    assert rules["code_exact"] == 1, rules
    written = [json.loads(line) for line in
               (Path(d) / "dropped.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["match_rule"] for row in written} == {
        "question_exact", "question_similar", "code_exact"}, written
    assert all(row["match_case"] == case["id"] for row in written), written

# 16d. correction is sequential, keeps ids, and records each attempt model
with tempfile.TemporaryDirectory() as d:
    out = Path(d) / "corrections.jsonl"
    model_correction.OUTPUT = out
    replies = iter([
        ("bad one\n===ANSWER===\n```python\nimport bpy\n```", "model-a"),
        ("bad two\n===ANSWER===\n```python\nimport bpy\n```", "model-b"),
        ("good\n===ANSWER===\n```python\nimport bpy\nbpy.ops.mesh.primitive_cube_add()\n```",
         "model-c"),
    ])
    calls = []
    model_correction.big.call_llm_ex = lambda *args, **kwargs: (calls.append(1) or next(replies))

    def fake_evaluate(question, answer, check="", require_check=False):
        # Judge by content so the initial small-answer check doesn't shift the sequence.
        if "primitive_cube_add" in answer:
            return True, "ok", model_correction._empty_info()
        return False, "no cube", model_correction._empty_info()

    model_correction.evaluate = fake_evaluate
    try:
        item = {"id": 137, "question": "q", "reference_check": "assert True",
                "reference": "", "attempts": 0, "error_history": [], "small_answer": "bad"}
        row, key = model_correction.process(item, small_answer="bad")
        assert key == "corrected" and row["id"] == 137, row
        assert row["attempts"] == 3, row["attempts"]  # third correction passed
        assert row["producing_model"] == "model-c", row
        assert [entry["model"] for entry in row["error_history"]] == ["model-a", "model-b", "model-c"]
        assert len(calls) == 3
    finally:
        model_correction.evaluate = verify_mod.evaluate

# 16e. failed attempts keep the last candidate and are not retried when exhausted
with tempfile.TemporaryDirectory() as d:
    out = Path(d) / "corrections.jsonl"
    model_correction.OUTPUT = out
    model_correction.big.call_llm_ex = lambda *args, **kwargs: (
        "===ANSWER===\n```python\nimport bpy\n# still bad\n```", "model-x")
    model_correction.evaluate = lambda question, answer, check="", require_check=False: (
        False, "boom", model_correction._empty_info())
    try:
        item = {"id": 9, "question": "q", "reference_check": "", "attempts": 0}
        row, key = model_correction.process(item, small_answer="bad", max_attempts=2)
        assert key == "failed" and row["attempts"] == 2, row
        assert "# still bad" in row["last_candidate"], row["last_candidate"]
        assert [entry["error"] for entry in row["error_history"]] == ["boom", "boom"]
        assert model_correction._select_pending([{**row, "attempts": 1}], 2)
        assert not model_correction._select_pending([row], 2)
    finally:
        model_correction.evaluate = verify_mod.evaluate

# 17. execution validation needs real Blender: coverage, hidden checks, class +
#     register exemption, and combine_data dropping "unverified" rows.
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
CLASS_REGISTER_ANSWER = "```python\n" + CLASS_REGISTER + "```"


def check_execution():
    codes = [UNCALLED, GUARDED, CALLED, ATTR_ERROR, CLASS_REGISTER, EMPTY, EMPTY]
    checks = ["", "", "", "", "", "assert len(bpy.data.objects) == 1",
              "assert len(bpy.data.objects) == 0"]
    res = execution_validator.run_batch(codes, checks=checks)
    assert len(res) == len(codes), res

    assert res[0]["code_ok"] and res[0]["coverage"] < execution_validator.COVERAGE_FLOOR, res[0]
    assert res[1]["code_ok"] and res[1]["coverage"] < execution_validator.COVERAGE_FLOOR, res[1]
    assert res[2]["code_ok"] and res[2]["coverage"] == 1.0, res[2]

    assert not res[3]["ok"] and res[3]["error_type"] == "AttributeError", res[3]

    assert res[4]["code_ok"] and execution_validator.is_class_register(CLASS_REGISTER), res[4]

    assert res[5]["ok"] and res[5]["check_ok"] is True, res[5]
    assert not res[6]["ok"] and res[6]["check_ok"] is False, res[6]
    assert res[6]["stage"] == "check", res[6]

    ok, reason = execution_validator.validate(UNCALLED)
    assert not ok and reason.startswith("low_coverage:"), (ok, reason)
    ok, reason = execution_validator.validate(CLASS_REGISTER)
    assert ok, (ok, reason)

    cam = execution_validator.run_with_check(
        "import bpy", "import bpy\nassert bpy.context.scene.camera is not None")
    assert cam.get("check_ok") is False, cam

    vacuous_answer = "```python\nimport bpy\nbpy.ops.mesh.primitive_cube_add()\n```"
    vacuous_check = "assert len(bpy.data.objects) >= 1"
    okv, rev, _info = model_correction.evaluate("How do I add a cube?", vacuous_answer, vacuous_check)
    assert not okv and rev == "vacuous_check", (okv, rev)

    fn_answer = "```python\ndef double(x):\n    return x * 2\n\ndouble(21)\n```"
    fn_check = "assert double(21) == 42"
    okf, reas, _info = model_correction.evaluate("Write a function that doubles.", fn_answer, fn_check)
    assert okf and reas == "ok", (okf, reas)

    okf2, reas2, _info = model_correction.evaluate("Write a function that doubles.", fn_answer,
                                           "assert False", require_check=True)
    assert not okf2 and reas2.startswith("check_failed"), (okf2, reas2)

    okc, reas3, _info = model_correction.evaluate("Add an operator.", CLASS_REGISTER_ANSWER,
                                          "assert False", require_check=True)
    assert not okc and reas3.startswith("check_failed"), (okc, reas3)

    okp, resp, _info = model_correction.evaluate("Add an operator.", CLASS_REGISTER_ANSWER,
                                         "bpy.ops.object.thing()", require_check=True)
    assert okp and resp == "ok", (okp, resp)

    has_answer = "```python\nimport bpy\ndef set_flag(obj, attr):\n    if hasattr(obj, attr):\n        setattr(obj, attr, True)\n```"
    okh, reh, _info = model_correction.evaluate("Set a flag on an object.", has_answer, "",
                                        require_check=False)
    assert not okh and reh.startswith("low_coverage:"), (okh, reh)

    obj_answer = "```python\ndef get_objects():\n    return list(bpy.data.objects)\n```"
    oko, rea, _info = model_correction.evaluate("List Blender objects.", obj_answer,
                                        "assert all(isinstance(o, str) for o in get_objects())",
                                        require_check=True)
    assert not oko and rea.startswith("check_failed"), (oko, rea)

    good = "```python\nimport bpy\nbpy.ops.mesh.primitive_cube_add(size=2, align='CURSOR')\n```"
    rows = [
        {"question": "How do I add a cube?", "answer": good, "validated": True},
        {"question": "How do I add a cube twice?", "answer": good,
         "validated": "unverified"},
        {"question": "How do I add a cube three times?", "answer": good},
    ]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "rows.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        kept = combine_data.load_clean("model_corrections", str(path))
    questions = [row["question"] for row in kept]
    assert "How do I add a cube?" in questions, questions
    assert "How do I add a cube three times?" in questions, questions
    assert "How do I add a cube twice?" not in questions, questions

    # Small model pass -> final producer is the small model, no reference shortcut.
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "corrections.jsonl"
        model_correction.OUTPUT = out
        model_correction.evaluate = lambda question, answer, check="", require_check=False: (
            True, "ok", model_correction._empty_info())
        item = {"id": 7, "question": "q1", "reference": "Ref answer.\n\n```python\nimport bpy\n```",
                "reference_check": "chk", "attempts": 0}
        small = "Use it.\n\n```python\nimport bpy\n```"
        row, key = model_correction.process(item, lambda s: None, small_answer=small)
        assert key == "passed" and row["producing_model"] == model_correction.SMALL_MODEL, (key, row)
        assert row["id"] == 7 and row["attempts"] == 0


if shutil.which(config.BLENDER_BIN) is None:
    print(f"SKIPPED execution checks: Blender not found ({config.BLENDER_BIN})")
else:
    check_execution()
    print("EXECUTION CHECKS PASSED")

print("ALL CHECKS PASSED")
