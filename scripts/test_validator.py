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

# 13. legacy two-section headings are stripped by the cleaner
import clean_existing
legacy = ("1) Architecture Overview\n\nProse.\n\n2) The Code\n"
          "```python\nimport bpy\n```")
assert clean_existing.normalize_answer(legacy) == (
    "Prose.\n\n```python\nimport bpy\n```")

# 14. type-4 pipeline parses the big model's ===ANSWER=== wrapper
import pipeline_type4
assert pipeline_type4.parse_answer("noise\n===ANSWER===\nfinal") == "final"
assert pipeline_type4.parse_answer("```python\nx = 1\n```") == "```python\nx = 1\n```"

print("ALL CHECKS PASSED")
