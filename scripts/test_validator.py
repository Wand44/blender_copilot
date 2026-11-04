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
assert ok(validator.validate_sample("q", a, 7))

print("ALL CHECKS PASSED")
