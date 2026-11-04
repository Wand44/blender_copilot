# Blender Copilot — Synthetic Data Pipeline v2

Fine-tuning data generator for a Blender 4.x Python coding assistant (QLoRA on a Qwen coder model).

## Pipeline

1. **Build API cache** — `blender -b --python scripts/build_api_cache.py -- data/bpy_api_cache.json`
   - Introspects the real installed Blender (`bpy.types`, `bpy.ops`, `bmesh`, `mathutils`, etc.).
   - Stores signatures, enum values, namespaces, and a small deprecated/removed table.

2. **Generate synthetic samples** — `python3 scripts/generate_synthetic.py`
   - Samples real API references, then generates task-specific samples:
     - `explanation` — short explanation + working code
     - `bugfix` — correct code is generated, one realistic bug is injected, verified to fail, then fixed
     - `fim` — fill-in-the-middle splits built programmatically from validated code
     - `refactor` — inefficient → optimized, behavioral equivalence checked where possible
     - `addon` — operator + panel + properties + register/unregister
   - Every candidate is validated statically and executed in a fresh headless Blender process.
   - The model may return `NO_VALID_SAMPLE`; the generator retries or skips.

3. **Combine sources (optional)** — `python3 scripts/combine_data.py`
   - Merges synthetic data with raw sources (doc, addons, general Python/Blender usage).
   - Outputs `data/final_dataset.jsonl` with only `question`, `answer`, `messages`, and `split`.

## Output schema

```json
{
  "messages": [
    {"role": "system", "content": "You are a Blender 4.x Python coding assistant."},
    {"role": "user", "content": "How do I add a cube at the cursor?"},
    {"role": "assistant", "content": "Use primitive_cube_add with align='CURSOR'.\n\n```python\nimport bpy\nbpy.ops.mesh.primitive_cube_add(align='CURSOR')\n```"}
  ],
  "task_type": "explanation",
  "blender_version": "4.2",
  "api_refs": ["bpy.ops.mesh.primitive_cube_add"],
  "validated": true
}
```

## Running

```bash
# Set the target Blender version and model
export TARGET_BLENDER=4.2
export OMNIRoute_MODEL=blender-qa

# Generate with real LLM calls
python3 scripts/generate_synthetic.py --limit 200

# Dry-run with deterministic templates (no LLM, useful for testing the harness)
python3 scripts/generate_synthetic.py --dry-run --limit 20 --no-execute

# Smoke tests
python3 scripts/test_validator.py
```

## Configuration knobs

- `scripts/generate_synthetic.py` top `CONFIG` dict: endpoint, model, cap, namespace cap.
- `scripts/execution_validator.py`: `BLENDER_EXEC_TIMEOUT`, `BLENDER_BIN`.
- `scripts/build_api_cache.py`: deprecated/removed API table.

## Notes / known limitations

- Execution validation is slow (one fresh Blender process per sample). Use `--limit` and parallelism carefully.
- Refactor behavioral equivalence is currently implemented for a narrow set of patterns; broaden the templates as needed.
- The pipeline introspects the installed Blender. If that is newer than the target version, review the deprecated table.
