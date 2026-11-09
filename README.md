# Blender Copilot

## Pipeline

1. **Build API cache** — `blender -b --python scripts/build_api_cache.py -- data/bpy_api_cache.json`
   - Introspects the real installed Blender (`bpy.types`, `bpy.ops`, `bmesh`, `mathutils`, etc.).
   - Stores signatures, enum values, namespaces, and a small deprecated/removed table.

2. **Generate synthetic samples** — `python3 scripts/generate_synthetic.py`
   - Samples real API references, then generates task-specific samples:
     - `explanation` — short explanation + working code
     - `bugfix` — the model injects a realistic bug; rows where the "fix" is identical to the buggy input are rejected
     - `fim` — `<|fim_prefix|>`/`<|fim_suffix|>`/`<|fim_middle|>` splits built programmatically from validated code (never character-masked LLM output)
     - `refactor` — inefficient → optimized, behavioral equivalence checked where possible
     - `addon` — operator + panel + properties + register/unregister
   - Every candidate is validated statically and executed in a fresh headless Blender process.
   - Fake fixes (answer code identical to the buggy input), unbalanced fences, and leaked reasoning are rejected.
   - The model may return `NO_VALID_SAMPLE`; the generator retries or skips.

3. **Type-4 verification pipeline** — `python3 scripts/pipeline_type4.py --questions 20`
   - The big LLM (OmniRoute) writes type-4 questions from topic-bucketed seeds.
   - `qwen3.5:4b` (Ollama) answers each question.
   - The answer's code block is executed in a fresh headless Blender.
   - If it fails (format, invented API, or execution error), the big LLM rewrites the
     correct answer, with the failure fed back to it; the big model's reference answer
     is tried first, so a valid one costs no extra call.
   - Everything lands in one file, `data/model_corrections.jsonl`: question,
     small-model answer, final answer, failure reason, and `validated`. Resume skips
     recorded questions. `combine_data.py` merges this file into the synthetic bucket
     and drops rows marked `validated: false`.

4. **Combine sources (optional)** — `python3 scripts/combine_data.py`
   - Merges synthetic data with raw sources (doc, addons, general Python/Blender usage).
   - Outputs `data/final_dataset.jsonl` with only `question`, `answer`, `messages`, and `split`.

## Output schema

```json
{
  "messages": [
    {"role": "system", "content": "You are a Blender 5.x Python coding assistant."},
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
export OMNIRoute_MODEL=auto/best-coding

# Generate with real LLM calls
python3 scripts/generate_synthetic.py --limit 200

# Re-clean an existing synthetic file (drops old type-6 rows, fake fixes,
# unbalanced fences, and code that fails in headless Blender)
python3 scripts/clean_existing.py --execute --chunk 25

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
