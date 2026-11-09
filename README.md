# Blender Copilot

Fine-tuning data generator for a Blender 5.1.1 Python coding assistant (QLoRA on a Qwen coder model).

## Repo layout

```
scripts/            pipeline code
data/raw/           hand-collected / extracted sources
data/generated/     pipeline outputs
data/final/         training-ready file (question/answer only)
logs/               runtime logs
```

All paths, the target Blender version, and task types live in `scripts/config.py`.

## Data sources

| File | Contents | Origin |
|------|----------|--------|
| `data/raw/doc_data.jsonl` | Blender 5.1.1 Python API reference, converted to Q&A | Sphinx HTML from the Blender docs |
| `data/raw/addons_data.jsonl` | Real Blender addon code | [Blender addons](https://huggingface.co/datasets/stodipro/blenderaddon) |
| `data/raw/general_data.jsonl` | General Blender usage Q&A | [General Blender Info](https://huggingface.co/datasets/Fikrat/blender) |
| `data/raw/general_python.jsonl` | General Python coding instructions (keeps basic Python alive) | [CodeAlpaca (general code)](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) |
| `data/generated/synthetic_training_data.jsonl` | Generated samples, statically and headlessly validated | `scripts/generate_synthetic.py` |
| `data/generated/model_corrections.jsonl` | Small-model answers that were checked and corrected | `scripts/model_correction.py` |

## Pipeline

1. **Build API cache** — `blender -b --python scripts/build_api_cache.py`
   - Introspects the real installed Blender (`bpy.types`, `bpy.ops`, `bmesh`, `mathutils`, etc.).
   - Writes `data/generated/bpy_api_cache.json`: signatures, enum values, namespaces,
     and a small deprecated/removed table. This is the static validator's ground truth.

2. **Extract docs** (one-time/upstream) — `python3 scripts/extract_docs.py`
   - Converts Sphinx HTML to Q&A pairs in `data/raw/doc_data.jsonl`.
   - Override with `--doc-dir` / `--output`.

3. **Generate synthetic samples** — `python3 scripts/generate_synthetic.py`
   - Samples real API references, then generates task-specific samples:
     - `explanation` — short explanation + working code
     - `bugfix` — the model injects a realistic bug; rows where the "fix" is identical to the buggy input are rejected
     - `fim` — `<|fim_prefix|>`/`<|fim_suffix|>`/`<|fim_middle|>` splits built programmatically from validated code (never character-masked LLM output)
     - `refactor` — inefficient → optimized, behavioral equivalence checked where possible
     - `addon` — operator + panel + properties + register/unregister
   - Every candidate is validated statically and executed in a fresh headless Blender process.
   - Output: `data/generated/synthetic_training_data.jsonl`.

4. **Model corrections** — `python3 scripts/model_correction.py --questions 20`
   - The big LLM (OmniRoute) writes Blender 5.x type-4 questions from topic-bucketed seeds.
   - `qwen3.5:4b` (Ollama) answers each question.
   - The answer's code block is executed in a fresh headless Blender.
   - If it fails, the big LLM rewrites the correct answer with the failure fed back;
     the big model's reference answer is tried first, so a valid one costs no extra call.
   - Output: `data/generated/model_corrections.jsonl` — question, small-model answer,
     final answer, failure reason, and `validated`. Resume skips recorded questions.

5. **Combine sources** — `python3 scripts/combine_data.py`
   - Cleaning happens here before anything is merged, so every source is already
     clean when combined: legacy headings stripped, raw code fenced, leaked prompts
     and `validated: false` rows dropped, duplicate code/questions removed, per-type
     35% cap. There is no separate cleaner.
   - Groups:
     - `blender_core` (70%) — `data/raw/doc_data.jsonl` fills the target;
       `data/generated/synthetic_training_data.jsonl` and
       `data/generated/model_corrections.jsonl` are kept whole
     - `general_python` (10%) — `data/raw/general_python.jsonl`, keeps basic Python ability alive
     - `addons` — `data/raw/addons_data.jsonl`, whole source
     - `general_usage` — `data/raw/general_data.jsonl`, whole source
   - Output is one question/answer-only file: `data/final/train.jsonl`.

## Output schema

```json
{"question": "How do I add a cube at the cursor?", "answer": "Use primitive_cube_add with align='CURSOR'.\n\n```python\nimport bpy\nbpy.ops.mesh.primitive_cube_add(align='CURSOR')\n```"}
```

## Running

```bash
# Points at the local OmniRoute server and qwen3.5:4b on Ollama.
omniroute serve
export OMNIRoute_MODEL=auto/best-coding

# Generate
python3 scripts/generate_synthetic.py --limit 200
python3 scripts/model_correction.py --questions 20

# Clean + merge everything into data/final/train.jsonl
python3 scripts/combine_data.py

# Smoke tests
python3 scripts/test_validator.py
```

## Configuration knobs

- `scripts/config.py`: Blender version/binary, all data paths, task-type map.
- `scripts/generate_synthetic.py` top `CONFIG` dict: endpoint, model, cap, namespace cap.
- `scripts/execution_validator.py`: `BLENDER_EXEC_TIMEOUT`, `BLENDER_BIN`.
- `scripts/build_api_cache.py`: deprecated/removed API table.

## Notes / known limitations

- Execution validation is slow (one fresh Blender process per sample). Use `--limit` and parallelism carefully.
- Refactor behavioral equivalence is currently implemented for a narrow set of patterns; broaden the templates as needed.
- The API cache is built against the installed Blender; keep it at 5.1.1 so docs, cache, and execution agree.
- `combine_data.py` still deduplicates across the whole/`keep` sources; "whole" means not ratio-sampled, not exempt from dedup or validation.
