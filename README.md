# Blender Copilot

Fine-tuning data generator for a Blender 5.1.1 Python coding assistant (QLoRA on a Qwen coder model).

## Repo layout

```
core/               shared Blender runner/validator, code + row helpers, config
pipeline/           runnable steps: synthetic, correction, verify, combine, benchmark
data/raw/           hand-collected / extracted sources
data/generated/     pipeline outputs (synthetic, corrections, API cache)
data/final/         training-ready train/val/preferred + manifest
bench/results/     benchmark JSON (format unchanged)
tests/              validator + pipeline smoke tests
scripts/            deprecated launchers kept for old commands
bench/              benchmark cases + headless Blender runner (read-only assets)
logs/               runtime logs
```

All paths and the target Blender version live in `core/config.py`.

## Data sources

| File | Contents | Origin |
|------|----------|--------|
| `data/raw/doc_data.jsonl` | Blender 5.1.1 Python API reference, converted to Q&A | Sphinx HTML from the Blender docs |
| `data/raw/addons_data.jsonl` | Real Blender addon code | [Blender addons](https://huggingface.co/datasets/stodipro/blenderaddon) |
| `data/raw/general_data.jsonl` | General Blender usage Q&A | [General Blender Info](https://huggingface.co/datasets/Fikrat/blender) |
| `data/raw/general_python.jsonl` | General Python coding instructions (keeps basic Python alive) | [CodeAlpaca (general code)](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) |
| `data/raw/replay_data.jsonl` | Optional replay rows (schema as usual, source `replay`) | your own file |
| `data/generated/synthetic_training_data.jsonl` | Generated samples, statically and headlessly validated | `pipeline/generate_synthetic.py` |
| `data/generated/model_corrections.jsonl` | Small-model answers, verified and sequentially corrected | `pipeline/model_correction.py` |

## Pipeline

1. **Build API cache** — `blender -b --python core/build_api_cache.py`
   - Introspects the installed Blender and writes `data/generated/bpy_api_cache.json`.
     This is the static validator's ground truth.

2. **Extract docs** (one-time/upstream) — `python3 -m pipeline.extract_docs`
   - Converts Sphinx HTML to Q&A pairs in `data/raw/doc_data.jsonl`.

3. **Generate synthetic samples** — `python3 -m pipeline.generate_synthetic --limit 200`
   - Every candidate is validated statically and executed in a fresh headless Blender.
   - Rows record `producing_model`, `validated`, `attempts`, `timestamp`.

4. **Model corrections** — `python3 -m pipeline.model_correction --questions 20`
   - Per question: ask the small model (`qwen3.5:4b` by default), verify it, then
     ask the big model (OmniRoute pool) to fix the exact failed code with the error.
   - A stored reference answer is tried first, so a valid one costs no extra call.
   - Up to 3 attempts, strictly sequential and on the same question. Stop on the
     first pass; otherwise mark `failed` and move on.
   - Rows keep a stable integer `id`, are resumable, and store `status`,
     `attempts`, `error_history`, `small_model`, `producing_model`, `final_model`.
   - `--questions N` targets N unfinished questions this run; new questions fill
     the gap. `--rows N` processes existing rows only. `--generate N` always adds
     N new questions first. `--revalidate` re-runs verification.
   - Each row is saved after every attempt, so a crash loses at most one call.

5. **Re-run verification** — `python3 -m pipeline.verify --target corrections|synthetic|all`

6. **Combine sources** — `python3 -m pipeline.combine_data`
   - Cleans, validates, drops benchmark-prompt matches/near-matches into
     `data/final/dropped_benchmark_overlap.jsonl`, removes duplicates, caps types.
   - Writes `data/final/train.jsonl`, `val.jsonl`, `preference_pairs.jsonl`, and
     `manifest.json` (per-source counts, verified %, duplicates, length stats,
     API-area/category coverage).
   - Groups: `blender_core` (70%), `general_python` (10%), addons, general usage,
     replay (when the file exists).

7. **Benchmark** — `python3 -m pipeline.benchmark --model qwen3.5:9b --samples 5 --temperature 0.2 --no-think`
   - Results go to `bench/results/` with `model`, `think`, `samples_per_case`,
     `temperature`, summary, and per-sample outcomes.

## Training row schema

```json
{
  "id": "correction:137",
  "question": "How do I add a cube at the cursor?",
  "answer": "Use primitive_cube_add with align='CURSOR'.\n\n```python\nimport bpy\nbpy.ops.mesh.primitive_cube_add(align='CURSOR')\n```",
  "source": "correction",
  "producing_model": "qwen3.5:4b",
  "verified": true,
  "attempts": 0,
  "api_areas": ["bpy.ops.mesh"],
  "content_hash": "…",
  "token_length": 63,
  "timestamp": "2026-11-14T12:00:00+00:00"
}
```

`source` is one of `docs`, `synthetic`, `correction`, `addon`, `general`,
`replay`. Raw docs/addon/general rows are not execution-verified. Legacy
correction rows have `producing_model: "legacy"`.

## Running

```bash
# Optional virtualenv for pipeline deps
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

omniroute serve
export OMNIRoute_MODEL=auto/best-coding

python3 -m pipeline.generate_synthetic --limit 200
python3 -m pipeline.model_correction --questions 20
python3 -m pipeline.combine_data
python3 -m pipeline.benchmark --model qwen3.5:9b --samples 5 --temperature 0.2

# Smoke tests (execution checks require Blender)
python3 tests/test_validator.py
```

Old commands still work through thin launchers in `scripts/` (they print a
deprecation-free passthrough and call the new module).

## Configuration knobs

- `core/config.py`: Blender version/binary, all data/results paths, task-type map.
- `pipeline/generate_synthetic.py` `CONFIG`: endpoint, models, cap, namespace cap.
- `core/execution_validator.py`: `BLENDER_EXEC_TIMEOUT`, `BLENDER_BIN`.
- `core/gpu.py`: `GPU_BUSY_MB` (default 1500) and `GPU_BUSY_UTIL` (default 25).
- `pipeline/model_correction.py`: `SMALL_MODEL`, `SMALL_MODEL_TIMEOUT`,
  `CORRECTION_TIMEOUT`, `MODEL_RETRIES`, max attempts.

## Notes / known limitations

- Execution validation is slow (one fresh Blender process per sample). Use `--limit` and parallelism carefully.
- Refactor behavioral equivalence is narrow; broaden templates as needed.
- The API cache is built against the installed Blender; keep it at 5.1.1.
- `bench/cases.py` and the results JSON format are intentionally unchanged.
- There is no training or export code in this repo.
