# Blender Copilot

[![Model on Hugging Face](https://img.shields.io/badge/Hugging%20Face-blender_copilot_v1-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/BlenderFace34/blender_copilot_v1)
![Blender 5.1.1](https://img.shields.io/badge/Blender-5.1.1-E87D0D?logo=blender&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)

Fine-tuning data generator for a Blender 5.1.1 Python coding assistant (QLoRA on a Qwen3.5-4B). Model: [BlenderFace34/blender_copilot_v1](https://huggingface.co/BlenderFace34/blender_copilot_v1).

## Model stats
> Each model was asked 18 Blender tasks, 5 samples per task (90 samples total). Every generated script runs in a fresh headless Blender and is checked against a task-specific assertion.

| Model | pass@1 | pass@5 | Passed | Crashed | Easy | Medium | Hard | Fix |
|-------|-------:|-------:|-------:|--------:|-----:|-------:|-----:|----:|
| `qwen3.5-4b` | 8.9% | 33.3% | 8/90 | 80 | 20% | 7% | 0% | 7% |
| `blender-copilot-v1`| 26.7% | 44.4% | 24/90 | 62 | 56% | 10% | 0% | 47% |
| `qwen3.5-9b` | 35.6% | 66.7% | 32/90 | 53 | 60% | 23% | 5% | 60% |

```text
qwen3.5-4b         | ████ 8.9%
blender-copilot-v1 | █████████████ 26.7%
qwen3.5-9b         | █████████████████ 35.6%
```

Blender-copilot-v1 triples pass@1 over its base (8.9% → 26.7%) and raises pass@5 to 44.4% coming closer to the 9B base.

## Training
 
Training code is not part of this repo; the settings are recorded here for reference.
 
- **Base model:** `unsloth/Qwen3.5-4B`
- **Method:** QLoRA, 3 epochs scheduled (405 steps), peak LR 2e-4 with linear decay
- **Data:** about 1,080 training and 38 validation examples (built with this pipeline)
- **Checkpoint:** the exported model is the checkpoint near step 130, where validation loss was lowest (about 0.54). Validation loss rose afterwards while training loss kept falling (overfitting).
- **Export:** merged and converted to GGUF (`Q4_K_M`)
## Using the model
 
The GGUF loads in current [llama.cpp](https://github.com/ggml-org/llama.cpp). It currently **fails to load in Ollama** (`tensor 'blk.0.ssm_beta.weight' not found`), because the exported file stores the SSM tensors in a fused layout that Ollama's bundled loader does not expect yet.
 
```bash
llama-server -m Qwen3.5-4B.Q4_K_M.gguf -ngl 99 -c 4096 --port 8080 \
  --chat-template-kwargs '{"enable_thinking":false}'
```
 
This serves a chat UI and an OpenAI-compatible API at `http://localhost:8080`. Run the model with thinking disabled; with thinking enabled, the answer may end up in `reasoning_content` instead of `content`.
 

## Repo layout

```
core/               shared Blender runner/validator, code + row helpers, config
pipeline/           runnable steps: synthetic, correction, verify, combine, benchmark
data/raw/           hand-collected / extracted sources
data/generated/     pipeline outputs (synthetic, corrections, API cache)
data/final/         training-ready train/val/preferred + manifest
bench/              benchmark cases + headless Blender runner (read-only assets)
bench/results/      benchmark JSON (format unchanged)
tests/              validator + pipeline smoke tests
scripts/            deprecated launchers kept for old commands
logs/               runtime logs
```

All paths and the target Blender version live in `core/config.py`.


## Pipeline

1. **Build API cache** - `blender -b --python core/build_api_cache.py`
    - Writes `data/generated/bpy_api_cache.json`.
     This is the static validator's ground truth.

2. **Extract docs** (one-time) - `python3 -m pipeline.extract_docs`
   - Converts Sphinx HTML to Q&A pairs in `data/raw/doc_data.jsonl`.

3. **Generate synthetic samples** - `python3 -m pipeline.generate_synthetic --limit 200`
   - Every candidate is validated statically and executed in a fresh headless Blender.
   - Rows record `producing_model`, `validated`, `attempts`, `timestamp`.

4. **Model corrections** - `python3 -m pipeline.model_correction --questions 20`
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

5. **Re-run verification** - `python3 -m pipeline.verify --target corrections|synthetic|all`

6. **Combine sources** - `python3 -m pipeline.combine_data`
   - Cleans, validates, drops benchmark-prompt matches/near-matches into
     `data/final/dropped_benchmark_overlap.jsonl`, removes duplicates, caps types.
   - Writes `data/final/train.jsonl`, `val.jsonl`, `preference_pairs.jsonl`, and
     `manifest.json` (per-source counts, verified %, duplicates, length stats,
     API-area/category coverage).
   - Groups: `blender_core` (70%), `general_python` (10%), replay (when the file exists).

7. **Benchmark** - `python3 -m pipeline.benchmark --model qwen3.5:9b --samples 5 --temperature 0.3 --no-think`
   - Results go to `bench/results/` with `model`, `think`, `samples_per_case`,
     `temperature`, summary, and per-sample outcomes.

## Data sources

| File | Contents | Origin |
|------|----------|--------|
| `data/raw/doc_data.jsonl` | Blender 5.1.1 Python API reference, converted to Q&A | Sphinx HTML from the Blender docs |
| `data/raw/general_python.jsonl` | General Python coding instructions (keeps basic Python alive) | [CodeAlpaca (general code)](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) |
| `data/raw/replay_data.jsonl` | Optional replay rows (schema as usual, source `replay`) | your own file |
| `data/generated/synthetic_training_data.jsonl` | Generated samples, statically and headlessly validated | `pipeline/generate_synthetic.py` |
| `data/generated/model_corrections.jsonl` | Small-model answers, verified and sequentially corrected | `pipeline/model_correction.py` |

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
python3 pipeline/benchmark.py --backend openai --model <label> --samples 5 --temperature 0.3

# Smoke tests (execution checks require Blender)
python3 tests/test_validator.py
```


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
- The API cache is built against the installed Blender (5.1.1).
- There is no training or export scripts in this repo.
- Blender API coverage is spotty.
- The model is not reliable for medium or hard tasks. Always review generated code before running it in a scene you care about.
- Thinking mode: run with thinking disabled (`enable_thinking: false`); with it on, the answer may land in `reasoning_content` and `content` can come back empty. All benchmarks used thinking off.
