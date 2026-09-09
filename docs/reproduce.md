# Reproduction protocol

## Environment and sources

Use the locked Python 3.12 environment: `uv sync --locked --dev`. Measured hardware was an Apple M5 Pro with 20 GPU cores and 48 GiB unified memory, running macOS 26.6.2. Keep the machine on AC power, close other GPU workloads and record thermal/power conditions. GPU CLI commands use an exclusive local lock; unrelated applications are not controlled by that lock.

The README pins both source revisions. Native int4 supplies the baseline, calibration activations, retained tensors and tokenizer assets. BF16 supplies the 192 MLP projection matrices. Inference from `.morph` requires neither source. Newly generated files include explicit encoder and source provenance. Their header lengths may differ from archived files even when their numeric payloads match.

Calibration uses the first 128-token chunk at validation-token offset 32,768, with a fresh model cache. The older capture archive contained more tokens, but the measured encoder consumed only that chunk. `morph32 calibrate` reproduces that input and saves 128 normalized channel-importance arrays. Gate projections share up-projection inputs. No test question or evaluation token is used for this calibration.

## Quality and throughput

Run each command in a fresh process. Use the same frozen text, offsets and token counts for all profiles. NLL is the mean next-token cross-entropy in float32, accumulated in 128-token chunks through a continuous cache; perplexity is `exp(NLL)`. Each selected window starts with an empty cache. No chat template is used for corpus scoring.

```bash
# Native reference, first test window and synchronized generation.
uv run morph32 evaluate --baseline models/Qwen3.8-27B-4bit \
  --tokens 4096 --offset 0 --output results/native-4096.json

# Both MORPH profiles, same window and generation parameters.
for profile in morph32-3s morph32-c; do
  uv run morph32 evaluate --checkpoint "checkpoints/$profile.morph" \
    --tokens 4096 --offset 0 --output "results/$profile-4096.json"
done

# Independent window used for the headline perplexity comparison.
uv run morph32 evaluate --baseline models/Qwen3.8-27B-4bit \
  --tokens 4096 --offset 8192 --quality-only --output results/native-heldout.json
for profile in morph32-3s morph32-c; do
  uv run morph32 evaluate --checkpoint "checkpoints/$profile.morph" \
    --tokens 4096 --offset 8192 --quality-only --output "results/$profile-heldout.json"
done
```

For a quick preliminary screen, use `--tokens 512 --quality-only`. Archived first-window 512-token perplexity increases were 5.66% for morph32-3s and 9.69% for morph32-c. Do not substitute that preliminary result for the 4,096-token checks.

Unless `--quality-only` is set, the evaluator warms generation, runs three greedy generations with 64 output tokens, synchronizes MLX and records each run. Report the median of each throughput field. It also records synchronized wall time, first-token latency and generated token IDs. Outputs may differ between quantizations. All source and checkpoint measurements use complete storage accounting; runtime parameters and allocator/process observations are separate fields.

Published timings came from separate sessions of the archived implementation. The new evaluator preserves the protocol but does not promise identical timings across processes, thermal conditions or future OS releases. A smaller checkpoint is not proof of lower peak memory: compare the recorded allocator and process measurements too.

## MMLU subset

The frozen 50-question file includes exact five-shot prompts, question IDs, gold answers and prompt hashes. It uses 50 distinct subjects, with category counts 13/13/12/12 and seed 20260910.

Score one example at a time. Append each of ` A`, ` B`, ` C`, ` D`; verify that each option adds exactly one token without changing the prompt prefix. Take the highest next-token log probability. Use no chat template, generated reasoning or sampling. This measures choice likelihood, not a model's willingness to emit a clean answer letter.

```bash
uv run python scripts/run_mmlu.py --subset data/mmlu-50.json \
  --output results/mmlu-50
```

The runner gives each fresh subprocess a hard 180-second wall-clock limit, including loading and kernel compilation. An internal deadline stops starting questions near the limit. Inspect `processes.json` and every result's `status` and question count. A killed or incomplete run is not a comparable full-subset score. A slower machine may need a smaller subset; the script does not silently report incomplete scores as 50-question results.

The archived 50-question process times were 81.53 seconds native, 105.85 seconds morph32-3s and 93.94 seconds morph32-c. These are a different workload from 64-token autoregressive generation; they do not establish that morph32-c generates faster.

Data packaging removed machine-specific paths and selection bookkeeping from the original subset JSON. Consequently, its whole-file hash differs from the archived subset hash. The audit compares unchanged question IDs and prompt hashes; tests verify prompt integrity and unique IDs.

## Reasoning, code and long context

`morph32 workloads` uses non-thinking chat templates, greedy decoding, a 1,024-token output cap and 512-token prefill chunks. It runs four reasoning questions, four Python-function tasks and a retrieval prompt of 17,644 tokens. The code grader checks generated functions in constrained child processes. It is a benchmark harness, not a general-purpose sandbox for arbitrary untrusted programs.

Archived results: all profiles passed 413/413 functional assertions across the four functions and recovered the long-context access code. Reasoning final-answer extraction scored native 2/4, morph32-3s 3/4 and morph32-c 2/4. These small smoke tests are not substitutes for a broad reasoning/coding evaluation. All generated text and per-task grading remain in `evidence/*-workloads.json`.

## Audit and release checks

```bash
# No model or GPU required.
uv run python scripts/audit_evidence.py
uv run pytest -m 'not metal'

# Small real GPU checks, not a model benchmark.
uv run pytest -m metal

# One archived question plus one cached decode step per checkpoint.
for profile in morph32-3s morph32-c; do
  uv run python scripts/smoke_checkpoint.py \
    --checkpoint "checkpoints/$profile.morph" --profile "$profile" \
    --output "results/smoke-$profile.json"
  uv run python scripts/check_conversion_rows.py \
    --bf16 models/Qwen3.8-27B-BF16 --calibration results/calibration.safetensors \
    --checkpoint "checkpoints/$profile.morph" --profile "$profile" \
    --output "results/rows-$profile.json"
done
```

The row check refits four rows from each gate/up/down projection in layers 0, 32 and 63. It compares stored seeds and FP16 scales exactly, covering both layouts. It needs BF16 and calibration inputs; the loader smoke test uses only the standalone checkpoint. Full conversion must still be followed by fresh-process quality and memory measurements before claiming it reproduces all archived results.
