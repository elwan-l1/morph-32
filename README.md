# MORPH-32

**Metal Orbit-Recurrent Packed Hypercode. Procedural weight quantization for MLX.**

MORPH-32 encodes each group of 32 weights as a few seed words and a scale. Metal regenerates related sign planes with XOR and rotation, then multiplies the reconstructed values directly with activations. The MLP weights stay compressed throughout inference.

On Qwen3.8-27B, **morph32-c reduces complete checkpoint size by 18.3% and resident weights by 19.5%** versus native MLX int4. Perplexity rises 5.32% on a separate 4,096-token test window. **Generation is slower**: this release trades speed for storage and memory.

[Format](docs/format.md) · [Reproduce](docs/reproduce.md) · [Evidence](evidence/summary.json) · [Validation](docs/validation.md) · [OpenCode](docs/serving.md)

## Results

Native baseline: [`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/tree/3e6447f082e89cc7f0bc6e5441afd38dfce760ff). Apple M5 Pro, 20 GPU cores, 48 GiB unified memory; MLX 0.32.2 and MLX-LM 0.31.3. Lower is better for size, memory and perplexity; higher is better for throughput and accuracy.

| Measurement                             | Native MLX int4 |  morph32-3s |    morph32-c |
| :-------------------------------------- | --------------: | ----------------: | ---------------: |
| Complete checkpoint                     |        16.08 GB |          13.67 GB |     **13.14 GB** |
| Complete physical bits/weight           |           4.703 |             3.998 |        **3.841** |
| MLP representation bits/weight          |             4.5 |               3.5 | **3.25 average** |
| Resident weights                        |        15.13 GB |          12.72 GB |     **12.19 GB** |
| Peak MLX memory                         |        18.26 GB |          15.87 GB |     **15.33 GB** |
| Prompt processing, tokens/s             |      **504.72** |            430.23 |           336.50 |
| Generation, tokens/s                    |       **18.05** |             11.24 |             8.46 |
| Perplexity, first 4,096 tokens          |      **5.4638** |            5.6266 |           5.8075 |
| Perplexity, separate 4,096-token window |      **4.7563** |            4.8712 |           5.0095 |
| Separate-window perplexity increase     |       Reference |            +2.42% |           +5.32% |
| MMLU subset, 50 questions               |   41/50 (82.0%) | **46/50 (92.0%)** |    42/50 (84.0%) |

GB means decimal GB. Complete physical bits/weight includes every tensor, metadata field, tokenizer asset and container byte, divided by 27,356,728,560 logical parameters. Matrix rates include scales; native int4 has 4-bit codes plus BF16 scale/bias per 64 weights. Memory rows overlap and must not be added.

Throughput is the median of three warmed 4,096-token prompt / 64-token generation runs. GPU execution was synchronized; models ran in separate sessions. The separate perplexity window starts at test-token offset 8,192 with a fresh cache. The 50-question MMLU subset uses five-shot choice likelihoods without generated thinking. **This small subset does not establish general MMLU superiority or quality equivalence.** Only text inference was evaluated; retained vision tensors still count toward complete storage.

Numbers are archived measurements, not a new full benchmark of this packaged release. [Machine-readable records](evidence/) include per-token losses, per-question predictions, generation outputs and memory observations. Rebuild and verify the comparison without a GPU:

```bash
uv sync --locked --dev
uv run python scripts/audit_evidence.py
```

## What makes it different

Native affine quantization assigns each weight an independent integer code. MORPH encodes a **shared state for 32 weights**. Three seed words define the first three sign planes; two additional planes come from a deterministic Boolean recurrence. Changing one stored bit can change several reconstructed weights, so the encoder optimizes the entire tile together.

For stored words `A`, `B`, `C`, the remaining planes are:

```text
D = B XOR rotl(C, 5) XOR rotl(C, 13) XOR rotl(C, 21)
E = C XOR rotl(D, 5) XOR rotl(D, 13) XOR rotl(D, 21)
w[j] = scale × (sign(A[j]) + sign(B[j])/2 + sign(C[j])/4
                         + sign(D[j])/16 + sign(E[j])/32)
```

Each sign is −1 or +1. There is no learned lookup table or dense MLP matrix to keep resident. This is lossy re-quantization from BF16 weights. The contribution here is the implemented recurrence, fitting procedure, packed layout and direct MLX/Metal execution; priority over all procedural or vector quantization research is not established.

**morph32-3s** (three seeds) stores three `uint32` words plus one FP16 scale per tile: 112 bits / 32 weights = **3.5 bpw**. It converts all 192 MLP projections and offers the stronger measured quality/speed tradeoff of the two MORPH profiles.

**morph32-c** (compact) stores only half of the third seed in selected projections and derives its remaining bits from the stored state: **3.0 bpw** there. It uses this format for 96 gate/up projections in layers 8–55, retaining morph32-3s for the other 96 MLP projections. The MLP average is **3.25 bpw**.

Attention, embeddings and other unconverted tensors retain native weights. Supported BF16 scale/bias metadata is compressed losslessly. Every retained tensor and model asset is embedded in the standalone `.morph` checkpoint.

## Built for Apple Metal

The format keeps 32-weight tiles and 32-bit seed words. The generation kernel assigns **one complete tile per lane**, with a 32-lane SIMD group reducing partial dot products. Prefill reconstructs 64×64 weight tiles in threadgroup memory and uses MLX's M5 NAX helpers for matrix multiplication. Neither path expands a complete MLP weight matrix.

**This release requires Apple M5 and macOS with working MLX Metal support.** Tested on M5 Pro with 48 GiB memory. M1–M4 support and other model architectures are not validated. The prefill kernel uses private MLX headers, so the MLX dependency is pinned.

## Build your checkpoints

Run commands from the repository root. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then download the exact source revisions. Conversion needs both native int4 and BF16 snapshots; inference needs only the resulting `.morph` file. Allow roughly 130 GB free disk space for sources, both outputs and working room. Conversion memory and duration are not characterized by the inference table.

```bash
uv sync --locked --dev

uv run hf download mlx-community/Qwen3.8-27B-4bit \
  --revision 3e6447f082e89cc7f0bc6e5441afd38dfce760ff \
  --local-dir models/Qwen3.8-27B-4bit
uv run hf download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir models/Qwen3.8-27B-BF16

uv run morph32 calibrate --source models/Qwen3.8-27B-4bit \
  --output results/calibration.safetensors

for profile in morph32-3s morph32-c; do
  uv run morph32 convert \
    --source models/Qwen3.8-27B-4bit --bf16 models/Qwen3.8-27B-BF16 \
    --calibration results/calibration.safetensors --profile "$profile" \
    --output "checkpoints/$profile.morph" \
    --report "results/conversion-$profile.json"
done
```

Calibration uses 128 tokens from the included WikiText-2 validation corpus, beginning at token 32,768. The encoder fits BF16 rows using activation-channel importance, three initial scales and two seed-bit search sweeps. No training or dataset download is needed beyond the frozen local evaluation data. Conversion refuses to overwrite existing outputs.

```bash
uv run morph32 inspect checkpoints/morph32-c.morph
uv run morph32 generate --checkpoint checkpoints/morph32-c.morph \
  --prompt "Write a Python function that merges overlapping intervals." \
  --max-tokens 256
```

Generation defaults to greedy decoding with `enable_thinking=False`. Loading verifies every tensor's SHA-256 and uses embedded config/tokenizer assets. `.morph` is a safetensors container with a versioned MORPH manifest, not an overlay requiring another checkpoint.

## Chat in OpenCode

Serve the converted checkpoint with a second command:

```bash
uv run morph32 serve --checkpoint checkpoints/morph32-3s.morph --port 8081
```

This starts a local OpenAI-compatible API at `http://127.0.0.1:8081/v1`, with streaming, tool-call parsing and thinking disabled. Add the [OpenCode provider example](docs/opencode.example.json) to your config, then select **MORPH-32 Local / Qwen3.8 27B · morph32-3s**. [Serving instructions](docs/serving.md) cover conversion, limits and shutdown.

## Reproduce evaluations

```bash
# One model: 4,096-token scoring and three warmed generation runs.
uv run morph32 evaluate --checkpoint checkpoints/morph32-3s.morph \
  --output results/morph32-3s-4096.json

# All three models: frozen 50-question subset, at most 180 seconds per process.
uv run python scripts/run_mmlu.py --output results/mmlu-50

# Longer reasoning, code-function checks and 17,644-token retrieval.
uv run morph32 workloads --checkpoint checkpoints/morph32-c.morph \
  --output results/morph32-c-workloads.json
```

Use `--baseline models/Qwen3.8-27B-4bit` instead of `--checkpoint` for native evaluation. See [the complete protocol](docs/reproduce.md) for both test windows, MMLU scoring, timeout handling and exact accounting.

## Repository and validation

```text
src/morph32/   Encoder, checkpoint I/O, Metal kernels, runtime and evaluation CLI
scripts/       Evidence audit, bounded MMLU runner and checkpoint smoke test
tests/         Format, dataset and small numerical kernel checks
docs/          Format specification, reproduction protocol and validation scope
evidence/      Archived measurements and separate release validation reports
data/          Frozen evaluation text and five-shot MMLU prompts
checkpoints/   Local .morph files; ignored by Git
```

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

Both measured checkpoints passed fresh-process loading and hash verification using this package. A checked MMLU question reproduced archived log probabilities exactly for each profile; cached single-token execution and resident byte counts also passed. Small Metal tests cover both layouts, prefill boundaries and retained metadata. morph32-3s was subsequently converted through the packaged CLI: all 2,610 tensor hashes and projection descriptors match the measured checkpoint. Local chat, streamed tool calls and an OpenCode conversation passed. morph32-c conversion and full-model benchmark suites were not rerun; [validation scope](docs/validation.md) records the distinction.

Code: [MIT](LICENSE). Model weights and bundled datasets retain their [upstream terms](NOTICE.md). The format documentation follows the explicit layout/reconstruction approach of [DeepSeek's weight documentation](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/README_WEIGHTS.md); evaluation presentation follows the separation of methods, measured tradeoffs and usage in [Unsloth's quantization documentation](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs). Neither method is implemented here.

Profile names are `morph32-3s` (three seeds) and `morph32-c` (compact). The CLI still accepts `three-seed` and `compact` as aliases. Existing checkpoint manifests remain readable; renaming a profile does not change its tensors.
