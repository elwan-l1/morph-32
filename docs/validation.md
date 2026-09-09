# Validation scope

## Archived model experiments

The comparison table comes from completed model evaluations of the measured checkpoints. It includes two 4,096-token windows, synchronized prompt/generation timings, physical file accounting and the frozen 50-question MMLU subset. The supporting records are curated copies: machine-specific paths and unrelated provenance were removed; numeric results, token losses, answer records and generated outputs were preserved. `evidence/provenance.json` gives SHA-256 hashes for the original and curated reports. The evidence audit recomputes summary values and verifies shared token/prompt identities.

These records were not regenerated during repository packaging. They establish the measured checkpoints' behavior under the recorded protocol, not bit-identical results for every new conversion or broader model quality.

## Packaged release checks

On Apple M5 Pro, the standalone package passed:

- Fresh-process loading of each complete `.morph` checkpoint with every tensor hash verified, no native model argument and no original research code dependency.
- Exact reproduction of archived four-choice MMLU log probabilities for `college_chemistry:17`, one checked question per profile.
- Finite cached single-token execution, exercising the generation path after prefill.
- Exact resident parameter counts and independently computed physical bits/weight for both checkpoints.
- Fresh 128-token calibration from the pinned native model. Refitting four BF16 rows of all three MLP projections in layers 0, 32 and 63 reproduced stored words and FP16 scales exactly: 36 rows per profile.
- Small numerical tests of morph32-3s and partial layouts: CPU/Metal reconstruction equality, exact matrix storage rates, direct generation and NAX prefill at batch sizes 3, 33 and 129. Prefill tests account for rounding reconstructed weights to activation dtype.
- Exact retained BF16 metadata reconstruction and sampled native-product equality, frozen-question identity and uniqueness, physical inventory checks, Ruff and package build checks.

Fresh checkpoint and row-check reports are `evidence/release-*.json` and `evidence/conversion-rows-*.json`. They are separate from archived benchmark records. GPU tests use small matrices; they are correctness checks, not performance measurements. The Linux CI job runs static, model-free and evidence checks only.

## morph32-3s conversion and local serving

The packaged CLI subsequently completed all 192 morph32-3s MLP projections in 161.76 seconds. The new checkpoint is 13,670,822,555 bytes, including 129 additional manifest bytes compared with the archived file. All 2,610 tensor payload hashes and every projection descriptor match the measured morph32-3s checkpoint. The server independently verified every new tensor at startup. See `evidence/morph32-3s-conversion-verification.json`.

The local HTTP server passed a non-streaming chat request and a streamed function-call request, including parsed arguments and the `tool_calls` finish reason. An OpenCode conversation through the custom provider returned the expected connection-check text with zero reasoning tokens. These checks exercise loading, prefill, cached generation, client configuration and streaming protocol handling.

Full morph32-c conversion, the complete new evaluation CLI and all model workloads were not rerun. The benchmark table continues to report archived experiments. Tensor identity establishes reproduction of morph32-3s weights; it does not replace fresh timing or broader quality measurements.

## Limits

Only the pinned Qwen3.8-27B architecture and M5 Pro runtime were tested. No BF16 full-model quality comparison, complete MMLU, image task evaluation, multi-user serving test or other Apple GPU validation is claimed. No statistical evidence establishes the higher 50-question score as a general improvement. Generation remains slower than native int4. Procedural quantization novelty beyond the specific implemented design has not been established.
