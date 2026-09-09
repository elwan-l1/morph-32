# Local chat with OpenCode

The `serve` command loads one standalone MORPH checkpoint and exposes MLX-LM's OpenAI-compatible chat API. It uses the MORPH Metal kernels for inference, streams replies, parses tool calls and disables thinking. Requests run sequentially; up to two prompt caches are retained.

## Convert, then serve

After the source downloads and calibration in the README, run these commands from the repository root:

```bash
uv run morph32 convert \
  --source models/Qwen3.8-27B-4bit \
  --bf16 models/Qwen3.8-27B-BF16 \
  --calibration results/calibration.safetensors \
  --profile morph32-3s \
  --output checkpoints/qwen3.8-27b-morph32-3s.morph \
  --report results/conversion-morph32-3s-local.json

uv run morph32 serve \
  --checkpoint checkpoints/qwen3.8-27b-morph32-3s.morph \
  --port 8081
```

The server verifies checkpoint integrity before listening. Keep its terminal open; Ctrl-C stops it. Conversion refuses to overwrite an existing checkpoint or report. If a source snapshot is already cached, its local snapshot directory can replace the corresponding `models/` path.

The endpoint is `http://127.0.0.1:8081/v1`; the model ID is `qwen3.8-27b-morph32-3s`. Defaults: 32,768 total context tokens, at most 4,096 output tokens, 512-token prefill chunks and greedy sampling. The total prompt plus requested output budget must fit the context limit. Clients can choose a lower output budget or a sampling temperature. Thinking remains disabled even if requested by the client. These are serving limits, not a new quality benchmark at maximum context length.

```bash
curl http://127.0.0.1:8081/health
curl http://127.0.0.1:8081/v1/models
curl http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-morph32-3s","messages":[{"role":"user","content":"Hello!"}],"max_tokens":128,"stream":true}'
```

This is a local development server bound to loopback. It has no external network listener and does not require an API key. Unsupported model IDs, adapters and draft models cannot trigger alternate model downloads. The server uses private interfaces from the pinned MLX-LM version; upgrade the dependency only after testing the adapter.

## OpenCode configuration

Merge the `provider` entry in [opencode.example.json](opencode.example.json) into `~/.config/opencode/opencode.json`. Preserve other providers. The example's `model` field selects MORPH as the default. `apiKey: "local"` is a placeholder for the compatible client, not a credential.

The provider uses `@ai-sdk/openai-compatible` and `options.baseURL`, following [OpenCode's custom provider documentation](https://opencode.ai/docs/providers/#custom-provider). After refreshing OpenCode's provider list, select **MORPH-32 Local / Qwen3.8 27B · morph32-3s**. Existing sessions can retain their previously selected model; use a new session or change its model explicitly.

The API supports ordinary chat and function-tool messages through MLX-LM's tokenizer-specific parser. A successful tool-call smoke test establishes protocol compatibility, not reliable autonomous coding on arbitrary tasks.
