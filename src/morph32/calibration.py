"""Capture channel importance from 128 real native-model activations."""

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from .sources import MODEL_REVISION


class Capture(nn.Module):
    def __init__(self, inner, key, pending):
        super().__init__()
        self.inner = inner
        self._key = key
        self._pending = pending

    def __call__(self, x):
        self._pending[self._key] = mx.contiguous(x.reshape(-1, x.shape[-1]))
        return self.inner(x)


def capture(source, text, output, *, offset=32768, tokens=128):
    source, text, output = Path(source), Path(text), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if tokens != 128 or offset < 0 or output.suffix != ".safetensors":
        raise ValueError("Use 128 calibration tokens, nonnegative offset, and .safetensors output")
    model, tokenizer = load(str(source))
    if len(model.layers) != 64:
        raise ValueError("Expected the 64-layer Qwen3.8-27B model")
    pending = {}
    for layer in range(64):
        for projection in ("up_proj", "down_proj"):
            inner = getattr(model.layers[layer].mlp, projection)
            setattr(
                model.layers[layer].mlp,
                projection,
                Capture(inner, f"layer{layer}.{projection}", pending),
            )
    corpus = text.read_text()
    ids = tokenizer.encode(corpus, add_special_tokens=False)[offset : offset + tokens]
    if len(ids) != tokens:
        raise ValueError("Calibration corpus is too short")
    y = model(mx.array(ids)[None], cache=model.make_cache())
    mx.eval(y, pending)
    importance = {}
    for key, x in pending.items():
        value = mx.mean(x.astype(mx.float32) ** 2, axis=0)
        value /= mx.mean(value)
        importance[key] = value
    mx.eval(importance)
    manifest = dict(
        format="morph32_channel_importance",
        source_revision=MODEL_REVISION,
        corpus_sha256=hashlib.sha256(corpus.encode()).hexdigest(),
        token_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        offset=offset,
        tokens=tokens,
        normalized=True,
        context="fresh cache at selected offset",
        captured_arrays=len(importance),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(output), importance, metadata={"calibration_manifest": json.dumps(manifest)}
    )
    return manifest
