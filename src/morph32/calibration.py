"""Capture channel energy and optional tile moments from native-model activations."""

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from .sources import LOGICAL_PARAMETERS, MODEL_REVISION, native_inventory


class Capture(nn.Module):
    def __init__(self, inner, key, pending):
        super().__init__()
        self.inner = inner
        self._key = key
        self._pending = pending

    def __call__(self, x):
        self._pending[self._key] = mx.contiguous(x.reshape(-1, x.shape[-1]))
        return self.inner(x)


def capture(source, text, output, *, offset=32768, tokens=128, chunks=1, covariance=False):
    source, text, output = Path(source), Path(text), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if tokens < 1 or chunks < 1 or offset < 0 or output.suffix != ".safetensors":
        raise ValueError(
            "Use positive calibration tokens/chunks, nonnegative offset, and .safetensors output"
        )
    model, tokenizer = load(str(source))
    pending = {}
    for layer in range(len(model.layers)):
        for projection in ("up_proj", "down_proj"):
            inner = getattr(model.layers[layer].mlp, projection)
            setattr(
                model.layers[layer].mlp,
                projection,
                Capture(inner, f"layer{layer}.{projection}", pending),
            )
    corpus = text.read_text()
    all_ids = tokenizer.encode(corpus, add_special_tokens=False)[offset : offset + tokens * chunks]
    if len(all_ids) != tokens * chunks:
        raise ValueError("Calibration corpus is too short")
    sums, moments = {}, {}
    from .optimization import second_moments

    for chunk in range(chunks):
        ids = all_ids[chunk * tokens : (chunk + 1) * tokens]
        y = model(mx.array(ids)[None], cache=make_prompt_cache(model))
        mx.eval(y, pending)
        for key, x in pending.items():
            value = mx.mean(x.astype(mx.float32) ** 2, axis=0)
            sums[key] = sums.get(key, 0) + value
            if covariance:
                moments[key] = moments.get(key, 0) + second_moments(x)
        mx.eval(sums, moments)
        pending.clear()
    importance = {key: value / mx.maximum(mx.mean(value), 1e-20) for key, value in sums.items()}
    importance.update({key + ".moments": value / chunks for key, value in moments.items()})
    mx.eval(importance)
    manifest = dict(
        format="morph32_tile_moments" if covariance else "morph32_channel_importance",
        source_revision=MODEL_REVISION
        if native_inventory(source)["logical_parameters"] == LOGICAL_PARAMETERS
        else None,
        source_config_sha256=hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
        corpus_sha256=hashlib.sha256(corpus.encode()).hexdigest(),
        token_sha256=hashlib.sha256(json.dumps(all_ids).encode()).hexdigest(),
        offset=offset,
        tokens=tokens * chunks,
        chunk_tokens=tokens,
        chunks=chunks,
        normalized=True,
        context="fresh cache at selected offset",
        captured_arrays=len(importance),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(output), importance, metadata={"calibration_manifest": json.dumps(manifest)}
    )
    return manifest
