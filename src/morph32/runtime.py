"""Standalone MORPH checkpoint loader; no original model dependency at runtime."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.utils import _get_classes, load_tokenizer

from .codec import matvec_fast
from .metadata import CorrelatedLinear, MetadataLinear
from .paths import cache_dir
from .prefill import matmul


def digest(value):
    mx.eval(value)
    return hashlib.sha256(np.array(value.view(mx.uint8)).tobytes()).hexdigest()


class MorphLinear(nn.Module):
    def __init__(self, words, scales, spec, bias=None):
        super().__init__()
        self.words = words
        self.scales = scales
        self._spec = spec
        partial = spec.get("partial", False)
        if (
            words.dtype != (mx.uint16 if partial else mx.uint32)
            or words.ndim != 3
            or words.shape[2] != (5 if partial else spec["seeds"])
        ):
            raise ValueError("Invalid MORPH seed tensor")
        if scales.dtype != mx.float16 or scales.shape != words.shape[:2]:
            raise ValueError("Invalid MORPH scale tensor")
        if bias is not None:
            self.bias = bias

    def __call__(self, x):
        if x.size == x.shape[-1]:
            variant = os.environ.get("MORPH32_DECODE_VARIANT", "original")
            if variant not in ("original", "split"):
                raise ValueError("Unknown MORPH32_DECODE_VARIANT")
            y = matvec_fast(
                x,
                self.words,
                self.scales,
                **self._spec,
                values=int(os.environ.get("MORPH32_DECODE_VALUES", "32")),
                optimized=variant == "split",
            )
        else:
            y = matmul(x, self.words, self.scales, **self._spec)
        return y + self.bias if "bias" in self else y


def load(path, verify=True):
    arrays, metadata = mx.load(str(path), format="safetensors", return_metadata=True)
    manifest = json.loads(metadata["morph_manifest"])
    if manifest["version"] != 1 or manifest["format"] != "morph32" or not manifest["standalone"]:
        raise ValueError("Unsupported MORPH checkpoint")
    if set(arrays) != set(manifest["hashes"]):
        raise ValueError("MORPH tensor inventory mismatch")
    if verify:
        for name, value in arrays.items():
            if digest(value) != manifest["hashes"][name]:
                raise ValueError("MORPH integrity mismatch: " + name)
    with tempfile.TemporaryDirectory(prefix="assets-", dir=cache_dir()) as temporary:
        folder = Path(temporary)
        for name, key in manifest["assets"].items():
            if name in ("", ".", "..") or Path(name).name != name:
                raise ValueError("Invalid asset path")
            (folder / name).write_bytes(np.array(arrays.pop(key)).tobytes())
        config = json.loads((folder / "config.json").read_text())
        cls, args = _get_classes(config)
        model = cls(args.from_dict(config))
        weights = model.sanitize(arrays)
        descriptors = manifest["descriptors"]
        nn.quantize(
            model,
            group_size=64,
            bits=4,
            mode="affine",
            class_predicate=lambda p, m: (
                hasattr(m, "to_quantized") and (p in descriptors or p + ".scales" in weights)
            ),
        )
        replacements = []
        loaded = set()
        for name, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
            if name not in descriptors:
                continue
            d = descriptors[name]
            bias = weights.get(name + ".bias")
            if d["kind"] == "morph32":
                if not isinstance(module, nn.QuantizedLinear) or list(d["shape"]) != [
                    module.weight.shape[0],
                    module.weight.shape[1] * 8,
                ]:
                    raise ValueError("MORPH architecture shape mismatch")
                module = MorphLinear(
                    weights[name + ".words"], weights[name + ".scales"], d["spec"], bias
                )
            elif d["kind"] == "affine3":
                from .scalar import Affine3Linear

                if not isinstance(module, nn.QuantizedLinear) or list(d["shape"]) != [
                    module.weight.shape[0],
                    module.weight.shape[1] * 8,
                ]:
                    raise ValueError("Affine3 architecture shape mismatch")
                module = Affine3Linear(
                    weights[name + ".words"],
                    weights[name + ".scales"],
                    weights[name + ".biases"],
                    bias,
                )
            elif d["kind"] == "correlated_bf16_metadata":
                module = CorrelatedLinear(
                    *[
                        weights[name + "." + field]
                        for field in ("weight", "fields", "bases", "index", "residual")
                    ],
                    d["bits"],
                    bias,
                    decode="packet",
                )
            elif d["kind"] == "lossless_bf16_metadata":
                module = MetadataLinear(
                    weights[name + ".weight"],
                    weights[name + ".metadata"],
                    weights[name + ".bases"],
                    d["bits"],
                    bias,
                )
            else:
                raise ValueError("Unsupported retained descriptor")
            replacements.append((name, module))
            loaded.add(name)
        if loaded != set(descriptors):
            raise ValueError("Unloaded checkpoint descriptors")
        model.update_modules(tree_unflatten(replacements))
        model.load_weights(list(weights.items()), strict=True)
        fusion = os.environ.get("MORPH32_FUSE_GATE_UP", "0")
        if fusion not in ("0", "1"):
            raise ValueError("MORPH32_FUSE_GATE_UP must be 0 or 1")
        if fusion == "1":
            from .fused import FusedMorphMLP

            for layer in model.layers:
                if isinstance(layer.mlp.gate_proj, MorphLinear) and isinstance(
                    layer.mlp.up_proj, MorphLinear
                ):
                    layer.mlp = FusedMorphMLP(layer.mlp)
        model.eval()
        mx.eval(model.parameters())
        tokenizer = load_tokenizer(folder, eos_token_ids=config.get("eos_token_id"))
    return model, tokenizer, manifest
