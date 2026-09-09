"""Build either standalone MORPH profile directly from the two Qwen snapshots."""

import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .codec import fit, reference, unpack
from .inventory import inventory
from .metadata_kernels import unpack_correlated, unpack_metadata
from .packing import pack_correlated_metadata, pack_metadata
from .paths import write_json
from .profiles import normalize_profile, recipe
from .runtime import digest
from .sources import (
    BF16_ID,
    BF16_REVISION,
    LOGICAL_PARAMETERS,
    MODEL_ID,
    MODEL_REVISION,
    native_inventory,
    tensor_rows,
)


def encode_metadata(scales, biases):
    """Return a smaller, exactly reversible metadata representation, or None."""
    try:
        fields, bases, index, residual, bits = pack_correlated_metadata(scales, biases)
        arrays = dict(fields=fields, bases=bases, index=index, residual=residual)
        restored = unpack_correlated(fields, bases, index, residual, bits, scales.shape)
        kind = "correlated_bf16_metadata"
    except ValueError:
        try:
            packed, bases, bits = pack_metadata(scales, biases)
        except ValueError:
            return None
        arrays = dict(metadata=packed, bases=bases)
        restored = unpack_metadata(packed, bases, bits, scales.shape)
        kind = "lossless_bf16_metadata"
    if sum(v.nbytes for v in arrays.values()) >= scales.nbytes + biases.nbytes:
        return None
    if not all(
        mx.array_equal(a, b).item() for a, b in zip(restored, (scales, biases), strict=True)
    ):
        raise ValueError("Metadata recovery mismatch")
    return arrays, dict(kind=kind, bits=bits, group_size=64)


def convert(source, bf16, calibration, output, report, *, profile="morph32-3s"):
    source, bf16, calibration, output, report = map(
        Path, (source, bf16, calibration, output, report)
    )
    profile = normalize_profile(profile)
    recipe(profile, 0, "up_proj")
    temporary = output.with_suffix(".safetensors")
    if (
        output.suffix != ".morph"
        or len({output.resolve(), report.resolve(), temporary.resolve()}) != 3
    ):
        raise ValueError("Use distinct .morph, .json, and temporary destinations")
    if any(p.exists() for p in (output, temporary, report)):
        raise FileExistsError("Conversion destination already exists")
    source_inventory = native_inventory(source)
    if source_inventory["logical_parameters"] != LOGICAL_PARAMETERS:
        raise ValueError("Source logical parameter count differs from Qwen3.8-27B")
    config = json.loads((source / "config.json").read_text())
    quant = config.get("quantization", config.get("quantization_config", {}))
    if (
        quant.get("bits") != 4
        or quant.get("group_size") != 64
        or quant.get("mode", "affine") != "affine"
    ):
        raise ValueError("Source must use affine int4 with group size 64")
    importance, meta = mx.load(str(calibration), return_metadata=True)
    calibration_info = json.loads(meta["calibration_manifest"])
    if (
        calibration_info.get("format") != "morph32_channel_importance"
        or calibration_info.get("tokens") != 128
        or not calibration_info.get("normalized")
    ):
        raise ValueError("Unsupported calibration artifact")
    tensors = {}
    for shard in sorted(source.glob("model*.safetensors")):
        part = mx.load(str(shard))
        if tensors.keys() & part.keys():
            raise ValueError("Duplicate tensor names across source shards")
        tensors.update(part)
    del part
    manifest = dict(
        version=1,
        format="morph32",
        standalone=True,
        profile=profile,
        logical_parameters=LOGICAL_PARAMETERS,
        descriptors={},
        assets={},
        hashes={},
        source_model=MODEL_ID,
        expected_source_revision=MODEL_REVISION,
        bf16_model=BF16_ID,
        expected_bf16_revision=BF16_REVISION,
        calibration=calibration_info,
        calibration_file_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
        retained_scope="All source tensors and top-level non-weight assets",
        encoder=dict(sweeps=2, starts=3, calibration_tokens=128),
    )
    started = time.monotonic()
    progress = dict(status="converting", profile=profile, projections=[], retained_metadata=[])
    write_json(report, progress)
    # Untouched projections retain native int4 codes; only their metadata is packed.
    for key in list(tensors):
        if not key.endswith(".scales"):
            continue
        prefix = key[:-7]
        if (
            not prefix.startswith("language_model.")
            or "embed_tokens" in prefix
            or ".mlp." in prefix
        ):
            continue
        scales, biases = tensors[key], tensors.get(prefix + ".biases")
        if biases is None or scales.dtype != mx.bfloat16:
            continue
        encoded = encode_metadata(scales, biases)
        if encoded is None:
            continue
        arrays, descriptor = encoded
        del tensors[key], tensors[prefix + ".biases"]
        tensors.update({prefix + "." + field: value for field, value in arrays.items()})
        manifest["descriptors"][prefix] = descriptor
        progress["retained_metadata"].append(prefix)
    for layer in range(64):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"language_model.model.layers.{layer}.mlp.{projection}"
            weight = tensor_rows(
                bf16, f"model.language_model.layers.{layer}.mlp.{projection}.weight"
            )
            native = tensors[prefix + ".weight"]
            if tuple(weight.shape) != (native.shape[0], native.shape[1] * 8):
                raise ValueError("BF16 and affine projection shapes differ")
            key = f"layer{layer}." + ("down_proj" if projection == "down_proj" else "up_proj")
            channel = importance[key]
            if (
                channel.shape != (weight.shape[1],)
                or not mx.all(mx.isfinite(channel) & (channel >= 0)).item()
                or mx.mean(channel).item() <= 0
            ):
                raise ValueError("Invalid channel importance")
            spec = recipe(profile, layer, projection)
            parts = []
            for row in range(0, weight.shape[0], 512):
                part = fit(weight[row : row + 512], **spec, importance=channel, sweeps=2, starts=3)
                mx.eval(part)
                parts.append(part)
            words, scales = [mx.concatenate([part[i] for part in parts]) for i in range(2)]
            mx.eval(words, scales)
            indices = mx.array(
                [0, weight.shape[0] // 3, 2 * weight.shape[0] // 3, weight.shape[0] - 1]
            )
            decoded = unpack(words[indices], scales[indices], **spec)
            mx.eval(decoded)
            if not np.array_equal(
                np.array(decoded), reference(words[indices], scales[indices], **spec)
            ):
                raise ValueError("CPU/Metal reconstruction mismatch")
            for field in ("weight", "scales", "biases"):
                del tensors[prefix + "." + field]
            tensors.update({prefix + ".words": words, prefix + ".scales": scales})
            manifest["descriptors"][prefix] = dict(
                kind="morph32", shape=list(weight.shape), spec=spec
            )
            progress["projections"].append(
                dict(
                    name=prefix,
                    shape=list(weight.shape),
                    seed_bytes=words.nbytes,
                    scale_bytes=scales.nbytes,
                    decoder_sample_exact=True,
                )
            )
            write_json(report, progress)
            del weight, native, words, scales, decoded, parts, part
            mx.clear_cache()
        print(f"Encoded layer {layer + 1}/64", flush=True)
    for asset in sorted(source.iterdir()):
        if asset.is_file() and asset.suffix != ".safetensors":
            key = "__assets__." + asset.name
            tensors[key] = mx.array(np.frombuffer(asset.read_bytes(), dtype=np.uint8))
            manifest["assets"][asset.name] = key
    manifest["hashes"] = {name: digest(value) for name, value in tensors.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(temporary),
        tensors,
        metadata={"morph_manifest": json.dumps(manifest, separators=(",", ":"))},
    )
    temporary.rename(output)
    progress.update(
        status="complete", seconds=time.monotonic() - started, inventory=inventory(output)
    )
    write_json(report, progress)
    return progress
