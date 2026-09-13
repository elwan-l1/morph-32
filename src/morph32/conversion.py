"""Build standalone MORPH profiles and opt-in fitting experiments from source snapshots."""

import hashlib
import json
import re
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .codec import fit, reference, unpack
from .inventory import inventory
from .metadata_kernels import unpack_correlated, unpack_metadata
from .packing import pack_correlated_metadata, pack_metadata
from .paths import write_json
from .profiles import normalize_profile, recipe, seed_search
from .research import sha256
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


def convert(
    source,
    bf16,
    calibration,
    output,
    report,
    *,
    profile="morph32-3s",
    compress_metadata=True,
    allow_other_model=False,
    optimization="profile",
    recipe_map=None,
):
    source, bf16, calibration, output, report = map(
        Path, (source, bf16, calibration, output, report)
    )
    profile = normalize_profile(profile)
    recipe(profile, 0, "up_proj")
    if optimization == "profile":
        optimization = seed_search(profile)
    temporary = output.with_suffix(".safetensors")
    if (
        output.suffix != ".morph"
        or len({output.resolve(), report.resolve(), temporary.resolve()}) != 3
    ):
        raise ValueError("Use distinct .morph, .json, and temporary destinations")
    if any(p.exists() for p in (output, temporary, report)):
        raise FileExistsError("Conversion destination already exists")
    source_inventory = native_inventory(source)
    if not allow_other_model and source_inventory["logical_parameters"] != LOGICAL_PARAMETERS:
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
        calibration_info.get("format") not in ("morph32_channel_importance", "morph32_tile_moments")
        or calibration_info.get("tokens", 0) < 1
        or not calibration_info.get("normalized")
    ):
        raise ValueError("Unsupported calibration artifact")
    config_hash = hashlib.sha256((source / "config.json").read_bytes()).hexdigest()
    if calibration_info.get("source_config_sha256", config_hash) != config_hash:
        raise ValueError("Calibration belongs to another source model")
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
        recipe_policy="explicit_map" if recipe_map is not None else "profile",
        recipe_map_sha256=hashlib.sha256(
            json.dumps(recipe_map, sort_keys=True).encode()
        ).hexdigest()
        if recipe_map is not None
        else None,
        logical_parameters=source_inventory["logical_parameters"],
        descriptors={},
        assets={},
        hashes={},
        source_model=str(source.resolve()) if allow_other_model else MODEL_ID,
        expected_source_revision=None if allow_other_model else MODEL_REVISION,
        bf16_model=str(bf16.resolve()) if allow_other_model else BF16_ID,
        expected_bf16_revision=None if allow_other_model else BF16_REVISION,
        calibration=calibration_info,
        calibration_file_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
        retained_scope="All source tensors and top-level non-weight assets",
        source_tensor_hashes={name: digest(value) for name, value in tensors.items()},
        compress_metadata=compress_metadata,
        source_config_sha256=hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
        encoder=dict(
            sweeps=2,
            starts=3,
            calibration_tokens=calibration_info["tokens"],
            optimization=optimization,
        ),
    )
    started = time.monotonic()
    progress = dict(status="converting", profile=profile, projections=[], retained_metadata=[])
    write_json(report, progress)
    # Untouched projections retain native int4 codes; only their metadata is packed.
    for key in list(tensors) if compress_metadata else []:
        if not key.endswith(".scales"):
            continue
        prefix = key[:-7]
        if (
            (not allow_other_model and not prefix.startswith("language_model."))
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
        before_bytes = scales.nbytes + biases.nbytes
        progress.setdefault("metadata_saved_bytes", 0)
        progress["metadata_saved_bytes"] += before_bytes - sum(v.nbytes for v in arrays.values())
        del tensors[key], tensors[prefix + ".biases"]
        tensors.update({prefix + "." + field: value for field, value in arrays.items()})
        manifest["descriptors"][prefix] = descriptor
        progress["retained_metadata"].append(prefix)
    prefixes = sorted(
        {
            key[:-7]
            for key in tensors
            if re.search(r"layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight$", key)
        }
    )
    if not prefixes or len(prefixes) % 3:
        raise ValueError("Expected complete gate/up/down MLP projections")
    if not allow_other_model and len(prefixes) != 192:
        raise ValueError("Expected 192 projections")
    if allow_other_model and profile == "morph32-c":
        raise ValueError("Compact layer selection is defined only for the original model")
    if recipe_map is not None and set(recipe_map) != set(prefixes):
        raise ValueError("Recipe map must cover every MLP projection exactly")
    if (optimization is not None or recipe_map is not None) and profile == "affine-3bit":
        raise ValueError("MORPH fitting options do not apply to affine-3bit")
    from .sources import tensor_index

    bf16_keys = tensor_index(bf16)
    for prefix in prefixes:
        layer = int(re.search(r"layers\.(\d+)\.", prefix)[1])
        projection = prefix.rsplit(".", 1)[-1]
        suffix = f"layers.{layer}.mlp.{projection}.weight"
        matches = [key for key in bf16_keys if key.endswith(suffix)]
        if len(matches) > 1 and prefix.startswith("language_model."):
            matches = [key for key in matches if key.startswith("model.language_model.")]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous or missing BF16 projection: {suffix}")
        weight = tensor_rows(bf16, matches[0])
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
        spec = recipe(profile, layer if not allow_other_model else 0, projection)
        if recipe_map is not None:
            spec = dict(recipe_map[prefix])
        parts = []
        for row in range(0, weight.shape[0], 512):
            if profile == "affine-3bit":
                from .scalar import fit as fit_scalar

                part = fit_scalar(weight[row : row + 512], channel, sweeps=2, starts=3)
            elif optimization is not None:
                from .optimization import fit_optimized

                opts = dict(optimization)
                use_covariance = opts.pop("covariance", False)
                if use_covariance and key + ".moments" not in importance:
                    raise ValueError("Covariance fitting requires tile-moment calibration")
                part = fit_optimized(
                    weight[row : row + 512],
                    spec=spec,
                    importance=channel,
                    moments=importance[key + ".moments"] if use_covariance else None,
                    **opts,
                )
            else:
                part = fit(weight[row : row + 512], **spec, importance=channel, sweeps=2, starts=3)
            mx.eval(part)
            parts.append(part)
        packed = [mx.concatenate([part[i] for part in parts]) for i in range(len(parts[0]))]
        words, scales = packed[:2]
        mx.eval(words, scales)
        indices = mx.array([0, weight.shape[0] // 3, 2 * weight.shape[0] // 3, weight.shape[0] - 1])
        if profile == "affine-3bit":
            from .scalar import reference as scalar_reference

            decoded = scalar_reference(words[indices], scales[indices], packed[2][indices])
            if not np.isfinite(decoded).all():
                raise ValueError("Nonfinite scalar reconstruction")
        else:
            decoded = unpack(words[indices], scales[indices], **spec)
            mx.eval(decoded)
            expected = reference(words[indices], scales[indices], **spec)
            sample_exact = np.array_equal(np.array(decoded), expected)
            if not sample_exact and not (
                "coefficients" in spec
                and np.allclose(np.array(decoded), expected, rtol=1e-6, atol=1e-8)
            ):
                raise ValueError("CPU/Metal reconstruction mismatch")
        native_bytes = sum(
            tensors[prefix + "." + field].nbytes for field in ("weight", "scales", "biases")
        )
        for field in ("weight", "scales", "biases"):
            del tensors[prefix + "." + field]
        tensors.update({prefix + ".words": words, prefix + ".scales": scales})
        if profile == "affine-3bit":
            tensors[prefix + ".biases"] = packed[2]
        manifest["descriptors"][prefix] = dict(
            kind="affine3" if profile == "affine-3bit" else "morph32",
            shape=list(weight.shape),
            spec=spec,
        )
        progress["projections"].append(
            dict(
                name=prefix,
                shape=list(weight.shape),
                seed_bytes=words.nbytes,
                scale_bytes=scales.nbytes,
                bias_bytes=packed[2].nbytes if profile == "affine-3bit" else 0,
                native_representation_bytes=native_bytes,
                representation_bytes=sum(v.nbytes for v in packed),
                decoder_sample_exact=None if profile == "affine-3bit" else bool(sample_exact),
            )
        )
        write_json(report, progress)
        del weight, native, words, scales, decoded, parts, part, packed
        mx.clear_cache()
    print(f"Encoded {len(prefixes)} projections", flush=True)
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
    progress["retained_tensor_hashes"] = {
        name: value
        for name, value in manifest["source_tensor_hashes"].items()
        if not any(name.startswith(prefix + ".") for prefix in prefixes)
    }
    progress["calibration_sha256"] = manifest["calibration_file_sha256"]
    progress["bf16_sha256"] = {
        path.name: sha256(path) for path in sorted(bf16.glob("model*.safetensors"))
    }
    progress["accounting"] = dict(
        procedural_saved_bytes=sum(
            p["native_representation_bytes"] - p["representation_bytes"]
            for p in progress["projections"]
        ),
        lossless_metadata_saved_bytes=progress.get("metadata_saved_bytes", 0),
        source_inventory=source_inventory,
        checkpoint_inventory=progress["inventory"],
        note="Payload reductions exclude container/asset differences; complete size is reported separately",
    )
    write_json(report, progress)
    return progress
