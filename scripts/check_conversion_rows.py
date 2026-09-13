"""Refit four real BF16 rows from early, middle and late layers against a saved checkpoint."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open

from morph32.codec import fit
from morph32.optimization import fit_optimized
from morph32.paths import gpu_lock, write_json
from morph32.profiles import PROFILES, normalize_profile, recipe, seed_search
from morph32.sources import tensor_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", type=normalize_profile, choices=PROFILES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = []
    with gpu_lock(), safe_open(args.checkpoint, framework="numpy") as saved:
        manifest = json.loads(saved.metadata()["morph_manifest"])
        options = seed_search(args.profile)
        assert manifest["encoder"].get("optimization") == options
        for layer in range(64):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                prefix = f"language_model.model.layers.{layer}.mlp.{projection}"
                assert manifest["descriptors"][prefix]["spec"] == recipe(
                    args.profile, layer, projection
                )
        importance = mx.load(str(args.calibration))
        for layer in (0, 32, 63):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                prefix = f"language_model.model.layers.{layer}.mlp.{projection}"
                source = f"model.language_model.layers.{layer}.mlp.{projection}.weight"
                weight = tensor_rows(args.bf16, source, slice(0, 4))
                channel_key = f"layer{layer}." + (
                    "down_proj" if projection == "down_proj" else "up_proj"
                )
                spec = recipe(args.profile, layer, projection)
                if options is None:
                    words, scales = fit(
                        weight, importance=importance[channel_key], **spec, sweeps=2, starts=3
                    )
                else:
                    words, scales = fit_optimized(
                        weight, importance=importance[channel_key], spec=spec, **options
                    )
                mx.eval(words, scales)
                for name, actual in [("words", words), ("scales", scales)]:
                    np.testing.assert_array_equal(
                        np.array(actual), saved.get_slice(prefix + "." + name)[:4]
                    )
                records.append(
                    dict(layer=layer, projection=projection, rows=4, seed_and_scale_exact=True)
                )
        write_json(args.output, dict(status="passed", profile=args.profile, projections=records))
    print(
        f"{args.profile}: {len(records) * 4} real rows reproduce checkpoint seeds and scales exactly"
    )


if __name__ == "__main__":
    main()
