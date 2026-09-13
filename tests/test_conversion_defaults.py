"""Exercise three-plane conversion through actual checkpoint serialization."""

# ruff: noqa: E402
import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.metal

from morph32 import conversion
from morph32.optimization import fit_optimized
from morph32.profiles import recipe, seed_search


@pytest.mark.parametrize("profile", ["morph32-3s", "morph32-c"])
def test_conversion_defaults(tmp_path, monkeypatch, profile):
    source, bf16 = tmp_path / "source", tmp_path / "bf16"
    source.mkdir()
    bf16.mkdir()
    (source / "config.json").write_text(json.dumps({"quantization": {"bits": 4, "group_size": 64}}))
    mx.random.seed(43)
    weight = (mx.random.normal((4, 64)) * 0.05).astype(mx.bfloat16)
    packed = mx.quantize(weight, group_size=64, bits=4)
    native, original, channels = {}, {}, {}
    for layer in range(64):
        channels[f"layer{layer}.up_proj"] = mx.ones(64)
        channels[f"layer{layer}.down_proj"] = mx.ones(64)
        for projection in ("gate_proj", "up_proj", "down_proj"):
            suffix = f"layers.{layer}.mlp.{projection}"
            prefix = "language_model.model." + suffix
            native.update(
                {
                    prefix + "." + k: v
                    for k, v in zip(("weight", "scales", "biases"), packed, strict=True)
                }
            )
            original["model.language_model." + suffix + ".weight"] = weight
    mx.save_safetensors(str(source / "model.safetensors"), native)
    mx.save_safetensors(str(bf16 / "model.safetensors"), original)
    calibration = tmp_path / "calibration.safetensors"
    mx.save_safetensors(
        str(calibration),
        channels,
        metadata={
            "calibration_manifest": json.dumps(
                dict(format="morph32_channel_importance", tokens=128, normalized=True)
            )
        },
    )
    # Keep the real 192-projection layout, reducing only matrix sizes for this fixture.
    monkeypatch.setattr(conversion, "LOGICAL_PARAMETERS", 192 * weight.size)
    output = tmp_path / "model.morph"
    report = conversion.convert(
        source,
        bf16,
        calibration,
        output,
        tmp_path / "report.json",
        profile=profile,
    )
    arrays, meta = mx.load(str(output), format="safetensors", return_metadata=True)
    manifest = json.loads(meta["morph_manifest"])
    assert report["status"] == "complete"
    options = seed_search(profile)
    assert options == dict(sweeps=2, restarts=3, seed=2026)
    assert manifest["encoder"]["optimization"] == options
    expected = {}
    for prefix, descriptor in manifest["descriptors"].items():
        layer = int(prefix.split("layers.")[1].split(".")[0])
        spec = recipe(profile, layer, prefix.rsplit(".", 1)[1])
        assert descriptor["spec"] == spec
        assert spec["planes"] == 3
        partial = spec.get("partial", False)
        if partial not in expected:
            expected[partial] = fit_optimized(weight, spec=spec, **options)
        for field, value in zip(("words", "scales"), expected[partial], strict=True):
            np.testing.assert_array_equal(np.array(arrays[prefix + "." + field]), np.array(value))
    assert sum(d["spec"].get("partial", False) for d in manifest["descriptors"].values()) == (
        96 if profile == "morph32-c" else 0
    )
