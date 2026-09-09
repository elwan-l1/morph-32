import hashlib
import json
from pathlib import Path

import pytest

from morph32.profiles import recipe

ROOT = Path(__file__).resolve().parents[1]


def test_profile_partition():
    projections = ("gate_proj", "up_proj", "down_proj")
    for profile, partial_count in [
        ("morph32-3s", 0),
        ("morph32-c", 96),
        ("three-seed", 0),
        ("compact", 96),
    ]:
        specs = [recipe(profile, layer, p) for layer in range(64) for p in projections]
        assert sum(s.get("partial", False) for s in specs) == partial_count
        assert all(s["tail"] == 0.5 and s["planes"] == 5 and s["seeds"] == 3 for s in specs)
    with pytest.raises(ValueError):
        recipe("morph32-c", 64, "up_proj")


def test_frozen_questions():
    subset = json.loads((ROOT / "data/mmlu-50.json").read_text())
    assert subset["fewshot"] == 5
    assert len(subset["records"]) == 50
    assert len({r["id"] for r in subset["records"]}) == 50
    for row in subset["records"]:
        assert hashlib.sha256(row["prompt"].encode()).hexdigest() == row["prompt_sha256"]
        assert row["answer"] in range(4)


def test_physical_inventory_includes_assets_and_rejects_false_count(tmp_path):
    import struct

    from morph32.inventory import inventory

    manifest = dict(
        format="morph32",
        version=1,
        standalone=True,
        logical_parameters=32,
        descriptors={
            "layer": {
                "kind": "morph32",
                "shape": [1, 32],
                "spec": recipe("morph32-3s", 0, "up_proj"),
            }
        },
    )
    header = {
        "__metadata__": {"morph_manifest": json.dumps(manifest)},
        "layer.words": {"dtype": "U32", "shape": [1, 1, 3], "data_offsets": [0, 12]},
        "layer.scales": {"dtype": "F16", "shape": [1, 1], "data_offsets": [12, 14]},
        "__assets__.config.json": {"dtype": "U8", "shape": [2], "data_offsets": [14, 16]},
    }
    path = tmp_path / "small.morph"

    def save():
        raw = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(14) + b"{}")

    save()
    report = inventory(path)
    assert report["file_bytes"] == path.stat().st_size
    assert report["physical_bpw"] == path.stat().st_size / 4
    assert report["embedded_asset_bytes"] == 2
    assert report["tensor_payload_bytes"] == 16
    manifest["logical_parameters"] = 64
    header["__metadata__"]["morph_manifest"] = json.dumps(manifest)
    save()
    with pytest.raises(ValueError, match="Independent logical count mismatch"):
        inventory(path)
