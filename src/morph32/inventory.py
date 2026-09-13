"""Physical checkpoint accounting without importing MLX."""

import json
import math
from pathlib import Path


def inventory(path):
    import struct

    with Path(path).open("rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(length))
    manifest = json.loads(header.pop("__metadata__")["morph_manifest"])
    payload = sum(t["data_offsets"][1] - t["data_offsets"][0] for t in header.values())
    assets = sum(
        t["data_offsets"][1] - t["data_offsets"][0]
        for name, t in header.items()
        if name.startswith("__assets__.")
    )
    logical = 0
    for name, item in header.items():
        if name.startswith("__assets__."):
            continue
        prefix, suffix = name.rsplit(".", 1)
        d = manifest["descriptors"].get(prefix)
        if d and d["kind"] in ("morph32", "affine3"):
            if suffix == "words":
                n, k = d["shape"]
                seeds = (
                    3
                    if d["kind"] == "affine3"
                    else (5 if d["spec"].get("partial", False) else d["spec"]["seeds"])
                )
                if item["shape"] != [n, k // 32, seeds] or header[prefix + ".scales"]["shape"] != [
                    n,
                    k // (64 if d["kind"] == "affine3" else 32),
                ]:
                    raise ValueError("MORPH physical/logical shape mismatch")
                logical += n * k
            elif suffix not in ("scales", "biases"):
                logical += math.prod(item["shape"])
            continue
        if (
            suffix in ("scales", "biases")
            or d
            and suffix in ("fields", "bases", "index", "residual", "metadata")
        ):
            continue
        count = math.prod(item["shape"])
        if suffix == "weight" and (d or prefix + ".scales" in header):
            count *= 8
        logical += count
    if logical != manifest["logical_parameters"]:
        raise ValueError("Independent logical count mismatch")
    size = Path(path).stat().st_size
    return dict(
        file_bytes=size,
        logical_parameters=logical,
        physical_bpw=8 * size / logical,
        tensor_payload_bytes=payload,
        container_bytes=size - payload,
        embedded_asset_bytes=assets,
    )
