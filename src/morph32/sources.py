"""Pinned source revisions and bounded safetensors row reads."""

import json
import math
import struct
from pathlib import Path

MODEL_ID = "mlx-community/Qwen3.8-27B-4bit"
MODEL_REVISION = "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
BF16_ID = "Qwen/Qwen3.8-27B"
BF16_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
LOGICAL_PARAMETERS = 27356728560


def tensor_rows(folder, key, rows=None):
    import mlx.core as mx
    import numpy as np

    folder = Path(folder)
    index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    path = folder / index[key]
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        descriptor = json.loads(stream.read(size))[key]
    types = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "U32": np.uint32}
    array = np.memmap(
        path,
        dtype=types[descriptor["dtype"]],
        mode="r",
        offset=8 + size + descriptor["data_offsets"][0],
        shape=tuple(descriptor["shape"]),
    )
    result = mx.array(np.array(array if rows is None else array[rows], copy=True))
    return result.view(mx.bfloat16) if descriptor["dtype"] == "BF16" else result


def native_inventory(folder):
    folder = Path(folder)
    tensors = {}
    storage = 0
    for path in sorted(folder.glob("model*.safetensors")):
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(size))
        storage += path.stat().st_size
        tensors.update({k: v for k, v in header.items() if k != "__metadata__"})
    if not tensors:
        raise FileNotFoundError("No source model tensors")
    logical = 0
    for name, tensor in tensors.items():
        if name.endswith((".scales", ".biases")):
            continue
        count = math.prod(tensor["shape"])
        if name.endswith(".weight") and name[:-7] + ".scales" in tensors:
            count *= 8
        logical += count
    assets = sum(
        p.stat().st_size for p in folder.iterdir() if p.is_file() and p.suffix != ".safetensors"
    )
    return dict(
        logical_parameters=logical,
        file_bytes=storage + assets,
        physical_bpw=8 * (storage + assets) / logical,
        tensor_payload_bytes=sum(
            t["data_offsets"][1] - t["data_offsets"][0] for t in tensors.values()
        ),
        embedded_asset_bytes=assets,
    )
