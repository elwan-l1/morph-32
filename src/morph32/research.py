"""Provenance and guards shared by the new research commands."""

import hashlib
import importlib.metadata
import json
import os
import platform
import struct
import subprocess
from pathlib import Path


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def checkpoint_manifest(path):
    with Path(path).open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
    return json.loads(header["__metadata__"]["morph_manifest"])


def check_calibration(checkpoint, text):
    if checkpoint:
        calibration = checkpoint_manifest(checkpoint)["calibration"]
        if calibration["corpus_sha256"] == hashlib.sha256(text.encode()).hexdigest():
            raise ValueError("Use a separate held-out corpus, not the calibration corpus")


def read_manifest(path):
    result = json.loads(Path(path).read_text())
    if result.get("status") != "complete":
        raise ValueError("Incomplete reference artifact")
    return result


def provenance(checkpoint=None, baseline=None):
    import mlx.core as mx

    from .metrics import power_state

    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "src/morph32").rglob("*.py"))
    files += sorted((root / "src/morph32/metal").glob("*.metal"))
    source_hashes = {str(p.relative_to(root)): sha256(p) for p in files}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    selected = Path(checkpoint or baseline)
    model_files = (
        [selected]
        if checkpoint
        else sorted(selected.glob("*.safetensors")) + sorted(selected.glob("*.json"))
    )
    return dict(
        model=str(selected.resolve()),
        model_sha256={p.name: sha256(p) for p in model_files},
        source_sha256=source_hashes,
        git_commit=commit,
        platform=platform.platform(),
        packages={name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "numpy")},
        device=mx.device_info(),
        power=power_state(),
        seed=0,
        decode_variant=os.environ.get("MORPH32_DECODE_VARIANT", "original"),
        prefill_variant=os.environ.get("MORPH32_PREFILL_VARIANT", "original"),
        decode_values=int(os.environ.get("MORPH32_DECODE_VALUES", "32")),
        fuse_gate_up=os.environ.get("MORPH32_FUSE_GATE_UP", "0"),
        metadata_variant=os.environ.get("MORPH32_METADATA_VARIANT", "packet"),
        prefill_block_m=int(os.environ.get("MORPH32_PREFILL_BLOCK_M", "128")),
    )
