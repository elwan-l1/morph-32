"""Portable paths and an exclusive lock for local GPU commands."""

import contextlib
import fcntl
import os
from pathlib import Path


def cache_dir():
    path = Path(os.environ.get("MORPH32_CACHE", ".cache/morph32")).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextlib.contextmanager
def gpu_lock():
    with (cache_dir() / "gpu.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another MORPH-32 GPU command is running") from error
        yield


def write_json(path, value):
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)
