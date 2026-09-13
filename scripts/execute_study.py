"""Execute a frozen manifest with one-minute timers, durable logs and resumable status."""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from morph32.paths import write_json
from morph32.research import sha256

ENV = dict(
    MORPH32_DECODE_VARIANT="original",
    MORPH32_DECODE_VALUES="32",
    MORPH32_PREFILL_VARIANT="original",
    MORPH32_PREFILL_BLOCK_M="128",
    MORPH32_FUSE_GATE_UP="0",
    MORPH32_METADATA_VARIANT="packet",
    TOKENIZERS_PARALLELISM="false",
    PYTHONDONTWRITEBYTECODE="1",
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    root = args.manifest.parent
    manifest = json.loads(args.manifest.read_text())
    manifest["environment"] = ENV
    manifest["status"] = "running"
    manifest.setdefault("started_utc", utc())

    def save():
        write_json(args.manifest, manifest)

    def event(**row):
        row["utc"] = utc()
        with (root / "events.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")

    save()
    for step in manifest["steps"]:
        if step["status"] == "complete":
            continue
        if step["status"] not in ("needs to run", "planned"):
            raise ValueError("Inspect failed/interrupted step before resuming: " + step["name"])
        name = step["name"]
        log = root / (name + ".log")
        if log.exists():
            raise FileExistsError(log)
        env = {**os.environ, **ENV, **step["env"]}
        step.update(status="running", started_utc=utc(), log=log.name)
        start = time.monotonic()
        save()
        event(step=name, event="start", argv=step["argv"])
        print("START", name, flush=True)
        with log.open("x") as out:
            process = subprocess.Popen(step["argv"], env=env, stdout=out, stderr=subprocess.STDOUT)
            while True:
                try:
                    code = process.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - start
                    write_json(
                        root / "timer.json",
                        dict(
                            step=name,
                            pid=process.pid,
                            checked_utc=utc(),
                            elapsed_seconds=elapsed,
                            next_check_seconds=60,
                        ),
                    )
                    event(step=name, event="timer", elapsed_seconds=elapsed)
                    print("CHECK", name, round(elapsed), flush=True)
                    if elapsed > 21600:
                        process.terminate()
                        process.wait(timeout=30)
                        code = 124
                        break
        step.update(
            returncode=code,
            elapsed_seconds=time.monotonic() - start,
            finished_utc=utc(),
            log_sha256=sha256(log),
            status="complete" if code == 0 else "failed",
        )
        if code == 0 and "--output" in step["argv"]:
            output = Path(step["argv"][step["argv"].index("--output") + 1])
            if output.suffix == ".json":
                data = json.loads(output.read_text())
                if data.get("status") != "complete":
                    step["status"] = "failed"
                    code = 1
                step["output_sha256"] = sha256(output)
        event(
            step=name,
            event=step["status"],
            elapsed_seconds=step["elapsed_seconds"],
            returncode=code,
        )
        save()
        print(step["status"].upper(), name, round(step["elapsed_seconds"], 1), flush=True)
        if code:
            manifest["status"] = "incomplete"
            save()
            raise SystemExit(code)
    manifest.update(status="complete", finished_utc=utc())
    save()
    event(event="suite complete")


if __name__ == "__main__":
    main()
