"""Run the native model and both MORPH profiles with a 180-second cap each."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("models/Qwen3.8-27B-4bit"))
    parser.add_argument(
        "--morph32-3s",
        "--three-seed",
        dest="three_seed",
        type=Path,
        default=Path("checkpoints/morph32-3s.morph"),
    )
    parser.add_argument(
        "--morph32-c",
        "--compact",
        dest="compact",
        type=Path,
        default=Path("checkpoints/morph32-c.morph"),
    )
    parser.add_argument("--subset", type=Path, default=Path("data/mmlu-50.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected_questions = len(json.loads(args.subset.read_text()).get("records", []))
    if not expected_questions:
        parser.error("Empty subset")
    for path in (args.baseline, args.three_seed, args.compact):
        if not path.exists():
            parser.error(f"Missing model: {path}")
    args.output.mkdir(parents=True, exist_ok=False)
    statuses = []
    for name, flag, path in (
        ("native", "--baseline", args.baseline),
        ("morph32-3s", "--checkpoint", args.three_seed),
        ("morph32-c", "--checkpoint", args.compact),
    ):
        start = time.monotonic()
        command = [
            sys.executable,
            "-m",
            "morph32",
            "mmlu",
            flag,
            str(path),
            "--subset",
            str(args.subset),
            "--output",
            str(args.output / f"{name}.json"),
        ]
        with (args.output / f"{name}.log").open("w") as log:
            try:
                child = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=dict(os.environ, MORPH32_DEADLINE=str(start + 175)),
                    timeout=180,
                    check=False,
                )
                status = dict(name=name, returncode=child.returncode, timed_out=False)
            except subprocess.TimeoutExpired:
                status = dict(name=name, returncode=None, timed_out=True)
        status["wall_seconds"] = time.monotonic() - start
        result_path = args.output / f"{name}.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        status["completed_questions"] = len(result.get("records", []))
        status["expected_questions"] = expected_questions
        status["complete"] = (
            status["returncode"] == 0
            and result.get("status") == "complete"
            and status["completed_questions"] == expected_questions
        )
        statuses.append(status)
        (args.output / "processes.json").write_text(json.dumps(statuses, indent=2) + "\n")
        print(status, flush=True)
    if any(not s["complete"] for s in statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
