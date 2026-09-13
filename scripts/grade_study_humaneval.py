"""Grade completed samples with a pinned official harness in a network-isolated container."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from morph32.paths import write_json
from morph32.research import sha256

ROOT = Path("paper/data/research/current-20260911")
HARNESS = Path("data/research-20260911/human-eval").resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation", type=Path)
    args = parser.parse_args()
    g = json.loads(args.generation.read_text())
    if g["status"] != "complete":
        raise ValueError("Incomplete generation")
    name = args.generation.stem
    folder = args.generation.parent
    output = folder / (name + "-scored.json")
    if output.exists():
        raise FileExistsError(output)
    samples = (folder / g["samples_file"]).resolve()
    if sha256(samples) != g["samples_sha256"]:
        raise ValueError("Samples changed")
    image = subprocess.check_output(
        ["docker", "image", "inspect", "morph-humaneval:20260911", "--format", "{{.Id}}"], text=True
    ).strip()
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--cpus",
        "2",
        "--memory",
        "2g",
        "--pids-limit",
        "256",
        "--user",
        "65534:65534",
        "--tmpfs",
        "/work:rw,noexec,size=256m,mode=1777",
        "--tmpfs",
        "/tmp:rw,noexec,size=256m,mode=1777",
        "-v",
        str(HARNESS) + ":/harness:ro",
        "-v",
        str(samples) + ":/input/samples.jsonl:ro",
        "-v",
        str((ROOT / "harness/grade.py").resolve()) + ":/runner/grade.py:ro",
        image,
        "python",
        "/runner/grade.py",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=1200)
    (folder / (name + "-harness.log")).write_text(
        completed.stdout + "\nSTDERR\n" + completed.stderr
    )
    completed.check_returncode()
    payloads = [
        line.removeprefix("MORPH_RESULT_JSON=")
        for line in completed.stdout.splitlines()
        if line.startswith("MORPH_RESULT_JSON=")
    ]
    if len(payloads) != 1:
        raise ValueError("Missing or ambiguous harness output")
    result = json.loads(payloads[0])
    results = folder / (name + "-official-results.jsonl")
    with results.open("x") as stream:
        for row in result["rows"]:
            stream.write(json.dumps(row) + "\n")
    revision = json.loads((ROOT / "upstream-discovery.json").read_text())["openai/human-eval"][
        "revision"
    ]
    subprocess.run(
        [
            sys.executable,
            "scripts/import_humaneval.py",
            "--generation",
            str(args.generation),
            "--results",
            str(results),
            "--harness-revision",
            revision,
            "--output",
            str(output),
        ],
        check=True,
    )
    write_json(
        folder / (name + "-harness-provenance.json"),
        dict(
            command=command,
            image_id=image,
            revision=revision,
            results_sha256=sha256(results),
            metrics=result["metrics"],
            network="none",
            read_only=True,
        ),
    )
    print(name, result["metrics"], flush=True)


if __name__ == "__main__":
    main()
