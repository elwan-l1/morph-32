"""Validate official harness results against the exact saved greedy completions."""

import argparse
import json
from pathlib import Path

from morph32.paths import write_json
from morph32.research import sha256
from morph32.statistics import accuracy_interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--harness-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    generation = json.loads(args.generation.read_text())
    if generation["status"] != "complete" or generation["kind"] != "humaneval":
        raise ValueError("Incomplete HumanEval generation")
    samples = args.generation.parent / generation["samples_file"]
    if sha256(samples) != generation["samples_sha256"]:
        raise ValueError("Changed sample file")
    expected = {r["id"]: r["text"] for r in generation["records"]}
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    if len(rows) != len(expected) or {r["task_id"] for r in rows} != set(expected):
        raise ValueError("Missing, duplicate or extra scored tasks")
    for row in rows:
        if row["completion"] != expected[row["task_id"]] or type(row["passed"]) is not bool:
            raise ValueError("Harness results differ from saved completions")
    write_json(
        args.output,
        dict(
            status="complete",
            kind="humaneval",
            records=rows,
            accuracy_summary=accuracy_interval(sum(r["passed"] for r in rows), len(rows)),
            metric="greedy pass@1; one completion per task",
            harness_revision=args.harness_revision,
            generation_sha256=sha256(args.generation),
            results_sha256=sha256(args.results),
        ),
    )


if __name__ == "__main__":
    main()
