"""Freeze official MMLU CSVs and a user-supplied held-out code corpus; never download data."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from morph32.paths import write_json
from morph32.research import sha256


def question(row, answer=False):
    if len(row) != 6 or row[-1] not in tuple("ABCD"):
        raise ValueError("Expected official MMLU question, four choices, answer")
    return (
        row[0]
        + "\n"
        + "\n".join(f"{letter}. {text}" for letter, text in zip("ABCD", row[1:5]))
        + "\nAnswer:"
        + (" " + row[5] + "\n\n" if answer else "")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="kind", required=True)
    mmlu = sub.add_parser("mmlu")
    mmlu.add_argument(
        "--source", type=Path, required=True, help="Official directory containing dev/ and test/"
    )
    mmlu.add_argument(
        "--per-subject", type=int, default=0, help="0 = full MMLU; otherwise at least 20"
    )
    code = sub.add_parser("code")
    code.add_argument("--source", type=Path, required=True, help="UTF-8 held-out code text")
    code.add_argument("--calibration", type=Path, default=Path("data/wikitext2-valid.txt"))
    code.add_argument("--dataset", required=True)
    code.add_argument("--split", required=True)
    code.add_argument("--revision", required=True)
    code.add_argument("--license", required=True)
    for command in (mmlu, code):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        raise FileExistsError(args.output)
    if args.kind == "mmlu":
        if args.per_subject != 0 and args.per_subject < 20:
            raise ValueError("Use full MMLU or at least 20 questions per subject")
        paths = sorted((args.source / "test").glob("*_test.csv"))
        if len(paths) != 57:
            raise ValueError("Expected all 57 official subjects")
        records, hashes = [], {}
        for path in paths:
            subject = path.name.removesuffix("_test.csv")
            dev = args.source / "dev" / f"{subject}_dev.csv"
            with dev.open(newline="") as stream:
                examples = list(csv.reader(stream))
            if len(examples) != 5:
                raise ValueError("Expected five dev examples")
            prefix = (
                f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
                + "".join(question(row, True) for row in examples)
            )
            with path.open(newline="") as stream:
                rows = list(csv.reader(stream))
            # Hash order gives a fixed subset, independent of model outcomes.
            selected = sorted(
                enumerate(rows),
                key=lambda pair: hashlib.sha256(json.dumps(pair).encode()).hexdigest(),
            )
            if args.per_subject:
                selected = selected[: args.per_subject]
            for index, row in sorted(selected):
                prompt = prefix + question(row)
                records.append(
                    dict(
                        id=f"{subject}/{index}",
                        subject=subject,
                        category="not aggregated",
                        prompt=prompt,
                        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                        answer="ABCD".index(row[5]),
                    )
                )
            hashes[str(path.relative_to(args.source))] = sha256(path)
            hashes[str(dev.relative_to(args.source))] = sha256(dev)
        write_json(
            args.output,
            dict(
                fewshot=5,
                records=records,
                source_sha256=hashes,
                selection="full" if not args.per_subject else "sha256 order per subject",
            ),
        )
    else:
        text = args.source.read_text()
        if not text.strip() or sha256(args.source) == sha256(args.calibration):
            raise ValueError("Code evaluation must use nonempty held-out data")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        write_json(
            args.output.with_suffix(".manifest.json"),
            dict(
                dataset=args.dataset,
                split=args.split,
                revision=args.revision,
                license=args.license,
                sha256=sha256(args.output),
                calibration_sha256=sha256(args.calibration),
                caveat="Distinct corpus files; pretraining contamination is not established",
            ),
        )


if __name__ == "__main__":
    main()
