"""Write a reproducible experiment command manifest; execute only with --execute.

Each inference command runs in a separate process. This runner never grades generated
code. HumanEval's official harness results must be imported separately.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from morph32.paths import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--code-text", type=Path, required=True)
    parser.add_argument("--mmlu", type=Path, required=True)
    parser.add_argument("--humaneval", type=Path, required=True)
    parser.add_argument("--label", required=True, help="Unique model label")
    parser.add_argument("--smaller-model", action="store_true")
    parser.add_argument(
        "--external", type=Path, help="Optional MLX-loadable recognized quantizer checkpoint"
    )
    parser.add_argument(
        "--external-description",
        help="Method, revisions, rate, retained tensors and runtime differences",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("paper/data/research"))
    args = parser.parse_args()
    if Path(args.label).name != args.label or args.label in (".", ".."):
        raise ValueError("Model label must be a single directory name")
    if args.external and not args.external_description:
        raise ValueError("External reference requires an explicit comparability description")
    root = args.output / args.label
    manifest_path = root / "commands.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    artifacts = Path("checkpoints/research") / args.label
    calibration = artifacts / "calibration.safetensors"
    steps = []

    def add(name, arguments, env=None):
        steps.append(
            dict(name=name, argv=[str(x) for x in arguments], env=env or {}, status="needs to run")
        )

    cli = [sys.executable, "-m", "morph32"]
    add("calibration", cli + ["calibrate", "--source", args.source, "--output", calibration])
    profiles = (
        ["morph32-3s", "independent-3s"]
        if args.smaller_model
        else ["morph32-3s", "morph32-c", "independent-3s", "affine-3bit", "morph32-3s-uncompressed"]
    )
    selected = {"native": ["--baseline", args.source]}
    for name in profiles:
        checkpoint = artifacts / f"{name}.morph"
        profile = name.removesuffix("-uncompressed")
        extra = (["--allow-other-model"] if args.smaller_model else []) + (
            ["--no-metadata-compression"] if name.endswith("-uncompressed") else []
        )
        add(
            "convert-" + name,
            cli
            + [
                "convert",
                "--source",
                args.source,
                "--bf16",
                args.bf16,
                "--calibration",
                calibration,
                "--profile",
                profile,
                "--output",
                checkpoint,
                "--report",
                root / f"{name}-conversion.json",
            ]
            + extra,
        )
        selected[name] = ["--checkpoint", checkpoint]
    if args.external:
        selected["external"] = ["--baseline", args.external]
    for domain, text, offset in [
        ("text", Path("data/wikitext2-test.txt"), 8192),
        ("code", args.code_text, 0),
    ]:
        reference = root / f"reference-{domain}"
        for name, model in selected.items():
            output = root / f"{name}-{domain}.json"
            ref = (
                ["--export-reference", reference]
                if name == "native"
                else ["--reference", reference]
            )
            add(
                f"{name}-{domain}",
                cli
                + ["evaluate"]
                + model
                + [
                    "--text",
                    text,
                    "--offset",
                    offset,
                    "--domain",
                    domain,
                    "--tokens",
                    4096,
                    "--runs",
                    10,
                    "--generation",
                    64,
                    "--output",
                    output,
                ]
                + ref
                + (["--quality-only"] if domain == "code" else []),
            )
    if not args.smaller_model:
        for name, model in selected.items():
            add(
                f"{name}-mmlu",
                cli
                + ["mmlu"]
                + model
                + ["--subset", args.mmlu, "--output", root / f"{name}-mmlu.json"],
            )
            for kind in ("humaneval", "retrieval"):
                add(
                    f"{name}-{kind}",
                    cli
                    + ["capabilities"]
                    + model
                    + ["--kind", kind, "--output", root / f"{name}-{kind}.json"]
                    + (["--problems", args.humaneval] if kind == "humaneval" else []),
                )
        checkpoint = artifacts / "morph32-3s.morph"
        for variant in ("original", "split"):
            add(
                "profile-" + variant,
                [
                    sys.executable,
                    "scripts/profile_generation.py",
                    "--checkpoint",
                    checkpoint,
                    "--variant",
                    variant,
                    "--output",
                    root / f"profile-{variant}.json",
                ],
            )
        add(
            "candidate-end-to-end",
            cli
            + [
                "evaluate",
                "--checkpoint",
                checkpoint,
                "--offset",
                8192,
                "--reference",
                root / "reference-text",
                "--runs",
                10,
                "--output",
                root / "morph32-3s-split-text.json",
            ],
            env={"MORPH32_DECODE_VARIANT": "split"},
        )
    manifest = dict(
        status="planned",
        model=args.label,
        smaller_model=args.smaller_model,
        external_description=args.external_description,
        steps=steps,
    )
    write_json(manifest_path, manifest)
    if args.execute:
        # Pin each comparison to the original kernel unless explicitly testing the candidate.
        for step in steps:
            env = {**os.environ, "MORPH32_DECODE_VARIANT": "original", **step["env"]}
            step["status"] = "running"
            write_json(manifest_path, manifest)
            try:
                subprocess.run(step["argv"], env=env, check=True)
            except subprocess.CalledProcessError:
                step["status"], manifest["status"] = "failed", "incomplete"
                write_json(manifest_path, manifest)
                raise
            step["status"] = "complete"
            write_json(manifest_path, manifest)
        manifest["status"] = "generation complete; HumanEval grading needs to run"
        write_json(manifest_path, manifest)
    print(
        json.dumps(dict(manifest=str(manifest_path), steps=len(steps), status=manifest["status"]))
    )


if __name__ == "__main__":
    main()
