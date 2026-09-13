"""Command-line interface for calibration, conversion, inference, and evaluation."""

import argparse
import json
import os
from pathlib import Path

from morph32.profiles import PROFILES, normalize_profile


def main():
    parser = argparse.ArgumentParser(prog="morph32")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect", help="Inspect physical storage without loading weights"
    )
    inspect.add_argument("checkpoint", type=Path)
    calibrate = commands.add_parser(
        "calibrate", help="Capture channel importance and optional tile second moments"
    )
    calibrate.add_argument("--source", type=Path, required=True)
    calibrate.add_argument("--text", type=Path, default=Path("data/wikitext2-valid.txt"))
    calibrate.add_argument("--offset", type=int, default=32768)
    calibrate.add_argument("--tokens", type=int, default=128)
    calibrate.add_argument("--chunks", type=int, default=1)
    calibrate.add_argument("--covariance", action="store_true")
    calibrate.add_argument("--output", type=Path, required=True)
    convert = commands.add_parser(
        "convert", help="Fit a standalone checkpoint directly from Qwen weights"
    )
    convert.add_argument("--source", type=Path, required=True)
    convert.add_argument("--bf16", type=Path, required=True)
    convert.add_argument("--calibration", type=Path, required=True)
    convert.add_argument("--profile", type=normalize_profile, choices=PROFILES, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--report", type=Path, required=True)
    convert.add_argument(
        "--optimization",
        type=Path,
        help="Override profile seed search with JSON: sweeps, restarts, seed, covariance",
    )
    convert.add_argument(
        "--recipe-map", type=Path, help="JSON mapping every projection name to its MORPH spec"
    )
    convert.add_argument("--no-metadata-compression", action="store_true")
    convert.add_argument(
        "--allow-other-model",
        action="store_true",
        help="Experimental compatible gate/up/down architecture; validates tensor shapes",
    )
    serve = commands.add_parser("serve", help="Serve a standalone checkpoint on a local chat API")
    serve.add_argument("--checkpoint", type=Path, required=True)
    serve.add_argument("--port", type=int, default=8081)
    serve.add_argument("--model-id", default="qwen3.8-27b-morph32-3s")
    serve.add_argument("--max-tokens", type=int, default=4096)
    serve.add_argument("--context-length", type=int, default=32768)
    generate = commands.add_parser("generate", help="Generate from a standalone checkpoint")
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-tokens", type=int, default=128)
    generate.add_argument("--temperature", type=float, default=0)
    for name, help_text in [
        ("evaluate", "Measure NLL, memory, and generation"),
        ("mmlu", "Score a frozen multiple-choice subset"),
        ("workloads", "Run reasoning, coding, and long-context smoke tests"),
        ("capabilities", "Generate HumanEval samples or repeated retrieval trials"),
    ]:
        task = commands.add_parser(name, help=help_text)
        model = task.add_mutually_exclusive_group(required=True)
        model.add_argument("--checkpoint", type=Path)
        model.add_argument("--baseline", type=Path)
        task.add_argument("--output", type=Path, required=True)
        if name == "evaluate":
            task.add_argument("--text", type=Path, default=Path("data/wikitext2-test.txt"))
            task.add_argument("--tokens", type=int, default=4096)
            task.add_argument("--offset", type=int, default=0)
            task.add_argument("--generation", type=int, default=64)
            task.add_argument(
                "--timing-tokens",
                type=int,
                help="Timing prompt length; quality still uses --tokens",
            )
            task.add_argument("--runs", type=int, default=10)
            task.add_argument("--quality-only", action="store_true")
            task.add_argument("--domain", choices=("text", "code"), default="text")
            reference = task.add_mutually_exclusive_group()
            reference.add_argument("--export-reference", type=Path)
            reference.add_argument("--reference", type=Path)
        elif name == "mmlu":
            task.add_argument("--subset", type=Path, default=Path("data/mmlu-50.json"))
        elif name == "capabilities":
            task.add_argument("--kind", choices=("humaneval", "retrieval"), required=True)
            task.add_argument("--problems", type=Path)
            task.add_argument("--max-tokens", type=int, default=512)
            task.add_argument("--positions", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1])
            task.add_argument("--trials", type=int, default=10)
            task.add_argument("--filler-records", type=int, default=1100)
        else:
            task.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.command == "inspect":
        from .inventory import inventory

        print(json.dumps(inventory(args.checkpoint), indent=2))
        return
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    import mlx.core as mx

    from .paths import gpu_lock

    if "M5" not in mx.device_info().get("device_name", ""):
        parser.error("This release requires Apple M5: its prefill kernel uses M5 NAX")
    with gpu_lock():
        if args.command == "calibrate":
            from .calibration import capture

            result = capture(
                args.source,
                args.text,
                args.output,
                offset=args.offset,
                tokens=args.tokens,
                chunks=args.chunks,
                covariance=args.covariance,
            )
        elif args.command == "convert":
            from .conversion import convert

            result = convert(
                args.source,
                args.bf16,
                args.calibration,
                args.output,
                args.report,
                profile=args.profile,
                compress_metadata=not args.no_metadata_compression,
                allow_other_model=args.allow_other_model,
                optimization=json.loads(args.optimization.read_text())
                if args.optimization
                else "profile",
                recipe_map=json.loads(args.recipe_map.read_text()) if args.recipe_map else None,
            )
        elif args.command == "serve":
            from .server import serve

            result = serve(
                args.checkpoint,
                port=args.port,
                model_id=args.model_id,
                max_tokens=args.max_tokens,
                context_length=args.context_length,
            )
        elif args.command == "generate":
            from .generation import generate

            result = generate(args.checkpoint, args.prompt, args.max_tokens, args.temperature)
        elif args.command == "evaluate":
            from .evaluation import evaluate

            result = evaluate(args)
        elif args.command == "mmlu":
            from .mmlu import evaluate

            result = evaluate(args)
        elif args.command == "capabilities":
            from .capabilities import evaluate

            result = evaluate(args)
        else:
            from .workloads import evaluate

            result = evaluate(args)
        if result is not None:
            print(json.dumps(result, indent=2))
