"""Frozen HumanEval completion export and positional retrieval; no generated code execution."""

import gzip
import hashlib
import json
import random
import time

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from .generation import load_selected
from .paths import write_json
from .research import provenance, sha256
from .statistics import accuracy_interval


def retrieval_tasks(positions, trials, filler_records):
    if trials < 1 or filler_records < 1 or not positions or any(not 0 <= p <= 1 for p in positions):
        raise ValueError("Invalid retrieval protocol")
    if len(set(positions)) != len(positions):
        raise ValueError("Duplicate insertion positions")
    rng = random.Random(21)
    for trial in range(trials):
        code = f"MAPLE-{rng.randrange(100000, 1000000)}"
        filler = [
            f"Archive entry {i}: routine weather record; no access code.\n"
            for i in range(filler_records)
        ]
        for position in positions:
            insertion = round(position * filler_records)
            prompt = (
                "".join(filler[:insertion])
                + f"The access code is {code}.\n"
                + "".join(filler[insertion:])
                + "What is the access code? Reply with the code only."
            )
            yield dict(
                id=f"retrieval-{trial}-{position:g}",
                prompt=prompt,
                expected=code,
                position=position,
                trial=trial,
                insertion_record=insertion,
            )


def human_eval_tasks(path):
    if path is None:
        raise ValueError("HumanEval requires --problems with the frozen official JSONL(.gz)")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if len(records) != 164 or len({r["task_id"] for r in records}) != 164:
        raise ValueError("Use the complete 164-task HumanEval dataset")
    return [dict(id=r["task_id"], prompt=r["prompt"]) for r in records]


def evaluate(args):
    samples_path = args.output.with_suffix(".samples.jsonl")
    if args.output.exists() or samples_path.exists():
        raise FileExistsError("Capability output exists")
    if args.max_tokens < 1:
        raise ValueError("Require positive generation budget")
    tasks = (
        human_eval_tasks(args.problems)
        if args.kind == "humaneval"
        else list(retrieval_tasks(args.positions, args.trials, args.filler_records))
    )
    mx.random.seed(0)
    model, tokenizer = load_selected(args.checkpoint, args.baseline)
    report = dict(
        status="running",
        kind=args.kind,
        records=[],
        requested_count=len(tasks),
        provenance=provenance(args.checkpoint, args.baseline),
        parameters=dict(
            max_tokens=args.max_tokens,
            temperature=0,
            seed=0,
            chat_template=args.kind != "humaneval",
            thinking=False,
            stop_sequences=["\nclass", "\ndef", "\n#", "\nif", "\nprint"]
            if args.kind == "humaneval"
            else [],
        ),
        problems_sha256=sha256(args.problems) if args.problems else None,
    )
    write_json(args.output, report)
    samples = []
    for task in tasks:
        # HumanEval uses raw prefix completion; never include tests/canonical_solution.
        prompt = (
            task["prompt"]
            if args.kind == "humaneval"
            else tokenizer.apply_chat_template(
                [{"role": "user", "content": task["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        tokens, text = [], ""
        mx.synchronize()
        start = time.monotonic()
        for response in stream_generate(
            model,
            tokenizer,
            prompt=ids,
            max_tokens=args.max_tokens,
            sampler=make_sampler(temp=0),
            prefill_step_size=512,
        ):
            tokens.append(response.token)
            text += response.text
            if args.kind == "humaneval":
                # Standard raw-completion boundary: stop before another top-level construct.
                boundaries = [
                    text.find(stop) for stop in ("\nclass", "\ndef", "\n#", "\nif", "\nprint")
                ]
                boundaries = [pos for pos in boundaries if pos >= 0]
                if boundaries:
                    text = text[: min(boundaries)]
                    break
        mx.synchronize()
        row = {k: v for k, v in task.items() if k != "prompt"}
        row.update(
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            token_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            prompt_tokens=len(ids),
            token_ids=tokens,
            text=text,
            seconds=time.monotonic() - start,
        )
        if args.kind == "retrieval":
            row["correct"] = text.strip() == task["expected"]
        else:
            samples.append(dict(task_id=task["id"], completion=text))
        report["records"].append(row)
        write_json(args.output, report)
    if args.kind == "humaneval":
        with samples_path.open("x") as stream:
            for sample in samples:
                stream.write(json.dumps(sample) + "\n")
        report.update(
            samples_file=samples_path.name,
            samples_sha256=sha256(samples_path),
            grading_status="needs to run: official HumanEval harness in an isolated sandbox",
        )
    else:
        report["by_position"] = {}
        for position in args.positions:
            rows = [r for r in report["records"] if r["position"] == position]
            report["by_position"][str(position)] = accuracy_interval(
                sum(r["correct"] for r in rows), len(rows)
            )
        # Cluster unit is an independently drawn code, with success at every position.
        successes = [
            all(r["correct"] for r in report["records"] if r["trial"] == trial)
            for trial in range(args.trials)
        ]
        report["accuracy_summary"] = accuracy_interval(sum(successes), len(successes))
        report["accuracy_unit"] = (
            "code trial: all insertion positions must succeed; synthetic task only"
        )
    report["status"] = "complete"
    write_json(args.output, report)
    return dict(status=report["status"], count=len(tasks), output=str(args.output))
