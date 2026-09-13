"""Five-shot single-token choice likelihoods; no generated reasoning or sampling."""

import hashlib
import json
import os
import time

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models.cache import make_prompt_cache

from .generation import load_selected
from .metrics import memory
from .paths import write_json
from .research import provenance
from .statistics import accuracy_interval


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    start = time.monotonic()
    raw = args.subset.read_bytes()
    subset = json.loads(raw)
    if subset.get("fewshot") != 5 or not subset.get("records"):
        raise ValueError("Expected a frozen five-shot MMLU subset")
    mx.random.seed(0)
    model, tokenizer = load_selected(args.checkpoint, args.baseline)
    report = dict(
        status="running",
        provenance=provenance(args.checkpoint, args.baseline),
        requested_count=len(subset["records"]),
        subset_sha256=hashlib.sha256(raw).hexdigest(),
        records=[],
        load_seconds=time.monotonic() - start,
        device=mx.device_info(),
        resident_parameter_bytes=sum(v.nbytes for _, v in tree_flatten(model.parameters())),
        parameters=dict(
            fewshot=5, batch_size=1, chat_template=False, thinking=False, sampling=None
        ),
    )
    encoded = []
    for row in subset["records"]:
        if hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
            raise ValueError("Prompt hash mismatch")
        ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        choices = []
        for letter in "ABCD":
            together = tokenizer.encode(row["prompt"] + " " + letter, add_special_tokens=False)
            if together[:-1] != ids or len(together) != len(ids) + 1:
                raise ValueError("Choices must each be one token with an unchanged prompt prefix")
            choices.append(together[-1])
        encoded.append((row, ids, choices))
    warm = model(mx.array(encoded[0][1][:64])[None], cache=make_prompt_cache(model))
    mx.eval(warm)
    mx.synchronize()
    del warm
    benchmark_start = time.monotonic()
    for row, ids, choices in encoded:
        if time.monotonic() + 5 >= float(os.environ.get("MORPH32_DEADLINE", "inf")):
            report["status"] = "budget_exhausted"
            break
        mx.clear_cache()
        mx.reset_peak_memory()
        mx.synchronize()
        question_start = time.monotonic()
        cache = make_prompt_cache(model)
        logits = model(mx.array(ids)[None], cache=cache)[0, -1].astype(mx.float32)
        scores = mx.take(logits, mx.array(choices))
        logprobs, probabilities = scores - mx.logsumexp(logits), mx.softmax(scores)
        prediction = mx.argmax(scores)
        mx.eval(logprobs, probabilities, prediction)
        mx.synchronize()
        pred = prediction.item()
        report["records"].append(
            dict(
                id=row["id"],
                category=row["category"],
                subject=row["subject"],
                prompt_sha256=row["prompt_sha256"],
                token_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                prompt_tokens=len(ids),
                answer_token_ids=choices,
                predicted="ABCD"[pred],
                expected="ABCD"[row["answer"]],
                correct=pred == row["answer"],
                choice_logprobs=logprobs.tolist(),
                choice_probabilities=probabilities.tolist(),
                seconds=time.monotonic() - question_start,
                memory=memory(),
            )
        )
        report["benchmark_seconds"] = time.monotonic() - benchmark_start
        write_json(args.output, report)
        del logits, scores, logprobs, probabilities, prediction, cache
        print(f"Scored {len(report['records'])}/{len(encoded)}", flush=True)
    else:
        report["status"] = "complete"
    correct = sum(r["correct"] for r in report["records"])
    report.update(
        correct=correct,
        count=len(report["records"]),
        accuracy=correct / len(report["records"]) if report["records"] else None,
        process_seconds=time.monotonic() - start,
    )
    if report["count"]:
        report["accuracy_summary"] = accuracy_interval(correct, report["count"])
    write_json(args.output, report)
    return {k: report[k] for k in ("status", "correct", "count", "accuracy", "process_seconds")}
