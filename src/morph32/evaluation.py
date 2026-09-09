"""Fresh-process quality and synchronized end-to-end timing measurements."""

import hashlib
import json
import math
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from .generation import load_selected
from .inventory import inventory
from .metrics import memory, power_state
from .paths import write_json
from .sources import native_inventory


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.tokens < 1 or args.offset < 0 or args.runs < 1 or args.generation < 2:
        raise ValueError("Require positive tokens/runs, nonnegative offset, generation >= 2")
    mx.random.seed(0)
    start = time.monotonic()
    model, tokenizer = load_selected(args.checkpoint, args.baseline)
    report = dict(
        status="running",
        load_seconds=time.monotonic() - start,
        inventory=inventory(args.checkpoint)
        if args.checkpoint
        else native_inventory(args.baseline),
        device=mx.device_info(),
        power=power_state(),
        loaded_memory=memory(),
        resident_parameter_bytes=sum(v.nbytes for _, v in tree_flatten(model.parameters())),
    )
    text = args.text.read_text()
    ids = tokenizer.encode(text, add_special_tokens=False)[
        args.offset : args.offset + args.tokens + 1
    ]
    if len(ids) != args.tokens + 1:
        raise ValueError("Scoring range exceeds corpus length")
    report["corpus"] = dict(
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        token_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        offset=args.offset,
        tokens=args.tokens,
        chunk_size=128,
    )
    warm = model(mx.array(ids[:128])[None], cache=model.make_cache())
    mx.eval(warm)
    mx.synchronize()
    del warm
    mx.clear_cache()
    mx.reset_peak_memory()
    cache = model.make_cache()
    losses = []
    start = time.monotonic()
    total = 0.0
    for offset in range(0, args.tokens, 128):
        end = min(offset + 128, args.tokens)
        logits = model(mx.array(ids[offset:end])[None], cache=cache)
        loss = nn.losses.cross_entropy(
            logits.astype(mx.float32), mx.array(ids[offset + 1 : end + 1])[None], reduction="none"
        )
        value = mx.sum(loss)
        mx.eval(value, loss)
        mx.synchronize()
        total += value.item()
        losses.extend(loss.reshape(-1).tolist())
        del logits, loss, value
    nll = total / args.tokens
    report.update(
        quality=dict(
            nll=nll,
            perplexity=math.exp(nll),
            scored_tokens=args.tokens,
            seconds=time.monotonic() - start,
        ),
        token_nll=losses,
        quality_memory=memory(),
    )
    del cache
    write_json(args.output, report)
    if not args.quality_only:
        for response in stream_generate(
            model, tokenizer, prompt=ids[:-1], max_tokens=2, sampler=make_sampler(temp=0)
        ):
            pass
        mx.synchronize()
        report["generation_runs"] = []
        for _ in range(args.runs):
            mx.clear_cache()
            mx.reset_peak_memory()
            mx.synchronize()
            start = time.monotonic()
            first = None
            tokens, output = [], ""
            for response in stream_generate(
                model,
                tokenizer,
                prompt=ids[:-1],
                max_tokens=args.generation,
                sampler=make_sampler(temp=0),
            ):
                if first is None:
                    mx.synchronize()
                    first = time.monotonic()
                tokens.append(response.token)
                output += response.text
            mx.synchronize()
            report["generation_runs"].append(
                dict(
                    prompt_tps=response.prompt_tps,
                    generation_tps=response.generation_tps,
                    synchronized_wall_seconds=time.monotonic() - start,
                    time_to_first_token_seconds=first - start,
                    token_ids=tokens,
                    text=output,
                    memory=memory(),
                )
            )
            write_json(args.output, report)
    report["status"] = "complete"
    write_json(args.output, report)
    return report["quality"]
