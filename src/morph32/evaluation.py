"""Fresh-process quality and synchronized end-to-end timing measurements."""

import hashlib
import json
import math
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from .generation import load_selected
from .inventory import inventory
from .metrics import memory, power_state
from .paths import write_json
from .research import check_calibration, provenance, read_manifest
from .sources import native_inventory
from .statistics import timing_summary


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
        provenance=provenance(args.checkpoint, args.baseline),
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
    check_calibration(args.checkpoint, text)
    report["corpus"] = dict(
        domain=args.domain,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        token_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        offset=args.offset,
        tokens=args.tokens,
        chunk_size=128,
    )
    reference_manifest = None
    tokenizer_hash = hashlib.sha256(
        json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()
    ).hexdigest()
    report["tokenizer_sha256"] = tokenizer_hash
    if args.export_reference:
        args.export_reference.mkdir(parents=True, exist_ok=False)
    if args.reference:
        reference_manifest = read_manifest(args.reference / "manifest.json")
        if (
            reference_manifest["corpus"] != report["corpus"]
            or reference_manifest["tokenizer_sha256"] != tokenizer_hash
        ):
            raise ValueError("Reference corpus/tokenization differs")
    divergences, agreements, reference_chunks = [], [], []
    warm = model(mx.array(ids[:128])[None], cache=make_prompt_cache(model))
    mx.eval(warm)
    mx.synchronize()
    del warm
    mx.clear_cache()
    mx.reset_peak_memory()
    cache = make_prompt_cache(model)
    losses = []
    start = time.monotonic()
    total = 0.0
    for offset in range(0, args.tokens, 128):
        end = min(offset + 128, args.tokens)
        logits = model(mx.array(ids[offset:end])[None], cache=cache)
        loss = nn.losses.cross_entropy(
            logits.astype(mx.float32), mx.array(ids[offset + 1 : end + 1])[None], reduction="none"
        )
        if args.reference or args.export_reference:
            logp = logits.astype(mx.float32)
            logp = logp - mx.logsumexp(logp, axis=-1, keepdims=True)
            mx.eval(logp)
            if args.export_reference:
                path = args.export_reference / f"{offset:08d}.npy"
                np.save(path, np.array(logp))
                reference_chunks.append(
                    dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                )
            else:
                item = reference_manifest["chunks"][offset // 128]
                path = args.reference / item["file"]
                if (
                    path.parent.resolve() != args.reference.resolve()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]
                ):
                    raise ValueError("Reference chunk integrity mismatch")
                ref = mx.array(np.load(path, allow_pickle=False))
                if ref.shape != logp.shape:
                    raise ValueError("Reference vocabulary/logit shape differs")
                # Forward KL: reference || candidate, full vocabulary, teacher-forced context.
                kl = mx.sum(mx.exp(ref) * (ref - logp), axis=-1)
                agreement = mx.argmax(ref, axis=-1) == mx.argmax(logp, axis=-1)
                mx.eval(kl, agreement)
                divergences.extend(kl.reshape(-1).tolist())
                agreements.extend(agreement.reshape(-1).tolist())
                del ref, kl, agreement
            del logp
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
    if args.reference:
        report["quality"].update(
            delta_nll_nats_per_token=nll - reference_manifest["quality"]["nll"],
            mean_kl_reference_to_candidate=sum(divergences) / len(divergences),
            reference_token_agreement=sum(agreements) / len(agreements),
            reference_manifest_sha256=hashlib.sha256(
                (args.reference / "manifest.json").read_bytes()
            ).hexdigest(),
        )
        report.update(token_kl=divergences, token_agreement=agreements)
    if args.export_reference:
        write_json(
            args.export_reference / "manifest.json",
            dict(
                status="complete",
                corpus=report["corpus"],
                tokenizer_sha256=tokenizer_hash,
                quality=report["quality"],
                chunks=reference_chunks,
                provenance=report["provenance"],
            ),
        )
        report["quality"].update(
            delta_nll_nats_per_token=0.0,
            mean_kl_reference_to_candidate=0.0,
            reference_token_agreement=1.0,
        )
        report["quality"]["reference_manifest_sha256"] = hashlib.sha256(
            (args.export_reference / "manifest.json").read_bytes()
        ).hexdigest()
    report["quality_memory_scope"] = (
        "includes reference chunk when comparing distributions; use generation peaks for runtime comparison"
    )
    del cache
    write_json(args.output, report)
    if not args.quality_only:
        timing_tokens = getattr(args, "timing_tokens", None)
        if timing_tokens is None:
            timing_tokens = args.tokens
        if not 1 <= timing_tokens <= args.tokens:
            raise ValueError("Timing tokens must be between 1 and scoring tokens")
        prompt_ids = ids[:timing_tokens]
        report["timing_prompt"] = dict(
            tokens=timing_tokens,
            token_sha256=hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest(),
        )
        for response in stream_generate(
            model, tokenizer, prompt=prompt_ids, max_tokens=2, sampler=make_sampler(temp=0)
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
                prompt=prompt_ids,
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
                    prompt_tokens=response.prompt_tokens,
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
    if report.get("generation_runs"):
        report["timing"] = {
            name: timing_summary([run[name] for run in report["generation_runs"]])
            for name in ("prompt_tps", "generation_tps")
        }
    report["status"] = "complete"
    write_json(args.output, report)
    return report["quality"]
