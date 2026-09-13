"""Attribute synchronized one-token MORPH projection time; instrumentation changes latency.

Use the largest measured projection family to decide whether the experimental
split-accumulator kernel merits adoption. End-to-end evaluate runs are authoritative.
"""

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text", type=Path, default=Path("data/wikitext2-test.txt"))
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--generation", type=int, default=64)
    parser.add_argument("--variant", choices=("original", "split"), default="original")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.tokens < 1 or args.generation < 2:
        raise ValueError("Invalid token budget")
    os.environ["MORPH32_DECODE_VARIANT"] = args.variant
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    from morph32.paths import gpu_lock, write_json
    from morph32.research import provenance
    from morph32.runtime import MorphLinear, load

    with gpu_lock():
        model, tokenizer, _ = load(args.checkpoint)
        ids = tokenizer.encode(args.text.read_text(), add_special_tokens=False)[: args.tokens]
        if len(ids) != args.tokens:
            raise ValueError("Insufficient prompt tokens")
        names = {
            id(module): name
            for name, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
            if isinstance(module, MorphLinear)
        }
        for _ in stream_generate(
            model, tokenizer, prompt=ids, max_tokens=2, sampler=make_sampler(temp=0)
        ):
            pass
        mx.synchronize()
        original = MorphLinear.__call__
        samples = defaultdict(list)

        def measured(module, x):
            if x.size != x.shape[-1]:
                return original(module, x)
            mx.eval(x)
            mx.synchronize()
            start = time.perf_counter()
            y = original(module, x)
            mx.eval(y)
            mx.synchronize()
            samples[names[id(module)]].append(time.perf_counter() - start)
            return y

        MorphLinear.__call__ = measured
        try:
            tokens = []
            for response in stream_generate(
                model,
                tokenizer,
                prompt=ids,
                max_tokens=args.generation,
                sampler=make_sampler(temp=0),
            ):
                tokens.append(response.token)
        finally:
            MorphLinear.__call__ = original
        ranked = sorted(
            (
                dict(name=name, seconds=sum(times), calls=len(times), samples=times)
                for name, times in samples.items()
            ),
            key=lambda row: row["seconds"],
            reverse=True,
        )
        write_json(
            args.output,
            dict(
                status="complete",
                provenance=provenance(args.checkpoint),
                variant=args.variant,
                projections=ranked,
                generated_token_ids=tokens,
                limitation="Synchronized per-projection wall time includes dispatch; not instruction-level profiling or end-to-end throughput",
            ),
        )


if __name__ == "__main__":
    main()
