"""Deterministic non-thinking generation by default."""

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from .runtime import load


def generate(checkpoint, prompt, max_tokens=128, temperature=0):
    if max_tokens <= 0 or temperature < 0:
        raise ValueError("Require positive max_tokens and nonnegative temperature")
    model, tokenizer, _ = load(checkpoint)
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    response = None
    for response in stream_generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temperature),
    ):
        print(response.text, end="", flush=True)
    mx.synchronize()
    print()
    if response is not None:
        return dict(prompt_tps=response.prompt_tps, generation_tps=response.generation_tps)


def load_selected(checkpoint=None, baseline=None):
    if (checkpoint is None) == (baseline is None):
        raise ValueError("Choose exactly one checkpoint or baseline directory")
    if baseline is not None:
        from mlx_lm import load as native_load

        return native_load(str(baseline))
    model, tokenizer, _ = load(checkpoint)
    return model, tokenizer
