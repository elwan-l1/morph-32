"""Full-budget reasoning, coding, and long retrieval for a standalone MORPH model."""

import hashlib
import json
import time

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from .generation import load_selected
from .grading import grade
from .metrics import memory
from .tasks import TASKS


def evaluate(a):
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if a.output.exists():
        raise FileExistsError(a.output)
    model, tokenizer = load_selected(a.checkpoint, a.baseline)
    tasks = list(TASKS)
    filler = (
        "The archive contains routine weather records. Each entry is unrelated to the access code. "
    )
    tasks.append(
        (
            "long_context_retrieval",
            "Remember the access code: MAPLE-736281.\n"
            + filler * 1100
            + "\nWhat is the access code at the start? Reply with the code only.",
            "MAPLE-736281",
        )
    )
    report = dict(
        checkpoint=str(a.checkpoint or a.baseline),
        tasks=[],
        max_generated_tokens=a.max_tokens,
        loaded_memory=memory(),
    )
    for name, prompt, expected in tasks:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer.encode(formatted, add_special_tokens=False)
        mx.clear_cache()
        mx.reset_peak_memory()
        mx.synchronize()
        start = time.perf_counter()
        tokens = []
        text = ""
        for response in stream_generate(
            model,
            tokenizer,
            prompt=ids,
            max_tokens=a.max_tokens,
            sampler=make_sampler(temp=0),
            prefill_step_size=512,
        ):
            tokens.append(response.token)
            text += response.text
        mx.synchronize()
        report["tasks"].append(
            dict(
                name=name,
                prompt_tokens=len(ids),
                prompt_sha256=hashlib.sha256(formatted.encode()).hexdigest(),
                expected=expected,
                text=text,
                token_ids=tokens,
                seconds=time.perf_counter() - start,
                memory=memory(),
                prompt_tps=response.prompt_tps,
                generation_tps=response.generation_tps,
            )
        )
        a.output.write_text(json.dumps(report, indent=2) + "\n")
        print(name, len(tokens), text[-100:].replace("\n", " "), flush=True)
    grade(a.output)
    return {"output": str(a.output), "tasks": len(tasks)}
