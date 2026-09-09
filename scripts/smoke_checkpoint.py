"""Verify a standalone checkpoint against one archived MMLU prediction in a fresh process."""

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from morph32.inventory import inventory
from morph32.paths import gpu_lock, write_json
from morph32.profiles import PROFILES, normalize_profile
from morph32.runtime import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", type=normalize_profile, choices=PROFILES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    row = json.loads((root / "data/mmlu-50.json").read_text())["records"][0]
    archived = json.loads((root / f"evidence/{args.profile}-mmlu50.json").read_text())
    expected = archived["records"][0]
    with gpu_lock():
        model, tokenizer, manifest = load(args.checkpoint, verify=True)
        ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == expected["token_sha256"]
        cache = model.make_cache()
        logits = model(mx.array(ids)[None], cache=cache)[0, -1].astype(mx.float32)
        choices = mx.take(logits, mx.array(expected["answer_token_ids"])) - mx.logsumexp(logits)
        mx.eval(choices)
        assert choices.tolist() == expected["choice_logprobs"], (
            choices.tolist(),
            expected["choice_logprobs"],
        )
        next_token = mx.argmax(logits).item()
        step = model(mx.array([[next_token]]), cache=cache)
        mx.eval(step)
        mx.synchronize()
        assert mx.all(mx.isfinite(step)).item()
        resident = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
        assert resident == archived["resident_parameter_bytes"]
        report = dict(
            status="passed",
            device=mx.device_info(),
            integrity_verified=True,
            profile=args.profile,
            checkpoint_inventory=inventory(args.checkpoint),
            descriptors=len(manifest["descriptors"]),
            resident_parameter_bytes=resident,
            question_id=row["id"],
            choice_logprobs=choices.tolist(),
            archived_logprobs_exact=True,
            cached_single_token_execution_finite=True,
        )
        write_json(args.output, report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
