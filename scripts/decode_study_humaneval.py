"""Recover exact HumanEval text from saved tokens, preserving raw inference evidence.

MLX-LM BPEStreamingDetokenizer removes an initial space. That display behavior
breaks Python indentation. Use the pinned tokenizer's complete token decoding,
then apply the same frozen top-level stop sequences. No model inference occurs.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

STOPS = ("\nclass", "\ndef", "\n#", "\nif", "\nprint")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path("models/Qwen3.8-27B-4bit"))
    args = parser.parse_args()
    source = json.loads(args.generation.read_text())
    assert source["status"] == "complete" and source["kind"] == "humaneval"
    output = args.generation.with_name(args.generation.stem + "-token-decoded.json")
    samples = output.with_suffix(".samples.jsonl")
    if output.exists() or samples.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    changed = []
    for row in source["records"]:
        original = row["text"]
        decoded = tokenizer.decode(
            row["token_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        boundaries = [decoded.find(stop) for stop in STOPS if stop in decoded]
        if boundaries:
            decoded = decoded[: min(boundaries)]
        row["text"] = decoded
        if decoded != original:
            changed.append(
                dict(
                    id=row["id"],
                    original_text_sha256=hashlib.sha256(original.encode()).hexdigest(),
                    decoded_text_sha256=hashlib.sha256(decoded.encode()).hexdigest(),
                    difference="one initial space"
                    if decoded == " " + original
                    else "token decoding",
                )
            )
    with samples.open("x") as stream:
        for row in source["records"]:
            stream.write(json.dumps(dict(task_id=row["id"], completion=row["text"])) + "\n")
    source.update(
        samples_file=samples.name,
        samples_sha256=sha(samples),
        grading_status="needs to run: official harness on token-decoded export",
        text_decoding_correction=dict(
            source_file=args.generation.name,
            source_sha256=sha(args.generation),
            utc=datetime.now(timezone.utc).isoformat(),
            reason="MLX-LM streaming decoder removes initial space; offline exact token decoding preserves Python indentation",
            inference_repeated=False,
            stop_sequences=list(STOPS),
            tokenizer_files={
                p.name: sha(p)
                for p in args.tokenizer.glob("*")
                if p.is_file()
                and (p.name.startswith("tokenizer") or p.name in ("vocab.json", "merges.txt"))
            },
            changed=changed,
        ),
    )
    with output.open("x") as stream:
        stream.write(json.dumps(source, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), changed=len(changed), total=len(source["records"]))))


if __name__ == "__main__":
    main()
