"""Record a separately labelled llama.cpp GGUF reference; never pool it with MLX."""

import json
import re
import subprocess
import time
from pathlib import Path

from morph32.paths import write_json
from morph32.research import sha256

ROOT = Path("paper/data/research/current-20260911/external")
BIN = Path("models/research-tools/llama-b10809/llama-b10809")
MODEL = Path("models/research-Qwen3-0.6B-GGUF/Qwen3-0.6B-Q3_K_S.gguf")


def main():
    ROOT.mkdir(exist_ok=True)
    if (ROOT / "results.json").exists():
        raise FileExistsError(ROOT / "results.json")
    commands = []
    jobs = [
        (
            "throughput",
            [
                str(BIN / "llama-bench"),
                "-m",
                str(MODEL),
                "-p",
                "4096",
                "-n",
                "64",
                "-r",
                "10",
                "-o",
                "json",
                "-ngl",
                "99",
            ],
        ),
        (
            "perplexity",
            [
                str(BIN / "llama-perplexity"),
                "-m",
                str(MODEL),
                "-f",
                "data/wikitext2-test.txt",
                "-c",
                "4096",
                "--chunks",
                "1",
                "-ngl",
                "99",
            ],
        ),
    ]
    for name, cmd in jobs:
        start = time.monotonic()
        with (
            (ROOT / (name + ".stdout")).open("x") as out,
            (ROOT / (name + ".stderr")).open("x") as err,
        ):
            p = subprocess.Popen(cmd, stdout=out, stderr=err)
            while True:
                try:
                    code = p.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    write_json(
                        ROOT / "timer.json",
                        dict(
                            job=name,
                            elapsed_seconds=time.monotonic() - start,
                            next_check_seconds=60,
                        ),
                    )
                    if time.monotonic() - start > 1800:
                        p.terminate()
                        raise TimeoutError(name)
        commands.append(
            dict(name=name, argv=cmd, returncode=code, seconds=time.monotonic() - start)
        )
        write_json(ROOT / "commands.json", commands)
        if code:
            raise RuntimeError(name)
    throughput = json.loads((ROOT / "throughput.stdout").read_text())
    text = (
        (ROOT / "perplexity.stdout").read_text() + "\n" + (ROOT / "perplexity.stderr").read_text()
    )
    matches = re.findall(r"Final estimate:\s*PPL\s*=\s*([\d.]+)", text)
    if not matches:
        raise ValueError("Missing final perplexity estimate")
    write_json(
        ROOT / "results.json",
        dict(
            status="complete",
            model="unsloth/Qwen3-0.6B-GGUF",
            revision="50968a4468ef4233ed78cd7c3de230dd1d61a56b",
            format="Q3_K_S",
            file_bytes=MODEL.stat().st_size,
            model_sha256=sha256(MODEL),
            runtime="llama.cpp b10809, official macOS arm64 release",
            throughput=throughput,
            perplexity=float(matches[-1]),
            corpus_sha256=sha256("data/wikitext2-test.txt"),
            comparison_scope="Separate-runtime descriptive reference only. The full GGUF quantizes a different tensor scope than MORPH MLP-only conversion; llama-bench uses synthetic tokens and llama-perplexity scores its standard first-chunk window. These values are not paired with MLX quality or throughput and are excluded from matched-cohort graphics. No superiority claim follows.",
            commands=commands,
            source_sha256={str(p): sha256(p) for p in ROOT.iterdir() if p.is_file()},
        ),
    )


if __name__ == "__main__":
    main()
