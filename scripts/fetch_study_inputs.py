"""Download pinned study inputs and record every upstream revision and file hash."""

import csv
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

from morph32.research import sha256

ROOT = Path("paper/data/research/current-20260911")
DATA = Path("data/research-20260911")
DATA.mkdir(parents=True, exist_ok=True)
upstream = json.loads((ROOT / "upstream-discovery.json").read_text())
commands = []


def download(repo, files, dest, kind="model"):
    argv = [
        "hf",
        "download",
        repo,
        *files,
        "--revision",
        upstream[repo]["revision"],
        "--local-dir",
        str(dest),
    ]
    if kind == "dataset":
        argv += ["--repo-type", "dataset"]
    commands.append(argv)
    subprocess.run(argv, check=True)


for repo, dest in [
    ("Qwen/Qwen3-0.6B", "models/research-Qwen3-0.6B-BF16"),
    ("mlx-community/Qwen3-0.6B-4bit", "models/research-Qwen3-0.6B-4bit"),
]:
    download(repo, [], Path(dest))
for repo, files, dest in [
    (
        "cais/mmlu",
        ["all/dev-00000-of-00001.parquet", "all/test-00000-of-00001.parquet", "README.md"],
        DATA / "mmlu-hub",
    ),
    (
        "code-search-net/code_search_net",
        ["python/test-00000-of-00001.parquet", "README.md"],
        DATA / "code-hub",
    ),
]:
    download(repo, files, dest, "dataset")
for split in ["dev", "test"]:
    rows = pq.read_table(DATA / f"mmlu-hub/all/{split}-00000-of-00001.parquet").to_pylist()
    subjects = sorted({r["subject"] for r in rows})
    for subject in subjects:
        p = DATA / f"mmlu-official/{split}/{subject}_{split}.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as f:
            writer = csv.writer(f)
            for r in rows:
                if r["subject"] == subject:
                    writer.writerow([r["question"], *r["choices"], "ABCD"[r["answer"]]])
rows = pq.read_table(DATA / "code-hub/python/test-00000-of-00001.parquet").to_pylist()
field = "whole_func_string" if "whole_func_string" in rows[0] else "func_code_string"
# Dataset order, first 100 test functions, fixed before evaluating any model.
(DATA / "code-original.txt").write_text("\n\n".join(r[field] for r in rows[:100]))
revision = upstream["openai/human-eval"]["revision"]
for rel in [
    "data/HumanEval.jsonl.gz",
    "LICENSE",
    "human_eval/evaluation.py",
    "human_eval/execution.py",
    "human_eval/data.py",
    "human_eval/__init__.py",
]:
    url = f"https://raw.githubusercontent.com/openai/human-eval/{revision}/{rel}"
    dest = DATA / "human-eval" / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(urllib.request.urlopen(url, timeout=60).read())
subprocess.run(
    [
        sys.executable,
        "scripts/prepare_research_data.py",
        "mmlu",
        "--source",
        str(DATA / "mmlu-official"),
        "--per-subject",
        "20",
        "--output",
        str(DATA / "mmlu-1140.json"),
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        "scripts/prepare_research_data.py",
        "code",
        "--source",
        str(DATA / "code-original.txt"),
        "--output",
        str(DATA / "code-test.txt"),
        "--dataset",
        "code-search-net/code_search_net:python",
        "--split",
        "test:first100",
        "--revision",
        upstream["code-search-net/code_search_net"]["revision"],
        "--license",
        "See archived dataset README and upstream per-repository licenses",
    ],
    check=True,
)
(ROOT / "inputs.json").write_text(
    json.dumps(
        dict(
            status="complete",
            commands=commands,
            upstream=upstream,
            code_selection=dict(rows=100, field=field, order="original test parquet order"),
            sha256={
                str(p): sha256(p)
                for p in DATA.rglob("*")
                if p.is_file() and ".cache" not in p.parts
            },
        ),
        indent=2,
    )
    + "\n"
)
print("INPUTS COMPLETE", flush=True)
