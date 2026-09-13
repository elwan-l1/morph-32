"""Run the pinned official harness entirely inside the restricted container."""

import json
import shutil
from pathlib import Path

from human_eval.evaluation import evaluate_functional_correctness

if __name__ == "__main__":
    shutil.copyfile("/input/samples.jsonl", "/work/samples.jsonl")
    metrics = evaluate_functional_correctness(
        "/work/samples.jsonl",
        k=[1],
        n_workers=2,
        timeout=3.0,
        problem_file="/harness/data/HumanEval.jsonl.gz",
    )
    result = Path("/work/samples.jsonl_results.jsonl")
    print(
        "MORPH_RESULT_JSON="
        + json.dumps(
            dict(metrics=metrics, rows=[json.loads(s) for s in result.read_text().splitlines()])
        )
    )
