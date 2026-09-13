"""Audit completed MMLU records without loading a model or changing an experiment."""

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(report_path, subset_path):
    report = json.loads(report_path.read_text())
    subset = json.loads(subset_path.read_text())
    rows = report["records"]
    expected = subset["records"]
    assert report["status"] == "complete"
    assert report["subset_sha256"] == digest(subset_path)
    assert report["count"] == report["requested_count"] == len(rows) == len(expected) == 1140
    assert len({r["id"] for r in rows}) == len(rows)
    assert Counter(r["subject"] for r in rows) == Counter(r["subject"] for r in expected)
    for actual, source in zip(rows, expected, strict=True):
        assert actual["id"] == source["id"]
        assert actual["prompt_sha256"] == source["prompt_sha256"]
        assert actual["expected"] == "ABCD"[source["answer"]]
        assert actual["correct"] == (actual["predicted"] == actual["expected"])
        scores = actual["choice_logprobs"]
        probabilities = actual["choice_probabilities"]
        assert len(scores) == len(probabilities) == 4
        assert all(math.isfinite(x) for x in scores + probabilities)
        assert actual["predicted"] == "ABCD"[max(range(4), key=scores.__getitem__)]
        assert math.isclose(sum(probabilities), 1, abs_tol=1e-6)
        softmax = [math.exp(x - max(scores)) for x in scores]
        assert all(
            math.isclose(p, x / sum(softmax), abs_tol=1e-6)
            for p, x in zip(probabilities, softmax, strict=True)
        )
    correct = sum(r["correct"] for r in rows)
    n, z = len(rows), 1.959963984540054
    proportion = correct / n
    denominator = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    summary = report["accuracy_summary"]
    assert report["correct"] == summary["correct"] == correct
    assert summary["count"] == n
    assert report["accuracy"] == summary["accuracy"] == proportion
    assert all(
        math.isclose(a, b, abs_tol=1e-12)
        for a, b in zip(summary["ci95"], [center - radius, center + radius], strict=True)
    )
    return dict(
        report=report_path.name,
        report_sha256=digest(report_path),
        subset_sha256=digest(subset_path),
        subjects=len(Counter(r["subject"] for r in rows)),
        accuracy_summary=summary,
        checks="coverage, prompt hashes, expected answers, predictions, probabilities, counts, Wilson interval",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--subset", type=Path, default=Path("data/research-20260911/mmlu-1140.json")
    )
    args = parser.parse_args()
    result = dict(
        status="complete",
        audited_utc=datetime.now(timezone.utc).isoformat(),
        **audit(args.report, args.subset),
    )
    output = args.report.with_name(args.report.stem + "-audit.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
