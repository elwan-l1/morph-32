"""Verify archived records and rebuild the README comparison without loading a model."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / "evidence" / name).read_text())


def memory_peak(report, field):
    snapshots = [report["loaded_memory"], report["quality_memory"]]
    snapshots.extend(run["memory"] for run in report.get("generation_runs", []))
    return max(snapshot[field] for snapshot in snapshots)


def audit():
    provenance = read("provenance.json")
    for name, item in provenance["files"].items():
        assert (
            hashlib.sha256((ROOT / "evidence" / name).read_bytes()).hexdigest()
            == item["curated_report_sha256"]
        ), name
    baseline_ppl = read("native-quality-heldout.json")["quality"]["perplexity"]
    summary = {}
    for profile in ("native", "morph32-3s", "morph32-c"):
        speed, heldout = (
            read(f"{profile}-{suffix}.json") for suffix in ("quality-speed", "quality-heldout")
        )
        for report in (speed, heldout):
            quality = report["quality"]
            assert len(report["token_nll"]) == quality["scored_tokens"] == 4096
            assert math.isclose(sum(report["token_nll"]) / 4096, quality["nll"], abs_tol=2e-7)
            assert math.isclose(math.exp(quality["nll"]), quality["perplexity"], rel_tol=1e-12)
        source = speed["inventory"]
        size = source.get(
            "file_bytes", source.get("safetensors_bytes", 0) + source.get("auxiliary_bytes", 0)
        )
        bpw = 8 * size / source["logical_parameters"]
        assert math.isclose(
            bpw, source.get("physical_bpw", source.get("checkpoint_bpw")), rel_tol=1e-12
        )
        runs = speed["generation_runs"]
        assert len(runs) == 3
        result = dict(
            file_bytes=size,
            physical_bpw=bpw,
            resident_parameter_bytes=speed["resident_parameter_bytes"],
            peak_mlx_bytes=memory_peak(speed, "mlx_peak_bytes"),
            peak_process_physical_bytes=memory_peak(
                speed, "process_lifetime_peak_phys_footprint_bytes"
            ),
            prompt_tokens_per_second=statistics.median(r["prompt_tps"] for r in runs),
            generation_tokens_per_second=statistics.median(r["generation_tps"] for r in runs),
            first_4096_nll=speed["quality"]["nll"],
            first_4096_perplexity=speed["quality"]["perplexity"],
            heldout_nll=heldout["quality"]["nll"],
            heldout_perplexity=heldout["quality"]["perplexity"],
            heldout_perplexity_increase_percent=100
            * (heldout["quality"]["perplexity"] / baseline_ppl - 1),
        )
        count = 50
        record = read(f"{profile}-mmlu{count}.json")
        questions = json.loads((ROOT / f"data/mmlu-{count}.json").read_text())["records"]
        assert record["status"] == "complete" and len(record["records"]) == count
        assert [r["id"] for r in record["records"]] == [r["id"] for r in questions]
        assert [r["prompt_sha256"] for r in record["records"]] == [
            r["prompt_sha256"] for r in questions
        ]
        assert sum(r["correct"] for r in record["records"]) == record["correct"]
        assert math.isclose(record["correct"] / count, record["accuracy"])
        result[f"mmlu{count}_correct"] = record["correct"]
        summary[profile] = result
    for suffix in ("quality-speed", "quality-heldout"):
        reports = [read(f"{profile}-{suffix}.json") for profile in summary]
        assert len({r["corpus"]["token_sha256"] for r in reports}) == 1
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Regenerate evidence/summary.json")
    args = parser.parse_args()
    summary = audit()
    path = ROOT / "evidence/summary.json"
    if args.write:
        path.write_text(json.dumps(summary, indent=2) + "\n")
    else:
        assert json.loads(path.read_text()) == summary, "Summary differs from archived records"
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
