"""Validate archived research records and derive every new plotted value; no model imports."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from morph32.paths import write_json
from morph32.statistics import accuracy_interval, timing_summary

PROFILES = (
    "native",
    "morph32-3s",
    "morph32-c",
    "independent-3s",
    "published-3bit",
    "morph32-3s-uncompressed",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=2e-7)


def summarize(folder, smaller=False):
    folder = Path(folder).resolve()
    paper_data = Path(__file__).resolve().parent / "data"
    require(
        folder.is_relative_to(paper_data), "Raw measurements must be archived under paper/data/"
    )
    profiles = ("native", "morph32-3s", "independent-3s") if smaller else PROFILES
    hashes, summary, matched = {}, {}, {}

    def read(name):
        path = folder / name
        raw = path.read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        value = json.loads(raw)
        require(value.get("status") == "complete", f"Incomplete record: {name}")
        return value

    for profile in profiles:
        row = {}
        for domain in ("text", "code"):
            report = read(f"{profile}-{domain}.json")
            q = report["quality"]
            losses = report["token_nll"]
            require(
                len(losses) == q["scored_tokens"] == report["corpus"]["tokens"],
                "Scored count mismatch",
            )
            require(close(sum(losses) / len(losses), q["nll"]), "NLL mismatch")
            require(close(math.exp(q["nll"]), q["perplexity"]), "Perplexity mismatch")
            require(q["mean_kl_reference_to_candidate"] >= -1e-5, "Invalid KL")
            if profile != "native":
                require(
                    len(report["token_kl"]) == len(losses) == len(report["token_agreement"]),
                    "Pairing count mismatch",
                )
                require(
                    close(
                        sum(report["token_kl"]) / len(losses), q["mean_kl_reference_to_candidate"]
                    ),
                    "KL mismatch",
                )
                require(
                    close(
                        sum(report["token_agreement"]) / len(losses), q["reference_token_agreement"]
                    ),
                    "Agreement mismatch",
                )
                require(
                    close(
                        q["nll"] - summary["native"][domain]["nll"], q["delta_nll_nats_per_token"]
                    ),
                    "NLL delta mismatch",
                )
            signature = dict(
                corpus=report["corpus"],
                tokenizer=report["tokenizer_sha256"],
                device=report["provenance"]["device"],
                packages=report["provenance"]["packages"],
                platform=report["provenance"]["platform"],
                source=report["provenance"]["source_sha256"],
                reference=q["reference_manifest_sha256"],
                variant=report["provenance"]["decode_variant"],
            )
            if profile == "native":
                matched[domain] = signature
            require(signature == matched[domain], f"Unmatched protocol: {profile}/{domain}")
            row[domain] = q
            if domain == "text":
                runs = report["generation_runs"]
                require(len(runs) >= 10, "At least ten timing repetitions are required")
                lengths = [(len(run["token_ids"])) for run in runs]
                require(min(lengths) == max(lengths) == 64, "Timing generation budget differs")
                row.update(
                    file_bytes=report["inventory"]["file_bytes"],
                    peak_mlx_bytes=max(run["memory"]["mlx_peak_bytes"] for run in runs),
                    resident_parameter_bytes=report["resident_parameter_bytes"],
                    generation=timing_summary([run["generation_tps"] for run in runs]),
                    prompt=timing_summary([run["prompt_tps"] for run in runs]),
                )
        if profile not in ("native", "published-3bit"):
            conversion = read(f"{profile}-conversion.json")
            signature = (
                conversion["retained_tensor_hashes"],
                conversion["calibration_sha256"],
                conversion["bf16_sha256"],
            )
            if profile == "morph32-3s":
                matched["retained"] = signature
            require(signature == matched["retained"], "Retained tensors or calibration differ")
            row["representation_bytes"] = sum(
                p["representation_bytes"] for p in conversion["projections"]
            )
            row["native_representation_bytes"] = sum(
                p["native_representation_bytes"] for p in conversion["projections"]
            )
            row["metadata_saved_bytes"] = conversion.get("metadata_saved_bytes", 0)
        if not smaller:
            knowledge = read(f"{profile}-mmlu.json")
            require(
                len(knowledge["records"]) == knowledge["requested_count"] >= 1000,
                "MMLU subset too small/incomplete",
            )
            require(
                len({r["id"] for r in knowledge["records"]}) == len(knowledge["records"]),
                "Duplicate MMLU task",
            )
            signature = [
                (
                    r["id"],
                    r["prompt_sha256"],
                    r["token_sha256"],
                    r["answer_token_ids"],
                    r["expected"],
                )
                for r in knowledge["records"]
            ]
            if profile == "native":
                matched["mmlu"] = signature
            require(signature == matched["mmlu"], "Unmatched MMLU tasks")
            for record in knowledge["records"]:
                probs = record["choice_probabilities"]
                require(
                    len(probs) == 4 and all(math.isfinite(v) and 0 <= v <= 1 for v in probs),
                    "Invalid MMLU probabilities",
                )
                require(
                    record["predicted"] == "ABCD"[max(range(4), key=lambda i: probs[i])],
                    "MMLU prediction mismatch",
                )
                require(
                    record["correct"] == (record["predicted"] == record["expected"]),
                    "MMLU scoring mismatch",
                )
            native_records = json.loads((folder / "native-mmlu.json").read_text())["records"]
            row["mmlu_flips"] = dict(
                lost=sum(
                    n["correct"] and not r["correct"]
                    for n, r in zip(native_records, knowledge["records"], strict=True)
                ),
                gained=sum(
                    not n["correct"] and r["correct"]
                    for n, r in zip(native_records, knowledge["records"], strict=True)
                ),
                changed_answers=sum(
                    n["predicted"] != r["predicted"]
                    for n, r in zip(native_records, knowledge["records"], strict=True)
                ),
            )
            row["mmlu"] = accuracy_interval(
                sum(r["correct"] for r in knowledge["records"]), len(signature)
            )
            generation = read(f"{profile}-humaneval.json")
            decoded = read(f"{profile}-humaneval-token-decoded.json")
            require(
                decoded["text_decoding_correction"]["source_sha256"]
                == hashes[f"{profile}-humaneval.json"],
                "Decoded source mismatch",
            )
            for raw_row, decoded_row in zip(generation["records"], decoded["records"], strict=True):
                require(
                    raw_row["token_ids"] == decoded_row["token_ids"]
                    and raw_row["id"] == decoded_row["id"],
                    "Changed generation tokens",
                )
                require(
                    decoded_row["text"] == " " + raw_row["text"], "Unexpected decoding correction"
                )
            coding = read(f"{profile}-humaneval-token-decoded-scored.json")
            require(
                {r["task_id"]: r["completion"] for r in coding["records"]}
                == {r["id"]: r["text"] for r in decoded["records"]},
                "Scored completions mismatch",
            )
            require(
                coding["generation_sha256"] == hashes[f"{profile}-humaneval-token-decoded.json"],
                "Coding provenance mismatch",
            )
            require(
                len(coding["records"]) == len({r["task_id"] for r in coding["records"]}) == 164,
                "Incomplete or duplicate HumanEval",
            )
            row["humaneval"] = accuracy_interval(
                sum(r["passed"] for r in coding["records"]), len(coding["records"])
            )
            retrieval = read(f"{profile}-retrieval.json")
            require(
                len(retrieval["records"]) == retrieval["requested_count"], "Incomplete retrieval"
            )
            signature = [
                (r["id"], r["prompt_sha256"], r["token_sha256"], r["expected"])
                for r in retrieval["records"]
            ]
            coding_signature = (
                generation["problems_sha256"],
                generation["parameters"],
                [(r["id"], r["prompt_sha256"], r["token_sha256"]) for r in generation["records"]],
            )
            if profile == "native":
                matched["retrieval"], matched["humaneval"] = signature, coding_signature
            require(
                signature == matched["retrieval"] and coding_signature == matched["humaneval"],
                "Unmatched capability tasks",
            )
            trials = sorted({r["trial"] for r in retrieval["records"]})
            require(
                len(trials) >= 10 and len({r["position"] for r in retrieval["records"]}) >= 5,
                "Insufficient retrieval coverage",
            )
            require(
                len({(r["trial"], r["position"]) for r in retrieval["records"]}) == 50,
                "Duplicate or missing retrieval positions",
            )
            require(
                all(len([r for r in retrieval["records"] if r["trial"] == t]) == 5 for t in trials),
                "Incomplete retrieval trial",
            )
            for record in retrieval["records"]:
                require(
                    record["correct"] == (record["text"].strip() == record["expected"]),
                    "Retrieval scoring mismatch",
                )
            correct = sum(
                all(r["correct"] for r in retrieval["records"] if r["trial"] == t) for t in trials
            )
            row["retrieval"] = accuracy_interval(correct, len(trials))
        summary[profile] = row
    if not smaller:
        require(
            summary["morph32-3s"]["representation_bytes"]
            == summary["independent-3s"]["representation_bytes"],
            "Controls are not storage matched",
        )
        require(
            summary["morph32-3s"]["representation_bytes"]
            == summary["morph32-3s-uncompressed"]["representation_bytes"],
            "Metadata ablation changes MLP payload",
        )
        require(
            summary["morph32-3s-uncompressed"]["metadata_saved_bytes"] == 0,
            "Metadata ablation still compresses metadata",
        )
    optimization = None
    if not smaller:
        original = read("morph32-3s-text.json")
        candidate = read("morph32-3s-split-text.json")
        require(
            candidate["corpus"] == original["corpus"]
            and candidate["tokenizer_sha256"] == original["tokenizer_sha256"],
            "Optimization changed evaluation inputs",
        )
        for key in ("model_sha256", "device", "packages", "platform", "source_sha256"):
            require(
                candidate["provenance"][key] == original["provenance"][key],
                "Optimization changed comparison settings: " + key,
            )
        require(candidate["provenance"]["decode_variant"] == "split", "Expected opt-in decoder")
        require(len(candidate["generation_runs"]) >= 10, "Insufficient candidate timings")
        require(
            all(len(r["token_ids"]) == 64 for r in candidate["generation_runs"]),
            "Candidate generation budget differs",
        )
        q = candidate["quality"]
        require(
            close(sum(candidate["token_nll"]) / len(candidate["token_nll"]), q["nll"])
            and close(math.exp(q["nll"]), q["perplexity"]),
            "Candidate quality mismatch",
        )
        require(
            q["reference_manifest_sha256"] == original["quality"]["reference_manifest_sha256"],
            "Candidate reference differs",
        )
        require(
            close(
                sum(candidate["token_kl"]) / len(candidate["token_kl"]),
                q["mean_kl_reference_to_candidate"],
            ),
            "Candidate KL mismatch",
        )
        profiles = {variant: read(f"profile-{variant}.json") for variant in ("original", "split")}
        require(
            all(report["projections"] for report in profiles.values()), "Empty generation profile"
        )
        optimization = dict(
            original_generation=summary["morph32-3s"]["generation"],
            candidate_generation=timing_summary(
                [r["generation_tps"] for r in candidate["generation_runs"]]
            ),
            original_quality=original["quality"],
            candidate_quality=q,
            candidate_file_bytes=candidate["inventory"]["file_bytes"],
            candidate_resident_parameter_bytes=candidate["resident_parameter_bytes"],
            largest_instrumented_projection=profiles["original"]["projections"][0]["name"],
            caveat="Instrumented projection ranking is not instruction-level bottleneck evidence; candidate remains opt-in",
        )
    return dict(
        optimization=optimization,
        status="complete",
        model=folder.name,
        smaller_model=smaller,
        profiles=summary,
        source_directory=str(folder.relative_to(paper_data)),
        source_sha256=hashes,
        interpretation="Current MORPH uses three planes. independent-3s is a base-fitting ablation; no recurrence is present. Historical recurrence results are separate. No superiority is assumed.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--smaller", action="store_true")
    args = parser.parse_args()
    summary = summarize(args.folder, args.smaller)
    write_json(args.folder / "summary.json", summary)


if __name__ == "__main__":
    main()
