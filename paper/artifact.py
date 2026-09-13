"""Rebuild the current publication from archived evidence; never run inference or PDF compilation."""

import hashlib
import json
from pathlib import Path

from publication import make
from research_artifact import require, summarize

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def main():
    primary_dir = PAPER / "data/research/primary-20260911"
    smaller_dir = PAPER / "data/research/smaller-20260911"
    primary = summarize(primary_dir)
    smaller = summarize(smaller_dir, True)
    for directory, summary in [(primary_dir, primary), (smaller_dir, smaller)]:
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        commands = json.loads((directory / "commands.json").read_text())
        require(commands["status"] == "complete", "Incomplete manifest")
        for step in commands["steps"]:
            require(step["status"] == "complete", "Incomplete step")
            for field, name in [("log_sha256", step["log"])]:
                require(
                    hashlib.sha256((directory / name).read_bytes()).hexdigest() == step[field],
                    "Changed step log: " + name,
                )
            if "output_sha256" in step:
                output = ROOT / step["argv"][step["argv"].index("--output") + 1]
                require(
                    hashlib.sha256(output.read_bytes()).hexdigest() == step["output_sha256"],
                    "Changed output: " + str(output),
                )
    # Recheck every saved MMLU/decoding audit's source chain.
    for path in primary_dir.glob("*-audit.json"):
        audit = json.loads(path.read_text())
        if "report_sha256" in audit:
            report = primary_dir / audit.get("report", path.name.replace("-audit.json", ".json"))
            require(
                report.exists()
                and hashlib.sha256(report.read_bytes()).hexdigest() == audit["report_sha256"],
                "Changed audited report " + str(report),
            )
    # Recheck the lossless control using full saved outputs, not aggregate scores.
    for task, keys in [
        ("mmlu", ("id", "choice_probabilities", "predicted")),
        ("humaneval", ("id", "token_ids")),
        ("retrieval", ("id", "token_ids", "text")),
    ]:
        left = json.loads((primary_dir / f"morph32-3s-{task}.json").read_text())["records"]
        right = json.loads((primary_dir / f"morph32-3s-uncompressed-{task}.json").read_text())[
            "records"
        ]
        require(
            [[r[k] for k in keys] for r in left] == [[r[k] for k in keys] for r in right],
            "Metadata control differs: " + task,
        )
    header_path = primary_dir / "metadata-header-equivalence.json"
    header = json.loads(header_path.read_text())
    a, b = header["checkpoints"].values()
    require(
        a["descriptors"] == b["descriptors"] and a["payload_sha256"] == b["payload_sha256"],
        "Metadata payload/header mismatch",
    )
    current = PAPER / "data/research/current-20260911"
    index = json.loads((current / "checkpoint-index.json").read_text())
    require(len(index) == 43, "Incomplete evidence index")
    for item in index:
        card = ROOT / item["card"]
        require(card.exists(), "Missing evidence card")
        for link in item["evidence"]:
            require((card.parent / link).exists(), "Broken evidence link: " + link)
    make(primary, smaller)
    files = [
        ROOT / "README.md",
        PAPER / "main.typ",
        PAPER / "data/final-summary.json",
        PAPER / "research_artifact.py",
        PAPER / "publication.py",
        PAPER / "artifact.py",
        current / "checkpoint-index.json",
        header_path,
        *sorted((current / "checkpoints").glob("CP*.md")),
        PAPER / "figures/morph32-c.svg",
        PAPER / "figures/morph32-3s.svg",
        PAPER / "figures/quality-size-paper.svg",
    ]
    sources = {
        str((primary_dir / name).relative_to(ROOT)): digest
        for name, digest in primary["source_sha256"].items()
    }
    sources.update(
        {
            str((smaller_dir / name).relative_to(ROOT)): digest
            for name, digest in smaller["source_sha256"].items()
        }
    )
    result = dict(
        status="complete",
        scope="Current primary and smaller saved-record arithmetic, matched inputs, exact token-decoding chain, execution output/log hashes, shared SVG figures and regenerated comparison table. No inference or PDF output.",
        source_sha256=sources,
        artifact_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    )
    (PAPER / "data/artifact-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "Verified archived evidence; rebuilt summaries and README table. No benchmarks or PDF output."
    )


if __name__ == "__main__":
    main()
