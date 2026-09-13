"""Build the shared comparison table from audited saved records."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
NAMES = ("native", "morph32-c", "morph32-3s", "published-3bit")
LABELS = {
    "native": "MLX 4-bit",
    "morph32-c": "MORPH32-c",
    "morph32-3s": "MORPH32-3s",
    "published-3bit": "MLX 3-bit",
}
START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def make(primary, smaller):
    """Refresh the paper's JSON source and README's matching four-model table."""
    (PAPER / "data/final-summary.json").write_text(
        json.dumps({"primary": primary, "smaller": smaller}, indent=2) + "\n"
    )
    rows = primary["profiles"]
    table = [
        "| Model | File GB | Resident GB | Text PPL ↓ | Code PPL ↓ | MMLU ↑ | HumanEval ↑ | Gen. tok/s ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in NAMES:
        row = rows[name]
        table.append(
            f"| {LABELS[name]} | {row['file_bytes'] / 1e9:.2f} | "
            f"{row['resident_parameter_bytes'] / 1e9:.2f} | "
            f"{row['text']['perplexity']:.3f} | {row['code']['perplexity']:.3f} | "
            f"{100 * row['mmlu']['accuracy']:.2f}% | "
            f"{100 * row['humaneval']['accuracy']:.2f}% | "
            f"{row['generation']['median']:.2f} |"
        )
    readme = ROOT / "README.md"
    content = readme.read_text()
    if content.count(START) != 1 or content.count(END) != 1:
        raise ValueError("README results markers must appear exactly once")
    before, rest = content.split(START, 1)
    _, after = rest.split(END, 1)
    readme.write_text(before + START + "\n\n" + "\n".join(table) + "\n\n" + END + after)
