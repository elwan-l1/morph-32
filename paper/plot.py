"""Rebuild the vector figure from the paper's frozen measurement snapshot."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def main() -> None:
    directory = Path(__file__).resolve().parent
    measurements = json.loads((directory / "data" / "summary.json").read_text())
    for path in Path("/System/Library/Fonts/Supplemental").glob("Times New Roman*.ttf"):
        font_manager.fontManager.addfont(str(path))
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "svg.fonttype": "path",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(5.5, 2.25), layout="constrained")
    configurations = [
        ("native", "Native", "o", "black"),
        ("morph32-3s", "morph32-3s", "s", "#444444"),
        ("morph32-c", "morph32-c", "^", "#777777"),
    ]
    for axis, metric, ylabel in zip(
        axes,
        ["heldout_perplexity", "generation_tokens_per_second"],
        ["Perplexity (lower is better)", "Generation (tokens/s)"],
    ):
        for key, label, marker, color in configurations:
            entry = measurements[key]
            x = entry["resident_parameter_bytes"] / 1e9
            y = entry[metric]
            axis.scatter(x, y, marker=marker, color=color, s=28, zorder=3)
            offset = (7, 7) if key != "native" else (-7, 7)
            axis.annotate(
                label,
                (x, y),
                xytext=offset,
                textcoords="offset points",
                ha="left" if key != "native" else "right",
                fontsize=9,
            )
        axis.set_xlim(11.9, 15.5)
        axis.set_xticks([12, 13, 14, 15])
        axis.set_xlabel("Resident weights (GB)", labelpad=3)
        axis.set_ylabel(ylabel, labelpad=3)
        axis.grid(axis="y", color="0.9", linewidth=0.6)
    axes[0].set_ylim(4.70, 5.12)
    axes[1].set_ylim(6, 21)
    figure.savefig(directory / "figures" / "tradeoffs.svg", metadata={"Date": None})
    plt.close(figure)


if __name__ == "__main__":
    main()
