"""Plot the two measured endpoints for the same nine fixed-work intervals."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FULL = Path(__file__).resolve().parents[1]
LABELS = {
    "mcf06_discovery": "SPEC06 MCF · 22B",
    "soplex06_heldout": "SPEC06 Soplex · 92B",
    "mcf17_discovery": "SPEC17 MCF · 472B, start",
    "mcf17_discovery_far": "SPEC17 MCF · 472B, end",
    "mcf17_transfer": "SPEC17 MCF · 1644B",
    "bfs_discovery": "GAP BFS · 3, start",
    "bfs_discovery_far": "GAP BFS · 3, end",
    "bfs_transfer": "GAP BFS · 14",
    "pr_heldout": "GAP PageRank · 3",
}
STYLE = {
    "stride": ("Stride", "#215b95", "o"),
    "multi": ("Multi-delta", "#c4580a", "s"),
    "spp": ("SPP, degree one", "#178065", "^"),
    "temporal": ("Temporal", "#774586", "D"),
}


def main():
    with (FULL / "native/performance.csv").open() as f:
        rows = {(r["case"], r["policy"]): r for r in csv.DictReader(f)}
    plt.rcParams.update(
        {"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "full-workload"}
    )
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 7.2), layout="constrained")
    for ax, field, title in zip(
        axes,
        ("timely_coverage", "time_reduction_vs_nopf"),
        (
            "Original LLC misses converted to timely hits",
            "Fixed-work cycle reduction vs NoPF",
        ),
    ):
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e9e9e9", linewidth=0.6)
        ax.axvline(0, color="#686868", linewidth=0.8)
        for j, (policy, (label, color, marker)) in enumerate(STYLE.items()):
            xs = [100 * float(rows[c, policy][field]) for c in LABELS]
            ys = [i + (j - 1.5) * 0.19 for i in range(len(xs))]
            ax.scatter(xs, ys, marker=marker, s=30, color=color, label=label, zorder=3)
            for value, y in zip(xs, ys):
                ax.annotate(
                    f"{value:.2f}%", (value, y), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=7, color=color,
                )
        margin = 0.14
        ax.margins(x=margin, y=0.035)
        ax.set_yticks(range(len(LABELS)), LABELS.values())
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left", fontsize=11, pad=10)
        ax.set_xlabel("Percent")
    axes[0].invert_yaxis()
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    fig.savefig(FULL / "figures/coverage_and_cycles.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
