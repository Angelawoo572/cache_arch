"""Regenerate the reported paired comparisons from the tracked tables."""

import csv
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def fig():
    placements = read(ROOT / "physical_replay/placement_results.csv")
    cues = read(ROOT / "deadlines/replay_results.csv")
    critical = read(ROOT / "conventional/criticality.csv")
    with plt.rc_context(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "prefetch-study",
        }
    ):
        f, axes = plt.subplots(
            1, 3, figsize=(13, 4.1), gridspec_kw={"width_ratios": [1.35, 1.25, 1]}
        )
        colors = ("#205d78", "#b05d34", "#799969")
        ax = axes[0]
        for i, (mode, label) in enumerate(
            [
                ("plus128", "Fixed +128"),
                ("recent", "Ranked"),
                ("recent16", "Ranked, 16-page gate"),
            ]
        ):
            data = [
                next(
                    float(r["time_saved_pct"])
                    for r in placements
                    if r["layout"] == layout and r["mode"] == mode
                )
                for layout in ("original", "perm")
            ]
            ax.bar(
                [j + (i - 1) * 0.23 for j in range(2)],
                data,
                width=0.22,
                color=colors[i],
                label=label,
            )
        ax.set_xticks([0, 1], ["Original", "Permuted"])
        ax.set_ylabel("Time saved vs. Stride (%)")
        ax.set_xlabel("Physical frame layout")
        ax.set_title("CactuBSSN 3477B", loc="left", fontweight="bold")
        ax.legend(
            frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.23)
        )
        ax = axes[1]
        for i, (mode, label) in enumerate(
            [
                ("current", "Current"),
                ("previous", "Previous event"),
                ("transition", "Stride transition"),
            ]
        ):
            data = [
                next(
                    int(r["cycles_saved"]) / 1000
                    for r in cues
                    if r["case"] == case and r["memory"] == mode
                )
                for case in ("mcf-cactus-r1", "lbm-r1")
            ]
            ax.bar(
                [j + (i - 1) * 0.23 for j in range(2)],
                data,
                width=0.22,
                color=colors[i],
                label=label,
            )
        ax.axhline(0, lw=0.8, color="black")
        ax.set_xticks([0, 1], ["MCF + Cactus", "LBM"])
        ax.set_ylabel("Cycles saved vs. Stride (thousands)")
        ax.set_title("Same-page cue tables", loc="left", fontweight="bold")
        ax.legend(
            frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.23)
        )
        ax = axes[2]
        data = [
            int(next(r["saved_makespan"] for r in critical if r["group"] == g))
            for g in ("high", "zero")
        ]
        bars = ax.bar([0, 1], data, color=[colors[0], colors[1]], width=0.6)
        ax.bar_label(bars, labels=[f"{v:,}" for v in data], padding=3, fontsize=9)
        ax.set_ylim(0, 2600)
        ax.set_xticks([0, 1], ["High wait", "Zero wait"])
        ax.set_ylabel("Cycles saved vs. Stride")
        ax.set_title("96 matched targets per arm", loc="left", fontweight="bold")
        for x in axes:
            x.yaxis.grid(True, color="#dddddd", linewidth=0.6)
            x.set_axisbelow(True)
        f.subplots_adjust(bottom=0.32, left=0.065, right=0.985, top=0.88, wspace=0.42)
        f.savefig(HERE / "performance_by_mechanism.svg", metadata={"Date": None})
        plt.close(f)


def page_history():
    comparison = read(ROOT / "page_memory/results.csv")
    history = next(
        r
        for r in read(ROOT / "history/page_memory_stability.csv")
        if r["train"] == "gap_bfs3_late-c0"
        and r["band"] == "all"
        and r["test"] == r["train"]
    )
    keys = [
        ("Current context", "current", "current"),
        ("Previous offset on page", "previous_offset", "offset"),
        ("Previous offset and PC", "previous_offset_and_pc", "link"),
        ("Next line", "next_line", "next"),
    ]
    with plt.rc_context(
        {"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "prefetch-study"}
    ):
        f, axes = plt.subplots(1, 2, figsize=(10.6, 3.0), sharey=True)
        for ax, metric in zip(axes, ("Targets", "Cycles")):
            if metric == "Targets":
                values = [int(history[k]) for _, k, _ in keys]
                ax.set_title(
                    "Correct offline choices in 1,460 grants", loc="left", fontsize=11
                )
            else:
                values = [
                    int(
                        next(
                            r["cycles_saved_vs_stride"]
                            for r in comparison
                            if r["mode"] == mode
                        )
                    )
                    for _, _, mode in keys
                ]
                ax.set_title("Cycles saved in native replay", loc="left", fontsize=11)
            bars = ax.barh(
                range(4),
                values,
                color=["#91a0a4", "#205d78", "#54879a", "#b05d34"],
                height=0.6,
            )
            ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=3, fontsize=9)
            ax.set_xlim(0, max(values) * 1.22)
            ax.set_yticks(range(4), [label for label, _, _ in keys])
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.xaxis.grid(True, color="#dddddd", linewidth=0.6)
            ax.set_axisbelow(True)
        axes[0].invert_yaxis()
        f.subplots_adjust(left=0.21, right=0.975, bottom=0.18, top=0.84, wspace=0.16)
        f.savefig(HERE / "bfs_page_history.svg", metadata={"Date": None})
        plt.close(f)


if __name__ == "__main__":
    fig()
    page_history()
