"""Render annotated baseline figures directly from the two archived performance tables."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from plot import LABELS, STYLE

STUDY = Path(__file__).resolve().parents[1]
PRIMARY_SOURCE = "native/performance.csv"
ENTIRE_SOURCE = "native/entire/performance.csv"
POLICIES = {
    "nopf": "NoPF",
    "stride": "Stride",
    "multi": "Multi-delta",
    "spp": "SPP",
    "temporal": "Temporal",
}
INK = "#172435"
MUTED = "#43556b"
PERCENT = mcolors.Normalize(0, 100)
DIVERGING = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=65)


def read_rows(source, cases):
    with (STUDY / source).open(newline="") as handle:
        entries = list(csv.DictReader(handle))
    rows = {(item["case"], item["policy"]): item for item in entries}
    if len(entries) != len(rows) or rows.keys() != {
        (case, policy) for case in cases for policy in POLICIES
    }:
        raise ValueError(f"{source} does not contain one row per case and policy")
    return rows


def figure(size, title, subtitle, source, notes):
    fig = plt.figure(figsize=size, facecolor="white")
    fig.text(0.02, 0.96, title, size=19, weight="bold")
    fig.text(0.02, 0.918, subtitle, size=10.3, color=MUTED)
    fig.text(
        0.02,
        0.03,
        f"{notes}\nSource  {source}",
        size=8.3,
        color=MUTED,
        linespacing=1.55,
    )
    return fig


def heatmap(
    ax,
    rows,
    cases,
    field,
    title,
    *,
    scale=1,
    cmap="Blues",
    norm=PERCENT,
    show_cases=True,
    value_format=".1f",
    size=8.4,
):
    data = np.array(
        [
            [
                (
                    scale * float(rows[case, policy][field])
                    if rows[case, policy][field]
                    else np.nan
                )
                for policy in POLICIES
            ]
            for case in cases
        ]
    )
    palette = plt.get_cmap(cmap).copy()
    palette.set_bad("#eff2f5")
    ax.imshow(data, aspect="auto", cmap=palette, norm=norm, interpolation="none")
    ax.set_title(title, loc="left", pad=10, size=11.2, weight="bold")
    ax.set_xticks(range(len(POLICIES)), POLICIES.values())
    ax.set_yticks(
        list(range(len(cases))), cases.values() if show_cases else [""] * len(cases)
    )
    ax.tick_params(axis="both", which="both", length=0, labelsize=8.3, pad=5)
    ax.set_xticks(np.arange(len(POLICIES) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(cases) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.spines[:].set_visible(False)
    for (row, column), value in np.ndenumerate(data):
        if np.isfinite(value):
            red, green, blue, _ = palette(norm(value))
            light = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            color = "white" if light < 0.48 else INK
            label = format(value, value_format)
        else:
            color = MUTED
            label = "–"
        ax.text(column, row, label, ha="center", va="center", size=size, color=color)


def cache_access(rows):
    fig = figure(
        (11.0, 8.4),
        "Demand cache hit and miss rates",
        "Nine fixed-work intervals · 50M warmup + 200M measured instructions per policy",
        PRIMARY_SOURCE,
        "Each percentage uses completed demand tag lookups at its cache level. Color intensity follows the percentage within each panel.",
    )
    grid = fig.add_gridspec(
        2, 2, left=0.183, right=0.975, top=0.85, bottom=0.112, hspace=0.38, wspace=0.10
    )
    for row, level in enumerate(("LLC", "L2")):
        for column, outcome in enumerate(("hit", "miss")):
            heatmap(
                fig.add_subplot(grid[row, column]),
                rows,
                LABELS,
                f"{level.lower()}_tag_{outcome}_rate",
                f"{level} tag {outcome} rate (%)",
                scale=100,
                cmap="Blues" if outcome == "hit" else "Oranges",
                show_cases=column == 0,
                value_format=".2f",
            )
    return fig


def prefetch_quality(rows):
    fig = figure(
        (13.9, 8.6),
        "Prefetch accuracy, coverage and timeliness",
        "Nine fixed-work intervals · accuracy counts accepted requests and coverage starts from original NoPF misses",
        PRIMARY_SOURCE,
        "First use is classified as a timely hit or late demand merge. A dash indicates zero denominator.\n"
        "Timely and inclusive coverage use matched dynamic operands. Net coverage divides the aggregate decrease in demand LLC misses by NoPF misses.",
    )
    grid = fig.add_gridspec(
        2, 3, left=0.145, right=0.984, top=0.84, bottom=0.115, hspace=0.41, wspace=0.13
    )
    panels = (
        ("useful_accuracy", "Useful accuracy (%)", "(timely + late) / accepted"),
        ("timely_accuracy", "Timely accuracy (%)", "timely / accepted"),
        ("timeliness_among_claims", "Timeliness (%)", "timely / (timely + late)"),
        (
            "timely_coverage",
            "Timely coverage (%)",
            "timely / original misses",
        ),
        (
            "inclusive_coverage",
            "Inclusive coverage (%)",
            "(timely + late) / original misses",
        ),
        (
            "net_miss_coverage",
            "Net miss coverage (%)",
            "aggregate misses saved / NoPF misses",
        ),
    )
    for index, (field, title, denominator) in enumerate(panels):
        row, column = divmod(index, 3)
        ax = fig.add_subplot(grid[row, column])
        heatmap(
            ax,
            rows,
            LABELS,
            field,
            f"{title}\n{denominator}",
            scale=100,
            cmap="RdBu" if field == "net_miss_coverage" else "Blues",
            norm=DIVERGING if field == "net_miss_coverage" else PERCENT,
            show_cases=column == 0,
        )
    return fig


def execution(rows):
    fig = figure(
        (11.7, 7.55),
        "Fixed-work performance",
        "All policies retire 200M instructions per interval. Positive cycle reduction means the policy finished sooner.",
        PRIMARY_SOURCE,
        "IPC counts retired instructions per cycle. Each cycle reduction uses the same interval's NoPF result.",
    )
    grid = fig.add_gridspec(
        1,
        2,
        left=0.176,
        right=0.97,
        top=0.835,
        bottom=0.151,
        width_ratios=(1.2, 1),
        wspace=0.14,
    )
    ax = fig.add_subplot(grid[0, 0])
    heatmap(
        fig.add_subplot(grid[0, 1]),
        rows,
        LABELS,
        "ipc",
        "Instructions per cycle (IPC)",
        cmap="GnBu",
        norm=mcolors.Normalize(0, 0.45),
        show_cases=False,
        value_format=".4f",
    )
    ax.set_title(
        "Fixed-work cycle reduction vs NoPF (%)",
        loc="left",
        pad=10,
        size=11.2,
        weight="bold",
    )
    policy_offsets = np.array([-0.28, -0.095, 0.095, 0.28])
    for pos, (policy, (_, color, _)) in enumerate(STYLE.items()):
        bar_y = np.arange(len(LABELS)) + policy_offsets[pos]
        values = [
            100 * float(rows[case, policy]["time_reduction_vs_nopf"]) for case in LABELS
        ]
        ax.barh(bar_y, values, height=0.16, color=color, zorder=2)
        for y, value in zip(bar_y, values):
            ax.annotate(
                f"{value:.2f}",
                (value, y),
                xytext=(-3 if value < 0 else 3, 0),
                textcoords="offset points",
                ha="right" if value < 0 else "left",
                va="center",
                size=7.2,
                color=INK,
            )
    ax.set(
        ylim=(len(LABELS) - 0.5, -0.5),
        xlim=(-5.0, 23),
        yticks=np.arange(len(LABELS)),
        yticklabels=LABELS.values(),
    )
    ax.tick_params(axis="both", labelsize=8.1, length=0, pad=5)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color("#bcc9d7")
    ax.axvline(0, linewidth=0.8, color="#546680")
    ax.set_axisbelow(True)
    ax.grid(axis="x", linewidth=0.6, color="#dce3eb")
    ax.legend(
        handles=[
            Patch(facecolor=color, label=POLICIES[key])
            for key, (_, color, _) in STYLE.items()
        ],
        bbox_to_anchor=(0, -0.050),
        loc="upper left",
        borderaxespad=0,
        ncol=2,
        frameon=False,
        fontsize=8.0,
    )
    return fig


def miss_intensity(rows):
    fig = figure(
        (11.0, 6.4),
        "Cache miss intensity and memory pressure",
        "Nine fixed-work intervals · values describe full measured runs",
        PRIMARY_SOURCE,
        "MPKI counts demand tag misses per 1,000 retired instructions. Occupancies are averages over measured cycles.",
    )
    grid = fig.add_gridspec(
        2, 2, left=0.183, right=0.975, top=0.815, bottom=0.137, hspace=0.44, wspace=0.10
    )
    panels = (
        ("llc_mpki", "LLC demand MPKI", 80, ".1f"),
        ("l2_mpki", "L2 demand MPKI", 175, ".1f"),
        ("avg_llc_mshr", "Mean occupied LLC MSHRs", 4, ".2f"),
        ("avg_dram_rq", "Mean DRAM read queue occupancy", 4, ".2f"),
    )
    for index, (field, title, upper, places) in enumerate(panels):
        row, column = divmod(index, 2)
        observed_max = max(float(entry[field]) for entry in rows.values())
        heatmap(
            fig.add_subplot(grid[row, column]),
            rows,
            LABELS,
            field,
            title,
            cmap="GnBu",
            norm=mcolors.Normalize(0, max(observed_max, upper)),
            show_cases=column == 0,
            value_format=places,
        )
    return fig


def pagerank_entire(rows):
    case = "pr_heldout_entire"
    fig = figure(
        (11.0, 5.8),
        "GAP PageRank-3 through end of recording",
        "500,000,001 original instructions · 50M warmup + 450,000,001 measured",
        ENTIRE_SOURCE,
        "Cycle labels give elapsed billions, IPC and elapsed-time change vs NoPF. Each coverage value starts from 35,499,013 original NoPF misses.\n"
        "Useful accuracy counts timely and late first uses per accepted request. Timeliness counts the timely share of those first uses.",
    )
    grid = fig.add_gridspec(
        1,
        2,
        left=0.124,
        right=0.98,
        top=0.83,
        bottom=0.19,
        wspace=0.25,
        width_ratios=(1.08, 1),
    )
    ax = fig.add_subplot(grid[0, 0])
    ax.set_title(
        "Fixed-work time by policy", loc="left", pad=10, size=11.2, weight="bold"
    )
    baseline = int(rows[case, "nopf"]["cycles"])
    for index, policy in enumerate(POLICIES):
        row = rows[case, policy]
        cycles = int(row["cycles"])
        color = STYLE[policy][1] if policy in STYLE else "#596578"
        value = cycles / 1e9
        ax.hlines(index, baseline / 1e9, value, color=color, linewidth=2)
        ax.scatter(value, index, color=color, s=35, zorder=3)
        ax.annotate(
            f"{value:.3f}B / {float(row['ipc']):.5f} / {(cycles - baseline) / baseline:+.3%}",
            (value, index),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            size=8.8,
            color=INK,
        )
    ax.axvline(baseline / 1e9, linewidth=0.75, color="#b5c1cf")
    ax.set(
        xlim=(10.060, 10.355),
        ylim=(4.5, -0.5),
        yticks=range(len(POLICIES)),
        yticklabels=POLICIES.values(),
        xlabel="Retirement cycles (billions)",
    )
    ax.tick_params(axis="both", labelsize=8.5, length=0, pad=5)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color("#bcc9d7")
    ax.set_axisbelow(True)
    ax.grid(axis="x", linewidth=0.65, color="#e1e9ee")
    quality = fig.add_subplot(grid[0, 1])
    quality.set_title(
        "Demand outcomes and prefetch quality (%)",
        loc="left",
        pad=10,
        size=11.2,
        weight="bold",
    )
    comparisons = (
        ("llc_tag_hit_rate", "LLC hit"),
        ("llc_tag_miss_rate", "LLC miss"),
        ("timely_coverage", "Timely\ncoverage"),
        ("useful_accuracy", "Useful\naccuracy"),
        ("timeliness_among_claims", "Timeliness"),
    )
    palette = plt.get_cmap("Blues").copy()
    palette.set_bad("#eff2f5")
    data = np.array(
        [
            [
                (
                    100 * float(rows[case, policy][key])
                    if rows[case, policy][key]
                    else np.nan
                )
                for key, _ in comparisons
            ]
            for policy in POLICIES
        ]
    )
    quality.imshow(
        data, aspect="auto", cmap=palette, norm=PERCENT, interpolation="none"
    )
    quality.set_xticks(range(len(comparisons)), [label for _, label in comparisons])
    quality.set_yticks(range(len(POLICIES)), [""] * len(POLICIES))
    quality.tick_params(axis="both", which="both", length=0, labelsize=8.5, pad=5)
    quality.set_xticks(np.arange(len(comparisons) + 1) - 0.5, minor=True)
    quality.set_yticks(np.arange(len(POLICIES) + 1) - 0.5, minor=True)
    quality.grid(which="minor", color="white", linewidth=1)
    quality.spines[:].set_visible(False)
    for (row, column), value in np.ndenumerate(data):
        visible = np.isfinite(value)
        color = "white" if visible and value >= 70 else INK
        quality.text(
            column,
            row,
            f"{value:.2f}" if visible else "–",
            color=color,
            ha="center",
            va="center",
            size=9,
        )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=STUDY / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "svg.fonttype": "none",
            "svg.hashsalt": "full-workload-annotated-baselines",
        }
    )
    primary = read_rows(PRIMARY_SOURCE, LABELS)
    entire = read_rows(ENTIRE_SOURCE, ("pr_heldout_entire",))
    for name, fig in (
        ("cache_access", cache_access(primary)),
        ("prefetch_quality", prefetch_quality(primary)),
        ("execution", execution(primary)),
        ("miss_intensity", miss_intensity(primary)),
        ("pagerank_entire", pagerank_entire(entire)),
    ):
        fig.savefig(args.out / f"baseline_{name}.svg", metadata={"Date": None})
        plt.close(fig)


if __name__ == "__main__":
    main()
