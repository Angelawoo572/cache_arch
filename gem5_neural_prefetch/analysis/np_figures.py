#!/usr/bin/env python3
"""Draw the measurement figures from measurement/tables/*.csv.

    python analysis/np_figures.py [--arm fast] [--kind gap]

Writes SVG files to measurement/figures/. Every figure has its table next to
it in measurement/tables/, which is the exact, colour-free view of the same
numbers. Colours are the validated reference palette of the charting method
used in this repository (checked for colour-vision deficiency, light surface).
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = Path(os.environ.get("GEM5_STUDY", HERE.parent))
TABLES = STUDY / "measurement" / "tables"
FIGURES = STUDY / "measurement" / "figures"
PREVIEW = None  # set by --preview: a folder that also receives PNG copies

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Categorical slots, in their fixed order.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
)
# One-hue ordinal ramp, light to dark.
RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#184f95")

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)

KERNEL = {"bfs": "BFS", "pr": "PageRank", "cc": "Components", "bc": "Betweenness", "sssp": "SSSP"}
ARM_TEXT = {
    "stride": "L1D stride",
    "fast": "stride + signature path",
    "bop": "stride + best offset",
    "ampm": "stride + access map",
}


def read(name, arm, kind):
    rows = []
    with (TABLES / name).open(newline="") as file:
        for row in csv.DictReader(file):
            if row["arm"] == arm and row["workload"].startswith(kind):
                rows.append(row)
    return rows


def label(row):
    parts = row["workload"].split("_")
    offset = int(row["offset"])
    where = "start" if offset == 0 else f"+{offset // 1_000_000}M"
    if parts[0] == "gap":
        return f"{KERNEL.get(parts[1], parts[1])} {parts[2]}, {where}"
    return f"{parts[1]}"


def number(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", "None") else 0.0


def stacked(rows, series, title, subtitle, xlabel, out, text_on=None):
    """Horizontal 100% stacked bars. series: (legend label, colour, values)."""
    names = [label(r) for r in rows]
    n = len(rows)
    columns = min(len(series), 4)
    legend_rows = -(-len(series) // columns)
    head = 1.1 + 0.24 * (legend_rows - 1)  # title, subtitle and legend, in inches
    height = head + 0.05 + 0.36 * n + 0.75
    fig, ax = plt.subplots(figsize=(8.6, height), dpi=100)
    fig.subplots_adjust(left=0.24, right=0.97, top=1 - head / height, bottom=0.62 / height)

    left = [0.0] * n
    for index, (name, colour, values) in enumerate(series):
        # A 2 px edge in the surface colour separates touching segments.
        ax.barh(
            range(n), values, left=left, height=0.56, color=colour,
            edgecolor=SURFACE, linewidth=1.5, label=name,
        )
        # Dark ink wherever it contrasts more than white does.
        ink = INK if colour in (YELLOW, ORANGE, AQUA, MAGENTA, RAMP[0], RAMP[1], NEUTRAL) else "#ffffff"
        for y, (start, width) in enumerate(zip(left, values)):
            if width >= 8.0:  # print a share only where it fits
                ax.text(start + width / 2, y, f"{width:.0f}%", ha="center", va="center",
                        fontsize=8.5, color=ink)
        left = [a + b for a, b in zip(left, values)]

    ax.set_yticks(range(n))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, color=AXIS)

    fig.text(0.02, 1 - 0.30 / height, title, fontsize=12.5, fontweight="bold", va="center")
    fig.text(0.02, 1 - 0.58 / height, subtitle, fontsize=9.5, color=INK_2, va="center")
    ax.legend(
        loc="lower left", bbox_to_anchor=(-0.31, 1.0), ncol=columns,
        frameon=False, fontsize=9, handlelength=1.1, handleheight=0.9,
        columnspacing=1.4, borderaxespad=0.3,
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / out)
    if PREVIEW is not None:
        PREVIEW.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREVIEW / (Path(out).stem + ".png"), dpi=160)
    plt.close(fig)
    print(f"wrote {FIGURES / out}")


def share(part, whole):
    return 100.0 * part / whole if whole > 0 else 0.0


def figure_residual(arm, kind):
    rows = read("m1_residual.csv", arm, kind)
    if not rows:
        return
    removed, late, miss, other = [], [], [], []
    for r in rows:
        before = number(r, "beyond_l2.stall_nopf")
        after_late = number(r, "beyond_l2.stall_fast_late")
        after_miss = number(r, "beyond_l2.stall_fast_miss")
        after_other = number(r, "beyond_l2.stall_fast_other") + number(r, "beyond_l2.stall_fast_timely")
        whole = max(before, after_late + after_miss + after_other)
        removed.append(share(whole - after_late - after_miss - after_other, whole))
        late.append(share(after_late, whole))
        miss.append(share(after_miss, whole))
        other.append(share(after_other, whole))
    stacked(
        rows,
        [
            ("removed by the fast path", BLUE, removed),
            ("left: prefetch arrived late", ORANGE, late),
            ("left: no prefetch", AQUA, miss),
            ("left on loads found in L1 or L2", YELLOW, other),
        ],
        "What the conventional prefetchers leave",
        f"Reorder-buffer stall on loads that miss beyond L2 without prefetching; with {ARM_TEXT.get(arm, arm)}",
        "share of the stall those loads cause without prefetching",
        f"residual_stall_{arm}.svg",
    )


def figure_mechanism(arm, kind):
    rows = read("m2_mechanism.csv", arm, kind)
    if not rows:
        return
    names = (
        ("same-PC delta", "same_pc_delta", BLUE),
        ("loaded value, 1 link", "value_depth_1", ORANGE),
        ("loaded value, chain of misses", "value_depth_2_plus", AQUA),
        ("offset on a known page", "offset_on_known_page", YELLOW),
        ("repeat of an earlier pair", "temporal_repeat", MAGENTA),
        ("none of these", "none", GREEN),
    )
    series = []
    for text, key, colour in names:
        series.append((text, colour, [share(number(r, f"{key}.stall"), number(r, "stall")) for r in rows]))
    stacked(
        rows,
        series,
        "What could name the addresses that are left",
        f"Residual stall by the cheapest construction that names the line; with {ARM_TEXT.get(arm, arm)}",
        "share of residual stall",
        f"residual_mechanism_{arm}.svg",
    )


def figure_lead(arm, kind):
    rows = [r for r in read("m3_lead.csv", arm, kind) if number(r, "stall") > 0]
    if not rows:
        return
    s25 = [100 * number(r, "stall_share_slack_at_least_25pct_of_latency") for r in rows]
    s50 = [100 * number(r, "stall_share_slack_at_least_50pct_of_latency") for r in rows]
    s100 = [100 * number(r, "stall_share_slack_at_least_100pct_of_latency") for r in rows]
    series = [
        ("under 25% of the miss latency", RAMP[0], [100 - a for a in s25]),
        ("25% to 50%", RAMP[1], [a - b for a, b in zip(s25, s50)]),
        ("50% to 100%", RAMP[2], [a - b for a, b in zip(s50, s100)]),
        ("the whole latency or more", RAMP[3], s100),
    ]
    stacked(
        rows,
        series,
        "How much lead a start at the value's return has",
        f"Value-derived residual stall by the time from the value's return to the consumer's issue; with {ARM_TEXT.get(arm, arm)}",
        "share of value-derived residual stall",
        f"traversal_lead_{arm}.svg",
    )


REACH = ("#86b6ef", "#2a78d6", "#104281")  # three ordinal steps, light to dark
NEUTRAL = "#c3c2b7"  # "not reached": a gray, deliberately not a step of the ramp


def figure_reach(arm, kind):
    """How much value-derived residual stall a bounded traversal could reach
    in time, by the number of links it must follow."""
    rows = [r for r in read("m3_walker_bound.csv", arm, kind) if number(r, "stall") > 0]
    if not rows:
        return
    h1 = [100 * number(r, "any.hops_1") for r in rows]
    h2 = [100 * number(r, "any.hops_2") for r in rows]
    h3 = [100 * number(r, "any.hops_3") for r in rows]
    series = [
        ("1 link", REACH[2], h1),
        ("2 links", REACH[1], [b - a for a, b in zip(h1, h2)]),
        ("3 links", REACH[0], [b - a for a, b in zip(h2, h3)]),
        ("not within 3 links", NEUTRAL, [100 - a for a in h3]),
    ]
    stacked(
        rows,
        series,
        "How far a traversal must follow to be in time",
        "Value-derived residual stall an early-started traversal reaches in time, by links followed "
        f"(upper bound); with {ARM_TEXT.get(arm, arm)}",
        "share of value-derived residual stall",
        f"traversal_reach_{arm}.svg",
    )


ARM_STYLE = {  # fixed colour and marker per arm: identity never rests on colour alone
    "stride": ("L1D stride", BLUE, "o"),
    "fast": ("stride + signature path", ORANGE, "s"),
    "bop": ("stride + best offset", AQUA, "^"),
    "ampm": ("stride + access map", YELLOW, "D"),
}


def figure_arms(kind):
    """Speed of every conventional arm against no prefetching, per window."""
    table = {}
    with (TABLES / "windows.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            if row["workload"].startswith(kind) and row["arm"] in ARM_STYLE:
                key = (row["workload"], int(row["offset"]))
                table.setdefault(key, {"row": row})[row["arm"]] = number(row, "cycles_nopf") / number(row, "cycles_fast")
    if not table:
        return
    keys = sorted(table)
    n = len(keys)
    height = 1.15 + 0.36 * n + 0.75
    fig, ax = plt.subplots(figsize=(8.6, height), dpi=100)
    fig.subplots_adjust(left=0.24, right=0.97, top=1 - 1.1 / height, bottom=0.62 / height)
    top = max(max(v for k, v in table[key].items() if k != "row") for key in keys)
    for y, key in enumerate(keys):
        values = [v for k, v in table[key].items() if k != "row"]
        ax.plot([min(values), max(values)], [y, y], color=GRID, linewidth=2, zorder=1, solid_capstyle="round")
    for arm, (name, colour, marker) in ARM_STYLE.items():
        xs = [table[key].get(arm) for key in keys]
        points = [(x, y) for y, x in enumerate(xs) if x is not None]
        if not points:
            continue
        ax.scatter(
            [x for x, _ in points], [y for _, y in points], s=46, color=colour, marker=marker,
            edgecolors=SURFACE, linewidths=1.2, zorder=3, label=name,
        )
    ax.axvline(1.0, color=AXIS, linewidth=1.0, zorder=0)
    ax.set_yticks(range(n))
    ax.set_yticklabels([label(table[key]["row"]) for key in keys])
    ax.invert_yaxis()
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(min(0.9, min(min(v for k, v in table[key].items() if k != "row") for key in keys) - 0.05), top + 0.1)
    ax.set_xlabel("speed relative to no prefetching (1.0 = equal)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, color=AXIS)
    fig.text(0.02, 1 - 0.30 / height, "What each conventional configuration already gains", fontsize=12.5, fontweight="bold", va="center")
    fig.text(0.02, 1 - 0.58 / height, "Fixed work of 20M instructions per window; one marker per configuration", fontsize=9.5, color=INK_2, va="center")
    ax.legend(
        loc="lower left", bbox_to_anchor=(-0.31, 1.0), ncol=5, frameon=False, fontsize=9,
        handletextpad=0.3, columnspacing=1.2, borderaxespad=0.3,
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = f"arms_speed_{kind.rstrip('_')}.svg"
    fig.savefig(FIGURES / out)
    if PREVIEW is not None:
        PREVIEW.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREVIEW / (Path(out).stem + ".png"), dpi=160)
    plt.close(fig)
    print(f"wrote {FIGURES / out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="fast")
    parser.add_argument("--kind", default="gap", help="workload name prefix")
    parser.add_argument("--preview", type=Path, default=None, help="also write PNG copies here")
    args = parser.parse_args()
    global PREVIEW
    PREVIEW = args.preview
    figure_residual(args.arm, args.kind)
    figure_mechanism(args.arm, args.kind)
    figure_lead(args.arm, args.kind)
    figure_reach(args.arm, args.kind)
    figure_arms(args.kind)


if __name__ == "__main__":
    main()
