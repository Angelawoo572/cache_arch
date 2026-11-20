"""Draw measured metrics from the aligned runs. Values and units appear in the plots."""

import csv
from pathlib import Path
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ARMS = ("nopf", "stride", "multi", "spp")
NAMES = ("NoPF", "Stride", "Multi-delta", "SPP")
COLORS = ("#526574", "#177899", "#cc681d", "#425ea2")
mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlelocation": "left",
        "svg.fonttype": "none",
        "font.size": 9,
    }
)


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.savefig(
        HERE / f"{name}.svg",
        bbox_inches="tight",
        pad_inches=0.18,
        metadata={"Date": None},
    )
    plt.close(fig)


def panels(rows, title, specs, file):
    cases = list(dict.fromkeys(r["case"] for r in rows))
    data = {(r["case"], r["policy"]): r for r in rows}
    fig, axs = plt.subplots(
        1,
        len(specs),
        figsize=(4.2 * len(specs), 0.36 * len(cases) + 1.8),
        layout="constrained",
        squeeze=False,
    )
    fig.suptitle(title, fontweight="bold", fontsize=14)
    for i, (field, label, scale, fmt, palette) in enumerate(specs):
        ax = axs[0, i]
        grid = np.full((len(cases), 4), np.nan)
        for y, case in enumerate(cases):
            for x, policy in enumerate(ARMS):
                val = data[case, policy].get(field, "")
                if val != "":
                    grid[y, x] = scale * float(val)
        vmax, wmin = float(np.nanmax(grid)), float(np.nanmin(grid))
        norm = (
            TwoSlopeNorm(vcenter=0, vmin=min(-0.01, wmin), vmax=max(0.01, vmax))
            if palette == "RdBu"
            else Normalize(vmin=0, vmax=max(vmax, 0.001))
        )
        cmap = mpl.colormaps[palette]
        ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm)
        for (j, k), v in np.ndenumerate(grid):
            if np.isnan(v):
                ax.text(k, j, "—", ha="center", va="center", color="#788187")
            else:
                rgb = cmap(norm(v))
                ink = (
                    "white"
                    if sum(a * b for a, b in zip(rgb, (0.21, 0.72, 0.07))) < 0.53
                    else "#17232c"
                )
                ax.text(
                    k,
                    j,
                    format(v, fmt),
                    ha="center",
                    va="center",
                    color=ink,
                    fontsize=8,
                )
        ax.set_xticks(range(4), NAMES, fontsize=8)
        ax.set_yticks(range(len(cases)), cases if i == 0 else [""] * len(cases))
        ax.tick_params(length=0)
        ax.set_title(label, fontsize=10)
    save(fig, file)


def pareto(rows):
    cases = list(
        dict.fromkeys(
            r["case"]
            for r in rows
            if r["case"][-6:] != "_slot1" and not r["case"].startswith("snap_")
        )
    )
    cols = 3
    fig, axes = plt.subplots(
        (len(cases) + cols - 1) // cols,
        cols,
        figsize=(12, 2.75 * ((len(cases) + cols - 1) // cols)),
        layout="constrained",
        squeeze=False,
    )
    fig.suptitle(
        "Execution time and memory traffic at fixed work",
        fontsize=14,
        fontweight="bold",
    )
    for ax, case in zip(axes.flat, cases):
        by_arm = {r["policy"]: r for r in rows if r["case"] == case}
        armdata = [by_arm[policy] for policy in ARMS]
        vals = [
            (
                sum(
                    int(r[k])
                    for k in (
                        "dram_demand_bytes",
                        "dram_prefetch_bytes",
                        "dram_write_bytes",
                    )
                )
                / 1e6,
                int(r["cycles"]) / 1e6,
            )
            for r in armdata
        ]
        points = set(vals)
        frontier = sorted(
            p
            for p in points
            if not any(q != p and q[0] <= p[0] and q[1] <= p[1] for q in points)
        )
        ax.plot(*zip(*frontier), color="#71836d", lw=1.3, zorder=1)
        for i, (x, y) in enumerate(vals):
            ax.scatter(
                x,
                y,
                s=45,
                marker=("o", "^", "s", "D")[i],
                color=COLORS[i],
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
            )
        ax.set_title(case, weight="bold", fontsize=10)
        coords = "  ".join(
            f"{abbr} {x:.1f}/{y:.1f}"
            for abbr, (x, y) in zip(("N", "S", "M", "P"), vals)
        )
        ax.text(
            0,
            -0.39,
            coords.replace("  M", "\nM"),
            ha="left",
            transform=ax.transAxes,
            fontsize=7,
        )
        ax.set_xlabel("Total DRAM (MB)", fontsize=8)
        ax.set_ylabel("Cycles (millions)", fontsize=8)
        ax.grid(alpha=0.15)
        ax.margins(0.18)
    for ax in axes.flat[len(cases) :]:
        ax.set_visible(False)
    fig.legend(
        [
            Line2D([], [], marker=m, color=c, lw=0)
            for m, c in zip(("o", "^", "s", "D"), COLORS)
        ],
        NAMES,
        ncols=4,
        loc="outside lower right",
    )
    save(fig, "traffic_time_pareto")


def trials():
    rows = read(STUDY / "trials/effects.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
    fig.suptitle(
        "What changes when SPP is offered a known target?", weight="bold", fontsize=14
    )
    labels = [
        r["case"].replace("_discovery", "").replace("06_heldout", "")
        + " / "
        + r["timing"]
        for r in rows
    ]
    for i, (key, title) in enumerate(
        (
            ("target_timely", "Selected loads timely"),
            ("cycle_savings", "Cycles saved vs SPP"),
            ("dram_byte_change", "Additional DRAM bytes"),
        )
    ):
        ax = axes[i]
        vals = [int(r[key]) for r in rows]
        bars = ax.barh(
            labels,
            vals,
            color=["#177899" if r["timing"] == "early" else "#bbbfc1" for r in rows],
        )
        ax.bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=8)
        ax.axvline(0, color="#545b66", lw=0.7)
        ax.margins(x=0.30)
        ax.set_title(title, fontsize=10)
        ax.invert_yaxis()
        if i:
            ax.tick_params(labelleft=False)
    save(fig, "timing_interventions")


def service():
    rows = [
        r
        for r in read(STUDY / "resources/request_service.csv")
        if r["claim"] == "late" and int(r["original_first_uses"])
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(10.8, 0.46 * len(rows) + 1.8), layout="constrained"
    )
    fig.suptitle(
        "How early did the conventional request arrive?", weight="bold", fontsize=14
    )
    labels = [r["case"].split("_")[0] + " / " + r["policy"] for r in rows]
    specs = [
        (
            ("proposal_to_admission_p50", "dispatch_to_fill_p50"),
            ("Proposal to demand", "Dispatch to fill"),
        ),
        (
            ("same_page_witness", "earlier_slack_ge0"),
            ("Earlier same-page cue", "Enough lead at observed service"),
        ),
    ]
    for i, (fields, legend) in enumerate(specs):
        ax = axes[i]
        y = np.arange(len(rows))
        for j, (field, label, color) in enumerate(
            zip(fields, legend, ("#177899", "#b25e29"))
        ):
            vals = [int(r[field]) for r in rows]
            if i:
                vals = [
                    100 * v / int(r["original_first_uses"]) for v, r in zip(vals, rows)
                ]
            bars = ax.barh(y + 0.34 * (j - 0.5), vals, 0.31, label=label, color=color)
            ax.bar_label(bars, fmt="%.0f" if i == 0 else "%.1f", padding=3, fontsize=7)
        ax.set_yticks(y, labels if not i else [""] * len(rows))
        ax.invert_yaxis()
        ax.margins(x=0.20)
        ax.set_xlabel(("Median cycles", "Share of sampled late original misses (%)")[i])
        ax.legend(
            fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False
        )
    save(fig, "service_deadlines")


def prospective():
    rows = read(STUDY / "one_offer/effects.csv")
    if len(rows) != 1:
        raise ValueError("This figure describes the isolated BFS replay")
    row = rows[0]
    origin = int(row["offered_at"])
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 2.85), layout="constrained")
    fig.suptitle(
        f"One held-out BFS forecast for physical line {int(row['predicted_line']):,}",
        fontweight="bold", fontsize=14,
    )
    stages = [
        (["Dispatch", "Fill"], [int(row[k]) - origin for k in ("dispatched_at", "filled_at")],
         "Request service", "Cycles after offer"),
        (["Baseline arrival", "Replay arrival", "Replay tag use"],
         [int(row["actual_eighth_lead_cycles"]), int(row["replay_eighth_lead_cycles"]),
          int(row["first_use_at"]) - origin], "Matched demand", "Cycles after cue"),
        (["Gained", "Displaced"], [int(row["timely_gained"]), int(row["timely_displaced"])],
         "All original misses", "Changed timely first uses"),
    ]
    for ax, (labels, values, title, units) in zip(axes, stages):
        if title == "Matched demand":
            ax.scatter(values, labels, color=COLORS[1], zorder=3)
            for y, value in enumerate(values):
                ax.annotate(f"{value:,}", (value, y), xytext=(0, 7),
                            textcoords="offset points", ha="center", fontsize=8)
            ax.set_ylim(len(values) - 0.55, -0.55)
            ax.margins(x=0.3)
            ax.ticklabel_format(style="plain", useOffset=False, axis="x")
            ax.tick_params(axis="x", labelsize=7)
        else:
            bars = ax.barh(labels, values, height=0.48, color=[COLORS[1], COLORS[2]])
            ax.bar_label(bars, fmt="{:,.0f}", padding=3)
            ax.invert_yaxis()
            ax.margins(x=0.22)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(units, fontsize=8)
    save(fig, "prospective_offer")


def main():
    rows = read(STUDY / "expansion/results/performance.csv") + read(
        STUDY / "domains/results/performance.csv"
    )
    panels(
        rows,
        "Demand lookups and execution",
        [
            ("llc_tag_hit_rate", "LLC hit rate (%)", 100, ".1f", "Blues"),
            ("llc_tag_miss_rate", "LLC miss rate (%)", 100, ".1f", "Oranges"),
            ("time_reduction_vs_nopf", "Time saved vs NoPF (%)", 100, "+.2f", "RdBu"),
        ],
        "demand_and_time",
    )
    panels(
        rows,
        "Use of accepted prefetches",
        [
            ("useful_accuracy", "Timely or late per accepted (%)", 100, ".1f", "Blues"),
            (
                "timely_coverage",
                "Direct timely / original misses (%)",
                100,
                ".1f",
                "Greens",
            ),
            (
                "timeliness_among_claims",
                "Timely / (timely + late) (%)",
                100,
                ".1f",
                "Purples",
            ),
        ],
        "prefetch_quality",
    )
    panels(
        rows,
        "Processor and cache traffic",
        [
            ("ipc", "Instructions per cycle", 1, ".2f", "Blues"),
            ("llc_mpki", "LLC demand misses / kinst.", 1, ".2f", "Oranges"),
            ("avg_llc_mshr", "Average LLC MSHR occupancy", 1, ".2f", "Purples"),
        ],
        "ipc_and_pressure",
    )
    pareto(rows)
    trials()
    service()
    prospective()


if __name__ == "__main__":
    main()
