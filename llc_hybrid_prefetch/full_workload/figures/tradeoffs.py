"""Plot branch-resident LLC performance trade-offs and replay diagnostics."""

import csv
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
CASES = {
    "mcf06_discovery": "SPEC06 MCF · 22B",
    "soplex06_heldout": "SPEC06 Soplex · 92B",
    "mcf17_discovery": "SPEC17 MCF · 472B start",
    "mcf17_discovery_far": "SPEC17 MCF · 472B end",
    "mcf17_transfer": "SPEC17 MCF · 1644B",
    "bfs_discovery": "GAP BFS · 3 start",
    "bfs_discovery_far": "GAP BFS · 3 end",
    "bfs_transfer": "GAP BFS · 14",
    "pr_heldout": "GAP PageRank · 3",
    "pr_heldout_entire": "GAP PageRank · 3 to EOF",
}
POLICIES = {
    "nopf": ("N", "NoPF", "#475569", "o"),
    "stride": ("S", "Stride", "#215b95", "o"),
    "multi": ("M", "Multi-delta", "#a64b0a", "s"),
    "spp": ("P", "SPP", "#13735b", "^"),
    "temporal": ("T", "Temporal", "#794389", "D"),
}
GAIN, LOSS = "#087d66", "#b64c30"
INK, MUTED = "#172437", "#526378"
TRAFFIC = ("dram_demand_bytes", "dram_prefetch_bytes", "dram_write_bytes")
FOCUS = ("mcf17_discovery", "bfs_discovery", "pr_heldout")


def read(path):
    with (STUDY / path).open() as stream:
        return list(csv.DictReader(stream))


def axes_style(ax, xgrid=False):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#a1afbe")
    ax.set_axisbelow(True)
    ax.grid(axis="x" if xgrid else "y", color="#e4eaf1", linewidth=0.65)
    ax.tick_params(colors=MUTED, length=0)


def headings(fig, title, subtitle):
    fig.text(0.075, 0.98, title, ha="left", va="top", fontsize=19, fontweight="bold")
    subtitle_y = 0.98 - 26 / (72 * fig.get_figheight())
    fig.text(0.075, subtitle_y, subtitle, ha="left", va="top", color=MUTED, fontsize=9.4)


def legend(fig, keys, **kwargs):
    handles = [
        Line2D([], [], color=color, marker=marker, linestyle="none", label=name)
        for key, (_, name, color, marker) in POLICIES.items()
        if key in keys
    ]
    fig.legend(handles=handles, frameon=False, **kwargs)


def save(fig, name):
    fig.savefig(HERE / f"tradeoff_{name}.svg", metadata={"Date": None})
    plt.close(fig)


def percent_change(value, ref):
    return 100 * (Fraction(value, ref) - 1)


def traffic(row):
    return sum(int(row[field]) for field in TRAFFIC)


def is_dominated(point, values):
    x, y = point
    return any(a <= x and b <= y and (a < x or b < y) for a, b in values)


def pareto_panel(ax, case, rows):
    axes_style(ax)
    ax.set_title(CASES[case], loc="left", fontweight="bold", fontsize=10.5, pad=7)
    base = rows[case, "nopf"]
    values = {
        policy: (
            percent_change(traffic(rows[case, policy]), traffic(base)),
            percent_change(int(rows[case, policy]["cycles"]), int(base["cycles"])),
        )
        for policy in POLICIES
    }
    grouped = defaultdict(list)
    for policy, point in values.items():
        grouped[point].append(policy)
    frontier = sorted(p for p in grouped if not is_dominated(p, grouped))
    ax.axhline(0, color="#9dabbb", linewidth=0.7)
    ax.axvline(0, color="#9dabbb", linewidth=0.7)
    if len(frontier) > 1:
        ax.plot(*zip(*frontier), color="#27384d", linewidth=1.25, zorder=3)
    for (x, y), keys in grouped.items():
        _, _, color, marker = POLICIES[keys[0]]
        ax.scatter(x, y, color=color, s=29, marker=marker, zorder=4)
        if (x, y) in frontier:
            ax.scatter(x, y, s=90, facecolor="none", edgecolor="#27384d", linewidth=0.9)
    ax.margins(x=0.16, y=0.20)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))

    box = ax.get_position()
    ax.set_position(box.shrunk(0.65, 1))
    table = ax.figure.add_axes(
        (box.x0 + 0.70 * box.width, box.y0, 0.37 * box.width, box.height)
    )
    table.set_axis_off()
    table.text(0.56, 0.93, "Traffic Δ", fontsize=7.7, color=MUTED, ha="right")
    table.text(1.0, 0.93, "Time Δ", fontsize=7.7, color=MUTED, ha="right")
    for i, ((x, y), keys) in enumerate(grouped.items()):
        code = "/".join(POLICIES[key][0] for key in keys)
        color = POLICIES[keys[0]][2]
        height = 0.72 - 0.14 * i
        table.text(0.02, height, code, color=color, fontsize=8.3, weight="bold")
        table.text(0.56, height, f"{float(x):+.2f}%", ha="right", fontsize=8.3)
        table.text(1.0, height, f"{float(y):+.2f}%", ha="right", fontsize=8.3)


def pareto(rows):
    fig, axes = plt.subplots(4, 3, figsize=(15.8, 12.7))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.09, top=0.90, hspace=0.48, wspace=0.32)
    headings(
        fig,
        "Pareto frontiers for execution time and DRAM traffic",
        "Percent change from NoPF. Lower is better on both axes. Rings identify nondominated points within each panel.",
    )
    for ax, case in zip(axes.flat, CASES):
        pareto_panel(ax, case, rows)
    for ax in axes.flat[len(CASES):]:
        ax.axis("off")
    fig.text(
        0.36, 0.171,
        "Primary panels retire 200M instructions each. The EOF panel retires 450,000,001 \n"
        "and overlaps the primary PageRank window. Each run warms up for 50M instructions.\n\n"
        "Traffic includes demand reads, prefetch reads, and writes. Tied policies share\n"
        "a marker. Connecting segments link measured frontier points.",
        fontsize=10,
        va="center",
        color=MUTED,
    )
    fig.supxlabel("Total DRAM traffic change (%)", y=0.060, fontsize=10)
    fig.supylabel("Fixed-work cycle change (%)", x=0.023, fontsize=10)
    legend(fig, POLICIES, loc="lower center", bbox_to_anchor=(0.5, 0.012), ncols=5)
    save(fig, "pareto")


def metrics_vs_time(rows):
    metrics = {
        "useful_accuracy": "Useful accuracy (%)",
        "timely_coverage": "Timely coverage (%)",
        "timeliness_among_claims": "Timeliness (%)",
    }
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharey="col", sharex="row")
    fig.subplots_adjust(left=0.11, right=0.965, bottom=0.15, top=0.89, hspace=0.43, wspace=0.31)
    headings(
        fig,
        "Accuracy, coverage, and timeliness versus speed",
        "Three 200M-instruction cases. Each label gives the x metric and cycle reduction. The y scale varies by workload.",
    )
    for col, case in enumerate(FOCUS):
        axes[0, col].set_title(CASES[case], loc="left", fontweight="bold", fontsize=11)
        for ax, (field, title) in zip(axes[:, col], metrics.items()):
            axes_style(ax)
            ax.axhline(0, color="#9dabbb", linewidth=0.7)
            points = []
            for policy in ("stride", "multi", "spp"):
                row = rows[case, policy]
                points.append((
                    100 * float(row["time_reduction_vs_nopf"]),
                    100 * float(row[field]), policy,
                ))
            lower, upper = (-1.60, 0.35) if case == "pr_heldout" else (-2.7, 22.5)
            span = upper - lower
            edge, separation = 0.045 * span, 0.115 * span
            last = lower - separation
            for index, (y, x, policy) in enumerate(sorted(points)):
                code, _, color, marker = POLICIES[policy]
                side = -1 if x > 67 else 1
                label_y = min(
                    max(y + edge, last + separation, lower + edge),
                    upper - edge - (len(points) - 1 - index) * separation,
                )
                last = label_y
                ax.scatter(x, y, color=color, marker=marker, s=37, zorder=3)
                ax.annotate(
                    f"{code}  {x:.1f}%, {y:+.2f}%", xy=(x, y),
                    xytext=(x + side * 4, label_y),
                    ha="right" if side < 0 else "left", va="center", color=color,
                    fontsize=8.1,
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 0.6},
                )
            ax.set_xlabel(title)
            ax.set_xlim(-5, 105)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.tick_params(labelbottom=True)
            ax.set_ylim(lower, upper)
            ax.yaxis.set_major_locator(MaxNLocator(5))
    fig.supylabel("Fixed-work cycle reduction vs NoPF (%)", x=0.030, fontsize=11)
    fig.text(
        0.075, 0.060,
        "Useful accuracy = (timely + late first uses) / accepted requests. "
        "Timely coverage = matched timely first uses / original NoPF LLC misses.\n"
        "Timeliness = timely / (timely + late first uses). "
        "NoPF and this temporal control accepted zero requests and have undefined accuracy and timeliness.",
        va="top",
        color=MUTED,
        fontsize=9.2,
    )
    legend(fig, ("stride", "multi", "spp"), loc="lower right", bbox_to_anchor=(0.97, 0.08), ncols=3)
    save(fig, "metrics_vs_time")


def spp_timing():
    rows = {
        (r["case"], r["claim"], r["interval_cycles"]): r
        for r in read("native/sampled_timeliness.csv")
        if r["policy"] == "spp"
    }
    fields = {
        "request_to_demand_ingress": "Request to demand ingress",
        "dispatch_to_fill": "Dispatch to fill",
        "fill_margin_at_demand_ingress": "Fill margin at demand ingress",
    }
    fig, axes = plt.subplots(
        1, 3, figsize=(15.5, 5.7), sharey=True,
        gridspec_kw={"width_ratios": [1.04, 0.93, 1.05]},
    )
    fig.subplots_adjust(left=0.26, right=0.95, bottom=0.21, top=0.76, wspace=0.50)
    headings(
        fig,
        "Lead time and service time under SPP",
        "Points are medians. Lines span the 10th to 90th percentiles. Labels give the three quantiles and observed sample size.",
    )
    for ax, (field, title) in zip(axes, fields.items()):
        axes_style(ax, xgrid=True)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10.6, pad=10)
        for index, case in enumerate(FOCUS):
            for offset, (claim, color) in enumerate((("timely", GAIN), ("late", LOSS))):
                row = rows[case, claim, field]
                low, med, high = (int(row[q]) for q in ("p10", "p50", "p90"))
                y = index * 2.5 + offset
                ax.hlines(y, low, high, color=color, linewidth=2, alpha=0.7)
                ax.scatter(med, y, color=color, s=27, zorder=3)
                ax.annotate(
                    f"{low:,} / {med:,} / {high:,}  ·  n={int(row['observed']):,}",
                    (0.98, y), xycoords=("axes fraction", "data"), xytext=(0, -11),
                    textcoords="offset points", ha="right", color=color, fontsize=7.9,
                )
        if field == "fill_margin_at_demand_ingress":
            ax.axvline(0, color="#a1afbe", linewidth=1)
            ax.set_xscale("symlog", linthresh=50)
            ax.set_xticks([-100, 0, 100, 1_000, 10_000])
            ax.set_xlabel("Cycles (symmetric log scale)")
        elif field == "request_to_demand_ingress":
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1_000, 10_000]))
            ax.set_xlabel("Cycles (log scale)")
        else:
            ax.set_xticks([70, 100, 150, 200])
            ax.set_xlabel("Cycles")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.tick_params(which="minor", length=0)
        ax.set_yticks([i * 2.5 + 0.5 for i in range(len(FOCUS))], [CASES[c] for c in FOCUS])
        ax.set_ylim(6.6, -0.62)
    fig.legend(
        handles=[
            Line2D([], [], color=GAIN, marker="o", label="Timely first use"),
            Line2D([], [], color=LOSS, marker="o", label="Late merge"),
        ],
        frameon=False, loc="upper right", bbox_to_anchor=(0.95, 0.865), ncols=2,
    )
    fig.text(
        0.075, 0.045,
        "SPP degree one. Four 250K-instruction logging islands per 200M window. Each interval uses requests with both endpoints visible.\n"
        "The margin is demand-ingress cycle minus fill cycle. A negative margin means the fill followed demand ingress.",
        fontsize=9.0, color=MUTED,
    )
    save(fig, "spp_timing")


def displacement():
    joined = {
        r["case"]: r for r in read("discovery/full_complete_load_alignment.csv")
        if r["reference"] == "stride" and r["actual_arm"] == "shape"
    }
    perf = {
        r["case"]: r for r in read("discovery/full_performance.csv") if r["arm"] == "shape"
    }
    cases = [c for c in CASES if c in joined]
    ys = list(range(len(cases) - 1)) + [len(cases) - 0.55]
    gains = [int(joined[c]["original_timely_gained"]) for c in cases]
    losses = [-int(joined[c]["original_timely_lost"]) for c in cases]
    savings = [float(perf[c]["savings_vs_stride_pct"]) for c in cases]
    fig, (counts, cycles) = plt.subplots(
        1, 2, figsize=(14, 6.2), sharey=True, gridspec_kw={"width_ratios": [1.5, 0.9]},
    )
    fig.subplots_adjust(left=0.22, right=0.95, bottom=0.17, top=0.80, wspace=0.25)
    headings(
        fig,
        "Preempting a Stride slot can gain and lose timely hits",
        "Frozen address generator with warmed Stride fallback. Every comparison uses the same retired work and recording.",
    )
    for ax, title in ((counts, "Original LLC misses"), (cycles, "Fixed-work cycle reduction")):
        axes_style(ax, xgrid=True)
        ax.set_title(title, loc="left", fontsize=10.8, fontweight="bold", pad=10)
        ax.axvline(0, color="#8d9bad", linewidth=0.8)
        ax.axhline(ys[-1] - 0.7, color="#a6b2c1", linestyle="--", linewidth=0.8)
    for values, color, side in ((gains, GAIN, 1), (losses, LOSS, -1)):
        counts.barh(ys, [v / 1_000 for v in values], height=0.6, color=color)
        for y, value in zip(ys, values):
            counts.annotate(
                f"{abs(value):,}", (value / 1_000, y), xytext=(side * 4, 0),
                textcoords="offset points", ha="left" if side > 0 else "right",
                va="center", color=color, fontsize=8.4,
            )
    counts.set_xlim(-160, 112)
    counts.set_xlabel("Newly timely (right) and no longer timely (left), thousands")
    cycles.scatter(savings, ys, color=INK, s=25, zorder=3)
    for y, value in zip(ys, savings):
        cycles.annotate(
            f"{value:+.4f}%" if value else "0%",
            (value, y), textcoords="offset points", xytext=(5, 0), va="center", fontsize=8.6,
        )
    cycles.set_xlim(-0.86, 0.27)
    cycles.set_xlabel("Cycle reduction vs Stride (%)")
    counts.set_yticks(ys, [CASES[c] for c in cases])
    counts.set_ylim(ys[-1] + 0.68, -0.7)
    fig.text(
        0.075, 0.046,
        "The first seven comparisons retire 200M instructions each. The separate EOF comparison retires 450,000,001\n"
        "and includes the first PageRank interval. Coverage is attributed to timely first use of a completed prefetch.",
        color=MUTED, fontsize=9.2,
    )
    save(fig, "displacement")



def address_examples():
    bfs = read("discovery/bfs_fixed3_witness.csv")
    mcf = [
        r for r in read("streams/mcf_transfer/target_details.csv")
        if r["case"] == "mcf17_discovery_far" and r["target_instr"] == "1800071866"
        and r["target_op"] == "0"
    ]
    bfs_names = {
        "nopf": "NoPF", "stride": "Stride", "multi": "Multi-delta", "spp": "SPP",
        "shape": "Frozen rule + Stride", "sparse3": "Fixed +3 + Stride",
    }
    mcf_names = {"shared_increment": "Cross-PC increment", "local": "Same-PC increment"}
    panels = (
        ("BFS-3  ·  target line 3,500,606 at instruction 100,101,506", bfs, bfs_names,
         ("generation", "dispatch", "fill"), "ingress", "outcome"),
        ("MCF-472B end  ·  target line 53,184 at instruction 1,800,071,866", mcf, mcf_names,
         ("actual_generate_cycle", "actual_dispatch_cycle", "actual_fill_cycle"),
         "actual_target_ingress_cycle", "actual_use"),
    )
    palette = (("#215b95", "o", "Generate"), ("#794389", "v", "Dispatch"), (GAIN, "s", "Fill"))
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), gridspec_kw={"height_ratios": [2.7, 1]})
    fig.subplots_adjust(left=0.23, right=0.73, top=0.81, bottom=0.16, hspace=0.50)
    headings(
        fig,
        "Two dynamic loads, observed request timelines",
        "Each row uses its own demand LLC ingress as cycle zero. The timeline below shows relative cycles.",
    )
    for ax, (title, rows, names, fields, target, status) in zip(axes, panels):
        axes_style(ax, xgrid=True)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10.8, pad=10)
        ax.set_xscale("symlog", linthresh=30)
        ax.set_xlim(-4_300, 95)
        ax.set_xticks([-3_000, -1_000, -300, -100, 0, 50])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.axvline(0, color="#a1afbe", linewidth=0.85)
        ax.set_ylim(len(rows) - 0.55, -0.75)
        ax.set_yticks(range(len(rows)), [names[row["arm"]] for row in rows])
        ax.annotate(
            "Generate / dispatch / fill  ·  Outcome", (1.03, 1),
            xycoords="axes fraction", ha="left", va="bottom", fontsize=8.5, color=MUTED,
        )
        for y, row in enumerate(rows):
            demand_cycle = int(row[target])
            relative = [int(row[f]) - demand_cycle if row[f] else None for f in fields]
            if relative[0] is not None and relative[-1] is not None:
                ax.hlines(y, relative[0], relative[-1], color="#90a2b6", linewidth=1.25)
            for value, (color, marker, _) in zip(relative, palette):
                if value is not None:
                    ax.scatter(value, y, color=color, marker=marker, s=29, zorder=3)
            ax.scatter(0, y, color=INK, marker="|", s=95, zorder=4)
            label = " / ".join("—" if x is None else f"{x:+,}" for x in relative)
            result = {
                "timely": "timely first use", "late": "late merge", "miss": "demand miss",
                "no_attributed_first_use": "no attributed first use",
            }[row[status]]
            ax.annotate(
                f"{label}  ·  {result}", (1.03, y), xycoords=("axes fraction", "data"),
                ha="left", va="center", fontsize=8.5,
            )
    axes[-1].set_xlabel("Cycles relative to demand ingress (symmetric log scale)")
    fig.legend(
        handles=[
            Line2D([], [], color=color, marker=marker, linestyle="none", label=name)
            for color, marker, name in (*palette, (INK, "|", "Demand ingress"))
        ],
        loc="lower left", bbox_to_anchor=(0.068, 0.030), ncols=4, frameon=False,
    )
    save(fig, "address_examples")


def core_interference():
    rows = {(r["policy"], r["program"]): r for r in read("interference/per_core.csv")}
    policies = ("nopf", "stride", "multi", "spp")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.15), sharey=True)
    fig.subplots_adjust(left=0.19, right=0.93, top=0.78, bottom=0.21, wspace=0.30)
    headings(
        fig,
        "Sharing the memory system changes each core's result",
        "MCF and BFS retire 20M instructions per core after 5M warmup. Solo and paired runs use the same physical placement.",
    )
    colors = {"mcf": "#845194", "bfs": "#187493"}
    for ax, title in zip(axes, ("Cost of sharing vs solo", "Time change vs paired NoPF")):
        axes_style(ax, xgrid=True)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10.7, pad=9)
        ax.axvline(0, color="#a1afbe", linewidth=0.8)
    for i, policy in enumerate(policies):
        for j, (program, color) in enumerate(colors.items()):
            row = rows[policy, program]
            solo = int(row["solo_cycles"])
            paired = int(row["paired_cycles"])
            nopf = int(rows["nopf", program]["paired_cycles"])
            for ax, ref in zip(axes, (solo, nopf)):
                x = float(percent_change(paired, ref))
                y = i + (j - 0.5) * 0.34
                ax.barh(y, x, height=0.28, color=color)
                side = 1 if x >= 0 else -1
                ax.annotate(
                    f"{x:+.1f}%", (x, y), xytext=(side * 4, 0),
                    textcoords="offset points", ha="left" if side > 0 else "right",
                    va="center", color=color, fontsize=8.6,
                )
    axes[0].set_xlim(-2, 38)
    axes[1].set_xlim(-33, 10)
    axes[0].set_yticks(range(len(policies)), [POLICIES[p][1] for p in policies])
    axes[0].invert_yaxis()
    for ax in axes:
        ax.set_xlabel("Fixed-work cycle change (%)")
    fig.legend(
        handles=[
            Line2D([], [], color=color, marker="s", linestyle="none", label=program.upper())
            for program, color in colors.items()
        ],
        loc="lower right", bbox_to_anchor=(0.93, 0.06), ncols=2, frameon=False,
    )
    fig.text(
        0.075, 0.056,
        "Positive values mean slower completion. The right panel compares policies under shared execution.",
        color=MUTED, fontsize=9.2,
    )
    save(fig, "core_interference")


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK,
        "svg.fonttype": "none",
        "svg.hashsalt": "llc-tradeoffs",
    })
    rows = {
        (r["case"], r["policy"]): r
        for r in read("native/performance.csv") + read("native/entire/performance.csv")
    }
    pareto(rows)
    metrics_vs_time(rows)
    spp_timing()
    displacement()
    address_examples()
    core_interference()


if __name__ == "__main__":
    main()
