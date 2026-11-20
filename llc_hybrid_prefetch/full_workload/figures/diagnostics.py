"""Plot phase sensitivity, cross-PC outcomes and sampling effects."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot import LABELS, STYLE

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
GREEN = "#137a65"
RED = "#a9432d"
BLUE = "#235889"
GREY = "#526279"


def read_csv(relative):
    with (STUDY / relative).open(newline="") as file:
        return list(csv.DictReader(file))


def save(fig, filename, source):
    fig.text(0.015, 0.012, f"Source: {source}", fontsize=8, color=GREY)
    fig.savefig(HERE / filename, metadata={"Date": None})
    plt.close(fig)


def quarter_plot():
    source = "native/quarters.csv"
    quarters = {
        (row["case"], row["policy"], int(row["quarter"])): int(row["cycles"])
        for row in read_csv(source)
    }
    policies = ("stride", "multi", "spp", "temporal")
    rows = [(case, policy) for case in LABELS for policy in policies]
    values = [
        [
            100 * (1 - quarters[case, policy, q] / quarters[case, "nopf", q])
            for q in range(4)
        ]
        for case, policy in rows
    ]

    fig, ax = plt.subplots(figsize=(8.5, 11.3), layout="constrained")
    fig.set_constrained_layout_pads(h_pad=0.3)
    cbar = ax.imshow(values, cmap="RdYlGn", vmin=-22, vmax=22, aspect="auto")
    bar = fig.colorbar(cbar, ax=ax, shrink=0.75, extend="both")
    bar.set_label("Time saved vs NoPF (%)")
    ax.set_title("Fixed-work time varies across 50M-instruction quarters", loc="left", pad=16)
    names = [
        f"{LABELS[case]}   {STYLE[policy][0].split(',')[0]}"
        if policy == "stride" else STYLE[policy][0].split(",")[0]
        for case, policy in rows
    ]
    ax.set_xticks(range(4), labels=["Q1", "Q2", "Q3", "Q4"])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), labels=names)
    ax.tick_params(length=0)
    for i, row in enumerate(values):
        if i % len(policies) == 0:
            ax.axhline(i - 0.5, color="white", linewidth=2)
        for j, value in enumerate(row):
            ax.text(j, i, f"{0 if abs(value) < 0.005 else value:+.2f}%",
                    ha="center", va="center", color="white" if abs(value) > 16 else "#13212b",
                    fontsize=8)
    save(fig, "diagnostic_quarters.svg", source)


def mcf_plot():
    targets_source = "streams/mcf_transfer/target_outcomes.csv"
    paired_source = "streams/mcf_transfer/paired.csv"
    cases = ("mcf17_discovery", "mcf17_discovery_far", "mcf17_transfer")
    labels = [LABELS[case].removeprefix("SPEC17 ") for case in cases]
    targets = {
        (r["case"], r["arm"]): r for r in read_csv(targets_source)
    }
    times = {
        r["case"]: float(r["time_saved_pct"]) for r in read_csv(paired_source)
        if r["interval"] == "all" and r["reference"] == "local"
    }
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.65))
    fig.subplots_adjust(left=0.20, right=0.96, top=0.78, bottom=0.22, wspace=0.43)
    fig.suptitle("Cross-PC increments can generate timely targets without consistent time savings",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    y = list(range(3))
    for ax in (left, right):
        ax.set_yticks(y, labels)
        ax.set_ylim(2.5, -0.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#e6e9ec")
        ax.set_axisbelow(True)
    counts = [int(targets[case, "shared_increment"]["timely_arithmetic"]) for case in cases]
    left.barh(y, counts, height=0.45, color=BLUE)
    left.set_xlim(0, 32)
    left.set_xlabel("Sampled targets made timely")
    left.set_title("Independent requests", loc="left", fontsize=10)
    for i, (case, count) in enumerate(zip(cases, counts)):
        total = targets[case, "shared_increment"]["original_sampled_targets"]
        local = targets[case, "local"]["timely"]
        left.text(count + 0.45, i, f"{count}/{total}\nLocal {local}", va="center", fontsize=8)
    effects = [times[case] for case in cases]
    right.barh(y, effects, height=0.45, color=[GREEN if v > 0 else RED for v in effects])
    right.axvline(0, color=GREY, linewidth=0.9)
    right.set_xlim(-0.90, 0.30)
    right.set_xlabel("Time saved vs local increment (%)")
    right.set_title("Complete 200M-instruction interval", loc="left", fontsize=10)
    for i, v in enumerate(effects):
        right.text(v + (0.028 if v >= 0 else -0.028), i, f"{v:+.3f}%",
                   ha="left" if v >= 0 else "right", va="center", fontsize=8)
    save(fig, "diagnostic_mcf_cross_pc.svg", f"{targets_source} and {paired_source}")


def sampling_plot():
    source = "continuous/counts.csv"
    cases = ("bfs_discovery", "soplex06_heldout")
    cohorts = {
        (r["case"], r["window"]): r for r in read_csv(source)
        if r["order"] == "arrival" and r["scope"] == "global" and r["cohort"] == "nopf_miss"
    }
    fig, ax = plt.subplots(figsize=(8.5, 3.75))
    fig.subplots_adjust(left=0.12, right=0.93, bottom=0.22, top=0.76)
    fig.suptitle("Short windows change the estimated rate of misses after timely hits",
                 x=0.03, ha="left", fontsize=12, fontweight="bold")
    for offset, window, title, color in (
        (-0.18, "full", "Full arrival stream", BLUE),
        (0.18, "four_islands", "Four short windows", "#d2914c"),
    ):
        positions = [i + offset for i in range(len(cases))]
        rates = []
        counts = []
        for case in cases:
            row = cohorts[case, window]
            n = int(row["adjacent_after_direct"])
            count = int(row["next_late"]) + int(row["next_plain_miss"])
            rates.append(100 * count / n)
            counts.append((count, n))
        ax.bar(positions, rates, width=0.30, color=color, label=title)
        for x, rate, (count, n) in zip(positions, rates, counts):
            ax.text(x, rate + 2, f"{rate:.2f}%\n{count:,}/{n:,}",
                    ha="center", va="bottom", fontsize=8)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ax.set_xticks(range(len(cases)), [LABELS[case] for case in cases])
    ax.set_ylim(0, 79)
    ax.set_ylabel("Late or plain miss after a timely use (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e9ec")
    ax.set_axisbelow(True)
    save(fig, "diagnostic_sampling.svg", source)


def main():
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "llc-diagnostics"})
    quarter_plot()
    mcf_plot()
    sampling_plot()


if __name__ == "__main__":
    main()
