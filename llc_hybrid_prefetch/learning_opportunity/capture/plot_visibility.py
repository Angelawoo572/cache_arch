"""Plot the fraction of original dynamic source loads observed at LLC admission."""

import argparse
import csv
import os

from build import HERE, RAW

os.environ.setdefault("MPLCONFIGDIR", str(RAW / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

NAMES = {
    "mcf17_discovery": "MCF17 · 472B",
    "mcf17_discovery_far": "MCF17 · 472B, later",
    "mcf17_transfer": "MCF17 · 1644B",
    "mcf06_discovery": "MCF06 · 22B",
    "bfs_discovery": "BFS · recording 3",
    "bfs_discovery_far": "BFS · recording 3, later",
    "bfs_transfer": "BFS · recording 14",
    "soplex06_heldout": "SoPlex06",
    "pr_heldout": "PageRank",
}


EXTERNAL = {
    "gcc734": "GCC17 · 734B",
    "cactus2421": "CactuBSSN17 · 2421B",
    "omnet874": "OMNeT++17 · 874B",
    "xalan700": "Xalancbmk17 · 700B",
    "astar42": "Astar06 · 42B",
    "omnet188": "OMNeT++06 · 188B",
    "soplex247": "SoPlex06 · 247B",
    "mcf51": "MCF06 · 51B",
    "bc3": "Betweenness · recording 3",
    "cc5": "Components · recording 5",
    "pr14": "PageRank · recording 14",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external", action="store_true")
    args = parser.parse_args()
    names = EXTERNAL if args.external else NAMES
    policy = "Stride" if args.external else "NoPF"
    prefix = "external_" if args.external else ""
    visible_key = policy.lower() + "_unique_llc_operands"
    with (HERE / (prefix + "source_parity.csv")).open(newline="") as file:
        rows = {r["case"]: r for r in csv.DictReader(file)}
    if rows.keys() != names.keys():
        raise ValueError("Expected every registered recording in the selected panel")
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(9.0, 0.39 * len(names) + 1.0))
    fig.subplots_adjust(left=0.24, right=0.95, bottom=0.17, top=0.89)
    for i, key in enumerate(names):
        record = rows[key]
        total = int(record["exact_operand_keys_per_policy"])
        visible = int(record[visible_key])
        rate = visible / total
        ax.barh(i, rate, color="#336d82", height=0.55)
        ax.text(
            rate + 0.005,
            i,
            f"{rate:.1%}   ({visible / 1e6:.2f}M of {total / 1e6:.2f}M)",
            va="center",
            fontsize=9,
        )
    ax.set_yticks(range(len(names)), labels=names.values())
    ax.invert_yaxis()
    top = max(
        int(r[visible_key]) / int(r["exact_operand_keys_per_policy"])
        for r in rows.values()
    )
    ax.set_xlim(0, max(top + 0.25, 0.4))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Share of measured source load operands admitted to the LLC")
    ax.set_title(
        "Source loads visible at the LLC under " + policy,
        loc="left",
        fontsize=13,
        pad=13,
    )
    fig.savefig(HERE / (prefix + "source_visibility.svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
