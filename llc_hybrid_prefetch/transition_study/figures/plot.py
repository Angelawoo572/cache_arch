"""Plot one aligned operand sequence and one matched service timeline."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def export_row():
    import numpy as np
    import sys

    sys.path.insert(0, str(STUDY / "episodes"))
    from extract import CLASSES, PATHS

    marks = [r for r in rows(STUDY / "episodes/timeline.csv")
             if r["case"] == "row_period2_paced" and r["view"] == "core_pc"
             and r["example"] == "P" and r["role"] == "last_T"]
    assert len(marks) == 1
    identifier = int(marks[0]["instr"]) * 256 + int(marks[0]["op"])
    data = np.load(STUDY / "raw/episodes/row_period2_paced/c0.npz")
    i = int(np.searchsorted(data["ids"], identifier))
    assert identifier == data["ids"][i]
    with (HERE / "row_example.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["instr", "op", "physical_line", "NoPF", "Stride", "lookup_cycle"])
        for j in range(i - 2, i + 5):
            w.writerow([*divmod(int(data["ids"][j]), 256), data["pa"][j],
                        PATHS[data["base_path"][j]], CLASSES[data["outcome"][j]],
                        data["tag_t"][j]])


def plot():
    data = rows(HERE / "row_example.csv")
    event = next(r for r in rows(STUDY / "timing/examples.csv")
                 if r["late_arm"] == "gap_bfs8_lbm" and r["projection"] == "core")
    fig = plt.figure(figsize=(9.6, 6.1), facecolor="white")
    ax1, ax2 = fig.subplots(2, gridspec_kw={"height_ratios": [1.1, 1]})
    fig.subplots_adjust(left=.15, right=.97, top=.89, bottom=.12, hspace=.81)
    ax1.set_title("A    Row transition · same PC, same physical page", loc="left",
                  fontsize=14, weight="bold", pad=12)
    colors = {"M": ("#e3e7eb", "#192b35"), "T": ("#167666", "white"),
              "P": ("#bb592f", "white")}
    for j, r in enumerate(data):
        assert r["NoPF"] == "llc_miss"
        for y, letter in ((1, "M"), (0, r["Stride"])):
            fill, ink = colors[letter]
            ax1.add_patch(FancyBboxPatch((j-.42,y-.33), .84,.66, boxstyle="round,pad=0.01,rounding_size=.07",
                                         facecolor=fill, edgecolor="none"))
            ax1.text(j, y,letter, ha="center", va="center", color=ink, weight="bold")
    ax1.set_xlim(-.52, len(data)-.48)
    ax1.set_ylim(-.48, 1.5)
    ax1.set_yticks([0, 1], labels=["Stride", "NoPF"])
    ax1.set_xticks(range(len(data)), labels=[r["physical_line"] for r in data])
    ax1.set_xlabel("Physical cache line, in dynamic load order")
    ax1.tick_params(length=0)
    for s in ax1.spines.values():
        s.set_visible(False)
    ax1.text(0, -.48, "M   original miss       T   attributed timely first use       P   persistent miss",
             transform=ax1.transAxes, fontsize=10, color="#374955")
    ax2.set_title("B    BFS-8 · same producer, physical target and consumer", loc="left",
                  fontsize=14, weight="bold", pad=12)
    for y, prefix, name in ((1, "other_", "BFS alone"), (0, "late_arm_", "BFS + LBM")):
        start = int(event[prefix+"proposed"])
        dispatch = int(event["other_dispatched" if y else "late_arm_dispatch"])-start
        fill = int(event[prefix+"fill"])-start
        demand = int(event["other_first_hit" if y else "late_arm_first_merge"])-start
        ax2.plot([0, dispatch],[y,y],color="#526c7c",lw=6,solid_capstyle="butt")
        ax2.plot([dispatch,fill],[y,y],color="#5686a4",lw=6,solid_capstyle="butt")
        ax2.scatter([fill],[y],color="#167666",s=56,zorder=3)
        ax2.plot([demand,demand],[y-.18,y+.18],color="#b14e28",lw=2)
        for x,t,dy in ((fill,"fill",.21),(demand,"demand lookup" if y else "demand merge",-.25)):
            ax2.text(x,y+dy,f"{t} {x}",va="center",ha="center",fontsize=9)
    ax2.set_xlim(-6, 380)
    ax2.set_ylim(-.48,1.47)
    ax2.set_yticks([0,1],labels=["BFS + LBM","BFS alone"])
    ax2.set_xlabel("Cycles since Stride proposed physical line " + f'{int(event["physical_line"]):,}')
    ax2.spines[["top","right","left"]].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    ax2.grid(axis="x", alpha=.15)
    fig.savefig(HERE / "transition_examples.png", dpi=165, metadata={"Software":"Matplotlib"})
    plt.close(fig)



def plot_lbm():
    witnesses = rows(STUDY / "replay/lbm_transfer_witness.csv")
    witness = next(r for r in witnesses if r["selection"] == "four_open_controls_without_timely"
                   and r["recording"] == "lbm-r2" and r["context"] == "exact")
    competitors = rows(STUDY / "replay/lbm_witness_controls.csv")
    events = [("PC + offset", int(witness["proposal"]), int(witness["fill"]), int(witness["target_ingress"]))]
    for key, name in (("multi", "Multi-delta"), ("spp", "SPP")):
        r = next(r for r in competitors if r["target_instr"] == witness["target_instr"] and r["control"] == key)
        assert r["target_line"] == witness["target_line"] and r["classification"] == "late"
        events.append((name, int(r["proposed"]), int(r["fill"]), int(r["demand_ingress"])))

    fig, ax = plt.subplots(figsize=(9.0, 3.4), facecolor="white")
    fig.subplots_adjust(left=.18, right=.92, top=.81, bottom=.20)
    ax.set_title("LBM R2   ·   When each prefetcher issued and filled the same line", loc="left",
                 fontsize=14, weight="bold", pad=16)
    ax.axvline(0, color="#5d646c", lw=1.2, linestyle="--")
    ax.text(0, 2.44, "Demand arrives", color="#3c505c", ha="center", fontsize=9, bbox=dict(facecolor="white", edgecolor="none", pad=1))
    for j, (_, start, end, demand) in enumerate(events):
        y = 2 - j
        start -= demand
        end -= demand
        color = "#167666" if end < 0 else "#b2512f"
        ax.plot([start,end],[y,y], lw=3, color=color)
        ax.scatter([start],[y], s=45, color=color, marker="|", zorder=3)
        ax.scatter([end],[y], s=43, color=color, zorder=3)
        ax.text(start, y-.23, f"issue {start:+,}", ha="center", fontsize=9, color="#33454f")
        ax.text(end, y+.16, f"fill {end:+,}", ha="center", fontsize=9, color=color)
    ax.set_yticks([2,1,0], labels=[r[0] for r in events])
    ax.set_xlim(-2300,300)
    ax.set_ylim(-.48,2.59)
    ax.set_xlabel("Cycles relative to the same demand's LLC ingress in each execution")
    ax.spines[["left","top","right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(HERE / "lbm_lookahead.png", dpi=165, metadata={"Software":"Matplotlib"})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true", help="refresh the row sample from paired raw operands")
    if parser.parse_args().extract:
        export_row()
    plot()
    plot_lbm()
