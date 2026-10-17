#!/usr/bin/env python3
"""Generate the offline-primary report figures and optional live appendix."""

import argparse
import json
import os
from pathlib import Path


COLORS = {8: "#0072B2", 16: "#D55E00"}
MARKERS = {8: "o", 16: "s"}


def load(path):
    return json.loads(Path(path).read_text())


def rows_for(rows, hidden, field):
    return sorted(
        (
            row for row in rows
            if row.get("hidden_size") == hidden
            and row.get(field) is not None
        ),
        key=lambda row: row["instruction_budget"],
    )


def curve(ax, rows, hidden, field, scale=1.0, label=None):
    selected = rows_for(rows, hidden, field)
    if not selected:
        return
    ax.plot(
        [row["instruction_budget"] for row in selected],
        [scale * row[field] for row in selected],
        marker=MARKERS[hidden], color=COLORS[hidden], linewidth=1.8,
        markersize=4.5, label=label or "h{}".format(hidden),
    )


def save(fig, path):
    fig.tight_layout()
    fig.savefig(str(path), dpi=190, bbox_inches="tight")


def best_aggressive(conclusion):
    candidates = conclusion.get(
        "offline_high_performing_small_budget_candidates"
    ) or []
    return max(
        candidates,
        key=lambda row: row.get(
            "relative_ipc_uplift_vs_same_h_20m", float("-inf")
        ),
    ) if candidates else None


def plot_offline_ipc(plt, analysis, out):
    rows = analysis["offline_points"]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    budgets = [
        row["instruction_budget"] for row in rows if row.get("ipc") is not None
    ]
    left, right = min(budgets), max(budgets)
    for hidden in (8, 16):
        curve(ax, rows, hidden, "ipc")
        conclusion = analysis["conclusions"]["h{}".format(hidden)]
        reference = conclusion.get("offline_20m_ipc")
        if reference is not None:
            low, high = reference * 0.995, reference * 1.005
            ax.fill_between(
                [left, right], [low, low], [high, high],
                color=COLORS[hidden], alpha=0.07,
                label="h{} i20m +/-0.5%".format(hidden),
            )
            ax.axhline(
                reference, color=COLORS[hidden], linestyle="--",
                linewidth=1.0, alpha=0.8,
            )
        plateau = conclusion.get("offline_stable_plateau")
        if plateau:
            budget = plateau["instruction_budget"]
            point = next(
                row for row in rows
                if row.get("hidden_size") == hidden
                and row.get("instruction_budget") == budget
            )
            ax.annotate(
                "h{} stable plateau: {}".format(
                    hidden, plateau["budget_tag"]
                ),
                xy=(budget, point["ipc"]), xytext=(8, -18 if hidden == 8 else 14),
                textcoords="offset points", fontsize=8,
                color=COLORS[hidden],
                arrowprops={"arrowstyle": "->", "color": COLORS[hidden]},
            )
        aggressive = best_aggressive(conclusion)
        if aggressive:
            ax.scatter(
                [aggressive["instruction_budget"]], [aggressive["ipc"]],
                marker="*", s=150, color=COLORS[hidden], edgecolor="black",
                linewidth=0.5, zorder=6,
                label="h{} aggressive {}".format(
                    hidden, aggressive["budget_tag"]
                ),
            )
    refs = analysis.get("references", {}).get("offline", {})
    for name, style, color in (
        ("no_pref", ":", "#444444"),
        ("offline_stride", "-.", "#009E73"),
        ("live_stride", (0, (5, 2)), "#7A5195"),
    ):
        value = (refs.get(name) or {}).get("ipc")
        if value is not None:
            ax.axhline(
                value, linestyle=style, color=color, linewidth=1.4,
                label=name.replace("_", " "),
            )
    ax.set_xscale("log")
    ax.set_xlabel("trace-start retired training instructions")
    ax.set_ylabel("offline keyed-replay IPC")
    ax.set_title("Fair offline IPC curve (same-H comparisons)")
    ax.grid(True, alpha=0.23)
    ax.legend(fontsize=7.5, ncol=2, loc="best")
    save(fig, out / "01_fair_offline_ipc_curve.png")


def plot_differences(plt, analysis, out):
    rows = analysis["offline_points"]
    panels = (
        ("relative_ipc_difference_vs_same_h_20m", 100.0,
         "IPC difference (%)"),
        ("coverage_percentage_point_difference_vs_same_h_20m", 1.0,
         "coverage difference (pp)"),
        ("l2_miss_rate_percentage_point_difference_vs_same_h_20m", 1.0,
         "L2 miss-rate difference (pp)"),
        ("request_pressure_difference_vs_same_h_20m", 1.0,
         "requests/L2-load difference"),
        ("student_act_rate_percentage_point_difference_vs_same_h_20m", 1.0,
         "student act-rate difference (pp)"),
    )
    fig, axes = plt.subplots(5, 1, figsize=(8.2, 10.3), sharex=True)
    for ax, (field, scale, ylabel) in zip(axes, panels):
        for hidden in (8, 16):
            curve(ax, rows, hidden, field, scale=scale)
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
        if field == "relative_ipc_difference_vs_same_h_20m":
            ax.axhspan(-0.5, 0.5, color="#009E73", alpha=0.08)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.grid(True, alpha=0.22)
    axes[0].legend(ncol=2, fontsize=8)
    axes[-1].set_xscale("log")
    axes[-1].set_xlabel("trace-start retired training instructions")
    fig.suptitle(
        "Difference from corresponding 20M model (h8->h8, h16->h16)",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.94)
    save(fig, out / "02_differences_from_same_h_20m.png")


def plot_supervision(plt, run_dir, out):
    data = load(run_dir / "training_prefix_data_summary.json")
    rows = sorted(data["points"], key=lambda row: row["instruction_budget"])
    panels = (
        ("decision_rows", "decision rows"),
        ("positive_count_rows", "teacher K>0 rows"),
        ("action_atoms", "teacher action atoms = sum(K)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.35))
    for ax, (field, label) in zip(axes, panels):
        ax.plot(
            [row["instruction_budget"] for row in rows],
            [row[field] for row in rows],
            marker="o", markersize=3.5, color="#009E73", linewidth=1.6,
        )
        ax.set_xscale("log")
        ax.set_xlabel("training instructions", fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, alpha=0.22)
        ax.tick_params(labelsize=7)
    fig.suptitle(
        "Training supervision (teacher-label statistics, not student act rate)",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.86)
    save(fig, out / "03_training_supervision.png")


def plot_live_appendix(plt, analysis, out):
    rows = analysis["live_secondary"]["points"]
    if not rows:
        return []
    outputs = []
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for hidden in (8, 16):
        curve(
            ax, rows, hidden, "live_minus_same_checkpoint_offline_ipc",
            label="h{} live_N - offline_N".format(hidden),
        )
    ax.axhline(0, color="black", linestyle=":", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("trace-start retired training instructions")
    ax.set_ylabel("IPC difference")
    ax.set_title("Same-checkpoint implementation parity")
    ax.grid(True, alpha=0.23)
    ax.legend(fontsize=8)
    path = out / "04_live_vs_offline_same_checkpoint.png"
    save(fig, path)
    outputs.append(path)

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for hidden in (8, 16):
        curve(
            ax, rows, hidden,
            "relative_live_ipc_difference_vs_same_h_live_20m",
            scale=100.0, label="h{}".format(hidden),
        )
    ax.axhspan(-0.5, 0.5, color="#009E73", alpha=0.08)
    ax.axhline(0, color="black", linestyle=":", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("trace-start retired training instructions")
    ax.set_ylabel("live IPC difference vs same-H live i20m (%)")
    ax.set_title("Secondary live training-size sufficiency")
    ax.grid(True, alpha=0.23)
    ax.legend(fontsize=8)
    path = out / "05_live_sufficiency_vs_same_h_20m.png"
    save(fig, path)
    outputs.append(path)
    return outputs


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    return parser


def main():
    args = build_parser().parse_args()
    analysis = load(args.run_dir / "offline_sufficiency.json")
    out = args.out_dir or args.run_dir / "report_plots"
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plots: {}".format(exc))
    plot_offline_ipc(plt, analysis, out)
    plot_differences(plt, analysis, out)
    plot_supervision(plt, args.run_dir, out)
    live_outputs = plot_live_appendix(plt, analysis, out)
    print("[ok] wrote 3 primary offline figures to {}".format(out))
    print("[ok] wrote {} optional live appendix figures".format(
        len(live_outputs)
    ))


if __name__ == "__main__":
    main()
