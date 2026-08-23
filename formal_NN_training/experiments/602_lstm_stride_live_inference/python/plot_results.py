#!/usr/bin/env python3
"""Generate offline-primary figures plus secondary deployment diagnostics."""

import argparse
import json
import os
from pathlib import Path

from analysis_policy import reference_20m, stable_plateau


COLORS = {8: "#0072B2", 16: "#D55E00"}


def load(path):
    if not Path(path).is_file():
        return {"points": [], "references": {}}
    return json.loads(Path(path).read_text())


def points_for(rows, hidden, field):
    selected = [
        row for row in rows
        if row.get("hidden_size") == hidden
        and row.get(field) is not None
    ]
    return sorted(selected, key=lambda row: row["instruction_budget"])


def curve(ax, rows, field, label_prefix="", marker="o"):
    for hidden in (8, 16):
        selected = points_for(rows, hidden, field)
        if selected:
            ax.plot(
                [row["instruction_budget"] for row in selected],
                [row[field] for row in selected],
                marker=marker,
                color=COLORS[hidden],
                label="{}h{}".format(label_prefix, hidden),
            )
    ax.set_xscale("log")
    ax.set_xlabel("retired training instructions")
    ax.grid(True, alpha=0.25)
    ax.legend()


def status_markers(ax, rows):
    failed = [
        row for row in rows
        if row.get("ipc") is None and row.get("instruction_budget")
    ]
    for hidden in (8, 16):
        selected = [
            row for row in failed if row.get("hidden_size") == hidden
        ]
        if selected:
            ax.scatter(
                [row["instruction_budget"] for row in selected],
                [0 for _ in selected],
                marker="x", color=COLORS[hidden],
                label="h{} untrainable/failed".format(hidden),
            )


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    fig.clf()


def main_offline_ipc_figure(plt, offline, references, out):
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for hidden in (8, 16):
        rows = points_for(offline, hidden, "ipc")
        reference = reference_20m(rows)
        if not rows or reference is None:
            continue
        x = [row["instruction_budget"] for row in rows]
        y = [row["ipc"] for row in rows]
        ax.plot(x, y, marker="o", color=COLORS[hidden], label="h{} offline keyed replay".format(hidden))
        lower, upper = reference["ipc"] * 0.995, reference["ipc"] * 1.005
        ax.fill_between(x, [lower] * len(x), [upper] * len(x),
                        color=COLORS[hidden], alpha=0.09,
                        label="h{} i20m +/-0.5%".format(hidden))
        plateau, _ = stable_plateau(rows, reference, 0.005)
        if plateau:
            ax.annotate("h{} stable plateau starts {}".format(hidden, plateau["budget_tag"]),
                        (plateau["instruction_budget"], plateau["ipc"]),
                        xytext=(6, 12), textcoords="offset points", fontsize=8,
                        color=COLORS[hidden])
        aggressive = "i250k" if hidden == 8 else "i100k"
        candidate = next((row for row in rows if row["budget_tag"] == aggressive), None)
        if candidate:
            ax.scatter([candidate["instruction_budget"]], [candidate["ipc"]],
                       marker="*", s=150, color=COLORS[hidden], edgecolor="black",
                       zorder=5)
            ax.annotate("{} aggressive candidate".format(aggressive),
                        (candidate["instruction_budget"], candidate["ipc"]),
                        xytext=(6, -16), textcoords="offset points", fontsize=8)
    for name in ("no_pref", "offline_stride"):
        reference = references.get(name, {})
        if reference.get("ipc") is not None:
            ax.axhline(reference["ipc"], linestyle="--", alpha=0.55,
                       label=name.replace("_", " "))
    status_markers(ax, offline)
    ax.set_xscale("log")
    ax.set_xlabel("retired training instructions")
    ax.set_ylabel("offline keyed-replay IPC")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    save(fig, out / "main_01_offline_ipc_equivalence.png")


def main_difference_figure(plt, equivalence, out):
    fields = (
        ("relative_ipc_difference_vs_same_h_20m", "relative IPC difference", 100.0),
        ("coverage_percentage_point_difference_vs_same_h_20m", "coverage difference (pp)", 1.0),
        ("l2_miss_rate_percentage_point_difference_vs_same_h_20m", "L2 miss-rate difference (pp)", 1.0),
        ("request_pressure_difference_vs_same_h_20m", "requests/load difference", 1.0),
        ("student_act_rate_percentage_point_difference_vs_same_h_20m", "student act-rate difference (pp)", 1.0),
    )
    fig, axes = plt.subplots(len(fields), 1, figsize=(8.2, 11.0), sharex=True)
    for ax, (field, ylabel, scale) in zip(axes, fields):
        for hidden in (8, 16):
            rows = points_for(equivalence, hidden, field)
            if rows:
                ax.plot([row["instruction_budget"] for row in rows],
                        [scale * row[field] for row in rows], marker="o",
                        color=COLORS[hidden], label="h{} vs h{}-i20m".format(hidden, hidden))
        ax.axhline(0.0, color="black", linestyle=":", linewidth=1)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[-1].set_xscale("log")
    axes[-1].set_xlabel("retired training instructions")
    save(fig, out / "main_02_same_h_20m_differences.png")


def main_supervision_figure(plt, prefix, out):
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), sharex=True)
    for ax, field, title in zip(
            axes, ("decision_rows", "positive_count_rows", "action_atoms"),
            ("decision rows", "teacher K>0 rows", "teacher action atoms = sum(K)")):
        rows = sorted((row for row in prefix if row.get(field) is not None),
                      key=lambda row: row["instruction_budget"])
        ax.plot([row["instruction_budget"] for row in rows],
                [row[field] for row in rows], marker="o", color="#009E73")
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("retired instructions")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("teacher-label count")
    save(fig, out / "main_03_training_supervision.png")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    return parser


def main():
    args = build_parser().parse_args()
    out = args.out_dir or args.run_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plots: {}".format(exc))
    prefix = load(args.run_dir / "training_prefix_data_summary.json")["points"]
    offline_data = load(args.run_dir / "offline_sweep_results.json")
    live_data = load(args.run_dir / "live_sweep_results.json")
    offline = offline_data["points"]
    live = [
        row for row in live_data["points"]
        if row.get("state_mode") == "parity"
    ]
    joined = load(args.run_dir / "offline_vs_live.json")["points"]
    frontier = load(args.run_dir / "pareto_frontier.json")["points"]
    equivalence = load(args.run_dir / "offline_20m_equivalence.json")["points"]

    main_offline_ipc_figure(
        plt, offline, offline_data.get("references", {}), out
    )
    main_difference_figure(plt, equivalence, out)
    main_supervision_figure(plt, prefix, out)

    for field, filename, ylabel in (
        ("decision_rows", "01_decision_rows.png", "decision rows"),
        (
            "positive_count_rows", "02_positive_count_rows.png",
            "positive-count rows",
        ),
        ("action_atoms", "03_action_atoms.png", "action atoms = sum(K)"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        ax.plot(
            [row["instruction_budget"] for row in prefix],
            [row[field] for row in prefix], marker="o", color="#009E73",
            label="shared h8/h16 stream",
        )
        ax.set_xscale("log")
        ax.set_xlabel("retired training instructions")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        save(fig, out / filename)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    curve(ax, offline, "ipc")
    status_markers(ax, offline)
    for name, reference in offline_data.get("references", {}).items():
        if reference.get("ipc") is not None:
            ax.axhline(
                reference["ipc"], linestyle="--", alpha=0.45, label=name
            )
    ax.set_ylabel("offline keyed-replay IPC")
    save(fig, out / "04_offline_ipc.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    curve(ax, live, "ipc")
    ax.set_ylabel("live IPC")
    save(fig, out / "05_live_ipc.png")

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for hidden in (8, 16):
        selected = [
            row for row in joined
            if row.get("hidden_size") == hidden
            and row.get("offline_ipc") is not None
            and row.get("live_ipc") is not None
        ]
        ax.scatter(
            [row["offline_ipc"] for row in selected],
            [row["live_ipc"] for row in selected],
            color=COLORS[hidden], label="h{}".format(hidden),
        )
    limits = ax.get_xlim()
    ax.plot(limits, limits, color="black", linestyle=":", label="equal")
    ax.set_xlabel("offline IPC")
    ax.set_ylabel("live IPC")
    ax.legend()
    ax.grid(True, alpha=0.25)
    save(fig, out / "06_offline_vs_live_ipc.png")

    for number, field, ylabel in (
        (7, "coverage", "coverage"),
        (8, "requests_per_l2_load", "requests per L2 load"),
        (9, "student_heldout_act_rate", "student held-out act rate"),
        (10, "l2_load_miss_rate", "L2 load miss rate"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        curve(ax, offline, field, label_prefix="offline ")
        if field != "student_heldout_act_rate":
            curve(ax, live, field, label_prefix="live ", marker="s")
        ax.set_ylabel(ylabel)
        save(fig, out / "{:02d}_{}.png".format(number, field))

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for hidden in (8, 16):
        selected = [
            row for row in live
            if row.get("ipc") is not None
            and row.get("live_weight_bytes") is not None
            and row.get("hidden_size") == hidden
        ]
        ax.scatter(
            [row["live_weight_bytes"] for row in selected],
            [row["ipc"] for row in selected],
            color=COLORS[hidden], label="h{}".format(hidden),
        )
    ax.set_xlabel("float32 weight bytes")
    ax.set_ylabel("live IPC")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save(fig, out / "11_weight_bytes_vs_live_ipc.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for hidden in (8, 16):
        selected = [
            row for row in live
            if row.get("ipc") is not None
            and row.get("live_peak_recurrent_state_bytes") is not None
            and row.get("hidden_size") == hidden
        ]
        ax.scatter(
            [row["live_peak_recurrent_state_bytes"] for row in selected],
            [row["ipc"] for row in selected],
            color=COLORS[hidden], label="h{}".format(hidden),
        )
    ax.set_xlabel("peak recurrent-state bytes")
    ax.set_ylabel("live IPC")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save(fig, out / "12_recurrent_state_bytes_vs_live_ipc.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for hidden in (8, 16):
        selected = points_for(live, hidden, "host_p50_nanoseconds")
        for field, linestyle in (
            ("host_p50_nanoseconds", "-"),
            ("host_p95_nanoseconds", "--"),
            ("host_p99_nanoseconds", ":"),
        ):
            ax.plot(
                [row["instruction_budget"] for row in selected],
                [row.get(field) for row in selected],
                color=COLORS[hidden], linestyle=linestyle,
                label="h{} {}".format(hidden, field[5:8]),
            )
    ax.set_xscale("log")
    ax.set_xlabel("retired training instructions")
    ax.set_ylabel("host inference nanoseconds")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    save(fig, out / "13_host_inference_latency.png")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    any_live = False
    for hidden in (8, 16):
        selected = [
            row for row in live
            if row.get("hidden_size") == hidden
            and row.get("ipc") is not None
        ]
        ax.scatter(
            [row["instruction_budget"] for row in selected],
            [row["ipc"] for row in selected],
            s=[
                20 + (row.get("live_total_deployment_bytes") or 0) / 400
                for row in selected
            ],
            alpha=0.55, color=COLORS[hidden], label="h{}".format(hidden),
        )
        any_live = any_live or bool(selected)
    if frontier:
        ax.scatter(
            [row["instruction_budget"] for row in frontier],
            [row["ipc"] for row in frontier],
            facecolors="none", edgecolors="black", s=150,
            label="Pareto frontier",
        )
    if any_live:
        ax.set_xscale("log")
    else:
        ax.text(0.5, 0.5, "NA---functional live runs not available",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("retired training instructions")
    ax.set_ylabel("live IPC")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save(fig, out / "14_pareto_frontier.png")
    print("[ok] wrote 3 offline-primary and 14 diagnostic plots to {}".format(out))


if __name__ == "__main__":
    main()
