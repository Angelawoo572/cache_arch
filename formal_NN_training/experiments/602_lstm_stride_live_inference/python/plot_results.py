#!/usr/bin/env python3
"""Generate the fourteen required h8/h16-separated diagnostic plots."""

import argparse
import json
import os
from pathlib import Path


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
    if frontier:
        ax.scatter(
            [row["instruction_budget"] for row in frontier],
            [row["ipc"] for row in frontier],
            facecolors="none", edgecolors="black", s=150,
            label="Pareto frontier",
        )
    ax.set_xscale("log")
    ax.set_xlabel("retired training instructions")
    ax.set_ylabel("live IPC")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save(fig, out / "14_pareto_frontier.png")
    print("[ok] wrote 14 plots to {}".format(out))


if __name__ == "__main__":
    main()
