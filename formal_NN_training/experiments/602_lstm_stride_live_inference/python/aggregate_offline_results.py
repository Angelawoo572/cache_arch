#!/usr/bin/env python3
"""Aggregate every configured point, including untrainable and failed probes."""

import argparse
from pathlib import Path

from result_utils import (
    base_point,
    load_json,
    parse_log,
    reference_logs,
    system_metrics,
    write_outputs,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser


def main():
    args = build_parser().parse_args()
    references = reference_logs(args.run_dir)
    baseline = references.get("no_pref")
    rows = []
    for path in sorted(args.run_dir.glob(
        "points/h*/*/seed*/point_metadata.json"
    )):
        metadata = load_json(path)
        row = base_point(metadata)
        behavior = metadata.get("heldout_behavior_metrics") or {}
        row["heldout_target_f1"] = behavior.get("target_f1")
        row["heldout_target_precision"] = behavior.get("target_precision")
        row["heldout_target_recall"] = behavior.get("target_recall")
        log = path.parent / "offline/replay.log"
        stats = None
        if log.is_file():
            try:
                stats = parse_log(log)
            except Exception as exc:
                row["status"] = "offline_replay_failed"
                row["failure_reason"] = str(exc)
        row.update(system_metrics(stats, baseline))
        row["offline_replay_log"] = str(log) if log.is_file() else None
        rows.append(row)
    rows.sort(key=lambda row: (
        row.get("hidden_size") or 0,
        row.get("instruction_budget") or 0,
        row.get("seed") or 0,
    ))
    write_outputs(
        rows,
        args.run_dir / "offline_sweep_results.csv",
        args.run_dir / "offline_sweep_results.json",
        references,
    )
    print("[ok] aggregated {} offline points".format(len(rows)))


if __name__ == "__main__":
    main()
