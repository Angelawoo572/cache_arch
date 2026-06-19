#!/usr/bin/env python3
"""Join counter-level and demand-centric prefetcher audit summaries.

This produces the table to feed the next NN planning step: one row per
trace/prefetcher with IPC/speedup, accuracy/timeliness, demand coverage,
late/residual rates, duplicate rates, top residual PCs/deltas, and failure flags.
"""

import argparse
import csv
from pathlib import Path


def read_csv(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def key(row):
    return (row.get("trace", ""), row.get("prefetcher", ""))


def pick(row, fields, prefix=""):
    out = {}
    for f in fields:
        out[prefix + f] = row.get(f, "") if row else ""
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior", required=True, type=Path)
    ap.add_argument("--residual", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    behavior_rows = read_csv(args.behavior)
    residual_rows = read_csv(args.residual)
    residual_by_key = {key(r): r for r in residual_rows}

    behavior_fields = [
        "ipc", "speedup_vs_no_pref", "l2_loads", "l2_load_miss", "l2_load_miss_rate",
        "miss_reduction_vs_no_pref", "pf_requested", "pf_dropped", "pf_issued",
        "nodup_issued", "pf_filled", "pf_useful", "pf_useless", "pf_late",
        "pq_merged_duplicate_proxy", "accuracy", "nodup_accuracy", "selected_accuracy",
        "timeliness", "late_per_issued", "drop_rate", "useless_per_issued",
        "run_failed", "fail_reason", "log_missing", "log",
    ]
    residual_fields = [
        "demand", "demand_hit", "demand_miss", "demand_miss_rate",
        "covered_on_time", "covered_on_time_rate", "coverage_among_misses",
        "late_prefetch", "late_rate_among_misses", "residual_miss",
        "residual_miss_rate", "residual_share_of_misses", "original_miss_pool",
        "pf_requested_events", "pf_accepted_events", "pf_duplicate_events",
        "pf_duplicate_rate", "pf_dropped_events", "pf_dropped_rate",
        "top_residual_pcs", "top_residual_deltas", "parse_error", "event_file",
    ]

    rows = []
    for b in behavior_rows:
        r = residual_by_key.get(key(b), {})
        row = {
            "trace": b.get("trace", ""),
            "prefetcher": b.get("prefetcher", ""),
        }
        row.update(pick(b, behavior_fields, "behavior_"))
        row.update(pick(r, residual_fields, "residual_"))
        row["has_residual_audit"] = "1" if r else "0"
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        args.out.write_text("")
        print(f"[write empty] {args.out}")
        return

    fieldnames = list(rows[0].keys())
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"[write] {args.out}")
    print(f"[rows] {len(rows)}")


if __name__ == "__main__":
    main()
