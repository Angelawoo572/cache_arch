#!/usr/bin/env python3
"""Summarize measured PQ/MSHR pressure from DEMAND_EVENT_LOG files.

The script uses csv/gzip only. It is Python 3.6 compatible and does not import
pandas, numpy, or any other third-party package.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path


def open_csv(path, mode="rt"):
    if str(path).endswith(".gz"):
        return gzip.open(str(path), mode, newline="")
    return open(str(path), mode, newline="")


def number(value):
    try:
        return float(value) if str(value).strip() else 0.0
    except (TypeError, ValueError):
        return 0.0


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(fraction * (len(ordered) - 1))
    return ordered[index]


def describe(values, prefix):
    if not values:
        return {
            prefix + "mean": 0.0,
            prefix + "p50": 0.0,
            prefix + "p95": 0.0,
            prefix + "max": 0.0,
        }
    return {
        prefix + "mean": float(sum(values)) / len(values),
        prefix + "p50": percentile(values, 0.50),
        prefix + "p95": percentile(values, 0.95),
        prefix + "max": max(values),
    }


def summarize_event_file(path, family, variant, trace):
    demand_pq = []
    demand_mshr = []
    prefetch_pq = []
    prefetch_mshr = []
    demand_loads = 0
    prefetch_attempts = 0
    prefetch_accepted = 0
    prefetch_duplicates = 0
    timely_demands = 0
    late_demands = 0

    with open_csv(path) as handle:
        for row in csv.DictReader(handle):
            if str(row.get("cache", "")).upper() != "L2C":
                continue
            event = str(row.get("event", "")).upper()
            pq = number(row.get("pq_occ"))
            mshr = number(row.get("mshr_occ"))
            if event == "DEMAND":
                demand_loads += 1
                demand_pq.append(pq)
                demand_mshr.append(mshr)
                if number(row.get("hit")) > 0 and number(row.get("was_prefetch")) > 0:
                    timely_demands += 1
                if number(row.get("late")) > 0:
                    late_demands += 1
            elif event == "PF":
                prefetch_attempts += 1
                prefetch_pq.append(pq)
                prefetch_mshr.append(mshr)
                if number(row.get("accepted")) > 0:
                    prefetch_accepted += 1
                if number(row.get("duplicate")) > 0:
                    prefetch_duplicates += 1

    result = {
        "trace": trace,
        "family": family,
        "variant": variant,
        "event_file": str(path),
        "demand_l2_loads": demand_loads,
        "prefetch_attempts": prefetch_attempts,
        "prefetch_accepted": prefetch_accepted,
        "prefetch_duplicate": prefetch_duplicates,
        "timely_prefetch_demand": timely_demands,
        "late_prefetch_demand": late_demands,
        "prefetch_attempts_per_l2_load": float(prefetch_attempts) / demand_loads if demand_loads else 0.0,
        "prefetch_accepted_per_l2_load": float(prefetch_accepted) / demand_loads if demand_loads else 0.0,
        "prefetch_reject_fraction": 1.0 - float(prefetch_accepted) / prefetch_attempts if prefetch_attempts else 0.0,
        "prefetch_duplicate_fraction": float(prefetch_duplicates) / prefetch_attempts if prefetch_attempts else 0.0,
    }
    result.update(describe(demand_pq, "demand_pq_occ_"))
    result.update(describe(demand_mshr, "demand_mshr_occ_"))
    result.update(describe(prefetch_pq, "pf_pq_occ_"))
    result.update(describe(prefetch_mshr, "pf_mshr_occ_"))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rows = []
    normal_dir = args.event_root / "normal" / "events"
    if normal_dir.is_dir():
        for path in sorted(normal_dir.glob("*.events.csv.gz")):
            stem = path.name[:-len(".events.csv.gz")]
            trace, variant = stem.rsplit(".", 1)
            rows.append(summarize_event_file(path, "normal", variant, trace))

    standalone_dir = args.event_root / "lstm"
    if standalone_dir.is_dir():
        for variant_dir in sorted(standalone_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            for path in sorted((variant_dir / "events").glob("*.events.csv.gz")):
                trace = path.name[:-len(".events.csv.gz")]
                rows.append(summarize_event_file(path, "standalone", variant_dir.name, trace))

    if not rows:
        raise RuntimeError("no compressed event logs found under {}".format(args.event_root))

    rows.sort(key=lambda row: (row["trace"], row["family"], row["variant"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set(name for row in rows for name in row))
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    args.out.with_suffix(".json").write_text(json.dumps({
        "event_root": str(args.event_root),
        "rows": len(rows),
    }, indent=2, sort_keys=True) + "\n")
    print("[resource summary] {}".format(args.out))


if __name__ == "__main__":
    main()
