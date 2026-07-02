#!/usr/bin/env python3
"""Join one decision ledger to normal-only timely attribution rows.

This is a standard-library-only Python 3.6 script.  The ledger must correspond
exactly to --standalone-variant.
"""
import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path


def open_csv(path, mode="rt"):
    if str(path).endswith(".gz"):
        return gzip.open(str(path), mode, newline="")
    return open(str(path), mode, newline="")


def integer(value):
    value = str(value).strip()
    return int(value, 16) if value.lower().startswith("0x") else int(float(value))


def load_oracle(path):
    result = {}
    with open_csv(path) as handle:
        for row in csv.DictReader(handle):
            result[integer(row["demand_idx"])] = (
                integer(row["pc"]), integer(row["line"]), integer(row["pc_line_occ"])
            )
    return result


def reason(event):
    if event is None:
        return "ledger_unmatched"
    if not integer(event["future_target_exists"]):
        return "outside_candidate_label_horizon"
    if not integer(event["target_reachable"]):
        return "candidate_bank_absent"
    if integer(event["target_selected"]):
        return "selected_but_not_timely"
    if integer(event["target_dedup_suppressed"]):
        return "dedup_suppressed"
    if integer(event["target_model_rejected"]):
        return "model_threshold_rejected"
    if integer(event["target_other_rejected"]):
        return "policy_rank_or_budget_rejected"
    return "ledger_unclassified"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribution-detail", required=True, type=Path)
    ap.add_argument("--oracle-dir", required=True, type=Path)
    ap.add_argument("--ledger-events", required=True, type=Path)
    ap.add_argument("--standalone-variant", required=True)
    ap.add_argument("--normal-prefetcher", default="")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    ledger = {}
    with open_csv(args.ledger_events) as handle:
        for row in csv.DictReader(handle):
            key = (row["trace"], integer(row["pc"]), integer(row["line"]), integer(row["pc_line_occ"]))
            if key in ledger:
                raise RuntimeError("duplicate ledger key {}".format(key))
            ledger[key] = row

    oracle_cache = {}
    rows = []
    with open_csv(args.attribution_detail) as handle:
        for row in csv.DictReader(handle):
            if row.get("category") != "normal_only_timely":
                continue
            if row.get("standalone_variant") != args.standalone_variant:
                continue
            if args.normal_prefetcher and row.get("normal_prefetcher") != args.normal_prefetcher:
                continue
            trace = row["trace"]
            if trace not in oracle_cache:
                oracle_cache[trace] = load_oracle(args.oracle_dir / (trace + ".oracle.csv.gz"))
            demand_idx = integer(row["demand_idx"])
            pc, line, occ = oracle_cache[trace][demand_idx]
            if integer(row["pc"]) != pc or integer(row["line"]) != line:
                raise RuntimeError("audit/oracle mismatch")
            event = ledger.get((trace, pc, line, occ))
            out = dict(row)
            out["pc_line_occ"] = occ
            out["ledger_joined"] = int(event is not None)
            out["ledger_reason"] = reason(event)
            if event is not None:
                if integer(event["demand_idx"]) != demand_idx:
                    raise RuntimeError("ledger demand index mismatch")
                for name, value in event.items():
                    out["ledger_" + name] = value
            rows.append(out)

    if not rows:
        raise RuntimeError("no matching normal-only timely rows")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set(name for row in rows for name in row))
    with open_csv(args.out, "wt") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(Counter(row["ledger_reason"] for row in rows))
    summary["rows"] = len(rows)
    summary["standalone_variant"] = args.standalone_variant
    summary["normal_prefetcher"] = args.normal_prefetcher
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("[ledger join] " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
