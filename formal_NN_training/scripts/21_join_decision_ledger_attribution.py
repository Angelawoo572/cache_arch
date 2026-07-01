#!/usr/bin/env python3
"""Classify audit normal-only misses using one notebook decision ledger.

The ledger is for exactly one replay candidate, so --standalone-variant is
required. The join key is (trace, pc, line, pc_line_occ), reconstructed from
the raw no-prefetch oracle and checked against the demand index.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path


def open_csv(path, mode="rt"):
    return gzip.open(str(path), mode, newline="") if str(path).endswith(".gz") else open(str(path), mode, newline="")


def integer(value):
    text = str(value).strip()
    return int(text, 16) if text.lower().startswith("0x") else int(float(text))


def load_oracle(path):
    rows = {}
    with open_csv(path) as handle:
        for row in csv.DictReader(handle):
            rows[integer(row["demand_idx"])] = (integer(row["pc"]), integer(row["line"]), integer(row["pc_line_occ"]))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribution-detail", required=True, type=Path)
    parser.add_argument("--oracle-dir", required=True, type=Path)
    parser.add_argument("--ledger-events", required=True, type=Path)
    parser.add_argument("--standalone-variant", required=True,
                        help="Exact replay-plan tag represented by --ledger-events.")
    parser.add_argument("--normal-prefetcher", default="",
                        help="Optional normal baseline filter, e.g. sms or sandbox.")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    ledger = {}
    with open_csv(args.ledger_events) as handle:
        for row in csv.DictReader(handle):
            key = (row["trace"], integer(row["pc"]), integer(row["line"]), integer(row["pc_line_occ"]))
            ledger[key] = row
    oracle_cache = {}
    joined = []
    skipped_other_variants = 0
    skipped_other_normals = 0
    with open_csv(args.attribution_detail) as handle:
        for row in csv.DictReader(handle):
            if row.get("category") != "normal_only_timely":
                continue
            if row.get("standalone_variant") != args.standalone_variant:
                skipped_other_variants += 1
                continue
            if args.normal_prefetcher and row.get("normal_prefetcher") != args.normal_prefetcher:
                skipped_other_normals += 1
                continue
            trace = row["trace"]
            if trace not in oracle_cache:
                oracle_cache[trace] = load_oracle(args.oracle_dir / (trace + ".oracle.csv.gz"))
            demand_idx = integer(row["demand_idx"])
            pc, line, occ = oracle_cache[trace][demand_idx]
            if integer(row["pc"]) != pc or integer(row["line"]) != line:
                raise RuntimeError("audit/oracle identity mismatch")
            event = ledger.get((trace, pc, line, occ))
            out = dict(row)
            out["pc_line_occ"] = occ
            out["ledger_joined"] = int(event is not None)
            if event is None:
                out["ledger_reason"] = "ledger_unmatched"
            else:
                out.update({"ledger_" + k: v for k, v in event.items()})
                if integer(event["demand_idx"]) != demand_idx:
                    raise RuntimeError("demand-index mismatch after exact ledger key join")
                if not integer(event["future_target_exists"]):
                    out["ledger_reason"] = "outside_candidate_label_horizon"
                elif not integer(event["target_reachable"]):
                    out["ledger_reason"] = "candidate_bank_absent"
                elif integer(event["target_selected"]):
                    out["ledger_reason"] = "selected_but_not_timely"
                elif integer(event["target_dedup_suppressed"]):
                    out["ledger_reason"] = "dedup_suppressed"
                elif integer(event["target_model_rejected"]):
                    out["ledger_reason"] = "model_threshold_rejected"
                elif integer(event["target_other_rejected"]):
                    out["ledger_reason"] = "policy_rank_or_budget_rejected"
                else:
                    out["ledger_reason"] = "ledger_unclassified"
            joined.append(out)
    if not joined:
        raise RuntimeError("no normal-only timely rows for requested candidate/baseline")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in joined for key in row})
    with open_csv(args.out, "wt") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(joined)
    summary = Counter(row["ledger_reason"] for row in joined)
    metadata = dict(summary, rows=len(joined), standalone_variant=args.standalone_variant,
                    normal_prefetcher=args.normal_prefetcher,
                    skipped_other_variants=skipped_other_variants,
                    skipped_other_normals=skipped_other_normals)
    args.out.with_suffix(".summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("[ledger join]", json.dumps(metadata))


if __name__ == "__main__":
    main()
