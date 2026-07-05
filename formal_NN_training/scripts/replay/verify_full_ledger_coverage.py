#!/usr/bin/env python3
"""Verify that a candidate ledger covers every oracle demand event."""
from __future__ import print_function

import argparse
import csv
import gzip
import json
from pathlib import Path


def open_text(path):
    return gzip.open(str(path), "rt", newline="") if str(path).endswith(".gz") else open(str(path), newline="")


def as_int(value, name):
    if value is None or str(value).strip() == "":
        raise ValueError("missing {}".format(name))
    value = str(value).strip()
    return int(value, 16) if value.lower().startswith("0x") else int(float(value))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True, type=Path)
    ap.add_argument("--ledger", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    oracle = set()
    trace = None
    with open_text(args.oracle) as handle:
        for raw in csv.DictReader(handle):
            current = str(raw.get("trace") or "")
            if not current:
                raise ValueError("oracle row has blank trace")
            if trace is None:
                trace = current
            elif trace != current:
                raise ValueError("oracle has multiple traces")
            oracle.add((as_int(raw.get("demand_idx"), "demand_idx"), as_int(raw.get("pc"), "pc"), as_int(raw.get("line"), "line")))
    if not oracle:
        raise ValueError("oracle has no rows")

    seen = set()
    unmatched = 0
    with open_text(args.ledger) as handle:
        for raw in csv.DictReader(handle):
            if str(raw.get("trace") or "") != trace:
                raise ValueError("ledger trace mismatch")
            key = (as_int(raw.get("demand_idx"), "demand_idx"), as_int(raw.get("pc"), "pc"), as_int(raw.get("line"), "line"))
            if key in oracle:
                seen.add(key)
            else:
                unmatched += 1

    missing = len(oracle) - len(seen)
    result = {"trace": trace, "oracle_events": len(oracle), "ledger_events_seen": len(seen), "unmatched_ledger_rows": unmatched, "full_coverage": missing == 0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[ledger coverage] " + json.dumps(result, sort_keys=True))
    if missing:
        raise SystemExit("ledger covers {}/{} oracle events".format(len(seen), len(oracle)))


if __name__ == "__main__":
    main()
