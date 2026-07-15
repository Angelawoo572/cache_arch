#!/usr/bin/env python3
"""Normalize one demand-event log into the 623 temporal-model stream contract."""
import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path


TRACE = "623.xalancbmk_s-700B"


def as_int(value):
    value = str(value).strip()
    return int(value, 16) if value.lower().startswith("0x") else int(float(value))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    opener = gzip.open if str(args.events).endswith(".gz") else open
    args.out.parent.mkdir(parents=True, exist_ok=True)
    occurrences = defaultdict(int)
    rows = 0
    with opener(args.events, "rt", newline="") as source, gzip.open(args.out, "wt", newline="") as target:
        reader = csv.DictReader(source)
        required = {"event", "cache", "op", "ip", "line", "cycle"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError("event log missing columns: {}".format(sorted(missing)))
        writer = csv.DictWriter(
            target,
            fieldnames=["trace", "demand_idx", "cycle", "pc", "line", "pc_line_occ"],
        )
        writer.writeheader()
        for row in reader:
            if row["event"] != "DEMAND" or row["cache"] != "L2C" or row["op"] != "read":
                continue
            pc = as_int(row["ip"])
            line = as_int(row["line"])
            pair = (pc, line)
            occ = occurrences[pair]
            occurrences[pair] += 1
            writer.writerow(
                {
                    "trace": TRACE,
                    "demand_idx": rows,
                    "cycle": as_int(row["cycle"]),
                    "pc": pc,
                    "line": line,
                    "pc_line_occ": occ,
                }
            )
            rows += 1
    if rows == 0:
        raise RuntimeError("no post-warmup L2 demand rows were found")
    print("[ok] wrote {} rows to {}".format(rows, args.out))


if __name__ == "__main__":
    main()


