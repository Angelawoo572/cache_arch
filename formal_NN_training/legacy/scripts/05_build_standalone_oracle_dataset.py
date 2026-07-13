#!/usr/bin/env python3
"""Build the raw no-prefetch demand dataset for the standalone NN.

This script deliberately does not read normal-prefetcher outputs.  It converts
one no-prefetch L2 demand-event stream into one stable oracle dataset used by
the standalone LSTM / tiny-Transformer notebooks.

Output key:
    pc_line_occ = zero-based occurrence count of (pc,line) in the no-prefetch
                  demand stream.

The notebook creates all future-horizon labels from this raw sequence.  Normal
prefetchers remain comparison baselines only.
"""
from __future__ import print_function

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

CACHE_LINE_BYTES = 64


def open_text(path):
    return gzip.open(str(path), "rt", newline="") if str(path).endswith(".gz") else open(str(path), "r", newline="")


def as_int(value, default=0):
    try:
        if value is None or str(value).strip() == "":
            return default
        s = str(value).strip()
        return int(s, 16) if s.lower().startswith("0x") else int(float(s))
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, type=Path, help="no-prefetch demand-event CSV or CSV.GZ")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--meta-out", type=Path, default=None)
    args = ap.parse_args()

    if not args.events.is_file():
        raise FileNotFoundError(str(args.events))

    rows = []
    occ = defaultdict(int)
    previous_line = None
    skipped = 0

    with open_text(args.events) as f:
        reader = csv.DictReader(f)
        required = {"event", "event_id", "cycle", "cache", "type", "ip", "addr", "line", "hit"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError("event stream missing columns: {}".format(sorted(missing)))

        for raw in reader:
            event = str(raw.get("event", "")).strip().upper()
            cache = str(raw.get("cache", "")).strip().upper()
            access_type = str(raw.get("type", "")).strip()
            if event != "DEMAND" or cache != "L2C" or access_type != "0":
                skipped += 1
                continue

            pc = as_int(raw.get("ip"))
            line = as_int(raw.get("line"))
            addr = as_int(raw.get("addr"))
            if not line and addr:
                line = addr // CACHE_LINE_BYTES
            if not addr and line:
                addr = line * CACHE_LINE_BYTES
            if not line:
                raise ValueError("zero line at event_id={}".format(raw.get("event_id")))

            key = (pc, line)
            pc_line_occ = occ[key]
            occ[key] += 1
            hit = int(as_int(raw.get("hit")) != 0)
            delta = 0 if previous_line is None else line - previous_line
            previous_line = line

            rows.append({
                "trace": args.trace,
                "demand_idx": len(rows),
                "base_event_id": as_int(raw.get("event_id")),
                "cycle": as_int(raw.get("cycle")),
                "pc": pc,
                "addr": addr,
                "line": line,
                "page": line // 64,
                "page_offset": line % 64,
                "delta": delta,
                "no_pref_hit": hit,
                "no_pref_miss": 1 - hit,
                "pc_line_occ": pc_line_occ,
            })

    if not rows:
        raise RuntimeError("no L2C LOAD demand rows found in {}".format(args.events))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "trace", "demand_idx", "base_event_id", "cycle", "pc", "addr", "line",
        "page", "page_offset", "delta", "no_pref_hit", "no_pref_miss", "pc_line_occ",
    ]
    opener = gzip.open if str(args.out).endswith(".gz") else open
    mode = "wt" if str(args.out).endswith(".gz") else "w"
    with opener(str(args.out), mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "trace": args.trace,
        "events": str(args.events),
        "out": str(args.out),
        "demand_rows": len(rows),
        "skipped_non_l2_load_rows": skipped,
        "unique_pc_line_pairs": len(occ),
        "key_format": "pc,line,pc_line_occ",
        "model_role": "standalone_raw_stream_only",
    }
    meta_out = args.meta_out or Path(str(args.out) + ".meta.json")
    meta_out.write_text(json.dumps(meta, indent=2) + "\n")
    print("[ok] " + json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    main()
