#!/usr/bin/env python3
"""Build normal-prefetch proposal tables keyed by (pc, line, occ).

This optional analysis builder is retained because it is the data bridge for a
future base-aware experiment. It does not affect the standalone-v4 pipeline.
"""
from __future__ import print_function

import argparse
import bisect
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

LINE_BYTES = 64


def open_text(path):
    if str(path).endswith(".gz"):
        return gzip.open(str(path), "rt", newline="")
    return open(str(path), "r", newline="")


def as_int(value, default=0):
    try:
        text = str(value).strip()
        if not text:
            return default
        return int(text, 16) if text.lower().startswith("0x") else int(float(text))
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--base-prefetcher", required=True)
    ap.add_argument("--meta-out", type=Path, default=None)
    ap.add_argument("--min-match-rate", type=float, default=0.99)
    args = ap.parse_args()

    demand_ids = defaultdict(list)
    demands = {}
    occ = defaultdict(int)
    prefetches = []
    with open_text(args.events) as handle:
        reader = csv.DictReader(handle)
        required = set(["event", "event_id", "cycle", "ip", "line", "base_addr", "pf_line"])
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError("event log missing columns: {}".format(sorted(missing)))
        for raw in reader:
            if str(raw.get("cache", "")).upper() != "L2C":
                continue
            kind = str(raw.get("event", "")).upper()
            event_id = as_int(raw.get("event_id"))
            cycle = as_int(raw.get("cycle"))
            pc = as_int(raw.get("ip"))
            if kind == "DEMAND":
                line = as_int(raw.get("line"))
                key = (pc, line)
                record = {"event_id": event_id, "cycle": cycle, "pc": pc, "line": line, "occ": occ[key]}
                occ[key] += 1
                demands[event_id] = record
                demand_ids[(cycle, pc, line)].append(event_id)
            elif kind == "PF":
                base_addr = as_int(raw.get("base_addr"))
                candidate = as_int(raw.get("pf_line"))
                if base_addr and candidate:
                    prefetches.append({
                        "event_id": event_id, "cycle": cycle, "pc": pc,
                        "base_line": base_addr // LINE_BYTES, "candidate_line": candidate,
                        "accepted": as_int(raw.get("accepted")), "duplicate": as_int(raw.get("duplicate")),
                    })

    for ids in demand_ids.values():
        ids.sort()
    rows = []
    ranks = defaultdict(int)
    unmatched = []
    fallback = 0
    for pf in sorted(prefetches, key=lambda row: row["event_id"]):
        ids = demand_ids.get((pf["cycle"], pf["pc"], pf["base_line"]), [])
        if not ids:
            unmatched.append(pf)
            continue
        pos = bisect.bisect_right(ids, pf["event_id"]) - 1
        trigger_id = ids[pos] if pos >= 0 else ids[0]
        fallback += int(pos < 0)
        trigger = demands[trigger_id]
        trigger_key = (trigger["pc"], trigger["line"], trigger["occ"])
        rank = ranks[trigger_key]
        ranks[trigger_key] += 1
        rows.append({
            "trace": args.trace, "base_prefetcher": args.base_prefetcher,
            "trigger_event_id": trigger_id, "cycle": trigger["cycle"],
            "pc": trigger["pc"], "line": trigger["line"], "occ": trigger["occ"],
            "pf_event_id": pf["event_id"], "candidate_rank": rank,
            "candidate_line": pf["candidate_line"],
            "candidate_delta": pf["candidate_line"] - trigger["line"],
            "prefetch_addr": pf["candidate_line"] * LINE_BYTES,
            "accepted": pf["accepted"], "duplicate": pf["duplicate"],
        })

    rate = float(len(rows)) / float(max(1, len(prefetches)))
    meta = {
        "trace": args.trace, "base_prefetcher": args.base_prefetcher,
        "events": str(args.events), "out": str(args.out),
        "l2_demand_rows": len(demands), "l2_pf_rows": len(prefetches),
        "matched_pf_rows": len(rows), "unmatched_pf_rows": len(unmatched),
        "fallback_order_matches": fallback, "match_rate": rate,
        "unique_trigger_keys": len(ranks), "key_format": "pc,line,occ",
        "semantics": "normal-prefetch proposals attached to same-cycle demand trigger",
        "unmatched_examples": unmatched[:5],
    }
    if rate < args.min_match_rate:
        raise RuntimeError("match rate {:.4f} < {:.4f}; refusing partial table: {}".format(rate, args.min_match_rate, json.dumps(meta)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["trace", "base_prefetcher", "trigger_event_id", "cycle", "pc", "line", "occ", "pf_event_id", "candidate_rank", "candidate_line", "candidate_delta", "prefetch_addr", "accepted", "duplicate"]
    if str(args.out).endswith(".gz"):
        handle = gzip.open(str(args.out), "wt", newline="")
    else:
        handle = open(str(args.out), "w", newline="")
    with handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    meta_out = args.meta_out or Path(str(args.out) + ".meta.json")
    meta_out.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print("[ok] " + json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    main()
