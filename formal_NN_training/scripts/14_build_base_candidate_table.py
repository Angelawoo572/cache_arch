#!/usr/bin/env python3
"""Build base-prefetcher proposal tables from residual-audit event logs.

Output rows are keyed by (pc,line,occ), the same stable no-prefetch demand key
used by the keyed ListReplayer.  This is the data bridge for a future
base-aware residual NN: it exposes which candidates the normal prefetcher
actually proposed for each dynamic demand trigger.

The source log records DEMAND and PF rows with one global event_id.  A PF row is
attached to the same-cycle (pc, base_line) demand whose event_id is nearest and
not after the PF event; if that ordering is unavailable, the nearest same-key
demand is used.  The script reports and enforces the match rate instead of
silently fabricating a partial base-candidate table.
"""
from __future__ import print_function

import argparse
import bisect
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


def is_l2(row):
    return str(row.get("cache", "")).strip().upper() == "L2C"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--base-prefetcher", required=True)
    ap.add_argument("--meta-out", type=Path, default=None)
    ap.add_argument("--min-match-rate", type=float, default=0.99)
    args = ap.parse_args()

    if not args.events.is_file():
        raise FileNotFoundError(str(args.events))

    demands_by_key = defaultdict(list)
    demand_records = {}
    pf_rows = []
    occ_counter = defaultdict(int)

    with open_text(args.events) as f:
        reader = csv.DictReader(f)
        need = {"event", "event_id", "cycle", "ip", "line", "base_addr", "pf_line"}
        missing = need.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError("event log missing required columns: {}".format(sorted(missing)))
        for raw in reader:
            row = {str(k).strip(): v for k, v in raw.items() if k is not None}
            if not is_l2(row):
                continue
            kind = str(row.get("event", "")).strip().upper()
            event_id = as_int(row.get("event_id"))
            cycle = as_int(row.get("cycle"))
            pc = as_int(row.get("ip"))
            if kind == "DEMAND":
                line = as_int(row.get("line"))
                pair = (pc, line)
                occ = occ_counter[pair]
                occ_counter[pair] += 1
                rec = {"event_id": event_id, "cycle": cycle, "pc": pc, "line": line, "occ": occ}
                demand_records[event_id] = rec
                demands_by_key[(cycle, pc, line)].append(event_id)
            elif kind == "PF":
                base_addr = as_int(row.get("base_addr"))
                base_line = base_addr // CACHE_LINE_BYTES if base_addr else 0
                pf_line = as_int(row.get("pf_line"))
                if base_line and pf_line:
                    pf_rows.append({
                        "event_id": event_id, "cycle": cycle, "pc": pc,
                        "base_line": base_line, "pf_line": pf_line,
                        "accepted": as_int(row.get("accepted")),
                        "duplicate": as_int(row.get("duplicate")),
                    })

    for ids in demands_by_key.values():
        ids.sort()

    out_rows = []
    per_trigger_rank = defaultdict(int)
    unmatched = []
    fallback_matches = 0
    for pf in sorted(pf_rows, key=lambda x: x["event_id"]):
        ids = demands_by_key.get((pf["cycle"], pf["pc"], pf["base_line"]), [])
        if not ids:
            unmatched.append(pf)
            continue
        pos = bisect.bisect_right(ids, pf["event_id"]) - 1
        if pos >= 0:
            trigger_event_id = ids[pos]
        else:
            trigger_event_id = ids[0]
            fallback_matches += 1
        d = demand_records[trigger_event_id]
        key = (d["pc"], d["line"], d["occ"])
        rank = per_trigger_rank[key]
        per_trigger_rank[key] += 1
        out_rows.append({
            "trace": args.trace,
            "base_prefetcher": args.base_prefetcher,
            "trigger_event_id": trigger_event_id,
            "cycle": d["cycle"],
            "pc": d["pc"],
            "line": d["line"],
            "occ": d["occ"],
            "pf_event_id": pf["event_id"],
            "candidate_rank": rank,
            "candidate_line": pf["pf_line"],
            "candidate_delta": pf["pf_line"] - d["line"],
            "prefetch_addr": pf["pf_line"] * CACHE_LINE_BYTES,
            "accepted": pf["accepted"],
            "duplicate": pf["duplicate"],
        })

    pf_total = len(pf_rows)
    rate = float(len(out_rows)) / float(max(pf_total, 1))
    meta = {
        "trace": args.trace,
        "base_prefetcher": args.base_prefetcher,
        "events": str(args.events),
        "out": str(args.out),
        "l2_demand_rows": len(demand_records),
        "l2_pf_rows": pf_total,
        "matched_pf_rows": len(out_rows),
        "unmatched_pf_rows": len(unmatched),
        "fallback_order_matches": fallback_matches,
        "match_rate": rate,
        "unique_trigger_keys": len(per_trigger_rank),
        "key_format": "pc,line,occ",
        "semantics": "base prefetch_line proposals attached to same-cycle demand trigger",
        "unmatched_examples": unmatched[:5],
    }
    if rate < args.min_match_rate:
        raise RuntimeError("base candidate match rate {:.4f} < {:.4f}; refusing partial table: {}".format(rate, args.min_match_rate, json.dumps(meta)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["trace", "base_prefetcher", "trigger_event_id", "cycle", "pc", "line", "occ", "pf_event_id", "candidate_rank", "candidate_line", "candidate_delta", "prefetch_addr", "accepted", "duplicate"]
    with gzip.open(str(args.out), "wt", newline="") if str(args.out).endswith(".gz") else open(str(args.out), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    meta_out = args.meta_out or Path(str(args.out) + ".meta.json")
    meta_out.write_text(json.dumps(meta, indent=2) + "\n")
    print("[ok] " + json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    main()
