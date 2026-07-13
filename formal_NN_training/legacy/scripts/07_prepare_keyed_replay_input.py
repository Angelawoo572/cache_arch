#!/usr/bin/env python3
"""Convert a standalone-NN export to a keyed ListReplayer CSV.

The notebook's order/cycle is not a stable post-prefetch callback index.  Each
exported row is therefore mapped to the no-prefetch oracle demand and written
as pc,line,occ,prefetch_addr.  `occ` is the zero-based occurrence count of the
(pc,line) pair in that no-prefetch stream.

For a rich export that already contains demand_idx, this script *also* requires
its pc and line to match the oracle row at that index.  A numeric index alone is
not accepted as proof of alignment.
"""
from __future__ import print_function

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

CACHE_LINE_BYTES = 64
U64_MAX = (1 << 64) - 1


def open_text(path):
    return gzip.open(str(path), "rt", newline="") if str(path).endswith(".gz") else path.open("r", newline="")


def to_int(value, name):
    if value is None or str(value).strip() == "":
        raise ValueError("missing {}".format(name))
    s = str(value).strip()
    return int(s, 16) if s.lower().startswith("0x") else int(float(s))


def load_oracle(path):
    by_cycle_pc_line = defaultdict(list)
    positions = defaultdict(int)
    identity = {}
    pair_occ = defaultdict(int)
    with open_text(path) as f:
        reader = csv.DictReader(f)
        required = {"demand_idx", "cycle", "pc", "line"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError("oracle missing columns: {}".format(sorted(missing)))
        previous = -1
        for row in reader:
            idx = to_int(row["demand_idx"], "demand_idx")
            if idx != previous + 1:
                raise ValueError("oracle demand_idx must be contiguous")
            previous = idx
            cycle = to_int(row["cycle"], "cycle")
            pc = to_int(row["pc"], "pc")
            line = to_int(row["line"], "line")
            pair = (pc, line)
            occ = pair_occ[pair]
            pair_occ[pair] += 1
            by_cycle_pc_line[(cycle, pc, line)].append(idx)
            identity[idx] = (pc, line, occ)
    if not identity:
        raise ValueError("oracle has no rows")
    return by_cycle_pc_line, positions, identity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rich-list", required=True, type=Path)
    ap.add_argument("--oracle", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--meta-out", type=Path, default=None)
    args = ap.parse_args()
    if not args.rich_list.is_file() or not args.oracle.is_file():
        raise FileNotFoundError("rich-list and oracle must exist")

    oracle_map, positions, identity = load_oracle(args.oracle)
    last_key, last_idx = None, None
    entries = defaultdict(list)
    unmatched, invalid = [], []
    direct_rows, fallback_rows, direct_verified = 0, 0, 0

    with args.rich_list.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if "prefetch_addr" not in fields:
            raise ValueError("rich list missing prefetch_addr")
        for row_no, row in enumerate(reader, start=2):
            addr = to_int(row.get("prefetch_addr"), "prefetch_addr")
            if addr <= 0 or addr > U64_MAX or addr % CACHE_LINE_BYTES:
                invalid.append(row_no)
                continue
            direct = row.get("replay_idx") or row.get("demand_idx")
            if direct not in (None, ""):
                idx = to_int(direct, "replay_idx")
                if idx not in identity:
                    raise ValueError("replay_idx outside oracle at CSV row {}: {}".format(row_no, idx))
                if "pc" not in fields or "line" not in fields:
                    raise ValueError("direct-index rich list must include pc and line for alignment verification")
                export_pc = to_int(row.get("pc"), "pc")
                export_line = to_int(row.get("line"), "line")
                oracle_pc, oracle_line, _ = identity[idx]
                if export_pc != oracle_pc or export_line != oracle_line:
                    raise ValueError(
                        "direct-index alignment mismatch at CSV row {}: idx={} export=(pc={},line={}) oracle=(pc={},line={})".format(
                            row_no, idx, export_pc, export_line, oracle_pc, oracle_line
                        )
                    )
                direct_rows += 1
                direct_verified += 1
            else:
                key = (to_int(row.get("order"), "order"), to_int(row.get("pc"), "pc"), to_int(row.get("line"), "line"))
                if key == last_key and last_idx is not None:
                    idx = last_idx
                else:
                    values = oracle_map.get(key, [])
                    pos = positions.get(key, 0)
                    if pos >= len(values):
                        unmatched.append(row_no)
                        continue
                    idx = values[pos]
                    positions[key] = pos + 1
                    last_key, last_idx = key, idx
                fallback_rows += 1
            entries[identity[idx]].append(addr)

    if unmatched:
        raise RuntimeError("{} exported rows cannot be mapped to oracle events".format(len(unmatched)))
    if not entries:
        raise RuntimeError("no valid replay entries")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        f.write("pc,line,occ,prefetch_addr\n")
        for pc, line, occ in sorted(entries):
            for addr in entries[(pc, line, occ)]:
                f.write("{},{},{},0x{:x}\n".format(pc, line, occ, addr))

    meta = {
        "rich_list": str(args.rich_list), "oracle": str(args.oracle), "out": str(args.out),
        "oracle_rows": len(identity), "entries": sum(len(x) for x in entries.values()),
        "unique_trigger_keys": len(entries), "unmatched_rows": len(unmatched),
        "dropped_invalid_address": len(invalid), "direct_index_rows": direct_rows,
        "direct_index_rows_verified_pc_line": direct_verified,
        "mapped_cycle_pc_line_rows": fallback_rows,
        "key_format": "pc,line,occ,prefetch_addr",
        "replay_semantics": "offline keyed replay; not in-simulator PyTorch inference",
    }
    meta_out = args.meta_out or args.out.with_suffix(args.out.suffix + ".meta.json")
    meta_out.write_text(json.dumps(meta, indent=2) + "\n")
    print("[ok] " + json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("[error] {}".format(exc), file=sys.stderr)
        raise SystemExit(2)
