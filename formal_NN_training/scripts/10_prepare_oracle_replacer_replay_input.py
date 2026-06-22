#!/usr/bin/env python3
"""Convert a rich base-LSTM export into a PC-line-occurrence replay list.

The notebook export has columns such as:
    order,pc,line,...,prefetch_addr
where ``order`` is a simulator cycle, not a stable runtime callback index.

A no-prefetch global L2-load ordinal is not safe after prefetching: a useful
prefetch changes memory timing and can reorder independent L2 callbacks. This
converter therefore maps each rich row to its no-prefetch oracle demand event,
then writes the event's stable local key:

    pc,line,occ,prefetch_addr

``occ`` is the zero-based occurrence number of that (pc,line) pair in the
oracle stream. The ListReplayer maintains the same per-(pc,line) counter at
runtime and emits a candidate only when that key occurs.

This is still an offline-policy replay, not in-simulator PyTorch inference. It
is robust to global L2 callback reordering but must be reported as keyed replay.
Python 3.6 compatible.
"""

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
    if value is None:
        raise ValueError("missing {}".format(name))
    s = str(value).strip()
    if not s:
        raise ValueError("missing {}".format(name))
    if s.lower().startswith("0x"):
        return int(s, 16)
    return int(float(s))


def trigger_key(cycle, pc, line):
    return (cycle, pc, line)


def load_oracle(oracle_path):
    """Return rich-trigger FIFO mapping and demand_idx -> (pc,line,occ)."""
    mapping = defaultdict(list)
    positions = defaultdict(int)
    identity_by_idx = {}
    per_pair_occ = defaultdict(int)

    with open_text(oracle_path) as f:
        reader = csv.DictReader(f)
        required = set(["demand_idx", "cycle", "pc", "line"])
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError("oracle {} missing columns: {}".format(oracle_path, sorted(missing)))

        previous = -1
        for row in reader:
            idx = to_int(row["demand_idx"], "oracle demand_idx")
            if idx != previous + 1:
                raise ValueError("oracle demand_idx must be contiguous: expected {}, saw {}".format(previous + 1, idx))
            previous = idx
            cycle = to_int(row["cycle"], "oracle cycle")
            pc = to_int(row["pc"], "oracle pc")
            line = to_int(row["line"], "oracle line")
            pair = (pc, line)
            occ = per_pair_occ[pair]
            per_pair_occ[pair] += 1
            mapping[trigger_key(cycle, pc, line)].append(idx)
            identity_by_idx[idx] = (pc, line, occ)

    if not identity_by_idx:
        raise ValueError("oracle has no rows: {}".format(oracle_path))
    return mapping, positions, identity_by_idx


def iter_rich_rows(path):
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = set(["prefetch_addr"])
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError("rich export {} missing columns: {}".format(path, sorted(missing)))
        for row in reader:
            yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rich-list", required=True, type=Path,
                    help="Notebook rich fair_dedup CSV")
    ap.add_argument("--oracle", required=True, type=Path,
                    help="Matching no-prefetch oracle CSV(.gz)")
    ap.add_argument("--out", required=True, type=Path,
                    help="Keyed sparse pc,line,occ,prefetch_addr replay CSV")
    ap.add_argument("--meta-out", type=Path, default=None,
                    help="Optional JSON conversion summary (default: <out>.meta.json)")
    ap.add_argument("--allow-unmatched", action="store_true",
                    help="Drop unmatched rich rows instead of failing. Do not use for formal keyed replay.")
    ap.add_argument("--fail-on-invalid-address", action="store_true",
                    help="Treat invalid neural addresses as fatal instead of filtering/counting them.")
    args = ap.parse_args()

    if not args.rich_list.is_file():
        raise FileNotFoundError(str(args.rich_list))
    if not args.oracle.is_file():
        raise FileNotFoundError(str(args.oracle))

    oracle_map, oracle_positions, identity_by_idx = load_oracle(args.oracle)

    # A degree-k export may carry several targets for one trigger. Reuse the
    # just-mapped demand_idx rather than consuming another oracle event.
    last_key = None
    last_idx = None
    trigger_to_addrs = defaultdict(list)
    unmatched = []
    invalid_examples = []
    direct_index_rows = 0
    mapped_rows = 0
    dropped_invalid_address = 0

    for row_no, row in enumerate(iter_rich_rows(args.rich_list), start=2):
        addr = to_int(row.get("prefetch_addr"), "prefetch_addr at rich row {}".format(row_no))
        valid_address = (addr > 0 and addr <= U64_MAX and addr % CACHE_LINE_BYTES == 0)
        if not valid_address:
            dropped_invalid_address += 1
            if len(invalid_examples) < 5:
                invalid_examples.append({"row": row_no, "prefetch_addr": str(addr)})
            if args.fail_on_invalid_address:
                raise ValueError("invalid prefetch_addr at rich row {}: {}".format(row_no, addr))
            continue

        direct = row.get("replay_idx") or row.get("demand_idx")
        if direct not in (None, ""):
            idx = to_int(direct, "replay_idx at rich row {}".format(row_no))
            if idx not in identity_by_idx:
                raise ValueError("replay_idx {} at rich row {} outside oracle range".format(idx, row_no))
            direct_index_rows += 1
        else:
            try:
                key = trigger_key(
                    to_int(row.get("order"), "order/cycle at rich row {}".format(row_no)),
                    to_int(row.get("pc"), "pc at rich row {}".format(row_no)),
                    to_int(row.get("line"), "line at rich row {}".format(row_no)),
                )
            except ValueError as exc:
                unmatched.append({"row": row_no, "reason": str(exc)})
                continue

            if key == last_key and last_idx is not None:
                idx = last_idx
            else:
                values = oracle_map.get(key)
                pos = oracle_positions.get(key, 0)
                if not values or pos >= len(values):
                    unmatched.append({"row": row_no, "reason": "no unused oracle trigger for cycle/pc/line={}".format(key)})
                    continue
                idx = values[pos]
                oracle_positions[key] = pos + 1
                last_key, last_idx = key, idx
            mapped_rows += 1

        trigger_to_addrs[identity_by_idx[idx]].append(addr)

    if unmatched and not args.allow_unmatched:
        preview = "; ".join("row {}: {}".format(x["row"], x["reason"]) for x in unmatched[:5])
        raise RuntimeError(
            "{} rich rows could not be mapped to oracle demand events. Examples: {}. "
            "Refusing to create a partial keyed replay list.".format(len(unmatched), preview)
        )
    entries = sum(len(v) for v in trigger_to_addrs.values())
    if not entries:
        raise RuntimeError("no replay entries produced")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        f.write("pc,line,occ,prefetch_addr\n")
        for key in sorted(trigger_to_addrs):
            pc, line, occ = key
            for addr in trigger_to_addrs[key]:
                f.write("{},{},{},0x{:x}\n".format(pc, line, occ, addr))

    meta = {
        "rich_list": str(args.rich_list),
        "oracle": str(args.oracle),
        "out": str(args.out),
        "oracle_rows": len(identity_by_idx),
        "entries": entries,
        "unique_trigger_keys": len(trigger_to_addrs),
        "direct_index_rows": direct_index_rows,
        "mapped_cycle_pc_line_rows": mapped_rows,
        "unmatched_rows": len(unmatched),
        "dropped_invalid_address": dropped_invalid_address,
        "invalid_address_examples": invalid_examples,
        "key_format": "pc,line,occ,prefetch_addr where occ is zero-based occurrence of (pc,line) in no-prefetch oracle",
        "replay_semantics": "PC-line-occ keyed replay; robust to global L2 callback reordering but not equivalent to in-simulator neural inference",
    }
    meta_out = args.meta_out or args.out.with_suffix(args.out.suffix + ".meta.json")
    meta_out.write_text(json.dumps(meta, indent=2) + "\n")
    print("[ok] " + json.dumps(meta, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("[error] {}".format(exc), file=sys.stderr)
        raise SystemExit(2)
