#!/usr/bin/env python3
"""Convert a rich oracle-LSTM export into strict list-replayer input.

Why this exists
---------------
The notebook's diagnostic CSV has columns such as
    order,pc,line,...,prefetch_addr
where ``order`` historically means simulator cycle, not a dynamic access index.
A list replayer that expects ``idx,0xprefetch_addr`` will otherwise parse:
    idx  <- cycle
    addr <- pc (as hexadecimal)
which silently produces zero matches or wrong targets.

This tool writes exactly two columns:
    idx,prefetch_addr
    <ROI L2 LOAD ordinal>,0x<byte-address>

For current rich exports without ``replay_idx`` / ``demand_idx``, it maps each
(order=cycle, pc, line) trigger back to the no-prefetch oracle table and uses
that row's demand_idx. Trigger mapping is strict by default: an unmatched
trigger is an error, never a silently dropped prefetch.

A model may occasionally reconstruct an invalid address (negative page/line,
64-bit overflow, or non-cache-line alignment). That is a legal *policy reject*,
not a mapping ambiguity. Such candidates are dropped explicitly and counted in
the JSON sidecar. The notebook should also filter them at export time, but this
converter keeps existing rich exports safe and replayable.

This file intentionally uses Python 3.6-compatible syntax because the
Sacramento login nodes may provide Python 3.6.
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


def load_oracle_index(oracle_path):
    """Build cycle/pc/line -> FIFO ROI-L2-load-index mapping from no-pref oracle."""
    mapping = defaultdict(list)
    positions = defaultdict(int)

    with open_text(oracle_path) as f:
        reader = csv.DictReader(f)
        required = set(["demand_idx", "cycle", "pc", "line"])
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError("oracle {} missing columns: {}".format(oracle_path, sorted(missing)))

        previous = -1
        for row in reader:
            idx = to_int(row["demand_idx"], "oracle demand_idx")
            if idx <= previous:
                raise ValueError("oracle demand_idx must be strictly increasing")
            previous = idx
            key = trigger_key(
                to_int(row["cycle"], "oracle cycle"),
                to_int(row["pc"], "oracle pc"),
                to_int(row["line"], "oracle line"),
            )
            mapping[key].append(idx)

    if not mapping:
        raise ValueError("oracle has no rows: {}".format(oracle_path))
    return mapping, positions


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
                    help="Notebook's rich fair_dedup CSV")
    ap.add_argument("--oracle", required=True, type=Path,
                    help="Matching no-prefetch oracle CSV(.gz); used only when rich-list lacks replay_idx/demand_idx")
    ap.add_argument("--out", required=True, type=Path,
                    help="Strict two-column list-replayer CSV")
    ap.add_argument("--meta-out", type=Path, default=None,
                    help="Optional JSON validation summary (default: <out>.meta.json)")
    ap.add_argument("--allow-unmatched", action="store_true",
                    help="Drop unmatched rich rows instead of failing. Do not use for formal replay.")
    ap.add_argument("--fail-on-invalid-address", action="store_true",
                    help="Treat invalid neural addresses as fatal instead of filtering/counting them.")
    args = ap.parse_args()

    if not args.rich_list.is_file():
        raise FileNotFoundError(str(args.rich_list))
    if not args.oracle.is_file():
        raise FileNotFoundError(str(args.oracle))

    oracle_map, oracle_positions = load_oracle_index(args.oracle)
    # Cache the most recently resolved key. A future degree-k export can carry
    # several targets for one trigger without consuming several oracle rows.
    last_key = None
    last_idx = None
    used_indices = set()
    rows = []
    unmatched = []
    invalid_examples = []
    direct_index_rows = 0
    mapped_rows = 0
    dropped_invalid_address = 0

    for row_no, row in enumerate(iter_rich_rows(args.rich_list), start=2):
        try:
            addr = to_int(row.get("prefetch_addr"), "prefetch_addr at rich row {}".format(row_no))
        except ValueError:
            # A missing/non-numeric address means the rich export schema is corrupt,
            # not merely a bad neural candidate.
            raise

        valid_address = (addr > 0 and addr <= U64_MAX and addr % CACHE_LINE_BYTES == 0)
        if not valid_address:
            dropped_invalid_address += 1
            if len(invalid_examples) < 5:
                invalid_examples.append({"row": row_no, "prefetch_addr": str(addr)})
            if args.fail_on_invalid_address:
                raise ValueError("invalid prefetch_addr at rich row {}: {}".format(row_no, addr))
            continue

        # New notebook exports carry replay_idx directly. Existing exports do
        # not; their `order` is a cycle count and must never be used as an idx.
        direct = row.get("replay_idx") or row.get("demand_idx")
        if direct not in (None, ""):
            idx = to_int(direct, "replay_idx at rich row {}".format(row_no))
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

        rows.append((idx, addr))
        used_indices.add(idx)

    if unmatched and not args.allow_unmatched:
        preview = "; ".join("row {}: {}".format(x["row"], x["reason"]) for x in unmatched[:5])
        raise RuntimeError(
            "{} rich rows could not be mapped to ROI L2 LOAD indices. Examples: {}. "
            "Refusing to create a partial replay list.".format(len(unmatched), preview)
        )
    if not rows:
        raise RuntimeError("no replay entries produced")

    rows.sort(key=lambda x: x[0])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        f.write("idx,prefetch_addr\n")
        for idx, addr in rows:
            f.write("{},0x{:x}\n".format(idx, addr))

    meta = {
        "rich_list": str(args.rich_list),
        "oracle": str(args.oracle),
        "out": str(args.out),
        "entries": len(rows),
        "unique_indices": len(used_indices),
        "min_idx": min(idx for idx, _ in rows),
        "max_idx": max(idx for idx, _ in rows),
        "direct_index_rows": direct_index_rows,
        "mapped_cycle_pc_line_rows": mapped_rows,
        "unmatched_rows": len(unmatched),
        "dropped_invalid_address": dropped_invalid_address,
        "invalid_address_examples": invalid_examples,
        "format": "idx,prefetch_addr where idx is ROI L2 LOAD demand_idx and address is an aligned positive uint64 byte-address in 0xhex",
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
