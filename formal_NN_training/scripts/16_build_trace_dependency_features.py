#!/usr/bin/env python3
"""
Build a prefix-only static dependency profile from a ChampSim input_instr trace.

Why this is a PC-keyed profile rather than an oracle-row sidecar:
the existing no-prefetch oracle is emitted at L2 request-service time, while
input_instr is a program-order trace. They are not a losslessly joinable
event stream. This tool therefore never fabricates an event-by-event join.

Instead, it scans only the raw-trace training prefix and writes one profile row
per instruction PC. The v3.9 notebook joins this profile by PC to add static
source/destination-register signatures and causal producer-PC statistics. It
does not use validation oracle labels or normal-prefetcher output.
"""
import argparse
import csv
import gzip
import json
import lzma
import os
import struct
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

INPUT_INSTR_RECORD = struct.Struct("<QBB2B4B2Q4Q")
CACHE_LINE_BYTES = 64

CSV_FIELDS = [
    "pc",
    "src_reg0", "src_reg1", "src_reg2", "src_reg3",
    "dst_reg0", "dst_reg1",
    "instruction_count", "load_instruction_count",
    "signature_variant_count",
    "dependency_observations",
    "parent_pc0", "parent_pc0_count",
    "parent_pc1", "parent_pc1_count",
    "parent_is_load_ppm",
    "parent_gap_median",
    "parent_gap_mean",
    "parent_depth_median",
    "parent_depth_mean",
]


def open_trace(path):
    return lzma.open(path, "rb") if str(path).endswith(".xz") else open(path, "rb")


def skip_records(handle, count):
    remaining = int(count)
    width = INPUT_INSTR_RECORD.size
    while remaining > 0:
        take = min(remaining, 1 << 20)
        block = handle.read(take * width)
        got = len(block) // width
        if got <= 0:
            raise RuntimeError(
                "trace ended while skipping warmup: skipped {}/{} records".format(
                    int(count) - remaining, int(count)
                )
            )
        if len(block) != got * width:
            raise RuntimeError("trace contains a partial input_instr record")
        remaining -= got


def median_from_hist(hist):
    total = sum(hist.values())
    if total <= 0:
        return 0
    target = (total - 1) // 2
    run = 0
    for value in sorted(hist):
        run += hist[value]
        if run > target:
            return int(value)
    return int(max(hist))


def bounded_hist_add(hist, value, max_keys):
    value = int(value)
    if value in hist:
        hist[value] += 1
        return
    if len(hist) < max_keys:
        hist[value] = 1
        return
    # Bounded histogram: retain heavy, representative buckets without allowing
    # a trace with many unique gaps to consume unbounded memory.
    victim = min(hist, key=hist.get)
    victim_count = hist.pop(victim)
    hist[value] = victim_count + 1


def add_counter_bounded(counter, value, max_keys):
    value = int(value)
    if value in counter:
        counter[value] += 1
        return
    if len(counter) < max_keys:
        counter[value] = 1
        return
    victim = min(counter, key=counter.get)
    victim_count = counter.pop(victim)
    counter[value] = victim_count + 1


def make_state():
    return {
        "instruction_count": 0,
        "load_instruction_count": 0,
        "signatures": Counter(),
        "dep_observations": 0,
        "parent_pcs": Counter(),
        "parent_is_load": 0,
        "gap_sum": 0,
        "depth_sum": 0,
        "gap_hist": {},
        "depth_hist": {},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, help="original .champsimtrace or .xz")
    parser.add_argument("--output", required=True, help="output .csv.gz profile")
    parser.add_argument("--meta", default=None, help="output metadata .json (default derived from --output)")
    parser.add_argument("--warmup-records", type=int, default=25000000)
    parser.add_argument(
        "--profile-records",
        type=int,
        default=20000000,
        help="number of post-warmup raw instructions used; default is the 80%% train prefix of 25M simulation",
    )
    parser.add_argument("--progress-every", type=int, default=1000000)
    parser.add_argument("--max-parent-pcs", type=int, default=8)
    parser.add_argument("--max-hist-buckets", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true", help="parse and validate only; do not write outputs")
    args = parser.parse_args()

    trace_path = Path(args.trace)
    output_path = Path(args.output)
    meta_path = Path(args.meta) if args.meta else output_path.with_suffix("").with_suffix(".json")
    if not trace_path.is_file():
        raise SystemExit("[error] trace not found: {}".format(trace_path))
    if output_path.suffix != ".gz" or not str(output_path).endswith(".csv.gz"):
        raise SystemExit("[error] --output must end in .csv.gz")
    if args.warmup_records < 0 or args.profile_records <= 0:
        raise SystemExit("[error] warmup must be >= 0 and profile-records must be > 0")

    # reg -> (producer_pc, producer_sequence, producer_depth, producer_is_load)
    origins = {}
    profiles = defaultdict(make_state)
    sequence = 0
    load_count = 0
    start = time.time()

    with open_trace(str(trace_path)) as handle:
        skip_records(handle, args.warmup_records)
        for local_index in range(int(args.profile_records)):
            record = handle.read(INPUT_INSTR_RECORD.size)
            if len(record) != INPUT_INSTR_RECORD.size:
                raise RuntimeError(
                    "trace ended or has partial record after {} post-warmup instructions".format(local_index)
                )
            fields = INPUT_INSTR_RECORD.unpack(record)
            pc = int(fields[0])
            # The two one-byte flags are not required for the dependency profile.
            dst = tuple(int(x) for x in fields[3:5])
            src = tuple(int(x) for x in fields[5:9])
            src_mem = tuple(int(x) for x in fields[11:15])
            is_load = int(any(src_mem))

            state = profiles[pc]
            state["instruction_count"] += 1
            signature = src + dst
            state["signatures"][signature] += 1
            parents = [origins[r] for r in src if r and r in origins]
            parent = max(parents, key=lambda item: item[1]) if parents else None

            if is_load:
                state["load_instruction_count"] += 1
                load_count += 1
                if parent is not None:
                    parent_pc, parent_seq, parent_depth, parent_is_load = parent
                    gap = max(0, sequence - parent_seq)
                    depth = max(1, parent_depth)
                    state["dep_observations"] += 1
                    add_counter_bounded(state["parent_pcs"], parent_pc, args.max_parent_pcs)
                    state["parent_is_load"] += int(parent_is_load)
                    state["gap_sum"] += gap
                    state["depth_sum"] += depth
                    bounded_hist_add(state["gap_hist"], gap, args.max_hist_buckets)
                    bounded_hist_add(state["depth_hist"], depth, args.max_hist_buckets)

            # Read dependencies before defining destinations. A load destination
            # becomes a fresh producer; arithmetic instructions propagate the
            # youngest causal producer they read.
            if is_load:
                depth = (parent[2] + 1) if parent is not None else 1
                new_origin = (pc, sequence, depth, 1)
            elif parent is not None:
                new_origin = parent
            else:
                new_origin = None
            if new_origin is not None:
                for reg in dst:
                    if reg:
                        origins[reg] = new_origin

            sequence += 1
            if args.progress_every and sequence % int(args.progress_every) == 0:
                elapsed = time.time() - start
                print(
                    "[progress] raw_train_records={:,}/{:,} pcs={:,} loads={:,} elapsed={:.1f}m".format(
                        sequence, args.profile_records, len(profiles), load_count, elapsed / 60.0
                    ),
                    flush=True,
                )

    rows = []
    for pc in sorted(profiles):
        state = profiles[pc]
        signature, _ = state["signatures"].most_common(1)[0]
        parent_top = state["parent_pcs"].most_common(2)
        parent0 = int(parent_top[0][0]) if parent_top else 0
        parent0_count = int(parent_top[0][1]) if parent_top else 0
        parent1 = int(parent_top[1][0]) if len(parent_top) > 1 else 0
        parent1_count = int(parent_top[1][1]) if len(parent_top) > 1 else 0
        deps = int(state["dep_observations"])
        rows.append({
            "pc": pc,
            "src_reg0": signature[0], "src_reg1": signature[1],
            "src_reg2": signature[2], "src_reg3": signature[3],
            "dst_reg0": signature[4], "dst_reg1": signature[5],
            "instruction_count": int(state["instruction_count"]),
            "load_instruction_count": int(state["load_instruction_count"]),
            "signature_variant_count": int(len(state["signatures"])),
            "dependency_observations": deps,
            "parent_pc0": parent0, "parent_pc0_count": parent0_count,
            "parent_pc1": parent1, "parent_pc1_count": parent1_count,
            "parent_is_load_ppm": int(round(1000000.0 * state["parent_is_load"] / deps)) if deps else 0,
            "parent_gap_median": median_from_hist(state["gap_hist"]),
            "parent_gap_mean": int(round(float(state["gap_sum"]) / deps)) if deps else 0,
            "parent_depth_median": median_from_hist(state["depth_hist"]),
            "parent_depth_mean": int(round(float(state["depth_sum"]) / deps)) if deps else 0,
        })

    dependency_pcs = sum(1 for row in rows if int(row["dependency_observations"]) > 0)
    meta = {
        "schema": "v3_9_pc_static_dependency_profile",
        "trace": str(trace_path),
        "record_bytes": INPUT_INSTR_RECORD.size,
        "warmup_records_skipped": int(args.warmup_records),
        "profile_records": int(args.profile_records),
        "profile_scope": "raw-trace training prefix only",
        "uses_oracle_alignment": False,
        "why_no_oracle_alignment": (
            "input_instr is program-order while the existing oracle is an L2 request-service stream; "
            "a lossless event-by-event join is not available from these two artifacts alone"
        ),
        "unique_pcs": len(rows),
        "pcs_with_dependency_observations": dependency_pcs,
        "load_instructions": int(load_count),
        "elapsed_seconds": round(time.time() - start, 3),
    }

    if args.dry_run:
        print(json.dumps(meta, indent=2))
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = Path(str(output_path) + ".partial")
    tmp_meta = Path(str(meta_path) + ".partial")
    with gzip.open(str(tmp_output), "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with open(str(tmp_meta), "w") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(tmp_output), str(output_path))
    os.replace(str(tmp_meta), str(meta_path))
    print("[saved] {}".format(output_path))
    print("[saved] {}".format(meta_path))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("[interrupted] no completed sidecar was committed", file=sys.stderr)
        sys.exit(130)
