#!/usr/bin/env python3
"""
Build, from one decoded raw-prefix scan of a ChampSim input_instr trace:

  (1) a PC-keyed static dependency profile; and optionally
  (2) a bounded producer-PC dependency edge vocabulary.

The raw input_instr stream is in program order, while the no-prefetch oracle is
an L2 request-service stream. They are not losslessly event-alignable. This
script therefore never creates an event-by-event join. Both outputs are static,
prefix-trained artifacts. At notebook time, the vocabulary may be used only with
the current oracle event's (pc, line) and past oracle history; it must never read
future raw-trace addresses.

The vocabulary is deliberately keyed by producer PC, not by raw instruction-gap
bucket or consumer PC. On a current producer load event (P, line L), runtime may
instantiate candidates L + delta for the top deltas learned for P. Consumer PC is
retained only as diagnostic evidence. This makes the lookup implementable from
the oracle stream and keeps the candidate budget bounded per producer.
"""
from __future__ import print_function

import argparse
import csv
import gzip
import hashlib
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
EDGE_SCHEMA = "v3_9_pc_dependency_edge_vocab"
EDGE_SCHEMA_VERSION = "v3_9_producer_delta_vocab_1"

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

EDGE_CSV_FIELDS = [
    "producer_pc",
    "producer_to_target_line_delta",
    "estimated_support",
    "support_lower_bound",
    "support_error_bound",
    "rank_within_producer",
    "representative_consumer_pc",
    "consumer_pc_slots",
    "producer_is_load",
]


def open_trace(path):
    return lzma.open(path, "rb") if str(path).endswith(".xz") else open(path, "rb")


def sha256_file(path, block=1 << 20):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        while True:
            chunk = handle.read(block)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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
    """Bounded Space-Saving-style counter used for profile diagnostics."""
    value = int(value)
    if value in hist:
        hist[value] += 1
        return
    if len(hist) < max_keys:
        hist[value] = 1
        return
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


def first_load_line(src_mem, line_bytes):
    for addr in src_mem:
        if addr:
            return int(addr) // int(line_bytes)
    return None


def make_profile_state():
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


def make_delta_state(consumer_pc):
    return {
        "estimate": 1,
        "error": 0,
        "consumers": {int(consumer_pc): 1},
    }


def observe_producer_delta(bucket, delta, consumer_pc, max_deltas, max_consumers):
    """Update a bounded per-producer Space-Saving delta table.

    estimate is an upper estimate; lower bound is estimate - error. Consumer
    counters are diagnostics only and are also bounded/approximate.
    """
    delta = int(delta)
    consumer_pc = int(consumer_pc)
    state = bucket.get(delta)
    if state is None:
        if len(bucket) < int(max_deltas):
            bucket[delta] = make_delta_state(consumer_pc)
            return False
        victim_delta, victim_state = min(
            bucket.items(), key=lambda item: (item[1]["estimate"], item[0])
        )
        victim_estimate = int(victim_state["estimate"])
        del bucket[victim_delta]
        bucket[delta] = {
            "estimate": victim_estimate + 1,
            "error": victim_estimate,
            "consumers": {consumer_pc: 1},
        }
        return True

    state["estimate"] += 1
    add_counter_bounded(state["consumers"], consumer_pc, max_consumers)
    return False


def representative_consumer(state):
    consumers = state.get("consumers", {})
    if not consumers:
        return 0
    return int(max(consumers.items(), key=lambda item: (item[1], -item[0]))[0])


def write_csv_gz(path, fieldnames, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".partial")
    with gzip.open(str(tmp), "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    os.replace(str(tmp), str(path))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".partial")
    with open(str(tmp), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(tmp), str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, help="original .champsimtrace or .xz")
    parser.add_argument("--output", required=True, help="output PC-static profile .csv.gz")
    parser.add_argument("--meta", default=None, help="profile metadata .json")
    parser.add_argument("--edge-output", default=None, help="output producer-keyed edge vocabulary .csv.gz")
    parser.add_argument("--edge-meta", default=None, help="edge vocabulary metadata .json")
    parser.add_argument("--edge-top-k", type=int, default=8,
                        help="maximum exported candidate deltas per producer PC")
    parser.add_argument("--edge-min-support", type=int, default=16,
                        help="minimum conservative support lower bound for an exported delta")
    parser.add_argument("--edge-max-deltas-per-producer", type=int, default=256,
                        help="bounded Space-Saving slots tracked per producer PC")
    parser.add_argument("--edge-max-consumers-per-delta", type=int, default=4,
                        help="bounded diagnostic consumer-PC slots tracked per active delta")
    parser.add_argument("--edge-max-producers", type=int, default=250000,
                        help="safety cap on producer PCs admitted to the edge table")
    parser.add_argument("--warmup-records", type=int, default=25000000)
    parser.add_argument("--profile-records", type=int, default=20000000,
                        help="post-warmup raw instructions used for the static training-prefix sidecars")
    parser.add_argument("--progress-every", type=int, default=1000000)
    parser.add_argument("--max-parent-pcs", type=int, default=8)
    parser.add_argument("--max-hist-buckets", type=int, default=64)
    parser.add_argument("--no-sha256", action="store_true",
                        help="skip the optional compressed-trace checksum pass")
    parser.add_argument("--dry-run", action="store_true", help="parse and validate only; do not write outputs")
    args = parser.parse_args()

    trace_path = Path(args.trace)
    output_path = Path(args.output)
    meta_path = Path(args.meta) if args.meta else output_path.with_suffix("").with_suffix(".json")
    edge_output_path = Path(args.edge_output) if args.edge_output else None
    edge_meta_path = None
    if edge_output_path is not None:
        edge_meta_path = Path(args.edge_meta) if args.edge_meta else edge_output_path.with_suffix("").with_suffix(".json")

    if not trace_path.is_file():
        raise SystemExit("[error] trace not found: {}".format(trace_path))
    if output_path.suffix != ".gz" or not str(output_path).endswith(".csv.gz"):
        raise SystemExit("[error] --output must end in .csv.gz")
    if edge_output_path is not None:
        if edge_output_path.suffix != ".gz" or not str(edge_output_path).endswith(".csv.gz"):
            raise SystemExit("[error] --edge-output must end in .csv.gz")
    if args.warmup_records < 0 or args.profile_records <= 0:
        raise SystemExit("[error] warmup must be >= 0 and profile-records must be > 0")
    if args.edge_top_k <= 0 or args.edge_min_support <= 0:
        raise SystemExit("[error] --edge-top-k and --edge-min-support must be > 0")
    if args.edge_max_deltas_per_producer < args.edge_top_k:
        raise SystemExit("[error] --edge-max-deltas-per-producer must be >= --edge-top-k")
    if args.edge_max_consumers_per_delta <= 0 or args.edge_max_producers <= 0:
        raise SystemExit("[error] edge safety caps must be > 0")

    want_edges = edge_output_path is not None

    # reg -> (producer_pc, producer_sequence, producer_depth, producer_is_load, producer_line)
    origins = {}
    profiles = defaultdict(make_profile_state)
    # producer_pc -> {delta -> bounded Space-Saving state}
    producer_delta_hist = {}
    producer_cap_dropped_observations = 0
    delta_slot_evictions = 0
    qualified_edge_observations = 0
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

            consumer_line = None
            if is_load:
                state["load_instruction_count"] += 1
                load_count += 1
                consumer_line = first_load_line(src_mem, CACHE_LINE_BYTES)

                if parent is not None:
                    parent_pc, parent_seq, parent_depth, parent_is_load, parent_line = parent
                    gap = max(0, sequence - parent_seq)
                    depth = max(1, parent_depth)
                    state["dep_observations"] += 1
                    add_counter_bounded(state["parent_pcs"], parent_pc, args.max_parent_pcs)
                    state["parent_is_load"] += int(parent_is_load)
                    state["gap_sum"] += gap
                    state["depth_sum"] += depth
                    bounded_hist_add(state["gap_hist"], gap, args.max_hist_buckets)
                    bounded_hist_add(state["depth_hist"], depth, args.max_hist_buckets)

                    # Candidate vocabulary: one producer-keyed delta source. Consumer
                    # PC and raw gap are learned evidence only; neither is a runtime key.
                    if want_edges and int(parent_is_load) == 1 and parent_line is not None and consumer_line is not None:
                        qualified_edge_observations += 1
                        bucket = producer_delta_hist.get(parent_pc)
                        if bucket is None:
                            if len(producer_delta_hist) >= int(args.edge_max_producers):
                                producer_cap_dropped_observations += 1
                            else:
                                bucket = {}
                                producer_delta_hist[parent_pc] = bucket
                        if bucket is not None:
                            evicted = observe_producer_delta(
                                bucket=bucket,
                                delta=consumer_line - parent_line,
                                consumer_pc=pc,
                                max_deltas=args.edge_max_deltas_per_producer,
                                max_consumers=args.edge_max_consumers_per_delta,
                            )
                            delta_slot_evictions += int(evicted)

            # Read dependencies before defining destinations. A load becomes a new
            # causal producer carrying its own raw load line; arithmetic instructions
            # propagate the youngest load-originating producer unchanged.
            if is_load:
                depth = (parent[2] + 1) if parent is not None else 1
                new_origin = (pc, sequence, depth, 1, consumer_line)
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
                    "[progress] raw_train_records={:,}/{:,} pcs={:,} loads={:,} "
                    "edge_producers={:,} edge_obs={:,} evictions={:,} elapsed={:.1f}m".format(
                        sequence, args.profile_records, len(profiles), load_count,
                        len(producer_delta_hist), qualified_edge_observations,
                        delta_slot_evictions, elapsed / 60.0
                    ),
                    flush=True,
                )

    profile_rows = []
    for pc in sorted(profiles):
        state = profiles[pc]
        signature, _ = state["signatures"].most_common(1)[0]
        parent_top = state["parent_pcs"].most_common(2)
        parent0 = int(parent_top[0][0]) if parent_top else 0
        parent0_count = int(parent_top[0][1]) if parent_top else 0
        parent1 = int(parent_top[1][0]) if len(parent_top) > 1 else 0
        parent1_count = int(parent_top[1][1]) if len(parent_top) > 1 else 0
        deps = int(state["dep_observations"])
        profile_rows.append({
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

    edge_rows = []
    if want_edges:
        for producer_pc in sorted(producer_delta_hist):
            bucket = producer_delta_hist[producer_pc]
            candidates = []
            for delta, state in bucket.items():
                estimate = int(state["estimate"])
                error = int(state["error"])
                lower_bound = max(0, estimate - error)
                if lower_bound < int(args.edge_min_support):
                    continue
                candidates.append((delta, estimate, error, lower_bound, state))
            candidates.sort(key=lambda item: (-item[1], -item[3], abs(item[0]), item[0]))
            for rank, (delta, estimate, error, lower_bound, state) in enumerate(candidates[:int(args.edge_top_k)]):
                edge_rows.append({
                    "producer_pc": int(producer_pc),
                    "producer_to_target_line_delta": int(delta),
                    "estimated_support": int(estimate),
                    "support_lower_bound": int(lower_bound),
                    "support_error_bound": int(error),
                    "rank_within_producer": int(rank),
                    "representative_consumer_pc": representative_consumer(state),
                    "consumer_pc_slots": int(len(state.get("consumers", {}))),
                    "producer_is_load": 1,
                })

    trace_sha = None if args.no_sha256 else sha256_file(trace_path)
    elapsed_seconds = round(time.time() - start, 3)
    profile_meta = {
        "schema": "v3_9_pc_static_dependency_profile",
        "trace": str(trace_path),
        "trace_sha256": trace_sha,
        "record_bytes": INPUT_INSTR_RECORD.size,
        "cache_line_bytes": CACHE_LINE_BYTES,
        "warmup_records_skipped": int(args.warmup_records),
        "profile_records": int(args.profile_records),
        "profile_scope": "raw-trace training prefix only",
        "uses_oracle_alignment": False,
        "why_no_oracle_alignment": (
            "input_instr is program-order while the existing oracle is an L2 request-service stream; "
            "a lossless event-by-event join is not available from these two artifacts alone"
        ),
        "unique_pcs": len(profile_rows),
        "pcs_with_dependency_observations": sum(
            1 for row in profile_rows if int(row["dependency_observations"]) > 0
        ),
        "load_instructions": int(load_count),
        "elapsed_seconds": elapsed_seconds,
    }

    edge_meta = None
    if want_edges:
        edge_meta = {
            "schema": EDGE_SCHEMA,
            "schema_version": EDGE_SCHEMA_VERSION,
            "trace": str(trace_path),
            "trace_sha256": trace_sha,
            "record_bytes": INPUT_INSTR_RECORD.size,
            "cache_line_bytes": CACHE_LINE_BYTES,
            "warmup_records": int(args.warmup_records),
            "profile_records": int(args.profile_records),
            "profile_scope": "raw-trace training prefix only",
            "uses_oracle_alignment": False,
            "runtime_lookup_key": "producer_pc",
            "runtime_generation_rule": (
                "on current oracle event (producer_pc, current_line), emit "
                "current_line + producer_to_target_line_delta for that producer_pc"
            ),
            "runtime_restriction": (
                "consumer_pc and raw instruction gaps are diagnostics only; runtime may use only "
                "the current oracle event and past oracle history, never future or live raw-trace addresses"
            ),
            "edge_definition": (
                "producer_to_target_line_delta = dependent_consumer_load_line - originating_producer_load_line; "
                "this is an address-relation proposal, not a value-based pointer chase"
            ),
            "support_semantics": (
                "estimated_support is a bounded Space-Saving upper estimate; support_lower_bound = "
                "estimated_support - support_error_bound; export requires lower_bound >= min_support_lower_bound"
            ),
            "min_support_lower_bound": int(args.edge_min_support),
            "top_k_per_producer": int(args.edge_top_k),
            "max_deltas_per_producer": int(args.edge_max_deltas_per_producer),
            "max_consumers_per_delta": int(args.edge_max_consumers_per_delta),
            "max_producers": int(args.edge_max_producers),
            "qualified_edge_observations": int(qualified_edge_observations),
            "producer_cap_dropped_observations": int(producer_cap_dropped_observations),
            "delta_slot_evictions": int(delta_slot_evictions),
            "producer_pcs_tracked": int(len(producer_delta_hist)),
            "rows": int(len(edge_rows)),
            "distinct_producers_exported": int(len(set(row["producer_pc"] for row in edge_rows))),
            "decoded_raw_prefix_passes": 1,
            "checksum_file_passes": 0 if args.no_sha256 else 1,
            "elapsed_seconds": elapsed_seconds,
        }

    if args.dry_run:
        print(json.dumps(profile_meta, indent=2, sort_keys=True))
        if want_edges:
            print(json.dumps(edge_meta, indent=2, sort_keys=True))
            for row in edge_rows[:8]:
                print("[dry-run edge] {}".format(row))
        return 0

    write_csv_gz(output_path, CSV_FIELDS, profile_rows)
    write_json(meta_path, profile_meta)
    print("[saved] {}".format(output_path))
    print("[saved] {}".format(meta_path))
    if want_edges:
        write_csv_gz(edge_output_path, EDGE_CSV_FIELDS, edge_rows)
        write_json(edge_meta_path, edge_meta)
        print("[saved] {}".format(edge_output_path))
        print("[saved] {}".format(edge_meta_path))
    print(json.dumps(profile_meta, indent=2, sort_keys=True))
    if want_edges:
        print(json.dumps(edge_meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("[interrupted] no completed sidecar was committed", file=sys.stderr)
        sys.exit(130)
