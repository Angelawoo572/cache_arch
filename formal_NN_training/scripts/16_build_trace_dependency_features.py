#!/usr/bin/env python3
"""Build causal dynamic-dependency sidecar features for a ChampSim trace.

This utility is deliberately Python-3.6-compatible because Sacramento's
``python3`` may predate Python 3.7. It never changes the trace, labels, or
ChampSim replay. It only aligns the existing no-prefetch oracle to the original
``input_instr`` stream and emits a derived NPZ sidecar keyed by ``demand_idx``.
"""

import argparse
import json
import lzma
import struct
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# IP, branch, taken, 2 destination registers, 4 source registers,
# 2 destination-memory operands, 4 source-memory operands.
INPUT_INSTR_RECORD = struct.Struct("<QBB2B4B2Q4Q")
CACHE_LINE_BYTES = 64


def parse_int_maybe_hex(value, default=0):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return default if np.isnan(value) else int(value)
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(text, 16) if text.lower().startswith("0x") else int(float(text))
    except (TypeError, ValueError):
        return default


def first_existing_col(frame, names):
    for name in names:
        if name in frame.columns:
            return name
    return None


def numeric_col(frame, name, default=0):
    if name is None:
        return np.full(len(frame), default, dtype=np.int64)
    # .values instead of Series.to_numpy for older pandas installations.
    return np.asarray(
        frame[name].map(lambda x: parse_int_maybe_hex(x, default)).values,
        dtype=np.int64,
    )


def resolve_oracle(path):
    frame = pd.read_csv(str(path))
    pc_col = first_existing_col(frame, ("pc", "ip", "PC", "IP"))
    line_col = first_existing_col(frame, ("line", "addr_line", "address_line", "block"))
    idx_col = first_existing_col(frame, ("demand_idx", "event_idx", "idx", "index"))
    if pc_col is None or line_col is None:
        raise ValueError("{}: need PC and line columns; found {}".format(path, list(frame.columns)))
    pc = numeric_col(frame, pc_col)
    line = numeric_col(frame, line_col)
    demand_idx = numeric_col(frame, idx_col) if idx_col else np.arange(len(frame), dtype=np.int64)
    if not np.array_equal(demand_idx, np.arange(len(frame), dtype=np.int64)):
        raise ValueError("{}: demand_idx must be contiguous 0..N-1".format(path))
    return pc, line, demand_idx


def skip_records(handle, n_records):
    remaining = int(n_records)
    record_bytes = INPUT_INSTR_RECORD.size
    while remaining:
        take = min(remaining, 1 << 20)
        block = handle.read(take * record_bytes)
        got = len(block) // record_bytes
        if got == 0:
            raise RuntimeError("trace ended while skipping warmup at {}/{} records".format(
                n_records - remaining, n_records
            ))
        if len(block) % record_bytes:
            raise RuntimeError("trace contains a partial input_instr record during warmup")
        remaining -= got


def open_trace(path):
    return lzma.open(str(path), "rb") if path.suffix == ".xz" else open(str(path), "rb")


def write_sidecar(output_path, arrays):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **arrays)


def build_sidecar(trace_path, oracle_path, output_path, warmup_records,
                  progress_every, min_alignment):
    oracle_pc, oracle_line, demand_idx = resolve_oracle(oracle_path)
    n_events = len(oracle_line)
    if n_events == 0:
        raise ValueError("oracle is empty")

    src_regs = np.zeros((n_events, 4), dtype=np.uint8)
    dst_regs = np.zeros((n_events, 2), dtype=np.uint8)
    dep_present = np.zeros(n_events, dtype=np.uint8)
    dep_parent_event = np.full(n_events, -1, dtype=np.int64)
    dep_parent_pc = np.zeros(n_events, dtype=np.uint64)
    dep_parent_line = np.zeros(n_events, dtype=np.int64)
    dep_parent_dynamic_gap = np.zeros(n_events, dtype=np.int32)
    dep_chain_depth = np.zeros(n_events, dtype=np.uint16)
    dep_parent_is_load = np.zeros(n_events, dtype=np.uint8)
    branches_since_parent = np.zeros(n_events, dtype=np.int32)
    taken_since_parent = np.zeros(n_events, dtype=np.int32)
    stores_since_parent = np.zeros(n_events, dtype=np.int32)
    latest_store_line = np.zeros(n_events, dtype=np.int64)
    latest_store_dynamic_gap = np.zeros(n_events, dtype=np.int32)
    dynamic_instruction_index = np.full(n_events, -1, dtype=np.int64)
    matched_source_mem_slot = np.full(n_events, -1, dtype=np.int8)

    # Register origin tuple:
    # dynamic_seq, oracle_event_or_-1, pc, line, chain_depth,
    # branch_count, taken_count, store_count, is_load.
    register_origin = {}
    dynamic_seq = 0
    event = 0
    branch_count = 0
    taken_count = 0
    store_count = 0
    last_store_seq = -1
    last_store_line = 0
    started = time.time()

    with open_trace(trace_path) as handle:
        skip_records(handle, warmup_records)
        while event < n_events:
            raw = handle.read(INPUT_INSTR_RECORD.size)
            if not raw:
                break
            if len(raw) != INPUT_INSTR_RECORD.size:
                raise RuntimeError("partial input_instr record at dynamic instruction {}".format(dynamic_seq))
            fields = INPUT_INSTR_RECORD.unpack(raw)
            pc = int(fields[0])
            is_branch = int(fields[1])
            taken = int(fields[2])
            dst = tuple(int(x) for x in fields[3:5])
            src = tuple(int(x) for x in fields[5:9])
            dst_mem = tuple(int(x) for x in fields[9:11])
            src_mem = tuple(int(x) for x in fields[11:15])
            source_lines = [int(addr) // CACHE_LINE_BYTES for addr in src_mem if addr]
            source_slots = [slot for slot, addr in enumerate(src_mem) if addr]
            is_load = int(bool(source_lines))
            is_store = int(any(dst_mem))

            parent_candidates = [register_origin[reg] for reg in src if reg and reg in register_origin]
            parent = max(parent_candidates, key=lambda item: item[0]) if parent_candidates else None
            current_event = -1

            # Oracle events are ordered dynamic demand loads. The next oracle
            # event can only be consumed by an exact PC+line match.
            if pc == int(oracle_pc[event]) and source_lines:
                try:
                    source_index = source_lines.index(int(oracle_line[event]))
                except ValueError:
                    source_index = -1
                if source_index >= 0:
                    src_regs[event] = np.asarray(src, dtype=np.uint8)
                    dst_regs[event] = np.asarray(dst, dtype=np.uint8)
                    dynamic_instruction_index[event] = dynamic_seq
                    matched_source_mem_slot[event] = source_slots[source_index]
                    latest_store_line[event] = int(last_store_line)
                    latest_store_dynamic_gap[event] = (
                        max(0, dynamic_seq - last_store_seq) if last_store_seq >= 0 else 0
                    )
                    if parent is not None:
                        (p_seq, p_event, p_pc, p_line, p_depth,
                         p_branches, p_taken, p_stores, p_is_load) = parent
                        dep_present[event] = 1
                        dep_parent_event[event] = p_event
                        dep_parent_pc[event] = np.uint64(p_pc)
                        dep_parent_line[event] = int(p_line)
                        dep_parent_dynamic_gap[event] = max(0, dynamic_seq - p_seq)
                        dep_chain_depth[event] = min(np.iinfo(np.uint16).max, p_depth)
                        dep_parent_is_load[event] = p_is_load
                        branches_since_parent[event] = max(0, branch_count - p_branches)
                        taken_since_parent[event] = max(0, taken_count - p_taken)
                        stores_since_parent[event] = max(0, store_count - p_stores)
                    current_event = event
                    event += 1

            # Observe source dependencies before updating destination registers.
            if is_load:
                origin_line = int(source_lines[0])
                origin_depth = int(parent[4]) + 1 if parent is not None else 1
                origin = (dynamic_seq, current_event, pc, origin_line, origin_depth,
                          branch_count, taken_count, store_count, 1)
            elif parent is not None:
                origin = parent
            else:
                origin = None
            if origin is not None:
                for reg in dst:
                    if reg:
                        register_origin[reg] = origin

            if is_store:
                store_count += 1
                store_addr = next((addr for addr in dst_mem if addr), 0)
                if store_addr:
                    last_store_line = int(store_addr) // CACHE_LINE_BYTES
                    last_store_seq = dynamic_seq
            if is_branch:
                branch_count += 1
                taken_count += int(bool(taken))

            dynamic_seq += 1
            if progress_every and dynamic_seq % progress_every == 0:
                elapsed = time.time() - started
                print("[progress] dynamic={:,} aligned={:,}/{:,} ({:.4%}) elapsed={:.1f}m".format(
                    dynamic_seq, event, n_events, float(event) / n_events, elapsed / 60.0
                ), flush=True)

    alignment = float(event) / n_events
    if alignment < min_alignment or event != n_events:
        raise RuntimeError(
            "unsafe/incomplete alignment: {}/{} ({:.4%}); no output written".format(
                event, n_events, alignment
            )
        )

    arrays = {
        "version": np.asarray(["v3_9_dependency_sidecar"], dtype="U32"),
        "demand_idx": demand_idx,
        "oracle_pc": oracle_pc.astype(np.uint64, copy=False),
        "oracle_line": oracle_line.astype(np.int64, copy=False),
        "src_regs": src_regs,
        "dst_regs": dst_regs,
        "dep_present": dep_present,
        "dep_parent_event": dep_parent_event,
        "dep_parent_pc": dep_parent_pc,
        "dep_parent_line": dep_parent_line,
        "dep_parent_dynamic_gap": dep_parent_dynamic_gap,
        "dep_chain_depth": dep_chain_depth,
        "dep_parent_is_load": dep_parent_is_load,
        "branches_since_parent": branches_since_parent,
        "taken_since_parent": taken_since_parent,
        "stores_since_parent": stores_since_parent,
        "latest_store_line": latest_store_line,
        "latest_store_dynamic_gap": latest_store_dynamic_gap,
        "dynamic_instruction_index": dynamic_instruction_index,
        "matched_source_mem_slot": matched_source_mem_slot,
    }
    write_sidecar(output_path, arrays)

    present = dep_present.astype(bool)
    meta = {
        "version": "v3_9_dependency_sidecar",
        "trace": str(trace_path),
        "oracle": str(oracle_path),
        "output": str(output_path),
        "record_bytes": INPUT_INSTR_RECORD.size,
        "warmup_records_skipped": int(warmup_records),
        "oracle_events": int(n_events),
        "aligned_events": int(event),
        "alignment": float(alignment),
        "dynamic_instructions_scanned_after_warmup": int(dynamic_seq),
        "dependency_present_events": int(dep_present.sum()),
        "dependency_present_fraction": float(dep_present.mean()),
        "median_parent_dynamic_gap": float(np.median(dep_parent_dynamic_gap[present])) if present.any() else 0.0,
        "source": "input_instr dynamic trace; no register or memory values are available",
    }
    meta_path = output_path.with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")
    print("[saved] {}".format(output_path))
    print("[saved] {}".format(meta_path))
    print(json.dumps(meta, indent=2))
    return meta


def choose_oracle(trace_stem, oracle, oracle_dir):
    if oracle:
        path = Path(oracle)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return path
    if not oracle_dir:
        raise ValueError("pass --oracle or --oracle-dir")
    root = Path(oracle_dir)
    matches = sorted(
        path for path in root.rglob("*")
        if path.is_file() and trace_stem in path.name and path.suffix in (".csv", ".gz")
    )
    if len(matches) != 1:
        raise RuntimeError("expected exactly one oracle under {} containing {!r}; found {}".format(
            root, trace_stem, matches
        ))
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, help="original .champsimtrace or .xz")
    parser.add_argument("--oracle", default=None, help="explicit no-prefetch oracle CSV/CSV.GZ")
    parser.add_argument("--oracle-dir", default=None, help="auto-find oracle by trace stem")
    parser.add_argument("--trace-stem", default="605.mcf_s-994B")
    parser.add_argument("--output", required=True, help="output .npz")
    parser.add_argument("--warmup-records", type=int, default=25000000)
    parser.add_argument("--progress-every", type=int, default=1000000)
    parser.add_argument("--min-alignment", type=float, default=0.9999)
    args = parser.parse_args()

    trace_path = Path(args.trace)
    if not trace_path.is_file():
        raise FileNotFoundError(str(trace_path))
    oracle_path = choose_oracle(args.trace_stem, args.oracle, args.oracle_dir)
    output_path = Path(args.output)
    if output_path.suffix != ".npz":
        raise ValueError("--output must end in .npz")
    build_sidecar(trace_path, oracle_path, output_path, args.warmup_records,
                  args.progress_every, args.min_alignment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
