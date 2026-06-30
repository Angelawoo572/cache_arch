#!/usr/bin/env python3
"""
Build causal dynamic-dependency sidecar features for one ChampSim input_instr trace.

This script does NOT change the trace, oracle labels, or ChampSim replay protocol.
It reads the existing no-prefetch oracle and original .champsimtrace(.xz) stream,
aligns each oracle demand event to the matching dynamic instruction by (PC, line),
and writes a compressed NumPy sidecar keyed by oracle demand_idx.

The sidecar is intentionally separate from Git-tracked sources/results. It is a
derived artifact for v3.9 605.mcf_s dependency-aware modeling.
"""
from __future__ import annotations

import argparse
import gzip
import json
import lzma
import os
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

INPUT_INSTR_RECORD = struct.Struct("<QBB2B4B2Q4Q")
CACHE_LINE_BYTES = 64
PAGE_LINES = 64


def parse_int_maybe_hex(value: object, default: int = 0) -> int:
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


def first_existing_col(frame: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    return next((name for name in names if name in frame.columns), None)


def numeric_col(frame: pd.DataFrame, name: Optional[str], default: int = 0) -> np.ndarray:
    if name is None:
        return np.full(len(frame), default, dtype=np.int64)
    # Keep hexadecimal string support, which pd.to_numeric does not provide.
    return frame[name].map(lambda x: parse_int_maybe_hex(x, default)).to_numpy(np.int64)


def resolve_oracle(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    pc_col = first_existing_col(frame, ("pc", "ip", "PC", "IP"))
    line_col = first_existing_col(frame, ("line", "addr_line", "address_line", "block"))
    idx_col = first_existing_col(frame, ("demand_idx", "event_idx", "idx", "index"))
    if pc_col is None or line_col is None:
        raise ValueError(
            f"{path}: need PC and line columns; found {list(frame.columns)}"
        )
    pc = numeric_col(frame, pc_col)
    line = numeric_col(frame, line_col)
    demand_idx = numeric_col(frame, idx_col) if idx_col else np.arange(len(frame), dtype=np.int64)
    expected = np.arange(len(frame), dtype=np.int64)
    if not np.array_equal(demand_idx, expected):
        raise ValueError(
            f"{path}: demand_idx must be contiguous 0..N-1 to make a safe sidecar"
        )
    return pc, line, demand_idx


def skip_records(handle, n_records: int) -> None:
    remaining = int(n_records)
    bytes_per = INPUT_INSTR_RECORD.size
    while remaining:
        take = min(remaining, 1 << 20)
        block = handle.read(take * bytes_per)
        got = len(block) // bytes_per
        if got == 0:
            raise RuntimeError(
                f"trace ended while skipping warmup: skipped {n_records - remaining}/{n_records}"
            )
        if len(block) % bytes_per:
            raise RuntimeError("trace has a partial input_instr record while skipping warmup")
        remaining -= got


def _save_npz(output: Path, arrays: Dict[str, np.ndarray]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)


def build_sidecar(
    *,
    trace_path: Path,
    oracle_path: Path,
    output_path: Path,
    warmup_records: int,
    progress_every: int,
    min_alignment: float,
) -> Dict[str, object]:
    oracle_pc, oracle_line, demand_idx = resolve_oracle(oracle_path)
    n = len(oracle_line)
    if n == 0:
        raise ValueError("oracle is empty")

    # Event-aligned arrays. int64 preserves true addresses; v3.9 hashes/buckets
    # them only after the causal alignment audit passes.
    src_regs = np.zeros((n, 4), dtype=np.uint8)
    dst_regs = np.zeros((n, 2), dtype=np.uint8)
    dep_present = np.zeros(n, dtype=np.uint8)
    dep_parent_event = np.full(n, -1, dtype=np.int64)
    dep_parent_pc = np.zeros(n, dtype=np.uint64)
    dep_parent_line = np.zeros(n, dtype=np.int64)
    dep_parent_dynamic_gap = np.zeros(n, dtype=np.int32)
    dep_chain_depth = np.zeros(n, dtype=np.uint16)
    dep_parent_is_load = np.zeros(n, dtype=np.uint8)
    branches_since_parent = np.zeros(n, dtype=np.int32)
    taken_since_parent = np.zeros(n, dtype=np.int32)
    stores_since_parent = np.zeros(n, dtype=np.int32)
    latest_store_line = np.zeros(n, dtype=np.int64)
    latest_store_dynamic_gap = np.zeros(n, dtype=np.int32)
    dynamic_instruction_index = np.full(n, -1, dtype=np.int64)
    matched_source_mem_slot = np.full(n, -1, dtype=np.int8)

    # Register origin:
    # (dynamic_seq, oracle_event_or_-1, pc, line, chain_depth,
    #  branch_counter, taken_counter, store_counter, is_load)
    register_origin: Dict[int, Tuple[int, int, int, int, int, int, int, int, int]] = {}
    dynamic_seq = 0
    event = 0
    branch_counter = 0
    taken_counter = 0
    store_counter = 0
    last_store_dynamic_seq = -1
    last_store_line = 0
    begin = time.time()
    opener = lzma.open if trace_path.suffix == ".xz" else open

    with opener(trace_path, "rb") as handle:
        skip_records(handle, warmup_records)

        while event < n:
            record = handle.read(INPUT_INSTR_RECORD.size)
            if len(record) == 0:
                break
            if len(record) != INPUT_INSTR_RECORD.size:
                raise RuntimeError(
                    f"partial input_instr record at dynamic instruction {dynamic_seq}"
                )
            fields = INPUT_INSTR_RECORD.unpack(record)
            pc = int(fields[0])
            is_branch = int(fields[1])
            taken = int(fields[2])
            dst = tuple(int(x) for x in fields[3:5])
            src = tuple(int(x) for x in fields[5:9])
            dst_mem = tuple(int(x) for x in fields[9:11])
            src_mem = tuple(int(x) for x in fields[11:15])
            source_lines = [addr // CACHE_LINE_BYTES for addr in src_mem if addr]
            source_slots = [slot for slot, addr in enumerate(src_mem) if addr]
            is_load = int(bool(source_lines))
            is_store = int(any(dst_mem))

            parents = [register_origin[r] for r in src if r and r in register_origin]
            parent = max(parents, key=lambda value: value[0]) if parents else None

            # Oracle rows are ordered dynamic demand events. We only advance on a
            # precise (PC, line) match and never write a partially aligned sidecar.
            if pc == int(oracle_pc[event]) and source_lines:
                try:
                    source_slot = source_lines.index(int(oracle_line[event]))
                except ValueError:
                    source_slot = -1
                if source_slot >= 0:
                    src_regs[event] = np.asarray(src, dtype=np.uint8)
                    dst_regs[event] = np.asarray(dst, dtype=np.uint8)
                    dynamic_instruction_index[event] = dynamic_seq
                    matched_source_mem_slot[event] = source_slots[source_slot]

                    if parent is not None:
                        (
                            parent_seq,
                            parent_event,
                            parent_pc,
                            parent_line,
                            parent_depth,
                            parent_branch_count,
                            parent_taken_count,
                            parent_store_count,
                            parent_is_load,
                        ) = parent
                        gap = max(0, dynamic_seq - parent_seq)
                        dep_present[event] = 1
                        dep_parent_event[event] = parent_event
                        dep_parent_pc[event] = np.uint64(parent_pc)
                        dep_parent_line[event] = int(parent_line)
                        dep_parent_dynamic_gap[event] = gap
                        dep_chain_depth[event] = min(np.iinfo(np.uint16).max, parent_depth)
                        dep_parent_is_load[event] = parent_is_load
                        branches_since_parent[event] = max(0, branch_counter - parent_branch_count)
                        taken_since_parent[event] = max(0, taken_counter - parent_taken_count)
                        stores_since_parent[event] = max(0, store_counter - parent_store_count)

                    latest_store_line[event] = int(last_store_line)
                    latest_store_dynamic_gap[event] = (
                        max(0, dynamic_seq - last_store_dynamic_seq)
                        if last_store_dynamic_seq >= 0 else 0
                    )
                    current_event = event
                    event += 1
                else:
                    current_event = -1
            else:
                current_event = -1

            # The current instruction writes its destination after its source
            # dependencies are observed. A load becomes a new causal origin.
            if is_load:
                line_for_origin = int(source_lines[0])
                depth_for_origin = (int(parent[4]) + 1) if parent is not None else 1
                origin = (
                    dynamic_seq,
                    current_event,
                    pc,
                    line_for_origin,
                    depth_for_origin,
                    branch_counter,
                    taken_counter,
                    store_counter,
                    1,
                )
            elif parent is not None:
                origin = parent
            else:
                origin = None

            if origin is not None:
                for reg in dst:
                    if reg:
                        register_origin[reg] = origin

            if is_store:
                store_counter += 1
                store_addr = next((addr for addr in dst_mem if addr), 0)
                if store_addr:
                    last_store_line = int(store_addr // CACHE_LINE_BYTES)
                    last_store_dynamic_seq = dynamic_seq
            if is_branch:
                branch_counter += 1
                taken_counter += int(bool(taken))

            dynamic_seq += 1
            if progress_every and dynamic_seq % progress_every == 0:
                elapsed = time.time() - begin
                print(
                    f"[progress] dynamic={dynamic_seq:,} aligned={event:,}/{n:,} "
                    f"({event / n:.4%}) elapsed={elapsed/60:.1f}m",
                    flush=True,
                )

    alignment = event / n
    if alignment < min_alignment:
        raise RuntimeError(
            f"unsafe alignment: only {event:,}/{n:,} ({alignment:.4%}) oracle events matched; "
            f"minimum is {min_alignment:.2%}. No output written."
        )
    if event != n:
        raise RuntimeError(
            f"trace ended with incomplete but superficially high alignment {event:,}/{n:,}; "
            "sidecar intentionally not written."
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
    _save_npz(output_path, arrays)

    meta = {
        "version": "v3_9_dependency_sidecar",
        "trace": str(trace_path),
        "oracle": str(oracle_path),
        "output": str(output_path),
        "record_bytes": INPUT_INSTR_RECORD.size,
        "warmup_records_skipped": int(warmup_records),
        "oracle_events": int(n),
        "aligned_events": int(event),
        "alignment": float(alignment),
        "dynamic_instructions_scanned_after_warmup": int(dynamic_seq),
        "dependency_present_events": int(dep_present.sum()),
        "dependency_present_fraction": float(dep_present.mean()),
        "median_parent_dynamic_gap": float(np.median(dep_parent_dynamic_gap[dep_present.astype(bool)]))
            if dep_present.any() else 0.0,
        "source": "input_instr dynamic trace; no register or memory values are available",
    }
    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("[saved]", output_path)
    print("[saved]", meta_path)
    print(json.dumps(meta, indent=2))
    return meta


def choose_oracle(trace_stem: str, oracle: Optional[str], oracle_dir: Optional[str]) -> Path:
    if oracle:
        path = Path(oracle)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    if not oracle_dir:
        raise ValueError("pass --oracle or --oracle-dir")
    root = Path(oracle_dir)
    matches = sorted(
        path for path in root.rglob("*")
        if path.is_file() and trace_stem in path.name and path.suffix in {".csv", ".gz"}
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one oracle under {root} containing {trace_stem!r}; found {matches}"
        )
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, help="original .champsimtrace or .xz")
    parser.add_argument("--oracle", default=None, help="explicit no-prefetch oracle CSV/CSV.GZ")
    parser.add_argument("--oracle-dir", default=None, help="auto-find an oracle by trace stem")
    parser.add_argument("--trace-stem", default="605.mcf_s-994B")
    parser.add_argument("--output", required=True, help="output .npz path")
    parser.add_argument("--warmup-records", type=int, default=25_000_000)
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument("--min-alignment", type=float, default=0.9999)
    args = parser.parse_args()

    trace_path = Path(args.trace)
    if not trace_path.is_file():
        raise FileNotFoundError(trace_path)
    oracle_path = choose_oracle(args.trace_stem, args.oracle, args.oracle_dir)
    output_path = Path(args.output)
    if output_path.suffix != ".npz":
        raise ValueError("--output must end in .npz")
    build_sidecar(
        trace_path=trace_path,
        oracle_path=oracle_path,
        output_path=output_path,
        warmup_records=args.warmup_records,
        progress_every=args.progress_every,
        min_alignment=args.min_alignment,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
