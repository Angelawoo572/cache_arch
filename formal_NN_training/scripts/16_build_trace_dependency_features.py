#!/usr/bin/env python3
"""Build causal dynamic-dependency features for one ChampSim input_instr trace.

Sacramento's system Python is older and does not provide the Python/Pandas stack
used by Colab. This utility therefore uses only the Python standard library.

It never changes the trace, oracle labels, or ChampSim replay protocol. It reads
an existing no-prefetch oracle and the original .champsimtrace(.xz), aligns each
oracle demand event to a matching dynamic instruction by ordered (PC, cache-line)
keys, and writes a compressed CSV sidecar keyed by demand_idx.
"""

import argparse
import csv
import gzip
import json
import lzma
import os
import struct
import time
from pathlib import Path

# IP, branch, taken, 2 destination registers, 4 source registers,
# 2 destination-memory operands, 4 source-memory operands.
INPUT_INSTR_RECORD = struct.Struct("<QBB2B4B2Q4Q")
CACHE_LINE_BYTES = 64

OUTPUT_FIELDS = [
    "demand_idx", "oracle_pc", "oracle_line",
    "src_reg_0", "src_reg_1", "src_reg_2", "src_reg_3",
    "dst_reg_0", "dst_reg_1",
    "dep_present", "dep_parent_event", "dep_parent_pc", "dep_parent_line",
    "dep_parent_dynamic_gap", "dep_chain_depth", "dep_parent_is_load",
    "branches_since_parent", "taken_since_parent", "stores_since_parent",
    "latest_store_line", "latest_store_dynamic_gap",
    "dynamic_instruction_index", "matched_source_mem_slot",
]


def parse_int_maybe_hex(value, default=0):
    """Parse decimal/hex CSV values without pandas or numpy."""
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "null"):
        return default
    try:
        if text.lower().startswith("0x"):
            return int(text, 16)
        return int(text)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return default


def first_existing_col(fieldnames, candidates):
    names = set(fieldnames or [])
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def open_text(path, mode):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(str(path), mode, newline="")
    return open(str(path), mode, newline="", encoding="utf-8")


def load_oracle(path):
    """Return ordered [(pc, line), ...] with strict contiguous demand_idx validation."""
    events = []
    with open_text(path, "rt") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        pc_col = first_existing_col(fieldnames, ("pc", "ip", "PC", "IP"))
        line_col = first_existing_col(fieldnames, ("line", "addr_line", "address_line", "block"))
        idx_col = first_existing_col(fieldnames, ("demand_idx", "event_idx", "idx", "index"))
        if pc_col is None or line_col is None:
            raise ValueError("{}: need PC and line columns; found {}".format(path, fieldnames))
        for expected_idx, row in enumerate(reader):
            if idx_col is not None:
                actual_idx = parse_int_maybe_hex(row.get(idx_col), -1)
                if actual_idx != expected_idx:
                    raise ValueError(
                        "{}: non-contiguous {} at row {}: got {}".format(
                            path, idx_col, expected_idx, actual_idx
                        )
                    )
            events.append((
                parse_int_maybe_hex(row.get(pc_col)),
                parse_int_maybe_hex(row.get(line_col)),
            ))
    if not events:
        raise ValueError("{}: oracle is empty".format(path))
    return events


def skip_records(handle, n_records):
    remaining = int(n_records)
    record_bytes = INPUT_INSTR_RECORD.size
    while remaining:
        take = min(remaining, 1 << 20)
        block = handle.read(take * record_bytes)
        got = len(block) // record_bytes
        if got == 0:
            raise RuntimeError(
                "trace ended while skipping warmup at {}/{} records".format(
                    n_records - remaining, n_records
                )
            )
        if len(block) % record_bytes:
            raise RuntimeError("trace contains a partial input_instr record during warmup")
        remaining -= got


def open_trace(path):
    path = Path(path)
    return lzma.open(str(path), "rb") if path.suffix == ".xz" else open(str(path), "rb")


def temp_path_for(output_path):
    return output_path.with_name(output_path.name + ".partial")


def write_row(writer, event_idx, oracle_pc, oracle_line, src, dst, parent,
              dynamic_seq, branch_count, taken_count, store_count,
              latest_store_line, last_store_seq, source_mem_slot):
    row = {
        "demand_idx": event_idx,
        "oracle_pc": oracle_pc,
        "oracle_line": oracle_line,
        "src_reg_0": src[0], "src_reg_1": src[1],
        "src_reg_2": src[2], "src_reg_3": src[3],
        "dst_reg_0": dst[0], "dst_reg_1": dst[1],
        "dep_present": 0,
        "dep_parent_event": -1,
        "dep_parent_pc": 0,
        "dep_parent_line": 0,
        "dep_parent_dynamic_gap": 0,
        "dep_chain_depth": 0,
        "dep_parent_is_load": 0,
        "branches_since_parent": 0,
        "taken_since_parent": 0,
        "stores_since_parent": 0,
        "latest_store_line": latest_store_line,
        "latest_store_dynamic_gap": max(0, dynamic_seq - last_store_seq) if last_store_seq >= 0 else 0,
        "dynamic_instruction_index": dynamic_seq,
        "matched_source_mem_slot": source_mem_slot,
    }
    parent_gap = None
    if parent is not None:
        # Dynamic sequence, matched oracle event or -1, PC, line, depth,
        # branch counter, taken counter, store counter, is-load.
        p_seq, p_event, p_pc, p_line, p_depth, p_branches, p_taken, p_stores, p_is_load = parent
        row.update({
            "dep_present": 1,
            "dep_parent_event": p_event,
            "dep_parent_pc": p_pc,
            "dep_parent_line": p_line,
            "dep_parent_dynamic_gap": max(0, dynamic_seq - p_seq),
            "dep_chain_depth": p_depth,
            "dep_parent_is_load": p_is_load,
            "branches_since_parent": max(0, branch_count - p_branches),
            "taken_since_parent": max(0, taken_count - p_taken),
            "stores_since_parent": max(0, store_count - p_stores),
        })
        parent_gap = row["dep_parent_dynamic_gap"]
    writer.writerow(row)
    return parent_gap


def build_sidecar(trace_path, oracle_path, output_path, warmup_records,
                  progress_every, min_alignment):
    oracle_events = load_oracle(oracle_path)
    n_events = len(oracle_events)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = temp_path_for(output_path)
    partial_meta_path = temp_path_for(output_path.with_suffix(".json"))
    for stale in (partial_path, partial_meta_path):
        if stale.exists():
            stale.unlink()

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
    latest_store_line = 0
    dependency_present_events = 0
    parent_gaps = []
    started = time.time()

    try:
        with open_trace(trace_path) as trace_handle:
            skip_records(trace_handle, warmup_records)
            with gzip.open(str(partial_path), "wt", newline="") as sidecar_handle:
                writer = csv.DictWriter(sidecar_handle, fieldnames=OUTPUT_FIELDS)
                writer.writeheader()

                while event < n_events:
                    raw = trace_handle.read(INPUT_INSTR_RECORD.size)
                    if not raw:
                        break
                    if len(raw) != INPUT_INSTR_RECORD.size:
                        raise RuntimeError(
                            "partial input_instr record at dynamic instruction {}".format(dynamic_seq)
                        )
                    fields = INPUT_INSTR_RECORD.unpack(raw)
                    pc = int(fields[0])
                    is_branch = int(fields[1])
                    taken = int(fields[2])
                    dst = tuple(int(value) for value in fields[3:5])
                    src = tuple(int(value) for value in fields[5:9])
                    dst_mem = tuple(int(value) for value in fields[9:11])
                    src_mem = tuple(int(value) for value in fields[11:15])
                    source_pairs = [
                        (slot, int(addr) // CACHE_LINE_BYTES)
                        for slot, addr in enumerate(src_mem) if addr
                    ]
                    source_lines = [line for _, line in source_pairs]
                    is_load = int(bool(source_pairs))
                    is_store = int(any(dst_mem))

                    parents = [register_origin[reg] for reg in src if reg and reg in register_origin]
                    parent = max(parents, key=lambda value: value[0]) if parents else None
                    current_event = -1

                    expected_pc, expected_line = oracle_events[event]
                    if pc == expected_pc and source_pairs:
                        matching_slots = [slot for slot, line in source_pairs if line == expected_line]
                        if matching_slots:
                            parent_gap = write_row(
                                writer, event, expected_pc, expected_line, src, dst, parent,
                                dynamic_seq, branch_count, taken_count, store_count,
                                latest_store_line, last_store_seq, matching_slots[0],
                            )
                            if parent_gap is not None:
                                dependency_present_events += 1
                                parent_gaps.append(parent_gap)
                            current_event = event
                            event += 1

                    # Read all source origins before assigning current destinations.
                    if is_load:
                        origin_line = source_lines[0]
                        origin_depth = (int(parent[4]) + 1) if parent is not None else 1
                        origin = (
                            dynamic_seq, current_event, pc, origin_line, origin_depth,
                            branch_count, taken_count, store_count, 1,
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
                        store_count += 1
                        store_addr = next((int(addr) for addr in dst_mem if addr), 0)
                        if store_addr:
                            latest_store_line = store_addr // CACHE_LINE_BYTES
                            last_store_seq = dynamic_seq
                    if is_branch:
                        branch_count += 1
                        taken_count += int(bool(taken))

                    dynamic_seq += 1
                    if progress_every and dynamic_seq % progress_every == 0:
                        elapsed = time.time() - started
                        print(
                            "[progress] dynamic={:,} aligned={:,}/{:,} ({:.4%}) elapsed={:.1f}m".format(
                                dynamic_seq, event, n_events, float(event) / n_events, elapsed / 60.0
                            ),
                            flush=True,
                        )

        alignment = float(event) / n_events
        if alignment < min_alignment or event != n_events:
            raise RuntimeError(
                "unsafe/incomplete alignment: {}/{} ({:.4%}); no output written".format(
                    event, n_events, alignment
                )
            )

        meta = {
            "version": "v3_9_dependency_sidecar_csv_v1",
            "trace": str(trace_path),
            "oracle": str(oracle_path),
            "output": str(output_path),
            "record_bytes": INPUT_INSTR_RECORD.size,
            "warmup_records_skipped": int(warmup_records),
            "oracle_events": int(n_events),
            "aligned_events": int(event),
            "alignment": float(alignment),
            "dynamic_instructions_scanned_after_warmup": int(dynamic_seq),
            "dependency_present_events": int(dependency_present_events),
            "dependency_present_fraction": float(dependency_present_events) / n_events,
            "median_parent_dynamic_gap": sorted(parent_gaps)[len(parent_gaps) // 2] if parent_gaps else 0,
            "source": "input_instr dynamic trace; no register or memory values are available",
            "format": "gzip CSV with one row per oracle demand_idx",
        }
        with open(str(partial_meta_path), "w", encoding="utf-8") as meta_handle:
            json.dump(meta, meta_handle, indent=2, sort_keys=True)
            meta_handle.write("\n")
        os.replace(str(partial_path), str(output_path))
        os.replace(str(partial_meta_path), str(output_path.with_suffix(".json")))
        print("[saved] {}".format(output_path))
        print("[saved] {}".format(output_path.with_suffix(".json")))
        print(json.dumps(meta, indent=2, sort_keys=True))
        return meta
    except Exception:
        for partial in (partial_path, partial_meta_path):
            if partial.exists():
                partial.unlink()
        raise


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
        raise RuntimeError(
            "expected exactly one oracle under {} containing {!r}; found {}".format(
                root, trace_stem, matches
            )
        )
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, help="original .champsimtrace or .xz")
    parser.add_argument("--oracle", default=None, help="explicit no-prefetch oracle CSV/CSV.GZ")
    parser.add_argument("--oracle-dir", default=None, help="auto-find oracle by trace stem")
    parser.add_argument("--trace-stem", default="605.mcf_s-994B")
    parser.add_argument("--output", required=True, help="output .csv.gz")
    parser.add_argument("--warmup-records", type=int, default=25000000)
    parser.add_argument("--progress-every", type=int, default=1000000)
    parser.add_argument("--min-alignment", type=float, default=1.0)
    args = parser.parse_args()

    trace_path = Path(args.trace)
    if not trace_path.is_file():
        raise FileNotFoundError(str(trace_path))
    oracle_path = choose_oracle(args.trace_stem, args.oracle, args.oracle_dir)
    output_path = Path(args.output)
    if not output_path.name.endswith(".csv.gz"):
        raise ValueError("--output must end in .csv.gz")

    build_sidecar(
        trace_path, oracle_path, output_path,
        args.warmup_records, args.progress_every, args.min_alignment,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
