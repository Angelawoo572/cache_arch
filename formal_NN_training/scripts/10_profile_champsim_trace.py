#!/usr/bin/env python3
"""Profile the dynamic fields stored in a standard ChampSim trace window.

ChampSim input_instr records PC, branch flags, register IDs, and memory
addresses. They do not contain opcode bytes, so this tool cannot recover
assembly without the original executable and a matching address map.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import lzma
import struct
from collections import Counter
from pathlib import Path

# input_instr: ip, branch, taken, 2 destination registers, 4 source registers,
# 2 destination-memory operands, 4 source-memory operands.
RECORD = struct.Struct("<QBB2B4B2Q4Q")
LINE_BYTES = 64


def as_int(value: str) -> int:
    return int(value, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--skip-records", type=as_int, default=25_000_000)
    parser.add_argument("--max-records", type=as_int, default=25_000_000,
                        help="0 reads through the end after skip; this can be expensive")
    parser.add_argument("--top-pcs", type=int, default=100)
    args = parser.parse_args()

    if not args.trace.is_file():
        raise FileNotFoundError(args.trace)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    opener = lzma.open if args.trace.suffix == ".xz" else open
    with opener(args.trace, "rb") as f:
        remaining = args.skip_records
        while remaining:
            records = min(remaining, 1 << 20)
            block = f.read(records * RECORD.size)
            if not block:
                break
            if len(block) % RECORD.size:
                raise RuntimeError("partial trace record while skipping")
            remaining -= len(block) // RECORD.size

        counts = Counter()
        pcs = Counter()
        deltas = Counter()
        prior_line = None
        first_pc = None
        last_pc = None
        read = 0

        while args.max_records == 0 or read < args.max_records:
            record = f.read(RECORD.size)
            if not record:
                break
            if len(record) != RECORD.size:
                raise RuntimeError("partial trace record")
            values = RECORD.unpack(record)
            pc, is_branch, branch_taken = values[0], values[1], values[2]
            destination_memory = values[9:11]
            source_memory = values[11:15]

            counts["instructions"] += 1
            pcs[pc] += 1
            first_pc = pc if first_pc is None else first_pc
            last_pc = pc
            if is_branch:
                counts["branch_instructions"] += 1
                counts["taken_branches"] += int(bool(branch_taken))

            loads = sum(address != 0 for address in source_memory)
            stores = sum(address != 0 for address in destination_memory)
            if loads:
                counts["load_instructions"] += 1
                counts["load_operands"] += loads
            if stores:
                counts["store_instructions"] += 1
                counts["store_operands"] += stores
            if loads or stores:
                counts["memory_instructions"] += 1

            line = next((address // LINE_BYTES for address in source_memory if address), None)
            if line is not None:
                if prior_line is not None:
                    deltas[line - prior_line] += 1
                prior_line = line
            read += 1

    total = counts["instructions"]
    fraction = lambda value: value / total if total else 0.0
    top_pcs = [
        {"pc": f"0x{pc:x}", "dynamic_instructions": count, "share": fraction(count)}
        for pc, count in pcs.most_common(args.top_pcs)
    ]
    delta_total = sum(deltas.values())
    top_deltas = [
        {"line_delta": delta, "count": count,
         "share_of_observed_load_deltas": count / delta_total if delta_total else 0.0}
        for delta, count in deltas.most_common(100)
    ]

    stem = args.trace.name.removesuffix(".champsimtrace.xz")
    summary = {
        "trace": str(args.trace),
        "trace_sha256": sha256(args.trace),
        "record_format": "ChampSim input_instr",
        "record_size_bytes": RECORD.size,
        "profile_window_skip_records": args.skip_records,
        "profile_window_records_read": total,
        "profile_window_ended_at_record": args.skip_records + total,
        "unique_pcs_in_window": len(pcs),
        "first_pc_in_window": None if first_pc is None else f"0x{first_pc:x}",
        "last_pc_in_window": None if last_pc is None else f"0x{last_pc:x}",
        **dict(counts),
        "branch_fraction": fraction(counts["branch_instructions"]),
        "taken_fraction_of_branches": (
            counts["taken_branches"] / counts["branch_instructions"]
            if counts["branch_instructions"] else 0.0),
        "load_instruction_fraction": fraction(counts["load_instructions"]),
        "store_instruction_fraction": fraction(counts["store_instructions"]),
        "memory_instruction_fraction": fraction(counts["memory_instructions"]),
        "assembly_available_from_trace": False,
        "assembly_note": "The trace has PC/branch/register/memory fields but no opcode bytes. Use the original executable plus a matching address map for assembly.",
        "top_pcs": top_pcs,
        "top_load_line_deltas": top_deltas,
    }
    (args.out_dir / f"{stem}.trace_profile.json").write_text(json.dumps(summary, indent=2) + "\n")

    for name, rows, fields in [
        ("top_pcs", top_pcs, ["pc", "dynamic_instructions", "share"]),
        ("top_load_line_deltas", top_deltas,
         ["line_delta", "count", "share_of_observed_load_deltas"]),
    ]:
        with (args.out_dir / f"{stem}.{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print("[write]", args.out_dir / f"{stem}.trace_profile.json")


if __name__ == "__main__":
    main()
