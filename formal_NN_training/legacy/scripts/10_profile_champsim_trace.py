#!/usr/bin/env python3
"""Profile dynamic information available in a ChampSim input_instr trace.

The standard ChampSim trace record contains dynamic PCs, branch flags,
register IDs, and memory addresses. It does not contain opcode bytes or
source-level symbols. Optional --binary mapping is therefore best-effort:
it maps observed PCs to static disassembly only when the supplied executable
uses the same PC address space as the trace.
"""
import argparse
import csv
import hashlib
import json
import lzma
import re
import struct
import subprocess
from collections import Counter
from pathlib import Path

# input_instr: IP, branch flag, branch-taken flag, 2 destination registers,
# 4 source registers, 2 destination-memory operands, 4 source-memory operands.
RECORD = struct.Struct("<QBB2B4B2Q4Q")
LINE_BYTES = 64
PAGE_LINES = 64


def as_int(value):
    return int(value, 0)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fraction(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def trace_stem(path):
    name = path.name
    suffix = ".champsimtrace.xz"
    if name.endswith(suffix):
        return name[:-len(suffix)]
    if name.endswith(".xz"):
        return name[:-3]
    return path.stem


def parse_disassembly(binary, objdump):
    """Return {address: instruction_text}; empty on a nonfatal tool failure."""
    if binary is None:
        return {}
    command = [objdump, "-d", "--no-show-raw-insn", str(binary)]
    try:
        text = subprocess.check_output(command, stderr=subprocess.STDOUT,
                                       universal_newlines=True)
    except (OSError, subprocess.CalledProcessError):
        return {}

    result = {}
    pattern = re.compile(r"^\s*([0-9a-fA-F]+):\s*(.+?)\s*$")
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            result[int(match.group(1), 16)] = match.group(2)
    return result


def addr2line_lookup(binary, addr2line, pcs):
    if binary is None:
        return {}
    result = {}
    for pc in pcs:
        try:
            text = subprocess.check_output(
                [addr2line, "-e", str(binary), "-f", "-C", "0x%x" % pc],
                stderr=subprocess.STDOUT, universal_newlines=True,
            ).strip().splitlines()
        except (OSError, subprocess.CalledProcessError):
            break
        if text:
            result[pc] = {
                "symbol": text[0],
                "source": text[1] if len(text) > 1 else "",
            }
    return result


def counter_rows(total_instructions, instruction_counts, load_counts, store_counts,
                 branch_counts, taken_counts, disassembly, source_map, limit):
    rows = []
    for pc, count in instruction_counts.most_common(limit):
        source = source_map.get(pc, {})
        rows.append({
            "pc": "0x%x" % pc,
            "dynamic_instructions": count,
            "instruction_share": fraction(count, total_instructions),
            "dynamic_load_instructions": load_counts.get(pc, 0),
            "dynamic_store_instructions": store_counts.get(pc, 0),
            "dynamic_branch_instructions": branch_counts.get(pc, 0),
            "dynamic_taken_branches": taken_counts.get(pc, 0),
            "assembly": disassembly.get(pc, ""),
            "symbol": source.get("symbol", ""),
            "source": source.get("source", ""),
        })
    return rows


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--skip-records", type=as_int, default=25_000_000)
    parser.add_argument("--max-records", type=as_int, default=25_000_000,
                        help="0 reads through the end after skip.")
    parser.add_argument("--top-pcs", type=int, default=100)
    parser.add_argument("--top-registers", type=int, default=64)
    parser.add_argument("--binary", type=Path, default=None,
                        help="Optional original executable for best-effort asm/symbol mapping.")
    parser.add_argument("--objdump", default="objdump")
    parser.add_argument("--addr2line", default="addr2line")
    args = parser.parse_args()

    if not args.trace.is_file():
        raise FileNotFoundError(str(args.trace))
    if args.binary is not None and not args.binary.is_file():
        raise FileNotFoundError(str(args.binary))
    if args.skip_records < 0 or args.max_records < 0:
        raise ValueError("skip/max records must be nonnegative")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    opener = lzma.open if args.trace.suffix == ".xz" else open
    counts = Counter()
    pc_instructions = Counter()
    pc_loads = Counter()
    pc_stores = Counter()
    pc_branches = Counter()
    pc_taken = Counter()
    source_regs = Counter()
    destination_regs = Counter()
    deltas = Counter()
    delta_abs_bucket = Counter()
    pages = Counter()
    first_pc = None
    last_pc = None
    prior_load_line = None

    with opener(args.trace, "rb") as handle:
        remaining = args.skip_records
        while remaining:
            records = min(remaining, 1 << 20)
            block = handle.read(records * RECORD.size)
            if not block:
                break
            if len(block) % RECORD.size:
                raise RuntimeError("partial trace record while skipping")
            remaining -= len(block) // RECORD.size

        while args.max_records == 0 or counts["instructions"] < args.max_records:
            raw = handle.read(RECORD.size)
            if not raw:
                break
            if len(raw) != RECORD.size:
                raise RuntimeError("partial trace record")
            values = RECORD.unpack(raw)
            pc, is_branch, branch_taken = values[0], values[1], values[2]
            destination_registers = values[3:5]
            source_register_values = values[5:9]
            destination_memory = values[9:11]
            source_memory = values[11:15]

            counts["instructions"] += 1
            pc_instructions[pc] += 1
            first_pc = pc if first_pc is None else first_pc
            last_pc = pc

            for reg in source_register_values:
                if reg:
                    source_regs[reg] += 1
            for reg in destination_registers:
                if reg:
                    destination_regs[reg] += 1

            if is_branch:
                counts["branch_instructions"] += 1
                pc_branches[pc] += 1
                if branch_taken:
                    counts["taken_branches"] += 1
                    pc_taken[pc] += 1

            load_lines = [address // LINE_BYTES for address in source_memory if address]
            store_lines = [address // LINE_BYTES for address in destination_memory if address]
            if load_lines:
                counts["load_instructions"] += 1
                counts["load_operands"] += len(load_lines)
                pc_loads[pc] += 1
                for line in load_lines:
                    pages[line // PAGE_LINES] += 1
                current_line = load_lines[0]
                if prior_load_line is not None:
                    delta = current_line - prior_load_line
                    deltas[delta] += 1
                    magnitude = abs(delta)
                    if magnitude == 0:
                        bucket = "0"
                    elif magnitude == 1:
                        bucket = "1"
                    elif magnitude <= 4:
                        bucket = "2_4"
                    elif magnitude <= 16:
                        bucket = "5_16"
                    elif magnitude <= 64:
                        bucket = "17_64"
                    elif magnitude <= 256:
                        bucket = "65_256"
                    else:
                        bucket = "gt_256"
                    delta_abs_bucket[bucket] += 1
                prior_load_line = current_line
            if store_lines:
                counts["store_instructions"] += 1
                counts["store_operands"] += len(store_lines)
                pc_stores[pc] += 1
            if load_lines or store_lines:
                counts["memory_instructions"] += 1

    disassembly = parse_disassembly(args.binary, args.objdump)
    top_pc_values = [pc for pc, _ in pc_instructions.most_common(args.top_pcs)]
    source_map = addr2line_lookup(args.binary, args.addr2line, top_pc_values)
    top_pcs = counter_rows(
        counts["instructions"], pc_instructions, pc_loads, pc_stores,
        pc_branches, pc_taken, disassembly, source_map, args.top_pcs,
    )
    delta_total = sum(deltas.values())
    top_deltas = [
        {"line_delta": delta, "count": count,
         "share_of_observed_load_deltas": fraction(count, delta_total)}
        for delta, count in deltas.most_common(100)
    ]
    top_pages = [
        {"page": page, "load_operands": count,
         "share_of_load_operands": fraction(count, counts["load_operands"])}
        for page, count in pages.most_common(100)
    ]
    register_rows = []
    for direction, table in (("source", source_regs), ("destination", destination_regs)):
        total = sum(table.values())
        for reg, count in table.most_common(args.top_registers):
            register_rows.append({
                "direction": direction,
                "register_id": reg,
                "dynamic_uses": count,
                "share_within_direction": fraction(count, total),
            })

    stem = trace_stem(args.trace)
    summary = {
        "trace": str(args.trace),
        "trace_sha256": sha256(args.trace),
        "compressed_trace_bytes": args.trace.stat().st_size,
        "record_format": "ChampSim input_instr",
        "record_size_bytes": RECORD.size,
        "profile_window_skip_records": args.skip_records,
        "profile_window_records_read": counts["instructions"],
        "profile_window_ended_at_record": args.skip_records + counts["instructions"],
        "full_trace_counted": bool(args.max_records == 0),
        "unique_pcs_in_window": len(pc_instructions),
        "unique_load_pages_observed": len(pages),
        "first_pc_in_window": "" if first_pc is None else "0x%x" % first_pc,
        "last_pc_in_window": "" if last_pc is None else "0x%x" % last_pc,
        "branch_fraction": fraction(counts["branch_instructions"], counts["instructions"]),
        "taken_fraction_of_branches": fraction(counts["taken_branches"], counts["branch_instructions"]),
        "load_instruction_fraction": fraction(counts["load_instructions"], counts["instructions"]),
        "store_instruction_fraction": fraction(counts["store_instructions"], counts["instructions"]),
        "memory_instruction_fraction": fraction(counts["memory_instructions"], counts["instructions"]),
        "delta_abs_buckets": dict(delta_abs_bucket),
        "assembly_available_from_trace": False,
        "assembly_mapping_attempted": args.binary is not None,
        "assembly_mapping_note": (
            "Trace records do not contain opcode bytes. Optional binary mapping is best-effort "
            "and is valid only when binary static addresses match the trace PC address space."
        ),
    }
    summary.update(dict(counts))

    (args.out_dir / (stem + ".trace_profile.json")).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_csv(
        args.out_dir / (stem + ".top_pcs.csv"),
        top_pcs,
        ["pc", "dynamic_instructions", "instruction_share",
         "dynamic_load_instructions", "dynamic_store_instructions",
         "dynamic_branch_instructions", "dynamic_taken_branches",
         "assembly", "symbol", "source"],
    )
    write_csv(
        args.out_dir / (stem + ".top_load_line_deltas.csv"),
        top_deltas,
        ["line_delta", "count", "share_of_observed_load_deltas"],
    )
    write_csv(
        args.out_dir / (stem + ".top_load_pages.csv"),
        top_pages,
        ["page", "load_operands", "share_of_load_operands"],
    )
    write_csv(
        args.out_dir / (stem + ".top_registers.csv"),
        register_rows,
        ["direction", "register_id", "dynamic_uses", "share_within_direction"],
    )
    print("[write] %s" % (args.out_dir / (stem + ".trace_profile.json")))


if __name__ == "__main__":
    main()
