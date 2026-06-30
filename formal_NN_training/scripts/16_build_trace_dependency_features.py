#!/usr/bin/env python3
# 16_build_trace_dependency_features.py
#
# Precompute register-dataflow ("dependency") features for one trace so the
# v3.9 notebook can load a small CSV instead of re-decompressing the full
# .champsimtrace.xz (~50M x 64B) inside Colab on every run.
#
# This reproduces, byte-for-byte and step-for-step, the alignment + last-writer
# logic already embedded in v3.8 `_enrich_dependency_context`, so the emitted
# columns drop straight into the model's feature dict.
#
# HARD CONSTRAINTS (sacramento runs CPython 3.6):
#   * NO `from __future__ import annotations`   (3.7+ only -> SyntaxError on 3.6)
#   * NO walrus `:=`, NO f-string `=` debug form (3.8+)
#   * NO pandas / numpy required (stdlib only: struct, lzma, csv, json, argparse)
#
# ChampSim input_instr record (confirmed by the notebook's own struct):
#   <Q  ip
#    B  is_branch
#    B  branch_taken
#   2B  destination_registers[2]
#   4B  source_registers[4]
#   2Q  destination_memory[2]
#   4Q  source_memory[4]        => 64 bytes total
#
# Alignment contract: events are matched as an in-order SUBSEQUENCE of the
# post-warmup load stream by (pc AND a source cache line). The script prints
# the alignment fraction; if it is below --min-alignment, DO NOT trust the CSV
# (fall back to in-notebook enrichment with the raw trace mounted).

import argparse
import csv
import json
import lzma
import os
import struct
import sys

RECORD = struct.Struct("<QBB2B4B2Q4Q")  # 64 bytes; matches v3.8 INPUT_INSTR_RECORD


def parse_int_maybe_hex(text):
    # Accept "123", "0x4a", "0X4A"; tolerate floats written as "123.0".
    s = (text or "").strip()
    if s == "":
        return 0
    try:
        if s.lower().startswith("0x"):
            return int(s, 16)
        if "." in s:
            return int(float(s))
        return int(s)
    except ValueError:
        return int(s, 0)


def read_event_targets(events_csv, pc_col, line_col, line_bytes):
    # Returns (pc_list, line_list) in file (chronological) order.
    pcs = []
    lines = []
    handle = open(events_csv, "r", newline="")
    try:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise RuntimeError("empty events csv: " + events_csv)
        name_to_idx = {}
        i = 0
        for col in header:
            name_to_idx[col.strip()] = i
            i += 1
        if pc_col not in name_to_idx:
            raise RuntimeError("pc column '%s' not in header: %s" % (pc_col, header))
        if line_col not in name_to_idx:
            raise RuntimeError("line column '%s' not in header: %s" % (line_col, header))
        pi = name_to_idx[pc_col]
        li = name_to_idx[line_col]
        for row in reader:
            if not row:
                continue
            pcs.append(parse_int_maybe_hex(row[pi]))
            lines.append(parse_int_maybe_hex(row[li]) // int(line_bytes))
    finally:
        handle.close()
    return pcs, lines


def open_trace(path):
    if path.endswith(".xz"):
        return lzma.open(path, "rb")
    return open(path, "rb")


def skip_warmup(handle, warmup_records):
    # Discard `warmup_records` records in large blocks; verify clean record framing.
    remaining = int(warmup_records)
    block_records = 1 << 20
    while remaining > 0:
        take = block_records if remaining > block_records else remaining
        block = handle.read(take * RECORD.size)
        got = len(block) // RECORD.size
        if got == 0:
            return False  # trace ended during warmup
        if len(block) % RECORD.size:
            raise RuntimeError("partial record while skipping warmup")
        remaining -= got
    return True


def build(trace, raw_dir, events_csv, out_csv, warmup, sim,
          line_bytes, pc_col, line_col, reg_none, min_alignment):
    raw_path = os.path.join(raw_dir, trace + ".champsimtrace.xz")
    if not os.path.isfile(raw_path):
        # also allow an uncompressed sibling
        alt = os.path.join(raw_dir, trace + ".champsimtrace")
        if os.path.isfile(alt):
            raw_path = alt
        else:
            raise RuntimeError("raw trace not found: " + raw_path)

    pc_target, line_target = read_event_targets(events_csv, pc_col, line_col, line_bytes)
    n_events = len(pc_target)
    if n_events == 0:
        raise RuntimeError("no events parsed from " + events_csv)

    # Output buffers (one row per event, in event order). Default = "no dependency".
    out = [None] * n_events
    for k in range(n_events):
        out[k] = dict(
            order=k, pc=pc_target[k], line=line_target[k],
            matched=0,
            src0=0, src1=0, src2=0, src3=0, dst0=0, dst1=0,
            dep_present=0, dep_parent_pc=0, dep_gap=-1, dep_parent_load=0,
            branch_is=0, branch_token=0,
        )

    handle = open_trace(raw_path)
    try:
        if not skip_warmup(handle, warmup):
            raise RuntimeError("trace shorter than warmup window")

        last_writer = {}     # reg_id -> (seq, pc, write_line, is_load)
        seq = 0
        next_event = 0
        max_records = int(sim) if int(sim) > 0 else (1 << 62)
        block_records = 1 << 20

        while next_event < n_events and seq < max_records:
            block = handle.read(block_records * RECORD.size)
            if len(block) < RECORD.size:
                break
            usable = (len(block) // RECORD.size) * RECORD.size
            for values in RECORD.iter_unpack(block[:usable]):
                if next_event >= n_events or seq >= max_records:
                    break
                pc = values[0]
                is_branch = values[1]
                taken = values[2]
                dst = (values[3], values[4])
                src = (values[5], values[6], values[7], values[8])
                src_mem = (values[11], values[12], values[13], values[14])

                mem_lines = []
                for addr in src_mem:
                    if addr:
                        mem_lines.append(addr // line_bytes)

                tgt_line = line_target[next_event]
                matched = (pc == pc_target[next_event]) and (tgt_line in mem_lines)

                if matched:
                    row = out[next_event]
                    row["matched"] = 1
                    row["src0"], row["src1"], row["src2"], row["src3"] = src
                    row["dst0"], row["dst1"] = dst
                    row["branch_is"] = 1 if is_branch else 0
                    row["branch_token"] = int(is_branch) * 2 + int(taken)
                    parents = []
                    for r in src:
                        if r and r != reg_none and r in last_writer:
                            parents.append(last_writer[r])
                    if parents:
                        # most-recent writer wins (largest seq)
                        parent = parents[0]
                        for cand in parents[1:]:
                            if cand[0] > parent[0]:
                                parent = cand
                        gap = seq - parent[0]
                        if gap < 0:
                            gap = 0
                        row["dep_present"] = 1
                        row["dep_parent_pc"] = parent[1]
                        row["dep_gap"] = gap
                        row["dep_parent_load"] = int(parent[3])
                    next_event += 1

                # update dataflow AFTER matching (a load can't be its own parent)
                write_line = mem_lines[0] if mem_lines else 0
                is_load = 1 if mem_lines else 0
                for reg in dst:
                    if reg and reg != reg_none:
                        last_writer[reg] = (seq, pc, write_line, is_load)
                seq += 1
    finally:
        handle.close()

    matched_total = 0
    parents_total = 0
    for row in out:
        matched_total += row["matched"]
        parents_total += row["dep_present"]
    alignment = matched_total / float(n_events)
    parent_rate = (parents_total / float(matched_total)) if matched_total else 0.0

    fieldnames = [
        "order", "pc", "line", "matched",
        "src0", "src1", "src2", "src3", "dst0", "dst1",
        "dep_present", "dep_parent_pc", "dep_gap", "dep_parent_load",
        "branch_is", "branch_token",
    ]
    wh = open(out_csv, "w", newline="")
    try:
        writer = csv.DictWriter(wh, fieldnames=fieldnames)
        writer.writeheader()
        for row in out:
            writer.writerow(row)
    finally:
        wh.close()

    meta = dict(
        trace=trace, raw_path=raw_path, events_csv=events_csv, out_csv=out_csv,
        record_bytes=RECORD.size, warmup_records=int(warmup), sim_records=int(sim),
        line_bytes=int(line_bytes), reg_none=int(reg_none),
        n_events=int(n_events), records_consumed=int(seq),
        events_matched=int(matched_total), alignment=round(alignment, 6),
        matched_with_parent_rate=round(parent_rate, 6),
        min_alignment=float(min_alignment),
        alignment_ok=bool(alignment >= float(min_alignment)),
        note=("USE: alignment passed; load dep features in v3.9."
              if alignment >= float(min_alignment) else
              "DO NOT USE: alignment below threshold; fall back to in-notebook "
              "enrichment with the raw trace mounted, or check pc/line columns."),
    )
    with open(out_csv + ".meta.json", "w") as mh:
        json.dump(meta, mh, indent=2)

    sys.stderr.write(
        "[dep] %s  events=%d  consumed=%d  matched=%d  alignment=%.4f  parent_rate=%.4f  ok=%s\n"
        % (trace, n_events, seq, matched_total, alignment, parent_rate, meta["alignment_ok"])
    )
    if not meta["alignment_ok"]:
        sys.stderr.write("[dep] WARNING: " + meta["note"] + "\n")
    return meta


def main():
    ap = argparse.ArgumentParser(description="Precompute register-dataflow features for one trace (3.6/stdlib only).")
    ap.add_argument("--trace", required=True, help="e.g. 605.mcf_s-994B")
    ap.add_argument("--raw-dir", default=os.path.expanduser("~/cache/traces"))
    ap.add_argument("--events-csv", required=True,
                    help="chronological event table, e.g. formal_NN_training/data/generated/lstm_events_605.mcf_s-994B.csv")
    ap.add_argument("--out-csv", required=True,
                    help="e.g. formal_NN_training/data/generated/dep_605.mcf_s-994B.csv")
    ap.add_argument("--pc-col", default="pc", help="PC column name in the events csv")
    ap.add_argument("--line-col", default="line", help="cache-line column name in the events csv")
    ap.add_argument("--warmup", type=int, default=25_000_000)
    ap.add_argument("--sim", type=int, default=25_000_000)
    ap.add_argument("--line-bytes", type=int, default=64)
    ap.add_argument("--reg-none", type=int, default=0, help="register id meaning 'no register'")
    ap.add_argument("--min-alignment", type=float, default=0.95)
    args = ap.parse_args()
    build(args.trace, args.raw_dir, args.events_csv, args.out_csv,
          args.warmup, args.sim, args.line_bytes, args.pc_col, args.line_col,
          args.reg_none, args.min_alignment)


if __name__ == "__main__":
    main()
