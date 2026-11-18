"""Generate specified dynamic microtraces. Addresses and ground-truth segment starts are visible here, not to the predictor."""
import argparse
import csv
import gzip
import json
from bisect import bisect_right
from collections import Counter
import struct
from matrix import SYNTHETIC, SYN_ROI, SYN_WARMUP, TRACE_DIR

RECORD = struct.Struct("<Q8B6Q")
DATA_BASE = 0x20000000
PC = 0x400000
EVENTS_PER_LOAD = 4
N_INSTRUCTIONS = SYN_ROI + SYN_WARMUP + 64_000


def line(case, index):
    if case == "steady":
        return index
    if case == "nested":
        row, column = divmod(index, 16)
        return row * 32 + column
    if case in ("nested_samepage", "nested_crosspage", "paced_nested_samepage"):
        pair, within = divmod(index, 32)
        row, column = divmod(within, 16)
        return pair * 128 + row * 32 + column + (32 if case == "nested_crosspage" else 0)
    if case == "nested_variable":
        block, within = divmod(index, 64)
        starts = (0, 8, 28, 40)
        row = bisect_right(starts, within) - 1
        return (4 * block + row) * 32 + within - starts[row]
    if case == "interleaved":
        stream, position = index % 2, index // 2
        return stream * (1 << 22) + position
    if case == "phase":
        block, column = divmod(index, 64)
        return 128 * block + (column if column < 32 else 32 + 3 * (column - 32))
    block, position = divmod(index, 8)
    return block * 32 + (0, 7, 2, 19, 5, 11, 29, 3)[position]


def boundary(case, index):
    if index == 0:
        return None
    if case in ("nested_samepage", "nested_crosspage", "paced_nested_samepage"):
        if index % 32 == 16:
            return "paired_second_row"
        if index % 32 == 0:
            return "pair_start"
    if case == "nested" and index % 16 == 0:
        return "row_start"
    if case == "nested_variable" and index % 64 in (0, 8, 28, 40):
        return "variable_row_start"
    if case == "phase" and index % 32 == 0:
        return "phase_start"
    if case == "recurring" and index % 8 == 0:
        return "motif_start"
    return None


def record(ip, dest=0, src=0, src2=0, memory=0, branch=False):
    return RECORD.pack(ip, int(branch), int(branch), dest, 0, src, src2, 0, 0,
                       0, 0, memory, 0, 0, 0)


def write(case):
    name = "synthetic-" + case
    paced = case == "paced_nested_samepage"
    # After consuming the preceding load, compute 256 dependent results.
    # Register 1 feeds the next effective-address computation.
    gap = 256 if paced else 0
    interval = EVENTS_PER_LOAD + gap
    nloads = (N_INSTRUCTIONS + interval - 1) // interval
    target = TRACE_DIR / (name + ".champsimtrace")
    partial = target.with_suffix(".part")
    address_step = record(PC, dest=1, src=1)
    consume = record(PC + 8, dest=1 if paced else 11, src=10, src2=11)
    chain = b"".join(record(PC + 12 + 4 * j, dest=1, src=1) for j in range(gap))
    # Standard x86 ChampSim trace convention for a conditional loop branch:
    # read flags (25) and IP (26); write IP, branch taken to the first instruction.
    branch = record(PC + 12 + 4 * gap, dest=26, src=25, src2=26, branch=True)
    counts = Counter()
    annotation_path = TRACE_DIR / (name + ".boundaries.csv.gz")
    with partial.open("wb") as output, gzip.open(str(annotation_path), "wt") as annotation:
        table = csv.writer(annotation)
        table.writerow(["instruction_id", "load_ordinal", "pattern", "measurement",
                        "cross_4k", "previous_vline", "vline"])
        chunk = bytearray()
        for n in range(nloads):
            kind = boundary(case, n)
            if kind:
                previous, current = line(case, n - 1), line(case, n)
                phase = ("warmup" if interval*n + 2 <= SYN_WARMUP else
                         "roi" if interval*n + 2 <= SYN_WARMUP + SYN_ROI else "tail")
                cross_page = int(previous // 64 != current // 64)
                table.writerow([interval*n + 2, n, kind, phase, cross_page,
                                DATA_BASE // 64 + previous, DATA_BASE // 64 + current])
                counts[(kind, phase, cross_page)] += 1
            chunk.extend(address_step)
            chunk.extend(record(PC + 4, dest=10, src=1,
                                memory=DATA_BASE + 64 * line(case, n)))
            chunk.extend(consume)
            chunk.extend(chain)
            chunk.extend(branch)
            if len(chunk) >= 1 << 20:
                output.write(chunk)
                chunk.clear()
        output.write(chunk)
    assert partial.stat().st_size == nloads * interval * RECORD.size
    partial.replace(target)
    target.with_suffix(".source.json").write_text(json.dumps({
        "type": "generated instruction trace (mechanism control)",
        "case": case, "records": nloads * interval, "record_bytes": RECORD.size,
        "load_pc": PC + 4, "events_per_load": interval, "dependent_compute_instructions": gap,
        "source": "experiments/synthetic.py",
        "line_bytes": 64, "data_virtual_base": DATA_BASE,
        "load_instruction_id": f"{interval}*n + 2 for zero-based load ordinal n",
        "stride_units": "cache lines", "boundaries_csv": annotation_path.name,
        "boundaries": [dict(pattern=k, measurement=m, cross_4k=bool(c), count=v)
                        for (k, m, c), v in sorted(counts.items())],
    }, indent=2) + "\n")


if __name__ == "__main__":
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=SYNTHETIC)
    args = parser.parse_args()
    for k in args.case or SYNTHETIC:
        write(k)
