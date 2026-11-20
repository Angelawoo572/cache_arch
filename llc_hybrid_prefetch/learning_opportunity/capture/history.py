"""Export a causal, bounded ingress history without future targets or baseline labels."""

import argparse
import csv
from pathlib import Path
import sys

from access import IndexedStream

PRESSURE = ("llc_mshr", "llc_rq", "llc_pq", "dram_rq", "dram_wq", "llc_wq")


def history(folder, anchor, length):
    if length < 1:
        raise ValueError("History length must be positive")
    begin = max(0, anchor - length + 1)
    data = IndexedStream(folder, "arrival").read(begin, anchor + 1)
    states = IndexedStream(folder, "ingress_state").read(begin, anchor + 1)
    seen_pc = {}
    prev_line = prev_cycle = prev_pc = None
    for index, (event, state) in enumerate(zip(data, states), start=begin):
        _, pc, line, cycle = map(int, event)
        order, pressure = map(int, state)
        row = dict(
            event=index,
            order=order,
            pc=pc,
            physical_line=line,
            page=line >> 6,
            offset=line & 63,
            cycle=cycle,
            previous_pc=prev_pc,
            previous_delta=line - prev_line if prev_line is not None else None,
            previous_same_pc_delta=line - seen_pc[pc] if pc in seen_pc else None,
            cycle_gap=cycle - prev_cycle if prev_cycle is not None else None,
        )
        for byte, name in enumerate(PRESSURE):
            row[name] = (pressure >> (byte * 8)) & 255
        yield row
        seen_pc[pc] = line
        prev_line, prev_cycle, prev_pc = line, cycle, pc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument(
        "anchor", type=int, help="Zero-based index in this run's LLC arrival order"
    )
    parser.add_argument("--length", type=int, default=64)
    args = parser.parse_args()
    rows = list(history(args.folder, args.anchor, args.length))
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
