"""Check every measured load operand, in source instruction order, across all four replays."""

import argparse
import csv
import gzip
import json

import numpy as np

from access import IndexedStream, TRACE_DTYPE
from build import FULL, HERE, RAW
from run import CASES, POLICIES, external_cases


class OperandCursor:
    def __init__(self, folder):
        self.stream = IndexedStream(folder, "source_load")
        self.blocks = self.stream.chunks()
        self.keys = np.empty(0, dtype="<u8")
        self.used = 0

    def compare(self, expected):
        while len(expected):
            if not len(self.keys):
                self.keys = next(self.blocks)[:, 0]
            size = min(len(self.keys), len(expected))
            if not np.array_equal(expected[:size], self.keys[:size]):
                raise ValueError(f"Mismatch at or after load operand {self.used}")
            self.used += size
            self.keys = self.keys[size:]
            expected = expected[size:]


def visible_operands(folder, trace, origin, start, stop):
    with gzip.open(folder / "pages.csv.gz", "rt") as file:
        layout = sorted(
            (int(r["virtual_page"]), int(r["physical_page"]))
            for r in csv.DictReader(file)
        )
    vpn, ppn = np.asarray(layout, dtype="<u8").T
    # Four possible load operands per instruction, one bit per possible operand.
    seen = np.zeros((4 * (stop - start) + 7) // 8, dtype=np.uint8)
    events = 0
    for block in IndexedStream(folder, "arrival").chunks():
        keys = block[:, 0]
        index = (keys >> 2) - origin - 1
        if np.any(index < start) or np.any(index >= stop):
            raise ValueError("LLC arrival outside measured source interval")
        records = trace[index]
        va = np.take_along_axis(
            records["load_va"], (keys & 3).astype("intp")[:, None], axis=1
        )[:, 0]
        page = va >> 12
        slots = np.searchsorted(vpn, page)
        if np.any(slots == len(vpn)) or not np.array_equal(vpn[slots], page):
            raise ValueError("Unknown source page at offline verification")
        line = (ppn[slots] << 6) | ((va >> 6) & 63)
        if not np.array_equal(line, block[:, 2]) or not np.array_equal(
            records["pc"], block[:, 1]
        ):
            raise ValueError(
                "Admission disagrees with source operand or simulated placement"
            )
        relative = keys - 4 * (origin + start + 1)
        np.bitwise_or.at(seen, relative >> 3, (1 << (relative & 7)).astype("u1"))
        events += len(block)
    popcount = np.asarray([n.bit_count() for n in range(256)], dtype=np.uint8)
    unique = sum(int(popcount[part].sum()) for part in np.array_split(seen, 64))
    return events, unique


def check(case, policies):
    folders = [RAW / "runs" / (case + "__" + arm) for arm in policies]
    metas = [
        json.loads((folder / "complete.json").read_text())["config"]
        for folder in folders
    ]
    common = {key: value for key, value in metas[0].items() if key != "policy"}
    if any({k: v for k, v in m.items() if k != "policy"} != common for m in metas):
        raise ValueError("Policy arms use different source boundaries")
    trace = np.memmap(
        FULL.parents[1] / common["traces"][0], mode="r", dtype=TRACE_DTYPE
    )
    start = common["skip"] + common["warmup"]
    stop = start + common["roi"]
    origin = common.get("source_ordinal_base", 0)
    cursors = [OperandCursor(folder) for folder in folders]
    for i in range(start, stop, 1_000_000):
        records = trace[i : min(i + 1_000_000, stop)]
        instr, operand = np.nonzero(records["load_va"])
        expected = ((instr.astype("<u8") + i + origin + 1) << 2) | operand.astype("<u8")
        for cursor in cursors:
            cursor.compare(expected)
    if any(c.used != c.stream.rows for c in cursors):
        raise ValueError("Extra captured source loads")
    events, unique = visible_operands(folders[0], trace, origin, start, stop)
    return {
        "case": case,
        "policies": len(cursors),
        "measured_instructions": common["roi"],
        "first_source_ordinal": origin + start + 1,
        "last_source_ordinal": origin + stop,
        "exact_operand_keys_per_policy": cursors[0].used,
        f"{policies[0]}_llc_admissions_verified": events,
        f"{policies[0]}_unique_llc_operands": unique,
        "input_view": common["traces"][0],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument(
        "--external", action="store_true", help="Check the external Stride captures"
    )
    args = parser.parse_args()
    cases, policies = (
        (external_cases(), ("stride",)) if args.external else (CASES, POLICIES)
    )
    selected = args.cases or cases
    if not set(selected) <= set(cases):
        parser.error("Case outside the chosen panel")
    target = HERE / (
        "external_source_parity.csv" if args.external else "source_parity.csv"
    )
    if target.is_file():
        with target.open(newline="") as file:
            rows = {r["case"]: r for r in csv.DictReader(file)}
    else:
        rows = {}
    for case in selected:
        row = check(case, policies)
        rows[case] = row
        temp = target.with_suffix(".tmp")
        with temp.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows[k] for k in sorted(rows))
        temp.replace(target)
        print(
            case,
            row["exact_operand_keys_per_policy"],
            "source operands per registered arm",
            flush=True,
        )


if __name__ == "__main__":
    main()
