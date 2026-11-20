"""Replay identical MCF instructions with the original and nonuniform physical allocations."""

import csv
import gzip

import numpy as np

from access import IndexedStream, source_records

from build import BUILD, FULL, HERE, RAW, build
from run import baseline


def execute(permuted):
    name = "mcf17_layout_reversed" if permuted else "mcf17_layout_original"
    return baseline.run(
        name,
        "stride",
        [FULL / "raw/traces/mcf17_discovery.champsimtrace"],
        [0],
        50_000_000,
        2_000_000,
    )


def pages(folder):
    with gzip.open(folder / "pages.csv.gz", "rt") as file:
        return {
            int(r["virtual_page"]): int(r["physical_page"])
            for r in csv.DictReader(file)
        }


def main():
    alternate = RAW / "layout_build/pythia"
    build(alternate, permute=True)
    baseline.RAW = RAW
    baseline.BIN = BUILD / "bin/champsim"
    original = execute(False)
    baseline.BIN = alternate / "bin/champsim"
    reversed_layout = execute(True)
    first, second = pages(original), pages(reversed_layout)
    if first.keys() != second.keys() or len(set(second.values())) != len(second):
        raise ValueError("Different virtual demand or aliasing in page map")
    changed = 0
    for va, ppn in first.items():
        offset = (ppn - 1) % 64
        expected = ppn - offset + int(format(offset, "06b")[::-1], 2)
        if second[va] != expected:
            raise ValueError("Incorrect reversed placement")
        changed += expected != ppn
    source_a = IndexedStream(original, "source_load")
    source_b = IndexedStream(reversed_layout, "source_load")
    if source_a.rows != source_b.rows:
        raise ValueError("Different source-load populations across layouts")
    matched = 0
    previous_key = None
    for left, right in zip(source_a.chunks(), source_b.chunks(), strict=True):
        keys = left[:, 0]
        if not np.array_equal(keys, right[:, 0]):
            raise ValueError("Different source operand identities across layouts")
        if np.any(keys[1:] <= keys[:-1]) or (
            previous_key is not None and keys[0] <= previous_key
        ):
            raise ValueError("Source operands are not in original instruction order")
        previous_key = keys[-1]
        # Both maps were populated from exactly this trace. Validate every referenced operand.
        records, va = source_records(original, FULL.parents[1], keys)
        if np.any(records["pc"] == 0) or any(
            int(v) not in first for v in np.unique(va >> 12)
        ):
            raise ValueError("Unmapped source load or missing instruction PC")
        matched += len(keys)
    if matched != source_a.rows:
        raise ValueError("Incomplete source stream")
    rows = []
    for folder in (original, reversed_layout):
        phases = baseline.rows(folder / "phases.csv")
        core = baseline.rows(folder / "cores.csv")[0]
        rows.append(
            dict(
                case=folder.name,
                measured=core["instructions"],
                cycles=core["cycles"],
                source_loads=sum(int(r["trace_loads"]) for r in phases),
                arrivals=sum(int(r["llc_ingress"]) for r in phases),
                matched_operand_keys=matched,
                matched_virtual_pages=len(first),
                changed_physical_pages=changed,
            )
        )
    if rows[0]["source_loads"] != rows[1]["source_loads"]:
        raise ValueError("Different source-load counts across layouts")
    with (HERE / "placement.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
