"""Validate complete indexed streams against the unchanged native simulator and source records."""

import argparse
from collections import Counter
import csv
import gzip
import io
import json
from pathlib import Path

import numpy as np

from access import COLUMNS, IndexedStream, source_records
from build import FULL

CACHE = FULL.parents[1]


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def match(reference, capture):
    if json.loads((reference / "config.json").read_text()) != json.loads(
        (capture / "config.json").read_text()
    ):
        raise ValueError("Different native capture and reference configurations")
    reference_arrivals = reference / "arrival.u64le.gz"
    if reference_arrivals.is_file():
        compare_gzip(reference_arrivals, capture / "arrival.u64le.gz")
    for name in ("cores.csv", "phases.csv", "stats.csv"):
        if read_csv(reference / name) != read_csv(capture / name):
            raise ValueError(f"Simulator parity failure: {capture.name}/{name}")
    with (
        gzip.open(reference / "pages.csv.gz", "rt") as left,
        gzip.open(capture / "pages.csv.gz", "rt") as right,
    ):
        if sorted(left) != sorted(right):
            raise ValueError("Physical maps differ")
    compare_gzip(
        reference / "outcomes_core0.u16le.gz", capture / "outcomes_core0.u16le.gz"
    )
    prior_arrivals = FULL / "raw/continuous/runs" / capture.name / "arrival.u64le.gz"
    if prior_arrivals.is_file():
        compare_gzip(prior_arrivals, capture / "arrival.u64le.gz")


def compare_gzip(one, two):
    # Keep small compressed references in memory. Alternating reads between two cold
    # files on the archive disk can otherwise seek once per comparison block.
    if one.stat().st_size <= 64 << 20:
        reference = gzip.GzipFile(fileobj=io.BytesIO(one.read_bytes()))
    else:
        reference = gzip.open(one)
    with reference as left, gzip.open(two) as right:
        while True:
            chunk = left.read(1 << 22)
            if chunk != right.read(1 << 22):
                raise ValueError(f"Native stream differs from reference: {two}")
            if not chunk:
                return


def histogram(target, codes):
    values, counts = np.unique(codes, return_counts=True)
    target.update({int(v): int(n) for v, n in zip(values, counts)})


def audit(folder, reference):
    match(reference, folder)
    config = json.loads((folder / "complete.json").read_text())["config"]
    stats = {r["key"]: int(r["value"]) for r in read_csv(folder / "stats.csv")}
    phases = read_csv(folder / "phases.csv")
    symbols = {r["value"]: int(r["id"]) for r in read_csv(folder / "symbols.csv")}
    streams = {name: IndexedStream(folder, name) for name in COLUMNS}
    for name, phase_count in (
        ("arrival", "llc_ingress"),
        ("source_load", "trace_loads"),
        ("ingress_state", "llc_ingress"),
    ):
        if streams[name].rows != sum(int(q[phase_count]) for q in phases):
            raise ValueError(f"Lost {name} rows")
    pages = {}
    with gzip.open(folder / "pages.csv.gz", "rt") as file:
        for row in csv.DictReader(file):
            pages[int(row["virtual_page"])] = int(row["physical_page"])
    prior_clock = 0
    samples = []
    for block in streams["arrival"].chunks():
        if np.any(block[1:, 3] < block[:-1, 3]) or block[0, 3] < prior_clock:
            raise ValueError("Nonmonotone arrivals")
        prior_clock = block[-1, 3]
        # Check each block edge and middle against the authoritative 64-byte source trace.
        samples.append(block[[0, len(block) // 2, -1]])
    # Read the arrival file sequentially, then resolve all source samples in one trace mapping.
    join_samples = sum(len(block) for block in samples)
    if samples:
        chosen = np.concatenate(samples)
        records, va = source_records(folder, CACHE, chosen[:, 0])
        phys = np.array(
            [pages[int(addr) >> 12] * 64 + (int(addr) // 64) % 64 for addr in va]
        )
        if not np.array_equal(phys, chosen[:, 2]) or not np.array_equal(
            records["pc"], chosen[:, 1]
        ):
            raise ValueError(
                "LLC arrival disagrees with source operand or physical placement"
            )
    counts = Counter()
    for block in streams["cache"].chunks():
        histogram(counts, block[:, 4])
    lookup = symbols.get("lookup", 0)
    for lev, prefix in ((2, "l2"), (3, "llc")):
        for result, code in (("hits", 2), ("misses", 1)):
            if counts[(lookup << 16) | (code << 8) | lev] != sum(
                int(q[f"{prefix}_tag_{result}"]) for q in phases
            ):
                raise ValueError("Incomplete cache lookup stream")
    events = Counter()
    for block in streams["offer"].chunks():
        histogram(events, block[:, 5] >> 32)
    for label in ("accepted", "timely", "late", "evicted", "redundant", "unresolved"):
        key = "prefetch_accepted" if label == "accepted" else "study_pf_" + label
        if events[symbols.get(label, -1)] != stats.get(key, 0):
            raise ValueError(f"Lifecycle disagrees at {label}")
    # Inflate every additional member to verify format and CRC, including the final partial member.
    for name in ("ingress_state", "source_load", "head_wait", "prefetch_victim"):
        if sum(len(b) for b in streams[name].chunks()) != streams[name].rows:
            raise ValueError(f"Incomplete {name} stream")
    return dict(
        case=config["case"],
        policy=config["policy"],
        warmup=config["warmup"],
        measured=config["roi"],
        origin=config.get("source_ordinal_base", 0),
        **{name + "_rows": s.rows for name, s in streams.items()},
        **{
            f"{level}_tag_{outcome}": counts[(lookup << 16) | (code << 8) | number]
            for number, level in ((1, "l1"), (2, "l2"), (3, "llc"))
            for code, outcome in ((1, "misses"), (2, "hits"))
        },
        proposed=events[symbols.get("proposed", -1)],
        accepted=events[symbols.get("accepted", -1)],
        rejected=events[symbols.get("rejected", -1)],
        timely=events[symbols.get("timely", -1)],
        late=events[symbols.get("late", -1)],
        compressed_stream_bytes=sum(s.path.stat().st_size for s in streams.values()),
        source_operand_checks=join_samples,
        baseline_counters_and_masks="match",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--references", type=Path, default=FULL / "raw/native/runs")
    args = parser.parse_args()
    for folder in args.folders:
        result = audit(folder, args.references / folder.name)
        path = folder / "capture_validation.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(path)
        print(
            "verified",
            result["case"],
            result["policy"],
            result["arrival_rows"],
            flush=True,
        )


if __name__ == "__main__":
    main()
