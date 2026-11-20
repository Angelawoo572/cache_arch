"""Compute exact LLC ingress spacing and request-queue occupancy at decision time."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY / "capture"))
from access import IndexedStream


def quantile(hist, fraction):
    if not hist:
        return ""
    rank = max(1, int(np.ceil(fraction * sum(hist.values()))))
    total = 0
    for value, count in sorted(hist.items()):
        total += count
        if total >= rank:
            return value


def tally(hist, data):
    values, counts = np.unique(data, return_counts=True)
    hist.update(dict(zip(map(int, values), map(int, counts))))


def measure(directory):
    json.loads((directory / "capture_validation.json").read_text())
    config = json.loads((directory / "complete.json").read_text())["config"]
    with (directory / "cores.csv").open() as f:
        cycles = int(next(csv.DictReader(f))["cycles"])
    with (directory / "stats.csv").open() as f:
        stats = {r["key"]: int(r["value"]) for r in csv.DictReader(f)}
    arrivals = IndexedStream(directory, "arrival")
    states = IndexedStream(directory, "ingress_state")
    if states.rows != arrivals.rows:
        raise ValueError("Pressure and arrivals are not one-to-one")
    gap_hist = Counter()
    names = ("llc_mshr", "llc_rq", "llc_pq", "dram_rq", "dram_wq", "llc_wq")
    queues = [Counter() for name in names]
    previous = None
    for a, state in zip(arrivals.chunks(), states.chunks(), strict=True):
        times = a[:, 3].astype("int64")
        gaps = np.diff(times) if previous is None else np.diff(times, prepend=previous)
        if len(gaps) and min(gaps) < 0:
            raise ValueError("LLC arrival clock went backwards")
        tally(gap_hist, gaps)
        previous = times[-1]
        for index, hist in enumerate(queues):
            tally(hist, (state[:, 1] >> (8 * index)) & 255)
    n = sum(gap_hist.values())
    if n != arrivals.rows - 1:
        raise ValueError("Lost an interarrival gap at a block boundary")
    row = dict(
        case=config["case"],
        policy=config["policy"],
        measured=config["roi"],
        ingress=arrivals.rows,
        cycles=cycles,
        triggers_per_kcycle=round(1000 * arrivals.rows / cycles, 4),
        accepted_per_kcycle=round(1000 * stats["prefetch_accepted"] / cycles, 4),
    )
    for percent in (10, 50, 90, 99):
        row[f"gap_p{percent}"] = quantile(gap_hist, percent / 100)
    for initiation in (1, 4, 8, 16):
        row[f"fraction_gap_lt_{initiation}"] = (
            round(sum(v for k, v in gap_hist.items() if k < initiation) / n, 6)
            if n
            else ""
        )
    for name, hist in zip(names, queues):
        if sum(hist.values()) != arrivals.rows:
            raise ValueError("Missing pressure snapshots")
        row[name + "_at_trigger_mean"] = round(
            sum(k * v for k, v in hist.items()) / arrivals.rows, 4
        )
        row[name + "_at_trigger_p90"] = quantile(hist, 0.90)
        row[name + "_at_trigger_max"] = max(hist)
    return row


def summarize(source, dest, expected, complete_only):
    with source.open() as f:
        captures = [r for r in csv.DictReader(f) if int(r["measured"]) == 200_000_000]
    if not complete_only and len(captures) != expected:
        raise ValueError(f"Expected {expected} complete runs in {source.name}")
    records = {}
    if dest.exists():
        with dest.open() as f:
            records = {(r["case"], r["policy"]): r for r in csv.DictReader(f)}
    for r in captures:
        k = r["case"], r["policy"]
        if k not in records:
            records[k] = measure(STUDY.parents[1] / r["raw_directory"])
    if records:
        with dest.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=next(iter(records.values())).keys())
            writer.writeheader()
            writer.writerows(records[k] for k in sorted(records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete-only", action="store_true")
    args = parser.parse_args()
    for source, dest, expected in (
        ("captures.csv", "at_ingress.csv", 36),
        ("external_captures.csv", "external_at_ingress.csv", 11),
    ):
        summarize(STUDY / "capture" / source, HERE / dest, expected, args.complete_only)


if __name__ == "__main__":
    main()
