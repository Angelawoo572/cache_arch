#!/usr/bin/env python3
"""Match NoPF-hit/Stride-miss loads to the preceding actual LLC eviction or refill."""

import argparse
import csv
import gzip
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def open_csv(directory, name, binary=False):
    path = directory / (name + ".csv")
    if path.is_file():
        return path.open("rb" if binary else "r")
    return gzip.open(path.with_suffix(".csv.gz"), "rb" if binary else "rt")


def nopf_hits(directory):
    hits = {}
    with open_csv(directory, "loads", True) as stream:
        header = stream.readline().strip().split(b",")
        required = [
            b"cpu",
            b"instr_id",
            b"operand",
            b"pc",
            b"virtual_line",
            b"physical_line",
            b"level",
            b"stage",
            b"cycle",
            b"hit",
        ]
        if header[: len(required)] != required:
            raise ValueError("unexpected load header")
        for line in stream:
            if b",LLC,lookup," not in line:
                continue
            r = line.rstrip().split(b",")
            if r[9] == b"1":
                key = tuple(map(int, r[:3]))
                if key in hits:
                    raise ValueError(f"repeated LLC lookup {key}")
                hits[key] = tuple(map(int, r[3:6]))
    return hits


def summary(directory):
    with open_csv(directory, "cores") as f:
        core = list(csv.DictReader(f))
    with open_csv(directory, "stats") as f:
        stats = {r["key"]: int(r["value"]) for r in csv.DictReader(f)}
    return core, stats


def calculate(nopf, stride, probe):
    a, sa = summary(stride)
    b, sb = summary(probe)
    performance_keys = (
        "dram_demand_bytes",
        "dram_prefetch_bytes",
        "dram_write_bytes",
        "prefetch_accepted",
        "pressure_cycles",
        "llc_mshr_cycle_sum",
        "dram_rq_cycle_sum",
    )
    if a != b or any(sa[k] != sb[k] for k in performance_keys):
        raise ValueError(
            f"{probe.name} pollution probe does not reproduce canonical counters and per-core cycles"
        )
    hits = nopf_hits(nopf)
    missed = []
    with open_csv(probe, "lookups") as f:
        for r in csv.DictReader(f):
            if r["hit"] != "0":
                continue
            key = tuple(int(r[k]) for k in ("cpu", "instr_id", "operand"))
            if key not in hits:
                continue
            identity = tuple(int(r[k]) for k in ("pc", "virtual_line", "line"))
            if identity != hits[key]:
                raise ValueError(
                    f"demand identity or physical placement changed for {key}"
                )
            missed.append((int(r["cycle"]), key, int(r["line"])))
    needed = {line for _, _, line in missed}
    counts = {int(c["cpu"]): Counter() for c in a}
    unique = set()
    last = {}
    with open_csv(probe, "fills") as f:
        events = iter(csv.DictReader(f))
        event = next(events, None)
        # The shared LLC handles fills and writebacks before read lookups within a cycle.
        serial = 0
        for cycle, key, line in missed:
            while event is not None and int(event["cycle"]) <= cycle:
                serial += 1
                new = int(event["line"])
                if event["victim_valid"] == "1":
                    victim = int(event["victim_line"])
                    if victim in needed:
                        last[victim] = (
                            event["kind"],
                            int(event["cpu"]),
                            int(event["cycle"]),
                            serial,
                        )
                if new in needed:
                    last[new] = (
                        "refill",
                        int(event["cpu"]),
                        int(event["cycle"]),
                        serial,
                    )
                event = next(events, None)
            c = key[0]
            counter = counts[c]
            counter["hit_to_miss"] += 1
            if line not in last:
                counter["no_line_change_observed"] += 1
                continue
            kind, source, previous_cycle, episode = last[line]
            if kind == "prefetch":
                counter["prefetch_was_last_evictor"] += 1
                if (c, episode) not in unique:
                    unique.add((c, episode))
                    counter["distinct_prefetch_evictions"] += 1
                counter[
                    "cross_core_evictions" if source != c else "same_core_evictions"
                ] += 1
                counter["evict_to_lookup_cycles_sum"] += cycle - previous_cycle
            elif kind == "refill":
                counter["last_change_was_refill"] += 1
            else:
                counter["demand_or_write_was_last_evictor"] += 1
    for c, k in counts.items():
        n = k["prefetch_was_last_evictor"]
        total = k["hit_to_miss"]
        if total != sum(
            k[x]
            for x in (
                "prefetch_was_last_evictor",
                "no_line_change_observed",
                "last_change_was_refill",
                "demand_or_write_was_last_evictor",
            )
        ):
            raise ValueError("nonexclusive classifications")
        yield {
            "case": probe.name.removesuffix("-stride"),
            "cpu": c,
            "nopf_hit_stride_miss": total,
            "last_evictor_prefetch": n,
            "distinct_prefetch_evictions": k["distinct_prefetch_evictions"],
            "same_core": k["same_core_evictions"],
            "other_core": k["cross_core_evictions"],
            "last_evictor_other": k["demand_or_write_was_last_evictor"],
            "no_line_change_observed": k["no_line_change_observed"],
            "last_change_refill": k["last_change_was_refill"],
            "mean_cycles_since_prefetch_eviction": round(
                k["evict_to_lookup_cycles_sum"] / n, 2
            )
            if n
            else "",
            "probe_matches_canonical_cycles_and_traffic": 1,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "raw/qualified")
    parser.add_argument("--probes", type=Path, default=ROOT / "raw/pollution")
    parser.add_argument("--output", type=Path, default=ROOT / "results/pollution.csv")
    args = parser.parse_args()
    dirs = sorted(args.probes.glob("*-stride"))
    if len(dirs) != 14:
        raise ValueError(f"expected all 14 SPEC Stride cases; found {len(dirs)}")
    results = []
    for probe in dirs:
        if not (probe / "complete.json").exists():
            raise ValueError(f"incomplete {probe.name}")
        nopf = args.runs / (probe.name.removesuffix("-stride") + "-nopf")
        results.extend(calculate(nopf, args.runs / probe.name, probe))
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0])
        w.writeheader()
        w.writerows(results)
    print(f"Validated {len(dirs)} SPEC pairs, {len(results)} per-core samples.")


if __name__ == "__main__":
    main()
