#!/usr/bin/env python3
"""Normalize a live stride/SPP event log into causal demand and candidate streams.

Every PF request is attached to the nearest demand callback with the same
trigger PC and base cache line (falling back to base line only).  Matching is
transport bookkeeping only: the models never receive PC or event identifiers.
The normalizer fails closed when a request cannot be assigned unambiguously.
"""
import argparse
import bisect
import csv
import gzip
from collections import defaultdict
from pathlib import Path


TRACE = "623.xalancbmk_s-700B"
POLICIES = {"stride", "spp"}
LOG2_BLOCK_SIZE = 6


def as_int(value):
    value = str(value).strip()
    return int(value, 16) if value.lower().startswith("0x") else int(float(value))


def nearest(entries, event_id):
    """Return the nearest (event_id, demand_idx) entry, preferring the past."""
    pos = bisect.bisect_right(entries, (event_id, 1 << 62))
    choices = []
    if pos:
        choices.append(entries[pos - 1])
    if pos < len(entries):
        choices.append(entries[pos])
    if not choices:
        return None
    return min(choices, key=lambda item: (abs(item[0] - event_id), item[0] > event_id))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--policy", required=True, choices=sorted(POLICIES))
    parser.add_argument("--stream-out", required=True, type=Path)
    parser.add_argument("--candidate-out", required=True, type=Path)
    parser.add_argument("--max-event-distance", type=int, default=256)
    args = parser.parse_args()

    opener = gzip.open if str(args.events).endswith(".gz") else open
    demands = []
    pf_rows = []
    occurrences = defaultdict(int)
    exact = defaultdict(list)
    by_line = defaultdict(list)

    with opener(args.events, "rt", newline="") as source:
        reader = csv.DictReader(source)
        required = {
            "event", "event_id", "cache", "op", "cycle", "ip", "line",
            "base_addr", "pf_line", "accepted", "duplicate",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError("event log missing columns: {}".format(sorted(missing)))
        for row in reader:
            if row["cache"] != "L2C":
                continue
            event_id = as_int(row["event_id"])
            if row["event"] == "DEMAND" and row["op"] == "read":
                pc = as_int(row["ip"])
                line = as_int(row["line"])
                pair = (pc, line)
                occ = occurrences[pair]
                occurrences[pair] += 1
                demand_idx = len(demands)
                demand = {
                    "trace": TRACE,
                    "demand_idx": demand_idx,
                    "cycle": as_int(row["cycle"]),
                    "pc": pc,
                    "line": line,
                    "pc_line_occ": occ,
                    "_event_id": event_id,
                }
                demands.append(demand)
                exact[pair].append((event_id, demand_idx))
                by_line[line].append((event_id, demand_idx))
            elif row["event"] == "PF":
                pf_rows.append({
                    "event_id": event_id,
                    "pc": as_int(row["ip"]),
                    "base_line": as_int(row["base_addr"]) >> LOG2_BLOCK_SIZE,
                    "pf_line": as_int(row["pf_line"]),
                    "accepted": as_int(row["accepted"]),
                    "duplicate": as_int(row["duplicate"]),
                })

    if not demands:
        raise RuntimeError("no post-warmup L2 demand rows were found")
    if not pf_rows:
        raise RuntimeError("{} emitted no logged PF requests".format(args.policy))

    attached = defaultdict(list)
    fallback_matches = 0
    for pf in pf_rows:
        candidates = exact.get((pf["pc"], pf["base_line"]), ())
        match = nearest(candidates, pf["event_id"]) if candidates else None
        match_mode = "pc_line"
        if match is None:
            match = nearest(by_line.get(pf["base_line"], ()), pf["event_id"])
            match_mode = "line_only"
            fallback_matches += 1
        if match is None:
            raise RuntimeError(
                "cannot attach PF event {} base line {}".format(
                    pf["event_id"], pf["base_line"]
                )
            )
        distance = abs(match[0] - pf["event_id"])
        if distance > args.max_event_distance:
            raise RuntimeError(
                "PF event {} is {} events from nearest demand trigger".format(
                    pf["event_id"], distance
                )
            )
        demand = demands[match[1]]
        if demand["line"] != pf["base_line"]:
            raise RuntimeError("candidate base-line transport mismatch")
        attached[match[1]].append({
            **pf,
            "match_mode": match_mode,
            "event_distance": distance,
        })

    args.stream_out.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.stream_out, "wt", newline="") as target:
        fields = ["trace", "demand_idx", "cycle", "pc", "line", "pc_line_occ"]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for demand in demands:
            writer.writerow({key: demand[key] for key in fields})

    candidate_count = 0
    max_candidates = 0
    with gzip.open(args.candidate_out, "wt", newline="") as target:
        fields = [
            "trace", "policy", "demand_idx", "pc", "line", "pc_line_occ",
            "candidate_rank", "pf_line", "accepted", "duplicate",
            "pf_event_id", "event_distance", "match_mode",
        ]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for demand in demands:
            rows = sorted(
                attached.get(demand["demand_idx"], ()),
                key=lambda item: item["event_id"],
            )
            max_candidates = max(max_candidates, len(rows))
            for rank, pf in enumerate(rows, 1):
                writer.writerow({
                    "trace": TRACE,
                    "policy": args.policy,
                    "demand_idx": demand["demand_idx"],
                    "pc": demand["pc"],
                    "line": demand["line"],
                    "pc_line_occ": demand["pc_line_occ"],
                    "candidate_rank": rank,
                    "pf_line": pf["pf_line"],
                    "accepted": pf["accepted"],
                    "duplicate": pf["duplicate"],
                    "pf_event_id": pf["event_id"],
                    "event_distance": pf["event_distance"],
                    "match_mode": pf["match_mode"],
                })
                candidate_count += 1

    print(
        "[ok] policy={} demands={} candidates={} max_per_demand={} "
        "line_only_transport_matches={} stream={} candidates_file={}".format(
            args.policy,
            len(demands),
            candidate_count,
            max_candidates,
            fallback_matches,
            args.stream_out,
            args.candidate_out,
        )
    )


if __name__ == "__main__":
    main()
