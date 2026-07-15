#!/usr/bin/env python3
"""Normalize the independent live 623 stride log with explicit trigger IDs.

The v5 event logger writes each completed L2 demand before invoking the normal
prefetcher and places that exact demand event ID on every synchronous PF row.
This normalizer never guesses from a future demand or from base_addr alone.
It fails closed on stale schemas, noncontiguous events, callback interleaving,
or any trigger identity mismatch.
"""
import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path


TRACE = "623.xalancbmk_s-700B"
POLICY = "stride"
LOG2_BLOCK_SIZE = 6
LOGGER_SCHEMA = "623_causal_trigger_v5"
ATTACHMENT_MODE = "explicit_trigger_event_id"


def as_int(value):
    value = str(value).strip()
    return int(value, 16) if value.lower().startswith("0x") else int(float(value))


def fail(message, event_id=None):
    suffix = "" if event_id is None else " at event {}".format(event_id)
    raise RuntimeError(message + suffix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--policy", required=True, choices=[POLICY])
    parser.add_argument("--stream-out", required=True, type=Path)
    parser.add_argument("--candidate-out", required=True, type=Path)
    args = parser.parse_args()

    opener = gzip.open if str(args.events).endswith(".gz") else open
    demands = []
    demand_by_event_id = {}
    attached = defaultdict(list)
    occurrences = defaultdict(int)
    last_event_id = -1
    latest_demand_event_id = None
    pf_count = 0
    max_event_distance = 0

    with opener(args.events, "rt", newline="") as source:
        reader = csv.DictReader(source)
        required = {
            "event", "event_id", "cpu", "cycle", "cache", "op", "ip",
            "line", "base_addr", "pf_line", "fill_level", "accepted", "duplicate",
            "trigger_event_id", "trigger_cpu", "trigger_ip", "trigger_line",
            "logger_schema",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                "event log has a stale/incomplete schema; missing {}. "
                "Rebuild collection with RESET_PATCH=1 and FORCE=1.".format(
                    sorted(missing)
                )
            )

        for row_number, row in enumerate(reader, 2):
            if row["logger_schema"] != LOGGER_SCHEMA:
                raise RuntimeError(
                    "event log row {} schema {!r}; expected {!r}".format(
                        row_number, row["logger_schema"], LOGGER_SCHEMA
                    )
                )
            if row["cache"] != "L2C":
                fail("non-L2C row in 623 L2 event log")

            event_id = as_int(row["event_id"])
            if event_id != last_event_id + 1:
                fail(
                    "event IDs are not contiguous (previous {})".format(
                        last_event_id
                    ),
                    event_id,
                )
            last_event_id = event_id
            cpu = as_int(row["cpu"])
            cycle = as_int(row["cycle"])
            event = row["event"]

            if event == "DEMAND":
                if row["op"] != "read":
                    fail("non-read demand row", event_id)
                pc = as_int(row["ip"])
                line = as_int(row["line"])
                trigger_id = as_int(row["trigger_event_id"])
                trigger_identity = (
                    as_int(row["trigger_cpu"]),
                    as_int(row["trigger_ip"]),
                    as_int(row["trigger_line"]),
                )
                if trigger_id != event_id or trigger_identity != (cpu, pc, line):
                    fail("demand self-trigger identity mismatch", event_id)
                pair = (pc, line)
                occ = occurrences[pair]
                occurrences[pair] += 1
                demand_idx = len(demands)
                demand = {
                    "trace": TRACE,
                    "demand_idx": demand_idx,
                    "cycle": cycle,
                    "pc": pc,
                    "line": line,
                    "pc_line_occ": occ,
                    "logger_schema": LOGGER_SCHEMA,
                    "_event_id": event_id,
                    "_cpu": cpu,
                }
                demands.append(demand)
                demand_by_event_id[event_id] = demand
                latest_demand_event_id = event_id
                continue

            if event != "PF":
                fail("unknown event kind {!r}".format(event), event_id)
            if latest_demand_event_id is None:
                fail("PF row precedes the first demand", event_id)

            trigger_id = as_int(row["trigger_event_id"])
            if trigger_id != latest_demand_event_id:
                fail(
                    "PF trigger is not the immediately active demand callback "
                    "({} != {})".format(trigger_id, latest_demand_event_id),
                    event_id,
                )
            demand = demand_by_event_id.get(trigger_id)
            if demand is None or trigger_id >= event_id:
                fail("PF references a missing/future trigger", event_id)

            trigger_identity = (
                as_int(row["trigger_cpu"]),
                as_int(row["trigger_ip"]),
                as_int(row["trigger_line"]),
            )
            expected_identity = (demand["_cpu"], demand["pc"], demand["line"])
            if trigger_identity != expected_identity:
                fail("PF trigger columns disagree with demand row", event_id)
            if cpu != demand["_cpu"] or cycle != demand["cycle"]:
                fail("PF is not synchronous with its demand callback", event_id)
            if as_int(row["ip"]) != demand["pc"]:
                fail("PF transport PC differs from trigger PC", event_id)
            if (as_int(row["base_addr"]) >> LOG2_BLOCK_SIZE) != demand["line"]:
                fail("PF base line differs from explicit trigger line", event_id)

            accepted = as_int(row["accepted"])
            duplicate = as_int(row["duplicate"])
            fill_level = as_int(row["fill_level"])
            if accepted not in (0, 1) or duplicate not in (0, 1):
                fail("PF accepted/duplicate field is not Boolean", event_id)
            if duplicate and not accepted:
                fail("rejected PF cannot be marked duplicate", event_id)
            if fill_level != 2:
                fail("stride candidate did not target FILL_L2", event_id)

            distance = event_id - trigger_id
            if distance <= 0:
                fail("PF event does not follow its explicit trigger", event_id)
            max_event_distance = max(max_event_distance, distance)
            attached[demand["demand_idx"]].append({
                "event_id": event_id,
                "trigger_event_id": trigger_id,
                "pf_line": as_int(row["pf_line"]),
                "fill_level": fill_level,
                "accepted": accepted,
                "duplicate": duplicate,
                "event_distance": distance,
            })
            pf_count += 1

    if not demands:
        raise RuntimeError("no post-warmup completed L2 demand callbacks were found")
    if not pf_count:
        raise RuntimeError("{} emitted no logged PF requests".format(args.policy))

    args.stream_out.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.stream_out, "wt", newline="") as target:
        fields = [
            "trace", "demand_idx", "cycle", "pc", "line", "pc_line_occ",
            "logger_schema",
        ]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for demand in demands:
            writer.writerow({key: demand[key] for key in fields})

    candidate_count = 0
    max_candidates = 0
    with gzip.open(args.candidate_out, "wt", newline="") as target:
        fields = [
            "trace", "policy", "demand_idx", "pc", "line", "pc_line_occ",
            "candidate_rank", "pf_line", "fill_level", "accepted", "duplicate",
            "trigger_event_id", "pf_event_id", "event_distance", "match_mode",
            "logger_schema",
        ]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for demand in demands:
            rows = attached.get(demand["demand_idx"], ())
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
                    "fill_level": pf["fill_level"],
                    "accepted": pf["accepted"],
                    "duplicate": pf["duplicate"],
                    "trigger_event_id": pf["trigger_event_id"],
                    "pf_event_id": pf["event_id"],
                    "event_distance": pf["event_distance"],
                    "match_mode": ATTACHMENT_MODE,
                    "logger_schema": LOGGER_SCHEMA,
                })
                candidate_count += 1

    print(
        "[ok] policy={} schema={} attachment={} demands={} candidates={} "
        "max_per_demand={} max_event_distance={} stream={} candidates_file={}".format(
            args.policy,
            LOGGER_SCHEMA,
            ATTACHMENT_MODE,
            len(demands),
            candidate_count,
            max_candidates,
            max_event_distance,
            args.stream_out,
            args.candidate_out,
        )
    )


if __name__ == "__main__":
    main()
