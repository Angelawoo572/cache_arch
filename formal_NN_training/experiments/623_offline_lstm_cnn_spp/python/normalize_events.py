#!/usr/bin/env python3
"""Normalize source-faithful SPP callback inputs and direct teacher actions.

The v5 logger writes a completed L2 demand immediately before its synchronous
prefetcher callback.  Each PF row carries that exact demand event ID.  This
normalizer never infers attachment from a future demand or from base_addr alone.
"""
import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path


TRACE = "623.xalancbmk_s-700B"
POLICY = "spp"
LOG2_BLOCK_SIZE = 6
PAGE_LINES = 64
LOGGER_SCHEMA = "623_causal_trigger_v5"
ATTACHMENT_MODE = "explicit_trigger_event_id"


def as_int(value):
    text = str(value).strip()
    return int(text, 16) if text.lower().startswith("0x") else int(float(text))


def fail(message, event_id=None):
    suffix = "" if event_id is None else " at event {}".format(event_id)
    raise RuntimeError(message + suffix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--policy", required=True, choices=[POLICY])
    parser.add_argument("--stream-out", required=True, type=Path)
    parser.add_argument("--teacher-actions-out", required=True, type=Path)
    parser.add_argument("--max-event-distance", type=int, default=256)
    args = parser.parse_args()

    opener = gzip.open if str(args.events).endswith(".gz") else open
    demands = []
    demand_by_event = {}
    actions_by_demand = defaultdict(list)
    occurrences = defaultdict(int)
    last_event_id = -1
    active_demand_event = None
    max_event_distance = 0

    with opener(args.events, "rt", newline="") as source:
        reader = csv.DictReader(source)
        required = {
            "event", "event_id", "cpu", "cycle", "cache", "op", "type",
            "ip", "addr", "line", "hit", "base_addr", "pf_line",
            "fill_level", "accepted", "duplicate", "trigger_event_id",
            "trigger_cpu", "trigger_ip", "trigger_line", "logger_schema",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                "event log has stale/incomplete schema; missing {}. Rebuild "
                "collection with RESET_PATCH=1 and FORCE=1.".format(
                    sorted(missing)
                )
            )

        for row_number, row in enumerate(reader, 2):
            if row["logger_schema"] != LOGGER_SCHEMA:
                fail(
                    "row {} schema {!r}; expected {!r}".format(
                        row_number, row["logger_schema"], LOGGER_SCHEMA
                    )
                )
            if row["cache"] != "L2C":
                fail("non-L2C row in SPP event log")
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

            if row["event"] == "DEMAND":
                if row["op"] != "read":
                    fail("non-read demand callback", event_id)
                pc = as_int(row["ip"])
                line = as_int(row["line"])
                full_addr = as_int(row["addr"])
                cache_hit = as_int(row["hit"])
                access_type = as_int(row["type"])
                if full_addr >> LOG2_BLOCK_SIZE != line:
                    fail("demand full address and line disagree", event_id)
                if cache_hit not in (0, 1):
                    fail("demand hit field is not Boolean", event_id)
                trigger_identity = (
                    as_int(row["trigger_event_id"]),
                    as_int(row["trigger_cpu"]),
                    as_int(row["trigger_ip"]),
                    as_int(row["trigger_line"]),
                )
                if trigger_identity != (event_id, cpu, pc, line):
                    fail("demand self-trigger identity mismatch", event_id)
                pair = (pc, line)
                occurrence = occurrences[pair]
                occurrences[pair] += 1
                demand = {
                    "trace": TRACE,
                    "demand_idx": len(demands),
                    "cycle": cycle,
                    "pc": pc,
                    # SPP is called with the line-aligned address, not the byte
                    # offset of the original load.
                    "address": line << LOG2_BLOCK_SIZE,
                    "line": line,
                    "cache_hit": cache_hit,
                    "access_type": access_type,
                    "pc_line_occ": occurrence,
                    "logger_schema": LOGGER_SCHEMA,
                    "_event_id": event_id,
                    "_cpu": cpu,
                }
                demands.append(demand)
                demand_by_event[event_id] = demand
                active_demand_event = event_id
                continue

            if row["event"] != "PF":
                fail("unknown event kind {!r}".format(row["event"]), event_id)
            if active_demand_event is None:
                fail("PF row precedes first demand", event_id)
            trigger_id = as_int(row["trigger_event_id"])
            if trigger_id != active_demand_event:
                fail(
                    "PF trigger is not the immediately active callback "
                    "({} != {})".format(trigger_id, active_demand_event),
                    event_id,
                )
            demand = demand_by_event.get(trigger_id)
            if demand is None or trigger_id >= event_id:
                fail("PF references missing/future trigger", event_id)
            trigger_identity = (
                as_int(row["trigger_cpu"]),
                as_int(row["trigger_ip"]),
                as_int(row["trigger_line"]),
            )
            expected = (demand["_cpu"], demand["pc"], demand["line"])
            if trigger_identity != expected:
                fail("PF trigger identity disagrees with demand", event_id)
            if cpu != demand["_cpu"] or cycle != demand["cycle"]:
                fail("PF is not synchronous with demand callback", event_id)
            if as_int(row["ip"]) != demand["pc"]:
                fail("PF transport IP differs from trigger IP", event_id)
            if as_int(row["base_addr"]) >> LOG2_BLOCK_SIZE != demand["line"]:
                fail("PF base address differs from trigger address", event_id)

            pf_line = as_int(row["pf_line"])
            fill_level = as_int(row["fill_level"])
            accepted = as_int(row["accepted"])
            duplicate = as_int(row["duplicate"])
            if fill_level not in (2, 4):
                fail("SPP output has invalid fill level", event_id)
            if accepted not in (0, 1) or duplicate not in (0, 1):
                fail("PF outcome bit is not Boolean", event_id)
            if duplicate and not accepted:
                fail("rejected PF cannot be a queue duplicate", event_id)
            if pf_line // PAGE_LINES != demand["line"] // PAGE_LINES:
                fail("SPP emitted a cross-page action", event_id)
            if pf_line == demand["line"]:
                fail("SPP emitted a self-prefetch action", event_id)

            distance = event_id - trigger_id
            if distance < 1 or distance > args.max_event_distance:
                fail(
                    "PF event distance {} outside [1, {}]".format(
                        distance, args.max_event_distance
                    ),
                    event_id,
                )
            max_event_distance = max(max_event_distance, distance)
            actions = actions_by_demand[demand["demand_idx"]]
            if any(item["pf_line"] == pf_line for item in actions):
                fail("SPP emitted two fill choices for one target", event_id)
            actions.append({
                "pf_event_id": event_id,
                "trigger_event_id": trigger_id,
                "pf_line": pf_line,
                "target_page_offset": pf_line % PAGE_LINES,
                "fill_level": fill_level,
                "accepted": accepted,
                "duplicate": duplicate,
                "event_distance": distance,
            })

    if not demands:
        raise RuntimeError("no completed post-warmup L2 demand callbacks")
    action_count = sum(len(items) for items in actions_by_demand.values())
    if action_count == 0:
        raise RuntimeError("SPP emitted no direct teacher actions")

    args.stream_out.parent.mkdir(parents=True, exist_ok=True)
    args.teacher_actions_out.parent.mkdir(parents=True, exist_ok=True)
    stream_fields = [
        "trace", "demand_idx", "cycle", "pc", "address", "line",
        "cache_hit", "access_type", "pc_line_occ", "logger_schema",
    ]
    with gzip.open(args.stream_out, "wt", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=stream_fields)
        writer.writeheader()
        for demand in demands:
            writer.writerow({key: demand[key] for key in stream_fields})

    action_fields = [
        "trace", "policy", "demand_idx", "pc", "line", "pc_line_occ",
        "action_rank", "pf_line", "target_page_offset", "fill_level",
        "accepted", "duplicate", "trigger_event_id", "pf_event_id",
        "event_distance", "match_mode", "logger_schema",
    ]
    max_actions = 0
    fill_counts = {2: 0, 4: 0}
    with gzip.open(args.teacher_actions_out, "wt", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=action_fields)
        writer.writeheader()
        for demand in demands:
            actions = actions_by_demand.get(demand["demand_idx"], ())
            max_actions = max(max_actions, len(actions))
            for rank, action in enumerate(actions, 1):
                fill_counts[action["fill_level"]] += 1
                writer.writerow({
                    "trace": TRACE,
                    "policy": POLICY,
                    "demand_idx": demand["demand_idx"],
                    "pc": demand["pc"],
                    "line": demand["line"],
                    "pc_line_occ": demand["pc_line_occ"],
                    "action_rank": rank,
                    "pf_line": action["pf_line"],
                    "target_page_offset": action["target_page_offset"],
                    "fill_level": action["fill_level"],
                    "accepted": action["accepted"],
                    "duplicate": action["duplicate"],
                    "trigger_event_id": action["trigger_event_id"],
                    "pf_event_id": action["pf_event_id"],
                    "event_distance": action["event_distance"],
                    "match_mode": ATTACHMENT_MODE,
                    "logger_schema": LOGGER_SCHEMA,
                })

    print(
        "[ok] policy={} schema={} demands={} direct_actions={} "
        "max_actions_per_demand={} fill_l2={} fill_llc={} "
        "max_event_distance={}".format(
            POLICY, LOGGER_SCHEMA, len(demands), action_count, max_actions,
            fill_counts[2], fill_counts[4], max_event_distance,
        )
    )


if __name__ == "__main__":
    main()
