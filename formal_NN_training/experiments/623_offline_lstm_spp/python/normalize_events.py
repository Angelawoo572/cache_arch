#!/usr/bin/env python3
"""Normalize source-faithful SPP callback inputs and direct teacher actions.

The v6 logger records every external callback that can affect source SPP:
completed L2 demand callbacks and L2 cache-fill eviction feedback.  PF rows
carry the exact demand event ID that synchronously caused them.  The normalized
model stream preserves DEMAND/FILL order and never infers attachment from a
future demand, a distance cutoff, or ``base_addr`` alone.
"""
import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path

from model_contract import (
    ADDRESS_BITS, CACHE_LINE_SHIFT, CACHE_LINE_BYTES, POLICY,
    TRACE, exact_int as as_int,
)

LOG2_BLOCK_SIZE = CACHE_LINE_SHIFT
SOURCE_SPP_PAGE_LINES = 4096 // CACHE_LINE_BYTES
LOGGER_SCHEMA = "623_causal_trigger_fill_v6"
ATTACHMENT_MODE = "explicit_trigger_event_id"
CANONICALIZATION_MODE = "per_target_min_fill_queue_effect"
NO_TRIGGER = (1 << ADDRESS_BITS) - 1


def fail(message, event_id=None):
    suffix = "" if event_id is None else " at event {}".format(event_id)
    raise RuntimeError(message + suffix)


def canonicalize_actions(raw_actions, demand_line):
    """Apply exactly the queue-visible repeated-target merge rule.

    SPP can revisit a target through multi-step lookahead.  CACHE::add_pq merges
    repeated target lines and upgrades the queued fill when the new numeric fill
    is smaller, so FILL_L2 (2) dominates FILL_LLC (4).  First-target order is
    retained; no action is removed by a learned-model threshold.
    """
    canonical = []
    by_line = {}
    for raw in raw_actions:
        pf_line = raw["pf_line"]
        current = by_line.get(pf_line)
        if current is None:
            current = dict(raw)
            current.update({
                "raw_action_count": 1,
                "source_first_pf_event_id": raw["pf_event_id"],
                "source_last_pf_event_id": raw["pf_event_id"],
                "is_self_target": int(pf_line == demand_line),
                "canonicalization": CANONICALIZATION_MODE,
            })
            by_line[pf_line] = current
            canonical.append(current)
            continue
        current["raw_action_count"] += 1
        current["source_last_pf_event_id"] = raw["pf_event_id"]
        current["accepted"] = max(current["accepted"], raw["accepted"])
        current["duplicate"] = max(current["duplicate"], raw["duplicate"])
        current["fill_level"] = min(current["fill_level"], raw["fill_level"])
    return canonical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--policy", required=True, choices=[POLICY])
    parser.add_argument("--stream-out", required=True, type=Path)
    parser.add_argument("--teacher-actions-out", required=True, type=Path)
    args = parser.parse_args()

    opener = gzip.open if str(args.events).endswith(".gz") else open
    events = []
    demands = []
    demand_by_event = {}
    actions_by_demand = defaultdict(list)
    occurrences = defaultdict(int)
    last_event_id = -1
    active_demand_event = None
    max_event_distance = 0
    fill_callback_count = 0

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
                demand_idx = len(demands)
                demand = {
                    "trace": TRACE,
                    "demand_idx": demand_idx,
                    "cycle": cycle,
                    "pc": pc,
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
                events.append({
                    "trace": TRACE,
                    "event_idx": len(events),
                    "raw_event_id": event_id,
                    "cycle": cycle,
                    "event_kind": "DEMAND",
                    "event_address": line << LOG2_BLOCK_SIZE,
                    "event_line": line,
                    "decision_idx": demand_idx,
                    "pc": pc,
                    "cache_hit": cache_hit,
                    "access_type": access_type,
                    "pc_line_occ": occurrence,
                    "logger_schema": LOGGER_SCHEMA,
                })
                active_demand_event = event_id
                continue

            if row["event"] == "FILL":
                # This is the exact evicted_addr passed to SPP_dev2::cache_fill.
                # The filled address is retained in base_addr for audit only.
                if row["op"] != "cache_fill":
                    fail("invalid cache-fill operation", event_id)
                evicted_addr = as_int(row["addr"])
                evicted_line = as_int(row["line"])
                filled_addr = as_int(row["base_addr"])
                if evicted_addr >> LOG2_BLOCK_SIZE != evicted_line:
                    fail("cache-fill evicted address and line disagree", event_id)
                if evicted_addr % (1 << LOG2_BLOCK_SIZE):
                    fail("cache-fill evicted address is not line aligned", event_id)
                if filled_addr % (1 << LOG2_BLOCK_SIZE):
                    fail("cache-fill installed address is not line aligned", event_id)
                if as_int(row["ip"]) != 0:
                    fail("cache-fill input unexpectedly carries PC", event_id)
                if (
                    as_int(row["trigger_event_id"]) != NO_TRIGGER
                    or any(as_int(row[name]) != 0 for name in (
                        "trigger_cpu", "trigger_ip", "trigger_line"
                    ))
                ):
                    fail("cache-fill input unexpectedly carries demand trigger", event_id)
                events.append({
                    "trace": TRACE,
                    "event_idx": len(events),
                    "raw_event_id": event_id,
                    "cycle": cycle,
                    "event_kind": "FILL",
                    "event_address": evicted_addr,
                    "event_line": evicted_line,
                    "decision_idx": -1,
                    "pc": 0,
                    "cache_hit": 0,
                    "access_type": as_int(row["type"]),
                    "pc_line_occ": -1,
                    "logger_schema": LOGGER_SCHEMA,
                })
                fill_callback_count += 1
                active_demand_event = None
                continue

            if row["event"] != "PF":
                fail("unknown event kind {!r}".format(row["event"]), event_id)
            if active_demand_event is None:
                fail("PF row is not inside an active demand callback", event_id)
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
            if not accepted:
                fail("teacher collection contains a dropped SPP request", event_id)
            if duplicate and not accepted:
                fail("rejected PF cannot be a queue duplicate", event_id)
            if pf_line // SOURCE_SPP_PAGE_LINES != demand["line"] // SOURCE_SPP_PAGE_LINES:
                fail("SPP emitted a cross-page action", event_id)

            distance = event_id - trigger_id
            if distance < 1:
                fail("PF event does not follow its explicit trigger", event_id)
            max_event_distance = max(max_event_distance, distance)
            actions_by_demand[demand["demand_idx"]].append({
                "pf_event_id": event_id,
                "trigger_event_id": trigger_id,
                "pf_line": pf_line,
                "fill_level": fill_level,
                "accepted": accepted,
                "duplicate": duplicate,
                "event_distance": distance,
            })

    if not demands:
        raise RuntimeError("no completed post-warmup L2 demand callbacks")
    if not fill_callback_count:
        raise RuntimeError("no post-warmup SPP cache-fill feedback callbacks")
    raw_action_count = sum(len(items) for items in actions_by_demand.values())
    if raw_action_count == 0:
        raise RuntimeError("SPP emitted no direct teacher actions")
    canonical_actions_by_demand = {
        demand["demand_idx"]: canonicalize_actions(
            actions_by_demand.get(demand["demand_idx"], ()), demand["line"]
        )
        for demand in demands
    }
    action_count = sum(
        len(items) for items in canonical_actions_by_demand.values()
    )

    args.stream_out.parent.mkdir(parents=True, exist_ok=True)
    args.teacher_actions_out.parent.mkdir(parents=True, exist_ok=True)
    stream_fields = [
        "trace", "event_idx", "raw_event_id", "cycle", "event_kind",
        "event_address", "event_line", "decision_idx", "pc", "cache_hit",
        "access_type", "pc_line_occ", "logger_schema",
    ]
    with gzip.open(args.stream_out, "wt", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=stream_fields)
        writer.writeheader()
        writer.writerows(events)

    action_fields = [
        "trace", "policy", "demand_idx", "pc", "line", "pc_line_occ",
        "action_rank", "pf_line", "fill_level",
        "accepted", "duplicate", "trigger_event_id", "pf_event_id",
        "event_distance", "raw_action_count", "source_first_pf_event_id",
        "source_last_pf_event_id", "is_self_target", "canonicalization",
        "match_mode", "logger_schema",
    ]
    max_actions = 0
    max_raw_actions = 0
    fill_counts = {2: 0, 4: 0}
    self_target_actions = 0
    collapsed_source_calls = 0
    with gzip.open(args.teacher_actions_out, "wt", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=action_fields)
        writer.writeheader()
        for demand in demands:
            raw_actions = actions_by_demand.get(demand["demand_idx"], ())
            actions = canonical_actions_by_demand.get(demand["demand_idx"], ())
            max_raw_actions = max(max_raw_actions, len(raw_actions))
            max_actions = max(max_actions, len(actions))
            for rank, action in enumerate(actions, 1):
                fill_counts[action["fill_level"]] += 1
                self_target_actions += action["is_self_target"]
                collapsed_source_calls += action["raw_action_count"] - 1
                writer.writerow({
                    "trace": TRACE,
                    "policy": POLICY,
                    "demand_idx": demand["demand_idx"],
                    "pc": demand["pc"],
                    "line": demand["line"],
                    "pc_line_occ": demand["pc_line_occ"],
                    "action_rank": rank,
                    "pf_line": action["pf_line"],
                    "fill_level": action["fill_level"],
                    "accepted": action["accepted"],
                    "duplicate": action["duplicate"],
                    "trigger_event_id": action["trigger_event_id"],
                    "pf_event_id": action["pf_event_id"],
                    "event_distance": action["event_distance"],
                    "raw_action_count": action["raw_action_count"],
                    "source_first_pf_event_id": action["source_first_pf_event_id"],
                    "source_last_pf_event_id": action["source_last_pf_event_id"],
                    "is_self_target": action["is_self_target"],
                    "canonicalization": action["canonicalization"],
                    "match_mode": ATTACHMENT_MODE,
                    "logger_schema": LOGGER_SCHEMA,
                })

    print(
        "[ok] policy={} schema={} context_events={} demands={} fills={} "
        "raw_source_calls={} canonical_actions={} collapsed_source_calls={} "
        "self_targets={} max_raw_actions_per_demand={} "
        "max_canonical_actions_per_demand={} fill_l2={} fill_llc={} "
        "observed_max_event_distance={}".format(
            POLICY, LOGGER_SCHEMA, len(events), len(demands),
            fill_callback_count, raw_action_count, action_count,
            collapsed_source_calls, self_target_actions, max_raw_actions,
            max_actions, fill_counts[2], fill_counts[4], max_event_distance,
        )
    )


if __name__ == "__main__":
    main()
