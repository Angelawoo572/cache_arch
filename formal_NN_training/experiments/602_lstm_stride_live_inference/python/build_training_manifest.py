#!/usr/bin/env python3
"""Normalize one exact retired-instruction prefix and describe its supervision."""

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formal_NN_training.common.normal_policy_reference import normal_actions


TRACE = "602.gcc_s-734B"
STREAM_FIELDS = [
    "trace", "demand_idx", "cycle", "pc", "line", "pc_line_occ",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gzip_content_sha256(path):
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_int(value):
    text = str(value).strip()
    return int(text, 16) if text.lower().startswith("0x") else int(float(text))


def normalize_events(events_path, stream_path):
    opener = gzip.open if str(events_path).endswith(".gz") else open
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    occurrences = defaultdict(int)
    rows = []
    first_raw_event = None
    with opener(events_path, "rt", newline="") as source, gzip.open(
        stream_path, "wt", newline=""
    ) as target:
        reader = csv.DictReader(source)
        required = {"event", "event_id", "cache", "op", "ip", "line", "cycle"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                "event log missing columns: {}".format(sorted(missing))
            )
        writer = csv.DictWriter(target, fieldnames=STREAM_FIELDS)
        writer.writeheader()
        for raw in reader:
            if (
                raw["event"] != "DEMAND"
                or raw["cache"] != "L2C"
                or raw["op"] != "read"
            ):
                continue
            pc = as_int(raw["ip"])
            line = as_int(raw["line"])
            pair = (pc, line)
            occurrence = occurrences[pair]
            occurrences[pair] += 1
            row = (pc, line, occurrence)
            rows.append(row)
            if first_raw_event is None:
                first_raw_event = {
                    "raw_event_id": as_int(raw["event_id"]),
                    "cycle": as_int(raw["cycle"]),
                    "decision_row": 0,
                }
            writer.writerow({
                "trace": TRACE,
                "demand_idx": len(rows) - 1,
                "cycle": as_int(raw["cycle"]),
                "pc": pc,
                "line": line,
                "pc_line_occ": occurrence,
            })
    return rows, first_raw_event


def point_status(decision_rows, silent_rows, positive_rows, minimum_rows):
    if decision_rows == 0:
        return "no_callbacks", "no post-boundary L2 demand callbacks"
    if positive_rows == 0:
        return "single_class_no_act", "training prefix contains only K=0"
    if silent_rows == 0:
        return "single_class_no_silent", "training prefix contains only K>0"
    if decision_rows < minimum_rows:
        return "insufficient_rows", "too few decision rows"
    return "trainable", None


def build_manifest(args):
    rows, first_callback = normalize_events(args.events, args.stream)
    actions, _ = normal_actions("stride", rows)
    counts = [len(items) for items in actions]
    positive_positions = [
        index for index, count in enumerate(counts) if count > 0
    ]
    silent_rows = sum(count == 0 for count in counts)
    positive_rows = len(counts) - silent_rows
    action_atoms = sum(counts)
    positive_pcs = {rows[index][0] for index in positive_positions}
    status, failure_reason = point_status(
        len(rows), silent_rows, positive_rows, args.minimum_rows
    )
    payload = {
        "schema_version": 1,
        "trace": TRACE,
        "instruction_budget": args.instruction_budget,
        "budget_tag": args.budget_tag,
        "collection_semantics": (
            "trace_start_to_exact_retired_instruction_budget"
        ),
        "collection_command": args.collection_command,
        "training_warmup_instructions": 0,
        "training_simulation_instructions": args.instruction_budget,
        "decision_rows": len(rows),
        "silent_rows": silent_rows,
        "positive_count_rows": positive_rows,
        "action_atoms": action_atoms,
        "k_histogram": {
            str(key): int(value)
            for key, value in sorted(Counter(counts).items())
        },
        "unique_pcs": len({pc for pc, _, _ in rows}),
        "unique_positive_pcs": len(positive_pcs),
        "first_callback_position": first_callback,
        "first_positive_label_position": (
            {
                "decision_row": positive_positions[0],
                "pc": rows[positive_positions[0]][0],
                "line": rows[positive_positions[0]][1],
            }
            if positive_positions else None
        ),
        "raw_event_log": str(args.events),
        "raw_event_log_sha256": sha256(args.events),
        "training_stream": str(args.stream),
        "training_stream_sha256": sha256(args.stream),
        "training_stream_content_sha256": gzip_content_sha256(args.stream),
        "status": status,
        "failure_reason": failure_reason,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--stream", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--instruction-budget", required=True, type=int)
    parser.add_argument("--budget-tag", required=True)
    parser.add_argument("--collection-command", required=True)
    parser.add_argument("--minimum-rows", type=int, default=2)
    return parser


def main():
    args = build_parser().parse_args()
    if args.instruction_budget < 1 or args.minimum_rows < 1:
        raise RuntimeError("budgets and minimum rows must be positive")
    payload = build_manifest(args)
    print(json.dumps({
        key: payload[key] for key in (
            "budget_tag", "instruction_budget", "decision_rows",
            "silent_rows", "positive_count_rows", "action_atoms", "status",
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()
