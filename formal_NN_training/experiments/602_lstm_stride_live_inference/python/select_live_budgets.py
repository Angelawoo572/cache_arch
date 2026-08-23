#!/usr/bin/env python3
"""Recommend distinct h8/h16 live budgets without launching simulations."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from analysis_policy import (
    first,
    reference_20m,
    smallest_reaching_at_least,
    smallest_within,
    stable_plateau,
)


def load_points(path):
    return json.loads(Path(path).read_text())["points"]


def transition(rows):
    valid = [row for row in rows if row.get("ipc") is not None]
    candidates = []
    for left, right in zip(valid, valid[1:]):
        distance = math.log10(right["instruction_budget"]) - math.log10(
            left["instruction_budget"]
        )
        candidates.append((
            (right["ipc"] - left["ipc"]) / distance,
            right,
        ))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def add(selected, row, reason):
    if row is None:
        return
    key = (row["hidden_size"], row["budget_tag"], row["seed"])
    selected.setdefault(key, {
        "hidden_size": row["hidden_size"],
        "instruction_budget": row["instruction_budget"],
        "budget_tag": row["budget_tag"],
        "seed": row["seed"],
        "reasons": [],
    })["reasons"].append(reason)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selection-config", required=True, type=Path)
    return parser


def main():
    args = build_parser().parse_args()
    grouped = defaultdict(list)
    for row in load_points(args.offline_results):
        grouped[row.get("hidden_size")].append(row)
    selected = {}
    confirmation = {}
    for hidden in (8, 16):
        rows = sorted(
            grouped[hidden], key=lambda row: row["instruction_budget"]
        )
        reference = reference_20m(rows)
        trainable = first(rows, lambda row: row.get("status") not in {
            "no_callbacks", "single_class_no_act",
            "single_class_no_silent", "insufficient_rows",
            "training_failed",
        })
        non_silent = first(rows, lambda row: (
            (row.get("offline_replay_entry_count") or 0) > 0
        ))
        near_transition = transition(rows)
        one_percent = smallest_within(rows, reference, 0.01)
        half_percent = smallest_within(rows, reference, 0.005)
        reaching_99 = smallest_reaching_at_least(rows, reference, 0.99)
        plateau, _ = stable_plateau(rows, reference, 0.005)
        above_plateau = None
        if plateau:
            above_plateau = first(
                rows,
                lambda row: (
                    row["instruction_budget"]
                    > plateau["instruction_budget"]
                    and row.get("ipc") is not None
                ),
            )
        for row, reason in (
            (trainable, "first_trainable_budget"),
            (non_silent, "first_non_all_silent_budget"),
            (near_transition, "near_offline_transition"),
            (reaching_99, "smallest_reaching_at_least_99_percent_of_20m_offline_ipc"),
            (one_percent, "smallest_within_1_percent_of_20m_offline_ipc"),
            (half_percent, "smallest_within_0.5_percent_of_20m_offline_ipc"),
            (plateau, "first_stable_offline_plateau_candidate"),
            (above_plateau, "one_point_above_stable_offline_plateau_candidate"),
            (reference, "20m_reference"),
        ):
            add(selected, row, reason)
        below = None
        if near_transition:
            below = next((
                row for row in reversed(rows)
                if row["instruction_budget"]
                < near_transition["instruction_budget"]
            ), None)
        confirmation["h{}".format(hidden)] = {
            "suggested_seeds": [7, 17, 27],
            "points": [
                item for item in (
                    below["budget_tag"] if below else None,
                    near_transition["budget_tag"]
                    if near_transition else None,
                    half_percent["budget_tag"] if half_percent else None,
                    reference["budget_tag"] if reference else None,
                ) if item
            ],
        }
    points = list(selected.values())
    points.sort(key=lambda item: (
        item["hidden_size"], item["instruction_budget"], item["seed"]
    ))
    recommendation = {
        "schema_version": 1,
        "approved": False,
        "launch_performed": False,
        "selected_points": points,
        "confirmation_set": confirmation,
    }
    for path in (args.output, args.selection_config):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(recommendation, indent=2, sort_keys=True) + "\n"
        )
    print("[recommendation only] {} deduplicated points".format(len(points)))
    print("[inspect and set approved=true] {}".format(args.selection_config))


if __name__ == "__main__":
    main()
