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
        near_transition = transition(rows)
        one_percent = smallest_within(rows, reference, 0.01)
        half_percent = smallest_within(rows, reference, 0.005)
        reaching_99 = smallest_reaching_at_least(rows, reference, 0.99)
        plateau, _ = stable_plateau(rows, reference, 0.005)
        by_tag = {row["budget_tag"]: row for row in rows}
        for budget_tag, reason in (
            ("i1m", "primary_1m_vs_20m_live_validation"),
            ("i20m", "same_hidden_size_live_reference"),
            ("i250k" if hidden == 8 else "i100k",
             "aggressive_high_performing_candidate"),
        ):
            row = by_tag.get(budget_tag)
            if row and row.get("ipc") is not None:
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
        "offline_diagnostics": {
            "selection_is_secondary_to_offline_fairness": True,
            "one_sided_and_two_sided_metrics_are_not_interchangeable": True,
        },
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
