#!/usr/bin/env python3
"""Primary same-hidden-size offline keyed-replay sufficiency analysis."""

import json
import math

from analysis_policy import (
    first,
    reference_20m,
    relative_deviation,
    smallest_reaching_at_least,
    smallest_within,
    stability_evidence,
    stable_plateau,
)


KEY_BUDGETS = {
    "i100k", "i250k", "i500k", "i1m", "i2m", "i5m", "i10m", "i20m",
}


def tag(row):
    return row.get("budget_tag") if row else None


def transition(rows, field="ipc"):
    valid = sorted(
        (row for row in rows if row.get(field) is not None),
        key=lambda row: row["instruction_budget"],
    )
    slopes = []
    for left, right in zip(valid, valid[1:]):
        dx = math.log10(right["instruction_budget"]) - math.log10(
            left["instruction_budget"]
        )
        slopes.append(((right[field] - left[field]) / dx, left, right))
    if not slopes:
        return None
    _, left, right = max(slopes, key=lambda item: item[0])
    return [left["budget_tag"], right["budget_tag"]]


def high_performing_candidates(rows, reference):
    if reference is None or reference.get("ipc") in (None, 0):
        return []
    candidates = []
    for row in sorted(rows, key=lambda item: item["instruction_budget"]):
        if (
            row.get("instruction_budget", 0) >= reference["instruction_budget"]
            or row.get("ipc") is None
            or row["ipc"] <= reference["ipc"]
        ):
            continue
        signals = []
        for field in (
            "coverage", "requests_per_l2_load",
            "student_heldout_act_rate",
        ):
            if (
                row.get(field) is not None
                and reference.get(field) is not None
                and row[field] > reference[field]
            ):
                signals.append(field)
        candidates.append({
            "budget_tag": row["budget_tag"],
            "instruction_budget": row["instruction_budget"],
            "ipc": row["ipc"],
            "relative_ipc_uplift_vs_same_h_20m": (
                (row["ipc"] - reference["ipc"]) / reference["ipc"]
            ),
            "aggressive_signals_vs_same_h_20m": signals,
        })
    return candidates


def plateau_summary(rows, reference, tolerance=0.005):
    candidate, suffix = stable_plateau(rows, reference, tolerance)
    if candidate is None:
        return None
    fields = (
        "coverage", "l2_load_miss_rate", "requests_per_l2_load",
        "student_heldout_act_rate", "timeliness",
        "offline_replay_entry_count",
    )
    return {
        "budget_tag": candidate["budget_tag"],
        "instruction_budget": candidate["instruction_budget"],
        "ipc_relative_tolerance": tolerance,
        "definition": (
            "candidate and every subsequent observed same-hidden-size "
            "offline IPC through 20M are within the two-sided tolerance"
        ),
        "observed_budget_tags": [row["budget_tag"] for row in suffix],
        "max_ipc_relative_deviation": max(
            relative_deviation(row["ipc"], reference["ipc"])
            for row in suffix
        ),
        "auxiliary_stability_near_same_h_20m": stability_evidence(
            suffix, reference, fields
        ),
    }


def conclusions_for_hidden(offline, live=None):
    """Return separate offline-primary and optional live-secondary results."""
    offline = sorted(offline, key=lambda row: row["instruction_budget"])
    live = sorted(live or [], key=lambda row: row["instruction_budget"])
    off_ref = reference_20m(offline)
    live_ref = reference_20m(live)
    first_callback = first(
        offline, lambda row: (row.get("decision_rows") or 0) > 0
    )
    first_positive = first(
        offline, lambda row: (row.get("positive_count_rows") or 0) > 0
    )
    first_trainable = first(
        offline,
        lambda row: (
            (row.get("decision_rows") or 0) >= 2
            and (row.get("silent_rows") or 0) > 0
            and (row.get("positive_count_rows") or 0) > 0
            and row.get("status") != "training_failed"
        ),
    )
    first_non_silent = first(
        offline,
        lambda row: (row.get("offline_replay_entry_count") or 0) > 0,
    )
    off_plateau = plateau_summary(offline, off_ref)
    live_plateau = plateau_summary(live, live_ref)
    result = {
        "comparison_reference": "same hidden size at i20m",
        "first_prefix_with_any_callback": tag(first_callback),
        "first_prefix_with_any_positive_label": tag(first_positive),
        "first_trainable_prefix": tag(first_trainable),
        "first_non_all_silent_model": tag(first_non_silent),
        "offline_transition_region": transition(offline),
        "offline_stable_plateau": off_plateau,
        "live_transition_region": transition(live),
        "live_stable_plateau": live_plateau,
        "offline_20m_ipc": off_ref.get("ipc") if off_ref else None,
        "live_20m_ipc": live_ref.get("ipc") if live_ref else None,
        "offline_high_performing_small_budget_candidates": (
            high_performing_candidates(offline, off_ref)
        ),
        "live_high_performing_small_budget_candidates": (
            high_performing_candidates(live, live_ref)
        ),
    }
    for label, fraction in (("99", 0.99), ("99_5", 0.995), ("99_9", 0.999)):
        result[
            "smallest_budget_reaching_at_least_{}_percent_of_20m_offline_ipc".format(label)
        ] = tag(smallest_reaching_at_least(offline, off_ref, fraction))
        result[
            "smallest_budget_reaching_at_least_{}_percent_of_20m_live_ipc".format(label)
        ] = tag(smallest_reaching_at_least(live, live_ref, fraction))
    for label, tolerance in (("1.0", 0.01), ("0.5", 0.005), ("0.1", 0.001)):
        result[
            "smallest_within_{}_percent_20m_offline_ipc".format(label)
        ] = tag(smallest_within(offline, off_ref, tolerance))
        result[
            "smallest_within_{}_percent_20m_live_ipc".format(label)
        ] = tag(smallest_within(live, live_ref, tolerance))
    return result


def difference(value, reference):
    if value is None or reference is None:
        return None
    return float(value) - float(reference)


def point_differences(rows):
    rows = sorted(rows, key=lambda row: row["instruction_budget"])
    reference = reference_20m(rows)
    plateau, suffix = stable_plateau(rows, reference, 0.005)
    plateau_tags = {row["budget_tag"] for row in suffix}
    output = []
    if reference is None:
        return output
    for row in rows:
        copy = dict(row)
        copy.update({
            "same_hidden_size_20m_budget_tag": reference["budget_tag"],
            "same_hidden_size_20m_ipc": reference.get("ipc"),
            "relative_ipc_difference_vs_same_h_20m": (
                (row["ipc"] - reference["ipc"]) / reference["ipc"]
                if row.get("ipc") is not None and reference.get("ipc")
                else None
            ),
            "coverage_percentage_point_difference_vs_same_h_20m": (
                100.0 * difference(row.get("coverage"), reference.get("coverage"))
                if difference(row.get("coverage"), reference.get("coverage"))
                is not None else None
            ),
            "l2_miss_rate_percentage_point_difference_vs_same_h_20m": (
                100.0 * difference(
                    row.get("l2_load_miss_rate"),
                    reference.get("l2_load_miss_rate"),
                )
                if difference(
                    row.get("l2_load_miss_rate"),
                    reference.get("l2_load_miss_rate"),
                ) is not None else None
            ),
            "request_pressure_difference_vs_same_h_20m": difference(
                row.get("requests_per_l2_load"),
                reference.get("requests_per_l2_load"),
            ),
            "student_act_rate_percentage_point_difference_vs_same_h_20m": (
                100.0 * difference(
                    row.get("student_heldout_act_rate"),
                    reference.get("student_heldout_act_rate"),
                )
                if difference(
                    row.get("student_heldout_act_rate"),
                    reference.get("student_heldout_act_rate"),
                ) is not None else None
            ),
            "timeliness_percentage_point_difference_vs_same_h_20m": (
                100.0 * difference(
                    row.get("timeliness"), reference.get("timeliness")
                )
                if difference(
                    row.get("timeliness"), reference.get("timeliness")
                ) is not None else None
            ),
            "replay_action_count_difference_vs_same_h_20m": difference(
                row.get("offline_replay_entry_count"),
                reference.get("offline_replay_entry_count"),
            ),
            "stable_0_5_percent_ipc_plateau_member": (
                row["budget_tag"] in plateau_tags
            ),
            "stable_0_5_percent_ipc_plateau_start": (
                plateau is not None
                and row["budget_tag"] == plateau["budget_tag"]
            ),
        })
        output.append(copy)
    return output


def live_comparisons(offline, live):
    offline_by_key = {
        (row["hidden_size"], row["instruction_budget"], row["seed"]): row
        for row in offline
    }
    refs = {}
    for hidden in (8, 16):
        refs[hidden] = reference_20m([
            row for row in live if row.get("hidden_size") == hidden
        ])
    output = []
    for row in sorted(live, key=lambda item: (
        item["hidden_size"], item["instruction_budget"], item["seed"]
    )):
        key = (row["hidden_size"], row["instruction_budget"], row["seed"])
        offline_row = offline_by_key.get(key)
        reference = refs.get(row["hidden_size"])
        copy = dict(row)
        copy.update({
            "same_checkpoint_offline_ipc": (
                offline_row.get("ipc") if offline_row else None
            ),
            "live_minus_same_checkpoint_offline_ipc": (
                difference(row.get("ipc"), offline_row.get("ipc"))
                if offline_row else None
            ),
            "same_hidden_size_live_20m_ipc": (
                reference.get("ipc") if reference else None
            ),
            "relative_live_ipc_difference_vs_same_h_live_20m": (
                (row["ipc"] - reference["ipc"]) / reference["ipc"]
                if reference and row.get("ipc") is not None
                and reference.get("ipc") else None
            ),
        })
        output.append(copy)
    return output


def build_analysis(offline, live):
    points = []
    conclusions = {}
    for hidden in (8, 16):
        off_rows = [row for row in offline if row.get("hidden_size") == hidden]
        live_rows = [row for row in live if row.get("hidden_size") == hidden]
        points.extend(point_differences(off_rows))
        conclusions["h{}".format(hidden)] = conclusions_for_hidden(
            off_rows, live_rows
        )
    return {
        "schema_version": 3,
        "primary_question": (
            "minimum trace-start prefix N whose offline keyed-replay system "
            "result is equivalent to the same-hidden-size 20M model"
        ),
        "scope": (
            "602.gcc_s-734B, seed 7, original offline keyed-replay protocol"
        ),
        "comparison_rule": "h8-N versus h8-20M; h16-N versus h16-20M",
        "analysis_policy": {
            "reaching_at_least": (
                "IPC_N >= requested_fraction * IPC_same_H_20M (one-sided)"
            ),
            "within": (
                "abs(IPC_N - IPC_same_H_20M) / abs(IPC_same_H_20M) "
                "<= tolerance (two-sided)"
            ),
            "stable_plateau": (
                "candidate and every subsequent observed same-H offline IPC "
                "through 20M are within +/-0.5% of same-H 20M IPC"
            ),
            "auxiliary_metrics": (
                "coverage, L2 miss rate, request pressure, student act rate, "
                "timeliness, and replay actions are reported separately"
            ),
        },
        "conclusions": conclusions,
        "offline_points": sorted(points, key=lambda row: (
            row["hidden_size"], row["instruction_budget"], row["seed"]
        )),
        "live_secondary": {
            "terminology": (
                "functional live inference with zero modeled NN inference latency"
            ),
            "points": live_comparisons(offline, live),
            "missing_points_are_interpolated": False,
        },
    }


def compact_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
