#!/usr/bin/env python3
"""Join offline/live results, compute conclusions, and find the Pareto set."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from result_utils import write_outputs
from analysis_policy import (
    first,
    reference_20m,
    relative_deviation,
    smallest_reaching_at_least,
    smallest_within,
    stability_evidence,
    stable_plateau,
)


def load(path):
    return json.loads(Path(path).read_text())


def transition(rows, field="ipc"):
    valid = [row for row in rows if row.get(field) is not None]
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


def tag(row):
    return row["budget_tag"] if row else None


def high_performing_candidates(rows, reference):
    if reference is None or reference.get("ipc") in (None, 0):
        return []
    result = []
    for row in rows:
        if (
            row.get("instruction_budget", 0) >= reference["instruction_budget"]
            or row.get("ipc") is None
            or row["ipc"] <= reference["ipc"]
        ):
            continue
        candidate = {
            "budget_tag": row["budget_tag"],
            "instruction_budget": row["instruction_budget"],
            "ipc": row["ipc"],
            "relative_ipc_uplift_vs_20m": (
                (row["ipc"] - reference["ipc"]) / reference["ipc"]
            ),
        }
        aggressive_signals = []
        for field in (
            "coverage", "requests_per_l2_load", "student_heldout_act_rate"
        ):
            candidate[field] = row.get(field)
            candidate[field + "_20m"] = reference.get(field)
            if (
                row.get(field) is not None
                and reference.get(field) is not None
                and row[field] > reference[field]
            ):
                aggressive_signals.append(field)
        candidate["aggressive_signals_vs_20m"] = aggressive_signals
        result.append(candidate)
    return result


def plateau_summary(rows, reference, tolerance=0.005):
    candidate, suffix = stable_plateau(rows, reference, tolerance)
    if candidate is None:
        return None
    return {
        "budget_tag": candidate["budget_tag"],
        "instruction_budget": candidate["instruction_budget"],
        "ipc_relative_tolerance": tolerance,
        "definition": (
            "candidate and every subsequent observed budget through 20M "
            "are within the two-sided IPC tolerance"
        ),
        "observed_budget_tags": [row["budget_tag"] for row in suffix],
        "max_ipc_relative_deviation": max(
            relative_deviation(row["ipc"], reference["ipc"])
            for row in suffix
        ),
        "auxiliary_stability_near_20m": stability_evidence(
            suffix,
            reference,
            (
                "coverage", "requests_per_l2_load",
                "student_heldout_act_rate",
            ),
        ),
    }


def conclusions_for_hidden(offline, live):
    offline = sorted(offline, key=lambda row: row["instruction_budget"])
    live = sorted(live, key=lambda row: row["instruction_budget"])
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
    all_silent = [
        row["budget_tag"] for row in offline
        if row.get("offline_replay_entry_count") == 0
        and row.get("optimizer_steps") is not None
    ]
    f1_rows = [
        row for row in offline if row.get("heldout_target_f1") is not None
    ]
    if len(f1_rows) >= 2:
        delta_f1 = (
            f1_rows[-1]["heldout_target_f1"]
            - f1_rows[0]["heldout_target_f1"]
        )
        teacher_trend = {
            "first_budget": f1_rows[0]["budget_tag"],
            "last_budget": f1_rows[-1]["budget_tag"],
            "target_f1_change": delta_f1,
            "direction": (
                "closer" if delta_f1 > 0 else
                "less_similar" if delta_f1 < 0 else "unchanged"
            ),
        }
    else:
        teacher_trend = None
    offline_plateau = plateau_summary(offline, off_ref)
    live_plateau = plateau_summary(live, live_ref)
    live_plateau_row = first(
        live,
        lambda row: (
            live_plateau is not None
            and row.get("budget_tag") == live_plateau["budget_tag"]
        ),
    )
    result = {
        "first_prefix_with_any_callback": tag(first_callback),
        "first_prefix_with_any_positive_label": tag(first_positive),
        "first_trainable_prefix": tag(first_trainable),
        "first_non_all_silent_model": tag(first_non_silent),
        "offline_transition_region": transition(offline),
        "offline_stable_plateau": offline_plateau,
        "live_transition_region": transition(live),
        "live_stable_plateau": live_plateau,
        "smallest_budget_reaching_at_least_99_percent_of_20m_offline_ipc": tag(
            smallest_reaching_at_least(offline, off_ref, 0.99)
        ),
        "smallest_budget_reaching_at_least_99_5_percent_of_20m_offline_ipc": tag(
            smallest_reaching_at_least(offline, off_ref, 0.995)
        ),
        "smallest_budget_reaching_at_least_99_9_percent_of_20m_offline_ipc": tag(
            smallest_reaching_at_least(offline, off_ref, 0.999)
        ),
        "smallest_budget_reaching_at_least_99_percent_of_20m_live_ipc": tag(
            smallest_reaching_at_least(live, live_ref, 0.99)
        ),
        "smallest_budget_reaching_at_least_99_5_percent_of_20m_live_ipc": tag(
            smallest_reaching_at_least(live, live_ref, 0.995)
        ),
        "smallest_budget_reaching_at_least_99_9_percent_of_20m_live_ipc": tag(
            smallest_reaching_at_least(live, live_ref, 0.999)
        ),
        "smallest_within_1.0_percent_20m_offline_ipc": tag(
            smallest_within(offline, off_ref, 0.01)
        ),
        "smallest_within_0.5_percent_20m_offline_ipc": tag(
            smallest_within(offline, off_ref, 0.005)
        ),
        "smallest_within_0.1_percent_20m_offline_ipc": tag(
            smallest_within(offline, off_ref, 0.001)
        ),
        "smallest_within_1.0_percent_20m_live_ipc": tag(
            smallest_within(live, live_ref, 0.01)
        ),
        "smallest_within_0.5_percent_20m_live_ipc": tag(
            smallest_within(live, live_ref, 0.005)
        ),
        "smallest_within_0.1_percent_20m_live_ipc": tag(
            smallest_within(live, live_ref, 0.001)
        ),
        "trained_all_silent_budgets": all_silent,
        "small_prefix_behavior": (
            "trained all-silent points precede useful behavior"
            if all_silent else
            "no trained all-silent point observed"
            if first_non_silent else None
        ),
        "teacher_similarity_trend": teacher_trend,
        "request_pressure_at_live_stable_plateau": (
            live_plateau_row.get("requests_per_l2_load")
            if live_plateau_row else None
        ),
        "offline_high_performing_small_budget_candidates": (
            high_performing_candidates(offline, off_ref)
        ),
        "live_high_performing_small_budget_candidates": (
            high_performing_candidates(live, live_ref)
        ),
        "offline_20m_ipc": off_ref.get("ipc") if off_ref else None,
        "live_20m_ipc": live_ref.get("ipc") if live_ref else None,
    }
    return result


def pareto(rows):
    candidates = [
        row for row in rows
        if row.get("ipc") is not None
        and row.get("live_total_deployment_bytes") is not None
    ]
    frontier = []
    for point in candidates:
        dominated = False
        for other in candidates:
            if other is point:
                continue
            no_worse = (
                other["instruction_budget"] <= point["instruction_budget"]
                and other["live_total_deployment_bytes"]
                <= point["live_total_deployment_bytes"]
                and other["ipc"] >= point["ipc"]
            )
            if (
                no_worse
                and point.get("host_mean_nanoseconds") is not None
                and other.get("host_mean_nanoseconds") is not None
            ):
                no_worse = (
                    other["host_mean_nanoseconds"]
                    <= point["host_mean_nanoseconds"]
                )
            strict = (
                other["instruction_budget"] < point["instruction_budget"]
                or other["live_total_deployment_bytes"]
                < point["live_total_deployment_bytes"]
                or other["ipc"] > point["ipc"]
                or (
                    point.get("host_mean_nanoseconds") is not None
                    and other.get("host_mean_nanoseconds") is not None
                    and other["host_mean_nanoseconds"]
                    < point["host_mean_nanoseconds"]
                )
            )
            if no_worse and strict:
                dominated = True
                break
        if not dominated:
            copy = dict(point)
            copy["pareto_frontier"] = True
            frontier.append(copy)
    return sorted(frontier, key=lambda row: (
        row["instruction_budget"], row["live_total_deployment_bytes"]
    ))


def tex_escape(value):
    if value is None:
        return "NA"
    text = json.dumps(value) if isinstance(value, list) else str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def display_value(value):
    if isinstance(value, dict) and "budget_tag" in value:
        return value["budget_tag"]
    return value


def percent(value):
    return "NA" if value is None else "{:.3f}\\%".format(100.0 * value)


def compact_candidates(candidates):
    if not candidates:
        return "none observed"
    items = []
    labels = {
        "coverage": "coverage",
        "requests_per_l2_load": "request pressure",
        "student_heldout_act_rate": "student act rate",
    }
    for candidate in candidates:
        signals = candidate.get("aggressive_signals_vs_20m") or []
        items.append(
            "{} (IPC uplift {}; higher than 20M: {})".format(
                tex_escape(candidate["budget_tag"]),
                percent(candidate.get("relative_ipc_uplift_vs_20m")),
                ", ".join(labels[item] for item in signals)
                if signals else "no reported traffic/act-rate metric",
            )
        )
    return "; ".join(items)


def compact_plateau(plateau):
    if plateau is None:
        return (
            "NA (no pre-20M candidate has an observed suffix entirely "
            "within 0.5\\% of 20M IPC)"
        )
    auxiliary = plateau["auxiliary_stability_near_20m"]
    diagnostics = []
    for field, label in (
        ("coverage", "coverage"),
        ("requests_per_l2_load", "request pressure"),
        ("student_heldout_act_rate", "student act rate"),
    ):
        item = auxiliary[field]
        verdict = item["all_observed_within_tolerance"]
        diagnostics.append("{} {} (max deviation {})".format(
            label,
            "stable" if verdict is True else
            "not stable" if verdict is False else "NA",
            percent(item["max_relative_deviation"]),
        ))
    return "{}; observed suffix {}; max IPC deviation {}; {}".format(
        tex_escape(plateau["budget_tag"]),
        tex_escape(plateau["observed_budget_tags"]),
        percent(plateau["max_ipc_relative_deviation"]),
        "; ".join(diagnostics),
    )


def write_conclusions_csv(path, conclusions):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("hidden_size", "conclusion", "value")
        )
        writer.writeheader()
        for hidden in (8, 16):
            for key in sorted(conclusions["h{}".format(hidden)]):
                value = conclusions["h{}".format(hidden)][key]
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, sort_keys=True)
                writer.writerow({
                    "hidden_size": hidden,
                    "conclusion": key,
                    "value": "NA" if value is None else value,
                })
        for key in sorted(conclusions["analysis_policy"]):
            writer.writerow({
                "hidden_size": "all",
                "conclusion": "analysis_policy." + key,
                "value": conclusions["analysis_policy"][key],
            })


def write_tex(path, conclusions, frontier):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        ("First prefix with any callback", "first_prefix_with_any_callback"),
        ("First prefix with any $K>0$ label", "first_prefix_with_any_positive_label"),
        ("First trainable prefix", "first_trainable_prefix"),
        ("First non-all-silent model", "first_non_all_silent_model"),
        ("Offline transition region", "offline_transition_region"),
        ("Offline stable plateau (0.5% IPC)", "offline_stable_plateau"),
        ("Live transition region", "live_transition_region"),
        ("Live stable plateau (0.5% IPC)", "live_stable_plateau"),
        ("Offline: minimum reaching at least 99% of 20M IPC", "smallest_budget_reaching_at_least_99_percent_of_20m_offline_ipc"),
        ("Offline: minimum reaching at least 99.5% of 20M IPC", "smallest_budget_reaching_at_least_99_5_percent_of_20m_offline_ipc"),
        ("Offline: minimum reaching at least 99.9% of 20M IPC", "smallest_budget_reaching_at_least_99_9_percent_of_20m_offline_ipc"),
        ("Live: minimum reaching at least 99% of 20M IPC", "smallest_budget_reaching_at_least_99_percent_of_20m_live_ipc"),
        ("Live: minimum reaching at least 99.5% of 20M IPC", "smallest_budget_reaching_at_least_99_5_percent_of_20m_live_ipc"),
        ("Live: minimum reaching at least 99.9% of 20M IPC", "smallest_budget_reaching_at_least_99_9_percent_of_20m_live_ipc"),
        ("Offline: minimum within 1.0% of 20M IPC", "smallest_within_1.0_percent_20m_offline_ipc"),
        ("Offline: minimum within 0.5% of 20M IPC", "smallest_within_0.5_percent_20m_offline_ipc"),
        ("Offline: minimum within 0.1% of 20M IPC", "smallest_within_0.1_percent_20m_offline_ipc"),
        ("Live: minimum within 1.0% of 20M IPC", "smallest_within_1.0_percent_20m_live_ipc"),
        ("Live: minimum within 0.5% of 20M IPC", "smallest_within_0.5_percent_20m_live_ipc"),
        ("Live: minimum within 0.1% of 20M IPC", "smallest_within_0.1_percent_20m_live_ipc"),
        ("Request pressure at live stable plateau", "request_pressure_at_live_stable_plateau"),
    ]
    lines = []
    for hidden in (8, 16):
        key = "h{}".format(hidden)
        lines.extend([
            r"\subsection*{" + key + "}",
            r"\begin{longtable}{@{}p{0.62\linewidth}p{0.30\linewidth}@{}}",
            r"\toprule",
            r"Conclusion & Result\\",
            r"\midrule",
            r"\endhead",
        ])
        for label, field in fields:
            lines.append("{} & {}\\\\".format(
                tex_escape(label),
                tex_escape(display_value(conclusions[key].get(field))),
            ))
        lines.extend([
            r"\bottomrule",
            r"\end{longtable}",
            r"\paragraph{High-performing small-budget candidates.} Offline: "
            + compact_candidates(conclusions[key].get(
                "offline_high_performing_small_budget_candidates"
            ))
            + r"; live: "
            + compact_candidates(conclusions[key].get(
                "live_high_performing_small_budget_candidates"
            )),
            "",
            r"\paragraph{Plateau evidence.} Offline: "
            + compact_plateau(conclusions[key].get("offline_stable_plateau"))
            + r"; live: "
            + compact_plateau(conclusions[key].get("live_stable_plateau")),
            "",
        ])
    lines.extend([
        r"\paragraph{Pareto points.} " + (
            ", ".join(
                "h{}--{}".format(row["hidden_size"], row["budget_tag"])
                for row in frontier
            ) if frontier else "NA---live runs required."
        ),
        r"\paragraph{h8 versus h16 at 20M.} "
        + tex_escape(conclusions.get("h8_compact_vs_h16")),
        r"\paragraph{Incremental h16 live benefit.} "
        + tex_escape(conclusions.get("h16_incremental_live_benefit")),
    ])
    path.write_text("\n".join(lines) + "\n")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser


def main():
    args = build_parser().parse_args()
    offline_data = load(args.run_dir / "offline_sweep_results.json")
    live_data = load(args.run_dir / "live_sweep_results.json")
    offline = offline_data["points"]
    live = [
        row for row in live_data["points"]
        if row.get("state_mode") == "parity"
        and row.get("live_run_mode") == "live"
    ]
    if not live:
        live = [
            row for row in live_data["points"]
            if row.get("state_mode") == "parity"
        ]
    live_by_key = {
        (row["hidden_size"], row["instruction_budget"], row["seed"]): row
        for row in live
    }
    joined = []
    for row in offline:
        key = (row["hidden_size"], row["instruction_budget"], row["seed"])
        live_row = live_by_key.get(key)
        combined = {
            "hidden_size": row["hidden_size"],
            "instruction_budget": row["instruction_budget"],
            "budget_tag": row["budget_tag"],
            "seed": row["seed"],
            "parameter_count": row.get("parameter_count"),
            "weight_bytes": row.get("weight_bytes"),
            "offline_ipc": row.get("ipc"),
            "offline_coverage": row.get("coverage"),
            "offline_request_pressure": row.get("requests_per_l2_load"),
            "live_ipc": live_row.get("ipc") if live_row else None,
            "live_coverage": live_row.get("coverage") if live_row else None,
            "live_request_pressure": (
                live_row.get("requests_per_l2_load") if live_row else None
            ),
            "live_minus_offline_ipc": (
                live_row["ipc"] - row["ipc"]
                if live_row and live_row.get("ipc") is not None
                and row.get("ipc") is not None else None
            ),
            "live_minus_offline_coverage": (
                live_row["coverage"] - row["coverage"]
                if live_row and live_row.get("coverage") is not None
                and row.get("coverage") is not None else None
            ),
            "live_minus_offline_request_pressure": (
                live_row["requests_per_l2_load"]
                - row["requests_per_l2_load"]
                if live_row
                and live_row.get("requests_per_l2_load") is not None
                and row.get("requests_per_l2_load") is not None else None
            ),
            "status": live_row.get("status") if live_row else row.get("status"),
            "failure_reason": (
                live_row.get("failure_reason")
                if live_row else row.get("failure_reason")
            ),
        }
        joined.append(combined)
    write_outputs(
        joined,
        args.run_dir / "offline_vs_live.csv",
        args.run_dir / "offline_vs_live.json",
        {
            "offline": offline_data.get("references", {}),
            "live": live_data.get("references", {}),
        },
    )
    frontier = pareto(live)
    write_outputs(
        frontier,
        args.run_dir / "pareto_frontier.csv",
        args.run_dir / "pareto_frontier.json",
    )
    grouped_offline = defaultdict(list)
    grouped_live = defaultdict(list)
    for row in offline:
        grouped_offline[row["hidden_size"]].append(row)
    for row in live:
        grouped_live[row["hidden_size"]].append(row)
    conclusions = {
        "h{}".format(hidden): conclusions_for_hidden(
            grouped_offline[hidden], grouped_live[hidden]
        )
        for hidden in (8, 16)
    }
    conclusions["schema_version"] = 2
    conclusions["analysis_policy"] = {
        "reaching_at_least": (
            "IPC_N >= requested_fraction * IPC_20M (one-sided)"
        ),
        "within": (
            "abs(IPC_N - IPC_20M) / abs(IPC_20M) <= tolerance "
            "(two-sided)"
        ),
        "stable_plateau": (
            "candidate and every subsequent observed IPC point through "
            "20M are within 0.5% of IPC_20M; at least two observed points "
            "are required"
        ),
        "auxiliary_stability": (
            "coverage, requests_per_l2_load, and student_heldout_act_rate "
            "are reported separately with a 5% relative diagnostic band"
        ),
        "missing_budget_rule": (
            "unrun or missing budgets are not counted as stable observations"
        ),
    }
    conclusions["pareto_points"] = [
        {
            "hidden_size": row["hidden_size"],
            "budget_tag": row["budget_tag"],
            "instruction_budget": row["instruction_budget"],
        }
        for row in frontier
    ]
    h8_live = conclusions["h8"].get("live_20m_ipc")
    h16_live = conclusions["h16"].get("live_20m_ipc")
    if h8_live is not None and h16_live not in (None, 0):
        relative_gap = (h16_live - h8_live) / h16_live
        conclusions["h8_compact_vs_h16"] = {
            "relative_live_ipc_gap": relative_gap,
            "h8_within_1_percent": abs(relative_gap) <= 0.01,
            "h8_weight_bytes": 7632,
            "h16_weight_bytes": 20880,
        }
        conclusions["h16_incremental_live_benefit"] = {
            "absolute_ipc": h16_live - h8_live,
            "relative_to_h8": (
                (h16_live - h8_live) / h8_live if h8_live else None
            ),
        }
    else:
        conclusions["h8_compact_vs_h16"] = None
        conclusions["h16_incremental_live_benefit"] = None
    conclusions["interpretation_guard"] = (
        "Winner selection requires match -> traffic -> timeliness -> "
        "coverage/miss rate -> IPC; loss or imitation accuracy alone is "
        "insufficient."
    )
    report_dir = args.run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "conclusions.json").write_text(
        json.dumps(conclusions, indent=2, sort_keys=True) + "\n"
    )
    write_conclusions_csv(report_dir / "conclusions.csv", conclusions)
    write_tex(
        report_dir / "generated_conclusions.tex", conclusions, frontier
    )
    print("[ok] offline/live comparison and Pareto frontier generated")


if __name__ == "__main__":
    main()
