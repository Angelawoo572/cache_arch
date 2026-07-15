#!/usr/bin/env python3
"""Fail-closed cross-directory comparison for standalone 623 LSTM/CNN runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PAIR_IDS = ("p0", "p1", "p2")
METRICS = (
    "ipc",
    "l2_load_miss_rate",
    "selected_accuracy",
    "coverage_vs_no_pref_l2_miss",
    "timeliness",
    "pollution_risk_proxy",
    "balanced_parity_index",
)
NORMAL_INVARIANTS = (
    "ipc",
    "instructions",
    "cycles",
    "l2_loads",
    "l2_load_miss",
    "pf_requested",
    "pf_issued",
    "pf_useful",
    "pf_useless",
    "pf_late",
)


def fail(message):
    raise RuntimeError(message)


def load_result(path, expected_family, policy):
    path = Path(path)
    result_path = path / "matched_comparison.json"
    if not result_path.is_file():
        fail("missing {}".format(result_path))
    payload = json.loads(result_path.read_text())
    if payload.get("status") != "PASS":
        fail("{} did not PASS".format(result_path))
    if not payload.get("fair_comparison_claim_allowed"):
        fail("{} does not permit a fair-comparison claim".format(result_path))
    if payload.get("model_family_track") != expected_family:
        fail("{} is not the {} track".format(result_path, expected_family))
    if "{}_track".format(policy) not in payload.get("primary_comparisons", {}):
        fail("{} is not the {} comparison".format(result_path, policy))
    return payload


def rows_by_method(payload):
    rows = payload.get("rows", [])
    mapping = {row.get("method"): row for row in rows}
    if len(mapping) != len(rows):
        fail("duplicate method in matched-comparison rows")
    return mapping


def recursive_hashes(value):
    if isinstance(value, dict):
        result = set()
        for child in value.values():
            result.update(recursive_hashes(child))
        return result
    if isinstance(value, list):
        result = set()
        for child in value:
            result.update(recursive_hashes(child))
        return result
    return {value} if isinstance(value, str) and len(value) == 64 else set()


def points_by_pair(payload, policy, family):
    track = payload.get("tracks", {}).get(policy, {})
    points = track.get("models", [])
    result = {}
    for point in points:
        pair_id = point.get("architecture_pair_id")
        if point.get("model_family") != family:
            fail("{} point leaked into {} track".format(
                point.get("model_family"), family
            ))
        if pair_id in result:
            fail("duplicate {} point {}".format(family, pair_id))
        result[pair_id] = point
    if set(result) != set(PAIR_IDS):
        fail("{} points {} != {}".format(
            family, sorted(result), list(PAIR_IDS)
        ))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("stride", "spp"), required=True)
    parser.add_argument("--lstm-run-dir", type=Path, required=True)
    parser.add_argument("--cnn-run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    lstm = load_result(args.lstm_run_dir, "lstm", args.policy)
    cnn = load_result(args.cnn_run_dir, "cnn", args.policy)
    if lstm.get("trace") != cnn.get("trace"):
        fail("LSTM/CNN traces differ")

    lstm_manifest = lstm.get("input_provenance", {}).get(
        "collection_manifest"
    )
    cnn_manifest = cnn.get("input_provenance", {}).get(
        "collection_manifest"
    )
    if not isinstance(lstm_manifest, dict) or lstm_manifest != cnn_manifest:
        fail("LSTM/CNN normalized input manifests are not byte-hash identical")
    if args.policy == "spp" and (
        lstm.get("input_provenance", {}).get("spp_source_contract_sha256")
        != cnn.get("input_provenance", {}).get("spp_source_contract_sha256")
    ):
        fail("LSTM/CNN SPP source contracts differ")

    lstm_hashes = recursive_hashes(
        lstm.get("offline_normal_list_hashes_by_model_tag", {})
    )
    cnn_hashes = recursive_hashes(
        cnn.get("offline_normal_list_hashes_by_model_tag", {})
    )
    if len(lstm_hashes) != 1 or lstm_hashes != cnn_hashes:
        fail("offline normal replay list is not identical across directories")

    lstm_rows = rows_by_method(lstm)
    cnn_rows = rows_by_method(cnn)
    normal_method = "offline_" + args.policy
    if normal_method not in lstm_rows or normal_method not in cnn_rows:
        fail("missing offline normal comparator row")
    normal_differences = {
        key: (lstm_rows[normal_method].get(key), cnn_rows[normal_method].get(key))
        for key in NORMAL_INVARIANTS
        if lstm_rows[normal_method].get(key) != cnn_rows[normal_method].get(key)
    }
    if normal_differences:
        fail("offline normal simulation differs: {}".format(normal_differences))

    lstm_points = points_by_pair(lstm, args.policy, "lstm")
    cnn_points = points_by_pair(cnn, args.policy, "cnn")
    rows = []
    for pair_id in PAIR_IDS:
        lstm_point = lstm_points[pair_id]
        cnn_point = cnn_points[pair_id]
        lstm_parameters = int(lstm_point["parameter_count"])
        cnn_parameters = int(cnn_point["parameter_count"])
        parameter_ratio = max(lstm_parameters, cnn_parameters) / float(
            min(lstm_parameters, cnn_parameters)
        )
        if parameter_ratio > 1.05:
            fail("{} parameter ratio {:.4f} exceeds 1.05".format(
                pair_id, parameter_ratio
            ))
        lstm_row = lstm_rows["offline_" + lstm_point["model_tag"]]
        cnn_row = cnn_rows["offline_" + cnn_point["model_tag"]]
        row = {
            "policy": args.policy,
            "architecture_pair_id": pair_id,
            "lstm_tag": lstm_point["model_tag"],
            "cnn_tag": cnn_point["model_tag"],
            "lstm_parameters": lstm_parameters,
            "cnn_parameters": cnn_parameters,
            "parameter_ratio": parameter_ratio,
        }
        for metric in METRICS:
            left = float(lstm_row[metric])
            right = float(cnn_row[metric])
            row["lstm_" + metric] = left
            row["cnn_" + metric] = right
            row["cnn_minus_lstm_" + metric] = right - left
        for field in (
            "behavior_count_exact_match_rate",
            "behavior_target_precision",
            "behavior_target_recall",
            "behavior_target_f1",
            "behavior_fill_accuracy_on_matched_targets",
        ):
            left = lstm_row.get(field, "")
            right = cnn_row.get(field, "")
            row["lstm_" + field] = left
            row["cnn_" + field] = right
            if left != "" and right != "" and left is not None and right is not None:
                row["cnn_minus_lstm_" + field] = float(right) - float(left)
            else:
                row["cnn_minus_lstm_" + field] = ""
        row["ipc_winner"] = (
            "cnn" if row["cnn_minus_lstm_ipc"] > 0 else "lstm"
            if row["cnn_minus_lstm_ipc"] < 0 else "tie"
        )
        row["balanced_parity_winner"] = (
            "cnn" if row["cnn_minus_lstm_balanced_parity_index"] > 0
            else "lstm" if row["cnn_minus_lstm_balanced_parity_index"] < 0
            else "tie"
        )
        rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "{}_lstm_vs_cnn.csv".format(args.policy)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "status": "PASS",
        "fair_cross_directory_comparison_allowed": True,
        "trace": lstm["trace"],
        "policy": args.policy,
        "input_manifest_exact_match": True,
        "offline_normal_list_exact_match": True,
        "offline_normal_simulation_exact_match": True,
        "architecture_contract": {
            "lstm": "complete chronological stateful history",
            "cnn": {
                "temporal_layers": 4,
                "kernel_size": 7,
                "dilations": [1, 6, 36, 216],
                "contiguous_receptive_field_events": 1555,
            },
        },
        "interpretation": {
            "cnn_wins": (
                "Local-to-medium causal correlations within 1,555 callbacks "
                "are sufficient at matched model size."
            ),
            "lstm_wins": (
                "Useful predictive state extends beyond or is represented "
                "more efficiently than the finite CNN history."
            ),
            "both_below_normal": (
                "The input restriction or behavior-cloning objective, not "
                "merely recurrent versus convolutional architecture, is the "
                "dominant limitation."
            ),
        },
        "rows": rows,
    }
    json_path = args.out_dir / "{}_lstm_vs_cnn.json".format(args.policy)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("[PASS] {}".format(json_path))
    print("[PASS] {}".format(csv_path))


if __name__ == "__main__":
    main()
