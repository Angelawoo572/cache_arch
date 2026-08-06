#!/usr/bin/env python3
"""Build a fail-closed diagnosis of a completed 623 SPP run.

This script reads existing analyzer and Colab metadata only.  It never changes
training, replay lists, or simulator outputs.  The output keeps request-count,
target-quality, and cache-lifecycle evidence separate so a PASS contract is not
mistaken for a successful neural policy.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from model_contract import (
    EXTERNAL_INPUT_FIELDS, MODEL_REVISION as ACTIVE_MODEL_REVISION, POLICY,
    RUN_ID as DEFAULT_RUN_ID, TRACE, describe_model_points,
)

V15_MODEL_REVISION = "compact_crn_joint_delta_fill_mixture_v15"
V16A_MODEL_REVISION = "compact_crn_joint_delta_fill_guard_map_v16a"
V17_MODEL_REVISION = "compact_crn_factorized_delta_keyed_fill_v17"
V18_MODEL_REVISION = "compact_crn_hard_distinct_delta_keyed_fill_v18"
ACTIVE_POINT_CONTRACT = describe_model_points()
REVISION_PROFILES = {
    V15_MODEL_REVISION: (
        "offline_joint_delta_fill_spp_lstm_",
        "joint_delta_fill_spp_lstm_h{}",
    ),
    V16A_MODEL_REVISION: (
        "offline_guard_joint_map_spp_lstm_",
        "guard_joint_map_spp_lstm_h{}",
    ),
    V17_MODEL_REVISION: (
        "offline_factorized_delta_fill_spp_lstm_",
        "factorized_delta_fill_spp_lstm_h{}",
    ),
    V18_MODEL_REVISION: (
        "offline_hard_distinct_delta_fill_spp_lstm_",
        "hard_distinct_delta_fill_spp_lstm_h{}",
    ),
    ACTIVE_MODEL_REVISION: (
        "offline_finite_joint_rank_spp_lstm_",
        "finite_joint_rank_spp_lstm_h{}",
    ),
}
SOURCE_INPUTS = list(EXTERNAL_INPUT_FIELDS)
EXPERIMENT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]


def div(numerator, denominator):
    return (
        float(numerator) / float(denominator)
        if denominator not in (None, 0) else None
    )


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def flatten(prefix, value, output):
    if isinstance(value, dict):
        for key in sorted(value):
            child = "{}.{}".format(prefix, key) if prefix else str(key)
            flatten(child, value[key], output)
    elif isinstance(value, (list, tuple)):
        output[prefix] = json.dumps(value, sort_keys=True)
    elif value is None:
        output[prefix] = ""
    else:
        output[prefix] = value


def input_contract_mismatches(metadata):
    expected = {
        "same_external_input_contract": True,
        "training_inference_input_encoder_identical": True,
        "training_runtime_fields": SOURCE_INPUTS,
        "inference_runtime_fields": SOURCE_INPUTS,
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "normal_policy_request_rate_used_as_budget": False,
        "future_label_window_used": False,
        "teacher_actions_are_model_inputs": False,
        "model_does_not_use_pc": True,
        "pc_is_replay_transport_only": True,
        "closed_loop_live_claim_allowed": False,
        "decoder_training_mode": ACTIVE_POINT_CONTRACT["decoder_training_mode"],
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_previous_predicted_action_used_as_input": False,
        "decoder_previous_sampled_action_used_as_input": False,
        "terminal_stop_supervised_for_every_teacher_sequence": False,
        "all_available_tail_stop_supervised": True,
        "maximum_length_sequences_terminate_by_finite_support": True,
        "separate_gate_head_used": False,
        "request_count_head_used": False,
        "request_count_regression_used": False,
        "separate_delta_head_used": False,
        "separate_fill_head_used": False,
        "fill_argmax_used": False,
        "stop_emit_head_used": False,
        "joint_action_training_objective": ACTIVE_POINT_CONTRACT[
            "joint_action_training_objective"
        ],
        "other_delta_training_objective": ACTIVE_POINT_CONTRACT[
            "other_delta_training_objective"
        ],
        "joint_action_prior_correction_at_decode_used": True,
        "joint_action_prior_correction_rule": ACTIVE_POINT_CONTRACT[
            "joint_action_prior_correction_rule"
        ],
        "finite_output_horizon_source": ACTIVE_POINT_CONTRACT[
            "finite_output_horizon_source"
        ],
        "finite_output_horizon_is_dataset_derived": True,
        "finite_output_horizon_is_normal_request_budget": False,
        "finite_output_horizon_is_tuned_degree": False,
        "neural_degree_cap": None,
        "probability_threshold_used": False,
        "inference_policy_hardcodes_used": False,
        "stochastic_decoding": False,
        "guard_selection_rule": ACTIVE_POINT_CONTRACT["guard_selection_rule"],
        "guard_selection_composite_or_mean_used": False,
        "delta_other_escape": ACTIVE_POINT_CONTRACT["delta_other_escape"],
        "full_signed_line_delta_range_reachable": False,
        "every_signed_line_delta_exactly_representable": False,
        "exact_delta_representability_scope": "train_vocabulary_only",
        "non_neural_control_uses_model": False,
        "non_neural_control_excluded_from_neural_claims": True,
        "cublas_workspace_config": ":4096:8",
        "torch_deterministic_algorithms_enabled": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "float32_matmul_precision": "highest",
        "determinism_fail_closed": True,
    }
    expected.update(ACTIVE_POINT_CONTRACT["training_config"])
    mismatches = []
    for key, value in expected.items():
        if metadata.get(key) != value:
            mismatches.append({
                "field": key,
                "actual": metadata.get(key),
                "expected": value,
            })
    if metadata.get("training_config") != ACTIVE_POINT_CONTRACT["training_config"]:
        mismatches.append({
            "field": "training_config",
            "actual": metadata.get("training_config"),
            "expected": ACTIVE_POINT_CONTRACT["training_config"],
        })
    horizon = metadata.get("train_action_horizon")
    if (
        not isinstance(horizon, int)
        or isinstance(horizon, bool)
        or horizon < 1
        or metadata.get("joint_decision_rank_count") != horizon
        or (metadata.get("teacher_max_actions_per_callback") or {}).get("train")
        != horizon
    ):
        mismatches.append({
            "field": "train_action_horizon",
            "actual": horizon,
            "expected": "positive TRAIN maximum used as exact finite support",
        })
    if "A100" not in str(metadata.get("cuda_device_name", "")):
        mismatches.append({
            "field": "cuda_device_name",
            "actual": metadata.get("cuda_device_name"),
            "expected": "NVIDIA A100",
        })
    provenance_paths = {
        "trainer_source_sha256": EXPERIMENT / "python" / "train_and_offline_infer.py",
        "model_contract_source_sha256": EXPERIMENT / "python" / "model_contract.py",
        "threshold_free_policy_source_sha256": (
            REPO_ROOT / "formal_NN_training" / "common" / "threshold_free_policy.py"
        ),
    }
    for key, path in provenance_paths.items():
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if metadata.get(key) != observed:
            mismatches.append({
                "field": key,
                "actual": metadata.get(key),
                "expected": observed,
            })
    encoder_hashes = {
        metadata.get("runtime_encoder_sha256"),
        metadata.get("training_runtime_encoder_sha256"),
        metadata.get("inference_runtime_encoder_sha256"),
    }
    if (
        len(encoder_hashes) != 1
        or not isinstance(next(iter(encoder_hashes)), str)
        or len(next(iter(encoder_hashes))) != 64
    ):
        mismatches.append({
            "field": "runtime_encoder_sha256",
            "actual": sorted(str(value) for value in encoder_hashes),
            "expected": "one identical 64-hex hash",
        })
    return mismatches


def analyzer_evidence_mismatches(matched, tag, metadata):
    """Bind current metadata and fill-list totals to analyzer evidence."""
    mismatches = []
    method = "offline_" + tag
    accounting = matched.get("replay_accounting") or {}
    nn_info = accounting.get(method) or {}
    normal_info = accounting.get("offline_spp") or {}
    control_info = accounting.get("offline_modal_llc_control") or {}
    checks = (
        ("runtime_encoder_sha256", metadata.get("runtime_encoder_sha256"),
         (matched.get("runtime_encoder_sha256_by_model_tag") or {}).get(tag)),
        ("normal_list_sha256", metadata.get("normal_list_sha256"),
         (matched.get("offline_normal_list_hashes_by_model_tag") or {})
         .get(POLICY, {}).get(tag)),
        ("normal_list_sha256_vs_replay", metadata.get("normal_list_sha256"),
         normal_info.get("sha256")),
        ("offline_normal_entries", metadata.get("offline_normal_entries"),
         normal_info.get("entries")),
        ("offline_normal_fill_level_counts",
         metadata.get("offline_normal_fill_level_counts"),
         normal_info.get("fill_counts")),
        ("nn_list_sha256", metadata.get("nn_list_sha256"),
         nn_info.get("sha256")),
        ("offline_nn_entries", metadata.get("offline_nn_entries"),
         nn_info.get("entries")),
        ("offline_nn_fill_level_counts",
         metadata.get("offline_nn_fill_level_counts"),
         nn_info.get("fill_counts")),
        ("non_neural_control_list_sha256",
         metadata.get("non_neural_control_list_sha256"),
         control_info.get("sha256")),
        ("non_neural_control_entries",
         metadata.get("non_neural_control_entries"),
         control_info.get("entries")),
        ("source_contract_sha256", metadata.get("source_contract_sha256"),
         (matched.get("input_provenance") or {}).get(
             "spp_source_contract_sha256"
         )),
    )
    for field, actual, expected in checks:
        if expected is None or actual != expected:
            mismatches.append({
                "field": field, "actual": actual, "expected": expected,
            })
    policy_inputs = (
        (matched.get("input_provenance") or {})
        .get("policy_inputs", {}).get(POLICY, {})
    )
    for role in ("train", "guard", "eval"):
        for kind in ("stream", "teacher_actions"):
            key = "{}_{}_content_sha256".format(role, kind)
            expected = policy_inputs.get(role, {}).get(kind, {}).get(
                "content_sha256"
            )
            if expected is None or metadata.get(key) != expected:
                mismatches.append({
                    "field": key,
                    "actual": metadata.get(key),
                    "expected": expected,
                })
    return mismatches


def model_record(row, metadata, normal, no_pref, matched):
    triggers = row.get("runtime_reachable_list_triggers")
    normal_triggers = normal.get("runtime_reachable_list_triggers")
    record = {
        "method": row.get("method"),
        "model_tag": metadata.get("model_tag"),
        "model_size": metadata.get("model_size"),
        "parameter_count": metadata.get("parameter_count"),
        "ipc": row.get("ipc"),
        "ipc_delta_vs_offline_normal": (
            row.get("ipc") - normal.get("ipc")
        ),
        "ipc_delta_vs_no_pref": row.get("ipc") - no_pref.get("ipc"),
        "l2_load_miss_rate": row.get("l2_load_miss_rate"),
        "l2_miss_rate_delta_vs_offline_normal": (
            row.get("l2_load_miss_rate")
            - normal.get("l2_load_miss_rate")
        ),
        "pf_requested": row.get("pf_requested"),
        "request_ratio_vs_offline_normal": div(
            row.get("pf_requested"), normal.get("pf_requested")
        ),
        "runtime_reachable_list_triggers": triggers,
        "trigger_ratio_vs_offline_normal": div(
            triggers, normal_triggers
        ),
        "actions_per_reached_trigger": div(
            row.get("pf_requested"), triggers
        ),
        "offline_normal_actions_per_reached_trigger": div(
            normal.get("pf_requested"), normal_triggers
        ),
        "pf_filled": row.get("pf_filled"),
        "pf_useful": row.get("pf_useful"),
        "pf_useless": row.get("pf_useless"),
        "pf_late": row.get("pf_late"),
        "pq_merged_duplicate_proxy": row.get(
            "pq_merged_duplicate_proxy"
        ),
        "selected_accuracy": row.get("selected_accuracy"),
        "coverage_vs_no_pref_l2_miss": row.get(
            "coverage_vs_no_pref_l2_miss"
        ),
        "timeliness": row.get("timeliness"),
        "l2_quality_metric_status": row.get("l2_quality_metric_status"),
        "l2_selected_accuracy_semantic": row.get(
            "l2_selected_accuracy_semantic"
        ),
        "l2_coverage_semantic": row.get("l2_coverage_semantic"),
        "l2_timeliness_semantic": row.get("l2_timeliness_semantic"),
        "runtime_unreached_list_entries": row.get(
            "runtime_unreached_list_entries"
        ),
        "runtime_unreached_list_triggers": row.get(
            "runtime_unreached_list_triggers"
        ),
        "prefetch_fill_l2_events": row.get("prefetch_fill_l2_events"),
        "prefetch_fill_llc_events": row.get("prefetch_fill_llc_events"),
        "event_l2_fill_fraction": div(
            row.get("prefetch_fill_l2_events"),
            (row.get("prefetch_fill_l2_events") or 0)
            + (row.get("prefetch_fill_llc_events") or 0),
        ),
    }
    diagnostic_keys = (
        "heldout_behavior_metrics",
        "decoder_training_mode",
        "decoder_previous_teacher_action_used_as_input",
        "decoder_previous_predicted_action_used_as_input",
        "decoder_previous_sampled_action_used_as_input",
        "joint_action_training_objective",
        "other_delta_training_objective",
        "joint_action_token_definition",
        "joint_action_token_count",
        "joint_action_train_token_counts",
        "joint_action_train_group_counts",
        "joint_action_train_group_weights",
        "joint_action_prior_correction_at_decode_used",
        "joint_action_prior_correction_rule",
        "train_action_horizon",
        "joint_decision_rank_count",
        "finite_output_horizon_source",
        "all_available_tail_stop_supervised",
        "maximum_length_sequences_terminate_by_finite_support",
        "hurdle_training_objective",
        "hurdle_train_class_counts",
        "hurdle_train_class_priors",
        "hurdle_class_weighting_used",
        "positive_count_training_objective",
        "positive_count_train_samples",
        "positive_count_train_min",
        "positive_count_train_max",
        "positive_log_count_train_mean",
        "positive_log_count_train_std",
        "positive_count_decoding_rule",
        "terminal_stop_supervised_for_every_teacher_sequence",
        "separate_gate_head_used",
        "request_count_head_used",
        "request_count_regression_used",
        "delta_training_objective",
        "delta_decoding_rule",
        "delta_vocabulary_source",
        "exact_delta_vocabulary_size",
        "delta_vocabulary_statistics",
        "delta_other_escape",
        "delta_other_decode_precision",
        "duplicate_target_handling",
        "fill_training_objective",
        "fill_train_class_counts",
        "fill_train_priors",
        "fill_train_inverse_frequency_weights",
        "fill_prior_correction_at_decode_used",
        "fill_prior_correction_rule",
        "fill_decoding_rule",
        "fill_conditioned_on_actual_emitted_target",
        "fill_argmax_used",
        "fill_target_conditioning_features",
        "decoder_guard_diagnostics",
        "decoder_eval_diagnostics",
        "global_chronological_lstm",
        "routed_demand_fill_recurrent_paths",
        "page_local_causal_state",
        "dynamic_page_state_pages",
        "peak_persistent_recurrent_state_bytes",
        "action_output_diagnostics",
        "raw_predicted_action_count",
        "materialized_action_count",
        "full_signed_line_delta_range_reachable",
        "every_signed_line_delta_exactly_representable",
        "exact_delta_representability_scope",
        "guard_selection_rule",
        "guard_selection_key_fields",
        "guard_selection_key",
        "guard_selection_metrics",
        "guard_selection_composite_or_mean_used",
        "selected_epoch",
        "offline_normal_fill_level_counts",
        "offline_nn_fill_level_counts",
        "decoder_revision",
        "weights_retrained",
        "checkpoint_reused",
        "non_neural_control_name",
        "non_neural_control_modal_delta",
        "non_neural_control_modal_delta_train_frequency",
        "non_neural_control_list_sha256",
    )
    for key in diagnostic_keys:
        if key in metadata:
            record[key] = metadata[key]
    history = metadata.get("train_history") or []
    if history:
        record["last_training_epoch"] = history[-1]
    mismatches = input_contract_mismatches(metadata)
    record["input_contract_verified"] = not mismatches
    record["input_contract_mismatches"] = mismatches
    evidence_mismatches = analyzer_evidence_mismatches(
        matched, metadata.get("model_tag"), metadata
    )
    record["analyzer_evidence_verified"] = not evidence_mismatches
    record["analyzer_evidence_mismatches"] = evidence_mismatches
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    run_dir = (
        args.run_dir
        if args.run_dir is not None
        else EXPERIMENT / "runs" / args.run_id
    )

    matched_path = run_dir / "matched_comparison.json"
    if not matched_path.is_file():
        raise SystemExit("missing {}".format(matched_path))
    matched = read_json(matched_path)
    if matched.get("status") != "PASS" or matched.get("failures"):
        raise SystemExit(
            "source matched comparison is not a root PASS with no failures"
        )
    if matched.get("trace") != TRACE:
        raise SystemExit("unexpected trace in {}".format(matched_path))
    model_revision = matched.get("model_revision", V15_MODEL_REVISION)
    profile = REVISION_PROFILES.get(model_revision)
    if profile is None:
        raise SystemExit("unsupported SPP model revision {}".format(
            model_revision
        ))
    neural_method_prefix, tag_template = profile
    model_sizes = tuple(
        point["size"] for point in ACTIVE_POINT_CONTRACT["points"]
    ) if model_revision == ACTIVE_MODEL_REVISION else (
        8, 16, 32, 64, 128
    )
    expected_model_tags = {tag_template.format(size) for size in model_sizes}

    rows = {
        row.get("method"): row for row in matched.get("rows", [])
    }
    no_pref = rows.get("no_pref")
    normal = rows.get("offline_" + POLICY)
    if no_pref is None or normal is None:
        raise SystemExit("missing no-prefetch or offline-normal row")

    expected_tags = {
        method[len("offline_"):]
        for method in rows
        if isinstance(method, str)
        and method.startswith(neural_method_prefix)
    }
    if not expected_tags:
        raise SystemExit("matched comparison contains no SPP neural rows")
    if expected_tags != expected_model_tags:
        raise SystemExit(
            "unexpected SPP model set: observed={} expected={}".format(
                sorted(expected_tags), sorted(expected_model_tags)
            )
        )

    records = []
    selected_metadata = []
    metadata_root = run_dir / "colab_output"
    for metadata_path in sorted(metadata_root.glob("*/run_metadata.json")):
        metadata = read_json(metadata_path)
        tag = metadata.get("model_tag")
        if not tag or metadata.get("matched_normal_prefetcher") != POLICY:
            continue
        row = rows.get("offline_" + tag)
        if row is None:
            raise SystemExit("missing replay row for {}".format(tag))
        records.append(model_record(row, metadata, normal, no_pref, matched))
        selected_metadata.append(metadata)
    if not records:
        raise SystemExit("no SPP neural metadata found")
    observed_tags = {record["model_tag"] for record in records}
    if observed_tags != expected_tags:
        raise SystemExit(
            "SPP metadata/replay model set mismatch: observed={} expected={}".format(
                sorted(observed_tags), sorted(expected_tags)
            )
        )

    encoder_hashes = {
        metadata.get("runtime_encoder_sha256")
        for metadata in selected_metadata
    }
    contract_verified = (
        all(record["input_contract_verified"] for record in records)
        and all(record["analyzer_evidence_verified"] for record in records)
        and len(encoder_hashes) == 1
    )
    if not contract_verified:
        problems = [{
            "model_tag": record["model_tag"],
            "input_contract_mismatches": record["input_contract_mismatches"],
            "analyzer_evidence_mismatches": record[
                "analyzer_evidence_mismatches"
            ],
        } for record in records if (
            not record["input_contract_verified"]
            or not record["analyzer_evidence_verified"]
        )]
        raise SystemExit(
            "cross-model evidence verification failed: {}".format(
                json.dumps(problems, sort_keys=True)
            )
        )

    best = max(records, key=lambda item: item["ipc"])
    payload = {
        "status": "PASS",
        "source_matched_comparison": str(matched_path),
        "source_matched_comparison_status": matched.get("status"),
        "trace": TRACE,
        "policy": POLICY,
        "model_revision": model_revision,
        "input_contract_verified": True,
        "current_metadata_bound_to_analyzer_evidence": True,
        "cross_capacity_runtime_encoder_identical": True,
        "runtime_encoder_sha256": next(iter(encoder_hashes)),
        "same_external_input": SOURCE_INPUTS,
        "capacity_action_list_audit": matched.get(
            "capacity_action_list_audit"
        ),
        "metric_applicability": matched.get("metric_applicability"),
        "diagnostic_control": (
            rows.get("offline_modal_llc_control")
        ),
        "teacher_role": (
            "captured SPP actions and fill choices are supervision and "
            "comparator replay only; they are not neural inference inputs"
        ),
        "no_pref_ipc": no_pref.get("ipc"),
        "offline_normal_ipc": normal.get("ipc"),
        "offline_normal_ipc_gain_vs_no_pref": (
            normal.get("ipc") - no_pref.get("ipc")
        ),
        "offline_normal_requests": normal.get("pf_requested"),
        "offline_normal_reached_triggers": normal.get(
            "runtime_reachable_list_triggers"
        ),
        "best_observed_neural_model_by_ipc": best["model_tag"],
        "best_observed_neural_ipc": best["ipc"],
        "best_neural_ipc_delta_vs_offline_normal": (
            best["ipc_delta_vs_offline_normal"]
        ),
        "any_neural_model_beats_offline_normal": any(
            record["ipc_delta_vs_offline_normal"] > 0
            for record in records
        ),
        "all_neural_models_lose_to_offline_normal": all(
            record["ipc_delta_vs_offline_normal"] <= 0
            for record in records
        ),
        "models": records,
        "interpretation_guardrails": [
            "PASS validates accounting and input fairness, not NN success.",
            "Request volume, target quality, miss rate, and IPC remain separate.",
            "Higher selected accuracy or coverage does not by itself prove "
            "better cache behavior.",
            "Replay-list fill totals and runtime issued-event fill counts are "
            "different accounting domains.",
            "L2-oriented quality ratios are N/A, not zero, when a method emits "
            "no FILL_L2 actions.",
            "The modal-LLC control is non-neural and excluded from neural claims.",
            "This schema does not directly measure harmful victim eviction.",
            "One trace and one seed identify a best observed point, not an optimum.",
        ],
    }

    json_path = run_dir / "model_diagnosis.json"
    csv_path = run_dir / "model_diagnosis.csv"
    atomic_json(json_path, payload)

    flat_records = []
    for record in records:
        flat = {}
        flatten("", record, flat)
        flat_records.append(flat)
    preferred = [
        "method", "model_tag", "model_size", "parameter_count",
        "ipc", "ipc_delta_vs_offline_normal", "ipc_delta_vs_no_pref",
        "l2_load_miss_rate", "l2_miss_rate_delta_vs_offline_normal",
        "pf_requested", "request_ratio_vs_offline_normal",
        "runtime_reachable_list_triggers",
        "trigger_ratio_vs_offline_normal",
        "actions_per_reached_trigger",
        "offline_normal_actions_per_reached_trigger",
        "pf_filled", "pf_useful", "pf_useless", "pf_late",
        "selected_accuracy", "coverage_vs_no_pref_l2_miss", "timeliness",
        "l2_quality_metric_status", "l2_selected_accuracy_semantic",
        "l2_coverage_semantic", "l2_timeliness_semantic",
        "prefetch_fill_l2_events", "prefetch_fill_llc_events",
        "event_l2_fill_fraction", "input_contract_verified",
    ]
    remaining = sorted({
        key for record in flat_records for key in record
        if key not in preferred
    })
    fields = preferred + remaining
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({
            key: record.get(key, "") for key in fields
        } for record in flat_records)

    print("[PASS] {}".format(json_path))
    print("[PASS] {}".format(csv_path))
    print(
        "best_observed={} ipc={:.6f} delta_vs_offline={:.6f}".format(
            best["model_tag"], best["ipc"],
            best["ipc_delta_vs_offline_normal"],
        )
    )


if __name__ == "__main__":
    main()
