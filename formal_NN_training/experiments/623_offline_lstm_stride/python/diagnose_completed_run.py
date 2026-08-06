#!/usr/bin/env python3
"""Build a fail-closed diagnosis of a completed 623 Stride run.

This script reads existing analyzer and Colab metadata only.  It never changes
training, replay lists, or simulator outputs.  The output keeps request-count,
target-quality, and cache-lifecycle evidence separate so a PASS contract is not
mistaken for a successful neural policy.
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from model_contract import POLICY, RUN_ID, TRACE, model_points_description

DEFAULT_RUN_ID = RUN_ID
SOURCE_INPUTS = ["pc", "addr"]
NEURAL_METHOD_PREFIX = "offline_natural_cardinality_stride_lstm_"
EXPECTED_TAGS = {
    point["model_tag"] for point in model_points_description()["points"]
}
EXPERIMENT = Path(__file__).resolve().parents[1]
REPOSITORY = Path(__file__).resolve().parents[4]
STRIDE_SOURCE = REPOSITORY / "external" / "ChampSim" / "prefetcher" / "stride.cc"


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


def input_contract_mismatches_v23_legacy(metadata):
    contract = model_points_description()
    source_hashes = {
        "trainer_source_sha256": hashlib.sha256(
            (EXPERIMENT / "python" / "train_and_offline_infer.py").read_bytes()
        ).hexdigest(),
        "redecoder_source_sha256": hashlib.sha256(
            (EXPERIMENT / "python" / "redecode_prior_corrected.py").read_bytes()
        ).hexdigest(),
        "model_contract_source_sha256": hashlib.sha256(
            (EXPERIMENT / "python" / "model_contract.py").read_bytes()
        ).hexdigest(),
        "threshold_free_policy_source_sha256": hashlib.sha256(
            (
                REPOSITORY / "formal_NN_training" / "common"
                / "threshold_free_policy.py"
            ).read_bytes()
        ).hexdigest(),
    }
    expected = {
        "same_external_input_contract": True,
        "training_inference_input_encoder_identical": True,
        "training_runtime_fields": SOURCE_INPUTS,
        "inference_runtime_fields": SOURCE_INPUTS,
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "teacher_actions_are_model_inputs": False,
        "future_label_window_used": False,
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_previous_predicted_action_used_as_input": False,
        "decoder_previous_sampled_action_used_as_input": False,
        "decoder_training_mode": contract["decoder_training_mode"],
        "terminal_stop_supervised_for_every_teacher_sequence": False,
        "hurdle_training_objective": contract[
            "hurdle_training_objective"
        ],
        "hurdle_equal_aggregate_train_mass": True,
        "hurdle_decoding_rule": (
            "deterministic_prior_corrected_two_class_argmax"
        ),
        "hurdle_prior_correction_at_decode_used": True,
        "hurdle_prior_correction_rule": (
            "weighted_logits_minus_log_TRAIN_inverse_frequency_class_weight"
        ),
        "separate_global_gate_used": True,
        "separate_count_head_used": True,
        "log_count_used": True,
        "positive_count_training_objective": contract[
            "positive_count_training_objective"
        ],
        "positive_count_support": "mathematically_unbounded_positive_integers",
        "positive_count_host_behavior": "fail_closed_no_clip_or_wrap",
        "runtime_encoding": "lossless_raw_pc64_plus_line58_only",
        "engineered_runtime_features": [],
        "causal_runtime_feature_count": 0,
        "normal_policy_templates_used_by_neural_inference": False,
        "probability_threshold_used": False,
        "inference_policy_hardcodes_used": False,
        "neural_degree_cap": None,
        "fixed_page_offset_classes": None,
        "same_page_rule_used_by_neural_inference": False,
        "delta_vocabulary_source": "train_labels_only",
        "delta_vocabulary_max_exact": 255,
        "delta_other_escape": "signed_log_continuous_bounded_approximation",
        "delta_other_decode_precision": "rounded_float32_approximate_except_exact_vocabulary",
        "full_signed_line_delta_range_reachable": False,
        "every_signed_line_delta_exactly_representable": False,
        "exact_delta_representability_scope": "train_vocabulary_only",
        "delta_coordinate_auxiliary_trained_on_all_teacher_actions": True,
        "delta_coordinate_used_for_decode_only_on_other": True,
        "decode_resource_watchdog_is_neural_degree_cap": False,
        "successful_run_hit_decode_resource_watchdog": False,
        "checkpoint_selection": contract["checkpoint_selection"],
        "checkpoint_selection_roles": [
            "parent_v22_guard_selection"
        ],
        "guard_selection_composite_or_mean_used": False,
        "evaluation_used_for_checkpoint_selection": False,
        "evaluation_decode_passes": 2,
        "parent_raw_reproduction_decode_passes": 1,
        "prior_corrected_evaluation_decode_passes": 1,
        "weights_retrained": False,
        "checkpoint_reused": True,
        "training_history_reused": True,
        "decoder_only_change": True,
        "parent_artifact_identity_required": True,
        "parent_raw_reproduction_matches": True,
        "training_config": contract["training_config"],
        "training_config_pinned_by_run_id": True,
        "training_device": "cuda",
        "cublas_workspace_config": contract["determinism_contract"][
            "cublas_workspace_config"
        ],
        "torch_deterministic_algorithms_enabled": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "float32_matmul_precision": contract["determinism_contract"][
            "float32_matmul_precision"
        ],
    }
    expected.update(source_hashes)
    mismatches = []
    for key, value in expected.items():
        if metadata.get(key) != value:
            mismatches.append({
                "field": key,
                "actual": metadata.get(key),
                "expected": value,
            })
    accelerator = contract["determinism_contract"][
        "required_accelerator_name_contains"
    ]
    if accelerator not in str(metadata.get("training_device_name")):
        mismatches.append({
            "field": "training_device_name",
            "actual": metadata.get("training_device_name"),
            "expected": "contains {}".format(accelerator),
        })
    if accelerator not in str(metadata.get("redecode_device_name")):
        mismatches.append({
            "field": "redecode_device_name",
            "actual": metadata.get("redecode_device_name"),
            "expected": "contains {}".format(accelerator),
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


def input_contract_mismatches(metadata):
    contract = model_points_description()
    source_hashes = {
        "trainer_source_sha256": hashlib.sha256(
            (EXPERIMENT / "python" / "train_and_offline_infer.py").read_bytes()
        ).hexdigest(),
        "model_contract_source_sha256": hashlib.sha256(
            (EXPERIMENT / "python" / "model_contract.py").read_bytes()
        ).hexdigest(),
        "threshold_free_policy_source_sha256": hashlib.sha256(
            (
                REPOSITORY / "formal_NN_training" / "common"
                / "threshold_free_policy.py"
            ).read_bytes()
        ).hexdigest(),
    }
    expected = {
        "same_external_input_contract": True,
        "training_inference_input_encoder_identical": True,
        "training_runtime_fields": SOURCE_INPUTS,
        "inference_runtime_fields": SOURCE_INPUTS,
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "teacher_actions_are_model_inputs": False,
        "future_label_window_used": False,
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_previous_predicted_action_used_as_input": False,
        "decoder_previous_sampled_action_used_as_input": False,
        "decoder_training_mode": contract["decoder_training_mode"],
        "count_training_objective": contract["count_training_objective"],
        "categorical_count_head_used": True,
        "count_regression_used": False,
        "log_count_used": False,
        "hurdle_head_used": False,
        "separate_global_gate_used": False,
        "separate_count_head_used": False,
        "stop_padding_used": False,
        "loss_class_reweighting_used": False,
        "decode_prior_correction_used": False,
        "count_zero_is_implicit_hurdle": True,
        "count_support_is_dataset_derived": True,
        "count_support_is_normal_request_budget": False,
        "count_support_is_tuned_degree": False,
        "runtime_encoding": "lossless_raw_pc64_plus_line58_only",
        "engineered_runtime_features": [],
        "causal_runtime_feature_count": 0,
        "normal_policy_templates_used_by_neural_inference": False,
        "probability_threshold_used": False,
        "inference_policy_hardcodes_used": False,
        "neural_degree_cap": None,
        "fixed_page_offset_classes": None,
        "same_page_rule_used_by_neural_inference": False,
        "delta_training_objective": contract[
            "delta_training_objective"
        ],
        "delta_other_escape": "signed_log_continuous_bounded_approximation",
        "delta_coordinate_auxiliary_scope": "OTHER_teacher_actions_only",
        "all_deltas_relative_to_current_demand": True,
        "stride_fill_level": "FILL_L2_only_no_learned_fill_head",
        "action_loss_scope": "teacher_action_ranks_only",
        "checkpoint_selection": contract["checkpoint_selection"],
        "blocked_validation_length_source": contract[
            "blocked_validation_length_source"
        ],
        "original_guard_role": contract["original_guard_role"],
        "blocked_validation_selected_checkpoint": True,
        "original_guard_used_for_checkpoint_selection": False,
        "evaluation_used_for_checkpoint_selection": False,
        "evaluation_policy_decode_count": 1,
        "diagnostic_eval_decode_count": 1,
        "oracle_diagnostics_replayed": False,
        "oracle_diagnostics_excluded_from_fair_claims": True,
        "weights_retrained": True,
        "checkpoint_reused": False,
        "decoder_only_change": False,
        "training_config": contract["training_config"],
        "cublas_workspace_config": contract["determinism_contract"][
            "cublas_workspace_config"
        ],
        "torch_deterministic_algorithms_enabled": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "float32_matmul_precision": "highest",
    }
    expected.update(source_hashes)
    mismatches = []
    for key, value in expected.items():
        if metadata.get(key) != value:
            mismatches.append({
                "field": key,
                "actual": metadata.get(key),
                "expected": value,
            })
    accelerator = contract["determinism_contract"][
        "required_accelerator_name_contains"
    ]
    if accelerator not in str(metadata.get("training_device_name")):
        mismatches.append({
            "field": "training_device_name",
            "actual": metadata.get("training_device_name"),
            "expected": "contains {}".format(accelerator),
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
    oracle = metadata.get("oracle_diagnostics") or {}
    if (
        oracle.get("diagnosis_only") is not True
        or oracle.get("excluded_from_fair_replay_claims") is not True
    ):
        mismatches.append({
            "field": "oracle_diagnostics",
            "actual": oracle,
            "expected": "diagnosis-only and excluded from replay claims",
        })
    return mismatches


def analyzer_evidence_mismatches(matched, tag, metadata):
    """Bind current metadata to hashes stored by the completed analyzer."""
    mismatches = []
    method = "offline_" + tag
    accounting = matched.get("replay_accounting") or {}
    nn_info = accounting.get(method) or {}
    normal_info = accounting.get("offline_stride") or {}
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
        ("nn_list_sha256", metadata.get("nn_list_sha256"),
         nn_info.get("sha256")),
        ("offline_nn_entries", metadata.get("offline_nn_entries"),
         nn_info.get("entries")),
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
        for kind in ("stream", "candidate"):
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


def audit_stride_source(path):
    """Prove that the live Stride decision uses only PC and address.

    cache_hit and type are part of ChampSim's generic prefetcher signature, but
    the pinned Stride implementation must not read them in the function body.
    This complements the metadata/encoder checks with an audit of the teacher's
    actual source-visible input.
    """
    if not path.is_file():
        raise SystemExit("missing Stride source for input audit: {}".format(path))
    raw = path.read_bytes()
    source = raw.decode("utf-8")
    marker = "void StridePrefetcher::invoke_prefetcher"
    next_marker = "uint32_t StridePrefetcher::generate_prefetch"
    start = source.find(marker)
    end = source.find(next_marker, start + len(marker))
    if start < 0 or end < 0:
        raise SystemExit("cannot isolate Stride invoke_prefetcher in {}".format(path))
    function = source[start:end]
    brace = function.find("{")
    if brace < 0:
        raise SystemExit("missing Stride invoke_prefetcher body in {}".format(path))
    signature = function[:brace]
    body = function[brace + 1:]
    expected_signature_fields = ("pc", "address", "cache_hit", "type")
    missing = [
        name for name in expected_signature_fields
        if re.search(r"\b{}\b".format(name), signature) is None
    ]
    used = {
        name: len(re.findall(r"\b{}\b".format(name), body))
        for name in expected_signature_fields
    }
    if missing or used["pc"] == 0 or used["address"] == 0:
        raise SystemExit(
            "Stride source input audit failed: missing={} body_uses={}".format(
                missing, used
            )
        )
    if used["cache_hit"] != 0 or used["type"] != 0:
        raise SystemExit(
            "Stride source uses excluded generic inputs: {}".format(used)
        )
    return {
        "status": "PASS",
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "invoke_prefetcher_signature_fields": list(expected_signature_fields),
        "invoke_prefetcher_body_reference_counts": used,
        "effective_source_inputs": ["pc", "address"],
        "signature_only_inputs": ["cache_hit", "type"],
        "provenance_scope": (
            "current checkout only; the completed run did not record this "
            "source blob SHA"
        ),
    }


def model_record(row, metadata, normal, no_pref, matched):
    triggers = row.get("runtime_reachable_list_triggers")
    normal_triggers = normal.get("runtime_reachable_list_triggers")
    record = {
        "method": row.get("method"),
        "model_tag": metadata.get("model_tag"),
        "model_revision": metadata.get("model_revision"),
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
        "runtime_unreached_list_entries": row.get(
            "runtime_unreached_list_entries"
        ),
        "runtime_unreached_list_triggers": row.get(
            "runtime_unreached_list_triggers"
        ),
    }
    diagnostic_keys = (
        "categorical_count_head_used",
        "count_training_objective",
        "count_support",
        "count_train_statistics",
        "count_fit_train_class_frequencies",
        "count_fit_train_add_one_natural_priors",
        "count_zero_is_implicit_hurdle",
        "count_regression_used",
        "log_count_used",
        "hurdle_head_used",
        "stop_padding_used",
        "loss_class_reweighting_used",
        "decode_prior_correction_used",
        "blocked_validation_source",
        "blocked_validation_length_source",
        "fit_train_callbacks",
        "blocked_validation_callbacks",
        "selected_epoch",
        "selected_blocked_validation",
        "original_guard_role",
        "original_guard_phase_shift_metrics",
        "blocked_validation_behavior_metrics",
        "oracle_diagnostics",
        "oracle_diagnostics_replayed",
        "decoder_blocked_validation_diagnostics",
        "decoder_original_guard_diagnostics",
        "decoder_eval_diagnostics",
        "deterministic_decoding",
        "deterministic_decoding_reproducible",
        "stochastic_decoding",
        "decoder_previous_teacher_action_used_as_input",
        "decoder_previous_predicted_action_used_as_input",
        "decoder_previous_sampled_action_used_as_input",
        "decoder_rank_conditioning",
        "all_teacher_ranks_supervised",
        "terminal_stop_supervised_for_every_teacher_sequence",
        "hurdle_training_objective",
        "hurdle_class_weighting",
        "hurdle_class_weights_ZERO_POSITIVE",
        "hurdle_training_statistics",
        "hurdle_decoding_rule",
        "hurdle_prior_correction_at_decode_used",
        "hurdle_prior_correction_rule",
        "positive_count_training_objective",
        "positive_count_decoding_rule",
        "positive_log_count_initial_bias",
        "decoded_count_definition",
        "separate_global_gate_used",
        "separate_count_head_used",
        "teacher_sequence_training_label_statistics",
        "delta_training_objective",
        "delta_decoding_rule",
        "delta_vocabulary_exact_size",
        "delta_vocabulary_statistics",
        "delta_other_escape",
        "delta_other_decode_precision",
        "decode_per_callback_resource_watchdog",
        "decode_per_role_resource_watchdog",
        "decode_resource_watchdog_behavior",
        "checkpoint_selection",
        "checkpoint_selection_roles",
        "checkpoint_selection_metrics",
        "guard_selection_composite_or_mean_used",
        "selected_guard_epoch",
        "selected_guard_key",
        "evaluation_decode_passes",
        "hurdle_count_decoder_diagnostics",
        "parent_raw_decoder_diagnostics",
        "parent_raw_heldout_behavior_metrics",
        "parent_run_id",
        "parent_model_tag",
        "parent_model_revision",
        "parent_decoder_revision",
        "parent_nn_list_sha256",
        "parent_raw_reproduction_sha256",
        "encoder_diagnostics",
        "heldout_behavior_metrics",
        "train_action_summary",
        "guard_action_summary",
        "eval_action_summary",
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
    parser.add_argument("--stride-source", type=Path, default=STRIDE_SOURCE)
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
        and method.startswith(NEURAL_METHOD_PREFIX)
    }
    if not expected_tags:
        raise SystemExit("matched comparison contains no Stride neural rows")
    if expected_tags != EXPECTED_TAGS:
        raise SystemExit(
            "unexpected Stride model set: observed={} expected={}".format(
                sorted(expected_tags), sorted(EXPECTED_TAGS)
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
        raise SystemExit("no Stride neural metadata found")
    observed_tags = {record["model_tag"] for record in records}
    if observed_tags != expected_tags:
        raise SystemExit(
            "Stride metadata/replay model set mismatch: observed={} expected={}".format(
                sorted(observed_tags), sorted(expected_tags)
            )
        )

    encoder_hashes = {
        metadata.get("runtime_encoder_sha256")
        for metadata in selected_metadata
    }
    model_revisions = {
        metadata.get("model_revision") for metadata in selected_metadata
    }
    if len(model_revisions) != 1 or None in model_revisions:
        raise SystemExit(
            "Stride metadata does not share one model revision: {}".format(
                sorted(repr(value) for value in model_revisions)
            )
        )
    model_revision = next(iter(model_revisions))
    analyzer_model_revision = matched.get("model_revision")
    if (
        analyzer_model_revision is not None
        and analyzer_model_revision != model_revision
    ):
        raise SystemExit(
            "analyzer/metadata model revision mismatch: {!r} != {!r}".format(
                analyzer_model_revision, model_revision
            )
        )
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

    source_audit = audit_stride_source(args.stride_source)
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
        "current_checkout_source_input_audit": source_audit,
        "completed_run_source_blob_provenance_verified": False,
        "same_external_input": SOURCE_INPUTS,
        "teacher_role": (
            "captured Stride actions are supervision and comparator replay "
            "only; they are not neural inference inputs"
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
            "This schema does not directly measure harmful victim eviction.",
            "The source audit covers the current checkout, not historical "
            "completed-run source provenance.",
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
        "method", "model_tag", "model_revision", "model_size",
        "parameter_count",
        "ipc", "ipc_delta_vs_offline_normal", "ipc_delta_vs_no_pref",
        "l2_load_miss_rate", "l2_miss_rate_delta_vs_offline_normal",
        "pf_requested", "request_ratio_vs_offline_normal",
        "runtime_reachable_list_triggers",
        "trigger_ratio_vs_offline_normal",
        "actions_per_reached_trigger",
        "offline_normal_actions_per_reached_trigger",
        "pf_filled", "pf_useful", "pf_useless", "pf_late",
        "selected_accuracy", "coverage_vs_no_pref_l2_miss", "timeliness",
        "input_contract_verified",
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
