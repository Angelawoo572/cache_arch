#!/usr/bin/env python3
"""Fail-closed, torch-free validation for active 623 v21 model metadata.

The validator imports only the selected track's ``model_contract.py``.  It
therefore remains usable on the Python-3.6 CPU replay host without importing
the CUDA trainer, torch, or numpy.
"""
from __future__ import print_function

import argparse
import ast
import hashlib
import inspect
import json
import math
import re
import runpy
import sys
from pathlib import Path


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def function_source(path, function_name):
    """Return the same undecorated function block used by inspect.getsource."""
    source = Path(path).read_text()
    tree = ast.parse(source, filename=str(path))
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(nodes) != 1:
        raise ValueError(
            "expected one top-level {} in {}".format(function_name, path)
        )
    lines = source.splitlines(True)
    return "".join(inspect.getblock(lines[nodes[0].lineno - 1:]))


def expected_encoder_sha256(track, trainer_path, contract):
    if track == "stride":
        payload = {
            "entrypoint": function_source(trainer_path, "runtime_features"),
            "bits": function_source(trainer_path, "_unsigned_bits"),
            "external_fields": list(contract["external_input_fields"]),
            "feature_count": contract["runtime_feature_count"],
            "runtime_encoding": "lossless_raw_pc64_plus_line58_only",
            "engineered_features": [],
        }
    else:
        payload = {
            "entrypoint_source": function_source(trainer_path, "runtime_bundle"),
            "primitive_source": function_source(trainer_path, "_unsigned_bits"),
            "fields": list(contract["external_input_fields"]),
            "use_pc": False,
            "line_address_bits": 58,
            "cache_line_bytes": 64,
            "bit_order": "least_significant_first",
            "callback_kind_encoding": {"DEMAND": 1.0, "FILL": 0.0},
            "derived_runtime_features": [],
        }
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def expected_router_source_sha256(track, trainer_path):
    if track == "stride":
        payload = "".join(function_source(trainer_path, name) for name in (
            "_pc_groups", "_initial_state", "_encode_chunk",
        ))
    else:
        payload = function_source(trainer_path, "decision_router_sha256")
    return hashlib.sha256(payload.encode()).hexdigest()


def is_integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


class Checks(object):
    def __init__(self):
        self.bad = {}

    def equal(self, key, actual, expected):
        if actual != expected:
            self.bad[key] = {"actual": actual, "expected": expected}

    def require(self, key, condition, actual, expected):
        if not condition:
            self.bad[key] = {"actual": actual, "expected": expected}

    def sha(self, key, value):
        self.require(
            key, SHA256_RE.fullmatch(str(value or "")) is not None,
            value, "64 lowercase hexadecimal SHA-256 characters",
        )

    def finish(self, track, metadata_path):
        if self.bad:
            raise SystemExit(
                "invalid 623 {} v21 metadata {}:\n{}".format(
                    track, metadata_path,
                    json.dumps(self.bad, indent=2, sort_keys=True),
                )
            )


def common_checks(checks, track, metadata, contract, experiment_dir,
                  trainer_path, contract_path):
    source_inputs = list(contract["external_input_fields"])
    expected = {
        "run_id": contract["run_id"],
        "operation": contract["operation"],
        "trace": contract["trace"],
        "matched_normal_prefetcher": contract["policy"],
        "neural_role": "standalone_direct_action_prefetcher",
        "model_family": "lstm",
        "track_model_family": "lstm",
        "experiment_revision": contract["experiment_revision"],
        "model_revision": contract["model_revision"],
        "decoder_revision": contract["decoder_revision"],
        "model_point_contract": contract,
        "training_config": contract["training_config"],
        "source_decision_effective_external_input": source_inputs,
        "training_runtime_fields": source_inputs,
        "inference_runtime_fields": source_inputs,
        "same_external_input_contract": True,
        "training_inference_input_encoder_identical": True,
        "teacher_actions_are_model_inputs": False,
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "normal_policy_request_rate_used_as_budget": False,
        "normal_policy_constants_used_by_neural_inference": False,
        "normal_policy_templates_used_by_neural_inference": False,
        "probability_threshold_used": False,
        "threshold_related_hardcodes_used": False,
        "inference_policy_hardcodes_used": False,
        "neural_degree_cap": None,
        "fixed_page_offset_classes": None,
        "same_page_rule_used_by_neural_inference": False,
        "future_label_window_used": False,
        "decoder_training_mode": contract["decoder_training_mode"],
        "delta_other_escape": contract["delta_other_escape"],
        "delta_other_decode_precision": contract[
            "delta_other_decode_precision"
        ],
        "full_signed_line_delta_range_reachable": False,
        "every_signed_line_delta_exactly_representable": False,
        "exact_delta_representability_scope": "train_vocabulary_only",
        "cublas_workspace_config": ":4096:8",
        "torch_deterministic_algorithms_enabled": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "float32_matmul_precision": "highest",
    }
    for key, value in expected.items():
        checks.equal(key, metadata.get(key), value)
    for key, value in contract["training_config"].items():
        checks.equal("pinned_training_" + key, metadata.get(key), value)

    common_dir = experiment_dir.parent.parent / "common"
    source_paths = {
        "trainer_source_sha256": trainer_path,
        "model_contract_source_sha256": contract_path,
        "threshold_free_policy_source_sha256": (
            common_dir / "threshold_free_policy.py"
        ),
    }
    for key, path in source_paths.items():
        checks.require(key + "_file", path.is_file(), str(path), "present")
        if path.is_file():
            checks.equal(key, metadata.get(key), sha256(path))

    expected_encoder = expected_encoder_sha256(track, trainer_path, contract)
    encoder_keys = (
        "runtime_encoder_sha256", "training_runtime_encoder_sha256",
        "inference_runtime_encoder_sha256",
    )
    for key in encoder_keys:
        checks.sha(key, metadata.get(key))
        checks.equal(key + "_current", metadata.get(key), expected_encoder)
    checks.require(
        "runtime_encoder_hash_equality",
        len(set(metadata.get(key) for key in encoder_keys)) == 1,
        [metadata.get(key) for key in encoder_keys], "three identical hashes",
    )


def validate_vocabulary(checks, vocabulary, size, maximum, frequencies=None):
    valid = (
        isinstance(vocabulary, list) and is_integer(size)
        and 0 < size <= maximum and len(vocabulary) == size
        and len(set(vocabulary)) == size
        and all(is_integer(value) for value in vocabulary)
    )
    checks.require(
        "dynamic_train_delta_vocabulary", valid,
        {"size": size, "vocabulary": vocabulary},
        "1..{} unique integer TRAIN deltas".format(maximum),
    )
    if frequencies is not None:
        frequency_valid = (
            isinstance(frequencies, list) and len(frequencies) == size
            and all(is_integer(value) and value > 0 for value in frequencies)
        )
        checks.require(
            "delta_vocabulary_train_frequencies", frequency_valid,
            frequencies, "one positive integer TRAIN frequency per delta",
        )
        if valid and frequency_valid:
            observed_order = list(zip(frequencies, vocabulary))
            checks.require(
                "delta_vocabulary_frequency_order",
                observed_order == sorted(
                    observed_order, key=lambda item: (-item[0], item[1])
                ), observed_order,
                "descending TRAIN frequency with signed-delta tie break",
            )
    return valid


def validate_stride(checks, metadata, contract, namespace, trainer_path):
    expected = {
        "runtime_encoding": "lossless_raw_pc64_plus_line58_only",
        "runtime_feature_count": 122,
        "raw_runtime_feature_count": 122,
        "causal_runtime_feature_count": 0,
        "runtime_feature_breakdown": {"pc_bits": 64, "line_bits": 58},
        "raw_runtime_input_only": True,
        "engineered_runtime_features": [],
        "causal_derived_features_from_same_external_input": [],
        "derived_features_use_teacher_or_future": False,
        "all_deltas_relative_to_current_demand": True,
        "delta_vocabulary_source": "train_labels_only",
        "delta_vocabulary_max_exact": 255,
        "maximum_delta_output_classes": 256,
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_previous_predicted_action_used_as_input": False,
        "decoder_previous_sampled_action_used_as_input": False,
        "sampled_outputs_used_as_decoder_feedback": False,
        "separate_global_gate_used": False,
        "separate_count_head_used": False,
        "log_count_used": False,
        "poisson_objective_used": False,
        "poisson_decoder_used": False,
        "terminal_stop_supervised_for_every_teacher_sequence": True,
        "rank_decision_training_objective": contract[
            "rank_decision_training_objective"
        ],
        "rank_decision_classes": ["STOP", "EMIT"],
        "rank_decision_class_weighting": (
            "TRAIN_inverse_frequency_N_over_2N_class"
        ),
        "rank_decision_equal_aggregate_train_mass": True,
        "data_derived_rank_class_weights_used": True,
        "manual_loss_weights_used": False,
        "rank_decision_decoding_rule": "deterministic_raw_argmax_each_rank",
        "delta_training_objective": contract["delta_training_objective"],
        "decision_rule": contract["decoding_rule"],
        "deterministic_decoding": True,
        "deterministic_decoding_reproducible": True,
        "stochastic_decoding": False,
        "decoder_sampling_roles": [],
        "checkpoint_selection": contract["checkpoint_selection"],
        "checkpoint_selection_primary_role": "guard",
        "checkpoint_selection_roles": [
            "guard_metrics", "TRAIN_loss_tiebreak_only"
        ],
        "training_loss_used_as_lexicographic_tiebreak": True,
        "guard_selection_composite_or_mean_used": False,
        "checkpoint_selection_metrics": [
            "maximize_target_f1", "maximize_trigger_f1",
            "maximize_count_exact_match_rate",
            "minimize_absolute_request_ratio_error", "minimize_train_loss",
            "prefer_earlier_epoch",
        ],
        "evaluation_used_for_checkpoint_selection": False,
        "evaluation_decode_passes": 1,
        "decode_per_callback_resource_watchdog": contract[
            "decode_per_callback_resource_watchdog"
        ],
        "decode_per_role_resource_watchdog": contract[
            "decode_per_role_resource_watchdog"
        ],
        "decode_resource_watchdog_behavior": contract[
            "decode_resource_watchdog_behavior"
        ],
        "decode_resource_watchdog_is_neural_degree_cap": False,
        "successful_run_hit_decode_resource_watchdog": False,
        "training_state_mode": "exact_pc_keyed_stateful_tbptt",
        "raw_pc64_line58_lossless_self_test": "PASS",
        "rank_stop_emit_equal_mass_self_test": "PASS",
        "separate_gate_and_count_heads_absent_self_test": "PASS",
        "rank_no_action_feedback_self_test": "PASS",
        "dynamic_realized_parameter_self_test": "PASS",
    }
    for key, value in expected.items():
        checks.equal(key, metadata.get(key), value)

    checks.equal("training_device", metadata.get("training_device"), "cuda")
    checks.require(
        "training_device_name", "A100" in str(metadata.get("training_device_name", "")),
        metadata.get("training_device_name"), "CUDA device name containing A100",
    )

    points = {point["model_size"]: point for point in contract["points"]}
    hidden = metadata.get("model_size")
    point = points.get(hidden)
    checks.require("model_point", point is not None, hidden, sorted(points))
    if point is not None:
        checks.equal(
            "architecture_pair_id", metadata.get("architecture_pair_id"),
            point["architecture_pair_id"],
        )
        checks.equal("model_tag", metadata.get("model_tag"), point["model_tag"])

    vocabulary = metadata.get("delta_vocabulary_exact")
    size = metadata.get("delta_vocabulary_exact_size")
    valid_vocabulary = validate_vocabulary(
        checks, vocabulary, size, 255,
        metadata.get("delta_vocabulary_train_frequencies"),
    )
    classes = metadata.get("realized_delta_output_classes")
    checks.equal("realized_delta_output_classes", classes,
                 size + 1 if is_integer(size) else "V+1")
    checks.equal("delta_other_class", metadata.get("delta_other_class"), size)
    if point is not None and valid_vocabulary:
        expected_parameters = namespace["expected_parameter_count"](
            hidden, size + 1
        )
        maximum_parameters = namespace["expected_parameter_count"](hidden)
        for key in (
            "parameter_count", "realized_parameter_count",
            "expected_parameter_count",
        ):
            checks.equal(key, metadata.get(key), expected_parameters)
        checks.equal(
            "maximum_parameter_count", metadata.get("maximum_parameter_count"),
            maximum_parameters,
        )
        checks.equal(
            "model_point_maximum_parameter_count",
            point["maximum_parameter_count"], maximum_parameters,
        )
        checks.require(
            "realized_parameter_count_within_maximum",
            expected_parameters <= maximum_parameters,
            expected_parameters, "<= {}".format(maximum_parameters),
        )

    statistics = metadata.get("rank_decision_training_statistics") or {}
    weights = metadata.get("rank_decision_class_weights_STOP_EMIT")
    stop = statistics.get("stop_labels")
    emit = statistics.get("emit_labels")
    total = statistics.get("total_rank_decisions")
    valid_counts = (
        is_integer(stop) and stop > 0 and is_integer(emit) and emit > 0
        and is_integer(total) and total == stop + emit
    )
    checks.require(
        "rank_decision_training_statistics", valid_counts, statistics,
        "positive TRAIN STOP/EMIT counts with N=STOP+EMIT",
    )
    if valid_counts:
        expected_weights = [
            total / float(2 * stop), total / float(2 * emit),
        ]
        valid_weights = (
            isinstance(weights, list) and len(weights) == 2
            and all(finite_number(value) for value in weights)
            and all(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
                    for actual, expected in zip(weights, expected_weights))
        )
        checks.require(
            "rank_decision_class_weights_STOP_EMIT", valid_weights, weights,
            expected_weights,
        )
        checks.equal(
            "statistics_class_weights_STOP_EMIT",
            statistics.get("class_weights_STOP_EMIT"), weights,
        )
        checks.equal(
            "rank_decision_weight_formula", statistics.get("weight_formula"),
            "N/(2*N_class)",
        )
        checks.equal(
            "rank_decision_weight_source", statistics.get("source"),
            "TRAIN teacher actions plus one terminal STOP per sequence",
        )
        checks.require(
            "rank_decision_equal_weighted_mass",
            finite_number(statistics.get("weighted_stop_mass"))
            and finite_number(statistics.get("weighted_emit_mass"))
            and math.isclose(
                statistics["weighted_stop_mass"], total / 2.0,
                rel_tol=1e-12, abs_tol=1e-9,
            )
            and math.isclose(
                statistics["weighted_emit_mass"], total / 2.0,
                rel_tol=1e-12, abs_tol=1e-9,
            ), statistics, "equal aggregate TRAIN class mass N/2",
        )

    expected_router = expected_router_source_sha256("stride", trainer_path)
    for key in ("training_state_router_sha256", "inference_state_router_sha256"):
        checks.sha(key, metadata.get(key))
        checks.equal(key + "_current", metadata.get(key), expected_router)


def validate_spp(checks, metadata, contract, namespace, experiment_dir,
                 trainer_path):
    expected = {
        "runtime_feature_count": 59,
        "runtime_encoding": (
            "lossless 58-bit cache-line number plus one DEMAND/FILL kind bit"
        ),
        "model_does_not_use_pc": True,
        "pc_is_replay_transport_only": True,
        "model_input_is_causal_external_event_sequence_only": True,
        "cache_fill_feedback_used_as_raw_external_input": True,
        "cache_fill_private_state_used_as_model_input": False,
        "cache_hit_and_type_are_audit_only": True,
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_previous_predicted_action_used_as_input": False,
        "decoder_previous_sampled_action_used_as_input": False,
        "teacher_action_values_used_as_decoder_feedback": False,
        "teacher_target_used_as_recurrent_feedback": False,
        "sampled_outputs_used_as_decoder_feedback": False,
        "separate_gate_head_used": False,
        "request_count_head_used": False,
        "request_count_regression_used": False,
        "action_or_byte_grammar_used": False,
        "stop_emit_training_objective": contract[
            "stop_emit_training_objective"
        ],
        "stop_emit_class_weighting_used": False,
        "stop_emit_prior_extremely_sparse": False,
        "stop_emit_train_class_order": ["STOP", "EMIT"],
        "stop_emit_prior_initialization": "TRAIN_natural_class_log_priors",
        "terminal_stop_supervised": True,
        "stop_emit_decoding_rule": (
            "rank_conditioned_two_class_MAP_until_STOP"
        ),
        "delta_training_objective": contract["delta_training_objective"],
        "delta_vocabulary_source": (
            "TRAIN_labels_only_top_frequency_then_signed_value_tie_break"
        ),
        "delta_vocabulary_architecture_budget": 255,
        "delta_vocabulary_max_exact": 255,
        "fill_training_objective": contract["fill_training_objective"],
        "fill_prior_correction_at_decode_used": True,
        "fill_prior_correction_rule": (
            "balanced_logits_plus_log_TRAIN_natural_prior"
        ),
        "fill_decoding_rule": contract["fill_decoding_rule"],
        "fill_conditioned_on_actual_emitted_target": True,
        "fill_argmax_used": True,
        "fill_probability_threshold": None,
        "stochastic_decoding": False,
        "keyed_sampling_used": False,
        "decoder_sampling_roles": [],
        "global_chronological_lstm": True,
        "routed_demand_fill_recurrent_paths": False,
        "page_local_causal_state": False,
        "handcrafted_semantic_features_used": False,
        "causal_derived_features": [],
        "manual_head_loss_weights_used": False,
        "data_derived_fill_class_weights_used": True,
        "teacher_target_conditions_loss_only_fill_factor": True,
        "guard_selection_rule": contract["guard_selection_rule"],
        "guard_selection_key_fields": [
            "joint_action_f1", "target_f1", "l2_joint_f1", "trigger_f1",
            "count_exact_match_rate", "fill_accuracy_on_matched_targets",
            "negative_normalized_train_loss", "negative_epoch",
        ],
        "guard_selection_composite_or_mean_used": False,
        "guard_selected_checkpoint": True,
        "guard_selected_decoder": False,
        "evaluation_used_for_selection": False,
        "evaluation_decode_count": 1,
        "determinism_fail_closed": True,
        "same_source_input_offline_claim_allowed": True,
        "closed_loop_live_claim_allowed": False,
        "comparison_claim_boundary": (
            "matched-input open-loop offline comparison only"
        ),
        "maximum_action_count_is_learned_not_fixed": True,
        "output_materialization_watchdog_role": (
            "fail_closed_resource_guard_no_truncation_or_forced_count"
        ),
        "output_materialization_watchdog_is_neural_degree_cap": False,
        "independent_rank_decoder_self_test": "PASS",
        "fill_prior_corrected_argmax_self_test": "PASS",
        "fail_closed_watchdog_self_test": "PASS",
    }
    for key, value in expected.items():
        checks.equal(key, metadata.get(key), value)
    checks.require(
        "cuda_device_name", "A100" in str(metadata.get("cuda_device_name", "")),
        metadata.get("cuda_device_name"), "CUDA device name containing A100",
    )

    points = {point["size"]: point for point in contract["points"]}
    hidden = metadata.get("model_size")
    point = points.get(hidden)
    checks.require("model_point", point is not None, hidden, sorted(points))
    if point is not None:
        checks.equal(
            "architecture_pair_id", metadata.get("architecture_pair_id"),
            point["pair_id"],
        )
        checks.equal("model_tag", metadata.get("model_tag"), point["tag"])

    vocabulary = metadata.get("exact_delta_vocabulary")
    size = metadata.get("realized_exact_delta_vocabulary_size")
    valid_vocabulary = validate_vocabulary(
        checks, vocabulary, size, 255, None,
    )
    histogram = metadata.get("train_delta_frequency_histogram")
    parsed_histogram = None
    if isinstance(histogram, dict) and histogram:
        try:
            parsed_histogram = {
                int(key): value for key, value in histogram.items()
            }
        except (TypeError, ValueError):
            parsed_histogram = None
    histogram_valid = (
        isinstance(parsed_histogram, dict) and parsed_histogram
        and len(parsed_histogram) == len(histogram)
        and all(is_integer(value) and value > 0
                for value in parsed_histogram.values())
    )
    checks.require(
        "train_delta_frequency_histogram", histogram_valid, histogram,
        "nonempty unique signed-integer TRAIN delta histogram",
    )
    if valid_vocabulary and histogram_valid:
        expected_vocabulary = sorted(
            parsed_histogram,
            key=lambda value: (-parsed_histogram[value], value),
        )[:255]
        checks.equal(
            "dynamic_train_delta_vocabulary_from_histogram", vocabulary,
            expected_vocabulary,
        )
    checks.equal("exact_delta_vocabulary_size",
                 metadata.get("exact_delta_vocabulary_size"), size)
    checks.equal("other_delta_class", metadata.get("other_delta_class"), size)
    if point is not None and valid_vocabulary:
        expected_parameters = namespace["expected_parameter_count"](hidden, size)
        maximum_parameters = namespace["expected_parameter_count"](hidden, 255)
        for key in ("parameter_count", "realized_parameter_count"):
            checks.equal(key, metadata.get(key), expected_parameters)
        for key in (
            "maximum_parameter_count",
            "maximum_parameter_count_at_255_exact_deltas",
        ):
            checks.equal(key, metadata.get(key), maximum_parameters)
        checks.equal(
            "model_point_maximum_parameter_count",
            point["maximum_parameter_count"], maximum_parameters,
        )
        checks.require(
            "realized_parameter_count_within_maximum",
            expected_parameters <= maximum_parameters,
            expected_parameters, "<= {}".format(maximum_parameters),
        )

    token_counts = metadata.get("stop_emit_train_class_counts")
    token_priors = metadata.get("stop_emit_train_class_priors")
    valid_counts = (
        isinstance(token_counts, list) and len(token_counts) == 2
        and all(is_integer(value) and value > 0 for value in token_counts)
    )
    checks.require(
        "stop_emit_train_class_counts", valid_counts, token_counts,
        "positive natural-frequency [STOP, EMIT] TRAIN counts",
    )
    if valid_counts:
        total = float(sum(token_counts))
        expected_priors = [value / total for value in token_counts]
        valid_priors = (
            isinstance(token_priors, list) and len(token_priors) == 2
            and all(finite_number(value) for value in token_priors)
            and all(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
                    for actual, expected in zip(token_priors, expected_priors))
        )
        checks.require(
            "stop_emit_train_class_priors", valid_priors, token_priors,
            expected_priors,
        )
        checks.equal(
            "terminal_stop_count_train",
            metadata.get("terminal_stop_count_train"), token_counts[0],
        )

    fill_counts = metadata.get("fill_train_class_counts")
    fill_priors = metadata.get("fill_train_priors")
    fill_weights = metadata.get("fill_train_inverse_frequency_weights")
    valid_fill_counts = (
        isinstance(fill_counts, list) and len(fill_counts) == 2
        and all(is_integer(value) and value > 0 for value in fill_counts)
    )
    checks.require(
        "fill_train_class_counts", valid_fill_counts, fill_counts,
        "positive TRAIN [FILL_L2, FILL_LLC] counts",
    )
    if valid_fill_counts:
        fill_total = float(sum(fill_counts))
        expected_fill_priors = [value / fill_total for value in fill_counts]
        expected_fill_weights = [0.5 / value for value in expected_fill_priors]
        for key, actual, expected_values in (
            ("fill_train_priors", fill_priors, expected_fill_priors),
            ("fill_train_inverse_frequency_weights", fill_weights,
             expected_fill_weights),
        ):
            valid_values = (
                isinstance(actual, list) and len(actual) == 2
                and all(finite_number(value) for value in actual)
                and all(math.isclose(value, expected, rel_tol=1e-12,
                                     abs_tol=1e-12)
                        for value, expected in zip(actual, expected_values))
            )
            checks.require(key, valid_values, actual, expected_values)

    for key in (
        "output_materialization_watchdog_actions_per_callback",
        "output_materialization_watchdog_actions_per_role",
    ):
        checks.require(
            key, is_integer(metadata.get(key)) and metadata.get(key) > 0,
            metadata.get(key), "positive fail-closed resource limit",
        )

    source_contract = experiment_dir / "data" / "spp_source_contract.json"
    checks.require(
        "source_contract_file", source_contract.is_file(),
        str(source_contract), "present",
    )
    if source_contract.is_file():
        checks.equal(
            "source_contract_sha256", metadata.get("source_contract_sha256"),
            sha256(source_contract),
        )

    expected_router = expected_router_source_sha256("spp", trainer_path)
    checks.sha("decision_router_source_sha256",
               metadata.get("decision_router_source_sha256"))
    checks.equal(
        "decision_router_source_sha256_current",
        metadata.get("decision_router_source_sha256"), expected_router,
    )
    for role in ("train", "guard", "eval"):
        checks.sha(
            role + "_decision_router_sha256",
            metadata.get(role + "_decision_router_sha256"),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", required=True, choices=("stride", "spp"))
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    args = parser.parse_args()

    metadata_path = args.metadata.resolve()
    experiment_dir = args.experiment_dir.resolve()
    trainer_path = experiment_dir / "python" / "train_and_offline_infer.py"
    contract_path = experiment_dir / "python" / "model_contract.py"
    for label, path in (
        ("metadata", metadata_path), ("trainer", trainer_path),
        ("model contract", contract_path),
    ):
        if not path.is_file():
            raise SystemExit("missing {} {}".format(label, path))

    try:
        metadata = json.loads(metadata_path.read_text())
        namespace = runpy.run_path(str(contract_path))
        describe = namespace.get("model_points_description")
        if describe is None:
            describe = namespace.get("describe_model_points")
        if describe is None:
            raise ValueError("model contract has no description entry point")
        contract = describe()
    except Exception as exc:
        raise SystemExit("cannot load v21 metadata/contract: {}".format(exc))

    checks = Checks()
    checks.equal("contract_policy", contract.get("policy"), args.track)
    common_checks(
        checks, args.track, metadata, contract, experiment_dir,
        trainer_path, contract_path,
    )
    if args.track == "stride":
        validate_stride(checks, metadata, contract, namespace, trainer_path)
    else:
        validate_spp(
            checks, metadata, contract, namespace, experiment_dir, trainer_path,
        )
    checks.finish(args.track, metadata_path)
    print("[PASS] validated 623 {} v21 metadata {}".format(
        args.track, metadata.get("model_tag")
    ))


if __name__ == "__main__":
    main()
