#!/usr/bin/env python3
"""Torch-free source of truth for the matched-input 623 Stride v25 model.

The only external runtime fields are lossless pc64 and line58.  Captured
normal-Stride actions are labels, the offline-normal replay, and diagnosis;
they are never neural inputs, candidates, templates, thresholds, or budgets.

Each configured H is a *total* recurrent width.  It is split evenly between a
global chronological LSTM and an exact-PC-local LSTM, then concatenated and
fused.  The decoder learns an unweighted ZERO/POSITIVE hurdle, a positive-only
categorical count, and rank-conditioned direct deltas.  Every real teacher
rank supervises all 58 modular delta bits with Bernoulli NLL; there is no
token vocabulary or escape head.
"""
import argparse
import json
import math
from collections import Counter
from decimal import Decimal, InvalidOperation


TRACE = "623.xalancbmk_s-700B"
POLICY = "stride"
RUN_ID = "623_offline_lstm_stride_dual_context_hurdle_unique_v25_seed7"
PARENT_INPUT_RUN_ID = (
    "623_offline_lstm_stride_prior_corrected_hurdle_v23_seed7"
)
EXPERIMENT_REVISION = "stride_dual_context_hurdle_unique_v25"
MODEL_REVISION = "dual_context_raw_hurdle_positive_count_rank_delta_bits_v25"
DECODER_REVISION = "ordered_unique_hurdle_count_delta_bits_map_v25"
OPERATION = "train-v25"
MODEL_TAG_PREFIX = "dual_context_hurdle_stride_lstm_h"
MODEL_POINTS = {
    "lstm": {8: "p0", 16: "p1", 32: "p2", 64: "p3", 128: "p4"}
}

TRAINING_SEED = 7
TRAINING_EPOCHS = 10
TRAINING_CHUNK_LEN = 1024
TRAINING_ACCUMULATE_CHUNKS = 16
TRAINING_LEARNING_RATE = 0.002
FIT_NUMERATOR = 4
FIT_DENOMINATOR = 5

ADDRESS_BITS = 64
CACHE_LINE_BYTES = 64
CACHE_LINE_OFFSET_BITS = CACHE_LINE_BYTES.bit_length() - 1
LINE_NUMBER_BITS = ADDRESS_BITS - CACHE_LINE_OFFSET_BITS
RUNTIME_FEATURE_BREAKDOWN = {
    "pc_bits": ADDRESS_BITS,
    "line_bits": LINE_NUMBER_BITS,
}
RUNTIME_FEATURES = sum(RUNTIME_FEATURE_BREAKDOWN.values())
RAW_RUNTIME_FEATURES = RUNTIME_FEATURES
CAUSAL_RUNTIME_FEATURES = 0
SOURCE_INPUTS = ("pc", "addr")

RANK_CODE_FEATURES = 8

DECODER_TRAINING_MODE = (
    "unweighted_callback_hurdle_then_positive_categorical_count_then_"
    "teacher_rank_all_58bit_modular_delta_payload_without_"
    "teacher_or_predicted_action_feedback"
)
HURDLE_OBJECTIVE = "unweighted_natural_ZERO_POSITIVE_cross_entropy"
POSITIVE_COUNT_OBJECTIVE = (
    "unweighted_positive_only_categorical_count_cross_entropy"
)
DELTA_OBJECTIVE = (
    "every_real_teacher_rank_all_58bit_modular_delta_Bernoulli_NLL"
)
FULL_OBJECTIVE = (
    "per_callback_sum_of_hurdle_positive_count_and_all_real_rank_"
    "58bit_negative_log_likelihood"
)
DECODING_RULE = (
    "hurdle_argmax_then_positive_count_argmax_then_ordered_rank_MAP_"
    "with_highest_scoring_feasible_target_selection_and_no_target_mutation"
)
CHECKPOINT_SELECTION = (
    "minimum_complete_last20pct_TRAIN_validation_NLL_then_earlier_epoch"
)
BLOCKED_VALIDATION_LENGTH_SOURCE = (
    "chronological_first80pct_FIT_last20pct_validation_of_original_TRAIN"
)
ORIGINAL_GUARD_ROLE = "phase_shift_audit_only"


def parse_exact_integer(value):
    """Parse integer text without ever rounding through binary float."""
    text = str(value).strip()
    if not text:
        raise ValueError("empty integer field")
    try:
        return int(text, 0)
    except ValueError:
        try:
            converted = Decimal(text)
        except InvalidOperation as exc:
            raise ValueError("non-integral integer field {!r}".format(text)) from exc
        if (
            not converted.is_finite()
            or converted != converted.to_integral_value()
        ):
            raise ValueError("non-integral integer field {!r}".format(text))
        return int(converted)


def positive_count_statistics(counts):
    values = [int(value) for value in counts]
    if not values or any(value < 0 for value in values):
        raise ValueError("count labels must be a nonempty nonnegative sequence")
    frequencies = Counter(value for value in values if value > 0)
    if not frequencies:
        raise ValueError("positive-count support cannot be empty")
    support = sorted(frequencies)
    positive_total = sum(frequencies.values())
    priors = [
        (frequencies[value] + 1.0)
        / float(positive_total + len(support))
        for value in support
    ]
    if not math.isclose(sum(priors), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("positive count prior does not sum to one")
    return {
        "positive_count_support": support,
        "positive_class_frequencies": [frequencies[value] for value in support],
        "positive_callbacks": positive_total,
        "zero_callbacks": len(values) - positive_total,
        "add_one_smoothed_positive_priors": priors,
        "loss_class_weights": None,
        "source": "labels in the named training partition only",
    }


def expected_parameter_count(hidden_size, positive_count_output_classes):
    """Exact DualContextHurdleStrideLSTM parameter count."""
    hidden = int(hidden_size)
    count_classes = int(positive_count_output_classes)
    if hidden not in MODEL_POINTS["lstm"] or hidden % 2:
        raise ValueError("unsupported even Stride v25 total hidden size")
    if count_classes < 1:
        raise ValueError("positive count output classes must be positive")
    # Two H/2 input projections + two H/2 LSTMs + H->H fusion + rank
    # projection + hurdle/count/58-bit heads.
    return (
        5 * hidden * hidden
        + (201 + count_classes) * hidden
        + count_classes + 60
    )


def model_tag(hidden_size):
    hidden = int(hidden_size)
    if hidden not in MODEL_POINTS["lstm"]:
        raise ValueError("unsupported Stride v25 hidden size")
    return MODEL_TAG_PREFIX + str(hidden)


def model_points_description():
    points = []
    for hidden, pair_id in sorted(MODEL_POINTS["lstm"].items()):
        points.append({
            "model_family": "lstm",
            "model_size": hidden,
            "total_recurrent_width": hidden,
            "global_recurrent_width": hidden // 2,
            "exact_pc_local_recurrent_width": hidden // 2,
            "architecture_pair_id": pair_id,
            "model_tag": model_tag(hidden),
            "parameter_count_is_dataset_dependent": True,
        })
    return {
        "operation": OPERATION,
        "run_id": RUN_ID,
        "parent_input_run_id": PARENT_INPUT_RUN_ID,
        "trace": TRACE,
        "policy": POLICY,
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": MODEL_REVISION,
        "decoder_revision": DECODER_REVISION,
        "decoder_training_mode": DECODER_TRAINING_MODE,
        "hurdle_training_objective": HURDLE_OBJECTIVE,
        "positive_count_training_objective": POSITIVE_COUNT_OBJECTIVE,
        "delta_training_objective": DELTA_OBJECTIVE,
        "complete_training_objective": FULL_OBJECTIVE,
        "decoding_rule": DECODING_RULE,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "blocked_validation_length_source": BLOCKED_VALIDATION_LENGTH_SOURCE,
        "fit_fraction": FIT_NUMERATOR / float(FIT_DENOMINATOR),
        "original_guard_role": ORIGINAL_GUARD_ROLE,
        "guard_selection_composite_or_mean_used": False,
        "neural_role": "standalone_direct_action_prefetcher",
        "external_input_fields": list(SOURCE_INPUTS),
        "teacher_actions_are_model_inputs": False,
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_request_rate_used_as_budget": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "dual_context_core_used": True,
        "global_chronological_lstm_used": True,
        "exact_pc_local_lstm_used": True,
        "learned_global_local_fusion_used": True,
        "total_hidden_split_rule": "equal_halves_global_and_exact_PC_local",
        "hurdle_head_used": True,
        "hurdle_classes": ["ZERO", "POSITIVE"],
        "hurdle_loss_class_weights": None,
        "positive_only_categorical_count_head_used": True,
        "count_zero_is_implicit_hurdle": True,
        "count_regression_used": False,
        "log_count_used": False,
        "stop_token_used": False,
        "stop_padding_used": False,
        "loss_class_reweighting_used": False,
        "decode_prior_correction_used": False,
        "positive_count_support_source_selection": "FIT_labels_only",
        "positive_count_support_source_final": "complete_TRAIN_labels_only",
        "count_support_is_dataset_derived": True,
        "count_support_is_normal_request_budget": False,
        "count_support_is_tuned_degree": False,
        "action_loss_scope": "all_58_bits_of_every_real_teacher_rank",
        "delta_token_head_used": False,
        "delta_vocabulary_used": False,
        "delta_escape_head_used": False,
        "rank_delta_payload_head": "one_direct_58bit_modular_Bernoulli_head",
        "rank_delta_payload_bits": LINE_NUMBER_BITS,
        "delta_bit_loss_scope": "all_58_bits_of_every_real_teacher_rank",
        "delta_decode_precision": "exact_all_58_modular_bits",
        "delta_bit_initialization": (
            "zero_weight_add_one_smoothed_partition_bit_marginal_logit_bias"
        ),
        "delta_bit_prior_source_selection": "all_real_FIT_teacher_actions",
        "delta_bit_prior_source_final": "all_real_complete_TRAIN_teacher_actions",
        "full_modular_line_delta_range_reachable": True,
        "all_deltas_relative_to_current_demand": True,
        "stride_fill_level": "FILL_L2_only_no_learned_fill_head",
        "fill_level": "FILL_L2_only_no_fill_head",
        "probability_threshold_used": False,
        "threshold_related_hardcodes_used": False,
        "inference_policy_hardcodes_used": False,
        "neural_degree_cap": None,
        "fixed_page_offset_classes": None,
        "same_page_rule_used_by_neural_inference": False,
        "normal_policy_templates_used_by_neural_inference": False,
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_previous_predicted_action_used_as_input": False,
        "decoder_previous_sampled_action_used_as_input": False,
        "deterministic_target_uniqueness_constraint_used": True,
        "target_uniqueness_constraint_is_neural_action_feedback": False,
        "target_uniqueness_rule": (
            "choose_highest_scoring_feasible_Bernoulli_payload_"
            "and_fail_closed_if_none_without_target_mutation"
        ),
        "decoded_target_projection_or_mutation_used": False,
        "runtime_feature_count": RUNTIME_FEATURES,
        "raw_runtime_feature_count": RAW_RUNTIME_FEATURES,
        "causal_runtime_feature_count": CAUSAL_RUNTIME_FEATURES,
        "runtime_feature_breakdown": dict(RUNTIME_FEATURE_BREAKDOWN),
        "runtime_encoding": "lossless_raw_pc64_plus_line58_only",
        "engineered_runtime_features": [],
        "training_state_routing": (
            "one_global_chronological_state_plus_one_local_state_per_exact_PC"
        ),
        "original_guard_used_for_selection": False,
        "evaluation_used_for_selection": False,
        "line_number_bits": LINE_NUMBER_BITS,
        "rank_code_features": RANK_CODE_FEATURES,
        "selection_protocol": {
            "fit": "first_80_percent_of_TRAIN",
            "validation": "last_20_percent_of_TRAIN",
            "selection_support": "FIT_only",
            "metric": "complete_validation_NLL_per_callback",
            "tie_break": "earlier_epoch",
        },
        "final_training_protocol": (
            "reset_seed_reinitialize_and_retrain_from_scratch_on_complete_"
            "TRAIN_for_selected_epoch_count"
        ),
        "training_config": {
            "seed": TRAINING_SEED,
            "epochs": TRAINING_EPOCHS,
            "chunk_len": TRAINING_CHUNK_LEN,
            "accumulate_chunks": TRAINING_ACCUMULATE_CHUNKS,
            "learning_rate": TRAINING_LEARNING_RATE,
            "fit_numerator": FIT_NUMERATOR,
            "fit_denominator": FIT_DENOMINATOR,
        },
        "determinism_contract": {
            "cublas_workspace_config": ":4096:8",
            "torch_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "float32_matmul_precision": "highest",
            "required_accelerator_name_contains": "A100",
        },
        "weights_retrained": True,
        "checkpoint_reused": False,
        "decoder_only_change": False,
        "input_archive_reused_byte_for_byte": True,
        "oracle_diagnostics": {
            "oracle_count_plus_nn_action": "diagnosis_only_not_replayed",
            "nn_count_plus_oracle_action": "diagnosis_only_not_replayed",
            "excluded_from_fair_neural_claims": True,
        },
        "parameter_formula": (
            "5*H^2 + (201+K)*H + K+60; H is total recurrent "
            "width split H/2 global and H/2 exact-PC local; "
            "K=positive count classes; every rank has one 58-bit head"
        ),
        "parameter_count_is_dataset_dependent": True,
        "points": points,
    }


def self_test_contract():
    exact_examples = {
        "12": 12,
        "12.0": 12,
        "1.2e1": 12,
        "0xc": 12,
        "9007199254740993.0": 9007199254740993,
        "9.007199254740993e15": 9007199254740993,
        "18446744073709551615.0": 18446744073709551615,
    }
    for value, expected in exact_examples.items():
        if parse_exact_integer(value) != expected:
            raise RuntimeError("exact Decimal integer parser self-test failed")
    for value in ("", "1.25", "nan", "inf", "0x1.0"):
        try:
            parse_exact_integer(value)
        except ValueError:
            continue
        raise RuntimeError("exact integer parser accepted {!r}".format(value))

    statistics = positive_count_statistics([0, 0, 1, 2, 2])
    if statistics["positive_count_support"] != [1, 2]:
        raise RuntimeError("positive count support changed")
    if statistics["positive_class_frequencies"] != [1, 2]:
        raise RuntimeError("positive count frequencies changed")
    if statistics["loss_class_weights"] is not None:
        raise RuntimeError("v25 positive count loss acquired class weights")

    previous = 0
    for hidden in sorted(MODEL_POINTS["lstm"]):
        parameters = expected_parameter_count(hidden, 2)
        if parameters <= previous:
            raise RuntimeError("parameter count is not monotone")
        previous = parameters

    contract = model_points_description()
    forbidden_true = (
        "count_regression_used", "log_count_used", "stop_padding_used",
        "loss_class_reweighting_used", "decode_prior_correction_used",
    )
    if any(contract[key] for key in forbidden_true):
        raise RuntimeError("forbidden decoder mechanism leaked into v25")
    if (
        not contract["dual_context_core_used"]
        or not contract["hurdle_head_used"]
        or not contract["positive_only_categorical_count_head_used"]
        or not contract["count_zero_is_implicit_hurdle"]
        or contract["delta_token_head_used"]
        or contract["delta_vocabulary_used"]
        or contract["delta_escape_head_used"]
        or contract["rank_delta_payload_bits"] != LINE_NUMBER_BITS
        or contract["delta_bit_loss_scope"]
        != "all_58_bits_of_every_real_teacher_rank"
        or contract["teacher_actions_are_model_inputs"]
        or contract["runtime_feature_count"] != 122
        or contract["causal_runtime_feature_count"] != 0
    ):
        raise RuntimeError("v25 dual-context/input contract changed")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--describe-model-points", action="store_true")
    group.add_argument("--field")
    group.add_argument("--tags-csv", action="store_true")
    group.add_argument("--base-tag", action="store_true")
    group.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    description = model_points_description()
    if args.self_test:
        self_test_contract()
        print("PASS")
    elif args.field:
        if args.field not in description:
            raise SystemExit("unknown model-contract field {}".format(args.field))
        value = description[args.field]
        print(json.dumps(value, sort_keys=True)
              if isinstance(value, (dict, list)) else value)
    elif args.tags_csv:
        print(",".join(point["model_tag"] for point in description["points"]))
    elif args.base_tag:
        print(description["points"][0]["model_tag"])
    else:
        print(json.dumps(description, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
