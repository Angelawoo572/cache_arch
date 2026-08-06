#!/usr/bin/env python3
"""Torch-free source of truth for the matched-input 623 Stride v24 model.

The neural runtime sees only lossless pc64 and aligned line58 bits. Captured
Stride actions are labels and the offline-normal comparator, never runtime
features, candidates, budgets, prefixes, or templates. v24 trains from scratch
and models the natural callback action list as categorical cardinality followed
by rank-conditioned deltas. There is no hurdle, count regression, class
reweighting, prior correction, STOP padding, or action feedback.
"""
import argparse
import json
import math
import re
from collections import Counter


TRACE = "623.xalancbmk_s-700B"
POLICY = "stride"
RUN_ID = "623_offline_lstm_stride_natural_cardinality_v24_seed7"
PARENT_INPUT_RUN_ID = (
    "623_offline_lstm_stride_prior_corrected_hurdle_v23_seed7"
)
EXPERIMENT_REVISION = "stride_source_input_variable_delta_free_running_v9"
MODEL_REVISION = "pc_keyed_raw_natural_cardinality_rank_delta_v24"
DECODER_REVISION = "categorical_count_then_conditional_rank_delta_map_v24"
OPERATION = "train-v24"
MODEL_TAG_PREFIX = "natural_cardinality_stride_lstm_h"
MODEL_POINTS = {
    "lstm": {8: "p0", 16: "p1", 32: "p2", 64: "p3", 128: "p4"}
}

TRAINING_SEED = 7
TRAINING_EPOCHS = 10
TRAINING_CHUNK_LEN = 1024
TRAINING_ACCUMULATE_CHUNKS = 16
TRAINING_LEARNING_RATE = 0.002

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

MAX_EXACT_DELTA_CLASSES = 255
MAX_DELTA_OUTPUT_CLASSES = MAX_EXACT_DELTA_CLASSES + 1
RANK_CODE_FEATURES = 8

DECODER_TRAINING_MODE = (
    "natural_categorical_callback_cardinality_then_teacher_rank_delta_"
    "without_teacher_or_predicted_action_feedback"
)
COUNT_OBJECTIVE = "unweighted_natural_categorical_count_cross_entropy"
DELTA_OBJECTIVE = (
    "teacher_action_rank_only_exact_TRAIN_delta_cross_entropy_plus_"
    "OTHER_only_signed_log_smooth_l1"
)
DECODING_RULE = (
    "deterministic_categorical_count_argmax_then_exactly_K_independent_"
    "rank_conditioned_delta_MAP"
)
CHECKPOINT_SELECTION = (
    "TRAIN_suffix_blocked_validation_natural_action_list_NLL_then_earlier_epoch"
)
BLOCKED_VALIDATION_LENGTH_SOURCE = "original_guard_callback_count"
ORIGINAL_GUARD_ROLE = "phase_shift_audit_only"


def parse_exact_integer(value):
    text = str(value).strip()
    try:
        return int(text, 0)
    except ValueError:
        if re.fullmatch(r"[+-]?[0-9]+(?:\.0+)?", text):
            return int(float(text))
        if re.fullmatch(r"[+-]?[0-9]+(?:\.[0-9]+)?[eE][+-]?[0-9]+", text):
            converted = float(text)
            if math.isfinite(converted) and converted.is_integer():
                return int(converted)
        raise ValueError("non-integral integer field {!r}".format(text))


def count_statistics(counts):
    values = [int(value) for value in counts]
    if not values or any(value < 0 for value in values):
        raise ValueError("count labels must be a nonempty nonnegative sequence")
    maximum = max(values)
    classes = maximum + 1
    frequencies = [0] * classes
    for value in values:
        frequencies[value] += 1
    total = len(values)
    priors = [
        (frequency + 1.0) / float(total + classes)
        for frequency in frequencies
    ]
    if not math.isclose(sum(priors), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("smoothed natural count prior does not sum to one")
    return {
        "maximum_train_count": maximum,
        "count_output_classes": classes,
        "class_order": list(range(classes)),
        "class_frequencies": frequencies,
        "add_one_smoothed_natural_priors": priors,
        "loss_class_weights": None,
        "source": "TRAIN callback action counts only",
    }


def expected_parameter_count(hidden_size, count_output_classes, delta_output_classes):
    """Exact NaturalCardinalityStrideLSTM parameter count."""
    hidden = int(hidden_size)
    count_classes = int(count_output_classes)
    delta_classes = int(delta_output_classes)
    if hidden not in MODEL_POINTS["lstm"]:
        raise ValueError("unsupported Stride v24 hidden size")
    if count_classes < 1:
        raise ValueError("count output classes must be positive")
    if not 2 <= delta_classes <= MAX_DELTA_OUTPUT_CLASSES:
        raise ValueError("delta output classes must be in [2, 256]")
    return (
        8 * hidden * hidden
        + (RUNTIME_FEATURES + RANK_CODE_FEATURES
           + count_classes + delta_classes + 11) * hidden
        + count_classes + delta_classes + 1
    )


def model_tag(hidden_size):
    hidden = int(hidden_size)
    if hidden not in MODEL_POINTS["lstm"]:
        raise ValueError("unsupported Stride v24 hidden size")
    return MODEL_TAG_PREFIX + str(hidden)


def model_points_description():
    points = []
    for hidden, pair_id in sorted(MODEL_POINTS["lstm"].items()):
        points.append({
            "model_family": "lstm",
            "model_size": hidden,
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
        "count_training_objective": COUNT_OBJECTIVE,
        "delta_training_objective": DELTA_OBJECTIVE,
        "decoding_rule": DECODING_RULE,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "blocked_validation_length_source": BLOCKED_VALIDATION_LENGTH_SOURCE,
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
        "separate_global_gate_used": False,
        "separate_count_head_used": False,
        "categorical_count_head_used": True,
        "count_head_used": True,
        "count_regression_used": False,
        "log_count_used": False,
        "hurdle_head_used": False,
        "stop_token_used": False,
        "stop_padding_used": False,
        "loss_class_reweighting_used": False,
        "decode_prior_correction_used": False,
        "count_support_source": "zero_through_maximum_TRAIN_teacher_count",
        "count_support_is_dataset_derived": True,
        "count_support_is_normal_request_budget": False,
        "count_support_is_tuned_degree": False,
        "count_zero_is_implicit_hurdle": True,
        "action_loss_scope": "teacher_action_ranks_only",
        "delta_vocabulary_source": "FIT_TRAIN_labels_only",
        "delta_vocabulary_max_exact": MAX_EXACT_DELTA_CLASSES,
        "delta_output_classes": "realized_exact_classes_plus_one_OTHER",
        "maximum_delta_output_classes": MAX_DELTA_OUTPUT_CLASSES,
        "delta_other_escape": "signed_log_continuous_bounded_approximation",
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
        "runtime_feature_count": RUNTIME_FEATURES,
        "raw_runtime_feature_count": RAW_RUNTIME_FEATURES,
        "causal_runtime_feature_count": CAUSAL_RUNTIME_FEATURES,
        "runtime_feature_breakdown": dict(RUNTIME_FEATURE_BREAKDOWN),
        "runtime_encoding": "lossless_raw_pc64_plus_line58_only",
        "engineered_runtime_features": [],
        "training_state_routing": "exact_observed_PC_keyed_hidden_cell",
        "original_guard_used_for_selection": False,
        "evaluation_used_for_selection": False,
        "line_number_bits": LINE_NUMBER_BITS,
        "rank_code_features": RANK_CODE_FEATURES,
        "training_config": {
            "seed": TRAINING_SEED,
            "epochs": TRAINING_EPOCHS,
            "chunk_len": TRAINING_CHUNK_LEN,
            "accumulate_chunks": TRAINING_ACCUMULATE_CHUNKS,
            "learning_rate": TRAINING_LEARNING_RATE,
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
            "8*H^2 + (RUNTIME_FEATURES+RANK_CODE_FEATURES+K+C+11)*H "
            "+ K+C+1; K=max_TRAIN_count+1; C=exact_delta_classes+1"
        ),
        "parameter_count_is_dataset_dependent": True,
        "points": points,
    }


def self_test_contract():
    examples = {"12": 12, "12.0": 12, "1.2e1": 12, "0xc": 12}
    for value, expected in examples.items():
        if parse_exact_integer(value) != expected:
            raise RuntimeError("exact integer parser self-test failed")
    for value in ("", "1.25", "nan", "inf"):
        try:
            parse_exact_integer(value)
        except ValueError:
            continue
        raise RuntimeError("exact integer parser accepted {!r}".format(value))

    statistics = count_statistics([0, 0, 1, 2, 2])
    if statistics["class_order"] != [0, 1, 2]:
        raise RuntimeError("natural count support changed")
    if statistics["class_frequencies"] != [2, 1, 2]:
        raise RuntimeError("natural count frequencies changed")
    if statistics["loss_class_weights"] is not None:
        raise RuntimeError("v24 count loss acquired class weights")

    previous = 0
    for hidden in sorted(MODEL_POINTS["lstm"]):
        parameters = expected_parameter_count(hidden, 3, 4)
        if parameters <= previous:
            raise RuntimeError("parameter count is not monotone")
        previous = parameters

    contract = model_points_description()
    forbidden_true = (
        "hurdle_head_used", "count_regression_used", "log_count_used",
        "stop_padding_used", "loss_class_reweighting_used",
        "decode_prior_correction_used",
    )
    if any(contract[key] for key in forbidden_true):
        raise RuntimeError("v23 decoder mechanism leaked into v24")
    if (
        not contract["categorical_count_head_used"]
        or contract["teacher_actions_are_model_inputs"]
        or contract["runtime_feature_count"] != 122
        or contract["causal_runtime_feature_count"] != 0
    ):
        raise RuntimeError("v24 natural-cardinality/input contract changed")


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
