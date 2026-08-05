#!/usr/bin/env python3
"""Torch-free source of truth for the independent 623 Stride v20 points.

Only ``pc`` and the current aligned ``addr`` cross the fair-comparison input
boundary.  The model is not a neural copy of Stride: it learns an exact-PC
recurrent representation and directly generates addresses with a generic
rank code.  A train-label vocabulary makes common integer deltas categorical;
one continuous signed-log ``OTHER`` coordinate provides a broad bounded
approximation without a source page, stride, or degree template. Only
vocabulary members are exact; float32 OTHER values decode approximately by
rounding the inverse signed-log coordinate and do not guarantee domain endpoints.
"""
import argparse
import json
from decimal import Decimal, InvalidOperation


TRACE = "623.xalancbmk_s-700B"
POLICY = "stride"
RUN_ID = "623_offline_lstm_stride_independent_delta_v20_seed7"

# The collected byte streams are intentionally reused.  This names the input
# revision, not the v20 model/decoder revision.
EXPERIMENT_REVISION = "stride_source_input_variable_delta_free_running_v9"
MODEL_REVISION = "pc_keyed_independent_rank_delta_v20"
DECODER_REVISION = "deterministic_train_vocab_other_escape_v20"
OPERATION = "train-v20"
MODEL_TAG_PREFIX = "independent_rank_delta_stride_lstm_h"
MODEL_POINTS = {"lstm": {16: "p0", 32: "p1"}}
TRAINING_SEED = 7
TRAINING_EPOCHS = 10
TRAINING_CHUNK_LEN = 1024
TRAINING_ACCUMULATE_CHUNKS = 16
TRAINING_LEARNING_RATE = 0.002

ADDRESS_BITS = 64
CACHE_LINE_BYTES = 64
CACHE_LINE_OFFSET_BITS = CACHE_LINE_BYTES.bit_length() - 1
LINE_NUMBER_BITS = ADDRESS_BITS - CACHE_LINE_OFFSET_BITS
REUSE_DISTANCE_BITS = 64
VALIDITY_BITS = 2

# Lossless raw values and lossless causal values derived from the same public
# PC/address history.  Signed deltas use LINE_NUMBER_BITS-bit two's complement.
RUNTIME_FEATURE_BREAKDOWN = {
    "pc_bits": ADDRESS_BITS,
    "line_bits": LINE_NUMBER_BITS,
    "current_same_pc_delta_bits": LINE_NUMBER_BITS,
    "prior_same_pc_delta_bits": LINE_NUMBER_BITS,
    "distinct_pc_reuse_distance_bits": REUSE_DISTANCE_BITS,
    "validity_bits": VALIDITY_BITS,
}
RUNTIME_FEATURES = sum(RUNTIME_FEATURE_BREAKDOWN.values())
RAW_RUNTIME_FEATURES = ADDRESS_BITS + LINE_NUMBER_BITS
CAUSAL_RUNTIME_FEATURES = RUNTIME_FEATURES - RAW_RUNTIME_FEATURES

# Byte-sized output alphabet: up to 255 exact TRAIN deltas plus one OTHER.
# This is a model-capacity choice, not a source page topology or request cap.
MAX_EXACT_DELTA_CLASSES = 255
DELTA_OUTPUT_CLASSES = 256
OTHER_DELTA_CLASS = 255
RANK_CODE_FEATURES = 8
GATE_CLASSES = 2
SOURCE_INPUTS = ("pc", "addr")
DECODE_PER_CALLBACK_WATCHDOG = 4096
DECODE_PER_ROLE_WATCHDOG = 10000000

DECODER_TRAINING_MODE = (
    "full_teacher_rank_supervision_without_teacher_or_predicted_action_feedback"
)
GATE_OBJECTIVE = (
    "natural_frequency_unweighted_two_class_cross_entropy_with_log_prior_bias_init"
)
COUNT_OBJECTIVE = "positive_log_count_smooth_l1"
DELTA_OBJECTIVE = (
    "train_frequency_exact_delta_cross_entropy_plus_all_rank_signed_log_auxiliary_smooth_l1"
)
DECODING_RULE = (
    "deterministic_gate_argmax_rounded_positive_log_count_and_rank_delta_MAP"
)
CHECKPOINT_SELECTION = (
    "guard_only_lexicographic_target_f1_trigger_f1_request_ratio_error"
)


def parse_exact_integer(value):
    text = str(value).strip()
    if not text:
        raise ValueError("empty integer text")
    lowered = text.lower()
    signless = lowered[1:] if lowered[:1] in ("+", "-") else lowered
    if signless.startswith("0x"):
        return int(text, 16)
    try:
        decimal = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("invalid integer text {!r}".format(text)) from exc
    if not decimal.is_finite() or decimal != decimal.to_integral_value():
        raise ValueError("non-integral integer text {!r}".format(text))
    return int(decimal)


def expected_parameter_count(hidden_size):
    """Projection + one LSTM + gate/count/rank/delta/escape heads."""
    hidden_size = int(hidden_size)
    return (
        8 * hidden_size * hidden_size
        + (RUNTIME_FEATURES + RANK_CODE_FEATURES + 270) * hidden_size
        + 260
    )


def model_tag(hidden_size):
    hidden_size = int(hidden_size)
    if hidden_size not in MODEL_POINTS["lstm"]:
        raise ValueError("unsupported Stride v20 hidden size")
    return MODEL_TAG_PREFIX + str(hidden_size)


def model_points_description():
    points = []
    for hidden_size, pair_id in sorted(MODEL_POINTS["lstm"].items()):
        points.append({
            "model_family": "lstm",
            "model_size": hidden_size,
            "architecture_pair_id": pair_id,
            "model_tag": model_tag(hidden_size),
            "parameter_count": expected_parameter_count(hidden_size),
        })
    return {
        "operation": OPERATION,
        "run_id": RUN_ID,
        "trace": TRACE,
        "policy": POLICY,
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": MODEL_REVISION,
        "decoder_revision": DECODER_REVISION,
        "decoder_training_mode": DECODER_TRAINING_MODE,
        "gate_training_objective": GATE_OBJECTIVE,
        "positive_count_training_objective": COUNT_OBJECTIVE,
        "delta_training_objective": DELTA_OBJECTIVE,
        "decoding_rule": DECODING_RULE,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "neural_role": "standalone_direct_action_prefetcher",
        "teacher_actions_are_model_inputs": False,
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_request_rate_used_as_budget": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "delta_vocabulary_source": "train_labels_only",
        "delta_vocabulary_max_exact": MAX_EXACT_DELTA_CLASSES,
        "delta_class_bias_initialization": (
            "log_add_one_smoothed_TRAIN_exact_plus_OTHER_frequency"
        ),
        "positive_log_count_bias_initialization": (
            "TRAIN_positive_mean_log_count"
        ),
        "delta_other_escape": "signed_log_continuous_bounded_approximation",
        "delta_other_decode_precision": (
            "rounded_float32_approximate_except_exact_vocabulary"
        ),
        "full_signed_line_delta_range_reachable": False,
        "every_signed_line_delta_exactly_representable": False,
        "exact_delta_representability_scope": "train_vocabulary_only",
        "fixed_page_offset_classes": None,
        "same_page_rule_used_by_neural_inference": False,
        "normal_policy_templates_used_by_neural_inference": False,
        "probability_threshold_used": False,
        "inference_policy_hardcodes_used": False,
        "neural_degree_cap": None,
        "decoder_previous_teacher_action_used_as_input": False,
        "runtime_feature_count": RUNTIME_FEATURES,
        "raw_runtime_feature_count": RAW_RUNTIME_FEATURES,
        "causal_runtime_feature_count": CAUSAL_RUNTIME_FEATURES,
        "runtime_feature_breakdown": dict(RUNTIME_FEATURE_BREAKDOWN),
        "line_number_bits": LINE_NUMBER_BITS,
        "max_exact_delta_classes": MAX_EXACT_DELTA_CLASSES,
        "delta_output_classes": DELTA_OUTPUT_CLASSES,
        "other_delta_class": OTHER_DELTA_CLASS,
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
        "required_source_hashes": [
            "trainer_source_sha256",
            "model_contract_source_sha256",
            "threshold_free_policy_source_sha256",
        ],
        "decode_per_callback_resource_watchdog": DECODE_PER_CALLBACK_WATCHDOG,
        "decode_per_role_resource_watchdog": DECODE_PER_ROLE_WATCHDOG,
        "decode_resource_watchdog_behavior": (
            "fail_closed_raise_before_replay_never_truncate_or_change_count"
        ),
        "decode_resource_watchdog_is_neural_degree_cap": False,
        "external_input_fields": list(SOURCE_INPUTS),
        "parameter_formula": (
            "8*H^2 + (RUNTIME_FEATURES+RANK_CODE_FEATURES+270)*H + 260"
        ),
        "points": points,
    }


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--describe-model-points", action="store_true")
    group.add_argument("--field")
    group.add_argument("--tags-csv", action="store_true")
    group.add_argument("--base-tag", action="store_true")
    args = parser.parse_args()
    description = model_points_description()
    if args.field:
        if args.field not in description:
            raise SystemExit("unknown model-contract field {}".format(args.field))
        value = description[args.field]
        print(
            json.dumps(value, sort_keys=True)
            if isinstance(value, (dict, list)) else value
        )
    elif args.tags_csv:
        print(",".join(point["model_tag"] for point in description["points"]))
    elif args.base_tag:
        print(description["points"][0]["model_tag"])
    else:
        print(json.dumps(description, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
