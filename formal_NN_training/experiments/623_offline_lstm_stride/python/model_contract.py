#!/usr/bin/env python3
"""Torch-free source of truth for the matched-input 623 Stride v21 sweep.

The neural runtime sees only lossless ``pc64`` and aligned ``line58`` bits.
Captured Stride actions are TRAIN labels and offline-normal replay entries, not
runtime features, candidates, budgets, prefixes, or templates.  A single LSTM
is routed by exact PC.  Its rank-conditioned decoder independently predicts a
binary STOP/EMIT action and, on EMIT, a current-demand-relative delta.

The exact delta alphabet is derived from TRAIN and is therefore known only
after the input archive is loaded.  Contract points expose the maximum weight
count at 255 exact classes plus OTHER; run metadata records the realized class
and parameter counts.
"""
import argparse
import json
from decimal import Decimal, InvalidOperation


TRACE = "623.xalancbmk_s-700B"
POLICY = "stride"
RUN_ID = "623_offline_lstm_stride_rank_stop_emit_v21_seed7"

# The collected bytes are reused unchanged.  This names their source-input
# revision, not the v21 model or decoder revision.
EXPERIMENT_REVISION = "stride_source_input_variable_delta_free_running_v9"
MODEL_REVISION = "pc_keyed_raw_rank_stop_emit_v21"
DECODER_REVISION = "deterministic_rank_stop_emit_train_vocab_v21"
OPERATION = "train-v21"
MODEL_TAG_PREFIX = "rank_stop_emit_stride_lstm_h"
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

# The realized alphabet has C=|TRAIN exact vocabulary|+1 rows.  The final row
# is OTHER.  255 is capacity, not a page topology or request-degree cap.
MAX_EXACT_DELTA_CLASSES = 255
MAX_DELTA_OUTPUT_CLASSES = MAX_EXACT_DELTA_CLASSES + 1
RANK_CODE_FEATURES = 8
RANK_DECISION_CLASSES = 2
STOP_CLASS = 0
EMIT_CLASS = 1
SOURCE_INPUTS = ("pc", "addr")
DECODE_PER_CALLBACK_WATCHDOG = 4096
DECODE_PER_ROLE_WATCHDOG = 10000000

DECODER_TRAINING_MODE = (
    "independent_rank_STOP_EMIT_with_each_teacher_action_and_terminal_STOP_"
    "without_teacher_or_predicted_action_feedback"
)
RANK_DECISION_OBJECTIVE = (
    "TRAIN_inverse_frequency_STOP_EMIT_cross_entropy_with_equal_aggregate_"
    "class_mass"
)
DELTA_OBJECTIVE = (
    "dynamic_TRAIN_exact_delta_cross_entropy_plus_every_emitted_rank_signed_"
    "log_auxiliary_smooth_l1"
)
DECODING_RULE = (
    "deterministic_rank_loop_argmax_STOP_or_EMIT_until_STOP_then_delta_MAP"
)
CHECKPOINT_SELECTION = (
    "guard_only_lexicographic_target_f1_trigger_f1_count_exact_negative_"
    "request_ratio_error_negative_train_loss_earlier_epoch"
)


def parse_exact_integer(value):
    """Parse decimal/scientific/hex integer text without float rounding."""
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


def expected_parameter_count(hidden_size, delta_output_classes=None):
    """Return realized weights for C classes, or the C=256 maximum.

    Modules are raw projection, one LSTM, rank projection, STOP/EMIT head,
    dynamic delta-class head, and one signed-log coordinate head.
    """
    hidden_size = int(hidden_size)
    classes = (
        MAX_DELTA_OUTPUT_CLASSES
        if delta_output_classes is None else int(delta_output_classes)
    )
    if hidden_size <= 0:
        raise ValueError("hidden size must be positive")
    if classes < 2 or classes > MAX_DELTA_OUTPUT_CLASSES:
        raise ValueError("delta output classes must be in [2, 256]")
    return (
        8 * hidden_size * hidden_size
        + (
            RUNTIME_FEATURES + RANK_CODE_FEATURES + classes + 13
        ) * hidden_size
        + classes + 3
    )


def model_tag(hidden_size):
    hidden_size = int(hidden_size)
    if hidden_size not in MODEL_POINTS["lstm"]:
        raise ValueError("unsupported Stride v21 hidden size")
    return MODEL_TAG_PREFIX + str(hidden_size)


def model_points_description():
    points = []
    for hidden_size, pair_id in sorted(MODEL_POINTS["lstm"].items()):
        maximum = expected_parameter_count(hidden_size)
        points.append({
            "model_family": "lstm",
            "model_size": hidden_size,
            "architecture_pair_id": pair_id,
            "model_tag": model_tag(hidden_size),
            "maximum_parameter_count": maximum,
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
        "rank_decision_training_objective": RANK_DECISION_OBJECTIVE,
        "delta_training_objective": DELTA_OBJECTIVE,
        "decoding_rule": DECODING_RULE,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "guard_selection_composite_or_mean_used": False,
        "neural_role": "standalone_direct_action_prefetcher",
        "teacher_actions_are_model_inputs": False,
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_request_rate_used_as_budget": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "separate_global_gate_used": False,
        "separate_count_head_used": False,
        "log_count_used": False,
        "rank_decision_classes": ["STOP", "EMIT"],
        "rank_decision_class_indices": {"STOP": STOP_CLASS, "EMIT": EMIT_CLASS},
        "rank_decision_class_weight_source": "TRAIN_actions_plus_terminal_STOPs",
        "rank_decision_class_weight_formula": "N/(2*N_class)",
        "rank_decision_equal_aggregate_train_mass": True,
        "rank_decision_bias_initialization": "zeros",
        "terminal_stop_supervised_for_every_teacher_sequence": True,
        "delta_vocabulary_source": "train_labels_only",
        "delta_vocabulary_max_exact": MAX_EXACT_DELTA_CLASSES,
        "delta_output_classes": "realized_exact_classes_plus_one_OTHER",
        "maximum_delta_output_classes": MAX_DELTA_OUTPUT_CLASSES,
        "delta_class_bias_initialization": (
            "log_add_one_smoothed_TRAIN_exact_plus_OTHER_frequency"
        ),
        "delta_other_escape": "signed_log_continuous_bounded_approximation",
        "delta_other_decode_precision": (
            "rounded_float32_approximate_except_exact_vocabulary"
        ),
        "full_signed_line_delta_range_reachable": False,
        "every_signed_line_delta_exactly_representable": False,
        "exact_delta_representability_scope": "train_vocabulary_only",
        "all_deltas_relative_to_current_demand": True,
        "fixed_page_offset_classes": None,
        "same_page_rule_used_by_neural_inference": False,
        "normal_policy_templates_used_by_neural_inference": False,
        "probability_threshold_used": False,
        "inference_policy_hardcodes_used": False,
        "neural_degree_cap": None,
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_previous_predicted_action_used_as_input": False,
        "decoder_previous_sampled_action_used_as_input": False,
        "runtime_feature_count": RUNTIME_FEATURES,
        "raw_runtime_feature_count": RAW_RUNTIME_FEATURES,
        "causal_runtime_feature_count": CAUSAL_RUNTIME_FEATURES,
        "runtime_feature_breakdown": dict(RUNTIME_FEATURE_BREAKDOWN),
        "runtime_encoding": "lossless_raw_pc64_plus_line58_only",
        "engineered_runtime_features": [],
        "line_number_bits": LINE_NUMBER_BITS,
        "max_exact_delta_classes": MAX_EXACT_DELTA_CLASSES,
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
            "fail_closed_raise_before_replay_never_truncate_or_change_actions"
        ),
        "decode_resource_watchdog_is_neural_degree_cap": False,
        "external_input_fields": list(SOURCE_INPUTS),
        "parameter_formula": (
            "8*H^2 + (RUNTIME_FEATURES+RANK_CODE_FEATURES+C+13)*H + C + 3; "
            "C=realized_exact_delta_classes+1"
        ),
        "parameter_count_contract": (
            "points expose C=256 maximum; run metadata records realized C and "
            "realized parameter count"
        ),
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
    previous = 0
    for hidden_size in sorted(MODEL_POINTS["lstm"]):
        realized = expected_parameter_count(hidden_size, 2)
        maximum = expected_parameter_count(hidden_size)
        if not 0 < realized <= maximum or maximum <= previous:
            raise RuntimeError("invalid realized/maximum parameter contract")
        previous = maximum
    if RUNTIME_FEATURES != 122 or CAUSAL_RUNTIME_FEATURES != 0:
        raise RuntimeError("raw pc64+line58 contract changed")


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
