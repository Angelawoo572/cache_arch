#!/usr/bin/env python3
"""Torch-free source of truth for the matched-input 623 Stride v23 ablation.

The neural runtime sees only lossless ``pc64`` and aligned ``line58`` bits.
Captured Stride actions are TRAIN labels and offline-normal replay entries, not
runtime features, candidates, budgets, prefixes, or templates.  v23 reuses the
five v22 checkpoints and training histories byte-for-byte.  The only policy
change is mathematically undoing the TRAIN inverse-frequency hurdle weights
before deterministic argmax.  One LSTM is routed by exact PC; its learned
positive log-count and rank-conditioned direct signed-delta heads are unchanged.

The exact delta alphabet is derived from TRAIN and is therefore known only
after the input archive is loaded.  Contract points expose the maximum weight
count at 255 exact classes plus OTHER; run metadata records the realized class
and parameter counts.
"""
import argparse
import json
import math
from decimal import Decimal, InvalidOperation


TRACE = "623.xalancbmk_s-700B"
POLICY = "stride"
RUN_ID = "623_offline_lstm_stride_prior_corrected_hurdle_v23_seed7"
PARENT_RUN_ID = "623_offline_lstm_stride_raw_hurdle_count_v22_seed7"
PARENT_MODEL_REVISION = "pc_keyed_raw_hurdle_count_rank_delta_v22"
PARENT_DECODER_REVISION = "deterministic_hurdle_log_count_train_vocab_v22"
PARENT_MODEL_TAG_PREFIX = "hurdle_count_stride_lstm_h"

# The collected bytes are reused unchanged.  This names their source-input
# revision, not the v23 decoder revision.
EXPERIMENT_REVISION = "stride_source_input_variable_delta_free_running_v9"
MODEL_REVISION = "pc_keyed_raw_hurdle_count_rank_delta_v22_reused_v23"
DECODER_REVISION = "train_weight_prior_corrected_hurdle_decode_v23"
OPERATION = "redecode-v23"
MODEL_TAG_PREFIX = "prior_corrected_hurdle_count_stride_lstm_h"
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
# is OTHER.  255 is vocabulary capacity, not a page topology or degree cap.
MAX_EXACT_DELTA_CLASSES = 255
MAX_DELTA_OUTPUT_CLASSES = MAX_EXACT_DELTA_CLASSES + 1
RANK_CODE_FEATURES = 8
ZERO_CLASS = 0
POSITIVE_CLASS = 1
SOURCE_INPUTS = ("pc", "addr")
MAX_HOST_ACTION_COUNT = (1 << 63) - 1
DECODE_PER_CALLBACK_WATCHDOG = 4096
DECODE_PER_ROLE_WATCHDOG = 10000000

DECODER_TRAINING_MODE = (
    "callback_hurdle_and_positive_log_count_plus_teacher_rank_direct_delta_"
    "without_teacher_or_predicted_action_feedback"
)
HURDLE_OBJECTIVE = (
    "TRAIN_inverse_frequency_zero_positive_cross_entropy_with_equal_aggregate_"
    "class_mass"
)
COUNT_OBJECTIVE = "positive_only_log_count_smooth_l1"
DELTA_OBJECTIVE = (
    "dynamic_TRAIN_exact_delta_cross_entropy_plus_every_emitted_rank_signed_"
    "log_auxiliary_smooth_l1"
)
DECODING_RULE = (
    "deterministic_TRAIN_weight_prior_corrected_hurdle_argmax_then_finite_"
    "rounded_exp_positive_log_count_and_rank_conditioned_delta_MAP"
)
CHECKPOINT_SELECTION = (
    "byte_identical_parent_v22_guard_selected_checkpoint_no_v23_reselection"
)


def parent_model_tag(hidden_size):
    hidden_size = int(hidden_size)
    if hidden_size not in MODEL_POINTS["lstm"]:
        raise ValueError("unsupported Stride v23 hidden size")
    return PARENT_MODEL_TAG_PREFIX + str(hidden_size)


def prior_correct_hurdle_logits(logits, class_weights):
    """Undo weighted-cross-entropy prior shift without a threshold.

    If weighted CE learns q(c|x) proportional to w_c p(c|x), natural-posterior
    MAP is argmax(log q(c|x) - log w_c).  Values are ordinary Python sequences
    so the contract remains torch-free and can be audited on Sacramento.
    """
    weights = [float(value) for value in class_weights]
    if len(weights) != 2 or any(
        not math.isfinite(value) or value <= 0 for value in weights
    ):
        raise ValueError("hurdle correction requires two positive finite weights")
    corrected = []
    for row in logits:
        values = [float(value) for value in row]
        if len(values) != 2 or not all(math.isfinite(value) for value in values):
            raise ValueError("hurdle logits must be finite two-class rows")
        corrected.append([
            values[index] - math.log(weights[index]) for index in range(2)
        ])
    return corrected


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


def hurdle_statistics_from_counts(counts):
    """Return the unique TRAIN-derived balanced hurdle weights and biases.

    Inverse-frequency weights make the aggregate weighted mass of zero and
    positive labels exactly equal.  The initial logits are the centered log of
    those effective masses.  They are therefore neutral (0, 0), but are
    derived and checked from the same TRAIN counts rather than hand selected.
    The positive log-count intercept is the finite mean TRAIN log count.
    """
    integers = [int(value) for value in counts]
    if not integers or any(value < 0 for value in integers):
        raise ValueError("hurdle counts must be a nonempty nonnegative sequence")
    zero = sum(value == 0 for value in integers)
    positive_values = [value for value in integers if value > 0]
    positive = len(positive_values)
    if zero <= 0 or positive <= 0:
        raise ValueError("hurdle training requires zero and positive TRAIN rows")
    total = zero + positive
    frequencies = [zero, positive]
    weights = [total / float(2 * frequency) for frequency in frequencies]
    weighted_mass = [
        frequencies[index] * weights[index] for index in range(2)
    ]
    effective_prior = [value / sum(weighted_mass) for value in weighted_mass]
    raw_bias = [math.log(value) for value in effective_prior]
    center = sum(raw_bias) / 2.0
    initial_bias = [value - center for value in raw_bias]
    positive_log_bias = sum(
        math.log(value) for value in positive_values
    ) / float(positive)
    values = weights + effective_prior + initial_bias + [positive_log_bias]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite TRAIN-derived hurdle initialization")
    return {
        "zero_labels": zero,
        "positive_labels": positive,
        "total_callbacks": total,
        "class_weights_ZERO_POSITIVE": weights,
        "weighted_zero_mass": weighted_mass[ZERO_CLASS],
        "weighted_positive_mass": weighted_mass[POSITIVE_CLASS],
        "effective_weighted_class_prior_ZERO_POSITIVE": effective_prior,
        "hurdle_initial_bias_ZERO_POSITIVE": initial_bias,
        "positive_log_count_initial_bias": positive_log_bias,
        "weight_formula": "N/(2*N_class)",
        "source": "TRAIN callback zero/positive action counts only",
    }


def positive_count_mode(log_count, host_max=MAX_HOST_ACTION_COUNT):
    """Map a learned real log-count to a finite positive host integer.

    The learned coordinate is mathematically unbounded.  The implementation
    never clips or wraps it: non-finite or out-of-domain results fail closed.
    """
    value = float(log_count)
    maximum = int(host_max)
    if maximum < 1:
        raise ValueError("host count maximum must be positive")
    if not math.isfinite(value):
        raise ValueError("positive log-count is not finite")
    if value > math.log(float(maximum)):
        raise ValueError("positive log-count exceeds host integer domain")
    try:
        magnitude = math.exp(value)
    except OverflowError as exc:
        raise ValueError("positive log-count exceeds host integer domain") from exc
    if not math.isfinite(magnitude):
        raise ValueError("positive log-count exceeds host integer domain")
    count = max(1, int(math.floor(magnitude + 0.5)))
    if count > maximum:
        raise ValueError("rounded positive count exceeds host integer domain")
    return count


def expected_parameter_count(hidden_size, delta_output_classes=None):
    """Return realized weights for C classes, or the C=256 maximum.

    Modules are raw projection, one LSTM, rank projection, two-class hurdle,
    positive log-count, dynamic delta-class, and signed-log coordinate heads.
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
        + (RUNTIME_FEATURES + RANK_CODE_FEATURES + classes + 14) * hidden_size
        + classes + 4
    )


def model_tag(hidden_size):
    hidden_size = int(hidden_size)
    if hidden_size not in MODEL_POINTS["lstm"]:
        raise ValueError("unsupported Stride v23 hidden size")
    return MODEL_TAG_PREFIX + str(hidden_size)


def model_points_description():
    points = []
    for hidden_size, pair_id in sorted(MODEL_POINTS["lstm"].items()):
        points.append({
            "model_family": "lstm",
            "model_size": hidden_size,
            "architecture_pair_id": pair_id,
            "model_tag": model_tag(hidden_size),
            "parent_model_tag": parent_model_tag(hidden_size),
            "maximum_parameter_count": expected_parameter_count(hidden_size),
        })
    return {
        "operation": OPERATION,
        "run_id": RUN_ID,
        "parent_run_id": PARENT_RUN_ID,
        "parent_model_revision": PARENT_MODEL_REVISION,
        "parent_decoder_revision": PARENT_DECODER_REVISION,
        "trace": TRACE,
        "policy": POLICY,
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": MODEL_REVISION,
        "decoder_revision": DECODER_REVISION,
        "decoder_training_mode": DECODER_TRAINING_MODE,
        "hurdle_training_objective": HURDLE_OBJECTIVE,
        "positive_count_training_objective": COUNT_OBJECTIVE,
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
        "separate_global_gate_used": True,
        "separate_count_head_used": True,
        "log_count_used": True,
        "hurdle_classes": ["ZERO", "POSITIVE"],
        "hurdle_class_indices": {"ZERO": ZERO_CLASS, "POSITIVE": POSITIVE_CLASS},
        "hurdle_class_weight_source": "TRAIN_callback_zero_positive_counts",
        "hurdle_class_weight_formula": "N/(2*N_class)",
        "hurdle_equal_aggregate_train_mass": True,
        "hurdle_prior_correction_at_decode_used": True,
        "hurdle_prior_correction_rule": (
            "weighted_logits_minus_log_TRAIN_inverse_frequency_class_weight"
        ),
        "hurdle_decoding_rule": (
            "deterministic_prior_corrected_two_class_argmax"
        ),
        "hurdle_bias_initialization": (
            "centered_log_effective_weighted_TRAIN_class_mass"
        ),
        "positive_log_count_bias_initialization": (
            "mean_log_positive_TRAIN_count"
        ),
        "positive_count_support": "mathematically_unbounded_positive_integers",
        "positive_count_host_behavior": "fail_closed_no_clip_or_wrap",
        "positive_count_mode": "max_1_round_exp_log_count",
        "terminal_stop_supervised_for_every_teacher_sequence": False,
        "delta_vocabulary_source": "train_labels_only",
        "delta_vocabulary_max_exact": MAX_EXACT_DELTA_CLASSES,
        "delta_output_classes": "realized_exact_classes_plus_one_OTHER",
        "maximum_delta_output_classes": MAX_DELTA_OUTPUT_CLASSES,
        "delta_class_bias_initialization": (
            "log_add_one_smoothed_TRAIN_exact_plus_OTHER_frequency"
        ),
        "delta_coordinate_bias_initialization": (
            "mean_TRAIN_signed_log_delta"
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
            "redecoder_source_sha256",
            "model_contract_source_sha256",
            "threshold_free_policy_source_sha256",
        ],
        "maximum_host_action_count": MAX_HOST_ACTION_COUNT,
        "decode_per_callback_resource_watchdog": DECODE_PER_CALLBACK_WATCHDOG,
        "decode_per_role_resource_watchdog": DECODE_PER_ROLE_WATCHDOG,
        "decode_resource_watchdog_behavior": (
            "fail_closed_raise_before_replay_never_truncate_or_change_actions"
        ),
        "decode_resource_watchdog_is_neural_degree_cap": False,
        "external_input_fields": list(SOURCE_INPUTS),
        "parameter_formula": (
            "8*H^2 + (RUNTIME_FEATURES+RANK_CODE_FEATURES+C+14)*H + C + 4; "
            "C=realized_exact_delta_classes+1"
        ),
        "parameter_count_contract": (
            "points expose C=256 maximum; run metadata records realized C and "
            "realized parameter count"
        ),
        "weights_retrained": False,
        "checkpoint_reused": True,
        "training_history_reused": True,
        "decoder_only_change": True,
        "parent_artifact_identity_required": True,
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

    statistics = hurdle_statistics_from_counts([0, 0, 0, 1, 2])
    weights = statistics["class_weights_ZERO_POSITIVE"]
    if not math.isclose(3 * weights[0], 2 * weights[1]):
        raise RuntimeError("balanced hurdle weights do not equalize TRAIN mass")
    if any(abs(value) > 1e-12 for value in statistics[
        "hurdle_initial_bias_ZERO_POSITIVE"
    ]):
        raise RuntimeError("effective balanced hurdle bias is not neutral")
    expected_log_bias = (math.log(1) + math.log(2)) / 2.0
    if not math.isclose(
        statistics["positive_log_count_initial_bias"], expected_log_bias
    ):
        raise RuntimeError("positive log-count bias is not TRAIN-derived")

    corrected = prior_correct_hurdle_logits(
        [[math.log(weights[0]) + math.log(0.8),
          math.log(weights[1]) + math.log(0.2)]],
        weights,
    )
    if corrected[0][0] <= corrected[0][1]:
        raise RuntimeError("TRAIN-weight prior correction changed natural MAP")

    modes = [positive_count_mode(value) for value in (
        -100.0, 0.0, math.log(1.6), math.log(2.6)
    )]
    if modes != [1, 1, 2, 3]:
        raise RuntimeError("finite positive count mode changed")
    for value in (float("nan"), float("inf"), math.log(100.0)):
        try:
            positive_count_mode(value, host_max=10)
        except ValueError:
            continue
        raise RuntimeError("host-domain count check accepted {!r}".format(value))

    previous = 0
    for hidden_size in sorted(MODEL_POINTS["lstm"]):
        realized = expected_parameter_count(hidden_size, 2)
        maximum = expected_parameter_count(hidden_size)
        if not 0 < realized <= maximum or maximum <= previous:
            raise RuntimeError("invalid realized/maximum parameter contract")
        previous = maximum
    if RUNTIME_FEATURES != 122 or CAUSAL_RUNTIME_FEATURES != 0:
        raise RuntimeError("raw pc64+line58 contract changed")
    description = model_points_description()
    if (
        description["separate_global_gate_used"] is not True
        or description["separate_count_head_used"] is not True
        or description["log_count_used"] is not True
        or description["terminal_stop_supervised_for_every_teacher_sequence"]
        is not False
        or description["decoder_previous_teacher_action_used_as_input"]
        or description["decoder_previous_predicted_action_used_as_input"]
    ):
        raise RuntimeError("v23 hurdle/count/no-feedback contract changed")
    if (
        description["hurdle_prior_correction_at_decode_used"] is not True
        or description["weights_retrained"] is not False
        or description["checkpoint_reused"] is not True
        or description["decoder_only_change"] is not True
    ):
        raise RuntimeError("v23 decoder-only parent-reuse contract changed")


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
