#!/usr/bin/env python3
"""Torch-free source of truth for the matched-input 623 SPP v24 model.

The input is unchanged: one DEMAND/FILL kind bit and one lossless 58-bit line
number in source chronology. v24 predicts natural categorical callback
cardinality, then only the K real actions as TRAIN-observed joint (delta, fill)
tokens plus fill-specific OTHER escapes. There is no STOP padding, hurdle,
count regression, class reweighting, prior correction, or action feedback.
"""
import argparse
import json
import math
import re


TRACE = "623.xalancbmk_s-700B"
POLICY = "spp"
RUN_ID = "623_offline_lstm_spp_natural_cardinality_v24_seed7"
PARENT_INPUT_RUN_ID = "623_offline_lstm_spp_finite_joint_rank_v23_seed7"
EXPERIMENT_REVISION = (
    "spp_source_input_variable_delta_fill_feedback_free_running_v11"
)
MODEL_REVISION = (
    "selected_chronological_lstm_natural_cardinality_joint_action_v24"
)
DECODER_REVISION = (
    "categorical_count_then_conditional_observed_joint_action_map_v24"
)
OPERATION = "train-v24"
MODEL_TAG_PREFIX = "natural_cardinality_spp_lstm_h"
MODEL_POINTS = {
    "lstm": {8: "p0", 16: "p1", 32: "p2", 64: "p3", 128: "p4"}
}

SEED = 7
EPOCHS = 10
CHUNK_LEN = 1024
ACCUMULATE_CHUNKS = 16
LEARNING_RATE = 0.002

ADDRESS_BITS = 64
CACHE_LINE_BYTES = 64
CACHE_LINE_SHIFT = CACHE_LINE_BYTES.bit_length() - 1
LINE_ADDRESS_BITS = ADDRESS_BITS - CACHE_LINE_SHIFT
LINE_ADDRESS_MODULUS = 1 << LINE_ADDRESS_BITS
RUNTIME_FEATURE_COUNT = LINE_ADDRESS_BITS + 1
FILL_LEVELS = (2, 4)
MAX_EXACT_ACTION_PAIRS = 255
RANK_CODE_SIZE = 4
OTHER_L2_NAME = "OTHER_L2"
OTHER_LLC_NAME = "OTHER_LLC"
EXTERNAL_INPUT_FIELDS = (
    "callback_kind", "invoke_prefetcher.addr", "cache_fill.evicted_addr",
)

CORE_TYPES = ("global", "event_routed")
CORE_SELECTION_HIDDEN_SIZE = 32
CORE_SELECTION_METRIC = "guard_natural_action_list_nll_per_callback"
CORE_SELECTION_TIE_BREAK = "global"
CORE_ABLATION_ROLE = "architecture_selection_only_not_replayed"

DECODER_TRAINING_MODE = (
    "natural_categorical_callback_cardinality_then_teacher_rank_observed_"
    "joint_delta_fill_without_action_feedback"
)
COUNT_OBJECTIVE = "unweighted_natural_categorical_count_cross_entropy"
ACTION_OBJECTIVE = (
    "teacher_action_rank_only_unweighted_observed_joint_delta_fill_"
    "cross_entropy"
)
OTHER_ACTION_OBJECTIVE = (
    "fill_specific_OTHER_token_only_signed_log_delta_smooth_l1"
)
DECODING_RULE = (
    "deterministic_categorical_count_argmax_then_exactly_K_independent_"
    "rank_conditioned_observed_joint_action_MAP"
)
CHECKPOINT_SELECTION = (
    "guard_natural_action_list_NLL_then_earlier_epoch"
)


def exact_int(value):
    text = str(value).strip()
    try:
        return int(text, 0)
    except ValueError:
        if re.fullmatch(r"[+-]?[0-9]+", text):
            return int(text, 10)
        raise ValueError("non-integral integer field {!r}".format(text))


def count_statistics(counts):
    values = [int(value) for value in counts]
    if not values or any(value < 0 for value in values):
        raise ValueError("counts must be nonempty and nonnegative")
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
        raise RuntimeError("count priors do not sum to one")
    return {
        "maximum_train_count": maximum,
        "count_output_classes": classes,
        "class_order": list(range(classes)),
        "class_frequencies": frequencies,
        "add_one_smoothed_natural_priors": priors,
        "loss_class_weights": None,
        "source": "TRAIN callback action counts only",
    }


def action_token_count(exact_pair_count):
    count = int(exact_pair_count)
    if not 1 <= count <= MAX_EXACT_ACTION_PAIRS:
        raise ValueError("exact joint-action pair count must be in [1, 255]")
    return count + 2


def other_action_token(fill_level, exact_pair_count):
    fill = int(fill_level)
    if fill not in FILL_LEVELS:
        raise ValueError("OTHER action requires L2 or LLC fill")
    return int(exact_pair_count) + FILL_LEVELS.index(fill)


def decode_action_token(token, exact_pairs):
    index = int(token)
    pairs = [(int(delta), int(fill)) for delta, fill in exact_pairs]
    if not 0 <= index < action_token_count(len(pairs)):
        raise ValueError("action token is outside realized vocabulary")
    if index < len(pairs):
        delta, fill = pairs[index]
        return "EXACT", delta, fill
    fill = FILL_LEVELS[index - len(pairs)]
    return "OTHER", None, fill


def expected_parameter_count(
    core_type, hidden_size, count_output_classes, action_output_classes,
):
    hidden = int(hidden_size)
    counts = int(count_output_classes)
    actions = int(action_output_classes)
    if core_type not in CORE_TYPES:
        raise ValueError("unsupported SPP v24 core")
    if hidden not in MODEL_POINTS["lstm"]:
        raise ValueError("unsupported SPP v24 hidden size")
    if counts < 1 or actions < 3:
        raise ValueError("invalid realized v24 output dimensions")
    if core_type == "global":
        return (
            9 * hidden * hidden
            + (74 + counts + actions) * hidden
            + counts + actions + 1
        )
    return (
        17 * hidden * hidden
        + (82 + counts + actions) * hidden
        + counts + actions + 1
    )


def model_tag(family, size):
    size = int(size)
    if family != "lstm" or size not in MODEL_POINTS["lstm"]:
        raise ValueError("unsupported SPP v24 model point")
    return MODEL_TAG_PREFIX + str(size)


def describe_model_points():
    points = []
    for size, pair_id in sorted(MODEL_POINTS["lstm"].items()):
        points.append({
            "family": "lstm",
            "size": size,
            "pair_id": pair_id,
            "tag": model_tag("lstm", size),
            "parameter_count_is_dataset_and_core_dependent": True,
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
        "seed": SEED,
        "epochs": EPOCHS,
        "chunk_len": CHUNK_LEN,
        "accumulate_chunks": ACCUMULATE_CHUNKS,
        "learning_rate": LEARNING_RATE,
        "training_config": {
            "seed": SEED,
            "epochs": EPOCHS,
            "chunk_len": CHUNK_LEN,
            "accumulate_chunks": ACCUMULATE_CHUNKS,
            "learning_rate": LEARNING_RATE,
        },
        "determinism_config": {
            "required_accelerator": "NVIDIA A100",
            "cublas_workspace_config": ":4096:8",
            "torch_deterministic_algorithms_enabled": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "float32_matmul_precision": "highest",
            "deterministic_argmax_decoding": True,
            "fail_closed": True,
        },
        "decoder_training_mode": DECODER_TRAINING_MODE,
        "count_training_objective": COUNT_OBJECTIVE,
        "joint_action_training_objective": ACTION_OBJECTIVE,
        "other_action_training_objective": OTHER_ACTION_OBJECTIVE,
        "decoding_rule": DECODING_RULE,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "neural_role": "standalone_direct_action_prefetcher",
        "teacher_actions_are_model_inputs": False,
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_request_rate_used_as_budget": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "runtime_feature_count": RUNTIME_FEATURE_COUNT,
        "runtime_encoding": (
            "lossless_58_bit_cache_line_plus_one_DEMAND_FILL_kind_bit"
        ),
        "external_input_fields": list(EXTERNAL_INPUT_FIELDS),
        "model_does_not_use_pc": True,
        "core_types": list(CORE_TYPES),
        "core_selection_hidden_size": CORE_SELECTION_HIDDEN_SIZE,
        "core_selection_metric": CORE_SELECTION_METRIC,
        "core_selection_tie_break": CORE_SELECTION_TIE_BREAK,
        "core_ablation_role": CORE_ABLATION_ROLE,
        "core_selection_uses_evaluation": False,
        "global_core": (
            "one global chronological single_layer_LSTM_state"
        ),
        "event_routed_core": (
            "one global chronological hidden_cell_state_with_distinct_"
            "DEMAND_and_FILL_LSTM_transitions_selected_only_by_callback_kind"
        ),
        "event_routed_core_adds_runtime_input": False,
        "count_head_used": True,
        "count_regression_used": False,
        "hurdle_head_used": False,
        "stop_token_used": False,
        "stop_padding_used": False,
        "loss_class_reweighting_used": False,
        "decode_prior_correction_used": False,
        "count_zero_is_implicit_hurdle": True,
        "count_support_source": "zero_through_maximum_TRAIN_teacher_count",
        "count_support_is_dataset_derived": True,
        "count_support_is_normal_request_budget": False,
        "count_support_is_tuned_degree": False,
        "joint_action_vocabulary_source": (
            "TRAIN_observed_delta_fill_pairs_only_plus_OTHER_L2_OTHER_LLC"
        ),
        "joint_action_vocabulary_max_exact_pairs": MAX_EXACT_ACTION_PAIRS,
        "joint_action_vocabulary_cartesian_product_used": False,
        "separate_delta_head_used": False,
        "separate_fill_head_used": False,
        "action_loss_scope": "teacher_action_ranks_only",
        "delta_other_escape": "signed_log_continuous_bounded_approximation",
        "probability_threshold_used": False,
        "threshold_related_hardcodes_used": False,
        "inference_policy_hardcodes_used": False,
        "fixed_page_offset_classes": None,
        "same_page_rule_used_by_neural_inference": False,
        "normal_policy_templates_used_by_neural_inference": False,
        "neural_degree_cap": None,
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_previous_predicted_action_used_as_input": False,
        "decoder_previous_sampled_action_used_as_input": False,
        "rank_code_size": RANK_CODE_SIZE,
        "fill_levels": list(FILL_LEVELS),
        "stochastic_decoding": False,
        "guard_selection_composite_or_mean_used": False,
        "parameter_formula": {
            "global": (
                "9*H^2 + (74+K+A)*H + K+A+1"
            ),
            "event_routed": (
                "17*H^2 + (82+K+A)*H + K+A+1"
            ),
            "K": "maximum_TRAIN_count+1",
            "A": "realized_exact_joint_pairs+2",
        },
        "parameter_count_is_dataset_and_core_dependent": True,
        "non_neural_control": {
            "name": "every_callback_TRAIN_modal_delta_FILL_LLC",
            "delta_source": "TRAIN_teacher_action_frequency_only",
            "fill_level": "FILL_LLC",
            "actions_per_callback": 1,
            "uses_neural_model": False,
            "excluded_from_neural_claims": True,
        },
        "oracle_diagnostics": {
            "oracle_count_plus_nn_action": "diagnosis_only_not_replayed",
            "nn_count_plus_oracle_action": "diagnosis_only_not_replayed",
            "excluded_from_fair_neural_claims": True,
        },
        "input_archive_reused_byte_for_byte": True,
        "points": points,
    }


def self_test_contract():
    large = (1 << 60) + 3
    if exact_int(str(large)) != large or exact_int("0008") != 8:
        raise RuntimeError("exact integer parser lost an integer field")
    for invalid in ("1.0", "1e3", "nan", "inf"):
        try:
            exact_int(invalid)
        except ValueError:
            continue
        raise RuntimeError("exact integer parser accepted {!r}".format(invalid))

    statistics = count_statistics([0, 1, 1, 3])
    if statistics["class_order"] != [0, 1, 2, 3]:
        raise RuntimeError("natural count support changed")
    pairs = [(-1, 2), (4, 4)]
    if decode_action_token(0, pairs) != ("EXACT", -1, 2):
        raise RuntimeError("exact joint token changed")
    if decode_action_token(
        other_action_token(2, len(pairs)), pairs
    ) != ("OTHER", None, 2):
        raise RuntimeError("OTHER_L2 token changed")
    if decode_action_token(
        other_action_token(4, len(pairs)), pairs
    ) != ("OTHER", None, 4):
        raise RuntimeError("OTHER_LLC token changed")

    for core in CORE_TYPES:
        previous = 0
        for hidden in sorted(MODEL_POINTS["lstm"]):
            parameters = expected_parameter_count(core, hidden, 4, 5)
            if parameters <= previous:
                raise RuntimeError("parameter count is not monotone")
            previous = parameters

    contract = describe_model_points()
    for key in (
        "count_regression_used", "hurdle_head_used", "stop_token_used",
        "stop_padding_used", "loss_class_reweighting_used",
        "decode_prior_correction_used",
        "joint_action_vocabulary_cartesian_product_used",
    ):
        if contract[key]:
            raise RuntimeError("{} leaked into SPP v24".format(key))
    if contract["core_selection_uses_evaluation"]:
        raise RuntimeError("evaluation leaked into core selection")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--describe-model-points", action="store_true")
    parser.add_argument("--tags-csv", action="store_true")
    parser.add_argument("--base-tag", action="store_true")
    parser.add_argument("--field")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    selected = sum((
        args.json, args.describe_model_points, args.tags_csv, args.base_tag,
        args.field is not None, args.self_test,
    ))
    if selected != 1:
        parser.error("select exactly one output mode")
    contract = describe_model_points()
    if args.self_test:
        self_test_contract()
        print("PASS")
    elif args.json or args.describe_model_points:
        print(json.dumps(contract, indent=2, sort_keys=True))
    elif args.tags_csv:
        print(",".join(point["tag"] for point in contract["points"]))
    elif args.base_tag:
        print(contract["points"][0]["tag"])
    elif args.field not in contract or isinstance(contract[args.field], (dict, list)):
        parser.error("--field must name a scalar contract field")
    else:
        print(contract[args.field])


if __name__ == "__main__":
    main()

