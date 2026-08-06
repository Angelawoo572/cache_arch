#!/usr/bin/env python3
"""Torch-free source of truth for the matched-input 623 SPP v23 model.

The recurrent input is unchanged from v22: one raw DEMAND/FILL bit and the
lossless 58-bit callback line number in global chronological order.  Captured
SPP actions are labels and comparator rows only.  Each TRAIN-derived rank is a
single joint categorical decision: STOP or EMIT(exact-delta-or-OTHER, fill).
The finite rank horizon is the maximum teacher action count observed in TRAIN;
it is data support, not a copied SPP degree, request budget, or tuned constant.
"""
import argparse
import json
import math
import re


TRACE = "623.xalancbmk_s-700B"
POLICY = "spp"
RUN_ID = "623_offline_lstm_spp_finite_joint_rank_v23_seed7"
PARENT_INPUT_RUN_ID = "623_offline_lstm_spp_hurdle_log_count_vocab_v22_seed7"
EXPERIMENT_REVISION = (
    "spp_source_input_variable_delta_fill_feedback_free_running_v11"
)
MODEL_REVISION = "global_chronological_lstm_finite_joint_rank_v23"
DECODER_REVISION = (
    "train_derived_horizon_joint_action_prior_corrected_map_v23"
)
OPERATION = "train-v23"
MODEL_TAG_PREFIX = "finite_joint_rank_spp_lstm_h"
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
MAX_EXACT_DELTAS = 255
RANK_CODE_SIZE = 4
STOP_TOKEN = 0
ACTION_GROUPS = ("STOP", "EMIT_L2", "EMIT_LLC")
OTHER_NAME = "OTHER"
EXTERNAL_INPUT_FIELDS = (
    "callback_kind", "invoke_prefetcher.addr", "cache_fill.evicted_addr",
)

DECODER_TRAINING_MODE = (
    "TRAIN_derived_finite_rank_joint_STOP_or_EMIT_delta_fill_without_action_"
    "feedback_with_all_tail_STOP_supervision"
)
JOINT_ACTION_OBJECTIVE = (
    "TRAIN_group_inverse_frequency_weighted_joint_action_token_cross_entropy"
)
OTHER_DELTA_OBJECTIVE = (
    "OTHER_token_only_signed_log_delta_auxiliary_smooth_l1"
)
DECODING_RULE = (
    "deterministic_group_prior_corrected_joint_token_argmax_at_each_TRAIN_"
    "derived_rank_until_first_STOP_or_finite_horizon"
)
CHECKPOINT_SELECTION = (
    "guard_lexicographic_joint_action_f1_target_f1_l2_joint_f1_trigger_f1_"
    "count_exact_fill_accuracy_then_TRAIN_loss_then_earlier_epoch"
)


def exact_int(value):
    text = str(value).strip()
    try:
        return int(text, 0)
    except ValueError:
        if re.fullmatch(r"[+-]?[0-9]+", text):
            return int(text, 10)
        raise ValueError("non-integral integer field {!r}".format(text))


def joint_token_count(exact_vocabulary_size):
    vocabulary = int(exact_vocabulary_size)
    if not 0 < vocabulary <= MAX_EXACT_DELTAS:
        raise ValueError("exact delta vocabulary must be in [1, 255]")
    return 1 + 2 * (vocabulary + 1)


def encode_emit_token(delta_class, fill_index, exact_vocabulary_size):
    vocabulary = int(exact_vocabulary_size)
    delta_class = int(delta_class)
    fill_index = int(fill_index)
    if not 0 <= delta_class <= vocabulary or fill_index not in (0, 1):
        raise ValueError("invalid joint EMIT token coordinates")
    return 1 + 2 * delta_class + fill_index


def decode_joint_token(token, exact_vocabulary_size):
    token = int(token)
    count = joint_token_count(exact_vocabulary_size)
    if not 0 <= token < count:
        raise ValueError("joint token is outside the realized alphabet")
    if token == STOP_TOKEN:
        return "STOP", None, None
    payload = token - 1
    return "EMIT", payload // 2, payload % 2


def token_group(token, exact_vocabulary_size):
    kind, _, fill_index = decode_joint_token(token, exact_vocabulary_size)
    return 0 if kind == "STOP" else 1 + int(fill_index)


def prior_correct_joint_scores(scores, group_weights, exact_vocabulary_size):
    """Undo TRAIN group weighting without a threshold or hand-tuned prior."""
    weights = [float(value) for value in group_weights]
    values = [float(value) for value in scores]
    if len(weights) != len(ACTION_GROUPS) or any(
        not math.isfinite(value) or value <= 0 for value in weights
    ):
        raise ValueError("joint group weights must be three finite positives")
    if len(values) != joint_token_count(exact_vocabulary_size) or any(
        not math.isfinite(value) for value in values
    ):
        raise ValueError("joint score vector has the wrong realized width")
    return [
        value - math.log(weights[token_group(index, exact_vocabulary_size)])
        for index, value in enumerate(values)
    ]


def expected_parameter_count(hidden_size, exact_vocabulary_size):
    """Exact GlobalSPPJointLSTM count for hidden H and TRAIN vocabulary V."""
    hidden = int(hidden_size)
    if hidden not in MODEL_POINTS["lstm"]:
        raise ValueError("unsupported SPP v23 hidden size")
    tokens = joint_token_count(exact_vocabulary_size)
    return 9 * hidden * hidden + (74 + tokens) * hidden + tokens + 1


def model_tag(family, size):
    size = int(size)
    if family != "lstm" or size not in MODEL_POINTS["lstm"]:
        raise ValueError("unsupported SPP v23 model point")
    return MODEL_TAG_PREFIX + str(size)


def describe_model_points():
    points = []
    for size, pair_id in sorted(MODEL_POINTS["lstm"].items()):
        points.append({
            "family": "lstm",
            "size": size,
            "pair_id": pair_id,
            "tag": model_tag("lstm", size),
            "maximum_parameter_count": expected_parameter_count(
                size, MAX_EXACT_DELTAS
            ),
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
        "joint_action_training_objective": JOINT_ACTION_OBJECTIVE,
        "other_delta_training_objective": OTHER_DELTA_OBJECTIVE,
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
        "global_chronological_lstm": True,
        "delta_vocabulary_source": "train_labels_only",
        "delta_vocabulary_max_exact": MAX_EXACT_DELTAS,
        "delta_other_escape": "signed_log_continuous_bounded_approximation",
        "delta_other_decode_precision": (
            "rounded_float32_approximate_except_exact_vocabulary"
        ),
        "full_signed_line_delta_range_reachable": False,
        "every_signed_line_delta_exactly_representable": False,
        "exact_delta_representability_scope": "train_vocabulary_only",
        "joint_action_token_definition": (
            "STOP_or_EMIT_exact_delta_or_OTHER_cross_FILL_L2_or_FILL_LLC"
        ),
        "joint_action_group_order": list(ACTION_GROUPS),
        "joint_action_group_weight_source": "TRAIN_rank_slot_labels_only",
        "joint_action_group_weight_formula": "N/(3*N_group)",
        "joint_action_bias_initialization": (
            "log_normalized_TRAIN_token_prior_times_group_weight"
        ),
        "joint_action_prior_correction_at_decode_used": True,
        "joint_action_prior_correction_rule": (
            "weighted_joint_logits_minus_log_TRAIN_group_weight"
        ),
        "separate_gate_head_used": False,
        "request_count_head_used": False,
        "separate_delta_head_used": False,
        "separate_fill_head_used": False,
        "stop_emit_head_used": False,
        "terminal_stop_supervised_for_every_teacher_sequence": False,
        "all_available_tail_stop_supervised": True,
        "maximum_length_sequences_terminate_by_finite_support": True,
        "finite_output_horizon_source": (
            "maximum_teacher_action_count_observed_in_TRAIN"
        ),
        "finite_output_horizon_is_dataset_derived": True,
        "finite_output_horizon_is_normal_request_budget": False,
        "finite_output_horizon_is_tuned_degree": False,
        "neural_degree_cap": None,
        "probability_threshold_used": False,
        "threshold_related_hardcodes_used": False,
        "inference_policy_hardcodes_used": False,
        "fixed_page_offset_classes": None,
        "same_page_rule_used_by_neural_inference": False,
        "normal_policy_templates_used_by_neural_inference": False,
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_previous_predicted_action_used_as_input": False,
        "decoder_previous_sampled_action_used_as_input": False,
        "rank_code_size": RANK_CODE_SIZE,
        "fill_levels": list(FILL_LEVELS),
        "stochastic_decoding": False,
        "guard_selection_rule": CHECKPOINT_SELECTION,
        "guard_selection_composite_or_mean_used": False,
        "parameter_formula": (
            "9*H^2 + (74+T)*H + T+1; T=1+2*(V+1), 1<=V<=255"
        ),
        "parameter_count_is_dataset_dependent": True,
        "non_neural_control": {
            "name": "every_callback_TRAIN_modal_delta_FILL_LLC",
            "delta_source": "TRAIN_teacher_action_frequency_only",
            "fill_level": "FILL_LLC",
            "actions_per_callback": 1,
            "uses_neural_model": False,
            "excluded_from_neural_claims": True,
        },
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
    for vocabulary in (1, 7, MAX_EXACT_DELTAS):
        count = joint_token_count(vocabulary)
        if count != 1 + 2 * (vocabulary + 1):
            raise RuntimeError("joint token width changed")
        for delta_class in range(vocabulary + 1):
            for fill_index in (0, 1):
                token = encode_emit_token(delta_class, fill_index, vocabulary)
                if decode_joint_token(token, vocabulary) != (
                    "EMIT", delta_class, fill_index
                ):
                    raise RuntimeError("joint token bijection failed")
        if decode_joint_token(STOP_TOKEN, vocabulary) != ("STOP", None, None):
            raise RuntimeError("STOP token changed")
    corrected = prior_correct_joint_scores(
        [0.0] * joint_token_count(1), [0.5, 2.0, 4.0], 1
    )
    if corrected[STOP_TOKEN] <= corrected[encode_emit_token(0, 0, 1)]:
        raise RuntimeError("group prior correction direction is wrong")
    contract = describe_model_points()
    if any("v23" not in value for value in (
        RUN_ID, MODEL_REVISION, DECODER_REVISION, OPERATION,
    )):
        raise RuntimeError("active SPP identifiers are not consistently v23")
    if contract["neural_degree_cap"] is not None:
        raise RuntimeError("TRAIN-derived horizon was mislabeled as a degree cap")
    for size in MODEL_POINTS["lstm"]:
        if expected_parameter_count(size, MAX_EXACT_DELTAS) <= 0:
            raise RuntimeError("invalid maximum parameter count")


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
