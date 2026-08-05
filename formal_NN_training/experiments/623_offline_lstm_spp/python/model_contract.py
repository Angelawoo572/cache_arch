#!/usr/bin/env python3
"""Torch-free source of truth for the independent 623 SPP v20 points.

The exact delta vocabulary is learned from TRAIN labels, so its realized size
and the corresponding parameter count are run metadata rather than constants.
"""
import argparse
import json
import re


TRACE = "623.xalancbmk_s-700B"
POLICY = "spp"
RUN_ID = "623_offline_lstm_spp_independent_vocab_v20_seed7"
EXPERIMENT_REVISION = "spp_source_input_variable_delta_fill_feedback_free_running_v11"
MODEL_REVISION = "global_chronological_lstm_independent_actions_v20"
DECODER_REVISION = "gate_logcount_rank_vocab_other_keyed_fill_v20"
OPERATION = "train-v20"
MODEL_TAG_PREFIX = "independent_vocab_spp_lstm_h"
MODEL_POINTS = {"lstm": {16: "p0", 32: "p1"}}

SEED = 7
DECODER_SEED = 7
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
EXTERNAL_INPUT_FIELDS = (
    "callback_kind", "invoke_prefetcher.addr", "cache_fill.evicted_addr",
)

DECODER_TRAINING_MODE = "fully_supervised_independent_ranks_no_action_feedback"
GATE_OBJECTIVE = "natural_prior_two_class_cross_entropy"
COUNT_OBJECTIVE = "positive_only_log_count_regression"
DELTA_OBJECTIVE = (
    "train_vocabulary_cross_entropy_plus_all_action_signed_log_regression"
)
FILL_OBJECTIVE = (
    "teacher_delta_class_and_rank_conditioned_inverse_frequency_cross_entropy"
)
DECODING_RULE = (
    "gate_MAP_rounded_exp_count_delta_class_MAP_OTHER_signed_log_"
    "and_prior_corrected_keyed_fill_draw"
)


def delta_embed_size(hidden_size):
    return max(4, int(hidden_size) // 4)


def expected_parameter_count(hidden_size, exact_vocabulary_size):
    """Exact count for GlobalSPPLSTM(H,V); OTHER makes C=V+1 classes."""
    hidden = int(hidden_size)
    vocab = int(exact_vocabulary_size)
    if hidden not in MODEL_POINTS["lstm"] or not 0 < vocab <= MAX_EXACT_DELTAS:
        raise ValueError("unsupported SPP v20 dimensions")
    classes = vocab + 1
    embed = delta_embed_size(hidden)
    # input projection + LSTM + gate + count + rank fusion + delta heads +
    # class embedding + target/rank-conditioned fill head.
    return (
        9 * hidden * hidden + 79 * hidden + 16
        + classes * (hidden + 1 + embed) + 2 * embed
    )


def model_tag(family, size):
    size = int(size)
    if family != "lstm" or size not in MODEL_POINTS["lstm"]:
        raise ValueError("unsupported SPP v20 model point")
    return MODEL_TAG_PREFIX + str(size)


def exact_int(value):
    text = str(value).strip()
    try:
        return int(text, 0)
    except ValueError:
        if re.fullmatch(r"[+-]?[0-9]+", text):
            return int(text, 10)
        raise ValueError("non-integral integer field {!r}".format(text))


def self_test_exact_int():
    large = (1 << 60) + 3
    if exact_int(str(large)) != large or exact_int("0008") != 8:
        raise RuntimeError("exact integer parser lost an integer field")
    for invalid in ("1.0", "1e3", "nan", "inf"):
        try:
            exact_int(invalid)
        except ValueError:
            continue
        raise RuntimeError("exact integer parser accepted {!r}".format(invalid))


def describe_model_points():
    return {
        "operation": OPERATION,
        "run_id": RUN_ID,
        "trace": TRACE,
        "policy": POLICY,
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": MODEL_REVISION,
        "decoder_revision": DECODER_REVISION,
        "seed": SEED,
        "decoder_seed": DECODER_SEED,
        "epochs": EPOCHS,
        "chunk_len": CHUNK_LEN,
        "accumulate_chunks": ACCUMULATE_CHUNKS,
        "learning_rate": LEARNING_RATE,
        "training_config": {
            "seed": SEED,
            "decoder_seed": DECODER_SEED,
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
            "fail_closed": True,
        },
        "decoder_training_mode": DECODER_TRAINING_MODE,
        "gate_training_objective": GATE_OBJECTIVE,
        "request_count_training_objective": COUNT_OBJECTIVE,
        "delta_training_objective": DELTA_OBJECTIVE,
        "fill_training_objective": FILL_OBJECTIVE,
        "decoding_rule": DECODING_RULE,
        "neural_role": "standalone_direct_action_prefetcher",
        "teacher_actions_are_model_inputs": False,
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_request_rate_used_as_budget": False,
        "delta_vocabulary_source": "train_labels_only",
        "delta_vocabulary_max_exact": MAX_EXACT_DELTAS,
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
        "teacher_target_fill_conditioning_scope": "conditional_loss_factor_only",
        "teacher_target_used_for_loss_local_fill_conditioning": True,
        "runtime_feature_count": RUNTIME_FEATURE_COUNT,
        "max_exact_train_delta_vocabulary": MAX_EXACT_DELTAS,
        "rank_code_size": RANK_CODE_SIZE,
        "fill_levels": list(FILL_LEVELS),
        "external_input_fields": list(EXTERNAL_INPUT_FIELDS),
        "parameter_formula": (
            "9*H^2 + 79*H + 16 + (V+1)*(H+1+E) + 2*E; "
            "E=max(4,H//4), 1<=V<=255"
        ),
        "parameter_count_is_dataset_dependent": True,
        "points": [
            {
                "family": "lstm", "size": size, "pair_id": pair_id,
                "tag": model_tag("lstm", size),
                "delta_embed_size": delta_embed_size(size),
                "maximum_parameter_count": expected_parameter_count(
                    size, MAX_EXACT_DELTAS
                ),
            }
            for size, pair_id in sorted(MODEL_POINTS["lstm"].items())
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--tags-csv", action="store_true")
    parser.add_argument("--base-tag", action="store_true")
    parser.add_argument("--field")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    selected = sum((
        args.json, args.tags_csv, args.base_tag, args.field is not None,
        args.self_test,
    ))
    if selected != 1:
        parser.error("select exactly one output mode")
    contract = describe_model_points()
    if args.self_test:
        self_test_exact_int()
        for size in MODEL_POINTS["lstm"]:
            if expected_parameter_count(size, MAX_EXACT_DELTAS) <= 0:
                raise RuntimeError("invalid maximum parameter count")
        print("PASS")
    elif args.json:
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
