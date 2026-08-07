#!/usr/bin/env python3
"""Torch-free source of truth for the matched-input 623 SPP v25 model."""
import argparse
import heapq
import json
import math
import re


TRACE = "623.xalancbmk_s-700B"
POLICY = "spp"
RUN_ID = "623_offline_lstm_spp_global_cardinality_unique_v25_seed7"
PARENT_INPUT_RUN_ID = "623_offline_lstm_spp_finite_joint_rank_v23_seed7"
EXPERIMENT_REVISION = "spp_recorded_fill_chronology_matched_input_v25"
MODEL_REVISION = (
    "global_chronological_lstm_natural_cardinality_direct_bits_unique_v25"
)
DECODER_REVISION = (
    "categorical_count_fill_exact_bits_kbest_unique_feasibility_v25"
)
OPERATION = "train-v25"
MODEL_TAG_PREFIX = "global_cardinality_unique_spp_lstm_h"
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
RANK_CODE_SIZE = 4
EXTERNAL_INPUT_FIELDS = (
    "callback_kind", "invoke_prefetcher.addr", "cache_fill.evicted_addr",
)
CORE_TYPE = "global"

DECODER_TRAINING_MODE = (
    "natural_categorical_callback_cardinality_then_real_rank_fill_and_"
    "teacher_fill_specific_exact_modular_delta_bits_without_action_feedback"
)
COUNT_OBJECTIVE = "unweighted_natural_categorical_count_cross_entropy"
FILL_OBJECTIVE = "real_teacher_rank_unweighted_L2_LLC_cross_entropy"
DELTA_BIT_OBJECTIVE = (
    "real_teacher_rank_teacher_fill_specific_exact_58_bit_modular_"
    "Bernoulli_negative_log_likelihood"
)
DECODING_RULE = (
    "deterministic_count_argmax_then_exactly_K_rank_ordered_cross_fill_"
    "maximum_log_probability_feasible_payload_selection_by_exact_kbest_"
    "Bernoulli_subset_enumeration_and_fail_if_infeasible"
)
CHECKPOINT_SELECTION = (
    "guard_per_callback_count_CE_plus_real_rank_fill_CE_plus_all_58_bit_"
    "NLL_then_earlier_epoch"
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


def modular_delta(base_line, target_line):
    return (int(target_line) - int(base_line)) % LINE_ADDRESS_MODULUS


def modular_delta_bits(delta):
    value = int(delta)
    if not 0 <= value < LINE_ADDRESS_MODULUS:
        raise ValueError("modular line delta is outside 58-bit domain")
    return [(value >> bit) & 1 for bit in range(LINE_ADDRESS_BITS)]


def bits_to_modular_delta(bits):
    values = [int(value) for value in bits]
    if len(values) != LINE_ADDRESS_BITS or any(value not in (0, 1) for value in values):
        raise ValueError("payload must contain exactly 58 bits")
    return sum(value << bit for bit, value in enumerate(values))


def modal_bernoulli_payload(logits):
    """Return deterministic modal bits, their log-probability, and flip costs."""
    values = [float(value) for value in logits]
    if len(values) != LINE_ADDRESS_BITS or any(not math.isfinite(x) for x in values):
        raise ValueError("payload logits must be 58 finite values")
    # A zero-logit tie chooses bit zero.  For either modal sign, the stable
    # modal log probability is -log(1 + exp(-abs(logit))).
    bits = [int(value > 0.0) for value in values]
    log_probability = -sum(
        math.log1p(math.exp(-abs(value))) for value in values
    )
    return bits, log_probability, [abs(value) for value in values]


def k_best_flip_subsets(flip_costs, limit):
    """Enumerate exact k-best bit-flip subsets in nondecreasing total cost."""
    costs = [float(value) for value in flip_costs]
    count = int(limit)
    if count < 1 or any(value < 0 or not math.isfinite(value) for value in costs):
        raise ValueError("invalid k-best subset request")
    order = sorted(range(len(costs)), key=lambda bit: (costs[bit], bit))
    yield 0.0, ()
    if count == 1 or not order:
        return
    # Each heap subset stores positions in the sorted-cost array.  Replacing
    # its last position or appending the next position enumerates every subset
    # exactly once; tuple order supplies deterministic tie breaking.
    heap = [(costs[order[0]], (0,))]
    emitted = 1
    while heap and emitted < count:
        total, positions = heapq.heappop(heap)
        yield total, tuple(order[position] for position in positions)
        emitted += 1
        last = positions[-1]
        if last + 1 < len(order):
            next_position = last + 1
            heapq.heappush(
                heap,
                (total + costs[order[next_position]], positions + (next_position,)),
            )
            heapq.heappush(
                heap,
                (
                    total - costs[order[last]] + costs[order[next_position]],
                    positions[:-1] + (next_position,),
                ),
            )


def expected_parameter_count(hidden_size, count_output_classes):
    hidden = int(hidden_size)
    counts = int(count_output_classes)
    if hidden not in MODEL_POINTS["lstm"] or counts < 1:
        raise ValueError("unsupported realized SPP v25 dimensions")
    # projection + one-layer LSTM + rank fusion + count head + 2-way fill
    # head + two fill-specific 58-bit payload heads.
    return 9 * hidden * hidden + (191 + counts) * hidden + counts + 118


def model_tag(family, size):
    size = int(size)
    if family != "lstm" or size not in MODEL_POINTS["lstm"]:
        raise ValueError("unsupported SPP v25 model point")
    return MODEL_TAG_PREFIX + str(size)


def describe_model_points():
    points = [{
        "family": "lstm", "size": size, "pair_id": pair_id,
        "tag": model_tag("lstm", size),
        "parameter_count_is_dataset_dependent": True,
    } for size, pair_id in sorted(MODEL_POINTS["lstm"].items())]
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
            "seed": SEED, "epochs": EPOCHS, "chunk_len": CHUNK_LEN,
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
        "fill_training_objective": FILL_OBJECTIVE,
        "delta_bit_training_objective": DELTA_BIT_OBJECTIVE,
        "per_callback_objective": (
            "count_CE_plus_sum_real_rank_fill_CE_plus_sum_all_58_bit_NLL"
        ),
        "training_and_guard_objective_identical": True,
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
        "runtime_encoding": "lossless_58_bit_cache_line_plus_one_DEMAND_FILL_kind_bit",
        "external_input_fields": list(EXTERNAL_INPUT_FIELDS),
        "model_does_not_use_pc": True,
        "core_type": CORE_TYPE,
        "global_core": "one_global_chronological_single_layer_LSTM_state",
        "core_selection_used": False,
        "event_routed_core_used": False,
        "count_head_used": True,
        "fill_head_used": True,
        "fill_specific_delta_bit_heads_used": True,
        "both_fill_bit_heads_require_train_supervision": True,
        "joint_action_token_head_used": False,
        "action_vocabulary_used": False,
        "other_token_used": False,
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
        "action_loss_scope": "teacher_action_ranks_only",
        "delta_payload_encoding": "exact_58_bit_modular_line_delta",
        "delta_payload_float_or_clip_used": False,
        "probability_threshold_used": False,
        "threshold_related_hardcodes_used": False,
        "inference_policy_hardcodes_used": False,
        "same_page_rule_used_by_neural_inference": False,
        "normal_policy_templates_used_by_neural_inference": False,
        "neural_degree_cap": None,
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_previous_predicted_action_used_as_input": False,
        "decoder_previous_sampled_action_used_as_input": False,
        "rank_logits_conditionally_independent_of_previous_actions": True,
        "rank_code_size": RANK_CODE_SIZE,
        "fill_levels": list(FILL_LEVELS),
        "target_uniqueness_feasibility_mask_used": True,
        "target_uniqueness_ignores_fill_level": True,
        "target_mutation_fallback_used": False,
        "count_reduction_fallback_used": False,
        "infeasible_unique_decode_behavior": "fail_closed",
        "kbest_payload_enumeration_exact": True,
        "fill_and_payload_log_probability_combined": True,
        "source_action_order_preserved": True,
        "stochastic_decoding": False,
        "parameter_formula": "9*H^2 + (191+K)*H + K+118",
        "parameter_count_is_dataset_dependent": True,
        "non_neural_control": {
            "name": "every_callback_TRAIN_modal_delta_FILL_LLC",
            "delta_source": "TRAIN_teacher_action_frequency_only",
            "fill_level": "FILL_LLC", "actions_per_callback": 1,
            "uses_neural_model": False, "excluded_from_neural_claims": True,
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
    if count_statistics([0, 1, 1, 3])["class_order"] != [0, 1, 2, 3]:
        raise RuntimeError("natural count support changed")
    for base, target in ((0, 0), (0, LINE_ADDRESS_MODULUS - 1), (17, 3)):
        delta = modular_delta(base, target)
        if bits_to_modular_delta(modular_delta_bits(delta)) != delta:
            raise RuntimeError("exact 58-bit payload codec changed")

    mixed = [2.0, -3.0, 0.0] + [1.0] * (LINE_ADDRESS_BITS - 3)
    bits, observed_logp, costs = modal_bernoulli_payload(mixed)
    expected_logp = -(
        math.log1p(math.exp(-2.0)) + math.log1p(math.exp(-3.0))
        + math.log(2.0)
        + (LINE_ADDRESS_BITS - 3) * math.log1p(math.exp(-1.0))
    )
    if bits[:3] != [1, 0, 0] or not math.isclose(
        observed_logp, expected_logp, rel_tol=0.0, abs_tol=1e-12
    ) or costs[:3] != [2.0, 3.0, 0.0]:
        raise RuntimeError("mixed-sign Bernoulli modal log-probability changed")
    subsets = list(k_best_flip_subsets([0.4, 0.1, 0.2], 8))
    if len({subset for _, subset in subsets}) != 8 or any(
        subsets[index][0] > subsets[index + 1][0]
        for index in range(len(subsets) - 1)
    ):
        raise RuntimeError("exact k-best subset enumeration changed")

    previous = 0
    for hidden in sorted(MODEL_POINTS["lstm"]):
        parameters = expected_parameter_count(hidden, 4)
        if parameters <= previous:
            raise RuntimeError("parameter count is not monotone")
        previous = parameters
    contract = describe_model_points()
    for key in (
        "joint_action_token_head_used", "action_vocabulary_used",
        "other_token_used", "count_regression_used", "hurdle_head_used",
        "stop_token_used", "stop_padding_used", "loss_class_reweighting_used",
        "decode_prior_correction_used", "core_selection_used",
        "event_routed_core_used", "delta_payload_float_or_clip_used",
        "target_mutation_fallback_used", "count_reduction_fallback_used",
    ):
        if contract[key]:
            raise RuntimeError("{} leaked into SPP v25".format(key))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--describe-model-points", action="store_true")
    parser.add_argument("--tags-csv", action="store_true")
    parser.add_argument("--base-tag", action="store_true")
    parser.add_argument("--field")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    selected = sum((args.json, args.describe_model_points, args.tags_csv,
                    args.base_tag, args.field is not None, args.self_test))
    if selected != 1:
        parser.error("select exactly one output mode")
    contract = describe_model_points()
    if args.self_test:
        self_test_contract(); print("PASS")
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
