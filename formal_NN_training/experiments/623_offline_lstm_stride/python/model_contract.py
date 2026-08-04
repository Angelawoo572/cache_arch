#!/usr/bin/env python3
"""Pure-stdlib single source for the Stride v19 model-point contract."""
import json
from decimal import Decimal, InvalidOperation


TRACE = "623.xalancbmk_s-700B"
POLICY = "stride"
RUN_ID = "623_offline_lstm_stride_global_local_grammar_v19_seed7"
EXPERIMENT_REVISION = "stride_source_input_variable_delta_free_running_v9"
MODEL_REVISION = "chronological_global_pc_local_stop_emit_leb128_v19"
DECODER_TRAINING_MODE = (
    "teacher_prefix_sequence_nll_with_isolated_hard_sampled_main_rollout"
)
REQUEST_COUNT_OBJECTIVE = (
    "on_policy_STOP_EMIT_prefix_nll_with_STOP_after_teacher_K_until_sampled_STOP"
)
DELTA_OBJECTIVE = (
    "canonical_autoregressive_ZigZag_LEB128_teacher_prefix_sequence_nll"
)
MODEL_TAG_PREFIX = "global_local_grammar_stride_lstm_h"
MODEL_POINTS = {"lstm": {8: "p0", 16: "p1"}}

ADDRESS_BITS = 64
CACHE_LINE_BYTES = 64
CACHE_LINE_OFFSET_BITS = CACHE_LINE_BYTES.bit_length() - 1
LINE_NUMBER_BITS = ADDRESS_BITS - CACHE_LINE_OFFSET_BITS
REUSE_AGE_BITS = 64
RAW_FEATURES = ADDRESS_BITS + LINE_NUMBER_BITS
LOCAL_FEATURES = LINE_NUMBER_BITS * 2 + REUSE_AGE_BITS + 1
RUNTIME_FEATURES = RAW_FEATURES + LOCAL_FEATURES
LEB128_PAYLOAD_BITS = 7
LEB128_MAX_BYTES = (
    LINE_NUMBER_BITS + LEB128_PAYLOAD_BITS - 1
) // LEB128_PAYLOAD_BITS
SAMPLER_GRID_BITS = 53
SAMPLER_GRID_POINTS = 1 << SAMPLER_GRID_BITS
SAMPLER_MIN_UNIFORM = 0.5 / float(SAMPLER_GRID_POINTS)
NONTERMINATION_WATCHDOG_RANKS = SAMPLER_GRID_BITS


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
    hidden_size = int(hidden_size)
    if hidden_size % 2:
        raise ValueError("configured hidden size must be even")
    input_size = hidden_size // 2
    return (
        16 * hidden_size * hidden_size
        + 8 * hidden_size * input_size
        + (RAW_FEATURES + LOCAL_FEATURES + 2) * input_size
        + 59 * hidden_size
        + 10
    )


def model_tag(hidden_size):
    return MODEL_TAG_PREFIX + str(int(hidden_size))


def model_points_description():
    points = []
    for hidden_size, pair_id in sorted(MODEL_POINTS["lstm"].items()):
        points.append({
            "model_family": "lstm",
            "model_size": hidden_size,
            "architecture_pair_id": pair_id,
            "model_tag": model_tag(hidden_size),
            "input_projection_size": hidden_size // 2,
            "parameter_count": expected_parameter_count(hidden_size),
        })
    return {
        "run_id": RUN_ID,
        "trace": TRACE,
        "policy": POLICY,
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": MODEL_REVISION,
        "decoder_training_mode": DECODER_TRAINING_MODE,
        "request_count_training_objective": REQUEST_COUNT_OBJECTIVE,
        "delta_training_objective": DELTA_OBJECTIVE,
        "runtime_feature_count": RUNTIME_FEATURES,
        "raw_runtime_feature_count": RAW_FEATURES,
        "pc_local_runtime_feature_count": LOCAL_FEATURES,
        "line_number_bits": LINE_NUMBER_BITS,
        "leb128_max_bytes": LEB128_MAX_BYTES,
        "sampler_grid_bits": SAMPLER_GRID_BITS,
        "sampler_min_uniform": SAMPLER_MIN_UNIFORM,
        "nontermination_watchdog_ranks": NONTERMINATION_WATCHDOG_RANKS,
        "parameter_formula": (
            "16*H^2 + 8*H*E + (RAW_FEATURES+LOCAL_FEATURES+2)*E "
            "+ 59*H + 10; E=H/2"
        ),
        "points": points,
    }


if __name__ == "__main__":
    print(json.dumps(model_points_description(), indent=2, sort_keys=True))
