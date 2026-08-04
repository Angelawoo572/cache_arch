#!/usr/bin/env python3
"""Torch-free single source for the 623 SPP v19 model-point contract."""
import argparse
import json
import re


OPERATION = "train-v19"
EXPERIMENT_REVISION = "spp_source_input_variable_delta_fill_feedback_free_running_v11"
MODEL_REVISION = "routed_page_lstm_rank_grammar_leb128_v19"
DECODER_REVISION = "keyed_stop_emit_zigzag_leb128_target_fill_v19"
RUN_ID = "623_offline_lstm_spp_routed_grammar_v19_seed7"
TRACE = "623.xalancbmk_s-700B"
POLICY = "spp"
EXTERNAL_INPUT_FIELDS = (
    "callback_kind", "invoke_prefetcher.addr", "cache_fill.evicted_addr",
)
ADDRESS_BITS = 64
CACHE_LINE_BYTES = 64
if CACHE_LINE_BYTES < 1 or CACHE_LINE_BYTES & (CACHE_LINE_BYTES - 1):
    raise RuntimeError("cache-line bytes must be a power of two")
CACHE_LINE_SHIFT = CACHE_LINE_BYTES.bit_length() - 1
LINE_ADDRESS_BITS = ADDRESS_BITS - CACHE_LINE_SHIFT
RUNTIME_FEATURE_COUNT = LINE_ADDRESS_BITS + 1
PAGE_BYTES = 4096
if PAGE_BYTES < CACHE_LINE_BYTES or PAGE_BYTES & (PAGE_BYTES - 1):
    raise RuntimeError("architectural page bytes must be a cache-line-aligned power of two")
PAGE_OFFSET_BITS = (PAGE_BYTES // CACHE_LINE_BYTES).bit_length() - 1
BYTE_PAYLOAD_BITS = 7
BYTE_VOCAB = 1 << (BYTE_PAYLOAD_BITS + 1)
LEB128_MAX_BYTES = (LINE_ADDRESS_BITS + BYTE_PAYLOAD_BITS - 1) // BYTE_PAYLOAD_BITS
SAMPLER_GRID_BITS = 52
KEYED_UNIFORM_HALF_BIN = 2.0 ** -(SAMPLER_GRID_BITS + 1)
ACTION_ROLLOUT_WATCHDOG_RANKS = SAMPLER_GRID_BITS
FILL_LEVELS = (2, 4)
MODEL_POINTS = {"lstm": {8: "p0", 16: "p1"}}
PARAMETER_FORMULA = (
    "(F+1)R + 2(8R^2+8R) + (8R^2+12R) + (R^2+2R) + "
    "(4R+1)H + 2(H+1) + (H+1)E + LE + (E+1)V + VE + "
    "6E^2+6E + (A+1)E + 2(H+E+1) + "
    "3H(2E+C)+3H^2+6H; R=H//2; E=H//4; "
    "F=runtime features; A=line bits; L=max codec bytes; "
    "V=byte vocabulary; C=fill classes"
)


def routed_state_size(hidden_size):
    return max(1, int(hidden_size) // 2)


def codec_embed_size(hidden_size):
    return max(1, int(hidden_size) // 4)


def expected_parameter_count(hidden_size):
    hidden = int(hidden_size)
    route = routed_state_size(hidden)
    codec = codec_embed_size(hidden)
    fill_classes = len(FILL_LEVELS)
    input_projection = (RUNTIME_FEATURE_COUNT + 1) * route
    routed_lstm = 2 * (8 * route * route + 8 * route)
    page_lstm = 8 * route * route + 12 * route
    validity = route * route + 2 * route
    fusion = (4 * route + 1) * hidden
    stop_emit = 2 * (hidden + 1)
    byte_condition = (hidden + 1) * codec
    byte_position = LEB128_MAX_BYTES * codec
    byte_head = (codec + 1) * BYTE_VOCAB
    byte_embedding = BYTE_VOCAB * codec
    byte_cell = 6 * codec * codec + 6 * codec
    target_encoder = (LINE_ADDRESS_BITS + 1) * codec
    fill_head = fill_classes * (hidden + codec + 1)
    action_input = 2 * codec + fill_classes
    action_cell = 3 * hidden * action_input + 3 * hidden * hidden + 6 * hidden
    return sum((
        input_projection, routed_lstm, page_lstm, validity, fusion,
        stop_emit, byte_condition, byte_position, byte_head, byte_embedding,
        byte_cell, target_encoder, fill_head, action_cell,
    ))


def model_tag(family, size):
    size = int(size)
    if family != "lstm" or size not in MODEL_POINTS["lstm"]:
        raise ValueError("unsupported SPP v19 model point")
    return "routed_grammar_spp_lstm_h{}".format(size)


def exact_int(value):
    """Parse one integer field without a floating-point round trip."""
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
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": MODEL_REVISION,
        "decoder_revision": DECODER_REVISION,
        "parameter_formula": PARAMETER_FORMULA,
        "runtime_feature_count": RUNTIME_FEATURE_COUNT,
        "run_id": RUN_ID, "trace": TRACE, "policy": POLICY,
        "address_bits": ADDRESS_BITS,
        "cache_line_bytes": CACHE_LINE_BYTES,
        "cache_line_shift": CACHE_LINE_SHIFT,
        "line_address_bits": LINE_ADDRESS_BITS,
        "page_bytes": PAGE_BYTES,
        "page_offset_bits": PAGE_OFFSET_BITS,
        "byte_payload_bits": BYTE_PAYLOAD_BITS,
        "byte_vocab": BYTE_VOCAB,
        "leb128_max_bytes": LEB128_MAX_BYTES,
        "sampler_grid_bits": SAMPLER_GRID_BITS,
        "action_rollout_watchdog_ranks": ACTION_ROLLOUT_WATCHDOG_RANKS,
        "fill_levels": list(FILL_LEVELS),
        "external_input_fields": list(EXTERNAL_INPUT_FIELDS),
        "points": [
            {
                "family": "lstm", "size": size,
                "pair_id": MODEL_POINTS["lstm"][size],
                "tag": model_tag("lstm", size),
                "route_hidden_size": routed_state_size(size),
                "codec_embed_size": codec_embed_size(size),
                "parameter_count": expected_parameter_count(size),
            }
            for size in MODEL_POINTS["lstm"]
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
