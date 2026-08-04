#!/usr/bin/env python3
"""Chronological global/local LSTM with an exact learned action grammar.

The external runtime contract is byte-for-byte unchanged from v18: each
callback exposes only ``pc`` and the current aligned ``addr``.  Captured
Stride requests are supervised labels and the offline-normal comparator. They
never enter the runtime encoder or main rollout state; teacher codec prefixes
advance only an isolated loss-branch likelihood state.

v19 replaces the per-PC-only hurdle/count/scalar decoder with:

* one chronological global LSTM over the complete PC/address stream;
* one dynamically PC-routed local LSTM.  Its input contains lossless causal
  same-PC delta and reuse-age encodings derived only from prior PC/address
  rows;
* a learned sigmoid validity gate that softly controls how much local state is
  fused with the global state;
* a rank-wise STOP/EMIT grammar.  Request count is the first sampled STOP, not
  a hurdle, rounded mean, probability threshold, budget, or degree cap;
* exact signed incremental targets encoded by ZigZag + canonical LEB128.
  Small strides normally require one byte while every legal signed 58-bit
  cache-line increment remains representable in at most nine bytes.

All runtime categorical choices use stateless event/rank/field-keyed
inverse-CDF sampling.  Keys are independent of model capacity, giving strict
common random numbers across the two v19 points.  Codec likelihood is computed
in an isolated teacher-prefix branch; only the hard sampled branch can update
the main recurrent decoder state and target origin used at the next rank.
"""
import argparse
import csv
import gzip
import hashlib
import inspect
import json
import platform
import random
import sys
from collections import Counter, OrderedDict
from pathlib import Path

from model_contract import (
    ADDRESS_BITS, CACHE_LINE_BYTES, CACHE_LINE_OFFSET_BITS,
    DECODER_TRAINING_MODE, DELTA_OBJECTIVE, EXPERIMENT_REVISION,
    LEB128_MAX_BYTES, LEB128_PAYLOAD_BITS, LINE_NUMBER_BITS, LOCAL_FEATURES,
    MODEL_POINTS, MODEL_REVISION, NONTERMINATION_WATCHDOG_RANKS, POLICY,
    RAW_FEATURES, REQUEST_COUNT_OBJECTIVE, REUSE_AGE_BITS, RUN_ID,
    RUNTIME_FEATURES, SAMPLER_GRID_BITS, SAMPLER_GRID_POINTS,
    SAMPLER_MIN_UNIFORM, TRACE,
    expected_parameter_count, model_points_description, model_tag,
    parse_exact_integer,
)

if __name__ == "__main__" and sys.argv[1:] == ["--describe-model-points"]:
    print(json.dumps(model_points_description(), indent=2, sort_keys=True))
    raise SystemExit(0)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formal_NN_training.common.threshold_free_policy import (
    ADDRESS_BITS as COMMON_ADDRESS_BITS,
    CACHE_LINE_BYTES as COMMON_CACHE_LINE_BYTES,
    apply_signed_line_delta, behavior_metrics,
)

if (COMMON_ADDRESS_BITS, COMMON_CACHE_LINE_BYTES) != (
    ADDRESS_BITS, CACHE_LINE_BYTES
):
    raise RuntimeError("shared address contract differs from v19 model contract")

EVENT_LOGGER_SCHEMA = "623_causal_trigger_v5"
CANDIDATE_ATTACHMENT_MODE = "explicit_trigger_event_id"
SAMPLER_REVISION = "splitmix64_event_rank_field_inverse_cdf_crn_v2"
SOURCE_INPUTS = ["pc", "addr"]

LINE_MODULUS = 1 << LINE_NUMBER_BITS
LINE_MASK = LINE_MODULUS - 1
SIGNED_LINE_MIN = -(1 << (LINE_NUMBER_BITS - 1))
SIGNED_LINE_MAX = (1 << (LINE_NUMBER_BITS - 1)) - 1
LEB128_PAYLOAD_CLASSES = 1 << LEB128_PAYLOAD_BITS
STOP = 0
EMIT = 1
FIELD_GRAMMAR = 1
FIELD_PAYLOAD_BIT = 2
FIELD_FINAL_PAYLOAD = 3
FIELD_CONTINUATION = 4
ROLE_CODES = {"train": 0x545241494E, "eval": 0x4556414C}
SAMPLER_DOMAIN = int.from_bytes(
    hashlib.sha256(SAMPLER_REVISION.encode()).digest()[:8], "little"
)
TRACK_DOMAIN = int.from_bytes(
    hashlib.sha256((TRACE + "|" + POLICY).encode()).digest()[:8], "little"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gzip_content_sha256(path):
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_int(value):
    return parse_exact_integer(value)


def load_stream(path):
    rows = []
    occurrences = {}
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "trace", "demand_idx", "pc", "line", "pc_line_occ",
            "logger_schema",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError("{} missing {}".format(path, sorted(missing)))
        for index, row in enumerate(reader):
            pc = as_int(row["pc"])
            line = as_int(row["line"])
            pair = (pc, line)
            expected = occurrences.get(pair, 0)
            occurrence = as_int(row["pc_line_occ"])
            occurrences[pair] = expected + 1
            if (
                row["trace"] != TRACE
                or as_int(row["demand_idx"]) != index
                or row["logger_schema"] != EVENT_LOGGER_SCHEMA
                or occurrence != expected
                or pc < 0 or pc >= (1 << ADDRESS_BITS)
                or line < 0 or line >= LINE_MODULUS
            ):
                raise RuntimeError(
                    "stream identity/ordering failure at row {}".format(index)
                )
            rows.append((pc, line, occurrence))
    if not rows:
        raise RuntimeError("empty stream: {}".format(path))
    return rows


def load_teacher_actions(path, rows):
    """Load captured actions as labels; no returned value is a model input."""
    actions = [[] for _ in rows]
    last_pf_event = -1
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "trace", "policy", "demand_idx", "pc", "line", "pc_line_occ",
            "candidate_rank", "pf_line", "fill_level", "accepted",
            "duplicate", "trigger_event_id", "pf_event_id", "event_distance",
            "match_mode", "logger_schema",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError("{} missing {}".format(path, sorted(missing)))
        for row in reader:
            index = as_int(row["demand_idx"])
            if index < 0 or index >= len(rows):
                raise RuntimeError("teacher action demand_idx out of range")
            pc, line, occurrence = rows[index]
            if (
                row["trace"] != TRACE or row["policy"] != POLICY
                or (
                    as_int(row["pc"]), as_int(row["line"]),
                    as_int(row["pc_line_occ"]),
                ) != (pc, line, occurrence)
                or row["logger_schema"] != EVENT_LOGGER_SCHEMA
                or row["match_mode"] != CANDIDATE_ATTACHMENT_MODE
            ):
                raise RuntimeError(
                    "teacher action identity failure at {}".format(index)
                )
            if as_int(row["candidate_rank"]) != len(actions[index]) + 1:
                raise RuntimeError(
                    "noncontiguous teacher action rank at {}".format(index)
                )
            trigger = as_int(row["trigger_event_id"])
            pf_event = as_int(row["pf_event_id"])
            distance = as_int(row["event_distance"])
            target = as_int(row["pf_line"])
            if (
                pf_event <= last_pf_event or trigger >= pf_event
                or distance != pf_event - trigger
                or target < 0 or target >= LINE_MODULUS
                or as_int(row["fill_level"]) != 2
                or as_int(row["accepted"]) not in (0, 1)
                or as_int(row["duplicate"]) not in (0, 1)
            ):
                raise RuntimeError(
                    "invalid captured Stride action at {}".format(index)
                )
            actions[index].append(target)
            last_pf_event = pf_event
    if not any(actions):
        raise RuntimeError("empty Stride teacher action stream {}".format(path))
    return actions


def _unsigned_bits(values, width):
    """Lossless LSB-first bit encoding stored compactly as uint8."""
    integers = [int(value) for value in values]
    limit = 1 << int(width)
    if any(value < 0 or value >= limit for value in integers):
        raise RuntimeError("runtime integer exceeds its lossless bit width")
    array = np.asarray(integers, dtype=np.uint64)
    shifts = np.arange(width, dtype=np.uint64)
    return (
        (array[:, None] >> shifts[None, :]) & np.uint64(1)
    ).astype(np.uint8)


def _canonical_signed_line_delta(current, previous):
    value = (int(current) - int(previous)) & LINE_MASK
    if value >= (1 << (LINE_NUMBER_BITS - 1)):
        value -= LINE_MODULUS
    return value


def runtime_features(rows):
    """Encode raw input plus only causal derivatives of prior PC/address rows."""
    if CACHE_LINE_BYTES != (1 << CACHE_LINE_OFFSET_BITS):
        raise RuntimeError("cache-line bytes must be a power of two")
    pcs = [pc for pc, _, _ in rows]
    lines = [line for _, line, _ in rows]
    raw = np.concatenate([
        _unsigned_bits(pcs, ADDRESS_BITS),
        _unsigned_bits(lines, LINE_NUMBER_BITS),
    ], axis=1)

    last_by_pc = {}
    delta_codes = np.zeros(len(rows), dtype=np.uint64)
    reuse_ages = np.zeros(len(rows), dtype=np.uint64)
    has_previous = np.zeros((len(rows), 1), dtype=np.uint8)
    for index, (pc, line, _) in enumerate(rows):
        previous = last_by_pc.get(pc)
        if previous is not None:
            previous_line, previous_index = previous
            signed = _canonical_signed_line_delta(line, previous_line)
            delta_codes[index] = np.uint64(signed & LINE_MASK)
            reuse_ages[index] = np.uint64(index - previous_index)
            has_previous[index, 0] = 1
        last_by_pc[pc] = (line, index)

    local = np.concatenate([
        _unsigned_bits(lines, LINE_NUMBER_BITS),
        _unsigned_bits(delta_codes, LINE_NUMBER_BITS),
        _unsigned_bits(reuse_ages, REUSE_AGE_BITS),
        has_previous,
    ], axis=1)
    if raw.shape[1] != RAW_FEATURES or local.shape[1] != LOCAL_FEATURES:
        raise RuntimeError("causal runtime feature width changed")
    return {"raw": raw, "local": local}


def runtime_encoder_sha256():
    payload = {
        "entrypoint_source": inspect.getsource(runtime_features),
        "bit_primitive_source": inspect.getsource(_unsigned_bits),
        "delta_primitive_source": inspect.getsource(
            _canonical_signed_line_delta
        ),
        "external_fields": SOURCE_INPUTS,
        "raw_feature_count": RAW_FEATURES,
        "local_feature_count": LOCAL_FEATURES,
        "total_feature_count": RUNTIME_FEATURES,
        "line_number_bits": LINE_NUMBER_BITS,
        "reuse_age_bits": REUSE_AGE_BITS,
        "derived_features_use_labels": False,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def _zigzag_encode(value):
    value = int(value)
    if value < SIGNED_LINE_MIN or value > SIGNED_LINE_MAX:
        raise RuntimeError("signed line increment exceeds 58-bit domain")
    return 2 * value if value >= 0 else -2 * value - 1


def _zigzag_decode(value):
    value = int(value)
    if value < 0 or value > LINE_MASK:
        raise RuntimeError("ZigZag value exceeds 58-bit domain")
    return value // 2 if value % 2 == 0 else -(value // 2) - 1


def _leb128_encode_signed_increment(value):
    remaining = _zigzag_encode(value)
    encoded = []
    while True:
        payload = remaining & 0x7F
        remaining >>= 7
        encoded.append(payload | (0x80 if remaining else 0))
        if not remaining:
            break
    if len(encoded) > LEB128_MAX_BYTES:
        raise RuntimeError("legal signed increment exceeded nine LEB128 bytes")
    return encoded


def _leb128_decode_signed_increment(encoded):
    if not encoded or len(encoded) > LEB128_MAX_BYTES:
        raise RuntimeError("invalid LEB128 byte count")
    value = 0
    terminated = False
    for position, byte in enumerate(encoded):
        byte = int(byte)
        if byte < 0 or byte > 255:
            raise RuntimeError("invalid LEB128 byte")
        payload = byte & 0x7F
        if position == LEB128_MAX_BYTES - 1 and payload >= 4:
            raise RuntimeError("LEB128 payload exceeds 58-bit address width")
        if position > 0 and not (byte & 0x80) and payload == 0:
            raise RuntimeError("noncanonical zero terminal LEB128 group")
        value |= payload << (7 * position)
        if not (byte & 0x80):
            terminated = True
            if position + 1 != len(encoded):
                raise RuntimeError("bytes follow LEB128 termination")
            break
    if not terminated or value > LINE_MASK:
        raise RuntimeError("unterminated/out-of-range LEB128 value")
    return _zigzag_decode(value)


def teacher_grammar(base_lines, actions):
    """Keep absolute teacher targets as labels; never precompute feedback."""
    if len(base_lines) != len(actions):
        raise RuntimeError("teacher grammar length mismatch")
    counts = np.asarray([len(items) for items in actions], dtype=np.int64)
    maximum = int(counts.max()) if len(counts) else 0
    targets = np.zeros((len(actions), maximum), dtype=np.uint64)
    for row, items in enumerate(actions):
        if items:
            targets[row, :len(items)] = np.asarray(items, dtype=np.uint64)
    return {
        "counts": counts,
        "targets": targets,
        "base_lines": np.asarray(base_lines, dtype=np.uint64),
    }


def _splitmix64(values):
    values = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        values = (values ^ (values >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
    return values ^ (values >> np.uint64(31))


def _keyed_uniform(
    event_ids, decoder_seed, role, epoch, action_rank, field, position,
):
    """One stateless U(0,1) variate per categorical decision."""
    if role not in ROLE_CODES:
        raise RuntimeError("unknown decoder sampling role {}".format(role))
    event_ids = np.asarray(event_ids, dtype=np.uint64)
    mask = (1 << 64) - 1
    constants = (
        int(decoder_seed) * 0xD6E8FEB86659FD93,
        int(ROLE_CODES[role]) * 0xA5A3564E27F8862B,
        int(epoch) * 0x9E3779B97F4A7C15,
        int(action_rank) * 0xBF58476D1CE4E5B9,
        int(field) * 0x94D049BB133111EB,
        int(position) * 0xDB4F0B9175AE2165,
        SAMPLER_DOMAIN,
        TRACK_DOMAIN,
    )
    key = event_ids.copy()
    for value in constants:
        key ^= np.uint64(value & mask)
        key = _splitmix64(key)
    bits = _splitmix64(key)
    return (
        (bits >> np.uint64(11)).astype(np.float64) + 0.5
    ) / float(1 << 53)


def _inverse_cdf_sample(
    logits, event_ids, decoder_seed, role, epoch, action_rank, field,
    position,
):
    if logits.ndim != 2 or logits.shape[0] != len(event_ids):
        raise RuntimeError("categorical sampler shape mismatch")
    if (
        torch.isnan(logits).any() or torch.isposinf(logits).any()
        or not torch.isfinite(logits).any(dim=1).all()
    ):
        raise RuntimeError("invalid categorical logits")
    uniforms = torch.from_numpy(_keyed_uniform(
        event_ids, decoder_seed, role, epoch, action_rank, field, position,
    )).to(device=logits.device, dtype=torch.float64)
    cumulative = torch.softmax(logits.to(torch.float64), dim=-1).cumsum(
        dim=-1
    )
    choices = (cumulative < uniforms.unsqueeze(1)).sum(dim=1)
    return choices.clamp(max=logits.shape[1] - 1).to(torch.long)


def _assert_stop_representable(logits):
    """Fail if finite-grid inverse-CDF cannot ever draw STOP.

    The keyed sampler uses open-midpoint 53-bit uniforms.  Its exact minimum
    is half a grid bin; rejecting STOP mass below that value is a numerical
    support check, not a policy threshold.
    """
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise RuntimeError("STOP support check requires binary grammar logits")
    probabilities = torch.softmax(logits.to(torch.float64), dim=1)
    if torch.any(probabilities[:, STOP] < SAMPLER_MIN_UNIFORM):
        raise RuntimeError(
            "learned STOP mass is not representable on the 53-bit sampler grid"
        )


def _assert_action_rank_within_watchdog(action_rank):
    """Abort a nonterminating run; never truncate or synthesize STOP."""
    if int(action_rank) >= NONTERMINATION_WATCHDOG_RANKS:
        raise RuntimeError(
            "sampled action grammar reached the fail-closed numerical "
            "nontermination watchdog"
        )


class GlobalLocalGrammarStrideLSTM(nn.Module):
    """Dual-time-scale recurrent encoder plus exact categorical decoder."""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = int(hidden_size)
        if self.hidden_size % 2:
            raise ValueError("configured hidden size must be even")
        self.input_size = self.hidden_size // 2
        self.global_input_projection = nn.Linear(RAW_FEATURES, self.input_size)
        self.local_input_projection = nn.Linear(LOCAL_FEATURES, self.input_size)
        self.global_lstm = nn.LSTM(
            self.input_size, hidden_size, batch_first=True
        )
        self.local_lstm = nn.LSTM(
            self.input_size, hidden_size, batch_first=True
        )
        self.local_validity = nn.Linear(2 * hidden_size, 1)
        self.fusion = nn.Linear(2 * hidden_size, hidden_size)

        self.action_cell = nn.GRUCell(hidden_size, hidden_size)
        self.grammar_head = nn.Linear(hidden_size, 2)
        self.continuation_head = nn.Linear(hidden_size, 2)
        self.payload_bit_head = nn.Linear(hidden_size, 2)
        self.final_payload_head = nn.Linear(hidden_size, 3)
        self.grammar_embedding = nn.Embedding(2, hidden_size)
        self.continuation_embedding = nn.Embedding(2, hidden_size)
        self.payload_bit_embedding = nn.Embedding(2, hidden_size)
        self.final_payload_embedding = nn.Embedding(3, hidden_size)
        self.byte_position_embedding = nn.Embedding(
            LEB128_MAX_BYTES, hidden_size
        )
        self.payload_bit_position_embedding = nn.Embedding(
            LEB128_PAYLOAD_BITS, hidden_size
        )

def _pc_groups(pcs):
    grouped = OrderedDict()
    for position, pc in enumerate(pcs):
        grouped.setdefault(int(pc), []).append(position)
    return sorted(
        grouped.items(), key=lambda item: (-len(item[1]), item[1][0])
    )


def _initial_local_state(state_map, keys, hidden_size, device):
    h_values = []
    c_values = []
    for key in keys:
        if key in state_map:
            h_value, c_value = state_map[key]
        else:
            h_value = torch.zeros(hidden_size, device=device)
            c_value = torch.zeros(hidden_size, device=device)
        h_values.append(h_value)
        c_values.append(c_value)
    return (
        torch.stack(h_values, dim=0).unsqueeze(0),
        torch.stack(c_values, dim=0).unsqueeze(0),
    )


def _encode_chunk(
    model, raw, local, pcs, global_state, local_state_map,
):
    """Encode one chronological TBPTT chunk with causal PC-local routing."""
    groups = _pc_groups(pcs)
    lengths = [len(indices) for _, indices in groups]
    padded = torch.zeros(
        len(groups), max(lengths), LOCAL_FEATURES,
        dtype=local.dtype, device=local.device,
    )
    for row, (_, indices) in enumerate(groups):
        index = torch.as_tensor(indices, dtype=torch.long, device=local.device)
        padded[row, :len(indices)] = local.index_select(0, index)
    projected_local = torch.tanh(model.local_input_projection(padded))
    packed = pack_padded_sequence(
        projected_local, lengths, batch_first=True, enforce_sorted=True
    )
    initial_local = _initial_local_state(
        local_state_map, [pc for pc, _ in groups],
        model.hidden_size, local.device,
    )
    packed_output, final_local = model.local_lstm(packed, initial_local)
    local_padded, _ = pad_packed_sequence(
        packed_output, batch_first=True, total_length=max(lengths)
    )
    local_context = torch.zeros(
        len(pcs), model.hidden_size, dtype=local.dtype, device=local.device
    )
    for row, (pc, indices) in enumerate(groups):
        index = torch.as_tensor(indices, dtype=torch.long, device=local.device)
        local_context = local_context.index_copy(
            0, index, local_padded[row, :len(indices)]
        )
        local_state_map[pc] = (
            final_local[0][0, row].detach(),
            final_local[1][0, row].detach(),
        )

    projected_global = torch.tanh(model.global_input_projection(raw))
    global_output, final_global = model.global_lstm(
        projected_global.unsqueeze(0), global_state
    )
    global_context = global_output.squeeze(0)
    validity = torch.sigmoid(model.local_validity(torch.cat(
        [global_context, local_context], dim=1
    )))
    fused = torch.tanh(model.fusion(torch.cat(
        [global_context, validity * local_context], dim=1
    )))
    detached_global = (
        final_global[0].detach(), final_global[1].detach()
    )
    return fused, detached_global, validity.squeeze(1)


def state_router_sha256():
    payload = (
        inspect.getsource(_pc_groups)
        + inspect.getsource(_initial_local_state)
        + inspect.getsource(_encode_chunk)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def sampler_source_sha256():
    payload = (
        inspect.getsource(_splitmix64)
        + inspect.getsource(_keyed_uniform)
        + inspect.getsource(_inverse_cdf_sample)
        + inspect.getsource(_assert_stop_representable)
        + inspect.getsource(_assert_action_rank_within_watchdog)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def codec_source_sha256():
    payload = (
        inspect.getsource(_zigzag_encode)
        + inspect.getsource(_zigzag_decode)
        + inspect.getsource(_leb128_encode_signed_increment)
        + inspect.getsource(_leb128_decode_signed_increment)
        + inspect.getsource(_legal_byte_mask)
        + inspect.getsource(_codec_position_embeddings)
        + inspect.getsource(_codec_payload_bit_logits)
        + inspect.getsource(_codec_payload_bit_transition)
        + inspect.getsource(_codec_continuation_logits)
        + inspect.getsource(_codec_continuation_transition)
        + inspect.getsource(_codec_final_payload_logits)
        + inspect.getsource(_codec_final_payload_transition)
        + inspect.getsource(_sample_codec_rollout)
        + inspect.getsource(_teacher_codec_nll)
        + inspect.getsource(_sample_codec_path)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _legal_byte_mask(position):
    """Return the canonical LEB128 support for one byte position."""
    if position < 0 or position >= LEB128_MAX_BYTES:
        raise RuntimeError("LEB128 byte position out of range")
    mask = np.zeros((2, LEB128_PAYLOAD_CLASSES), dtype=np.bool_)
    if position == 0:
        mask[:, :] = True
    elif position < LEB128_MAX_BYTES - 1:
        mask[EMIT, :] = True       # continuation bit set
        mask[STOP, 1:] = True      # terminal high group must be nonzero
    else:
        mask[STOP, 1:4] = True     # final 58-bit group is 1, 2, or 3
    return mask


def _codec_position_embeddings(model, state, byte_position):
    positions = torch.full(
        (len(state),), int(byte_position), dtype=torch.long,
        device=state.device,
    )
    return model.byte_position_embedding(positions)


def _codec_payload_bit_logits(
    model, state, byte_embedding, bit_position,
):
    positions = torch.full(
        (len(state),), int(bit_position), dtype=torch.long,
        device=state.device,
    )
    bit_embedding = model.payload_bit_position_embedding(positions)
    logits = model.payload_bit_head(
        state + byte_embedding + bit_embedding
    )
    return logits, bit_embedding


def _codec_payload_bit_transition(
    model, state, token, byte_embedding, bit_embedding,
):
    return model.action_cell(
        model.payload_bit_embedding(token) + byte_embedding + bit_embedding,
        state,
    )


def _codec_continuation_logits(model, state, byte_embedding, payload, position):
    """Apply the exact same canonical mask in likelihood and rollout."""
    logits = model.continuation_head(state + byte_embedding)
    payload_numpy = payload.detach().cpu().numpy().astype(np.int64)
    legal = _legal_byte_mask(position)[:, payload_numpy].T
    legal_tensor = torch.from_numpy(legal).to(device=state.device)
    return logits.masked_fill(~legal_tensor, float("-inf"))


def _codec_continuation_transition(
    model, state, token, byte_embedding,
):
    return model.action_cell(
        model.continuation_embedding(token) + byte_embedding, state
    )


def _codec_final_payload_logits(model, state, byte_embedding):
    return model.final_payload_head(state + byte_embedding)


def _codec_final_payload_transition(
    model, state, payload_class, byte_embedding,
):
    return model.action_cell(
        model.final_payload_embedding(payload_class) + byte_embedding, state
    )


def _sample_codec_rollout(
    model, state, event_ids, decoder_seed, role, epoch, action_rank,
):
    """Sample the main codec path; this function cannot accept labels."""
    event_ids = np.asarray(event_ids, dtype=np.int64)
    if len(state) != len(event_ids):
        raise RuntimeError("codec state/event length mismatch")
    active_numpy = np.arange(len(event_ids), dtype=np.int64)
    values = torch.zeros(len(event_ids), dtype=torch.long, device=state.device)
    lengths = np.zeros(len(event_ids), dtype=np.int64)
    for byte_position in range(LEB128_MAX_BYTES):
        if not len(active_numpy):
            break
        active = torch.from_numpy(active_numpy).to(
            device=state.device, dtype=torch.long
        )
        active_state = state.index_select(0, active)
        active_events = event_ids[active_numpy]
        byte_embedding = _codec_position_embeddings(
            model, active_state, byte_position
        )
        payload = torch.zeros(
            len(active_numpy), dtype=torch.long, device=state.device
        )

        if byte_position == LEB128_MAX_BYTES - 1:
            payload_logits = _codec_final_payload_logits(
                model, active_state, byte_embedding
            )
            payload_class = _inverse_cdf_sample(
                payload_logits, active_events, decoder_seed, role, epoch,
                action_rank, FIELD_FINAL_PAYLOAD, byte_position,
            )
            payload = payload_class + 1
            active_state = _codec_final_payload_transition(
                model, active_state, payload_class, byte_embedding
            )
            continuation = torch.zeros_like(payload)
        else:
            for bit_position in range(LEB128_PAYLOAD_BITS):
                bit_logits, bit_embedding = _codec_payload_bit_logits(
                    model, active_state, byte_embedding, bit_position
                )
                sampled_bit = _inverse_cdf_sample(
                    bit_logits, active_events, decoder_seed, role, epoch,
                    action_rank, FIELD_PAYLOAD_BIT,
                    byte_position * LEB128_PAYLOAD_BITS + bit_position,
                )
                payload = payload | (
                    sampled_bit.to(torch.long) << bit_position
                )
                active_state = _codec_payload_bit_transition(
                    model, active_state, sampled_bit, byte_embedding,
                    bit_embedding,
                )
            continuation_logits = _codec_continuation_logits(
                model, active_state, byte_embedding, payload, byte_position
            )
            continuation = _inverse_cdf_sample(
                continuation_logits, active_events, decoder_seed, role,
                epoch, action_rank, FIELD_CONTINUATION, byte_position,
            )
            active_state = _codec_continuation_transition(
                model, active_state, continuation, byte_embedding
            )

        current = values.index_select(0, active)
        current = current | (payload << (7 * byte_position))
        values = values.index_copy(0, active, current)
        state = state.index_copy(0, active, active_state)
        lengths[active_numpy] += 1
        keep = continuation.detach().cpu().numpy().astype(bool)
        active_numpy = active_numpy[keep]

    if len(active_numpy):
        raise RuntimeError("canonical codec did not terminate by final byte")
    signs = values & 1
    magnitudes = values >> 1
    signed = torch.where(signs == 0, magnitudes, -magnitudes - 1)
    return state, signed, lengths


def _teacher_codec_nll(model, state, target_increments, target_valid):
    """Full teacher-prefix NLL in a loss-only, isolated recurrent branch."""
    target_valid = np.asarray(target_valid, dtype=np.bool_)
    target_increments = np.asarray(target_increments, dtype=np.int64)
    if len(state) != len(target_valid) or len(state) != len(target_increments):
        raise RuntimeError("codec target/state length mismatch")
    target_lengths = np.zeros(len(state), dtype=np.int64)
    target_bytes = np.zeros((len(state), LEB128_MAX_BYTES), dtype=np.uint8)
    for index in np.flatnonzero(target_valid):
        encoded = _leb128_encode_signed_increment(target_increments[index])
        target_lengths[index] = len(encoded)
        target_bytes[index, :len(encoded)] = encoded

    branch_state = state
    active_numpy = np.flatnonzero(target_valid).astype(np.int64)
    total_nll = state.new_zeros(())
    payload_atoms = 0
    termination_atoms = 0
    atoms_by_position = Counter()
    for byte_position in range(LEB128_MAX_BYTES):
        if not len(active_numpy):
            break
        if np.any(target_lengths[active_numpy] <= byte_position):
            raise RuntimeError("teacher codec branch passed its terminal byte")
        active = torch.from_numpy(active_numpy).to(
            device=state.device, dtype=torch.long
        )
        active_state = branch_state.index_select(0, active)
        byte_embedding = _codec_position_embeddings(
            model, active_state, byte_position
        )
        target_payload_numpy = (
            target_bytes[active_numpy, byte_position] & np.uint8(0x7F)
        ).astype(np.int64)
        target_payload = torch.from_numpy(target_payload_numpy).to(
            device=state.device, dtype=torch.long
        )

        if byte_position == LEB128_MAX_BYTES - 1:
            if np.any(target_payload_numpy < 1) or np.any(
                target_payload_numpy > 3
            ):
                raise RuntimeError("illegal teacher final LEB128 payload")
            logits = _codec_final_payload_logits(
                model, active_state, byte_embedding
            )
            target_class = target_payload - 1
            total_nll = total_nll + F.cross_entropy(
                logits, target_class, reduction="sum"
            )
            payload_atoms += len(active_numpy)
            atoms_by_position["byte8.final_payload"] += len(active_numpy)
            active_state = _codec_final_payload_transition(
                model, active_state, target_class, byte_embedding
            )
            target_continuation = torch.zeros_like(target_payload)
        else:
            for bit_position in range(LEB128_PAYLOAD_BITS):
                bit_logits, bit_embedding = _codec_payload_bit_logits(
                    model, active_state, byte_embedding, bit_position
                )
                target_bit = (target_payload >> bit_position) & 1
                total_nll = total_nll + F.cross_entropy(
                    bit_logits, target_bit, reduction="sum"
                )
                payload_atoms += len(active_numpy)
                atoms_by_position[
                    "byte{}.bit{}".format(byte_position, bit_position)
                ] += len(active_numpy)
                active_state = _codec_payload_bit_transition(
                    model, active_state, target_bit, byte_embedding,
                    bit_embedding,
                )

            target_continuation_numpy = (
                (
                    target_bytes[active_numpy, byte_position]
                    & np.uint8(0x80)
                ) != 0
            ).astype(np.int64)
            target_continuation = torch.from_numpy(
                target_continuation_numpy
            ).to(device=state.device, dtype=torch.long)
            continuation_logits = _codec_continuation_logits(
                model, active_state, byte_embedding, target_payload,
                byte_position,
            )
            selected = continuation_logits.gather(
                1, target_continuation.unsqueeze(1)
            ).squeeze(1)
            if not torch.isfinite(selected).all():
                raise RuntimeError("teacher continuation violates canonical mask")
            total_nll = total_nll + F.cross_entropy(
                continuation_logits, target_continuation, reduction="sum"
            )
            termination_atoms += len(active_numpy)
            atoms_by_position[
                "byte{}.continuation".format(byte_position)
            ] += len(active_numpy)
            active_state = _codec_continuation_transition(
                model, active_state, target_continuation, byte_embedding
            )

        branch_state = branch_state.index_copy(0, active, active_state)
        keep = target_continuation_numpy.astype(bool) if (
            byte_position < LEB128_MAX_BYTES - 1
        ) else np.zeros(len(active_numpy), dtype=np.bool_)
        active_numpy = active_numpy[keep]

    if len(active_numpy):
        raise RuntimeError("teacher codec did not terminate by final byte")
    return total_nll, {
        "codec_exact_atoms": payload_atoms,
        "codec_termination_atoms": termination_atoms,
        "codec_teacher_prefix_atoms_by_position": dict(atoms_by_position),
    }


def _sample_codec_path(
    model, state, event_ids, decoder_seed, role, epoch, action_rank,
    target_increments=None, target_valid=None,
):
    """Run isolated teacher-loss and sampled-main codec branches."""
    event_ids = np.asarray(event_ids, dtype=np.int64)
    if len(state) != len(event_ids):
        raise RuntimeError("codec state/event length mismatch")
    sampled_state, signed, lengths = _sample_codec_rollout(
        model, state, event_ids, decoder_seed, role, epoch, action_rank
    )
    if target_valid is None:
        target_valid = np.zeros(len(event_ids), dtype=np.bool_)
    else:
        target_valid = np.asarray(target_valid, dtype=np.bool_)
    if target_increments is None:
        target_increments = np.zeros(len(event_ids), dtype=np.int64)
    else:
        target_increments = np.asarray(target_increments, dtype=np.int64)
    teacher_nll, components = _teacher_codec_nll(
        model, state, target_increments, target_valid
    )
    return sampled_state, signed, lengths, teacher_nll, components


def _sequence_nll(
    model, context, grammar, event_ids, decoder_seed, epoch,
):
    """True on-policy STOP/EMIT and canonical-byte sequence NLL."""
    counts = grammar["counts"]
    targets = grammar["targets"]
    base_lines = grammar["base_lines"]
    event_ids = np.asarray(event_ids, dtype=np.int64)
    if (
        len(context) != len(counts) or len(event_ids) != len(counts)
        or len(base_lines) != len(counts)
    ):
        raise RuntimeError("grammar batch length mismatch")

    state = context
    origins = torch.as_tensor(
        base_lines.astype(np.int64), dtype=torch.long, device=context.device
    )
    active_numpy = np.arange(len(counts), dtype=np.int64)
    total = context.new_zeros(())
    grammar_nll_sum = 0.0
    codec_nll_sum = 0.0
    grammar_atoms = 0
    codec_exact_atoms = 0
    codec_termination_atoms = 0
    codec_teacher_prefix_atoms_by_position = Counter()
    sampled_emits = 0
    sampled_stops = 0
    action_rank = 0

    # Rows leave active_numpy permanently on their first sampled STOP.  If a
    # row emits beyond teacher K, STOP remains its grammar label until the
    # sampled policy actually stops; no teacher count reactivates a row.
    while len(active_numpy):
        _assert_action_rank_within_watchdog(action_rank)
        active = torch.from_numpy(active_numpy).to(
            device=context.device, dtype=torch.long
        )
        active_state = state.index_select(0, active)
        active_events = event_ids[active_numpy]
        labels_numpy = (action_rank < counts[active_numpy]).astype(np.int64)
        labels = torch.from_numpy(labels_numpy).to(context.device)
        logits = model.grammar_head(active_state)
        grammar_loss = F.cross_entropy(logits, labels, reduction="sum")
        total = total + grammar_loss
        grammar_nll_sum += float(grammar_loss.detach().item())
        grammar_atoms += len(active_numpy)
        _assert_stop_representable(logits)

        sampled = _inverse_cdf_sample(
            logits, active_events, decoder_seed, "train", epoch,
            action_rank, FIELD_GRAMMAR, 0,
        )
        sampled_numpy = sampled.detach().cpu().numpy()
        after_grammar = model.action_cell(
            model.grammar_embedding(sampled), active_state
        )
        emit_local = np.flatnonzero(sampled_numpy == EMIT)
        sampled_emits += len(emit_local)
        sampled_stops += len(active_numpy) - len(emit_local)
        if not len(emit_local):
            break

        emit_index = torch.from_numpy(emit_local).to(
            device=context.device, dtype=torch.long
        )
        emit_rows = active_numpy[emit_local]
        emit_events = active_events[emit_local]
        emit_origins = origins.index_select(0, active).index_select(
            0, emit_index
        )
        teacher_valid = action_rank < counts[emit_rows]
        target_increments = np.zeros(len(emit_rows), dtype=np.int64)
        emit_origins_numpy = emit_origins.detach().cpu().numpy()
        for local_position in np.flatnonzero(teacher_valid):
            target_increments[local_position] = _canonical_signed_line_delta(
                int(targets[emit_rows[local_position], action_rank]),
                int(emit_origins_numpy[local_position]),
            )

        (
            codec_state, signed, _, codec_loss, codec_components,
        ) = _sample_codec_path(
            model,
            after_grammar.index_select(0, emit_index),
            emit_events,
            decoder_seed,
            "train",
            epoch,
            action_rank,
            target_increments=target_increments,
            target_valid=teacher_valid,
        )
        total = total + codec_loss
        codec_nll_sum += float(codec_loss.detach().item())
        codec_exact_atoms += codec_components["codec_exact_atoms"]
        codec_termination_atoms += codec_components[
            "codec_termination_atoms"
        ]
        codec_teacher_prefix_atoms_by_position.update(
            codec_components["codec_teacher_prefix_atoms_by_position"]
        )

        emitted_targets = (emit_origins + signed) & LINE_MASK
        global_emit = active.index_select(0, emit_index)
        origins = origins.index_copy(0, global_emit, emitted_targets)
        after_grammar = after_grammar.index_copy(
            0, emit_index, codec_state
        )
        state = state.index_copy(0, active, after_grammar)
        active_numpy = emit_rows
        action_rank += 1

    atoms = grammar_atoms + codec_exact_atoms + codec_termination_atoms
    if atoms <= 0:
        raise RuntimeError("training chunk contains no sequence atoms")
    return total / float(atoms), {
        "total_nll_sum": float(total.detach().item()),
        "grammar_nll_sum": grammar_nll_sum,
        "codec_nll_sum": codec_nll_sum,
        "grammar_atoms": grammar_atoms,
        "codec_exact_atoms": codec_exact_atoms,
        "codec_termination_atoms": codec_termination_atoms,
        "codec_teacher_prefix_atoms_by_position": dict(
            codec_teacher_prefix_atoms_by_position
        ),
        "categorical_atoms": atoms,
        "sampled_emit_decisions": sampled_emits,
        "sampled_stop_decisions": sampled_stops,
    }


def _slice_grammar(grammar, start, stop):
    return {key: value[start:stop] for key, value in grammar.items()}


def train_model(
    model, rows, runtime, grammar, device, epochs, chunk_len,
    accumulate_chunks, learning_rate, decoder_seed,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    pcs = np.asarray([pc for pc, _, _ in rows], dtype=np.uint64)
    for epoch in range(1, epochs + 1):
        model.train()
        global_state = None
        local_state_map = {}
        totals = Counter()
        codec_teacher_prefix_atoms_by_position = Counter()
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        pending_categorical_atoms = 0
        steps = 0
        validity_sum = 0.0
        validity_atoms = 0
        for start in range(0, len(rows), chunk_len):
            stop = min(start + chunk_len, len(rows))
            raw = torch.from_numpy(runtime["raw"][start:stop]).to(
                device=device, dtype=torch.float32
            )
            local = torch.from_numpy(runtime["local"][start:stop]).to(
                device=device, dtype=torch.float32
            )
            context, global_state, validity = _encode_chunk(
                model, raw, local, pcs[start:stop],
                global_state, local_state_map,
            )
            loss, components = _sequence_nll(
                model, context, _slice_grammar(grammar, start, stop),
                np.arange(start, stop, dtype=np.int64),
                decoder_seed, epoch,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training sequence NLL")
            categorical_atoms = int(components["categorical_atoms"])
            (loss * float(categorical_atoms)).backward()
            pending += 1
            pending_categorical_atoms += categorical_atoms
            validity_sum += float(validity.detach().sum().item())
            validity_atoms += len(validity)
            for key, value in components.items():
                if key == "codec_teacher_prefix_atoms_by_position":
                    codec_teacher_prefix_atoms_by_position.update(value)
                else:
                    totals[key] += value
            if pending == accumulate_chunks or stop == len(rows):
                if pending_categorical_atoms <= 0:
                    raise RuntimeError("gradient window contains no atoms")
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(float(pending_categorical_atoms))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                pending_categorical_atoms = 0
                steps += 1
        row = {
            "epoch": epoch,
            "joint_sequence_nll_per_categorical_atom": (
                totals["total_nll_sum"]
                / max(1, totals["categorical_atoms"])
            ),
            "grammar_nll_per_categorical_decision": (
                totals["grammar_nll_sum"]
                / max(1, totals["grammar_atoms"])
            ),
            "codec_nll_sum": totals["codec_nll_sum"],
            "grammar_atoms": int(totals["grammar_atoms"]),
            "codec_exact_atoms": int(totals["codec_exact_atoms"]),
            "codec_termination_atoms": int(
                totals["codec_termination_atoms"]
            ),
            "codec_teacher_prefix_atoms_by_position": dict(sorted(
                codec_teacher_prefix_atoms_by_position.items()
            )),
            "categorical_atoms": int(totals["categorical_atoms"]),
            "sampled_emit_decisions": int(totals["sampled_emit_decisions"]),
            "sampled_stop_decisions": int(totals["sampled_stop_decisions"]),
            "mean_local_validity": validity_sum / max(1, validity_atoms),
            "optimizer_steps": steps,
            "observed_pc_states": len(local_state_map),
        }
        history.append(row)
        print(
            "[train:global-local-stop-emit-leb128] epoch={} nll={:.8f} "
            "grammar={:.8f} validity={:.6f}".format(
                epoch,
                row["joint_sequence_nll_per_categorical_atom"],
                row["grammar_nll_per_categorical_decision"],
                row["mean_local_validity"],
            ),
            flush=True,
        )
    return history


def _grammar_atom_count(counts):
    return int(np.asarray(counts, dtype=np.int64).sum() + len(counts))


def score_model(model, rows, runtime, device, chunk_len):
    output = np.empty((len(rows), model.hidden_size), dtype=np.float32)
    pcs = np.asarray([pc for pc, _, _ in rows], dtype=np.uint64)
    global_state = None
    local_state_map = {}
    validity_sum = 0.0
    validity_min = 1.0
    validity_max = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), chunk_len):
            stop = min(start + chunk_len, len(rows))
            raw = torch.from_numpy(runtime["raw"][start:stop]).to(
                device=device, dtype=torch.float32
            )
            local = torch.from_numpy(runtime["local"][start:stop]).to(
                device=device, dtype=torch.float32
            )
            context, global_state, validity = _encode_chunk(
                model, raw, local, pcs[start:stop],
                global_state, local_state_map,
            )
            output[start:stop] = context.cpu().numpy()
            validity_sum += float(validity.sum().item())
            validity_min = min(validity_min, float(validity.min().item()))
            validity_max = max(validity_max, float(validity.max().item()))
    return output, {
        "rows": len(rows),
        "unique_pc_states": len(local_state_map),
        "mean_local_validity": validity_sum / float(len(rows)),
        "min_local_validity": validity_min,
        "max_local_validity": validity_max,
    }


def decode(
    model, context_numpy, base_lines, device, decoder_seed, chunk_len=4096,
):
    """Decode until learned STOP, with only a fail-closed numeric watchdog."""
    if len(context_numpy) != len(base_lines):
        raise RuntimeError("decoder row counts differ")
    predicted_lines = [[] for _ in base_lines]
    predicted_fills = [[] for _ in base_lines]
    emit_probability_sum = 0.0
    grammar_decisions = 0
    sampled_emits = 0
    sampled_stops = 0
    codec_lengths = Counter()
    max_actions = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(base_lines), chunk_len):
            stop = min(start + chunk_len, len(base_lines))
            state = torch.from_numpy(context_numpy[start:stop]).to(device)
            anchors = torch.as_tensor(
                base_lines[start:stop], dtype=torch.long, device=device
            )
            active_numpy = np.arange(stop - start, dtype=np.int64)
            action_rank = 0
            while len(active_numpy):
                _assert_action_rank_within_watchdog(action_rank)
                active = torch.from_numpy(active_numpy).to(
                    device=device, dtype=torch.long
                )
                active_state = state.index_select(0, active)
                logits = model.grammar_head(active_state)
                probabilities = torch.softmax(logits, dim=1)
                emit_probability_sum += float(probabilities[:, EMIT].sum().item())
                grammar_decisions += len(active_numpy)
                _assert_stop_representable(logits)
                event_ids = start + active_numpy
                sampled = _inverse_cdf_sample(
                    logits, event_ids, decoder_seed, "eval", 0,
                    action_rank, FIELD_GRAMMAR, 0,
                )
                after_grammar = model.action_cell(
                    model.grammar_embedding(sampled), active_state
                )
                emit_local = np.flatnonzero(
                    sampled.cpu().numpy() == EMIT
                )
                sampled_emits += len(emit_local)
                sampled_stops += len(active_numpy) - len(emit_local)
                if not len(emit_local):
                    break
                emit_index = torch.from_numpy(emit_local).to(
                    device=device, dtype=torch.long
                )
                emit_events = event_ids[emit_local]
                emit_start_state = after_grammar.index_select(0, emit_index)
                codec_state, signed, lengths, codec_loss, codec_components = (
                    _sample_codec_path(
                        model, emit_start_state, emit_events, decoder_seed,
                        "eval", 0, action_rank,
                    )
                )
                if (
                    float(codec_loss.item()) != 0.0
                    or codec_components["codec_exact_atoms"] != 0
                    or codec_components["codec_termination_atoms"] != 0
                ):
                    raise RuntimeError("inference codec unexpectedly used labels")
                selected_anchors = anchors.index_select(0, active).index_select(
                    0, emit_index
                )
                targets = (selected_anchors + signed) & LINE_MASK
                global_positions = active_numpy[emit_local]
                if not (
                    len(global_positions) == len(targets) == len(lengths)
                ):
                    raise RuntimeError("decoded codec output lengths differ")
                for local_position, target, byte_count in zip(
                    global_positions, targets.cpu().numpy(), lengths
                ):
                    predicted_lines[start + int(local_position)].append(
                        int(target)
                    )
                    predicted_fills[start + int(local_position)].append(-1)
                    codec_lengths[int(byte_count)] += 1
                anchors = anchors.index_copy(0, active.index_select(0, emit_index), targets)
                after_grammar = after_grammar.index_copy(
                    0, emit_index, codec_state
                )
                state = state.index_copy(0, active, after_grammar)
                active_numpy = active_numpy[emit_local]
                action_rank += 1
                max_actions = max(max_actions, action_rank)

    counts = np.asarray([len(items) for items in predicted_lines], dtype=np.int64)
    diagnostics = {
        "callbacks": len(base_lines),
        "grammar_decisions": grammar_decisions,
        "sampled_emit_decisions": sampled_emits,
        "sampled_stop_decisions": sampled_stops,
        "mean_learned_emit_probability": (
            emit_probability_sum / max(1, grammar_decisions)
        ),
        "decoded_positive_callbacks": int(np.count_nonzero(counts)),
        "decoded_total_actions": int(counts.sum()),
        "decoded_mean_actions_per_callback": float(counts.mean()),
        "decoded_mean_actions_per_positive_callback": (
            float(counts[counts > 0].mean()) if np.any(counts > 0) else 0.0
        ),
        "decoded_max_actions_per_callback": int(counts.max()),
        "decoded_count_distribution": {
            str(key): int(value)
            for key, value in sorted(Counter(counts.tolist()).items())
        },
        "sampled_leb128_byte_length_distribution": {
            str(key): int(value) for key, value in sorted(codec_lengths.items())
        },
        "degree_cap_applied": False,
    }
    return counts, predicted_lines, predicted_fills, diagnostics


def trigger_metrics(predicted_counts, target_actions):
    if len(predicted_counts) != len(target_actions):
        raise RuntimeError("trigger metric row counts differ")
    predicted_positive = np.asarray(predicted_counts) > 0
    target_positive = np.asarray(
        [bool(items) for items in target_actions], dtype=np.bool_
    )
    true_positive = int(np.count_nonzero(
        predicted_positive & target_positive
    ))
    false_positive = int(np.count_nonzero(
        predicted_positive & ~target_positive
    ))
    false_negative = int(np.count_nonzero(
        ~predicted_positive & target_positive
    ))
    precision = (
        true_positive / float(true_positive + false_positive)
        if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / float(true_positive + false_negative)
        if true_positive + false_negative else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        "predicted_positive_callbacks": int(np.count_nonzero(predicted_positive)),
        "normal_positive_callbacks": int(np.count_nonzero(target_positive)),
        "true_positive_trigger_callbacks": true_positive,
        "false_positive_trigger_callbacks": false_positive,
        "false_negative_trigger_callbacks": false_negative,
        "trigger_precision": precision,
        "trigger_recall": recall,
        "trigger_f1": f1,
    }


def write_table(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_replay(path, rows, actions):
    if len(rows) != len(actions):
        raise RuntimeError("replay row/action counts differ")
    entries = 0
    triggers = 0
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pc", "line", "occ", "prefetch_addr"])
        for (pc, line, occurrence), targets in zip(rows, actions):
            if targets:
                triggers += 1
            for target in targets:
                writer.writerow([
                    pc, line, occurrence,
                    "0x{:x}".format(int(target) * CACHE_LINE_BYTES),
                ])
                entries += 1
    return entries, triggers


def _count_summary(actions):
    counts = [len(items) for items in actions]
    distribution = Counter(counts)
    return {
        "rows": len(counts),
        "actions": int(sum(counts)),
        "trigger_rows": int(sum(value > 0 for value in counts)),
        "mean_actions_per_row": (
            float(sum(counts)) / len(counts) if counts else 0.0
        ),
        "count_distribution": {
            str(key): int(value) for key, value in sorted(distribution.items())
        },
    }


def self_test_codec():
    values = [
        SIGNED_LINE_MIN, -16385, -129, -128, -2, -1,
        0, 1, 2, 63, 64, 127, 128, 16384, SIGNED_LINE_MAX,
    ]
    for value in values:
        encoded = _leb128_encode_signed_increment(value)
        if not 1 <= len(encoded) <= LEB128_MAX_BYTES:
            raise RuntimeError("LEB128 length self-test failed")
        if _leb128_decode_signed_increment(encoded) != value:
            raise RuntimeError("ZigZag/LEB128 round-trip failed")
    if len(_leb128_encode_signed_increment(1)) != 1:
        raise RuntimeError("small stride does not use compact one-byte codec")
    masks = [
        _legal_byte_mask(position) for position in range(LEB128_MAX_BYTES)
    ]
    if (
        not masks[0][STOP, 0]
        or masks[1][STOP, 0]
        or not masks[1][EMIT, 0]
        or not masks[1][STOP, 1]
        or int(masks[-1].sum()) != 3
        or not np.all(masks[-1][STOP, 1:4])
    ):
        raise RuntimeError("canonical LEB128 legality mask changed")
    try:
        _leb128_decode_signed_increment([0x80, 0x00])
    except RuntimeError:
        pass
    else:
        raise RuntimeError("noncanonical LEB128 form was accepted")


def self_test_sampler():
    events = np.arange(32, dtype=np.int64)
    first = _keyed_uniform(events, 7, "eval", 0, 1, FIELD_GRAMMAR, 0)
    second = _keyed_uniform(events, 7, "eval", 0, 1, FIELD_GRAMMAR, 0)
    changed = _keyed_uniform(events, 7, "eval", 0, 2, FIELD_GRAMMAR, 0)
    if not np.array_equal(first, second) or np.array_equal(first, changed):
        raise RuntimeError("stateless keyed CRN sampler self-test failed")
    if np.any(first <= 0.0) or np.any(first >= 1.0):
        raise RuntimeError("keyed uniform left open unit interval")
    _assert_stop_representable(torch.tensor([[0.0, 0.0]]))
    _assert_stop_representable(torch.tensor([
        [-(SAMPLER_GRID_BITS - 1) * float(np.log(2.0)), 0.0]
    ], dtype=torch.float64))
    try:
        _assert_stop_representable(torch.tensor([
            [-(SAMPLER_GRID_BITS + 1) * float(np.log(2.0)), 0.0]
        ], dtype=torch.float64))
    except RuntimeError:
        pass
    else:
        raise RuntimeError("unrepresentable STOP mass was accepted")


def self_test_nontermination_watchdog():
    simulated_always_emit_rank = 0
    try:
        while True:
            _assert_action_rank_within_watchdog(simulated_always_emit_rank)
            simulated_token = EMIT
            if simulated_token == STOP:
                break
            simulated_always_emit_rank += 1
    except RuntimeError:
        if simulated_always_emit_rank != NONTERMINATION_WATCHDOG_RANKS:
            raise RuntimeError("nontermination watchdog fired at wrong rank")
    else:
        raise RuntimeError("always-EMIT path bypassed nontermination watchdog")


def self_test_exact_integer_parser():
    large = (1 << 60) + 123
    if as_int(str(large)) != large or as_int("1.0e3") != 1000:
        raise RuntimeError("exact integer parser changed a legal integer")
    for value in ("1.5", "nan", "inf"):
        try:
            as_int(value)
        except ValueError:
            pass
        else:
            raise RuntimeError("exact integer parser accepted {}".format(value))


def self_test_parameter_count(hidden_size):
    if hidden_size not in MODEL_POINTS["lstm"]:
        raise RuntimeError("self-test size is not a configured model point")
    model = GlobalLocalGrammarStrideLSTM(hidden_size)
    observed = sum(parameter.numel() for parameter in model.parameters())
    expected = expected_parameter_count(hidden_size)
    if observed != expected:
        raise RuntimeError(
            "parameter formula mismatch: {} != {}".format(observed, expected)
        )
    described = {
        point["model_size"]: point["parameter_count"]
        for point in model_points_description()["points"]
    }
    if described.get(hidden_size) != observed or observed >= 10000:
        raise RuntimeError("configured v19 parameter point changed")


def self_test_causal_encoder():
    prefix = [
        (11, 100, 0), (22, 200, 0), (11, 104, 0), (11, 108, 0)
    ]
    changed_future = prefix + [(22, 999, 0), (11, 1, 0)]
    original_future = prefix + [(22, 201, 0), (11, 112, 0)]
    first = runtime_features(changed_future)
    second = runtime_features(original_future)
    for branch in ("raw", "local"):
        if not np.array_equal(first[branch][:len(prefix)], second[branch][:len(prefix)]):
            raise RuntimeError("future row changed a causal runtime feature")
    local = first["local"]
    age_bits = local[
        2,
        2 * LINE_NUMBER_BITS:2 * LINE_NUMBER_BITS + REUSE_AGE_BITS,
    ]
    age = int(sum(int(bit) << index for index, bit in enumerate(age_bits)))
    if age != 2 or int(local[2, -1]) != 1:
        raise RuntimeError("same-PC reuse-age feature self-test failed")


def self_test_global_local_encoder_behavior():
    torch.manual_seed(1729)
    model = GlobalLocalGrammarStrideLSTM(4)
    first_rows = [(11, 100, 0), (22, 200, 0), (11, 104, 0)]
    second_rows = [(11, 100, 0), (11, 104, 0), (22, 200, 0)]

    def encode(rows):
        runtime = runtime_features(rows)
        with torch.no_grad():
            return _encode_chunk(
                model,
                torch.from_numpy(runtime["raw"]).to(torch.float32),
                torch.from_numpy(runtime["local"]).to(torch.float32),
                np.asarray([row[0] for row in rows], dtype=np.uint64),
                None,
                {},
            )

    first_context, first_global, first_validity = encode(first_rows)
    second_context, second_global, second_validity = encode(second_rows)
    if (
        first_context.shape != (len(first_rows), model.hidden_size)
        or len(first_global) != 2
        or torch.equal(first_global[0], second_global[0])
        or torch.any(first_validity <= 0)
        or torch.any(first_validity >= 1)
        or torch.any(second_validity <= 0)
        or torch.any(second_validity >= 1)
    ):
        raise RuntimeError("global chronology/local validity behavior changed")

    with torch.no_grad():
        model.fusion.weight.zero_()
        model.fusion.bias.zero_()
        model.fusion.weight[:, model.hidden_size:] = torch.eye(
            model.hidden_size
        )
        model.local_validity.weight.zero_()
        model.local_validity.bias.fill_(-8.0)
    low_context, _, low_validity = encode(first_rows)
    with torch.no_grad():
        model.local_validity.bias.fill_(8.0)
    high_context, _, high_validity = encode(first_rows)
    if (
        not torch.all(high_validity > low_validity)
        or torch.allclose(low_context, high_context)
    ):
        raise RuntimeError("learned local-validity gate has no behavioral effect")


def self_test_rankwise_stop_absorption():
    torch.manual_seed(1730)
    model = GlobalLocalGrammarStrideLSTM(4)
    with torch.no_grad():
        model.grammar_head.weight.zero_()
        model.grammar_head.bias.copy_(torch.tensor([100.0, -100.0]))
    context = torch.zeros(2, model.hidden_size)
    grammar = {
        "counts": np.asarray([2, 1], dtype=np.int64),
        "targets": np.asarray([[2, 3], [4, 0]], dtype=np.uint64),
        "base_lines": np.asarray([1, 1], dtype=np.uint64),
    }
    loss, components = _sequence_nll(
        model, context, grammar, np.asarray([0, 1], dtype=np.int64), 7, 1
    )
    if (
        not torch.isfinite(loss)
        or components["sampled_stop_decisions"] != 2
        or components["sampled_emit_decisions"] != 0
        or components["grammar_atoms"] != 2
        or components["codec_exact_atoms"] != 0
    ):
        raise RuntimeError("sampled STOP was not absorbing across action ranks")


def self_test_teacher_loss_isolation():
    torch.manual_seed(2718)
    model = GlobalLocalGrammarStrideLSTM(4)
    state = torch.randn(3, 4)
    events = np.asarray([1, 2, 3], dtype=np.int64)
    first_targets = np.asarray([1, -128, SIGNED_LINE_MAX], dtype=np.int64)
    second_targets = np.asarray([-1, 64, SIGNED_LINE_MIN], dtype=np.int64)
    target_valid = np.ones(len(events), dtype=np.bool_)
    first_state, first_delta, first_lengths, first_loss, first_atoms = (
        _sample_codec_path(
            model, state, events, 7, "train", 1, 0,
            target_increments=first_targets, target_valid=target_valid,
        )
    )
    second_state, second_delta, second_lengths, second_loss, second_atoms = (
        _sample_codec_path(
            model, state, events, 7, "train", 1, 0,
            target_increments=second_targets, target_valid=target_valid,
        )
    )
    if (
        not torch.equal(first_delta, second_delta)
        or not torch.equal(first_state, second_state)
        or not np.array_equal(first_lengths, second_lengths)
        or not torch.isfinite(first_loss)
        or not torch.isfinite(second_loss)
        or first_atoms["codec_exact_atoms"] <= 0
        or second_atoms["codec_exact_atoms"] <= 0
    ):
        raise RuntimeError("teacher loss changed sampled main codec rollout")
    no_label_state, no_label_delta, no_label_lengths, no_label_loss, atoms = (
        _sample_codec_path(model, state, events, 7, "train", 1, 0)
    )
    if (
        not torch.equal(first_delta, no_label_delta)
        or not torch.equal(first_state, no_label_state)
        or not np.array_equal(first_lengths, no_label_lengths)
        or float(no_label_loss.item()) != 0.0
        or atoms["codec_exact_atoms"] != 0
        or atoms["codec_termination_atoms"] != 0
    ):
        raise RuntimeError("label-free codec rollout isolation failed")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--describe-model-points", action="store_true",
        help="print the canonical v19 model-point contract as JSON",
    )
    parser.add_argument("--policy", required=True, choices=[POLICY])
    for role in ("train", "guard", "eval"):
        parser.add_argument(
            "--{}-stream".format(role), required=True, type=Path
        )
        parser.add_argument(
            "--{}-candidates".format(role), required=True, type=Path
        )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-family", choices=["lstm"], required=True)
    parser.add_argument("--model-size", type=int, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--decoder-seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--chunk-len", type=int, default=1024)
    parser.add_argument("--accumulate-chunks", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    return parser


def main():
    if sys.argv[1:] == ["--describe-model-points"]:
        print(json.dumps(model_points_description(), indent=2, sort_keys=True))
        return
    args = build_parser().parse_args()
    expected_pair = MODEL_POINTS["lstm"].get(args.model_size)
    if expected_pair is None or args.pair_id != expected_pair:
        raise RuntimeError("model size/pair is not a configured v19 LSTM point")
    if (
        args.model_size < 1 or args.epochs < 1 or args.chunk_len < 1
        or args.accumulate_chunks < 1 or args.learning_rate <= 0
    ):
        raise RuntimeError("model/training dimensions must be positive")

    self_test_codec()
    self_test_sampler()
    self_test_nontermination_watchdog()
    self_test_exact_integer_parser()
    self_test_parameter_count(args.model_size)
    self_test_causal_encoder()
    self_test_global_local_encoder_behavior()
    self_test_rankwise_stop_absorption()
    self_test_teacher_loss_isolation()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    roles = ("train", "guard", "eval")
    stream_paths = {
        role: getattr(args, role + "_stream") for role in roles
    }
    action_paths = {
        role: getattr(args, role + "_candidates") for role in roles
    }
    rows = {role: load_stream(stream_paths[role]) for role in roles}
    teacher = {
        role: load_teacher_actions(action_paths[role], rows[role])
        for role in roles
    }

    train_runtime = runtime_features(rows["train"])
    if not all(
        np.array_equal(train_runtime[key], runtime_features(rows["train"])[key])
        for key in ("raw", "local")
    ):
        raise RuntimeError("training/inference runtime encoder differs")
    train_grammar = teacher_grammar(
        [line for _, line, _ in rows["train"]], teacher["train"]
    )

    model = GlobalLocalGrammarStrideLSTM(args.model_size)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    expected_parameters = expected_parameter_count(args.model_size)
    if parameter_count != expected_parameters:
        raise RuntimeError("v19 parameter count changed")
    history = train_model(
        model, rows["train"], train_runtime, train_grammar, device,
        args.epochs, args.chunk_len, args.accumulate_chunks,
        args.learning_rate, args.decoder_seed,
    )

    history_rows = rows["train"] + rows["guard"] + rows["eval"]
    history_runtime = runtime_features(history_rows)
    for branch in ("raw", "local"):
        if not np.array_equal(
            history_runtime[branch][:len(rows["train"])],
            train_runtime[branch],
        ):
            raise RuntimeError("complete-history encoder changed train prefix")
    encoded_history, encoder_diagnostics = score_model(
        model, history_rows, history_runtime, device, args.chunk_len
    )
    eval_start = len(rows["train"]) + len(rows["guard"])
    eval_context = encoded_history[eval_start:]
    (
        predicted_counts, predicted_lines, predicted_fills,
        decoder_diagnostics,
    ) = decode(
        model, eval_context,
        [line for _, line, _ in rows["eval"]],
        device, args.decoder_seed,
    )
    behavior = behavior_metrics(
        predicted_counts, predicted_lines, predicted_fills, teacher["eval"]
    )
    behavior.update(trigger_metrics(predicted_counts, teacher["eval"]))

    normal_path = args.out_dir / "offline_stride.replay.csv"
    nn_path = args.out_dir / "offline_nn.replay.csv"
    normal_entries, normal_triggers = write_replay(
        normal_path, rows["eval"], teacher["eval"]
    )
    nn_entries, nn_triggers = write_replay(
        nn_path, rows["eval"], predicted_lines
    )
    write_table(args.out_dir / "training_history.csv", history)

    tag = model_tag(args.model_size)
    torch.save({
        "state_dict": model.state_dict(),
        "model_family": "lstm",
        "model_size": args.model_size,
        "raw_runtime_features": RAW_FEATURES,
        "local_runtime_features": LOCAL_FEATURES,
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": MODEL_REVISION,
    }, args.out_dir / "model.pt")

    train_counts = train_grammar["counts"]
    train_unique_pc_count = len({pc for pc, _, _ in rows["train"]})
    history_unique_pc_count = len({pc for pc, _, _ in history_rows})
    sampler_key_fields = [
        "sampler_revision", "decoder_seed", "trace", "policy", "role",
        "epoch", "event_index", "action_rank", "field", "codec_position",
    ]
    encoder_hash = runtime_encoder_sha256()
    router_hash = state_router_sha256()
    metadata = {
        "trace": TRACE,
        "model_tag": tag,
        "matched_normal_prefetcher": POLICY,
        "neural_role": "standalone_direct_action_prefetcher",
        "model_family": "lstm",
        "track_model_family": "lstm",
        "model_size": args.model_size,
        "architecture_pair_id": args.pair_id,
        "parameter_count": parameter_count,
        "expected_parameter_count": expected_parameters,
        "parameter_formula": model_points_description()["parameter_formula"],
        "model_point_contract": model_points_description(),
        "parameter_bytes_float32": parameter_count * 4,
        "input_projection_size": model.input_size,
        "seed": args.seed,
        "decoder_seed": args.decoder_seed,
        "epochs": args.epochs,
        "chunk_len": args.chunk_len,
        "accumulate_chunks": args.accumulate_chunks,
        "learning_rate": args.learning_rate,
        "runtime_feature_count": RUNTIME_FEATURES,
        "raw_runtime_feature_count": RAW_FEATURES,
        "pc_local_runtime_feature_count": LOCAL_FEATURES,
        "runtime_encoding": (
            "lossless PC64+line58 global branch plus causal lossless "
            "line58+same-PC signed-delta58+reuse-age64+valid1 local branch"
        ),
        "runtime_pc_bits": ADDRESS_BITS,
        "runtime_line_number_bits": LINE_NUMBER_BITS,
        "runtime_reuse_age_bits": REUSE_AGE_BITS,
        "runtime_constant_offset_bits_removed": CACHE_LINE_OFFSET_BITS,
        "source_decision_effective_external_input": SOURCE_INPUTS,
        "same_external_input_contract": True,
        "causal_derived_features_from_same_external_input": [
            "same_pc_signed_line_delta", "same_pc_reuse_age",
            "same_pc_history_valid",
        ],
        "derived_features_use_teacher_or_future": False,
        "training_inference_input_encoder_identical": True,
        "runtime_encoder_entrypoint": (
            "623_offline_lstm_stride.train_and_offline_infer.runtime_features"
        ),
        "runtime_encoder_sha256": encoder_hash,
        "training_runtime_encoder_sha256": encoder_hash,
        "inference_runtime_encoder_sha256": encoder_hash,
        "training_runtime_fields": SOURCE_INPUTS,
        "inference_runtime_fields": SOURCE_INPUTS,
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "normal_policy_request_rate_used_as_budget": False,
        "normal_policy_constants_used_by_neural_inference": False,
        "normal_tracker_capacity_used_by_neural_inference": False,
        "normal_degree_used_by_neural_inference": False,
        "probability_threshold_used": False,
        "threshold_related_hardcodes_used": False,
        "neural_degree_cap": None,
        "fixed_page_offset_classes": None,
        "same_page_rule_used_by_neural_inference": False,
        "future_label_window_used": False,
        "handcrafted_semantic_features_used": True,
        "handcrafted_features_scope": (
            "causal relative delta/reuse age derived losslessly from PC+addr"
        ),
        "manual_loss_weights_used": False,
        "gradient_accumulation_weighting": (
            "exact_categorical_atom_count_for_global_natural_sequence_NLL"
        ),
        "training_regularization_used": False,
        "inference_policy_hardcodes_used": False,
        "learned_request_count": True,
        "nn_generates_own_target_addresses": True,
        "complete_action_space": (
            "learned unbounded STOP/EMIT grammar with exact signed 58-bit "
            "increments; sampler-precision watchdog aborts the whole run "
            "instead of materializing a nonterminating sequence"
        ),
        "decision_rule": (
            "rankwise_keyed_inverse_cdf_STOP_EMIT_then_exact_"
            "ZigZag_LEB128_increment"
        ),
        "decoder_training_mode": DECODER_TRAINING_MODE,
        "decoder_previous_teacher_action_used_as_input": True,
        "decoder_previous_teacher_action_input_scope": (
            "isolated_loss_only_teacher_prefix_likelihood_branch"
        ),
        "decoder_previous_teacher_action_used_as_main_rollout_input": False,
        "teacher_prefix_tokens_condition_loss_logits": True,
        "teacher_prefix_tokens_recurrently_advance_loss_branch_state": True,
        "teacher_prefix_tokens_mutate_main_rollout_state": False,
        "teacher_prefix_branch_role": (
            "loss_only_canonical_autoregressive_sequence_likelihood"
        ),
        "sampled_prefix_branch_role": (
            "only_branch_that_mutates_next_rank_state_and_origin"
        ),
        "decoder_free_running_self_test": "PASS",
        "request_count_model": "rankwise learned STOP_EMIT action grammar",
        "request_count_training_objective": REQUEST_COUNT_OBJECTIVE,
        "request_count_decoding_rule": (
            "stateless_event_rank_keyed_categorical_inverse_cdf_until_STOP"
        ),
        "request_count_residual_scope": "none_rankwise_action_grammar",
        "gate_training_objective": "NOT_APPLICABLE_no_separate_hurdle_gate",
        "gate_decoding_rule": "NOT_APPLICABLE_STOP_EMIT_is_action_token",
        "gate_class_weighting_used": False,
        "data_derived_gate_class_weights_used": False,
        "gate_class_weights_source": None,
        "gate_class_weights": None,
        "poisson_objective_used": False,
        "poisson_decoder_used": False,
        "delta_mixture_components": 0,
        "gmm_objective_used": False,
        "gmm_decoder_used": False,
        "delta_training_objective": DELTA_OBJECTIVE,
        "delta_decoding_rule": (
            "stateless_keyed_inverse_cdf_exact_ZigZag_LEB128_signed_increment"
        ),
        "delta_decoder_feedback_rule": (
            "main_rollout_uses_only_actual_hard_sampled_STOP_EMIT_"
            "payload_bits_continuation_tokens"
        ),
        "delta_codec": "signed_ZigZag_then_canonical_LEB128",
        "delta_codec_max_bytes": LEB128_MAX_BYTES,
        "delta_codec_complete_signed_bits": LINE_NUMBER_BITS,
        "delta_codec_payload_factorization": (
            "seven shared binary payload decisions; final group 3 classes"
        ),
        "delta_codec_canonical_legality_mask": True,
        "on_policy_rollout": True,
        "teacher_prefix_likelihood_branch_isolated_from_rollout": True,
        "sampled_stop_rows_reactivated": False,
        "teacher_selected_emit_used_as_feedback": False,
        "teacher_codec_token_used_as_feedback": True,
        "teacher_codec_token_feedback_scope": (
            "isolated_loss_only_teacher_prefix_likelihood_branch"
        ),
        "teacher_codec_token_used_as_main_rollout_feedback": False,
        "increment_supervision_origin": (
            "actual_previous_sampled_target_or_current_demand"
        ),
        "host_nontermination_guard": (
            "fail_closed_if_STOP_has_no_53bit_sampler_support_or_if_"
            "sampled_path_reaches_sampler_precision_watchdog"
        ),
        "fail_closed_nontermination_watchdog_ranks": (
            NONTERMINATION_WATCHDOG_RANKS
        ),
        "nontermination_watchdog_is_policy_degree_cap": False,
        "successful_run_hit_nontermination_watchdog": False,
        "nontermination_watchdog_behavior": (
            "raise_and_produce_no_replay_never_truncate_or_force_STOP"
        ),
        "stop_sampler_representability_check": True,
        "sampler_uniform_grid_bits": SAMPLER_GRID_BITS,
        "sampler_minimum_open_midpoint_uniform": SAMPLER_MIN_UNIFORM,
        "delta_codec_source_sha256": codec_source_sha256(),
        "fill_decoding_rule": "fixed_track_contract_FILL_L2",
        "request_count_training_label_statistics": {
            "decision_callbacks": int(len(train_counts)),
            "positive_callbacks": int(np.count_nonzero(train_counts)),
            "zero_callbacks": int(np.count_nonzero(train_counts == 0)),
            "teacher_actions": int(train_counts.sum()),
            "grammar_categorical_atoms": _grammar_atom_count(train_counts),
        },
        "request_count_decoder_diagnostics": decoder_diagnostics,
        "decoder_sampler": {
            "sampler_revision": SAMPLER_REVISION,
            "key_fields": sampler_key_fields,
            "backend": (
                "vectorized_splitmix64_53bit_open_midpoints_float64_cdf"
            ),
            "categorical_method": "inverse_cdf",
            "cross_event_rng_state": False,
        },
        "sampler_revision": SAMPLER_REVISION,
        "decoder_key_fields": sampler_key_fields,
        "decoder_event_key_definition": "zero_based_role_event_index",
        "decoder_codec_position_definition": (
            "generic token coordinate: byte index for continuation/final "
            "payload, flattened byte*7+bit for payload-bit decisions"
        ),
        "decoder_event_key_uses_teacher_information": False,
        "decoder_action_rank_origin": 0,
        "decoder_key_includes_sampler_revision": True,
        "decoder_sampler_source_sha256": sampler_source_sha256(),
        "decoder_sampling_schedule_sha256": hashlib.sha256(
            json.dumps({
                "grammar_field": FIELD_GRAMMAR,
                "payload_bit_field": FIELD_PAYLOAD_BIT,
                "final_payload_field": FIELD_FINAL_PAYLOAD,
                "continuation_field": FIELD_CONTINUATION,
                "max_codec_bytes": LEB128_MAX_BYTES,
                "payload_bits_per_regular_byte": LEB128_PAYLOAD_BITS,
                "roles": sorted(ROLE_CODES),
            }, sort_keys=True).encode()
        ).hexdigest(),
        "deterministic_decoding": False,
        "deterministic_decoding_reproducible": False,
        "stochastic_decoding": True,
        "stochastic_decoding_reproducible": True,
        "common_random_numbers_across_capacities": True,
        "strict_common_random_numbers_across_capacities": True,
        "cross_event_rng_state_used": False,
        "decoder_sampling_roles": ["train", "eval"],
        "decoder_train_sampling_performed": True,
        "decoder_guard_sampling_performed": False,
        "sampled_outputs_used_as_decoder_feedback": True,
        "decoder_probability_mass_carries_train_guard_history": False,
        "cross_event_probability_credit_used": False,
        "training_labels": (
            "captured Stride actions; grammar labels, isolated teacher-prefix "
            "codec likelihood, and comparator replay only"
        ),
        "forbidden_inputs": [
            "normal_actions_at_inference", "Stride_tracker_table",
            "last_stride", "normal_degree", "cycle", "cache_hit",
            "queue_state", "future_rows",
        ],
        "training_chunks_shuffled": False,
        "training_state_mode": "chronological_global_and_pc_local_tbptt",
        "training_state_carried_across_chunks": True,
        "training_state_detached_between_chunks": True,
        "training_state_routing": (
            "one chronological global state plus dynamic exact-PC local map"
        ),
        "training_state_reset": "only_at_epoch_start",
        "inference_history_mode": (
            "fresh_state_then_complete_train_guard_eval_chronology"
        ),
        "inference_state_routing": (
            "one chronological global state plus dynamic exact-PC local map"
        ),
        "learned_local_validity_gate": True,
        "local_validity_gate_rule": (
            "sigmoid learned from chronological global and PC-local contexts"
        ),
        "encoder_diagnostics": encoder_diagnostics,
        "guard_role": "causal_input_history_warmup_and_audit_only",
        "train_unique_pc_count": train_unique_pc_count,
        "history_unique_pc_count": history_unique_pc_count,
        "global_recurrent_state_bytes_float32": 2 * args.model_size * 4,
        "local_recurrent_state_bytes_per_observed_pc_float32": (
            2 * args.model_size * 4
        ),
        "training_state_router_sha256": router_hash,
        "inference_state_router_sha256": router_hash,
        "peak_training_recurrent_state_bytes_float32": (
            (train_unique_pc_count + 1) * 2 * args.model_size * 4
        ),
        "peak_inference_recurrent_state_bytes_float32": (
            (history_unique_pc_count + 1) * 2 * args.model_size * 4
        ),
        "peak_persistent_recurrent_state_bytes": (
            (history_unique_pc_count + 1) * 2 * args.model_size * 4
        ),
        "causal_no_future_self_test": "PASS",
        "pc_keyed_causality_self_test": "PASS",
        "global_chronology_self_test": "PASS",
        "learned_validity_self_test": "PASS",
        "event_keyed_crn_self_test": "PASS",
        "rankwise_stop_emit_self_test": "PASS",
        "main_rollout_isolation_self_test": "PASS",
        "teacher_prefix_loss_isolation_self_test": "PASS",
        "stop_sampler_representability_self_test": "PASS",
        "always_emit_nontermination_watchdog_self_test": "PASS",
        "exact_integer_parser_self_test": "PASS",
        "zigzag_leb128_exact_codec_self_test": "PASS",
        "hard_self_action_feedback_self_test": "PASS",
        "compact_parameter_self_test": "PASS",
        "cnn_architecture_self_test": "NOT_APPLICABLE",
        "cnn_temporal_layers": 0,
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": MODEL_REVISION,
        "event_logger_schema": EVENT_LOGGER_SCHEMA,
        "candidate_attachment_mode": CANDIDATE_ATTACHMENT_MODE,
        "offline_normal_entries": normal_entries,
        "offline_normal_triggers": normal_triggers,
        "offline_nn_entries": nn_entries,
        "offline_nn_triggers": nn_triggers,
        "normal_list_sha256": sha256(normal_path),
        "nn_list_sha256": sha256(nn_path),
        "heldout_behavior_metrics": behavior,
        "train_action_summary": _count_summary(teacher["train"]),
        "guard_action_summary": _count_summary(teacher["guard"]),
        "eval_action_summary": _count_summary(teacher["eval"]),
        "train_history": history,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    for role in roles:
        metadata[role + "_stream_gzip_sha256"] = sha256(stream_paths[role])
        metadata[role + "_stream_content_sha256"] = gzip_content_sha256(
            stream_paths[role]
        )
        metadata[role + "_candidate_gzip_sha256"] = sha256(action_paths[role])
        metadata[role + "_candidate_content_sha256"] = gzip_content_sha256(
            action_paths[role]
        )
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "status": "PASS",
        "model_tag": tag,
        "parameters": parameter_count,
        "decision_rule": metadata["decision_rule"],
        "offline_normal_entries": normal_entries,
        "offline_nn_entries": nn_entries,
    }, indent=2))


if __name__ == "__main__":
    main()
