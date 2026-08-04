#!/usr/bin/env python3
"""Routed/page-local LSTM with an exact learned action grammar for 623 SPP.

Only source-visible chronological callbacks are runtime inputs: DEMAND(addr)
and CACHE_FILL(evicted_addr).  Source-SPP actions are supervision and the
offline-normal comparator only.  The v19 decoder learns a rank-wise
STOP/EMIT grammar, an exact signed ZigZag+canonical-LEB128 increment, and fill
after the model's actual emitted target.  All categorical choices use the
stateless keyed inverse CDF; no probability threshold, Poisson count, GMM,
candidate table, degree cap, or nearest-delta fallback exists.
"""
import argparse
import csv
import gzip
import hashlib
import inspect
import json
import math
import platform
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model_points_v19 import (
    ACTION_ROLLOUT_WATCHDOG_RANKS, ADDRESS_BITS, BYTE_PAYLOAD_BITS,
    BYTE_VOCAB, CACHE_LINE_BYTES, CACHE_LINE_SHIFT, DECODER_REVISION,
    EXPERIMENT_REVISION, EXTERNAL_INPUT_FIELDS, FILL_LEVELS,
    KEYED_UNIFORM_HALF_BIN, LEB128_MAX_BYTES, LINE_ADDRESS_BITS,
    MODEL_POINTS, MODEL_REVISION, OPERATION, PAGE_BYTES,
    PAGE_OFFSET_BITS, PARAMETER_FORMULA, POLICY, RUNTIME_FEATURE_COUNT,
    RUN_ID, SAMPLER_GRID_BITS, TRACE, codec_embed_size,
    describe_model_points, exact_int as as_int, expected_parameter_count,
    model_tag, routed_state_size, self_test_exact_int,
)

from formal_NN_training.common.keyed_sampling import (
    KEY_FIELDS, SAMPLER_REVISION, key_schedule_sha256, key_stream_sha256,
    keyed_uniform, sampler_metadata, sampler_source_sha256,
    sampling_schedule_sha256, self_test_keyed_crn,
)
from formal_NN_training.common.threshold_free_policy import (
    ADDRESS_BITS as COMMON_ADDRESS_BITS,
    CACHE_LINE_BYTES as COMMON_CACHE_LINE_BYTES,
    CACHE_LINE_SHIFT as COMMON_CACHE_LINE_SHIFT,
    behavior_metrics,
)

if (
    ADDRESS_BITS != COMMON_ADDRESS_BITS
    or CACHE_LINE_BYTES != COMMON_CACHE_LINE_BYTES
    or CACHE_LINE_SHIFT != COMMON_CACHE_LINE_SHIFT
):
    raise RuntimeError("SPP v19 pure model contract differs from common address ABI")
LINE_ADDRESS_MODULUS = 1 << LINE_ADDRESS_BITS
LINE_ADDRESS_HALF_RANGE = 1 << (LINE_ADDRESS_BITS - 1)
PAGE_LINES = PAGE_BYTES // CACHE_LINE_BYTES
if (
    PAGE_BYTES % CACHE_LINE_BYTES or PAGE_LINES < 1
    or PAGE_LINES & (PAGE_LINES - 1)
    or PAGE_OFFSET_BITS != PAGE_LINES.bit_length() - 1
):
    raise RuntimeError("page/line architecture decomposition must be a power of two")
RUNTIME_FEATURES = LINE_ADDRESS_BITS + 1
if RUNTIME_FEATURES != RUNTIME_FEATURE_COUNT:
    raise RuntimeError("SPP v19 runtime feature contract changed")
EVENT_LOGGER_SCHEMA = "623_causal_trigger_fill_v6"
ACTION_ATTACHMENT_MODE = "explicit_trigger_event_id"
CANONICALIZATION_MODE = "per_target_min_fill_queue_effect"
SOURCE_INPUTS = list(EXTERNAL_INPUT_FIELDS)
if SAMPLER_GRID_BITS != int(np.finfo(np.float64).nmant):
    raise RuntimeError("SPP v19 sampler-grid contract differs from float64")


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


def require_equal_lengths(label, *sequences):
    lengths = tuple(len(sequence) for sequence in sequences)
    if len(set(lengths)) != 1:
        raise RuntimeError("{} parallel lengths differ: {}".format(label, lengths))


def load_stream(path):
    context, demands, demand_positions = [], [], []
    occurrences = defaultdict(int)
    last_raw_event_id, last_cycle = -1, -1
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "trace", "event_idx", "raw_event_id", "cycle", "event_kind",
            "event_address", "event_line", "decision_idx", "pc",
            "cache_hit", "access_type", "pc_line_occ", "logger_schema",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError("{} missing {}".format(path, sorted(missing)))
        for index, row in enumerate(reader):
            raw_event_id, cycle = as_int(row["raw_event_id"]), as_int(row["cycle"])
            kind, decision_idx = row["event_kind"], as_int(row["decision_idx"])
            pc, address = as_int(row["pc"]), as_int(row["event_address"])
            line, hit = as_int(row["event_line"]), as_int(row["cache_hit"])
            occurrence = as_int(row["pc_line_occ"])
            if (
                row["trace"] != TRACE or as_int(row["event_idx"]) != index
                or row["logger_schema"] != EVENT_LOGGER_SCHEMA
                or address != line << CACHE_LINE_SHIFT
                or line < 0 or line >= LINE_ADDRESS_MODULUS
                or pc < 0 or pc >= 1 << ADDRESS_BITS
                or raw_event_id <= last_raw_event_id or cycle < last_cycle
                or kind not in ("DEMAND", "FILL")
            ):
                raise RuntimeError("stream identity/input failure at row {}".format(index))
            if kind == "DEMAND":
                expected = occurrences[(pc, line)]
                occurrences[(pc, line)] += 1
                if decision_idx != len(demands) or occurrence != expected or hit not in (0, 1):
                    raise RuntimeError("demand identity failure at row {}".format(index))
                demands.append((pc, address, line, occurrence))
                demand_positions.append(index)
            elif decision_idx != -1 or pc != 0 or hit != 0 or occurrence != -1:
                raise RuntimeError("cache-fill context leaks transport fields at {}".format(index))
            context.append((kind, address, line, decision_idx))
            last_raw_event_id, last_cycle = raw_event_id, cycle
    if not context or not demands or len(context) == len(demands):
        raise RuntimeError("empty/no-fill SPP stream {}".format(path))
    return {
        "context": context, "demands": demands,
        "demand_positions": np.asarray(demand_positions, dtype=np.int64),
    }


def load_teacher_actions(path, rows):
    actions = [[] for _ in rows]
    last_pf_event = -1
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "trace", "policy", "demand_idx", "pc", "line", "pc_line_occ",
            "action_rank", "pf_line", "fill_level", "accepted", "duplicate",
            "trigger_event_id", "pf_event_id", "event_distance", "raw_action_count",
            "source_first_pf_event_id", "source_last_pf_event_id", "is_self_target",
            "canonicalization", "match_mode", "logger_schema",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError("{} missing {}".format(path, sorted(missing)))
        for row in reader:
            index = as_int(row["demand_idx"])
            if index < 0 or index >= len(rows):
                raise RuntimeError("teacher action demand_idx out of range")
            pc, _, line, occurrence = rows[index]
            if (
                row["trace"] != TRACE or row["policy"] != POLICY
                or (as_int(row["pc"]), as_int(row["line"]), as_int(row["pc_line_occ"]))
                != (pc, line, occurrence)
                or row["logger_schema"] != EVENT_LOGGER_SCHEMA
                or row["match_mode"] != ACTION_ATTACHMENT_MODE
                or as_int(row["action_rank"]) != len(actions[index]) + 1
            ):
                raise RuntimeError("teacher action identity/rank failure at {}".format(index))
            pf_event, trigger = as_int(row["pf_event_id"]), as_int(row["trigger_event_id"])
            if (
                pf_event <= last_pf_event or trigger >= pf_event
                or as_int(row["event_distance"]) != pf_event - trigger
            ):
                raise RuntimeError("invalid action attachment at {}".format(index))
            pf_line, fill = as_int(row["pf_line"]), as_int(row["fill_level"])
            if (
                pf_line < 0 or pf_line >= LINE_ADDRESS_MODULUS
                or fill not in FILL_LEVELS or as_int(row["accepted"]) != 1
                or as_int(row["duplicate"]) not in (0, 1)
                or as_int(row["raw_action_count"]) < 1
                or as_int(row["source_first_pf_event_id"]) != pf_event
                or as_int(row["source_last_pf_event_id"]) < pf_event
                or as_int(row["is_self_target"]) != int(pf_line == line)
                or row["canonicalization"] != CANONICALIZATION_MODE
                or any(existing == pf_line for existing, _ in actions[index])
            ):
                raise RuntimeError("invalid captured SPP action at {}".format(index))
            actions[index].append((pf_line, fill))
            last_pf_event = pf_event
    if not any(actions):
        raise RuntimeError("empty teacher action stream {}".format(path))
    return actions


def _unsigned_bits(values, width):
    integers = [int(value) for value in values]
    if width < 1 or any(value < 0 or value >= 1 << width for value in integers):
        raise RuntimeError("runtime line number is outside the encoded domain")
    array = np.asarray(integers, dtype=np.uint64)
    shifts = np.arange(width, dtype=np.uint64)
    return ((array[:, None] >> shifts[None, :]) & 1).astype(np.float32)


def runtime_bundle(stream):
    context = stream["context"]
    lines = np.asarray([line for _, _, line, _ in context], dtype=np.int64)
    kinds = np.asarray([kind == "DEMAND" for kind, _, _, _ in context], dtype=np.bool_)
    features = np.concatenate([
        _unsigned_bits(lines, LINE_ADDRESS_BITS), kinds.astype(np.float32)[:, None],
    ], axis=1)
    return {
        "features": features,
        "lines": lines,
        "pages": np.right_shift(lines, PAGE_OFFSET_BITS),
        "demand_kind": kinds,
    }


def runtime_encoder_sha256():
    payload = {
        "entrypoint_source": inspect.getsource(runtime_bundle),
        "primitive_source": inspect.getsource(_unsigned_bits),
        "fields": SOURCE_INPUTS, "use_pc": False, "address_bits": ADDRESS_BITS,
        "line_address_bits": LINE_ADDRESS_BITS, "cache_line_bytes": CACHE_LINE_BYTES,
        "bit_order": "least_significant_first",
        "page_key_derivation": "line>>{}".format(PAGE_OFFSET_BITS),
        "callback_kind_encoding": {"DEMAND": 1.0, "FILL": 0.0},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def sampling_event_keys(stream):
    require_equal_lengths(
        "decision router", stream["demand_positions"], stream["demands"],
    )
    keys = []
    for decision_idx, (position, demand) in enumerate(zip(stream["demand_positions"], stream["demands"])):
        kind, _, context_line, routed = stream["context"][int(position)]
        demand_line = demand[2]
        if kind != "DEMAND" or routed != decision_idx or context_line != demand_line:
            raise RuntimeError("SPP decision router changed")
        keys.append("decision_idx={}|kind=DEMAND|line={}".format(decision_idx, demand_line))
    return keys


def decision_router_sha256(stream):
    payload = {
        "context_rows": len(stream["context"]),
        "demand_positions": [int(value) for value in stream["demand_positions"]],
        "decision_indices": [int(stream["context"][int(pos)][3]) for pos in stream["demand_positions"]],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def decision_router_source_sha256():
    return hashlib.sha256(inspect.getsource(decision_router_sha256).encode()).hexdigest()


def build_context_targets(stream, actions):
    require_equal_lengths(
        "teacher decision targets", stream["demand_positions"],
        stream["demands"], actions,
    )
    counts = np.full(len(stream["context"]), -1, dtype=np.int64)
    width = max(len(items) for items in actions)
    lines = np.full((len(counts), width), -1, dtype=np.int64)
    fills = np.full((len(counts), width), -1, dtype=np.int64)
    fill_to_index = {value: index for index, value in enumerate(FILL_LEVELS)}
    for decision, position in enumerate(stream["demand_positions"]):
        items = actions[decision]
        counts[position] = len(items)
        for rank, (target, fill) in enumerate(items):
            lines[position, rank] = int(target)
            fills[position, rank] = fill_to_index[fill]
    return counts, lines, fills


def write_table(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_teacher_replay(path, rows, actions):
    require_equal_lengths("teacher replay", rows, actions)
    entries = triggers = 0
    fill_counts = {"FILL_L2": 0, "FILL_LLC": 0}
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pc", "line", "occ", "prefetch_addr", "fill_level"])
        for (pc, _, line, occurrence), items in zip(rows, actions):
            triggers += int(bool(items))
            for pf_line, fill in items:
                writer.writerow([pc, line, occurrence, hex(pf_line << CACHE_LINE_SHIFT), fill])
                fill_counts["FILL_L2" if fill == 2 else "FILL_LLC"] += 1
                entries += 1
    return entries, triggers, fill_counts


def write_prediction_replay(path, rows, predicted_lines, predicted_fills):
    require_equal_lengths("prediction replay", rows, predicted_lines, predicted_fills)
    entries = triggers = 0
    fill_counts = {"FILL_L2": 0, "FILL_LLC": 0}
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pc", "line", "occ", "prefetch_addr", "fill_level"])
        for (pc, _, line, occurrence), targets, fills in zip(rows, predicted_lines, predicted_fills):
            require_equal_lengths("prediction callback", targets, fills)
            triggers += int(bool(targets))
            for pf_line, fill_index in zip(targets, fills):
                fill = FILL_LEVELS[int(fill_index)]
                writer.writerow([pc, line, occurrence, hex(int(pf_line) << CACHE_LINE_SHIFT), fill])
                fill_counts["FILL_L2" if fill == 2 else "FILL_LLC"] += 1
                entries += 1
    return entries, triggers, fill_counts


def canonical_signed_delta(base, target):
    difference = (int(target) - int(base)) % LINE_ADDRESS_MODULUS
    return difference - LINE_ADDRESS_MODULUS if difference >= LINE_ADDRESS_HALF_RANGE else difference


def zigzag_encode(delta):
    delta = int(delta)
    if delta < -LINE_ADDRESS_HALF_RANGE or delta >= LINE_ADDRESS_HALF_RANGE:
        raise RuntimeError("signed delta is outside the canonical 58-bit domain")
    return 2 * delta if delta >= 0 else -2 * delta - 1


def zigzag_decode(unsigned):
    unsigned = int(unsigned)
    if unsigned < 0 or unsigned >= LINE_ADDRESS_MODULUS:
        raise RuntimeError("ZigZag word is outside the 58-bit domain")
    return unsigned >> 1 if not (unsigned & 1) else -((unsigned >> 1) + 1)


def leb128_encode_delta(delta):
    value = zigzag_encode(delta)
    result = []
    while True:
        byte = value & ((1 << BYTE_PAYLOAD_BITS) - 1)
        value >>= BYTE_PAYLOAD_BITS
        if value:
            result.append(byte | (1 << BYTE_PAYLOAD_BITS))
        else:
            result.append(byte)
            break
    if len(result) > LEB128_MAX_BYTES:
        raise RuntimeError("canonical LEB128 delta exceeds nine bytes")
    return result


def leb128_decode_delta(tokens):
    value = 0
    for position, raw in enumerate(tokens):
        token = int(raw)
        if token < 0 or token >= BYTE_VOCAB:
            raise RuntimeError("LEB128 token is outside byte domain")
        value |= (token & ((1 << BYTE_PAYLOAD_BITS) - 1)) << (BYTE_PAYLOAD_BITS * position)
        if token < (1 << BYTE_PAYLOAD_BITS):
            if position and token == 0:
                raise RuntimeError("noncanonical overlong LEB128 delta")
            if position == LEB128_MAX_BYTES - 1 and token > 3:
                raise RuntimeError("LEB128 delta exceeds the 58-bit domain")
            if leb128_encode_delta(zigzag_decode(value)) != list(tokens):
                raise RuntimeError("LEB128 delta is not canonical")
            return zigzag_decode(value)
    raise RuntimeError("unterminated LEB128 delta")


def _zigzag_decode_tensor(unsigned):
    positive = torch.bitwise_right_shift(unsigned, 1)
    return torch.where(torch.bitwise_and(unsigned, 1) == 0, positive, -(positive + 1))


def _zigzag_encode_tensor(delta):
    return torch.where(delta >= 0, 2 * delta, -2 * delta - 1)


def _line_bits_tensor(lines, dtype):
    shifts = torch.arange(LINE_ADDRESS_BITS, device=lines.device, dtype=torch.int64)
    return torch.bitwise_and(torch.bitwise_right_shift(lines.unsqueeze(1), shifts), 1).to(dtype)


def _keyed_uniform_tensor(event_keys, decoder_seed, role, head, action_rank, device):
    values = np.asarray([
        np.float64(keyed_uniform(decoder_seed, TRACE, POLICY, role, key, head, action_rank))
        for key in event_keys
    ], dtype=np.float64)
    return torch.from_numpy(values).to(device=device, dtype=torch.float64)


def _sample_categorical_st(logits, uniforms, legal=None):
    if (
        logits.ndim != 2 or not bool(torch.isfinite(logits).all())
        or uniforms.dtype != torch.float64
        or uniforms.shape != (len(logits),)
    ):
        raise RuntimeError("categorical logits/uniform contract changed")
    if legal is not None:
        if legal.shape != logits.shape or not bool(legal.any(dim=1).all()):
            raise RuntimeError("categorical legality mask removed all probability mass")
        logits = logits.masked_fill(~legal, -torch.inf)
    probabilities64 = F.softmax(logits.detach().to(torch.float64), dim=-1)
    cumulative = probabilities64.cumsum(dim=-1)
    cumulative[:, -1] = 1.0
    hard = (uniforms.unsqueeze(1) >= cumulative).sum(dim=1).clamp(max=logits.shape[1] - 1)
    hard_one_hot = F.one_hot(hard.to(torch.long), logits.shape[1]).to(logits.dtype)
    soft = F.softmax(logits, dim=-1)
    feedback = hard_one_hot + soft - soft.detach()
    return hard.to(torch.long), feedback


def _assert_stop_representable(logits):
    if logits.ndim != 2 or logits.shape[1] != 2 or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("STOP/EMIT logits must be finite two-class rows")
    stop_probability = F.softmax(logits.detach().to(torch.float64), dim=-1)[:, 0]
    if bool((stop_probability <= KEYED_UNIFORM_HALF_BIN).any()):
        raise RuntimeError(
            "learned STOP interval is below the keyed inverse-CDF grid; "
            "refusing an unbounded action rollout"
        )


def _assert_action_rollout_watchdog(rank, active):
    if int(rank) >= ACTION_ROLLOUT_WATCHDOG_RANKS and bool(active.any()):
        raise RuntimeError(
            "learned STOP/EMIT rollout reached the fail-closed sampler-grid "
            "watchdog; no replay will be emitted"
        )


class RoutedPageState:
    def __init__(self, demand=None, fill=None, pages=None, page_last=None, offset=0, peak_pages=0):
        self.demand = demand
        self.fill = fill
        self.pages = {} if pages is None else pages
        self.page_last = {} if page_last is None else page_last
        self.offset = int(offset)
        self.peak_pages = int(peak_pages)


def detach_encoder_state(state):
    def detached(pair):
        return None if pair is None else tuple(value.detach() for value in pair)
    return RoutedPageState(
        demand=detached(state.demand), fill=detached(state.fill),
        pages={key: tuple(value.detach() for value in pair) for key, pair in state.pages.items()},
        page_last=dict(state.page_last), offset=state.offset, peak_pages=state.peak_pages,
    )


class ExactActionDecoder(nn.Module):
    def __init__(self, hidden_size, embed_size):
        super().__init__()
        self.hidden_size, self.embed_size = hidden_size, embed_size
        self.stop_emit = nn.Linear(hidden_size, 2)
        self.byte_condition = nn.Linear(hidden_size, embed_size)
        self.byte_position = nn.Embedding(LEB128_MAX_BYTES, embed_size)
        self.byte_head = nn.Linear(embed_size, BYTE_VOCAB)
        self.byte_embedding = nn.Embedding(BYTE_VOCAB, embed_size)
        self.byte_cell = nn.GRUCell(embed_size, embed_size)
        self.target_encoder = nn.Linear(LINE_ADDRESS_BITS, embed_size)
        self.fill_head = nn.Linear(hidden_size + embed_size, len(FILL_LEVELS))
        self.action_cell = nn.GRUCell(2 * embed_size + len(FILL_LEVELS), hidden_size)

    def initial_byte_state(self, state):
        return torch.tanh(self.byte_condition(state))

    def byte_logits(self, byte_state, position):
        pos = self.byte_position(torch.full(
            (len(byte_state),), int(position), device=byte_state.device,
            dtype=torch.long,
        ))
        return self.byte_head(torch.tanh(byte_state + pos))

    def _legal_byte_mask(self, prefix, position, origin, used_targets, used_mask):
        rows = len(prefix)
        legal = torch.zeros((rows, BYTE_VOCAB), dtype=torch.bool, device=prefix.device)
        terminal_count = 1 << BYTE_PAYLOAD_BITS
        payload = torch.arange(terminal_count, device=prefix.device, dtype=torch.int64)
        words = prefix.unsqueeze(1) + torch.bitwise_left_shift(
            payload, BYTE_PAYLOAD_BITS * position,
        )
        terminal = words < LINE_ADDRESS_MODULUS
        if position:
            terminal[:, 0] = False
        legal[:, :terminal_count] = terminal
        if used_targets.shape[1]:
            deltas = _zigzag_decode_tensor(words)
            targets = torch.remainder(origin.unsqueeze(1) + deltas, LINE_ADDRESS_MODULUS)
            duplicates = ((targets.unsqueeze(2) == used_targets.unsqueeze(1)) & used_mask.unsqueeze(1)).any(dim=2)
            legal[:, :terminal_count] &= ~duplicates
        if position < LEB128_MAX_BYTES - 1:
            # A continuation byte is legal only when its prefix has at least
            # one complete, in-domain, nonduplicate terminal descendant.  By
            # excluding an empty subtree now, byte nine can never dead-end and
            # sampling needs neither backtracking nor an address fallback.
            bits = BYTE_PAYLOAD_BITS * (position + 1)
            step = 1 << bits
            candidate_prefix = words
            possible = torch.div(
                (LINE_ADDRESS_MODULUS - 1) - candidate_prefix,
                step,
                rounding_mode="floor",
            ).clamp(min=0)
            blocked = torch.zeros_like(possible)
            if used_targets.shape[1]:
                difference = torch.remainder(
                    used_targets - origin.unsqueeze(1), LINE_ADDRESS_MODULUS,
                )
                signed = torch.where(
                    difference >= LINE_ADDRESS_HALF_RANGE,
                    difference - LINE_ADDRESS_MODULUS,
                    difference,
                )
                used_words = _zigzag_encode_tensor(signed)
                low_mask = step - 1
                descendants = (
                    (torch.bitwise_and(used_words.unsqueeze(1), low_mask)
                     == candidate_prefix.unsqueeze(2))
                    & (torch.bitwise_right_shift(used_words.unsqueeze(1), bits) >= 1)
                    & used_mask.unsqueeze(1)
                )
                blocked = descendants.sum(dim=2)
            legal[:, terminal_count:] = possible > blocked
        return legal

    def sample_delta(self, state, origin, used_targets, used_mask, event_keys, rank, decoder_seed, role):
        prefix = torch.zeros(len(state), dtype=torch.int64, device=state.device)
        done = torch.zeros(len(state), dtype=torch.bool, device=state.device)
        byte_state = self.initial_byte_state(state)
        lengths = torch.zeros(len(state), dtype=torch.int64, device=state.device)
        token_matrix = torch.full((len(state), LEB128_MAX_BYTES), -1, dtype=torch.int64, device=state.device)
        coordinates = []
        for position in range(LEB128_MAX_BYTES):
            active = torch.nonzero(~done, as_tuple=False).squeeze(1)
            if not len(active):
                break
            active_keys = [event_keys[int(index)] for index in active.cpu().tolist()]
            active_byte_state = byte_state.index_select(0, active)
            logits = self.byte_logits(active_byte_state, position)
            legal = self._legal_byte_mask(
                prefix.index_select(0, active), position, origin.index_select(0, active),
                used_targets.index_select(0, active), used_mask.index_select(0, active),
            )
            head = "delta_leb128_byte_{}".format(position)
            uniforms = _keyed_uniform_tensor(active_keys, decoder_seed, role, head, rank, state.device)
            hard, feedback = _sample_categorical_st(logits, uniforms, legal)
            token_matrix[active, position] = hard
            prefix[active] += torch.bitwise_left_shift(
                torch.bitwise_and(hard, (1 << BYTE_PAYLOAD_BITS) - 1),
                BYTE_PAYLOAD_BITS * position,
            )
            next_byte_state = self.byte_cell(
                feedback @ self.byte_embedding.weight, active_byte_state,
            )
            byte_state = byte_state.index_copy(0, active, next_byte_state)
            lengths[active] += 1
            ended = hard < (1 << BYTE_PAYLOAD_BITS)
            done[active[ended]] = True
            coordinates.extend((key, head, rank) for key in active_keys)
        if not bool(done.all()):
            raise RuntimeError("exact SPP LEB128 decoder did not terminate by byte nine")
        delta = _zigzag_decode_tensor(prefix)
        target = torch.remainder(origin + delta, LINE_ADDRESS_MODULUS)
        if used_targets.shape[1] and bool(((target.unsqueeze(1) == used_targets) & used_mask).any(dim=1).any()):
            raise RuntimeError("probability-masked SPP codec emitted a duplicate target")
        delta_feedback = byte_state
        return delta, target, delta_feedback, token_matrix, lengths, coordinates

    def teacher_delta_loss(self, state, origin, teacher_target, used_targets, used_mask):
        """Exact loss-only autoregressive likelihood on the teacher prefix.

        The teacher prefix conditions only this shared codec likelihood path.
        It never produces a runtime target and never mutates action state,
        origin, or duplicate history.
        """
        require_equal_lengths(
            "teacher delta likelihood", state, origin, teacher_target,
            used_targets, used_mask,
        )
        duplicate = torch.zeros(len(state), dtype=torch.bool, device=state.device)
        if used_targets.shape[1]:
            duplicate = ((teacher_target.unsqueeze(1) == used_targets) & used_mask).any(dim=1)
        eligible = ~duplicate
        skipped = int(duplicate.sum().item())
        if not bool(eligible.any()):
            return state.new_zeros(()), 0, skipped
        states = state[eligible]
        origins = origin[eligible]
        targets = teacher_target[eligible]
        used = used_targets[eligible]
        mask = used_mask[eligible]
        token_lists = [
            leb128_encode_delta(canonical_signed_delta(base, target))
            for base, target in zip(origins.cpu().tolist(), targets.cpu().tolist())
        ]
        byte_state = self.initial_byte_state(states)
        prefix = torch.zeros(len(states), dtype=torch.int64, device=states.device)
        total = states.new_zeros(())
        atoms = 0
        for position in range(max(map(len, token_lists))):
            active_list = [index for index, tokens in enumerate(token_lists) if position < len(tokens)]
            active = torch.as_tensor(active_list, device=states.device, dtype=torch.long)
            active_byte_state = byte_state.index_select(0, active)
            logits = self.byte_logits(active_byte_state, position)
            legal = self._legal_byte_mask(
                prefix.index_select(0, active), position,
                origins.index_select(0, active), used.index_select(0, active),
                mask.index_select(0, active),
            )
            tokens = torch.as_tensor(
                [token_lists[index][position] for index in active_list],
                device=states.device, dtype=torch.long,
            )
            if not bool(legal.gather(1, tokens.unsqueeze(1)).all()):
                raise RuntimeError("canonical teacher LEB128 token left the inference support")
            masked_logits = logits.masked_fill(~legal, -torch.inf)
            total = total + F.cross_entropy(masked_logits, tokens, reduction="sum")
            feedback = F.one_hot(tokens, BYTE_VOCAB).to(logits.dtype)
            next_state = self.byte_cell(
                feedback @ self.byte_embedding.weight, active_byte_state,
            )
            byte_state = byte_state.index_copy(0, active, next_state)
            prefix[active] += torch.bitwise_left_shift(
                torch.bitwise_and(tokens, (1 << BYTE_PAYLOAD_BITS) - 1),
                BYTE_PAYLOAD_BITS * position,
            )
            atoms += len(active_list)
        return total, atoms, skipped

    def fill_distribution(self, state, target):
        target_bits = _line_bits_tensor(target, state.dtype)
        target_embedding = torch.tanh(self.target_encoder(target_bits))
        logits = self.fill_head(torch.cat([state, target_embedding], dim=1))
        return logits, target_embedding

    def sample_fill(self, logits, event_keys, rank, decoder_seed, role):
        uniforms = _keyed_uniform_tensor(event_keys, decoder_seed, role, "fill_after_target", rank, logits.device)
        hard, feedback = _sample_categorical_st(logits, uniforms)
        coordinates = [(key, "fill_after_target", rank) for key in event_keys]
        return hard, feedback, coordinates

    def advance(self, state, delta_feedback, target_embedding, fill_feedback):
        return self.action_cell(torch.cat([delta_feedback, target_embedding, fill_feedback], dim=1), state)


class RoutedPageSPPLSTM(nn.Module):
    def __init__(self, feature_count, hidden_size):
        super().__init__()
        if feature_count != RUNTIME_FEATURES or hidden_size not in MODEL_POINTS["lstm"]:
            raise ValueError("unsupported routed SPP model dimensions")
        self.feature_count, self.hidden_size = feature_count, hidden_size
        self.route_size = routed_state_size(hidden_size)
        self.codec_size = codec_embed_size(hidden_size)
        self.input_projection = nn.Linear(feature_count, self.route_size)
        self.demand_lstm = nn.LSTM(self.route_size, self.route_size, batch_first=True)
        self.fill_lstm = nn.LSTM(self.route_size, self.route_size, batch_first=True)
        self.page_lstm = nn.LSTM(self.route_size + 1, self.route_size, batch_first=True)
        self.page_validity = nn.Linear(self.route_size + 1, self.route_size)
        self.fusion = nn.Linear(4 * self.route_size, hidden_size)
        self.decoder = ExactActionDecoder(hidden_size, self.codec_size)

    def _route(self, embedded, mask, cell, initial):
        if initial is None:
            zero = embedded.new_zeros((1, 1, self.route_size))
            initial = (zero, zero.clone())
        selected = embedded[mask]
        if len(selected):
            output, final = cell(selected.unsqueeze(0), initial)
            output = output.squeeze(0)
        else:
            output, final = embedded.new_zeros((0, self.route_size)), initial
        cumulative = mask.to(torch.int64).cumsum(0)
        padded = torch.cat([initial[0][0], output], dim=0)
        return padded.index_select(0, cumulative), final

    def _pages(self, embedded, page_ids, state):
        groups = defaultdict(list)
        for index, page in enumerate(page_ids):
            groups[int(page)].append(index)
        active_pages = list(groups)
        lengths = [len(groups[page]) for page in active_pages]
        maximum = max(lengths)
        padded = embedded.new_zeros((len(active_pages), maximum, self.route_size + 1))
        positions = []
        age_by_position = embedded.new_zeros((len(embedded), 1))
        initial_h, initial_c = [], []
        for row, page in enumerate(active_pages):
            indices = groups[page]
            positions.append(indices)
            initial = state.pages.get(page)
            if initial is None:
                initial_h.append(embedded.new_zeros(self.route_size))
                initial_c.append(embedded.new_zeros(self.route_size))
            else:
                initial_h.append(initial[0])
                initial_c.append(initial[1])
            previous = state.page_last.get(page)
            for column, index in enumerate(indices):
                absolute = state.offset + index
                gap = 1 if previous is None else max(1, absolute - previous)
                age = math.log1p(gap)
                padded[row, column, :self.route_size] = embedded[index]
                padded[row, column, self.route_size] = age
                age_by_position[index, 0] = age
                previous = absolute
            state.page_last[page] = previous
        h0 = torch.stack(initial_h).unsqueeze(0)
        c0 = torch.stack(initial_c).unsqueeze(0)
        packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.page_lstm(packed, (h0, c0))
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=maximum)
        page_context = embedded.new_zeros((len(embedded), self.route_size))
        for row, page in enumerate(active_pages):
            indices = positions[row]
            page_context[torch.as_tensor(indices, device=embedded.device)] = output[row, :len(indices)]
            state.pages[page] = (hn[0, row], cn[0, row])
        state.peak_pages = max(state.peak_pages, len(state.pages))
        return page_context, age_by_position

    def encode(self, features, demand_kind, page_ids, state=None):
        if features.ndim != 2 or features.shape[1] != self.feature_count:
            raise RuntimeError("routed SPP encoder shape changed")
        if state is None:
            state = RoutedPageState()
        embedded = torch.tanh(self.input_projection(features))
        demand_context, state.demand = self._route(embedded, demand_kind, self.demand_lstm, state.demand)
        fill_context, state.fill = self._route(embedded, ~demand_kind, self.fill_lstm, state.fill)
        page_context, page_age = self._pages(embedded, page_ids, state)
        validity = torch.sigmoid(self.page_validity(torch.cat([page_context, page_age], dim=1)))
        fused = torch.tanh(self.fusion(torch.cat([
            demand_context, fill_context, validity * page_context, embedded,
        ], dim=1)))
        state.offset += len(features)
        return fused, state


def _iter_chunks(length, width):
    for start in range(0, length, width):
        yield start, min(length, start + width)


def _teacher_fill_loss(decoder, states, teacher_targets, teacher_fill_classes):
    require_equal_lengths(
        "teacher conditional fill likelihood", states, teacher_targets,
        teacher_fill_classes,
    )
    logits, _ = decoder.fill_distribution(states, teacher_targets)
    return F.cross_entropy(logits, teacher_fill_classes, reduction="sum"), len(states)


def _normalized_categorical_loss(losses, atom_count):
    if not losses or atom_count < 1:
        raise RuntimeError("categorical loss normalization needs positive support")
    return torch.stack(losses).sum() / int(atom_count)


def structured_loss(model, context, counts, teacher_lines, teacher_fills, base_lines, event_keys, decoder_seed):
    valid = counts >= 0
    if not bool(valid.any()):
        raise RuntimeError("training chunk has no SPP decision callbacks")
    state = context[valid]
    decision_counts = counts[valid]
    teacher_lines = teacher_lines[valid]
    teacher_fills = teacher_fills[valid]
    origin = base_lines[valid].clone()
    keys = [event_keys[index] for index in torch.nonzero(valid, as_tuple=False).squeeze(1).cpu().tolist()]
    width = teacher_lines.shape[1]
    used_targets = torch.zeros((len(state), max(1, width)), dtype=torch.int64, device=state.device)
    used_mask = torch.zeros((len(state), max(1, width)), dtype=torch.bool, device=state.device)
    rollout_active = torch.ones(len(state), dtype=torch.bool, device=state.device)
    grammar_sum = state.new_zeros(())
    byte_sum = state.new_zeros(())
    fill_sum = state.new_zeros(())
    grammar_atoms = byte_atoms = action_atoms = 0
    sampled_stop_atoms = sampled_emit_atoms = 0
    teacher_delta_duplicate_illegal_skips = 0
    coordinates = []
    rank = 0
    while bool(rollout_active.any()):
        # Only rows reached by the sampled rollout bear this rank's grammar
        # NLL.  Teacher count supplies the EMIT/STOP label; it cannot keep a
        # stopped row alive or create a later stale-state loss.
        reached = torch.nonzero(rollout_active, as_tuple=False).squeeze(1)
        reached_state = state.index_select(0, reached)
        reached_keys = [keys[int(index)] for index in reached.cpu().tolist()]
        grammar_logits = model.decoder.stop_emit(reached_state)
        _assert_stop_representable(grammar_logits)
        grammar_targets = (decision_counts[reached] > rank).to(torch.long)
        grammar_sum = grammar_sum + F.cross_entropy(
            grammar_logits, grammar_targets, reduction="sum",
        )
        grammar_atoms += len(reached)

        # Teacher-prefix target and teacher-target-conditioned fill likelihoods
        # are loss-only branches at the reached state.  They share the runtime
        # heads and legality support but cannot mutate the sampled trajectory.
        teacher_at_reached = grammar_targets == 1
        if bool(teacher_at_reached.any()):
            local_teacher = torch.nonzero(teacher_at_reached, as_tuple=False).squeeze(1)
            teacher_rows = reached.index_select(0, local_teacher)
            teacher_targets_at_rank = teacher_lines[teacher_rows, rank]
            prior_used = used_targets.index_select(0, teacher_rows)[:, :rank]
            prior_mask = used_mask.index_select(0, teacher_rows)[:, :rank]
            token_loss, token_atoms, skipped = model.decoder.teacher_delta_loss(
                state.index_select(0, teacher_rows), origin.index_select(0, teacher_rows),
                teacher_targets_at_rank, prior_used, prior_mask,
            )
            byte_sum = byte_sum + token_loss
            byte_atoms += token_atoms
            teacher_delta_duplicate_illegal_skips += skipped
            legal_teacher = torch.ones(
                len(teacher_rows), dtype=torch.bool, device=state.device,
            )
            if prior_used.shape[1]:
                legal_teacher = ~(
                    (teacher_targets_at_rank.unsqueeze(1) == prior_used) & prior_mask
                ).any(dim=1)
            if bool(legal_teacher.any()):
                fill_rows = teacher_rows[legal_teacher]
                fill_loss, fill_atoms = _teacher_fill_loss(
                    model.decoder, state.index_select(0, fill_rows),
                    teacher_lines[fill_rows, rank], teacher_fills[fill_rows, rank],
                )
                fill_sum = fill_sum + fill_loss
                action_atoms += fill_atoms

        # STOP/EMIT is always a stateless keyed draw from learned logits.
        uniforms = _keyed_uniform_tensor(
            reached_keys, decoder_seed, "train", "stop_emit", rank, state.device,
        )
        sampled_grammar, _ = _sample_categorical_st(grammar_logits, uniforms)
        coordinates.extend((key, "stop_emit", rank) for key in reached_keys)
        stopped = reached[sampled_grammar == 0]
        emitted = reached[sampled_grammar == 1]
        sampled_stop_atoms += len(stopped)
        sampled_emit_atoms += len(emitted)
        rollout_active[stopped] = False

        # Only actual sampled EMIT rows run the sampled-prefix codec, sample
        # fill after their actual target, and change the next-rank trajectory.
        if len(emitted):
            if rank >= used_targets.shape[1]:
                growth = max(1, used_targets.shape[1])
                used_targets = torch.cat([
                    used_targets,
                    torch.zeros((len(state), growth), dtype=torch.int64, device=state.device),
                ], dim=1)
                used_mask = torch.cat([
                    used_mask,
                    torch.zeros((len(state), growth), dtype=torch.bool, device=state.device),
                ], dim=1)
            emitted_state = state.index_select(0, emitted)
            emitted_keys = [keys[int(index)] for index in emitted.cpu().tolist()]
            _, sampled_target, delta_feedback, _, _, delta_coords = model.decoder.sample_delta(
                emitted_state, origin.index_select(0, emitted),
                used_targets.index_select(0, emitted)[:, :rank],
                used_mask.index_select(0, emitted)[:, :rank],
                emitted_keys, rank, decoder_seed, "train",
            )
            fill_logits, target_embedding = model.decoder.fill_distribution(
                emitted_state, sampled_target,
            )
            _, fill_feedback, fill_coords = model.decoder.sample_fill(
                fill_logits, emitted_keys, rank, decoder_seed, "train",
            )
            advanced = model.decoder.advance(
                emitted_state, delta_feedback, target_embedding, fill_feedback,
            )
            state = state.index_copy(0, emitted, advanced)
            origin[emitted] = sampled_target.detach()
            used_targets[emitted, rank] = sampled_target.detach()
            used_mask[emitted, rank] = True
            coordinates.extend(delta_coords)
            coordinates.extend(fill_coords)
        rank += 1
        _assert_action_rollout_watchdog(rank, rollout_active)
    loss = grammar_sum + byte_sum + fill_sum
    return loss, {
        "grammar_nll_sum": float(grammar_sum.detach()), "byte_nll_sum": float(byte_sum.detach()),
        "fill_nll_sum": float(fill_sum.detach()), "grammar_atoms": grammar_atoms,
        "byte_atoms": byte_atoms, "action_atoms": action_atoms,
        "sampled_stop_atoms": sampled_stop_atoms,
        "sampled_emit_atoms": sampled_emit_atoms,
        "teacher_delta_duplicate_illegal_skips": teacher_delta_duplicate_illegal_skips,
        "total_categorical_atoms": grammar_atoms + byte_atoms + action_atoms,
    }, coordinates


def train_model(
    model, bundle, targets, event_key_by_context, device, epochs, chunk_len,
    accumulate_chunks, learning_rate, decoder_seed,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    features = torch.from_numpy(bundle["features"])
    kinds = torch.from_numpy(bundle["demand_kind"])
    counts = torch.from_numpy(targets[0]).to(torch.long)
    lines = torch.from_numpy(targets[1]).to(torch.long)
    fills = torch.from_numpy(targets[2]).to(torch.long)
    bases = torch.from_numpy(bundle["lines"]).to(torch.long)
    chunks = list(_iter_chunks(len(features), chunk_len))
    history, all_coordinates = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        encoder_state = None
        totals = {
            "grammar_nll_sum": 0.0, "byte_nll_sum": 0.0,
            "fill_nll_sum": 0.0, "grammar_atoms": 0,
            "byte_atoms": 0, "action_atoms": 0,
            "sampled_stop_atoms": 0, "sampled_emit_atoms": 0,
            "teacher_delta_duplicate_illegal_skips": 0,
            "total_categorical_atoms": 0,
        }
        steps = 0
        epoch_coordinates = []
        for group_start in range(0, len(chunks), accumulate_chunks):
            group = chunks[group_start:group_start + accumulate_chunks]
            optimizer.zero_grad(set_to_none=True)
            group_losses = []
            group_atoms = 0
            for start, stop in group:
                xb = features[start:stop].to(device)
                kb = kinds[start:stop].to(device)
                context, encoder_state = model.encode(xb, kb, bundle["pages"][start:stop], encoder_state)
                encoder_state = detach_encoder_state(encoder_state)
                if not np.any(targets[0][start:stop] >= 0):
                    continue
                loss, components, coordinates = structured_loss(
                    model, context, counts[start:stop].to(device), lines[start:stop].to(device),
                    fills[start:stop].to(device), bases[start:stop].to(device),
                    event_key_by_context[start:stop], decoder_seed=decoder_seed,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite SPP v19 training loss")
                group_losses.append(loss)
                group_atoms += components["total_categorical_atoms"]
                for key, value in components.items():
                    totals[key] += value
                epoch_coordinates.extend(coordinates)
            if not group_losses:
                continue
            if group_atoms < 1:
                raise RuntimeError("optimizer group has zero categorical SPP v19 atoms")
            _normalized_categorical_loss(group_losses, group_atoms).backward()
            optimizer.step()
            steps += 1
        row = {
            "epoch": epoch,
            "categorical_nll_per_atom": (
                totals["grammar_nll_sum"] + totals["byte_nll_sum"] + totals["fill_nll_sum"]
            ) / max(1, totals["total_categorical_atoms"]),
            "grammar_nll_per_token": totals["grammar_nll_sum"] / max(1, totals["grammar_atoms"]),
            "leb128_byte_nll_per_byte": totals["byte_nll_sum"] / max(1, totals["byte_atoms"]),
            "fill_nll_per_action": totals["fill_nll_sum"] / max(1, totals["action_atoms"]),
            "grammar_atoms": totals["grammar_atoms"], "leb128_byte_atoms": totals["byte_atoms"],
            "action_atoms": totals["action_atoms"], "chronological_chunks": len(chunks),
            "total_categorical_atoms": totals["total_categorical_atoms"],
            "sampled_stop_atoms": totals["sampled_stop_atoms"],
            "sampled_emit_atoms": totals["sampled_emit_atoms"],
            "teacher_delta_duplicate_illegal_skips": totals["teacher_delta_duplicate_illegal_skips"],
            "optimizer_steps": steps,
        }
        history.append(row)
        all_coordinates = epoch_coordinates
        print("[train:spp-v19] epoch={} grammar={:.8f} byte={:.8f} fill={:.8f}".format(
            epoch, row["grammar_nll_per_token"], row["leb128_byte_nll_per_byte"], row["fill_nll_per_action"]
        ))
    return history, all_coordinates


def score_context(model, bundle, device, initial_state=None, chunk_len=8192):
    model.eval()
    parts, state = [], initial_state
    with torch.no_grad():
        for start, stop in _iter_chunks(len(bundle["features"]), chunk_len):
            features = torch.from_numpy(bundle["features"][start:stop]).to(device)
            kinds = torch.from_numpy(bundle["demand_kind"][start:stop]).to(device)
            context, state = model.encode(features, kinds, bundle["pages"][start:stop], state)
            state = detach_encoder_state(state)
            parts.append(context.cpu().numpy())
    return np.concatenate(parts, axis=0), state


def decode_actions(model, contexts, base_lines, event_keys, device, decoder_seed, role, chunk_len=8192):
    if not (len(contexts) == len(base_lines) == len(event_keys)):
        raise RuntimeError("SPP v19 decoder row counts differ")
    counts = np.zeros(len(contexts), dtype=np.int64)
    predicted_lines, predicted_fills, coordinates = [[] for _ in contexts], [[] for _ in contexts], []
    model.eval()
    with torch.no_grad():
        for start, stop in _iter_chunks(len(contexts), chunk_len):
            state = torch.from_numpy(contexts[start:stop]).to(device)
            origin = torch.as_tensor(base_lines[start:stop], dtype=torch.int64, device=device)
            active = torch.ones(len(state), dtype=torch.bool, device=device)
            used = [[] for _ in range(len(state))]
            rank = 0
            while bool(active.any()):
                indices = torch.nonzero(active, as_tuple=False).squeeze(1)
                active_keys = [event_keys[start + int(index)] for index in indices.cpu().tolist()]
                logits = model.decoder.stop_emit(state.index_select(0, indices))
                _assert_stop_representable(logits)
                uniforms = _keyed_uniform_tensor(active_keys, decoder_seed, role, "stop_emit", rank, device)
                grammar, _ = _sample_categorical_st(logits, uniforms)
                coordinates.extend((key, "stop_emit", rank) for key in active_keys)
                stop_indices = indices[grammar == 0]
                active[stop_indices] = False
                emit_indices = indices[grammar == 1]
                if not len(emit_indices):
                    rank += 1
                    continue
                emit_keys = [event_keys[start + int(index)] for index in emit_indices.cpu().tolist()]
                max_used = max((len(used[int(index)]) for index in emit_indices.cpu().tolist()), default=0)
                used_tensor = torch.zeros((len(emit_indices), max_used), dtype=torch.int64, device=device)
                used_mask = torch.zeros((len(emit_indices), max_used), dtype=torch.bool, device=device)
                for row, index in enumerate(emit_indices.cpu().tolist()):
                    if used[index]:
                        used_tensor[row, :len(used[index])] = torch.as_tensor(used[index], dtype=torch.int64, device=device)
                        used_mask[row, :len(used[index])] = True
                active_state = state.index_select(0, emit_indices)
                _, target, delta_feedback, _, _, delta_coords = model.decoder.sample_delta(
                    active_state, origin.index_select(0, emit_indices), used_tensor, used_mask,
                    emit_keys, rank, decoder_seed, role,
                )
                fill_logits, target_embedding = model.decoder.fill_distribution(active_state, target)
                hard_fill, fill_feedback, fill_coords = model.decoder.sample_fill(
                    fill_logits, emit_keys, rank, decoder_seed, role,
                )
                advanced = model.decoder.advance(active_state, delta_feedback, target_embedding, fill_feedback)
                state = state.index_copy(0, emit_indices, advanced)
                origin[emit_indices] = target
                emitted_rows = emit_indices.cpu().tolist()
                target_rows = target.cpu().tolist()
                fill_rows = hard_fill.cpu().tolist()
                require_equal_lengths(
                    "decoded emitted actions", emitted_rows, target_rows, fill_rows,
                )
                for index, target_line, fill_class in zip(emitted_rows, target_rows, fill_rows):
                    used[index].append(int(target_line))
                    predicted_lines[start + index].append(int(target_line))
                    predicted_fills[start + index].append(int(fill_class))
                    counts[start + index] += 1
                coordinates.extend(delta_coords)
                coordinates.extend(fill_coords)
                rank += 1
                _assert_action_rollout_watchdog(rank, active)
    return counts, predicted_lines, predicted_fills, coordinates


def trigger_behavior_metrics(predicted_counts, teacher_actions):
    require_equal_lengths("trigger behavior", predicted_counts, teacher_actions)
    predicted = np.asarray(predicted_counts, dtype=np.int64)
    normal = np.asarray([len(items) for items in teacher_actions], dtype=np.int64)
    pp, np_ = predicted > 0, normal > 0
    tp, fp, fn = int((pp & np_).sum()), int((pp & ~np_).sum()), int((~pp & np_).sum())
    ratio = lambda a, b: float(a) / float(b) if b else 0.0
    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    return {
        "normal_positive_callbacks": int(np_.sum()), "normal_zero_callbacks": int((~np_).sum()),
        "predicted_positive_callbacks": int(pp.sum()), "true_positive_trigger_callbacks": tp,
        "false_positive_trigger_callbacks": fp, "false_negative_trigger_callbacks": fn,
        "normal_positive_callback_rate": ratio(int(np_.sum()), len(np_)),
        "predicted_positive_callback_rate": ratio(int(pp.sum()), len(pp)),
        "trigger_precision": precision, "trigger_recall": recall,
        "trigger_f1": ratio(2 * precision * recall, precision + recall),
        "mean_normal_actions_per_positive_callback": ratio(int(normal.sum()), int(np_.sum())),
        "mean_predicted_actions_per_positive_callback": ratio(int(predicted.sum()), int(pp.sum())),
        "predicted_to_normal_action_ratio": ratio(int(predicted.sum()), int(normal.sum())),
    }


def joint_action_metrics(predicted_lines, predicted_fills, teacher_actions):
    require_equal_lengths(
        "joint action metrics", predicted_lines, predicted_fills,
        teacher_actions,
    )
    tp = pred_total = teacher_total = l2_pred = l2_teacher = l2_tp = 0
    for lines, fills, truth in zip(predicted_lines, predicted_fills, teacher_actions):
        require_equal_lengths("joint predicted action", lines, fills)
        predicted = Counter((int(line), FILL_LEVELS[int(fill)]) for line, fill in zip(lines, fills))
        teacher = Counter((int(line), int(fill)) for line, fill in truth)
        intersection = predicted & teacher
        tp += sum(intersection.values()); pred_total += sum(predicted.values()); teacher_total += sum(teacher.values())
        l2_pred += sum(value for (_, fill), value in predicted.items() if fill == 2)
        l2_teacher += sum(value for (_, fill), value in teacher.items() if fill == 2)
        l2_tp += sum(value for (_, fill), value in intersection.items() if fill == 2)
    ratio = lambda a, b: float(a) / float(b) if b else 0.0
    precision, recall = ratio(tp, pred_total), ratio(tp, teacher_total)
    l2_precision, l2_recall = ratio(l2_tp, l2_pred), ratio(l2_tp, l2_teacher)
    return {
        "joint_true_positive_actions": int(tp), "joint_action_precision": precision,
        "joint_action_recall": recall, "joint_action_f1": ratio(2 * precision * recall, precision + recall),
        "predicted_l2_actions": int(l2_pred), "teacher_l2_actions": int(l2_teacher),
        "l2_joint_true_positive_actions": int(l2_tp), "l2_joint_precision": l2_precision,
        "l2_joint_recall": l2_recall, "l2_joint_f1": ratio(2 * l2_precision * l2_recall, l2_precision + l2_recall),
        "predicted_l2_fraction": ratio(l2_pred, pred_total), "teacher_l2_fraction": ratio(l2_teacher, teacher_total),
    }


def complete_behavior_metrics(counts, lines, fills, teacher):
    result = behavior_metrics(counts, lines, fills, teacher, fill_levels=FILL_LEVELS)
    result.update(trigger_behavior_metrics(counts, teacher))
    result.update(joint_action_metrics(lines, fills, teacher))
    return result


def action_legality_diagnostics(base_lines, counts, predicted_lines):
    require_equal_lengths(
        "action legality", base_lines, counts, predicted_lines,
    )
    self_targets = duplicate_targets = 0
    for base, targets in zip(base_lines, predicted_lines):
        seen = set()
        for target in targets:
            self_targets += int(int(target) == int(base))
            duplicate_targets += int(int(target) in seen)
            seen.add(int(target))
    if duplicate_targets or int(np.asarray(counts).sum()) != sum(map(len, predicted_lines)):
        raise RuntimeError("v19 exact action grammar legality/accounting failed")
    return {
        "raw_predicted_action_count": int(np.asarray(counts).sum()),
        "materialized_distinct_action_count": int(sum(map(len, predicted_lines))),
        "raw_positive_callback_count": int((np.asarray(counts) > 0).sum()),
        "materialized_positive_callback_count": int(sum(bool(items) for items in predicted_lines)),
        "count_to_materialized_shortfall": 0, "self_target_actions": self_targets,
        "duplicate_target_actions": duplicate_targets, "self_target_actions_allowed": True,
        "duplicate_mask_mode": "categorical_probability_mask_then_renormalize",
        "delta_legality_fallback": None,
    }


def self_test_model(hidden_size):
    for size in MODEL_POINTS["lstm"]:
        expected = expected_parameter_count(size)
        model = RoutedPageSPPLSTM(RUNTIME_FEATURES, size)
        observed = sum(parameter.numel() for parameter in model.parameters())
        if observed != expected or observed >= 10000:
            raise RuntimeError("SPP v19 parameter formula mismatch at h{}: {} != {}".format(size, observed, expected))
    for delta in (0, 1, -1, 63, -64, 64, -65, LINE_ADDRESS_HALF_RANGE - 1, -LINE_ADDRESS_HALF_RANGE):
        tokens = leb128_encode_delta(delta)
        if leb128_decode_delta(tokens) != delta or len(tokens) > LEB128_MAX_BYTES:
            raise RuntimeError("exact signed ZigZag/LEB128 round trip failed")
    self_test_keyed_crn()
    if sampler_metadata().get("uniform_mapping") != (
        "sha256_top_{}_bits_half_bin_open_interval".format(SAMPLER_GRID_BITS)
    ):
        raise RuntimeError("SPP action watchdog and keyed sampler precision differ")
    model = RoutedPageSPPLSTM(RUNTIME_FEATURES, hidden_size)
    model.eval()
    features = torch.zeros((5, RUNTIME_FEATURES))
    kinds = torch.tensor([True, False, True, True, False])
    pages = np.asarray([1, 2, 1, 3, 2], dtype=np.int64)
    changed = features.clone(); changed[-1, 0] = 1.0
    with torch.no_grad():
        first, _ = model.encode(features, kinds, pages)
        second, _ = model.encode(changed, kinds, pages)
    if not torch.equal(first[:-1], second[:-1]):
        raise RuntimeError("future callback changed prior routed/page-local states")
    required = {
        "demand_lstm.weight_ih_l0", "fill_lstm.weight_ih_l0", "page_lstm.weight_ih_l0",
        "page_validity.weight", "decoder.stop_emit.weight", "decoder.byte_head.weight",
        "decoder.byte_cell.weight_ih", "decoder.fill_head.weight",
        "decoder.action_cell.weight_ih",
    }
    if not required.issubset(dict(model.named_parameters())):
        raise RuntimeError("SPP v19 architecture is missing a routed/grammar component")
    stop_emit, _ = _sample_categorical_st(
        torch.log(torch.tensor([[0.75, 0.25], [0.75, 0.25]])),
        torch.tensor([0.50, 0.90], dtype=torch.float64),
    )
    if stop_emit.tolist() != [0, 1]:
        raise RuntimeError("learned STOP/EMIT categorical inverse CDF changed")
    try:
        _sample_categorical_st(
            torch.tensor([[float("nan"), 0.0]]),
            torch.tensor([0.5], dtype=torch.float64),
        )
    except RuntimeError:
        pass
    else:
        raise RuntimeError("nonfinite categorical logits were accepted")
    origin = torch.tensor([10], dtype=torch.int64)
    first_rank = model.decoder._legal_byte_mask(
        torch.zeros(1, dtype=torch.int64), 0, origin,
        torch.empty((1, 0), dtype=torch.int64),
        torch.empty((1, 0), dtype=torch.bool),
    )
    after_self_target = model.decoder._legal_byte_mask(
        torch.zeros(1, dtype=torch.int64), 0, origin,
        torch.tensor([[10]], dtype=torch.int64),
        torch.tensor([[True]], dtype=torch.bool),
    )
    if not bool(first_rank[0, 0]) or bool(after_self_target[0, 0]) or not bool(after_self_target[0, 2]):
        raise RuntimeError("SPP v19 self-target/duplicate probability mask changed")

    # Exhaust the three possible terminal descendants of one byte-eight
    # continuation prefix.  The prefix-feasibility mask must remove that
    # continuation before it can produce a byte-nine dead end.
    position = LEB128_MAX_BYTES - 2
    step = 1 << (BYTE_PAYLOAD_BITS * (position + 1))
    blocked_words = torch.tensor([step, 2 * step, 3 * step], dtype=torch.int64)
    blocked_targets = torch.remainder(
        origin.unsqueeze(1) + _zigzag_decode_tensor(blocked_words).unsqueeze(0),
        LINE_ADDRESS_MODULUS,
    )
    feasible = model.decoder._legal_byte_mask(
        torch.zeros(1, dtype=torch.int64), position, origin,
        blocked_targets, torch.ones_like(blocked_targets, dtype=torch.bool),
    )
    continuation_zero = 1 << BYTE_PAYLOAD_BITS
    if bool(feasible[0, continuation_zero]) or not bool(feasible[0].any()):
        raise RuntimeError("duplicate-prefix feasibility mask can dead-end")

    # The loss-only teacher-prefix path uses the same canonical/duplicate
    # support and cannot mutate the sampled codec inputs or history.
    state = torch.zeros((1, hidden_size))
    empty_targets = torch.empty((1, 0), dtype=torch.int64)
    empty_mask = torch.empty((1, 0), dtype=torch.bool)
    state_before, origin_before = state.clone(), origin.clone()
    teacher_loss, teacher_atoms, skipped = model.decoder.teacher_delta_loss(
        state, origin, origin.clone(), empty_targets, empty_mask,
    )
    if (
        not torch.isfinite(teacher_loss) or teacher_atoms != 1 or skipped != 0
        or not torch.equal(state, state_before) or not torch.equal(origin, origin_before)
    ):
        raise RuntimeError("canonical teacher-prefix loss/state isolation changed")
    duplicate_loss, duplicate_atoms, duplicate_skipped = model.decoder.teacher_delta_loss(
        state, origin, origin.clone(), origin.unsqueeze(1),
        torch.ones((1, 1), dtype=torch.bool),
    )
    if duplicate_atoms != 0 or duplicate_skipped != 1 or float(duplicate_loss) != 0.0:
        raise RuntimeError("duplicate-illegal teacher target was not skipped")

    sample_args = (
        state, origin, empty_targets, empty_mask,
        ["self_test_event"], 0, 7, "train",
    )
    first_sample = model.decoder.sample_delta(*sample_args)
    model.decoder.teacher_delta_loss(
        state, origin, torch.remainder(origin + 12345, LINE_ADDRESS_MODULUS),
        empty_targets, empty_mask,
    )
    second_sample = model.decoder.sample_delta(*sample_args)
    if not torch.equal(first_sample[1], second_sample[1]) or not torch.equal(first_sample[3], second_sample[3]):
        raise RuntimeError("teacher-prefix likelihood changed the sampled-prefix path")

    fill_loss, fill_atoms = _teacher_fill_loss(
        model.decoder, state, torch.remainder(origin + 1, LINE_ADDRESS_MODULUS),
        torch.zeros(1, dtype=torch.long),
    )
    if not torch.isfinite(fill_loss) or fill_atoms != 1:
        raise RuntimeError("teacher-target-conditioned fill likelihood changed")

    # A teacher with two actions cannot keep a model-forced STOP trajectory
    # alive.  Only one reached grammar coordinate may be sampled and no actual
    # byte/fill coordinate may exist.
    rollout_model = RoutedPageSPPLSTM(RUNTIME_FEATURES, hidden_size)
    with torch.no_grad():
        rollout_model.decoder.stop_emit.weight.zero_()
        rollout_model.decoder.stop_emit.bias.copy_(torch.tensor([100.0, -100.0]))
    _, components, coordinates = structured_loss(
        rollout_model, torch.zeros((1, hidden_size)), torch.tensor([2]),
        torch.tensor([[11, 12]], dtype=torch.int64),
        torch.tensor([[0, 1]], dtype=torch.int64), origin,
        ["grammar_rollout_self_test"], 7,
    )
    if (
        components["grammar_atoms"] != 1
        or components["sampled_stop_atoms"] != 1
        or components["sampled_emit_atoms"] != 0
        or coordinates != [("grammar_rollout_self_test", "stop_emit", 0)]
    ):
        raise RuntimeError("teacher grammar leaked into sampled recurrent rollout")

    try:
        _assert_stop_representable(torch.tensor([[-1000.0, 1000.0]]))
    except RuntimeError:
        pass
    else:
        raise RuntimeError("unrepresentable learned STOP interval was accepted")
    always_emit_logits = torch.tensor([[math.log(2 * KEYED_UNIFORM_HALF_BIN), 0.0]])
    _assert_stop_representable(always_emit_logits)
    always_emit_rank = 0
    always_emit_active = torch.ones(1, dtype=torch.bool)
    while always_emit_rank < ACTION_ROLLOUT_WATCHDOG_RANKS:
        action, _ = _sample_categorical_st(
            always_emit_logits, torch.tensor([0.5], dtype=torch.float64),
        )
        if action.item() != 1:
            raise RuntimeError("always-EMIT watchdog self-test did not emit")
        always_emit_rank += 1
        if always_emit_rank < ACTION_ROLLOUT_WATCHDOG_RANKS:
            _assert_action_rollout_watchdog(always_emit_rank, always_emit_active)
    try:
        _assert_action_rollout_watchdog(always_emit_rank, always_emit_active)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("always-EMIT rollout bypassed fail-closed watchdog")
    if float(_normalized_categorical_loss([torch.tensor(2.0), torch.tensor(4.0)], 3)) != 2.0:
        raise RuntimeError("categorical atom normalization changed")
    self_test_exact_int()


def main():
    if sys.argv[1:] == ["--describe-model-points"]:
        print(json.dumps(describe_model_points(), indent=2, sort_keys=True))
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, choices=[POLICY])
    for role in ("train", "guard", "eval"):
        parser.add_argument("--{}-stream".format(role), required=True, type=Path)
        parser.add_argument("--{}-teacher-actions".format(role), required=True, type=Path)
    parser.add_argument("--source-contract", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-family", choices=["lstm"], required=True)
    parser.add_argument("--model-size", type=int, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--decoder-seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--chunk-len", type=int, default=1024)
    parser.add_argument("--accumulate-chunks", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.decoder_seed is None:
        args.decoder_seed = args.seed
    source_contract = json.loads(args.source_contract.read_text())
    if source_contract.get("decision_effective_external_input") != SOURCE_INPUTS:
        raise RuntimeError("unexpected SPP source input contract")
    if MODEL_POINTS["lstm"].get(args.model_size) != args.pair_id:
        raise RuntimeError("model size/pair is not a configured v19 point")
    if min(args.epochs, args.chunk_len, args.accumulate_chunks) < 1:
        raise RuntimeError("model/training dimensions must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    self_test_model(args.model_size)
    # Self-tests intentionally instantiate/sample models.  Production RNG
    # state is established only afterwards so test evolution cannot change a
    # pinned seed's initial weights or training trajectory.
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    roles = ("train", "guard", "eval")
    stream_paths = {role: getattr(args, role + "_stream") for role in roles}
    action_paths = {role: getattr(args, role + "_teacher_actions") for role in roles}
    streams = {role: load_stream(stream_paths[role]) for role in roles}
    teachers = {role: load_teacher_actions(action_paths[role], streams[role]["demands"]) for role in roles}
    bundles = {role: runtime_bundle(streams[role]) for role in roles}
    for role in roles:
        if bundles[role]["features"].shape[1] != RUNTIME_FEATURES or not np.array_equal(bundles[role]["features"], runtime_bundle(streams[role])["features"]):
            raise RuntimeError("{} training/inference runtime encoder differs".format(role))
    event_keys = {role: sampling_event_keys(streams[role]) for role in roles}
    targets = {role: build_context_targets(streams[role], teachers[role]) for role in roles}
    train_key_by_context = [None] * len(streams["train"]["context"])
    require_equal_lengths(
        "train event-key routing", event_keys["train"],
        streams["train"]["demand_positions"],
    )
    for key, position in zip(event_keys["train"], streams["train"]["demand_positions"]):
        train_key_by_context[int(position)] = key

    model = RoutedPageSPPLSTM(RUNTIME_FEATURES, args.model_size).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != expected_parameter_count(args.model_size):
        raise RuntimeError("measured SPP v19 parameter count changed")
    history, train_coordinates = train_model(
        model, bundles["train"], targets["train"], train_key_by_context, device,
        args.epochs, args.chunk_len, args.accumulate_chunks, args.learning_rate,
        args.decoder_seed,
    )

    contexts, encoder_state = {}, None
    for role in roles:
        contexts[role], encoder_state = score_context(model, bundles[role], device, encoder_state)
    eval_positions = streams["eval"]["demand_positions"]
    base_lines = np.asarray([row[2] for row in streams["eval"]["demands"]], dtype=np.int64)
    predicted_counts, predicted_lines, predicted_fills, eval_coordinates = decode_actions(
        model, contexts["eval"][eval_positions], base_lines, event_keys["eval"],
        device, args.decoder_seed, "eval",
    )
    legality = action_legality_diagnostics(base_lines, predicted_counts, predicted_lines)
    behavior = complete_behavior_metrics(predicted_counts, predicted_lines, predicted_fills, teachers["eval"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    normal_path, nn_path = args.out_dir / "offline_spp.replay.csv", args.out_dir / "offline_nn.replay.csv"
    normal_entries, normal_triggers, normal_fill_counts = write_teacher_replay(normal_path, streams["eval"]["demands"], teachers["eval"])
    nn_entries, nn_triggers, nn_fill_counts = write_prediction_replay(nn_path, streams["eval"]["demands"], predicted_lines, predicted_fills)
    history_path, model_path = args.out_dir / "training_history.csv", args.out_dir / "model.pt"
    write_table(history_path, history)
    torch.save({
        "state_dict": model.state_dict(), "model_family": "lstm", "model_size": args.model_size,
        "runtime_features": RUNTIME_FEATURES, "fill_levels": FILL_LEVELS,
        "codec": "signed_zigzag_canonical_leb128", "decoder_seed": args.decoder_seed,
        "sampler_revision": SAMPLER_REVISION, "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": MODEL_REVISION, "decoder_revision": DECODER_REVISION,
    }, model_path)

    tag = model_tag("lstm", args.model_size)
    training_pages = len(set(bundles["train"]["pages"].tolist()))
    history_pages = len(set(np.concatenate([
        bundles[role]["pages"] for role in roles
    ]).tolist()))
    route_size = routed_state_size(args.model_size)
    codec_size = codec_embed_size(args.model_size)
    global_state_bytes = 4 * route_size * 4
    page_state_bytes = 2 * route_size * 4 + 8
    training_state_bytes = global_state_bytes + training_pages * page_state_bytes
    inference_state_bytes = global_state_bytes + history_pages * page_state_bytes
    persistent_bytes = max(training_state_bytes, inference_state_bytes)
    train_schedule = sampling_schedule_sha256(args.decoder_seed, TRACE, POLICY, "train", train_coordinates)
    eval_schedule = sampling_schedule_sha256(args.decoder_seed, TRACE, POLICY, "eval", eval_coordinates)
    metadata = {
        "run_id": RUN_ID, "trace": TRACE, "model_tag": tag,
        "matched_normal_prefetcher": POLICY,
        "neural_role": "standalone_direct_action_prefetcher", "model_family": "lstm",
        "track_model_family": "lstm", "model_size": args.model_size,
        "architecture_pair_id": args.pair_id, "parameter_count": parameter_count,
        "parameter_formula": PARAMETER_FORMULA,
        "configured_parameter_counts": {
            str(size): expected_parameter_count(size) for size in MODEL_POINTS["lstm"]
        },
        "model_point_contract": describe_model_points(),
        "route_hidden_size": route_size, "codec_embed_size": codec_size,
        "parameter_storage_bytes_float32": parameter_count * 4,
        "dynamic_page_state_pages": history_pages,
        "training_dynamic_page_state_pages": training_pages,
        "inference_history_dynamic_page_state_pages": history_pages,
        "recurrent_state_bytes_per_global_stream": 4 * route_size * 4,
        "recurrent_state_bytes_per_page": 2 * route_size * 4 + 8,
        "peak_recurrent_state_bytes": persistent_bytes,
        "peak_persistent_recurrent_state_bytes": persistent_bytes,
        "peak_training_persistent_recurrent_state_bytes": training_state_bytes,
        "peak_inference_persistent_recurrent_state_bytes": inference_state_bytes,
        "recurrent_state_dtype": "float32_plus_int64_page_last_position",
        "persistent_recurrent_state": "separate demand/fill LSTM state plus page-keyed causal LSTM state",
        "seed": args.seed, "decoder_seed": args.decoder_seed, "operation": OPERATION,
        "production_rng_seeded_after_self_tests": True,
        "experiment_revision": EXPERIMENT_REVISION, "model_revision": MODEL_REVISION,
        "decoder_revision": DECODER_REVISION, "weights_retrained": True,
        "checkpoint_reused": False, "decoder_only_change": False,
        "guard_selected_decoder": False, "joint_map_used": False,
        "selected_decoder_mode": "keyed_rank_stop_emit_plus_exact_zigzag_leb128_plus_target_conditioned_fill",
        "decoder_candidate_modes": [], "decoder_sampler": sampler_metadata(),
        "sampler_revision": SAMPLER_REVISION, "decoder_sampler_revision": SAMPLER_REVISION,
        "decoder_sampler_source_sha256": sampler_source_sha256(),
        "decoder_sampler_key_schedule_sha256": key_schedule_sha256(),
        "decoder_sampler_key_fields": list(KEY_FIELDS), "decoder_key_fields": list(KEY_FIELDS),
        "decoder_event_key_fields": ["decision_index", "constant_DEMAND_kind", "source_line"],
        "decoder_event_key_definition": "zero_based_role_decision_idx_plus_source_line",
        "decoder_event_key_uses_teacher_information": False,
        "decoder_key_includes_sampler_revision": True,
        "decoder_forbidden_key_fields": ["pc", "raw_teacher_event_id"],
        "decoder_action_rank_origin": 0,
        "decoder_eval_event_key_stream_sha256": key_stream_sha256(event_keys["eval"]),
        "decoder_eval_sampling_schedule_sha256": eval_schedule,
        "decoder_eval_sampling_coordinates": len(eval_coordinates),
        "decoder_train_sampling_schedule_sha256": train_schedule,
        "decoder_train_sampling_coordinates": len(train_coordinates),
        "decoder_guard_sampling_schedule_sha256": None,
        "decoder_guard_sampling_coordinates": 0,
        "common_random_numbers_across_capacities": True,
        "strict_common_random_numbers_across_capacities": True,
        "cross_event_rng_state_used": False, "train_guard_decoder_rng_burn_used": False,
        "decoder_sampling_roles": ["train", "eval"], "decoder_roles_sampled": ["train", "eval"],
        "decoder_train_sampling_performed": True, "decoder_guard_sampling_performed": False,
        "decoder_action_sampling_performed": True, "decoder_count_sampling_performed": True,
        "keyed_sampling_self_test": "PASS", "stochastic_decoding_reproducible": True,
        "stochastic_decoding": "stateless event/rank/head keyed categorical inverse-CDF for STOP/EMIT, bytes, and fill",
        "epochs": args.epochs, "chunk_len": args.chunk_len,
        "accumulate_chunks": args.accumulate_chunks, "learning_rate": args.learning_rate,
        "guard_rows": len(streams["guard"]["context"]), "eval_rows": len(streams["eval"]["context"]),
        "guard_demand_callbacks": len(streams["guard"]["demands"]),
        "eval_demand_callbacks": len(streams["eval"]["demands"]),
        "guard_cache_fill_callbacks": len(streams["guard"]["context"]) - len(streams["guard"]["demands"]),
        "eval_cache_fill_callbacks": len(streams["eval"]["context"]) - len(streams["eval"]["demands"]),
        "guard_role": "causal_input_history_warmup_and_audit_only",
        "runtime_feature_count": RUNTIME_FEATURES,
        "runtime_encoding": "lossless 58-bit cache-line number plus one DEMAND/FILL kind bit",
        "runtime_address_alignment_bits_removed": CACHE_LINE_SHIFT,
        "runtime_address_alignment_bits_were_constant_zero": True,
        "source_decision_effective_external_input": SOURCE_INPUTS,
        "same_external_input_contract": True,
        "training_inference_input_encoder_identical": True,
        "decoder_training_mode": "sampled_rank_grammar_rollout_with_separate_teacher_prefix_output_nll",
        "decoder_previous_teacher_action_used_as_input": True,
        "decoder_previous_teacher_action_used_as_input_scope": "isolated_loss_only_teacher_prefix_likelihood_branch",
        "decoder_previous_teacher_action_used_as_main_rollout_input": False,
        "decoder_free_running_self_test": "PASS",
        "teacher_count_role": "labels_STOP_or_EMIT_only_at_ranks_reached_by_sampled_rollout",
        "teacher_count_used_as_decoder_feedback": False,
        "teacher_prefix_role": "loss_only_exact_autoregressive_target_likelihood_branch",
        "teacher_prefix_advances_loss_only_likelihood_byte_state": True,
        "teacher_prefix_used_as_main_rollout_recurrent_feedback": False,
        "teacher_target_conditions_loss_only_fill_factor": True,
        "teacher_action_values_used_as_main_rollout_recurrent_feedback": False,
        "runtime_encoder_entrypoint": "623_offline_lstm_spp.routed_grammar_v19.runtime_bundle",
        "runtime_encoder_sha256": runtime_encoder_sha256(),
        "training_runtime_encoder_sha256": runtime_encoder_sha256(),
        "inference_runtime_encoder_sha256": runtime_encoder_sha256(),
        "training_runtime_fields": SOURCE_INPUTS, "inference_runtime_fields": SOURCE_INPUTS,
        "decision_router_source_sha256": decision_router_source_sha256(),
        "model_does_not_use_pc": True, "pc_is_replay_transport_only": True,
        "model_input_is_causal_external_event_sequence_only": True,
        "cache_fill_feedback_used_as_raw_external_input": True,
        "cache_fill_private_state_used_as_model_input": False,
        "cache_hit_and_type_are_audit_only": True,
        "teacher_actions_are_model_inputs": False,
        "teacher_actions_are_model_inputs_scope": "external_or_runtime_inference_inputs_only",
        "teacher_actions_used_as_supervised_output_conditioning": True,
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_request_rate_used_as_budget": False,
        "normal_policy_constants_used_by_neural_inference": False,
        "nn_generates_own_target_addresses_and_fill_levels": True,
        "complete_action_space": "rank STOP/EMIT plus exact signed 58-bit increments and learned fill",
        "decision_rule": "keyed learned STOP/EMIT; exact keyed ZigZag+canonical-LEB128 increment; fill after actual target",
        "probability_threshold_used": False, "threshold_related_hardcodes_used": False,
        "neural_degree_cap": None, "fixed_page_offset_classes": None,
        "same_page_rule_used_by_neural_inference": False, "future_label_window_used": False,
        "fill_lead_cutoff_used": False, "handcrafted_semantic_features_used": True,
        "causal_derived_features": ["raw_line_page_key", "log1p_page_reuse_age"],
        "causal_derived_features_use_external_input_only": True,
        "external_input_fields_unchanged": True,
        "manual_loss_weights_used": False, "gate_class_weighting_used": False,
        "gate_training_objective": None, "gate_decoding_rule": None,
        "request_count_training_objective": "rankwise_unweighted_stop_emit_categorical_nll",
        "request_count_decoding_rule": "first_keyed_learned_STOP_token_ends_action_sequence",
        "request_count_residual_scope": None, "request_count_sampling_performed": True,
        "stop_emit_sampling_rule": "event_rank_keyed_categorical_inverse_cdf",
        "stop_emit_sampler_representability_check": "STOP_mass_strictly_above_open_uniform_half_bin",
        "action_rollout_fail_closed_watchdog_ranks": ACTION_ROLLOUT_WATCHDOG_RANKS,
        "action_rollout_watchdog_role": "error_without_replay_not_truncation_or_forced_STOP",
        "action_rollout_watchdog_is_neural_degree_cap": False,
        "joint_delta_fill_dependency_modeled": True, "joint_pair_classes": 0,
        "joint_delta_fill_training_objective": "target_nll_then_fill_ce_conditioned_on_same_teacher_target_in_loss_only_branch",
        "joint_delta_fill_decoding_rule": "delta_then_fill_after_actual_target",
        "delta_mixture_components": 0, "decoder_mixture_components": 0,
        "delta_training_objective": "exact_autoregressive_teacher_prefix_canonical_leb128_nll_with_sampled_history_duplicate_support",
        "delta_mixture_decoding_rule": None,
        "delta_decoding_rule": "keyed_exact_signed_zigzag_canonical_leb128",
        "delta_codec": "signed_zigzag_canonical_leb128_max_{}_bytes_complete_{}_bit_support".format(
            LEB128_MAX_BYTES, LINE_ADDRESS_BITS,
        ),
        "delta_zero_allowed": True, "self_target_actions_allowed": True,
        "delta_legality_constraints": ["distinct_target_within_callback"],
        "delta_legality_fallback": None,
        "duplicate_target_handling": "mask_categorical_probability_and_renormalize",
        "duplicate_prefix_feasibility_mask_used": True,
        "fill_training_objective": "unweighted_two_class_cross_entropy_conditioned_on_teacher_target_loss_only",
        "fill_decoding_rule": "event_rank_keyed_categorical_inverse_cdf_after_actual_target",
        "fill_conditioned_on_actual_emitted_target": True, "fill_argmax_used": False,
        "fill_probability_feedback_used": False, "hard_fill_one_hot_feedback_used": True,
        "keyed_fill_uniform_dtype": "float64", "address_confidence_fill_heuristic_used": False,
        "decoder_probability_mass_carries_train_guard_history": False,
        "cross_event_probability_credit_used": False, "sampled_outputs_used_as_decoder_feedback": True,
        "delta_decoder_feedback_rule": "actual_hard_leb128_increment_and_target_embedding",
        "fill_decoder_feedback_rule": "actual_keyed_hard_fill_one_hot",
        "straight_through_hard_action_feedback_used": True,
        "loss_design": "sum of rank STOP/EMIT, exact byte, and target-conditioned fill NLL divided by all categorical atoms",
        "optimizer_gradient_normalization": "total_categorical_atom_count_per_accumulation_group",
        "training_regularization_used": False, "inference_policy_hardcodes_used": False,
        "learned_request_count": True, "address_interface_bits": ADDRESS_BITS,
        "factorized_delta_fill_heads": False,
        "routed_demand_fill_recurrent_paths": True, "page_local_causal_state": True,
        "page_key_source": "raw_line_address_shifted_by_derived_log2_4096B_page_lines",
        "page_state_validity_rule": "learned_vector_gate_conditioned_on_page_state_and_log_causal_reuse_age",
        "eviction_feedback_role": "raw chronological input event routed through the fill LSTM",
        "training_labels": "canonicalized source-SPP actions and fill; supervision only",
        "teacher_action_files_role": "normal replay, supervised labels, and audit only",
        "forbidden_inputs": ["normal_actions_at_inference", "SPP_signature_tables", "pattern_tables", "normal_thresholds", "normal_degree", "global_history_register_contents", "prefetch_filter_contents", "cycle", "cache_hit", "access_type", "queue_state", "future_rows"],
        "training_chunks_shuffled": False, "training_state_mode": "chronological_stateful_tbptt",
        "training_state_carried_across_chunks": True, "training_state_detached_between_chunks": True,
        "inference_history_mode": "fresh_state_then_complete_train_guard_eval_chronology",
        "causal_no_future_self_test": "PASS", "rank_stop_emit_grammar_self_test": "PASS",
        "exact_leb128_codec_self_test": "PASS", "duplicate_probability_mask_self_test": "PASS",
        "duplicate_prefix_no_dead_end_self_test": "PASS",
        "teacher_prefix_state_isolation_self_test": "PASS",
        "stop_sampler_representability_self_test": "PASS",
        "categorical_nonfinite_rejection_self_test": "PASS",
        "always_emit_watchdog_self_test": "PASS",
        "integer_csv_exactness_self_test": "PASS",
        "target_conditioned_fill_self_test": "PASS", "routed_page_state_self_test": "PASS",
        "cnn_architecture_self_test": "NOT_APPLICABLE", "cnn_temporal_layers": 0,
        "event_logger_schema": EVENT_LOGGER_SCHEMA, "action_attachment_mode": ACTION_ATTACHMENT_MODE,
        "canonicalization_mode": CANONICALIZATION_MODE,
        "teacher_action_canonicalization": CANONICALIZATION_MODE,
        "replay_preserves_explicit_fill_level": True,
        "same_source_input_offline_claim_allowed": True, "closed_loop_live_claim_allowed": False,
        "offline_input_feedback_origin": "recorded cache-fill callbacks produced by the source SPP run",
        "comparison_claim_boundary": "matched-input open-loop offline comparison only; live NN actions did not regenerate cache-fill feedback",
        "collection_manifest_role": "historical_input_package_provenance_only",
        "collection_manifest_decoder_fields_are_current_contract": False,
        "source_contract_sha256": sha256(args.source_contract),
        "offline_normal_entries": normal_entries, "offline_normal_triggers": normal_triggers,
        "offline_normal_fill_counts": normal_fill_counts,
        "offline_normal_fill_level_counts": normal_fill_counts,
        "offline_nn_entries": nn_entries, "offline_nn_triggers": nn_triggers,
        "offline_nn_fill_counts": nn_fill_counts, "offline_nn_fill_level_counts": nn_fill_counts,
        "action_legality_diagnostics": legality,
        "raw_predicted_action_count": legality["raw_predicted_action_count"],
        "materialized_distinct_action_count": legality["materialized_distinct_action_count"],
        "normal_list_sha256": sha256(normal_path), "nn_list_sha256": sha256(nn_path),
        "heldout_behavior_metrics": behavior, "train_history": history,
        "source_contract": source_contract, "model_checkpoint_sha256": sha256(model_path),
        "training_history_sha256": sha256(history_path), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__,
    }
    for role in roles:
        metadata[role + "_decision_router_sha256"] = decision_router_sha256(streams[role])
        metadata[role + "_decoder_event_key_stream_sha256"] = key_stream_sha256(event_keys[role])
        metadata[role + "_stream_gzip_sha256"] = sha256(stream_paths[role])
        metadata[role + "_stream_content_sha256"] = gzip_content_sha256(stream_paths[role])
        metadata[role + "_teacher_actions_gzip_sha256"] = sha256(action_paths[role])
        metadata[role + "_teacher_actions_content_sha256"] = gzip_content_sha256(action_paths[role])
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "PASS", "model_tag": tag, "parameters": parameter_count,
        "decision_rule": metadata["decision_rule"], "offline_normal_entries": normal_entries,
        "offline_nn_entries": nn_entries, "offline_nn_fill_level_counts": nn_fill_counts,
    }, indent=2))


if __name__ == "__main__":
    main()
