#!/usr/bin/env python3
"""Train and decode the matched-input 623 Stride v25 model.

Runtime input is exactly raw pc64 + line58.  The fixed model family splits the
total recurrent width equally between a global chronological LSTM and an
exact-PC-local LSTM, then learns their fusion.  The decoder uses an unweighted
ZERO/POSITIVE hurdle, a positive-only categorical count, and rank-conditioned
direct 58-bit modular deltas.  Every real rank supervises the same bit head;
there is no delta-token vocabulary or escape path.
"""
import argparse
import csv
import gzip
import hashlib
import heapq
import inspect
import json
import math
import os
import platform
import random
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import model_contract as model_contract_module
from model_contract import (
    ADDRESS_BITS, BLOCKED_VALIDATION_LENGTH_SOURCE, CACHE_LINE_BYTES,
    CAUSAL_RUNTIME_FEATURES, CHECKPOINT_SELECTION, DECODER_REVISION,
    DECODER_TRAINING_MODE, DECODING_RULE, DELTA_OBJECTIVE,
    EXPERIMENT_REVISION, FIT_DENOMINATOR, FIT_NUMERATOR, FULL_OBJECTIVE,
    HURDLE_OBJECTIVE, LINE_NUMBER_BITS, MODEL_POINTS,
    MODEL_REVISION, OPERATION, ORIGINAL_GUARD_ROLE, PARENT_INPUT_RUN_ID,
    POLICY, POSITIVE_COUNT_OBJECTIVE, RANK_CODE_FEATURES,
    RAW_RUNTIME_FEATURES, RUN_ID,
    RUNTIME_FEATURES, SOURCE_INPUTS, TRACE,
    TRAINING_ACCUMULATE_CHUNKS, TRAINING_CHUNK_LEN, TRAINING_EPOCHS,
    TRAINING_LEARNING_RATE, TRAINING_SEED, positive_count_statistics,
    expected_parameter_count, model_points_description, model_tag,
    parse_exact_integer,
)

if __name__ == "__main__" and sys.argv[1:] == ["--describe-model-points"]:
    print(json.dumps(model_points_description(), indent=2, sort_keys=True))
    raise SystemExit(0)
if __name__ == "__main__" and sys.argv[1:] == ["--self-test"]:
    model_contract_module.self_test_contract()
    print("PASS")
    raise SystemExit(0)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
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
    behavior_metrics,
)

EVENT_LOGGER_SCHEMA = "623_causal_trigger_v5"
CANDIDATE_ATTACHMENT_MODE = "explicit_trigger_event_id"
SOURCE_INPUT_LIST = list(SOURCE_INPUTS)
LINE_MODULUS = 1 << LINE_NUMBER_BITS
LINE_MASK = LINE_MODULUS - 1

if (COMMON_ADDRESS_BITS, COMMON_CACHE_LINE_BYTES) != (
    ADDRESS_BITS, CACHE_LINE_BYTES
):
    raise RuntimeError("shared address contract differs from v25 contract")


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
                row["trace"] != TRACE
                or row["policy"] != POLICY
                or (
                    as_int(row["pc"]), as_int(row["line"]),
                    as_int(row["pc_line_occ"]),
                ) != (pc, line, occurrence)
                or row["logger_schema"] != EVENT_LOGGER_SCHEMA
                or row["match_mode"] != CANDIDATE_ATTACHMENT_MODE
                or as_int(row["candidate_rank"]) != len(actions[index]) + 1
            ):
                raise RuntimeError(
                    "teacher action identity/rank failure at {}".format(index)
                )
            trigger = as_int(row["trigger_event_id"])
            pf_event = as_int(row["pf_event_id"])
            target = as_int(row["pf_line"])
            if (
                pf_event <= last_pf_event
                or trigger >= pf_event
                or as_int(row["event_distance"]) != pf_event - trigger
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
    integers = [int(value) for value in values]
    limit = 1 << int(width)
    if any(value < 0 or value >= limit for value in integers):
        raise RuntimeError("runtime integer exceeds its lossless bit width")
    array = np.asarray(integers, dtype=np.uint64)
    shifts = np.arange(width, dtype=np.uint64)
    return ((array[:, None] >> shifts[None, :]) & 1).astype(np.uint8)


def runtime_features(rows):
    encoded = np.concatenate([
        _unsigned_bits([pc for pc, _, _ in rows], ADDRESS_BITS),
        _unsigned_bits([line for _, line, _ in rows], LINE_NUMBER_BITS),
    ], axis=1)
    if encoded.shape != (len(rows), RUNTIME_FEATURES):
        raise RuntimeError("raw runtime feature width changed")
    return encoded


def runtime_encoder_sha256():
    payload = {
        "entrypoint": inspect.getsource(runtime_features),
        "bits": inspect.getsource(_unsigned_bits),
        "external_fields": SOURCE_INPUT_LIST,
        "feature_count": RUNTIME_FEATURES,
        "runtime_encoding": "lossless_raw_pc64_plus_line58_only",
        "engineered_features": [],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def _pc_groups(pcs):
    grouped = OrderedDict()
    for position, pc in enumerate(pcs):
        grouped.setdefault(int(pc), []).append(position)
    return sorted(grouped.items(), key=lambda item: (-len(item[1]), item[1][0]))


def _initial_local_state(state_map, keys, hidden_size, device):
    hidden, cell = [], []
    for key in keys:
        if key in state_map:
            h_value, c_value = state_map[key]
        else:
            h_value = torch.zeros(hidden_size, device=device)
            c_value = torch.zeros(hidden_size, device=device)
        hidden.append(h_value)
        cell.append(c_value)
    return torch.stack(hidden).unsqueeze(0), torch.stack(cell).unsqueeze(0)


def new_recurrent_state():
    return {"global": None, "local": {}}


def _encode_chunk(model, features, pcs, recurrent_state):
    if set(recurrent_state) != {"global", "local"}:
        raise RuntimeError("invalid dual-context recurrent state")
    branch = model.branch_hidden_size

    global_projected = torch.tanh(model.global_input_projection(
        features.unsqueeze(0)
    ))
    global_initial = recurrent_state["global"]
    if global_initial is None:
        global_initial = (
            torch.zeros(1, 1, branch, device=features.device),
            torch.zeros(1, 1, branch, device=features.device),
        )
    global_output, global_final = model.global_lstm(
        global_projected, global_initial
    )
    recurrent_state["global"] = (
        global_final[0].detach(), global_final[1].detach()
    )

    groups = _pc_groups(pcs)
    lengths = [len(indices) for _, indices in groups]
    padded = torch.zeros(
        len(groups), max(lengths), RUNTIME_FEATURES,
        dtype=features.dtype, device=features.device,
    )
    for row, (_, indices) in enumerate(groups):
        positions = torch.as_tensor(
            indices, dtype=torch.long, device=features.device
        )
        padded[row, :len(indices)] = features.index_select(0, positions)
    projected = torch.tanh(model.local_input_projection(padded))
    packed = pack_padded_sequence(
        projected, lengths, batch_first=True, enforce_sorted=True
    )
    initial = _initial_local_state(
        recurrent_state["local"], [pc for pc, _ in groups],
        branch, features.device,
    )
    packed_output, final = model.local_lstm(packed, initial)
    padded_output, _ = pad_packed_sequence(
        packed_output, batch_first=True, total_length=max(lengths)
    )
    local_context = torch.zeros(
        len(pcs), branch, dtype=features.dtype, device=features.device
    )
    for row, (pc, indices) in enumerate(groups):
        positions = torch.as_tensor(
            indices, dtype=torch.long, device=features.device
        )
        local_context = local_context.index_copy(
            0, positions, padded_output[row, :len(indices)]
        )
        recurrent_state["local"][pc] = (
            final[0][0, row].detach(), final[1][0, row].detach()
        )
    joined = torch.cat((global_output.squeeze(0), local_context), dim=1)
    if joined.shape != (len(pcs), model.hidden_size):
        raise RuntimeError("dual-context width accounting changed")
    return torch.tanh(model.fusion(joined))


def state_router_sha256():
    payload = (
        inspect.getsource(_pc_groups)
        + inspect.getsource(_initial_local_state)
        + inspect.getsource(new_recurrent_state)
        + inspect.getsource(_encode_chunk)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _rank_code(rank, count, device, dtype):
    positions = torch.full(
        (int(count), 1), float(int(rank) + 1), device=device, dtype=dtype
    )
    scales = torch.pow(
        torch.tensor(10000.0, device=device, dtype=dtype),
        torch.arange(
            0, RANK_CODE_FEATURES, 2, device=device, dtype=dtype
        ) / float(RANK_CODE_FEATURES),
    ).reshape(1, -1)
    angles = positions / scales
    return torch.stack((torch.sin(angles), torch.cos(angles)), dim=2).reshape(
        int(count), RANK_CODE_FEATURES
    )


def _rank_context(model, context, rank):
    code = _rank_code(rank, len(context), context.device, context.dtype)
    return torch.tanh(context + model.rank_projection(code))


def _modular_line_delta(target, base):
    return (int(target) - int(base)) & LINE_MASK


def _modular_delta_bits(values):
    return _unsigned_bits(values, LINE_NUMBER_BITS)


def _bits_to_modular_delta(logits):
    if len(logits) != LINE_NUMBER_BITS:
        raise RuntimeError("delta payload width is not 58 bits")
    value = 0
    for bit, logit in enumerate(logits):
        if float(logit) >= 0.0:
            value |= 1 << bit
    return value


def hurdle_prior(actions):
    positives = sum(bool(items) for items in actions)
    zeros = len(actions) - positives
    denominator = float(len(actions) + 2)
    return [(zeros + 1.0) / denominator, (positives + 1.0) / denominator]


def delta_bit_prior(rows, actions):
    modular = []
    for (_, base, _), targets in zip(rows, actions):
        modular.extend(_modular_line_delta(target, base) for target in targets)
    if not modular:
        raise RuntimeError("delta bit prior requires real teacher actions")
    bits = _modular_delta_bits(modular)
    ones = bits.sum(axis=0).astype(np.int64)
    denominator = float(len(modular) + 2)
    return [float((int(value) + 1.0) / denominator) for value in ones]


def realized_label_design(rows, actions):
    count_stats = positive_count_statistics([len(items) for items in actions])
    return {
        "hurdle_prior": hurdle_prior(actions),
        "positive_count_statistics": count_stats,
        "positive_count_support": count_stats["positive_count_support"],
        "positive_count_prior": count_stats[
            "add_one_smoothed_positive_priors"
        ],
        "delta_bit_prior": delta_bit_prior(rows, actions),
    }


def instantiate_model(hidden_size, design):
    return DualContextHurdleStrideLSTM(
        hidden_size,
        design["hurdle_prior"],
        design["positive_count_prior"],
        design["delta_bit_prior"],
    )


def delta_label_statistics(rows, actions):
    values = []
    for (_, base, _), targets in zip(rows, actions):
        values.extend(_modular_line_delta(target, base) for target in targets)
    return {
        "teacher_actions": len(values),
        "unique_modular_teacher_deltas": len(set(values)),
        "all_teacher_actions_supervise_58_bits": True,
        "supervised_bit_atoms": len(values) * LINE_NUMBER_BITS,
    }


class DualContextHurdleStrideLSTM(nn.Module):
    def __init__(
        self, hidden_size, hurdle_prior, positive_count_prior, bit_prior,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.branch_hidden_size = self.hidden_size // 2
        self.positive_count_output_classes = len(positive_count_prior)
        if (
            self.hidden_size not in MODEL_POINTS["lstm"]
            or self.hidden_size % 2
            or self.positive_count_output_classes < 1
        ):
            raise ValueError("unsupported realized Stride v25 dimensions")
        branch = self.branch_hidden_size
        self.global_input_projection = nn.Linear(RUNTIME_FEATURES, branch)
        self.local_input_projection = nn.Linear(RUNTIME_FEATURES, branch)
        self.global_lstm = nn.LSTM(branch, branch, batch_first=True)
        self.local_lstm = nn.LSTM(branch, branch, batch_first=True)
        self.fusion = nn.Linear(hidden_size, hidden_size)
        self.rank_projection = nn.Linear(RANK_CODE_FEATURES, hidden_size)
        self.hurdle_head = nn.Linear(hidden_size, 2)
        self.positive_count_head = nn.Linear(
            hidden_size, self.positive_count_output_classes
        )
        self.delta_bit_head = nn.Linear(hidden_size, LINE_NUMBER_BITS)

        hurdle_tensor = torch.as_tensor(
            hurdle_prior, dtype=self.hurdle_head.bias.dtype
        )
        count_tensor = torch.as_tensor(
            positive_count_prior, dtype=self.positive_count_head.bias.dtype
        )
        bit_tensor = torch.as_tensor(
            bit_prior, dtype=self.delta_bit_head.bias.dtype
        )
        if (
            bool((hurdle_tensor <= 0).any())
            or bool((count_tensor <= 0).any())
            or bit_tensor.shape != (LINE_NUMBER_BITS,)
            or bool((bit_tensor <= 0).any())
            or bool((bit_tensor >= 1).any())
            or not bool(torch.isfinite(hurdle_tensor).all())
            or not bool(torch.isfinite(count_tensor).all())
            or not bool(torch.isfinite(bit_tensor).all())
        ):
            raise ValueError("TRAIN-derived initialization is invalid")
        with torch.no_grad():
            self.hurdle_head.weight.zero_()
            self.hurdle_head.bias.copy_(torch.log(hurdle_tensor))
            self.positive_count_head.weight.zero_()
            self.positive_count_head.bias.copy_(torch.log(count_tensor))
            self.delta_bit_head.weight.zero_()
            self.delta_bit_head.bias.copy_(
                torch.log(bit_tensor) - torch.log1p(-bit_tensor)
            )


def _chunk_objective(
    model, context, base_lines, actions, positive_count_support,
):
    counts = np.asarray([len(items) for items in actions], dtype=np.int64)
    if (
        len(counts) != len(context)
        or len(counts) == 0
        or model.positive_count_output_classes != len(positive_count_support)
    ):
        raise RuntimeError("v25 chunk labels are outside realized support")
    hurdle_targets = torch.from_numpy((counts > 0).astype(np.int64)).to(
        device=context.device, dtype=torch.long
    )
    hurdle_sum = F.cross_entropy(
        model.hurdle_head(context), hurdle_targets, reduction="sum"
    )
    positive_np = np.flatnonzero(counts > 0).astype(np.int64)
    count_sum = context.sum() * 0.0
    count_index = {
        int(value): index for index, value in enumerate(positive_count_support)
    }
    if len(positive_np):
        unsupported = sorted({
            int(counts[row]) for row in positive_np
            if int(counts[row]) not in count_index
        })
        if unsupported:
            raise RuntimeError(
                "positive count labels outside partition-derived support: {}"
                .format(unsupported)
            )
        positive = torch.from_numpy(positive_np).to(
            device=context.device, dtype=torch.long
        )
        count_targets = torch.as_tensor(
            [count_index[int(counts[row])] for row in positive_np],
            device=context.device, dtype=torch.long,
        )
        count_sum = F.cross_entropy(
            model.positive_count_head(context.index_select(0, positive)),
            count_targets, reduction="sum",
        )
    bit_sum = context.sum() * 0.0
    action_atoms = 0
    base_array = np.asarray(base_lines, dtype=np.uint64)
    for rank in range(int(counts.max()) if len(counts) else 0):
        active_np = np.flatnonzero(counts > rank).astype(np.int64)
        if not len(active_np):
            continue
        active = torch.from_numpy(active_np).to(
            device=context.device, dtype=torch.long
        )
        ranked = _rank_context(
            model, context.index_select(0, active), rank
        )
        predictions = model.delta_bit_head(ranked)
        modular_values = [
            _modular_line_delta(actions[row][rank], int(base_array[row]))
            for row in active_np
        ]
        truth = torch.as_tensor(
            _modular_delta_bits(modular_values),
            dtype=context.dtype, device=context.device,
        )
        bit_sum = bit_sum + F.binary_cross_entropy_with_logits(
            predictions, truth, reduction="sum"
        )
        action_atoms += len(active_np)

    decision_atoms = len(counts)
    complete_sum = hurdle_sum + count_sum + bit_sum
    objective = complete_sum / float(decision_atoms)
    return objective, {
        "hurdle_nll_sum": float(hurdle_sum.detach()),
        "positive_count_nll_sum": float(count_sum.detach()),
        "delta_bit_nll_sum": float(bit_sum.detach()),
        "decision_atoms": decision_atoms,
        "positive_callback_atoms": len(positive_np),
        "action_atoms": action_atoms,
        "delta_bit_atoms": action_atoms * LINE_NUMBER_BITS,
        "complete_nll_per_callback": float(objective.detach()),
        "objective_chunks": 1,
    }


def score_suffix(model, rows, runtime, device, chunk_len, output_start):
    if not 0 <= output_start <= len(rows):
        raise RuntimeError("invalid scored suffix")
    output = np.empty(
        (len(rows) - output_start, model.hidden_size), dtype=np.float32
    )
    pcs = np.asarray([pc for pc, _, _ in rows], dtype=np.uint64)
    recurrent_state = new_recurrent_state()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), chunk_len):
            stop = min(start + chunk_len, len(rows))
            features = torch.from_numpy(runtime[start:stop]).to(
                device=device, dtype=torch.float32
            )
            context = _encode_chunk(
                model, features, pcs[start:stop], recurrent_state
            )
            copy_start = max(start, output_start)
            if copy_start < stop:
                output[copy_start - output_start:stop - output_start] = (
                    context[copy_start - start:].cpu().numpy()
                )
    return output, {
        "rows": len(rows),
        "global_state_count": 1,
        "unique_pc_local_states": len(recurrent_state["local"]),
    }


def validation_nll(
    model, context_numpy, rows, actions, positive_count_support,
    device, chunk_len=4096,
):
    if not (len(context_numpy) == len(rows) == len(actions)):
        raise RuntimeError("blocked-validation lengths differ")
    totals = Counter()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), chunk_len):
            stop = min(start + chunk_len, len(rows))
            context = torch.from_numpy(context_numpy[start:stop]).to(device)
            _, components = _chunk_objective(
                model, context,
                [line for _, line, _ in rows[start:stop]],
                actions[start:stop], positive_count_support,
            )
            totals.update(components)
    decisions = max(1, totals["decision_atoms"])
    complete = (
        totals["hurdle_nll_sum"]
        + totals["positive_count_nll_sum"]
        + totals["delta_bit_nll_sum"]
    ) / decisions
    return {
        "complete_nll_per_callback": float(complete),
        "hurdle_nll_per_callback": (
            totals["hurdle_nll_sum"] / decisions
        ),
        "positive_count_nll_per_callback": (
            totals["positive_count_nll_sum"] / decisions
        ),
        "delta_bit_nll_per_callback": (
            totals["delta_bit_nll_sum"] / decisions
        ),
        "decision_atoms": int(totals["decision_atoms"]),
        "positive_callback_atoms": int(totals["positive_callback_atoms"]),
        "action_atoms": int(totals["action_atoms"]),
        "delta_bit_atoms": int(totals["delta_bit_atoms"]),
    }


def _highest_scoring_feasible_payload(base, bit_logits, forbidden):
    """Choose the best feasible Bernoulli payload without changing a target.

    Starting from the modal payload, flipping bit i costs abs(logit_i) in log
    probability.  A min-heap enumerates assignments by exact subset-sum cost.
    At most len(forbidden)+1 assignments are needed: payload-to-target mapping
    is bijective, so one of that many distinct assignments must be feasible.
    """
    values = [float(value) for value in bit_logits]
    if len(values) != LINE_NUMBER_BITS or not all(map(math.isfinite, values)):
        raise RuntimeError("delta bit logits are invalid")
    modal = _bits_to_modular_delta(values)
    modal_log_probability = 0.0
    for value in values:
        if value >= 0.0:
            modal_log_probability -= math.log1p(math.exp(-value))
        else:
            modal_log_probability -= math.log1p(math.exp(value))
    ordered_flips = sorted(
        (abs(value), bit) for bit, value in enumerate(values)
    )
    # Heap item: cumulative penalty, modular delta (deterministic tie-break),
    # next sorted flip position, and raw flip mask.
    heap = [(0.0, modal, 0, 0)]
    popped = 0
    limit = len(forbidden) + 1
    while heap and popped < limit:
        penalty, modular_delta, next_position, flip_mask = heapq.heappop(heap)
        popped += 1
        target = (int(base) + int(modular_delta)) & LINE_MASK
        if target not in forbidden:
            return (
                target,
                modular_delta != modal,
                modal_log_probability - float(penalty),
            )
        for position in range(next_position, len(ordered_flips)):
            weight, bit = ordered_flips[position]
            new_flip_mask = flip_mask | (1 << bit)
            new_delta = modal ^ new_flip_mask
            heapq.heappush(heap, (
                penalty + weight, new_delta, position + 1, new_flip_mask,
            ))
    raise RuntimeError("exact Bernoulli feasibility enumeration failed")


def decode(
    model, context_numpy, base_lines, positive_count_support,
    device, count_override=None, role="eval", chunk_len=4096,
):
    if len(context_numpy) != len(base_lines):
        raise RuntimeError("decoder row counts differ")
    if (
        model.positive_count_output_classes != len(positive_count_support)
        or not positive_count_support
        or any(int(value) <= 0 for value in positive_count_support)
    ):
        raise RuntimeError("decoder support differs from model")
    hurdle_parts, count_parts = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(base_lines), chunk_len):
            stop = min(start + chunk_len, len(base_lines))
            context = torch.from_numpy(context_numpy[start:stop]).to(device)
            hurdle_logits = model.hurdle_head(context)
            count_logits = model.positive_count_head(context)
            if (
                not bool(torch.isfinite(hurdle_logits).all())
                or not bool(torch.isfinite(count_logits).all())
            ):
                raise RuntimeError("hurdle/count logits are non-finite")
            hurdle_parts.append(hurdle_logits.cpu())
            count_parts.append(count_logits.cpu())
    hurdle_logits = torch.cat(hurdle_parts, dim=0)
    count_logits = torch.cat(count_parts, dim=0)
    hurdle_probabilities = torch.softmax(hurdle_logits.to(torch.float64), dim=1)
    hurdle_entropy = -(
        hurdle_probabilities
        * torch.log(hurdle_probabilities.clamp_min(1e-300))
    ).sum(dim=1)
    count_probabilities = torch.softmax(count_logits.to(torch.float64), dim=1)
    count_entropy = -(
        count_probabilities * torch.log(count_probabilities.clamp_min(1e-300))
    ).sum(dim=1)
    hurdle_choice = hurdle_logits.argmax(dim=1).numpy().astype(np.int64)
    count_choice = count_logits.argmax(dim=1).numpy().astype(np.int64)
    support_array = np.asarray(positive_count_support, dtype=np.int64)
    natural_counts = np.where(
        hurdle_choice == 0, 0, support_array[count_choice]
    ).astype(np.int64)
    if count_override is None:
        counts = natural_counts
    else:
        counts = np.asarray(count_override, dtype=np.int64)
        if len(counts) != len(base_lines) or bool((counts < 0).any()):
            raise RuntimeError("oracle count override is invalid")

    predicted_lines = [[] for _ in base_lines]
    predicted_fills = [[] for _ in base_lines]
    bit_entropy_sum = 0.0
    bit_atoms = alternate_payloads = 0
    with torch.no_grad():
        for start in range(0, len(base_lines), chunk_len):
            stop = min(start + chunk_len, len(base_lines))
            context = torch.from_numpy(context_numpy[start:stop]).to(device)
            local_counts = counts[start:stop]
            for rank in range(int(local_counts.max()) if len(local_counts) else 0):
                active_np = np.flatnonzero(local_counts > rank).astype(np.int64)
                if not len(active_np):
                    continue
                active = torch.from_numpy(active_np).to(
                    device=device, dtype=torch.long
                )
                ranked = _rank_context(
                    model, context.index_select(0, active), rank
                )
                bit_logits = model.delta_bit_head(ranked)
                if not bool(torch.isfinite(bit_logits).all()):
                    raise RuntimeError("rank action output is non-finite")
                probabilities = torch.sigmoid(bit_logits.to(torch.float64))
                bit_entropy_sum += float((-(
                    probabilities * torch.log(probabilities.clamp_min(1e-300))
                    + (1.0 - probabilities) * torch.log(
                        (1.0 - probabilities).clamp_min(1e-300)
                    )
                )).sum().item())
                bit_atoms += len(active_np) * LINE_NUMBER_BITS
                payloads = bit_logits.cpu().tolist()
                for local, payload in zip(active_np, payloads):
                    row = start + int(local)
                    used = set(predicted_lines[row])
                    selected_target, alternate, _ = (
                        _highest_scoring_feasible_payload(
                            int(base_lines[row]), payload, used
                        )
                    )
                    alternate_payloads += int(alternate)
                    predicted_lines[row].append(selected_target)
                    predicted_fills[row].append(-1)
    if any(
        len(items) != int(count) or len(items) != len(set(items))
        for items, count in zip(predicted_lines, counts)
    ):
        raise RuntimeError("ordered decoder did not realize K unique targets")
    count_width = model.positive_count_output_classes
    diagnostics = {
        "role": role,
        "count_override_used": count_override is not None,
        "hurdle_classes": ["ZERO", "POSITIVE"],
        "positive_count_output_classes": count_width,
        "positive_count_support": [int(value) for value in positive_count_support],
        "decoded_count_distribution": {
            str(key): int(value)
            for key, value in sorted(Counter(counts.tolist()).items())
        },
        "decoded_positive_callbacks": int((counts > 0).sum()),
        "decoded_total_actions": int(counts.sum()),
        "decoded_max_actions_per_callback": (
            int(counts.max()) if len(counts) else 0
        ),
        "mean_hurdle_entropy": float(hurdle_entropy.mean().item()),
        "mean_hurdle_entropy_normalized": (
            float(hurdle_entropy.mean().item()) / math.log(2.0)
        ),
        "mean_positive_count_entropy": float(count_entropy.mean().item()),
        "mean_positive_count_entropy_normalized": (
            float(count_entropy.mean().item()) / math.log(count_width)
            if count_width > 1 else 0.0
        ),
        "mean_delta_bit_entropy": (
            bit_entropy_sum / bit_atoms if bit_atoms else None
        ),
        "mean_delta_bit_entropy_normalized": (
            bit_entropy_sum / bit_atoms / math.log(2.0)
            if bit_atoms else None
        ),
        "alternate_feasible_payloads_selected": alternate_payloads,
        "all_emitted_target_lines_unique_within_callback": True,
        "deterministic_target_uniqueness_feasibility_mask_used": True,
        "decoded_target_projection_or_mutation_used": False,
        "uniqueness_constraint_used_as_neural_input": False,
        "probability_threshold_used": False,
        "class_reweighting_used": False,
        "decode_prior_correction_used": False,
        "action_feedback_used": False,
        "normal_request_budget_used": False,
    }
    return counts, predicted_lines, predicted_fills, diagnostics


def trigger_metrics(predicted_counts, target_actions):
    predicted = np.asarray(predicted_counts) > 0
    target = np.asarray([bool(items) for items in target_actions])
    true_positive = int(np.logical_and(predicted, target).sum())
    precision = (
        true_positive / float(predicted.sum()) if predicted.sum() else 0.0
    )
    recall = true_positive / float(target.sum()) if target.sum() else 0.0
    return {
        "predicted_positive_callbacks": int(predicted.sum()),
        "normal_positive_callbacks": int(target.sum()),
        "true_positive_trigger_callbacks": true_positive,
        "trigger_precision": precision,
        "trigger_recall": recall,
        "trigger_f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        ),
    }


def complete_metrics(counts, lines, fills, teacher):
    result = behavior_metrics(counts, lines, fills, teacher)
    result.update(trigger_metrics(counts, teacher))
    teacher_counts = [len(items) for items in teacher]
    confusion = Counter(
        (int(truth), int(prediction))
        for truth, prediction in zip(teacher_counts, counts)
    )
    result["count_confusion"] = {
        "{}->{}".format(truth, prediction): int(value)
        for (truth, prediction), value in sorted(confusion.items())
    }
    result["count_mae"] = (
        float(np.abs(
            np.asarray(counts, dtype=np.int64)
            - np.asarray(teacher_counts, dtype=np.int64)
        ).mean()) if teacher_counts else 0.0
    )
    normal = result["normal_actions"]
    result["request_ratio_vs_teacher"] = (
        result["predicted_actions"] / float(normal) if normal else 0.0
    )
    return result


def count_oracle_upper_bound(predicted_counts, teacher_actions):
    predicted = np.asarray(predicted_counts, dtype=np.int64)
    teacher = np.asarray([len(items) for items in teacher_actions], dtype=np.int64)
    true_positive = int(np.minimum(predicted, teacher).sum())
    predicted_total = int(predicted.sum())
    teacher_total = int(teacher.sum())
    precision = true_positive / float(predicted_total) if predicted_total else 0.0
    recall = true_positive / float(teacher_total) if teacher_total else 0.0
    return {
        "diagnostic_only": True,
        "replayed": False,
        "true_positive_actions_with_oracle_targets": true_positive,
        "target_precision_upper_bound": precision,
        "target_recall_upper_bound": recall,
        "target_f1_upper_bound": (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        ),
    }


def reset_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_one_epoch(
    model, optimizer, rows, actions, positive_count_support, device, args,
):
    runtime = runtime_features(rows)
    pcs = np.asarray([pc for pc, _, _ in rows], dtype=np.uint64)
    model.train()
    recurrent_state = new_recurrent_state()
    totals = Counter()
    optimizer.zero_grad(set_to_none=True)
    pending_chunks = pending_callbacks = optimizer_steps = 0
    for start in range(0, len(rows), args.chunk_len):
        stop = min(start + args.chunk_len, len(rows))
        features = torch.from_numpy(runtime[start:stop]).to(
            device=device, dtype=torch.float32
        )
        context = _encode_chunk(
            model, features, pcs[start:stop], recurrent_state
        )
        objective, components = _chunk_objective(
            model, context,
            [line for _, line, _ in rows[start:stop]],
            actions[start:stop], positive_count_support,
        )
        if not torch.isfinite(objective):
            raise RuntimeError("non-finite Stride v25 complete NLL")
        callbacks = int(components["decision_atoms"])
        (objective * float(callbacks)).backward()
        pending_chunks += 1
        pending_callbacks += callbacks
        totals.update(components)
        if pending_chunks == args.accumulate_chunks or stop == len(rows):
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(float(pending_callbacks))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending_chunks = pending_callbacks = 0
            optimizer_steps += 1
    return totals, optimizer_steps, len(recurrent_state["local"])


def _history_row(
    phase, epoch, totals, optimizer_steps, unique_pc_states,
    fit_callbacks, validation_callbacks, validation=None, selected=False,
):
    decisions = max(1, totals["decision_atoms"])
    complete = (
        totals["hurdle_nll_sum"]
        + totals["positive_count_nll_sum"]
        + totals["delta_bit_nll_sum"]
    ) / decisions
    validation = validation or {}
    return {
        "phase": phase,
        "epoch": epoch,
        "train_complete_nll_per_callback": complete,
        "train_hurdle_nll_per_callback": (
            totals["hurdle_nll_sum"] / decisions
        ),
        "train_positive_count_nll_per_callback": (
            totals["positive_count_nll_sum"] / decisions
        ),
        "train_delta_bit_nll_per_callback": (
            totals["delta_bit_nll_sum"] / decisions
        ),
        "blocked_validation_complete_nll_per_callback": validation.get(
            "complete_nll_per_callback"
        ),
        "blocked_validation_hurdle_nll_per_callback": validation.get(
            "hurdle_nll_per_callback"
        ),
        "blocked_validation_positive_count_nll_per_callback": validation.get(
            "positive_count_nll_per_callback"
        ),
        "blocked_validation_delta_bit_nll_per_callback": validation.get(
            "delta_bit_nll_per_callback"
        ),
        "fit_callbacks": fit_callbacks,
        "blocked_validation_callbacks": validation_callbacks,
        "optimizer_steps": optimizer_steps,
        "observed_pc_local_states": unique_pc_states,
        "selected_epoch": bool(selected),
    }


def select_epoch(
    model, fit_rows, fit_actions, full_train_rows, blocked_rows,
    blocked_actions, positive_count_support, device, args,
):
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate
    )
    runtime_full_train = runtime_features(full_train_rows)
    history, best_epoch, best_validation = [], None, None
    for epoch in range(1, args.epochs + 1):
        totals, optimizer_steps, unique_pc_states = _train_one_epoch(
            model, optimizer, fit_rows, fit_actions,
            positive_count_support, device, args,
        )
        blocked_context, _ = score_suffix(
            model, full_train_rows, runtime_full_train, device,
            args.chunk_len, len(fit_rows),
        )
        validation = validation_nll(
            model, blocked_context, blocked_rows, blocked_actions,
            positive_count_support, device,
        )
        selected = (
            best_validation is None
            or validation["complete_nll_per_callback"]
            < best_validation["complete_nll_per_callback"]
        )
        if selected:
            best_epoch, best_validation = epoch, dict(validation)
        row = _history_row(
            "selection_fit", epoch, totals, optimizer_steps,
            unique_pc_states, len(fit_rows), len(blocked_rows),
            validation=validation, selected=selected,
        )
        history.append(row)
        print(
            "[select:stride-v25] epoch={} train_nll={:.8f} "
            "validation_nll={:.8f} selected={}".format(
                epoch, row["train_complete_nll_per_callback"],
                validation["complete_nll_per_callback"],
                selected,
            ), flush=True,
        )
    if best_epoch is None:
        raise RuntimeError("blocked validation selected no checkpoint")
    return history, best_epoch, best_validation


def retrain_complete_train(
    model, rows, actions, positive_count_support, selected_epochs, device, args,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    for epoch in range(1, int(selected_epochs) + 1):
        totals, optimizer_steps, unique_pc_states = _train_one_epoch(
            model, optimizer, rows, actions, positive_count_support,
            device, args,
        )
        row = _history_row(
            "final_complete_TRAIN_retrain", epoch, totals, optimizer_steps,
            unique_pc_states, len(rows), 0,
            selected=(epoch == int(selected_epochs)),
        )
        history.append(row)
        print(
            "[retrain:stride-v25] epoch={}/{} complete_train_nll={:.8f}"
            .format(
                epoch, selected_epochs,
                row["train_complete_nll_per_callback"],
            ), flush=True,
        )
    return history


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
    entries = triggers = 0
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pc", "line", "occ", "prefetch_addr"])
        for (pc, line, occurrence), targets in zip(rows, actions):
            triggers += int(bool(targets))
            for target in targets:
                writer.writerow([
                    pc, line, occurrence,
                    "0x{:x}".format(int(target) * CACHE_LINE_BYTES),
                ])
                entries += 1
    return entries, triggers


def count_summary(actions):
    counts = [len(items) for items in actions]
    return {
        "rows": len(counts),
        "actions": int(sum(counts)),
        "trigger_rows": int(sum(value > 0 for value in counts)),
        "mean_actions_per_row": (
            float(sum(counts)) / len(counts) if counts else 0.0
        ),
        "count_distribution": {
            str(key): int(value)
            for key, value in sorted(Counter(counts).items())
        },
    }


def self_test_model(hidden_size):
    hurdle_prior_values = [0.6, 0.4]
    count_prior = [0.7, 0.3]
    model = DualContextHurdleStrideLSTM(
        hidden_size, hurdle_prior_values, count_prior,
        [0.5] * LINE_NUMBER_BITS,
    )
    observed = sum(parameter.numel() for parameter in model.parameters())
    expected = expected_parameter_count(hidden_size, 2)
    if observed != expected:
        raise RuntimeError(
            "Stride v25 parameter formula mismatch {} != {}".format(
                observed, expected
            )
        )
    names = [name for name, _ in model.named_parameters()]
    if any(
        token in name
        for name in names
        for token in ("log_count", "stop", "coordinate", "class", "token")
    ):
        raise RuntimeError("forbidden decoder mechanism leaked into v25 model")
    context = torch.zeros((2, hidden_size), dtype=torch.float32)
    with torch.no_grad():
        model.hurdle_head.weight.zero_()
        model.hurdle_head.bias[:] = torch.tensor([-5.0, 5.0])
        model.positive_count_head.weight.zero_()
        model.positive_count_head.bias[:] = torch.tensor([-5.0, 5.0])
        model.delta_bit_head.weight.zero_()
        model.delta_bit_head.bias.fill_(1.0)
    decoded = decode(
        model, context.numpy(), [10, 20], [1, 2],
        torch.device("cpu"), role="self-test",
    )
    if decoded[0].tolist() != [2, 2] or any(
        len(items) != 2 or len(set(items)) != 2 for items in decoded[1]
    ):
        raise RuntimeError("ordered decoder did not schedule K unique actions")
    target, alternate, payload_score = _highest_scoring_feasible_payload(
        10, [1.0] * LINE_NUMBER_BITS, {LINE_MASK & 9}
    )
    if (
        target in {LINE_MASK & 9}
        or not isinstance(alternate, bool)
        or not math.isfinite(payload_score)
    ):
        raise RuntimeError("payload feasibility decoder self-test failed")
    mixed = [2.0, -1.0] + [0.5] * (LINE_NUMBER_BITS - 2)
    _, alternate, observed_score = _highest_scoring_feasible_payload(
        0, mixed, set()
    )
    explicit_score = sum(
        -math.log1p(math.exp(-abs(value))) for value in mixed
    )
    if alternate or not math.isclose(
        observed_score, explicit_score, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("mixed-sign Bernoulli log-probability is wrong")


def build_parser():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--seed", type=int, default=TRAINING_SEED)
    parser.add_argument("--epochs", type=int, default=TRAINING_EPOCHS)
    parser.add_argument("--chunk-len", type=int, default=TRAINING_CHUNK_LEN)
    parser.add_argument(
        "--accumulate-chunks", type=int,
        default=TRAINING_ACCUMULATE_CHUNKS,
    )
    parser.add_argument(
        "--learning-rate", type=float, default=TRAINING_LEARNING_RATE
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    return parser


def main():
    args = build_parser().parse_args()
    expected_pair = MODEL_POINTS["lstm"].get(args.model_size)
    if expected_pair is None or args.pair_id != expected_pair:
        raise RuntimeError("model size/pair is not a configured v25 point")
    pinned = model_points_description()["training_config"]
    observed = {
        "seed": args.seed,
        "epochs": args.epochs,
        "chunk_len": args.chunk_len,
        "accumulate_chunks": args.accumulate_chunks,
        "learning_rate": args.learning_rate,
        "fit_numerator": FIT_NUMERATOR,
        "fit_denominator": FIT_DENOMINATOR,
    }
    if observed != pinned:
        raise RuntimeError(
            "RUN_ID pins training config: observed={} expected={}".format(
                observed, pinned
            )
        )

    reset_random_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not hasattr(torch, "set_float32_matmul_precision"):
        raise RuntimeError("v25 requires torch matmul precision control")
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    )
    if device.type != "cuda" or "A100" not in device_name:
        raise RuntimeError(
            "the pinned v25 run requires an A100; observed {}".format(
                device_name
            )
        )
    model_contract_module.self_test_contract()
    self_test_model(args.model_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    roles = ("train", "guard", "eval")
    stream_paths = {role: getattr(args, role + "_stream") for role in roles}
    action_paths = {
        role: getattr(args, role + "_candidates") for role in roles
    }
    rows = {role: load_stream(stream_paths[role]) for role in roles}
    actions = {
        role: load_teacher_actions(action_paths[role], rows[role])
        for role in roles
    }

    fit_stop = len(rows["train"]) * FIT_NUMERATOR // FIT_DENOMINATOR
    if not 0 < fit_stop < len(rows["train"]):
        raise RuntimeError("TRAIN cannot supply the fixed 80/20 split")
    fit_rows = rows["train"][:fit_stop]
    fit_actions = actions["train"][:fit_stop]
    blocked_rows = rows["train"][fit_stop:]
    blocked_actions = actions["train"][fit_stop:]

    selection_design = realized_label_design(fit_rows, fit_actions)
    validation_positive_counts = {
        len(items) for items in blocked_actions if items
    }
    unsupported_validation_counts = sorted(
        validation_positive_counts
        - set(selection_design["positive_count_support"])
    )
    if unsupported_validation_counts:
        raise RuntimeError(
            "validation positive counts absent from FIT support: {}"
            .format(unsupported_validation_counts)
        )
    reset_random_seed(args.seed)
    selection_model = instantiate_model(args.model_size, selection_design)
    selection_history, selected_epoch, selected_validation = select_epoch(
        selection_model, fit_rows, fit_actions, rows["train"],
        blocked_rows, blocked_actions,
        selection_design["positive_count_support"], device, args,
    )

    # The selected FIT checkpoint is intentionally discarded.  Reinitialize
    # from the same seed and train a new model on all TRAIN callbacks for the
    # selected epoch count, with count support derived from all TRAIN.
    del selection_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reset_random_seed(args.seed)
    final_design = realized_label_design(rows["train"], actions["train"])
    model = instantiate_model(args.model_size, final_design)
    final_history = retrain_complete_train(
        model, rows["train"], actions["train"],
        final_design["positive_count_support"],
        selected_epoch, device, args,
    )
    history = selection_history + final_history
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    expected_parameters = expected_parameter_count(
        args.model_size, len(final_design["positive_count_support"]),
    )
    if parameter_count != expected_parameters:
        raise RuntimeError("realized Stride v25 parameter count changed")

    positive_count_support = final_design["positive_count_support"]

    train_guard_rows = rows["train"] + rows["guard"]
    train_guard_runtime = runtime_features(train_guard_rows)
    guard_context, guard_encoder = score_suffix(
        model, train_guard_rows, train_guard_runtime, device,
        args.chunk_len, len(rows["train"]),
    )
    guard_decode = decode(
        model, guard_context,
        [line for _, line, _ in rows["guard"]],
        positive_count_support, device,
        role="phase-shift-guard-audit",
    )
    guard_metrics = complete_metrics(
        guard_decode[0], guard_decode[1], guard_decode[2],
        actions["guard"],
    )

    complete_rows = train_guard_rows + rows["eval"]
    complete_runtime = runtime_features(complete_rows)
    eval_context, eval_encoder = score_suffix(
        model, complete_rows, complete_runtime, device,
        args.chunk_len, len(train_guard_rows),
    )
    eval_bases = [line for _, line, _ in rows["eval"]]
    eval_decode = decode(
        model, eval_context, eval_bases, positive_count_support,
        device, role="eval",
    )
    heldout = complete_metrics(
        eval_decode[0], eval_decode[1], eval_decode[2],
        actions["eval"],
    )

    teacher_eval_counts = np.asarray([
        len(items) for items in actions["eval"]
    ], dtype=np.int64)
    oracle_count_decode = decode(
        model, eval_context, eval_bases, positive_count_support,
        device, count_override=teacher_eval_counts,
        role="diagnostic-oracle-count",
    )
    oracle_count_metrics = complete_metrics(
        oracle_count_decode[0], oracle_count_decode[1],
        oracle_count_decode[2], actions["eval"],
    )
    oracle_diagnostics = {
        "diagnosis_only": True,
        "excluded_from_fair_replay_claims": True,
        "oracle_count_plus_nn_action": {
            "replayed": False,
            "behavior_metrics": oracle_count_metrics,
            "decoder_diagnostics": oracle_count_decode[3],
        },
        "nn_count_plus_oracle_action": count_oracle_upper_bound(
            eval_decode[0], actions["eval"]
        ),
    }

    normal_path = args.out_dir / "offline_stride.replay.csv"
    nn_path = args.out_dir / "offline_nn.replay.csv"
    normal_entries, normal_triggers = write_replay(
        normal_path, rows["eval"], actions["eval"]
    )
    nn_entries, nn_triggers = write_replay(
        nn_path, rows["eval"], eval_decode[1]
    )
    history_path = args.out_dir / "training_history.csv"
    model_path = args.out_dir / "model.pt"
    write_table(history_path, history)
    torch.save({
        "state_dict": model.state_dict(),
        "run_id": RUN_ID,
        "operation": OPERATION,
        "model_family": "lstm",
        "model_size": args.model_size,
        "positive_count_support": [
            int(value) for value in positive_count_support
        ],
        "hurdle_prior": final_design["hurdle_prior"],
        "positive_count_prior": final_design["positive_count_prior"],
        "delta_bit_prior": final_design["delta_bit_prior"],
        "rank_delta_payload_bits": LINE_NUMBER_BITS,
        "selected_epoch": selected_epoch,
        "selected_blocked_validation": selected_validation,
        "final_retrained_from_scratch": True,
        "final_retrain_epochs": selected_epoch,
        "realized_parameter_count": parameter_count,
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": MODEL_REVISION,
        "decoder_revision": DECODER_REVISION,
    }, model_path)

    contract = model_points_description()
    source_hashes = {
        "trainer_source_sha256": sha256(Path(__file__)),
        "model_contract_source_sha256": sha256(
            Path(__file__).with_name("model_contract.py")
        ),
        "threshold_free_policy_source_sha256": sha256(
            ROOT / "formal_NN_training/common/threshold_free_policy.py"
        ),
    }
    encoder_hash = runtime_encoder_sha256()
    router_hash = state_router_sha256()
    metadata = {
        "run_id": RUN_ID,
        "operation": OPERATION,
        "parent_input_run_id": PARENT_INPUT_RUN_ID,
        "input_reuse": "v23 input package reused byte-for-byte",
        "input_archive_reused_byte_for_byte": True,
        "trace": TRACE,
        "model_tag": model_tag(args.model_size),
        "matched_normal_prefetcher": POLICY,
        "neural_role": "standalone_direct_action_prefetcher",
        "model_family": "lstm",
        "track_model_family": "lstm",
        "model_size": args.model_size,
        "total_recurrent_width": args.model_size,
        "global_recurrent_width": args.model_size // 2,
        "exact_pc_local_recurrent_width": args.model_size // 2,
        "architecture_pair_id": args.pair_id,
        "parameter_count": parameter_count,
        "realized_parameter_count": parameter_count,
        "expected_parameter_count": expected_parameters,
        "parameter_count_is_dataset_dependent": True,
        "parameter_formula": contract["parameter_formula"],
        "model_point_contract": contract,
        "parameter_storage_bytes_float32": parameter_count * 4,
        "seed": args.seed,
        "epochs": args.epochs,
        "chunk_len": args.chunk_len,
        "accumulate_chunks": args.accumulate_chunks,
        "learning_rate": args.learning_rate,
        "training_config": contract["training_config"],
        "training_device": str(device),
        "training_device_name": device_name,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "determinism_fail_closed": True,
        "stochastic_decoding": False,
        "weights_retrained": True,
        "checkpoint_reused": False,
        "decoder_only_change": False,
        "source_decision_effective_external_input": SOURCE_INPUT_LIST,
        "training_runtime_fields": SOURCE_INPUT_LIST,
        "inference_runtime_fields": SOURCE_INPUT_LIST,
        "same_external_input_contract": True,
        "training_inference_input_encoder_identical": True,
        "runtime_feature_count": RUNTIME_FEATURES,
        "raw_runtime_feature_count": RAW_RUNTIME_FEATURES,
        "causal_runtime_feature_count": CAUSAL_RUNTIME_FEATURES,
        "runtime_feature_breakdown": contract["runtime_feature_breakdown"],
        "runtime_encoding": contract["runtime_encoding"],
        "engineered_runtime_features": [],
        "runtime_encoder_sha256": encoder_hash,
        "training_runtime_encoder_sha256": encoder_hash,
        "inference_runtime_encoder_sha256": encoder_hash,
        "teacher_actions_are_model_inputs": False,
        "teacher_actions_are_model_inputs_scope": (
            "labels_comparator_and_diagnosis_only"
        ),
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "normal_policy_request_rate_used_as_budget": False,
        "normal_policy_templates_used_by_neural_inference": False,
        "probability_threshold_used": False,
        "threshold_related_hardcodes_used": False,
        "inference_policy_hardcodes_used": False,
        "neural_degree_cap": None,
        "fixed_page_offset_classes": None,
        "same_page_rule_used_by_neural_inference": False,
        "future_label_window_used": False,
        "decoder_training_mode": DECODER_TRAINING_MODE,
        "decoding_rule": DECODING_RULE,
        "decision_rule": DECODING_RULE,
        "complete_training_objective": FULL_OBJECTIVE,
        "hurdle_training_objective": HURDLE_OBJECTIVE,
        "positive_count_training_objective": POSITIVE_COUNT_OBJECTIVE,
        "positive_only_categorical_count_head_used": True,
        "categorical_count_head_used": True,
        "count_head_used": True,
        "count_regression_used": False,
        "log_count_used": False,
        "hurdle_head_used": True,
        "hurdle_classes": ["ZERO", "POSITIVE"],
        "hurdle_loss_class_weights": None,
        "stop_token_used": False,
        "separate_global_gate_used": True,
        "separate_count_head_used": True,
        "stop_padding_used": False,
        "loss_class_reweighting_used": False,
        "decode_prior_correction_used": False,
        "manual_loss_weights_used": False,
        "count_zero_is_implicit_hurdle": True,
        "positive_count_support": [
            int(value) for value in positive_count_support
        ],
        "positive_count_support_source": "complete_original_TRAIN_labels_only",
        "selection_positive_count_support": [
            int(value) for value in selection_design["positive_count_support"]
        ],
        "selection_positive_count_support_source": "FIT_labels_only",
        "maximum_complete_TRAIN_teacher_count": max(
            len(items) for items in actions["train"]
        ),
        "maximum_count_exposed_as_normal_request_budget": False,
        "count_support_is_dataset_derived": True,
        "count_support_is_normal_request_budget": False,
        "count_support_is_tuned_degree": False,
        "count_train_statistics": final_design[
            "positive_count_statistics"
        ],
        "count_fit_train_statistics": selection_design[
            "positive_count_statistics"
        ],
        "hurdle_complete_TRAIN_natural_prior": final_design["hurdle_prior"],
        "hurdle_FIT_natural_prior": selection_design["hurdle_prior"],
        "delta_bit_initialization": (
            "zero_weight_add_one_smoothed_partition_bit_marginal_logit_bias"
        ),
        "delta_bit_prior_source_selection": "all_real_FIT_teacher_actions",
        "delta_bit_prior_source_final": "all_real_complete_TRAIN_teacher_actions",
        "delta_bit_FIT_add_one_priors": selection_design["delta_bit_prior"],
        "delta_bit_complete_TRAIN_add_one_priors": final_design[
            "delta_bit_prior"
        ],
        "delta_training_objective": DELTA_OBJECTIVE,
        "delta_token_head_used": False,
        "delta_vocabulary_used": False,
        "delta_escape_head_used": False,
        "rank_delta_payload_head": "one_direct_58bit_modular_Bernoulli_head",
        "rank_delta_payload_bits": LINE_NUMBER_BITS,
        "delta_decode_precision": "exact_all_58_modular_bits",
        "full_modular_line_delta_range_reachable": True,
        "delta_bit_loss_scope": "all_58_bits_of_every_real_teacher_rank",
        "all_deltas_relative_to_current_demand": True,
        "stride_fill_level": "FILL_L2_only_no_learned_fill_head",
        "fill_level": "FILL_L2_only_no_fill_head",
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_previous_predicted_action_used_as_input": False,
        "decoder_previous_sampled_action_used_as_input": False,
        "deterministic_target_uniqueness_constraint_used": True,
        "target_uniqueness_constraint_is_neural_action_feedback": False,
        "decoded_target_projection_or_mutation_used": False,
        "target_uniqueness_rule": contract["target_uniqueness_rule"],
        "action_loss_scope": "all_58_bits_of_every_real_teacher_rank",
        "blocked_validation_source": (
            "chronological_last20pct_of_original_TRAIN"
        ),
        "blocked_validation_length_source": (
            BLOCKED_VALIDATION_LENGTH_SOURCE
        ),
        "fit_train_callbacks": len(fit_rows),
        "blocked_validation_callbacks": len(blocked_rows),
        "blocked_validation_selected_checkpoint": True,
        "selected_epoch": selected_epoch,
        "selected_blocked_validation": selected_validation,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "checkpoint_selection_roles": [
            "blocked_TRAIN_validation_NLL", "earlier_epoch_tiebreak"
        ],
        "original_guard_role": ORIGINAL_GUARD_ROLE,
        "original_guard_used_for_checkpoint_selection": False,
        "original_guard_used_for_selection": False,
        "original_guard_phase_shift_metrics": guard_metrics,
        "blocked_validation_behavior_metrics": None,
        "selection_support_derived_from_FIT_only": True,
        "final_support_derived_from_complete_TRAIN_only": True,
        "selected_FIT_checkpoint_reused_for_final_model": False,
        "final_retrained_from_scratch": True,
        "final_retrain_seed_reset": True,
        "final_retrain_epochs": selected_epoch,
        "final_retrain_training_partition": "complete_original_TRAIN",
        "evaluation_used_for_checkpoint_selection": False,
        "evaluation_used_for_selection": False,
        "evaluation_policy_decode_count": 1,
        "diagnostic_eval_decode_count": 1,
        "oracle_diagnostics": oracle_diagnostics,
        "oracle_diagnostics_replayed": False,
        "oracle_diagnostics_excluded_from_fair_claims": True,
        "training_chunks_shuffled": False,
        "dual_context_core_used": True,
        "global_chronological_lstm_used": True,
        "exact_pc_local_lstm_used": True,
        "learned_global_local_fusion_used": True,
        "training_state_mode": "dual_global_chronological_and_exact_pc_local_tbptt",
        "training_state_carried_across_chunks": True,
        "training_state_detached_between_chunks": True,
        "training_state_routing": (
            "one_global_chronological_state_plus_one_local_state_per_exact_PC"
        ),
        "inference_state_routing": (
            "one_global_chronological_state_plus_one_local_state_per_exact_PC"
        ),
        "training_state_router_sha256": router_hash,
        "inference_state_router_sha256": router_hash,
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": MODEL_REVISION,
        "decoder_revision": DECODER_REVISION,
        "event_logger_schema": EVENT_LOGGER_SCHEMA,
        "candidate_attachment_mode": CANDIDATE_ATTACHMENT_MODE,
        "offline_normal_entries": normal_entries,
        "offline_normal_triggers": normal_triggers,
        "offline_nn_entries": nn_entries,
        "offline_nn_triggers": nn_triggers,
        "normal_list_sha256": sha256(normal_path),
        "nn_list_sha256": sha256(nn_path),
        "heldout_behavior_metrics": heldout,
        "decoder_blocked_validation_diagnostics": None,
        "decoder_original_guard_diagnostics": guard_decode[3],
        "decoder_eval_diagnostics": eval_decode[3],
        "encoder_original_guard_diagnostics": guard_encoder,
        "encoder_eval_diagnostics": eval_encoder,
        "train_action_summary": count_summary(actions["train"]),
        "guard_action_summary": count_summary(actions["guard"]),
        "eval_action_summary": count_summary(actions["eval"]),
        "delta_label_statistics": {
            role: delta_label_statistics(rows[role], actions[role])
            for role in roles
        },
        "selection_history": selection_history,
        "final_retrain_history": final_history,
        "train_history": history,
        "model_checkpoint_sha256": sha256(model_path),
        "training_history_sha256": sha256(history_path),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        **source_hashes,
    }
    for role in roles:
        metadata[role + "_stream_gzip_sha256"] = sha256(stream_paths[role])
        metadata[role + "_stream_content_sha256"] = gzip_content_sha256(
            stream_paths[role]
        )
        metadata[role + "_candidate_gzip_sha256"] = sha256(
            action_paths[role]
        )
        metadata[role + "_candidate_content_sha256"] = gzip_content_sha256(
            action_paths[role]
        )
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "status": "PASS",
        "model_tag": metadata["model_tag"],
        "parameters": parameter_count,
        "selected_epoch": selected_epoch,
        "blocked_validation_complete_nll": selected_validation[
            "complete_nll_per_callback"
        ],
        "offline_normal_entries": normal_entries,
        "offline_nn_entries": nn_entries,
    }, indent=2))


if __name__ == "__main__":
    main()
