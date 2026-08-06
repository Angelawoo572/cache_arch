#!/usr/bin/env python3
"""Train and decode the matched-input finite joint-action 623 SPP v23 model.

The NN sees only the source-visible global chronology: DEMAND(addr) and
CACHE_FILL(evicted_addr).  PC remains replay transport.  Teacher SPP actions
are labels and comparator rows only.  A rank decision is one joint token:
STOP or EMIT(delta, fill).  The model has no separate hurdle, count, delta, or
fill argmax and no teacher/predicted action feedback.
"""
import argparse
import copy
import csv
import gzip
import hashlib
import inspect
import json
import math
import os
import platform
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# CUDA deterministic GEMM configuration must precede torch import.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_contract import (
    ACCUMULATE_CHUNKS, ACTION_GROUPS, ADDRESS_BITS, CACHE_LINE_BYTES,
    CACHE_LINE_SHIFT, CHECKPOINT_SELECTION, CHUNK_LEN, DECODER_REVISION,
    DECODER_TRAINING_MODE, EPOCHS, EXPERIMENT_REVISION,
    EXTERNAL_INPUT_FIELDS, FILL_LEVELS, JOINT_ACTION_OBJECTIVE,
    LEARNING_RATE, LINE_ADDRESS_BITS, LINE_ADDRESS_MODULUS,
    MAX_EXACT_DELTAS, MODEL_POINTS, MODEL_REVISION, OPERATION,
    OTHER_DELTA_OBJECTIVE, PARENT_INPUT_RUN_ID, POLICY, RANK_CODE_SIZE,
    RUNTIME_FEATURE_COUNT, RUN_ID, SEED, STOP_TOKEN, TRACE,
    decode_joint_token, describe_model_points, encode_emit_token,
    exact_int as as_int, expected_parameter_count, joint_token_count,
    model_tag, self_test_contract, token_group,
)

# Contract inspection must work on the CPU-only replay host.
if __name__ == "__main__" and sys.argv[1:] == ["--describe-model-points"]:
    print(json.dumps(describe_model_points(), indent=2, sort_keys=True))
    raise SystemExit(0)
if __name__ == "__main__" and sys.argv[1:] == ["--self-test"]:
    self_test_contract()
    print("PASS")
    raise SystemExit(0)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from formal_NN_training.common.threshold_free_policy import behavior_metrics


EVENT_LOGGER_SCHEMA = "623_causal_trigger_fill_v6"
ACTION_ATTACHMENT_MODE = "explicit_trigger_event_id"
CANONICALIZATION_MODE = "per_target_min_fill_queue_effect"
SOURCE_INPUTS = list(EXTERNAL_INPUT_FIELDS)
LINE_ADDRESS_HALF_RANGE = 1 << (LINE_ADDRESS_BITS - 1)
RUNTIME_FEATURES = LINE_ADDRESS_BITS + 1

if RUNTIME_FEATURES != RUNTIME_FEATURE_COUNT:
    raise RuntimeError("SPP v23 raw runtime feature contract changed")


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
            raw_event_id = as_int(row["raw_event_id"])
            cycle = as_int(row["cycle"])
            kind = row["event_kind"]
            decision_idx = as_int(row["decision_idx"])
            pc = as_int(row["pc"])
            address = as_int(row["event_address"])
            line = as_int(row["event_line"])
            hit = as_int(row["cache_hit"])
            occurrence = as_int(row["pc_line_occ"])
            if (
                row["trace"] != TRACE
                or as_int(row["event_idx"]) != index
                or row["logger_schema"] != EVENT_LOGGER_SCHEMA
                or address != line << CACHE_LINE_SHIFT
                or line < 0 or line >= LINE_ADDRESS_MODULUS
                or pc < 0 or pc >= 1 << ADDRESS_BITS
                or raw_event_id <= last_raw_event_id
                or cycle < last_cycle
                or kind not in ("DEMAND", "FILL")
            ):
                raise RuntimeError(
                    "stream identity/input failure at row {}".format(index)
                )
            if kind == "DEMAND":
                expected = occurrences[(pc, line)]
                occurrences[(pc, line)] += 1
                if (
                    decision_idx != len(demands)
                    or occurrence != expected
                    or hit not in (0, 1)
                ):
                    raise RuntimeError(
                        "demand identity failure at row {}".format(index)
                    )
                demands.append((pc, address, line, occurrence))
                demand_positions.append(index)
            elif decision_idx != -1 or pc != 0 or hit != 0 or occurrence != -1:
                raise RuntimeError(
                    "cache-fill context leaks transport fields at {}".format(index)
                )
            context.append((kind, address, line, decision_idx))
            last_raw_event_id, last_cycle = raw_event_id, cycle
    if not context or not demands or len(context) == len(demands):
        raise RuntimeError("empty/no-fill SPP stream {}".format(path))
    return {
        "context": context,
        "demands": demands,
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
            "trigger_event_id", "pf_event_id", "event_distance",
            "raw_action_count", "source_first_pf_event_id",
            "source_last_pf_event_id", "is_self_target", "canonicalization",
            "match_mode", "logger_schema",
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
                row["trace"] != TRACE
                or row["policy"] != POLICY
                or (
                    as_int(row["pc"]), as_int(row["line"]),
                    as_int(row["pc_line_occ"]),
                ) != (pc, line, occurrence)
                or row["logger_schema"] != EVENT_LOGGER_SCHEMA
                or row["match_mode"] != ACTION_ATTACHMENT_MODE
                or as_int(row["action_rank"]) != len(actions[index]) + 1
            ):
                raise RuntimeError(
                    "teacher action identity/rank failure at {}".format(index)
                )
            pf_event = as_int(row["pf_event_id"])
            trigger = as_int(row["trigger_event_id"])
            if (
                pf_event <= last_pf_event
                or trigger >= pf_event
                or as_int(row["event_distance"]) != pf_event - trigger
            ):
                raise RuntimeError("invalid action attachment at {}".format(index))
            pf_line = as_int(row["pf_line"])
            fill = as_int(row["fill_level"])
            if (
                pf_line < 0 or pf_line >= LINE_ADDRESS_MODULUS
                or fill not in FILL_LEVELS
                or as_int(row["accepted"]) != 1
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
    kinds = np.asarray(
        [kind == "DEMAND" for kind, _, _, _ in context], dtype=np.bool_
    )
    features = np.concatenate([
        _unsigned_bits(lines, LINE_ADDRESS_BITS),
        kinds.astype(np.float32)[:, None],
    ], axis=1)
    return {"features": features, "lines": lines, "demand_kind": kinds}


def runtime_encoder_sha256():
    payload = {
        "entrypoint_source": inspect.getsource(runtime_bundle),
        "primitive_source": inspect.getsource(_unsigned_bits),
        "fields": SOURCE_INPUTS,
        "use_pc": False,
        "line_address_bits": LINE_ADDRESS_BITS,
        "cache_line_bytes": CACHE_LINE_BYTES,
        "bit_order": "least_significant_first",
        "callback_kind_encoding": {"DEMAND": 1.0, "FILL": 0.0},
        "derived_runtime_features": [],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def decision_router_sha256(stream):
    payload = {
        "context_rows": len(stream["context"]),
        "demand_positions": [
            int(value) for value in stream["demand_positions"]
        ],
        "decision_indices": [
            int(stream["context"][int(position)][3])
            for position in stream["demand_positions"]
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def decision_router_source_sha256():
    return hashlib.sha256(
        inspect.getsource(decision_router_sha256).encode()
    ).hexdigest()


def canonical_signed_delta(base, target):
    difference = (int(target) - int(base)) % LINE_ADDRESS_MODULUS
    if difference >= LINE_ADDRESS_HALF_RANGE:
        difference -= LINE_ADDRESS_MODULUS
    return difference


def signed_log(delta):
    value = int(delta)
    return math.copysign(math.log1p(abs(value)), value) if value else 0.0


def inverse_signed_log(value):
    scalar = float(value)
    maximum = math.log1p(LINE_ADDRESS_HALF_RANGE)
    scalar = max(-maximum, min(maximum, scalar))
    magnitude = int(round(math.expm1(abs(scalar))))
    delta = -magnitude if scalar < 0 else magnitude
    return max(
        -LINE_ADDRESS_HALF_RANGE,
        min(LINE_ADDRESS_HALF_RANGE - 1, delta),
    )


def build_delta_vocabulary(train_stream, train_actions):
    require_equal_lengths(
        "TRAIN vocabulary", train_stream["demands"], train_actions
    )
    frequencies = Counter()
    for demand, items in zip(train_stream["demands"], train_actions):
        base = demand[2]
        frequencies.update(
            canonical_signed_delta(base, target) for target, _ in items
        )
    ordered = sorted(frequencies, key=lambda value: (-frequencies[value], value))
    exact = ordered[:MAX_EXACT_DELTAS]
    if not exact:
        raise RuntimeError("TRAIN delta vocabulary is empty")
    return exact, frequencies


def vocabulary_statistics(stream, actions, exact_vocabulary):
    exact = set(exact_vocabulary)
    frequencies = Counter()
    for demand, items in zip(stream["demands"], actions):
        frequencies.update(
            canonical_signed_delta(demand[2], target) for target, _ in items
        )
    total = sum(frequencies.values())
    in_vocabulary = sum(
        count for value, count in frequencies.items() if value in exact
    )
    return {
        "action_count": total,
        "unique_signed_deltas": len(frequencies),
        "in_vocabulary_actions": in_vocabulary,
        "other_actions": total - in_vocabulary,
        "in_vocabulary_fraction": (
            in_vocabulary / float(total) if total else 0.0
        ),
        "other_fraction": (
            (total - in_vocabulary) / float(total) if total else 0.0
        ),
    }


def train_action_horizon(actions):
    if not actions:
        raise RuntimeError("TRAIN teacher callback set is empty")
    horizon = max(map(len, actions))
    if horizon < 1:
        raise RuntimeError("TRAIN teacher has no actions")
    return int(horizon)


def build_context_targets(stream, actions, vocabulary, action_horizon):
    """Build EMIT labels plus every available tail STOP through rank H-1."""
    require_equal_lengths(
        "teacher decision targets",
        stream["demand_positions"], stream["demands"], actions,
    )
    if action_horizon < 1:
        raise RuntimeError("TRAIN-derived action horizon must be positive")
    slots = action_horizon
    tokens = np.full((len(stream["context"]), slots), -1, dtype=np.int64)
    other_signed_logs = np.zeros(tokens.shape, dtype=np.float32)
    class_by_delta = {value: index for index, value in enumerate(vocabulary)}
    other_class = len(vocabulary)
    fill_to_index = {value: index for index, value in enumerate(FILL_LEVELS)}
    for decision, position in enumerate(stream["demand_positions"]):
        items = actions[decision]
        if len(items) > action_horizon:
            raise RuntimeError("teacher action list exceeds TRAIN-derived horizon")
        tokens[position, :] = STOP_TOKEN
        base = stream["demands"][decision][2]
        for rank, (target, fill) in enumerate(items):
            delta = canonical_signed_delta(base, target)
            delta_class = class_by_delta.get(delta, other_class)
            tokens[position, rank] = encode_emit_token(
                delta_class, fill_to_index[fill], len(vocabulary)
            )
            other_signed_logs[position, rank] = signed_log(delta)
    return tokens, other_signed_logs


def joint_training_statistics(targets, exact_vocabulary_size):
    tokens, _ = targets
    observed = tokens[tokens >= 0]
    token_count = joint_token_count(exact_vocabulary_size)
    token_counts = np.bincount(observed, minlength=token_count).astype(np.int64)
    groups = np.asarray([
        token_group(token, exact_vocabulary_size) for token in observed
    ], dtype=np.int64)
    group_counts = np.bincount(groups, minlength=len(ACTION_GROUPS)).astype(
        np.int64
    )
    if not bool((group_counts > 0).all()):
        raise RuntimeError(
            "TRAIN must contain STOP, EMIT_L2, and EMIT_LLC rank labels"
        )
    total = float(group_counts.sum())
    group_weights = total / (len(ACTION_GROUPS) * group_counts.astype(np.float64))
    class_weights = np.asarray([
        group_weights[token_group(token, exact_vocabulary_size)]
        for token in range(token_count)
    ], dtype=np.float64)
    token_priors = (token_counts.astype(np.float64) + 1.0) / (
        float(token_counts.sum()) + token_count
    )
    effective_priors = token_priors * class_weights
    effective_priors = effective_priors / effective_priors.sum()
    return {
        "token_counts": token_counts,
        "token_priors": token_priors,
        "effective_weighted_token_priors": effective_priors,
        "group_counts": group_counts,
        "group_weights": group_weights,
        "class_weights": class_weights,
    }


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
                writer.writerow([
                    pc, line, occurrence, hex(pf_line << CACHE_LINE_SHIFT), fill,
                ])
                fill_counts["FILL_L2" if fill == 2 else "FILL_LLC"] += 1
                entries += 1
    return entries, triggers, fill_counts


def write_prediction_replay(path, rows, predicted_lines, predicted_fills):
    require_equal_lengths(
        "prediction replay", rows, predicted_lines, predicted_fills
    )
    entries = triggers = 0
    fill_counts = {"FILL_L2": 0, "FILL_LLC": 0}
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pc", "line", "occ", "prefetch_addr", "fill_level"])
        for (pc, _, line, occurrence), targets, fills in zip(
            rows, predicted_lines, predicted_fills
        ):
            require_equal_lengths("prediction callback", targets, fills)
            triggers += int(bool(targets))
            for pf_line, fill_index in zip(targets, fills):
                fill = FILL_LEVELS[int(fill_index)]
                writer.writerow([
                    pc, line, occurrence,
                    hex(int(pf_line) << CACHE_LINE_SHIFT), fill,
                ])
                fill_counts["FILL_L2" if fill == 2 else "FILL_LLC"] += 1
                entries += 1
    return entries, triggers, fill_counts


def build_modal_llc_control(base_lines, modal_delta):
    lines = [[(int(base) + int(modal_delta)) % LINE_ADDRESS_MODULUS]
             for base in base_lines]
    fills = [[1] for _ in base_lines]
    return lines, fills


def _iter_chunks(length, width):
    for start in range(0, length, width):
        yield start, min(length, start + width)


def rank_code(ranks, dtype):
    ranks = ranks.to(dtype)
    frequencies = ranks.new_tensor([1.0, 0.01])
    phase = ranks.unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)


class GlobalSPPJointLSTM(nn.Module):
    def __init__(self, hidden_size, exact_vocabulary_size):
        super().__init__()
        if (
            hidden_size not in MODEL_POINTS["lstm"]
            or not 0 < exact_vocabulary_size <= MAX_EXACT_DELTAS
        ):
            raise ValueError("unsupported joint SPP dimensions")
        self.hidden_size = int(hidden_size)
        self.exact_vocabulary_size = int(exact_vocabulary_size)
        self.token_count = joint_token_count(exact_vocabulary_size)
        self.input_projection = nn.Linear(RUNTIME_FEATURES, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.rank_fusion = nn.Linear(
            hidden_size + RANK_CODE_SIZE, hidden_size
        )
        self.joint_action = nn.Linear(hidden_size, self.token_count)
        self.other_signed_log = nn.Linear(hidden_size, 1)

    def initialize_train_token_prior(self, token_priors, class_weights):
        priors = torch.as_tensor(
            token_priors, dtype=self.joint_action.bias.dtype
        )
        weights = torch.as_tensor(
            class_weights, dtype=self.joint_action.bias.dtype
        )
        if (
            priors.shape != (self.token_count,)
            or weights.shape != (self.token_count,)
            or bool((priors <= 0).any())
            or bool((weights <= 0).any())
            or not bool(torch.isfinite(priors).all())
            or not bool(torch.isfinite(weights).all())
        ):
            raise RuntimeError("invalid TRAIN joint-token prior")
        effective = priors * weights
        effective = effective / effective.sum()
        with torch.no_grad():
            self.joint_action.weight.zero_()
            self.joint_action.bias.copy_(torch.log(effective))

    def encode(self, features, state=None):
        embedded = torch.tanh(self.input_projection(features))
        output, state = self.lstm(embedded.unsqueeze(0), state)
        return output.squeeze(0), state

    def ranked_heads(self, contexts, ranks):
        code = rank_code(ranks, contexts.dtype)
        ranked = torch.tanh(
            self.rank_fusion(torch.cat([contexts, code], dim=1))
        )
        return (
            self.joint_action(ranked),
            self.other_signed_log(ranked).squeeze(1),
        )


def detach_state(state):
    if state is None:
        return None
    return tuple(value.detach() for value in state)


def chunk_loss(model, contexts, targets, class_weights, device):
    tokens_np, signed_logs_np = targets
    decision_rows = np.flatnonzero(tokens_np[:, 0] >= 0)
    if not len(decision_rows):
        return None, None
    slots = tokens_np.shape[1]
    row_tensor = torch.as_tensor(
        np.repeat(decision_rows, slots), dtype=torch.long, device=device
    )
    ranks = torch.as_tensor(
        np.tile(np.arange(slots, dtype=np.int64), len(decision_rows)),
        dtype=torch.long, device=device,
    )
    truth = torch.as_tensor(
        tokens_np[decision_rows].reshape(-1), dtype=torch.long, device=device
    )
    ranked_contexts = contexts.index_select(0, row_tensor)
    logits, other_predictions = model.ranked_heads(ranked_contexts, ranks)
    weights = torch.as_tensor(
        class_weights, dtype=logits.dtype, device=device
    )
    joint_sum = F.cross_entropy(
        logits, truth, weight=weights, reduction="sum"
    )
    other_class = model.exact_vocabulary_size
    other_tokens = torch.as_tensor([
        encode_emit_token(other_class, fill_index, other_class)
        for fill_index in (0, 1)
    ], dtype=torch.long, device=device)
    other_mask = (truth == other_tokens[0]) | (truth == other_tokens[1])
    if bool(other_mask.any()):
        signed_truth = torch.as_tensor(
            signed_logs_np[decision_rows].reshape(-1),
            dtype=other_predictions.dtype, device=device,
        )
        other_sum = F.smooth_l1_loss(
            other_predictions[other_mask], signed_truth[other_mask],
            reduction="sum",
        )
        other_atoms = int(other_mask.sum().item())
    else:
        other_sum = logits.new_zeros(())
        other_atoms = 0
    total = joint_sum + other_sum
    return total, {
        "joint_sum": float(joint_sum.detach()),
        "joint_atoms": int(truth.numel()),
        "other_sum": float(other_sum.detach()),
        "other_atoms": other_atoms,
        "total_atoms": int(truth.numel()) + other_atoms,
    }


def score_context(model, bundle, device, initial_state=None, chunk_len=8192):
    model.eval()
    parts, state = [], initial_state
    with torch.no_grad():
        for start, stop in _iter_chunks(len(bundle["features"]), chunk_len):
            features = torch.from_numpy(bundle["features"][start:stop]).to(device)
            context, state = model.encode(features, state)
            state = detach_state(state)
            parts.append(context.cpu().numpy())
    return np.concatenate(parts, axis=0), state


def score_role_history(model, bundles, roles, device):
    contexts, state = {}, None
    for role in roles:
        contexts[role], state = score_context(
            model, bundles[role], device, state
        )
    return contexts


def corrected_joint_logits(logits, class_weights):
    weights = torch.as_tensor(
        class_weights, dtype=torch.float64, device=logits.device
    )
    if (
        weights.ndim != 1
        or weights.shape[0] != logits.shape[1]
        or bool((weights <= 0).any())
        or not bool(torch.isfinite(weights).all())
    ):
        raise RuntimeError("invalid TRAIN-derived joint class weights")
    corrected = logits.detach().to(torch.float64) - torch.log(weights).unsqueeze(0)
    if not bool(torch.isfinite(corrected).all()):
        raise RuntimeError("prior-corrected joint logits are non-finite")
    return corrected


def decode_actions(
    model, contexts, base_lines, vocabulary, class_weights,
    train_action_horizon_value, device, chunk_len=8192,
):
    """Decode finite joint rank slots with no action feedback or threshold."""
    require_equal_lengths("decode", contexts, base_lines)
    if (
        len(vocabulary) < 1
        or len(vocabulary) > MAX_EXACT_DELTAS
        or len(set(map(int, vocabulary))) != len(vocabulary)
        or any(
            int(value) < -LINE_ADDRESS_HALF_RANGE
            or int(value) >= LINE_ADDRESS_HALF_RANGE
            for value in vocabulary
        )
    ):
        raise RuntimeError("decode received an invalid TRAIN delta vocabulary")
    if train_action_horizon_value < 1:
        raise RuntimeError("decode received an invalid TRAIN action horizon")
    base_array = np.asarray(base_lines)
    if (
        base_array.ndim != 1
        or not np.issubdtype(base_array.dtype, np.integer)
        or bool((base_array < 0).any())
        or bool((base_array >= LINE_ADDRESS_MODULUS).any())
    ):
        raise RuntimeError("decode base lines are outside the 58-bit domain")

    count = len(contexts)
    decision_slots = int(train_action_horizon_value)
    predicted_lines = [[] for _ in range(count)]
    predicted_fills = [[] for _ in range(count)]
    predicted_tokens = [[] for _ in range(count)]
    stopped = np.zeros(count, dtype=np.bool_)
    token_histogram = Counter()
    group_histogram = Counter()
    entropy_sum = 0.0
    entropy_atoms = 0
    final_rank_emits = 0
    model.eval()
    with torch.no_grad():
        for start, stop in _iter_chunks(count, chunk_len):
            callback_contexts = torch.from_numpy(contexts[start:stop]).to(device)
            local_stopped = torch.zeros(
                stop - start, dtype=torch.bool, device=device
            )
            for rank in range(decision_slots):
                active = torch.nonzero(~local_stopped, as_tuple=False).squeeze(1)
                if int(active.numel()) == 0:
                    break
                ranks = torch.full(
                    (len(active),), rank, dtype=torch.long, device=device
                )
                logits, other_values = model.ranked_heads(
                    callback_contexts.index_select(0, active), ranks
                )
                if (
                    not bool(torch.isfinite(logits).all())
                    or not bool(torch.isfinite(other_values).all())
                ):
                    raise RuntimeError("joint decoder produced non-finite output")
                corrected = corrected_joint_logits(logits, class_weights)
                probabilities = torch.softmax(corrected, dim=1)
                entropy = -(
                    probabilities * torch.log(probabilities.clamp_min(1e-300))
                ).sum(dim=1)
                entropy_sum += float(entropy.sum().item())
                entropy_atoms += len(active)
                chosen = corrected.argmax(dim=1)
                for offset, token, other_value in zip(
                    active.cpu().tolist(), chosen.cpu().tolist(),
                    other_values.cpu().tolist(),
                ):
                    global_row = start + int(offset)
                    token = int(token)
                    predicted_tokens[global_row].append(token)
                    token_histogram[token] += 1
                    kind, delta_class, fill_index = decode_joint_token(
                        token, len(vocabulary)
                    )
                    if kind == "STOP":
                        group_histogram["STOP"] += 1
                        local_stopped[offset] = True
                        stopped[global_row] = True
                        continue
                    group_histogram[
                        "EMIT_L2" if fill_index == 0 else "EMIT_LLC"
                    ] += 1
                    delta = (
                        int(vocabulary[delta_class])
                        if delta_class < len(vocabulary)
                        else inverse_signed_log(other_value)
                    )
                    target = (
                        int(base_lines[global_row]) + delta
                    ) % LINE_ADDRESS_MODULUS
                    predicted_lines[global_row].append(target)
                    predicted_fills[global_row].append(int(fill_index))
                    if rank == decision_slots - 1:
                        final_rank_emits += 1
    counts = np.asarray([len(items) for items in predicted_lines], dtype=np.int64)
    if any(len(lines) != len(fills) for lines, fills in zip(
        predicted_lines, predicted_fills
    )):
        raise RuntimeError("joint output target/fill cardinalities differ")
    token_width = joint_token_count(len(vocabulary))
    mean_entropy = entropy_sum / entropy_atoms if entropy_atoms else 0.0
    diagnostics = {
        "train_action_horizon": int(train_action_horizon_value),
        "joint_decision_rank_count": decision_slots,
        "maximum_possible_actions_from_finite_support": decision_slots,
        "joint_token_evaluations": entropy_atoms,
        "explicit_stop_callbacks": int(stopped.sum()),
        "finite_slot_exhaustion_callbacks": int((~stopped).sum()),
        "decoded_zero_callbacks": int((counts == 0).sum()),
        "decoded_positive_callbacks": int((counts > 0).sum()),
        "materialized_rank_actions": int(counts.sum()),
        "maximum_decoded_count": int(counts.max()) if len(counts) else 0,
        "final_supervised_rank_emit_predictions": final_rank_emits,
        "joint_token_histogram": {
            str(key): token_histogram[key] for key in sorted(token_histogram)
        },
        "joint_group_histogram": {
            key: int(group_histogram[key]) for key in ACTION_GROUPS
        },
        "mean_prior_corrected_joint_token_entropy": mean_entropy,
        "mean_prior_corrected_joint_token_entropy_normalized": (
            mean_entropy / math.log(token_width) if token_width > 1 else 0.0
        ),
        "deterministic_argmax": True,
        "probability_threshold_used": False,
        "action_feedback_used": False,
    }
    return (
        counts, predicted_lines, predicted_fills, predicted_tokens, diagnostics,
    )


def ratio(numerator, denominator):
    return numerator / float(denominator) if denominator else 0.0


def trigger_behavior_metrics(predicted_counts, teacher_actions):
    predicted = np.asarray(predicted_counts) > 0
    teacher = np.asarray([bool(items) for items in teacher_actions])
    true_positive = int(np.logical_and(predicted, teacher).sum())
    precision = ratio(true_positive, int(predicted.sum()))
    recall = ratio(true_positive, int(teacher.sum()))
    return {
        "trigger_true_positive": true_positive,
        "trigger_precision": precision,
        "trigger_recall": recall,
        "trigger_f1": ratio(2 * precision * recall, precision + recall),
    }


def joint_action_metrics(predicted_lines, predicted_fills, teacher_actions):
    true_positive = predicted_total = teacher_total = 0
    l2_true_positive = l2_predicted = l2_teacher = 0
    for lines, fills, items in zip(
        predicted_lines, predicted_fills, teacher_actions
    ):
        predicted = Counter(zip(map(int, lines), map(int, fills)))
        teacher = Counter(
            (int(line), FILL_LEVELS.index(fill)) for line, fill in items
        )
        predicted_total += sum(predicted.values())
        teacher_total += sum(teacher.values())
        true_positive += sum((predicted & teacher).values())
        predicted_l2 = Counter({
            line: count for (line, fill), count in predicted.items() if fill == 0
        })
        teacher_l2_rows = Counter({
            line: count for (line, fill), count in teacher.items() if fill == 0
        })
        l2_predicted += sum(predicted_l2.values())
        l2_teacher += sum(teacher_l2_rows.values())
        l2_true_positive += sum((predicted_l2 & teacher_l2_rows).values())
    precision = ratio(true_positive, predicted_total)
    recall = ratio(true_positive, teacher_total)
    l2_precision = ratio(l2_true_positive, l2_predicted)
    l2_recall = ratio(l2_true_positive, l2_teacher)
    return {
        "joint_true_positive_actions": true_positive,
        "joint_action_precision": precision,
        "joint_action_recall": recall,
        "joint_action_f1": ratio(2 * precision * recall, precision + recall),
        "predicted_l2_actions": l2_predicted,
        "teacher_l2_actions": l2_teacher,
        "l2_joint_true_positive_actions": l2_true_positive,
        "l2_joint_precision": l2_precision,
        "l2_joint_recall": l2_recall,
        "l2_joint_f1": ratio(
            2 * l2_precision * l2_recall, l2_precision + l2_recall
        ),
        "predicted_l2_fraction": ratio(l2_predicted, predicted_total),
        "teacher_l2_fraction": ratio(l2_teacher, teacher_total),
    }


def complete_behavior_metrics(counts, lines, fills, teacher):
    result = behavior_metrics(
        counts, lines, fills, teacher, fill_levels=FILL_LEVELS
    )
    result.update(trigger_behavior_metrics(counts, teacher))
    result.update(joint_action_metrics(lines, fills, teacher))
    return result


def guard_selection_key(metrics, normalized_train_loss, epoch):
    fill_accuracy = metrics.get("fill_accuracy_on_matched_targets")
    fill_accuracy = 0.0 if fill_accuracy is None else float(fill_accuracy)
    return (
        metrics["joint_action_f1"],
        metrics["target_f1"],
        metrics["l2_joint_f1"],
        metrics["trigger_f1"],
        metrics["count_exact_match_rate"],
        fill_accuracy,
        -float(normalized_train_loss),
        -int(epoch),
    )


def output_diagnostics(
    base_lines, counts, predicted_lines, predicted_tokens, vocabulary_size,
):
    duplicate_targets = self_targets = other_actions = 0
    for base, lines, tokens in zip(
        base_lines, predicted_lines, predicted_tokens
    ):
        duplicate_targets += len(lines) - len(set(lines))
        self_targets += sum(int(int(line) == int(base)) for line in lines)
        for token in tokens:
            kind, delta_class, _ = decode_joint_token(token, vocabulary_size)
            if kind == "EMIT" and delta_class == vocabulary_size:
                other_actions += 1
    total = sum(map(len, predicted_lines))
    if int(np.asarray(counts).sum()) != total:
        raise RuntimeError("joint output accounting differs from emitted actions")
    return {
        "raw_predicted_action_count": total,
        "materialized_action_count": total,
        "raw_positive_callback_count": int((np.asarray(counts) > 0).sum()),
        "materialized_positive_callback_count": sum(
            bool(items) for items in predicted_lines
        ),
        "self_target_actions": self_targets,
        "duplicate_target_actions": duplicate_targets,
        "other_escape_actions": other_actions,
        "duplicate_outputs_are_preserved_for_replay": True,
        "delta_legality_fallback": None,
    }


def train_model(
    model, bundles, targets, streams, teachers, vocabulary, priors,
    action_horizon, device, args,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    features = torch.from_numpy(bundles["train"]["features"])
    chunks = list(_iter_chunks(len(features), args.chunk_len))
    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        model.train()
        recurrent = None
        totals = Counter()
        optimizer_steps = 0
        for group_start in range(0, len(chunks), args.accumulate_chunks):
            optimizer.zero_grad(set_to_none=True)
            group_losses, group_atoms = [], 0
            for start, stop in chunks[
                group_start:group_start + args.accumulate_chunks
            ]:
                features_chunk = features[start:stop].to(device)
                context, recurrent = model.encode(features_chunk, recurrent)
                recurrent = detach_state(recurrent)
                sliced = tuple(value[start:stop] for value in targets["train"])
                loss, components = chunk_loss(
                    model, context, sliced, priors["class_weights"], device
                )
                if loss is None:
                    continue
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite SPP v23 training loss")
                group_losses.append(loss)
                group_atoms += components["total_atoms"]
                totals.update(components)
            if not group_losses:
                continue
            (torch.stack(group_losses).sum() / group_atoms).backward()
            optimizer.step()
            optimizer_steps += 1
        normalized = (
            totals["joint_sum"] + totals["other_sum"]
        ) / max(1, totals["total_atoms"])
        guard_contexts = score_role_history(
            model, bundles, ("train", "guard"), device
        )["guard"]
        guard_positions = streams["guard"]["demand_positions"]
        guard_bases = np.asarray(
            [row[2] for row in streams["guard"]["demands"]], dtype=np.int64
        )
        decoded = decode_actions(
            model, guard_contexts[guard_positions], guard_bases, vocabulary,
            priors["class_weights"], action_horizon, device,
        )
        metrics = complete_behavior_metrics(
            decoded[0], decoded[1], decoded[2], teachers["guard"]
        )
        selection = guard_selection_key(metrics, normalized, epoch)
        row = {
            "epoch": epoch,
            "normalized_train_loss": normalized,
            "weighted_joint_token_nll": (
                totals["joint_sum"] / max(1, totals["joint_atoms"])
            ),
            "other_signed_log_loss": (
                totals["other_sum"] / max(1, totals["other_atoms"])
            ),
            "optimizer_steps": optimizer_steps,
            "guard_joint_action_f1": metrics["joint_action_f1"],
            "guard_target_f1": metrics["target_f1"],
            "guard_l2_joint_f1": metrics["l2_joint_f1"],
            "guard_trigger_f1": metrics["trigger_f1"],
            "guard_count_exact_match_rate": metrics["count_exact_match_rate"],
            "guard_fill_accuracy_on_matched_targets": (
                0.0 if metrics.get("fill_accuracy_on_matched_targets") is None
                else metrics["fill_accuracy_on_matched_targets"]
            ),
            "guard_joint_token_entropy": decoded[4][
                "mean_prior_corrected_joint_token_entropy"
            ],
            "guard_selection_key": json.dumps(selection),
        }
        history.append(row)
        if best is None or selection > best["selection_key"]:
            best = {
                "epoch": epoch,
                "selection_key": selection,
                "guard_metrics": metrics,
                "guard_decoder_diagnostics": decoded[4],
                "state_dict": copy.deepcopy({
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                }),
            }
        print(
            "[train:spp-v23] epoch={} loss={:.8f} joint_f1={:.8f} "
            "target_f1={:.8f}".format(
                epoch, normalized, metrics["joint_action_f1"],
                metrics["target_f1"],
            )
        )
    if best is None:
        raise RuntimeError("SPP v23 produced no checkpoint")
    model.load_state_dict(best["state_dict"])
    return history, best


def self_test_model(hidden_size):
    for size in MODEL_POINTS["lstm"]:
        model = GlobalSPPJointLSTM(size, 7)
        observed = sum(parameter.numel() for parameter in model.parameters())
        expected = expected_parameter_count(size, 7)
        if observed != expected:
            raise RuntimeError(
                "SPP v23 parameter formula mismatch: {} != {}".format(
                    observed, expected
                )
            )
    if (
        inverse_signed_log(signed_log(-12345)) != -12345
        or inverse_signed_log(signed_log(6789)) != 6789
    ):
        raise RuntimeError("signed-log OTHER codec round trip failed")
    sample = GlobalSPPJointLSTM(hidden_size, 7)
    sample.eval()
    features = torch.zeros((5, RUNTIME_FEATURES))
    changed = features.clone()
    changed[-1, 0] = 1.0
    with torch.no_grad():
        first, _ = sample.encode(features)
        second, _ = sample.encode(changed)
    if not torch.equal(first[:-1], second[:-1]):
        raise RuntimeError("future callback changed a prior global LSTM state")
    forbidden = ("gate", "count", "fill_head", "delta_head", "action_cell")
    if any(
        any(token in name for token in forbidden)
        for name, _ in sample.named_parameters()
    ):
        raise RuntimeError("factorized head or action feedback leaked into v23")
    sample.initialize_train_token_prior(
        np.ones(joint_token_count(7), dtype=np.float64)
        / joint_token_count(7),
        np.ones(joint_token_count(7), dtype=np.float64),
    )
    with torch.no_grad():
        sample.joint_action.weight.zero_()
        sample.joint_action.bias.fill_(-10.0)
        sample.joint_action.bias[STOP_TOKEN] = 10.0
    stopped = decode_actions(
        sample, np.zeros((2, hidden_size), dtype=np.float32),
        np.asarray([0, 1], dtype=np.int64), list(range(7)),
        np.ones(joint_token_count(7), dtype=np.float64), 2,
        torch.device("cpu"),
    )
    if stopped[0].tolist() != [0, 0] or stopped[4][
        "finite_slot_exhaustion_callbacks"
    ] != 0:
        raise RuntimeError("joint STOP decoder self-test failed")
    emit = encode_emit_token(0, 1, 7)
    with torch.no_grad():
        sample.joint_action.bias.fill_(-10.0)
        sample.joint_action.bias[emit] = 10.0
    finite = decode_actions(
        sample, np.zeros((1, hidden_size), dtype=np.float32),
        np.asarray([0], dtype=np.int64), list(range(7)),
        np.ones(joint_token_count(7), dtype=np.float64), 2,
        torch.device("cpu"),
    )
    if finite[0].tolist() != [2] or finite[4][
        "finite_slot_exhaustion_callbacks"
    ] != 1:
        raise RuntimeError("finite joint-rank support self-test failed")
    toy_stream = {
        "context": [("DEMAND", 0, 0, 0), ("DEMAND", 64, 1, 1)],
        "demands": [(1, 0, 0, 0), (2, 64, 1, 0)],
        "demand_positions": np.asarray([0, 1], dtype=np.int64),
    }
    toy_actions = [[(1, 2)], []]
    toy_targets = build_context_targets(toy_stream, toy_actions, [1], 1)[0]
    if toy_targets.tolist() != [
        [encode_emit_token(0, 0, 1)],
        [STOP_TOKEN],
    ]:
        raise RuntimeError("all-tail STOP supervision self-test failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, choices=[POLICY])
    for role in ("train", "guard", "eval"):
        parser.add_argument(
            "--{}-stream".format(role), required=True, type=Path
        )
        parser.add_argument(
            "--{}-teacher-actions".format(role), required=True, type=Path
        )
    parser.add_argument("--source-contract", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-family", choices=["lstm"], required=True)
    parser.add_argument("--model-size", type=int, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--chunk-len", type=int, default=CHUNK_LEN)
    parser.add_argument(
        "--accumulate-chunks", type=int, default=ACCUMULATE_CHUNKS
    )
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    pinned_training_config = describe_model_points()["training_config"]
    actual_training_config = {
        "seed": args.seed,
        "epochs": args.epochs,
        "chunk_len": args.chunk_len,
        "accumulate_chunks": args.accumulate_chunks,
        "learning_rate": args.learning_rate,
    }
    if actual_training_config != pinned_training_config:
        raise RuntimeError(
            "CLI training config {} differs from pinned run contract {}".format(
                actual_training_config, pinned_training_config
            )
        )
    source_contract = json.loads(args.source_contract.read_text())
    if source_contract.get("decision_effective_external_input") != SOURCE_INPUTS:
        raise RuntimeError("unexpected SPP source input contract")
    if MODEL_POINTS["lstm"].get(args.model_size) != args.pair_id:
        raise RuntimeError("model size/pair is not a configured v23 point")
    if min(
        args.epochs, args.chunk_len, args.accumulate_chunks
    ) < 1:
        raise RuntimeError("model/training dimensions must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cuda_device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else None
    )
    if device.type != "cuda" or "A100" not in str(cuda_device_name):
        raise RuntimeError(
            "pinned v23 run requires an NVIDIA A100; observed {!r}".format(
                cuda_device_name
            )
        )
    if not hasattr(torch, "use_deterministic_algorithms"):
        raise RuntimeError(
            "this torch build cannot enforce deterministic algorithms"
        )
    if not hasattr(torch, "set_float32_matmul_precision") or not hasattr(
        torch, "get_float32_matmul_precision"
    ):
        raise RuntimeError("this torch build cannot pin matmul precision")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    self_test_model(args.model_size)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    roles = ("train", "guard", "eval")
    stream_paths = {
        role: getattr(args, role + "_stream") for role in roles
    }
    action_paths = {
        role: getattr(args, role + "_teacher_actions") for role in roles
    }
    streams = {role: load_stream(stream_paths[role]) for role in roles}
    teachers = {
        role: load_teacher_actions(
            action_paths[role], streams[role]["demands"]
        ) for role in roles
    }
    bundles = {role: runtime_bundle(streams[role]) for role in roles}
    for role in roles:
        if not np.array_equal(
            bundles[role]["features"], runtime_bundle(streams[role])["features"]
        ):
            raise RuntimeError(
                "{} runtime encoder is not reproducible".format(role)
            )

    vocabulary, train_delta_frequencies = build_delta_vocabulary(
        streams["train"], teachers["train"]
    )
    action_horizon = train_action_horizon(teachers["train"])
    targets = {
        "train": build_context_targets(
            streams["train"], teachers["train"], vocabulary, action_horizon
        )
    }
    priors = joint_training_statistics(targets["train"], len(vocabulary))
    vocabulary_stats = {
        role: vocabulary_statistics(
            streams[role], teachers[role], vocabulary
        ) for role in roles
    }

    model = GlobalSPPJointLSTM(args.model_size, len(vocabulary)).to(device)
    model.initialize_train_token_prior(
        priors["token_priors"], priors["class_weights"]
    )
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    if parameter_count != expected_parameter_count(
        args.model_size, len(vocabulary)
    ):
        raise RuntimeError("measured SPP v23 parameter count changed")
    history, best = train_model(
        model, bundles, targets, streams, teachers, vocabulary, priors,
        action_horizon, device, args,
    )

    # Reproduce selected GUARD evidence, then touch EVAL exactly once.
    selected_contexts = score_role_history(
        model, bundles, ("train", "guard"), device
    )
    guard_bases = np.asarray(
        [row[2] for row in streams["guard"]["demands"]], dtype=np.int64
    )
    guard_decode = decode_actions(
        model,
        selected_contexts["guard"][streams["guard"]["demand_positions"]],
        guard_bases, vocabulary, priors["class_weights"], action_horizon,
        device,
    )
    selected_guard_metrics = complete_behavior_metrics(
        guard_decode[0], guard_decode[1], guard_decode[2], teachers["guard"]
    )
    if selected_guard_metrics != best["guard_metrics"]:
        raise RuntimeError("selected guard checkpoint did not reproduce")

    eval_context = score_role_history(model, bundles, roles, device)["eval"]
    eval_bases = np.asarray(
        [row[2] for row in streams["eval"]["demands"]], dtype=np.int64
    )
    eval_decode = decode_actions(
        model, eval_context[streams["eval"]["demand_positions"]], eval_bases,
        vocabulary, priors["class_weights"], action_horizon, device,
    )
    behavior = complete_behavior_metrics(
        eval_decode[0], eval_decode[1], eval_decode[2], teachers["eval"]
    )
    diagnostics = output_diagnostics(
        eval_bases, eval_decode[0], eval_decode[1], eval_decode[3],
        len(vocabulary),
    )
    modal_delta = min(
        train_delta_frequencies,
        key=lambda value: (-train_delta_frequencies[value], value),
    )
    control_lines, control_fills = build_modal_llc_control(
        eval_bases, modal_delta
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    normal_path = args.out_dir / "offline_spp.replay.csv"
    nn_path = args.out_dir / "offline_nn.replay.csv"
    control_path = args.out_dir / "offline_modal_llc_control.replay.csv"
    normal_entries, normal_triggers, normal_fill_counts = write_teacher_replay(
        normal_path, streams["eval"]["demands"], teachers["eval"]
    )
    nn_entries, nn_triggers, nn_fill_counts = write_prediction_replay(
        nn_path, streams["eval"]["demands"], eval_decode[1], eval_decode[2]
    )
    control_entries, control_triggers, control_fill_counts = (
        write_prediction_replay(
            control_path, streams["eval"]["demands"],
            control_lines, control_fills,
        )
    )
    history_path = args.out_dir / "training_history.csv"
    model_path = args.out_dir / "model.pt"
    write_table(history_path, history)
    torch.save({
        "state_dict": model.state_dict(),
        "model_family": "lstm",
        "model_size": args.model_size,
        "runtime_features": RUNTIME_FEATURES,
        "exact_delta_vocabulary": vocabulary,
        "other_class": len(vocabulary),
        "joint_token_count": joint_token_count(len(vocabulary)),
        "joint_group_order": ACTION_GROUPS,
        "joint_group_weights": priors["group_weights"].tolist(),
        "joint_class_weights": priors["class_weights"].tolist(),
        "joint_effective_weighted_token_priors": (
            priors["effective_weighted_token_priors"].tolist()
        ),
        "train_action_horizon": action_horizon,
        "joint_decision_rank_count": action_horizon,
        "selected_epoch": best["epoch"],
        "stochastic_decoding": False,
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": MODEL_REVISION,
        "decoder_revision": DECODER_REVISION,
    }, model_path)

    contract = describe_model_points()
    tag = model_tag("lstm", args.model_size)
    state_bytes = 2 * args.model_size * 4
    token_count = joint_token_count(len(vocabulary))
    metadata = {
        "run_id": RUN_ID,
        "parent_input_run_id": PARENT_INPUT_RUN_ID,
        "trace": TRACE,
        "model_tag": tag,
        "matched_normal_prefetcher": POLICY,
        "neural_role": "standalone_direct_action_prefetcher",
        "model_family": "lstm",
        "track_model_family": "lstm",
        "model_size": args.model_size,
        "architecture_pair_id": args.pair_id,
        "parameter_count": parameter_count,
        "realized_parameter_count": parameter_count,
        "maximum_parameter_count": expected_parameter_count(
            args.model_size, MAX_EXACT_DELTAS
        ),
        "maximum_parameter_count_at_255_exact_deltas": (
            expected_parameter_count(args.model_size, MAX_EXACT_DELTAS)
        ),
        "parameter_count_is_dataset_dependent": True,
        "parameter_formula": contract["parameter_formula"],
        "parameter_storage_bytes_float32": parameter_count * 4,
        "peak_persistent_recurrent_state_bytes": state_bytes,
        "persistent_recurrent_state": (
            "one bounded global chronological LSTM hidden/cell pair"
        ),
        "dynamic_page_state_pages": 0,
        "recurrent_state_dtype": "float32",
        "model_point_contract": contract,
        "seed": args.seed,
        "operation": OPERATION,
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": MODEL_REVISION,
        "decoder_revision": DECODER_REVISION,
        "weights_retrained": True,
        "checkpoint_reused": False,
        "decoder_only_change": False,
        "guard_selected_checkpoint": True,
        "guard_selected_decoder": False,
        "selected_epoch": best["epoch"],
        "guard_selection_key": list(best["selection_key"]),
        "guard_selection_metrics": selected_guard_metrics,
        "guard_selection_rule": contract["guard_selection_rule"],
        "guard_selection_key_fields": [
            "joint_action_f1", "target_f1", "l2_joint_f1", "trigger_f1",
            "count_exact_match_rate", "fill_accuracy_on_matched_targets",
            "negative_normalized_train_loss", "negative_epoch",
        ],
        "guard_selection_composite_or_mean_used": False,
        "evaluation_decode_count": 1,
        "evaluation_used_for_selection": False,
        "training_config": contract["training_config"],
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "determinism_fail_closed": True,
        "cuda_device_name": cuda_device_name,
        "epochs": args.epochs,
        "chunk_len": args.chunk_len,
        "accumulate_chunks": args.accumulate_chunks,
        "learning_rate": args.learning_rate,
        "runtime_feature_count": RUNTIME_FEATURES,
        "runtime_encoding": contract["runtime_encoding"],
        "source_decision_effective_external_input": SOURCE_INPUTS,
        "training_runtime_fields": SOURCE_INPUTS,
        "inference_runtime_fields": SOURCE_INPUTS,
        "same_external_input_contract": True,
        "training_inference_input_encoder_identical": True,
        "runtime_encoder_entrypoint": "train_and_offline_infer.runtime_bundle",
        "runtime_encoder_sha256": runtime_encoder_sha256(),
        "training_runtime_encoder_sha256": runtime_encoder_sha256(),
        "inference_runtime_encoder_sha256": runtime_encoder_sha256(),
        "model_does_not_use_pc": True,
        "pc_is_replay_transport_only": True,
        "model_input_is_causal_external_event_sequence_only": True,
        "cache_fill_feedback_used_as_raw_external_input": True,
        "cache_fill_private_state_used_as_model_input": False,
        "cache_hit_and_type_are_audit_only": True,
        "teacher_actions_are_model_inputs": False,
        "teacher_actions_are_model_inputs_scope": "labels_and_comparator_only",
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "normal_policy_request_rate_used_as_budget": False,
        "normal_policy_constants_used_by_neural_inference": False,
        "probability_threshold_used": False,
        "threshold_related_hardcodes_used": False,
        "inference_policy_hardcodes_used": False,
        "neural_degree_cap": None,
        "same_page_rule_used_by_neural_inference": False,
        "fixed_page_offset_classes": None,
        "normal_policy_templates_used_by_neural_inference": False,
        "future_label_window_used": False,
        "fill_lead_cutoff_used": False,
        "normal_candidate_bank_is_fixed": False,
        "nn_generates_own_target_addresses_and_fill_levels": True,
        "decoder_training_mode": DECODER_TRAINING_MODE,
        "decoding_rule": contract["decoding_rule"],
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_previous_predicted_action_used_as_input": False,
        "decoder_previous_sampled_action_used_as_input": False,
        "teacher_action_values_used_as_decoder_feedback": False,
        "teacher_target_used_as_recurrent_feedback": False,
        "sampled_outputs_used_as_decoder_feedback": False,
        "rank_conditioning": "generic_four_component_sinusoidal_position_code",
        "joint_action_training_objective": JOINT_ACTION_OBJECTIVE,
        "other_delta_training_objective": OTHER_DELTA_OBJECTIVE,
        "joint_action_token_definition": contract[
            "joint_action_token_definition"
        ],
        "joint_action_token_order": (
            "STOP_then_for_each_delta_class_EMIT_L2_then_EMIT_LLC"
        ),
        "joint_action_token_count": token_count,
        "joint_action_group_order": list(ACTION_GROUPS),
        "joint_action_train_token_counts": priors["token_counts"].tolist(),
        "joint_action_train_token_priors_add_one": (
            priors["token_priors"].tolist()
        ),
        "joint_action_initial_effective_weighted_token_priors": (
            priors["effective_weighted_token_priors"].tolist()
        ),
        "joint_action_bias_initialization": contract[
            "joint_action_bias_initialization"
        ],
        "joint_action_train_group_counts": priors["group_counts"].tolist(),
        "joint_action_train_group_weights": priors["group_weights"].tolist(),
        "joint_action_group_weight_formula": "N/(3*N_group)",
        "joint_action_prior_correction_at_decode_used": True,
        "joint_action_prior_correction_rule": contract[
            "joint_action_prior_correction_rule"
        ],
        "joint_action_decoding_rule": contract["decoding_rule"],
        "separate_gate_head_used": False,
        "request_count_head_used": False,
        "request_count_regression_used": False,
        "separate_delta_head_used": False,
        "separate_fill_head_used": False,
        "fill_argmax_used": False,
        "stop_emit_head_used": False,
        "terminal_stop_supervised_for_every_teacher_sequence": False,
        "all_available_tail_stop_supervised": True,
        "maximum_length_sequences_terminate_by_finite_support": True,
        "train_action_horizon": action_horizon,
        "joint_decision_rank_count": action_horizon,
        "finite_output_horizon_source": contract[
            "finite_output_horizon_source"
        ],
        "finite_output_horizon_is_dataset_derived": True,
        "finite_output_horizon_is_normal_request_budget": False,
        "finite_output_horizon_is_tuned_degree": False,
        "maximum_possible_actions_from_finite_support": action_horizon,
        "all_train_teacher_sequences_have_terminal_stop_label": False,
        "delta_training_objective": (
            "joint_action_token_cross_entropy_plus_OTHER_only_signed_log"
        ),
        "delta_decoding_rule": (
            "joint_token_exact_TRAIN_delta_or_signed_log_OTHER_relative_to_"
            "callback_line"
        ),
        "delta_vocabulary_source": (
            "TRAIN_labels_only_top_frequency_then_signed_value_tie_break"
        ),
        "delta_vocabulary_architecture_budget": MAX_EXACT_DELTAS,
        "delta_vocabulary_max_exact": MAX_EXACT_DELTAS,
        "exact_delta_vocabulary": vocabulary,
        "exact_delta_vocabulary_size": len(vocabulary),
        "realized_exact_delta_vocabulary_size": len(vocabulary),
        "other_delta_class": len(vocabulary),
        "delta_vocabulary_statistics": vocabulary_stats,
        "delta_other_escape": (
            "signed_log_continuous_bounded_approximation"
        ),
        "delta_other_decode_precision": (
            "rounded_float32_approximate_except_exact_vocabulary"
        ),
        "full_signed_line_delta_range_reachable": False,
        "every_signed_line_delta_exactly_representable": False,
        "exact_delta_representability_scope": "train_vocabulary_only",
        "train_delta_frequency_histogram": {
            str(key): value
            for key, value in sorted(train_delta_frequencies.items())
        },
        "delta_zero_allowed": True,
        "self_target_actions_allowed": True,
        "delta_legality_constraints": [],
        "delta_legality_fallback": None,
        "duplicate_target_handling": (
            "preserve_all_learned_outputs_for_replay"
        ),
        "fill_training_objective": (
            "inside_single_joint_action_token_cross_entropy"
        ),
        "fill_probability_threshold": None,
        "stochastic_decoding": False,
        "keyed_sampling_used": False,
        "decoder_sampling_roles": [],
        "decoder_guard_diagnostics": guard_decode[4],
        "decoder_eval_diagnostics": eval_decode[4],
        "training_chunks_shuffled": False,
        "training_state_mode": "chronological_stateful_tbptt",
        "training_state_carried_across_chunks": True,
        "training_state_detached_between_chunks": True,
        "inference_history_mode": (
            "fresh_state_then_complete_train_guard_eval_chronology"
        ),
        "global_chronological_lstm": True,
        "routed_demand_fill_recurrent_paths": False,
        "page_local_causal_state": False,
        "handcrafted_semantic_features_used": False,
        "causal_derived_features": [],
        "manual_head_loss_weights_used": False,
        "data_derived_joint_group_weights_used": True,
        "teacher_max_actions_per_callback": {
            role: max(map(len, teachers[role])) for role in roles
        },
        "maximum_action_count_is_learned_not_fixed": False,
        "finite_support_is_train_derived_not_hardcoded": True,
        "causal_no_future_self_test": "PASS",
        "joint_token_bijection_self_test": "PASS",
        "all_tail_stop_supervision_self_test": "PASS",
        "finite_rank_termination_self_test": "PASS",
        "joint_group_prior_correction_self_test": "PASS",
        "rank_no_action_feedback_self_test": "PASS",
        "signed_log_other_codec_self_test": "PASS",
        "integer_csv_exactness_self_test": "PASS",
        "event_logger_schema": EVENT_LOGGER_SCHEMA,
        "action_attachment_mode": ACTION_ATTACHMENT_MODE,
        "teacher_action_canonicalization": CANONICALIZATION_MODE,
        "replay_preserves_explicit_fill_level": True,
        "same_source_input_offline_claim_allowed": True,
        "closed_loop_live_claim_allowed": False,
        "offline_input_feedback_origin": (
            "recorded cache-fill callbacks produced by the source SPP run"
        ),
        "comparison_claim_boundary": (
            "matched-input open-loop offline comparison only"
        ),
        "collection_manifest_role": (
            "historical_input_package_provenance_only"
        ),
        "collection_manifest_decoder_fields_are_current_contract": False,
        "input_reuse": "v22 input package reused byte-for-byte",
        "non_neural_control_name": (
            "every_callback_TRAIN_modal_delta_FILL_LLC"
        ),
        "non_neural_control_uses_model": False,
        "non_neural_control_excluded_from_neural_claims": True,
        "non_neural_control_actions_per_callback": 1,
        "non_neural_control_fill_level": "FILL_LLC",
        "non_neural_control_delta_source": (
            "TRAIN_teacher_action_frequency_only"
        ),
        "non_neural_control_modal_delta": int(modal_delta),
        "non_neural_control_modal_delta_train_frequency": int(
            train_delta_frequencies[modal_delta]
        ),
        "non_neural_control_entries": control_entries,
        "non_neural_control_triggers": control_triggers,
        "non_neural_control_fill_counts": control_fill_counts,
        "non_neural_control_list_sha256": sha256(control_path),
        "source_contract_sha256": sha256(args.source_contract),
        "trainer_source_sha256": sha256(Path(__file__)),
        "model_contract_source_sha256": sha256(
            Path(__file__).with_name("model_contract.py")
        ),
        "threshold_free_policy_source_sha256": sha256(
            REPO_ROOT / "formal_NN_training" / "common"
            / "threshold_free_policy.py"
        ),
        "offline_normal_entries": normal_entries,
        "offline_normal_triggers": normal_triggers,
        "offline_normal_fill_counts": normal_fill_counts,
        "offline_normal_fill_level_counts": normal_fill_counts,
        "offline_nn_entries": nn_entries,
        "offline_nn_triggers": nn_triggers,
        "offline_nn_fill_counts": nn_fill_counts,
        "offline_nn_fill_level_counts": nn_fill_counts,
        "action_output_diagnostics": diagnostics,
        "raw_predicted_action_count": diagnostics[
            "raw_predicted_action_count"
        ],
        "materialized_action_count": diagnostics[
            "materialized_action_count"
        ],
        "normal_list_sha256": sha256(normal_path),
        "nn_list_sha256": sha256(nn_path),
        "heldout_behavior_metrics": behavior,
        "train_history": history,
        "source_contract": source_contract,
        "model_checkpoint_sha256": sha256(model_path),
        "training_history_sha256": sha256(history_path),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "decision_router_source_sha256": decision_router_source_sha256(),
    }
    for role in roles:
        metadata[role + "_decision_router_sha256"] = decision_router_sha256(
            streams[role]
        )
        metadata[role + "_stream_gzip_sha256"] = sha256(stream_paths[role])
        metadata[role + "_stream_content_sha256"] = gzip_content_sha256(
            stream_paths[role]
        )
        metadata[role + "_teacher_actions_gzip_sha256"] = sha256(
            action_paths[role]
        )
        metadata[role + "_teacher_actions_content_sha256"] = (
            gzip_content_sha256(action_paths[role])
        )
    metadata_path = args.out_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "status": "PASS",
        "model_tag": tag,
        "parameters": parameter_count,
        "selected_epoch": best["epoch"],
        "exact_delta_vocabulary_size": len(vocabulary),
        "train_action_horizon": action_horizon,
        "joint_token_count": token_count,
        "offline_normal_entries": normal_entries,
        "offline_nn_entries": nn_entries,
        "offline_nn_fill_level_counts": nn_fill_counts,
        "non_neural_control_entries": control_entries,
    }, indent=2))


if __name__ == "__main__":
    main()
