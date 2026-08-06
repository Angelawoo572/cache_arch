#!/usr/bin/env python3
"""Train and decode the matched-input 623 Stride v24 model.

Runtime input is exactly raw pc64 + line58. The model learns one natural
categorical action count per callback and, only for the teacher's real action
ranks, one direct demand-relative delta. K=0 is the implicit no-request case.
There is no hurdle, count regression, STOP padding, class reweighting, prior
correction, page rule, request budget, or action feedback.
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
from collections import Counter, OrderedDict
from pathlib import Path

import model_contract as model_contract_module
from model_contract import (
    ADDRESS_BITS, BLOCKED_VALIDATION_LENGTH_SOURCE, CACHE_LINE_BYTES,
    CAUSAL_RUNTIME_FEATURES, CHECKPOINT_SELECTION, COUNT_OBJECTIVE,
    DECODER_REVISION, DECODER_TRAINING_MODE, DECODING_RULE,
    DELTA_OBJECTIVE, EXPERIMENT_REVISION, LINE_NUMBER_BITS,
    MAX_DELTA_OUTPUT_CLASSES, MAX_EXACT_DELTA_CLASSES, MODEL_POINTS,
    MODEL_REVISION, OPERATION, ORIGINAL_GUARD_ROLE, PARENT_INPUT_RUN_ID,
    POLICY, RANK_CODE_FEATURES, RAW_RUNTIME_FEATURES, RUN_ID,
    RUNTIME_FEATURES, SOURCE_INPUTS, TRACE,
    TRAINING_ACCUMULATE_CHUNKS, TRAINING_CHUNK_LEN, TRAINING_EPOCHS,
    TRAINING_LEARNING_RATE, TRAINING_SEED, count_statistics,
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
    apply_signed_line_delta,
    behavior_metrics,
)

EVENT_LOGGER_SCHEMA = "623_causal_trigger_v5"
CANDIDATE_ATTACHMENT_MODE = "explicit_trigger_event_id"
SOURCE_INPUT_LIST = list(SOURCE_INPUTS)
LINE_MODULUS = 1 << LINE_NUMBER_BITS
LINE_MASK = LINE_MODULUS - 1
SIGNED_LINE_MIN = -(1 << (LINE_NUMBER_BITS - 1))
SIGNED_LINE_MAX = (1 << (LINE_NUMBER_BITS - 1)) - 1

if (COMMON_ADDRESS_BITS, COMMON_CACHE_LINE_BYTES) != (
    ADDRESS_BITS, CACHE_LINE_BYTES
):
    raise RuntimeError("shared address contract differs from v24 contract")


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


def _initial_state(state_map, keys, hidden_size, device):
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


def _encode_chunk(model, features, pcs, state_map):
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
    projected = torch.tanh(model.input_projection(padded))
    packed = pack_padded_sequence(
        projected, lengths, batch_first=True, enforce_sorted=True
    )
    initial = _initial_state(
        state_map, [pc for pc, _ in groups], model.hidden_size, features.device
    )
    packed_output, final = model.lstm(packed, initial)
    padded_output, _ = pad_packed_sequence(
        packed_output, batch_first=True, total_length=max(lengths)
    )
    context = torch.zeros(
        len(pcs), model.hidden_size, dtype=features.dtype, device=features.device
    )
    for row, (pc, indices) in enumerate(groups):
        positions = torch.as_tensor(
            indices, dtype=torch.long, device=features.device
        )
        context = context.index_copy(
            0, positions, padded_output[row, :len(indices)]
        )
        state_map[pc] = (
            final[0][0, row].detach(), final[1][0, row].detach()
        )
    return context


def state_router_sha256():
    payload = (
        inspect.getsource(_pc_groups)
        + inspect.getsource(_initial_state)
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


def _canonical_signed_line_delta(target, base):
    value = (int(target) - int(base)) & LINE_MASK
    if value >= (1 << (LINE_NUMBER_BITS - 1)):
        value -= LINE_MODULUS
    return value


def _signed_log(value):
    integer = int(value)
    return math.copysign(math.log1p(abs(integer)), integer) if integer else 0.0


def _coordinate_to_delta(value):
    scalar = float(value)
    if not math.isfinite(scalar):
        raise RuntimeError("OTHER delta coordinate is not finite")
    maximum = math.log1p(abs(SIGNED_LINE_MIN))
    scalar = max(-maximum, min(maximum, scalar))
    magnitude = int(round(math.expm1(abs(scalar))))
    delta = -magnitude if scalar < 0 else magnitude
    return max(SIGNED_LINE_MIN, min(SIGNED_LINE_MAX, delta))


def _teacher_deltas(rows, actions):
    values = []
    for (_, base, _), targets in zip(rows, actions):
        values.extend(
            _canonical_signed_line_delta(target, base) for target in targets
        )
    return values


def build_delta_vocabulary(rows, actions):
    frequencies = Counter(_teacher_deltas(rows, actions))
    if not frequencies:
        raise RuntimeError("cannot build an empty delta vocabulary")
    ordered = sorted(
        frequencies, key=lambda value: (-frequencies[value], value)
    )
    return ordered[:MAX_EXACT_DELTA_CLASSES], frequencies


def delta_class_prior(exact_vocabulary, frequencies):
    counts = [int(frequencies[value]) for value in exact_vocabulary]
    exact_total = sum(counts)
    counts.append(sum(frequencies.values()) - exact_total)
    denominator = float(sum(counts) + len(counts))
    return [(value + 1.0) / denominator for value in counts]


def delta_coordinate_initial_bias(rows, actions):
    values = _teacher_deltas(rows, actions)
    if not values:
        raise RuntimeError("delta coordinate initialization requires actions")
    return sum(_signed_log(value) for value in values) / float(len(values))


def vocabulary_statistics(rows, actions, exact_vocabulary):
    exact = set(exact_vocabulary)
    values = _teacher_deltas(rows, actions)
    in_vocabulary = sum(value in exact for value in values)
    return {
        "teacher_actions": len(values),
        "unique_teacher_deltas": len(set(values)),
        "exact_vocabulary_actions": int(in_vocabulary),
        "other_escape_actions": int(len(values) - in_vocabulary),
        "exact_vocabulary_coverage": (
            float(in_vocabulary) / len(values) if values else 0.0
        ),
    }


class NaturalCardinalityStrideLSTM(nn.Module):
    def __init__(
        self, hidden_size, count_prior, delta_prior,
        delta_coordinate_bias,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.count_output_classes = len(count_prior)
        self.delta_output_classes = len(delta_prior)
        self.other_delta_class = self.delta_output_classes - 1
        if (
            self.hidden_size not in MODEL_POINTS["lstm"]
            or self.count_output_classes < 1
            or not 2 <= self.delta_output_classes <= MAX_DELTA_OUTPUT_CLASSES
        ):
            raise ValueError("unsupported realized Stride v24 dimensions")
        self.input_projection = nn.Linear(RUNTIME_FEATURES, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.rank_projection = nn.Linear(RANK_CODE_FEATURES, hidden_size)
        self.count_head = nn.Linear(hidden_size, self.count_output_classes)
        self.delta_class_head = nn.Linear(
            hidden_size, self.delta_output_classes
        )
        self.delta_coordinate_head = nn.Linear(hidden_size, 1)

        count_tensor = torch.as_tensor(
            count_prior, dtype=self.count_head.bias.dtype
        )
        delta_tensor = torch.as_tensor(
            delta_prior, dtype=self.delta_class_head.bias.dtype
        )
        if (
            bool((count_tensor <= 0).any())
            or bool((delta_tensor <= 0).any())
            or not bool(torch.isfinite(count_tensor).all())
            or not bool(torch.isfinite(delta_tensor).all())
            or not math.isfinite(float(delta_coordinate_bias))
        ):
            raise ValueError("TRAIN-derived initialization is invalid")
        with torch.no_grad():
            self.count_head.weight.zero_()
            self.count_head.bias.copy_(torch.log(count_tensor))
            self.delta_class_head.weight.zero_()
            self.delta_class_head.bias.copy_(torch.log(delta_tensor))
            self.delta_coordinate_head.bias.fill_(
                float(delta_coordinate_bias)
            )


def _chunk_objective(
    model, context, base_lines, actions, exact_vocabulary,
):
    counts = np.asarray([len(items) for items in actions], dtype=np.int64)
    if (
        len(counts) != len(context)
        or len(counts) == 0
        or int(counts.max()) >= model.count_output_classes
        or model.other_delta_class != len(exact_vocabulary)
    ):
        raise RuntimeError("v24 chunk labels are outside realized support")
    count_targets = torch.from_numpy(counts).to(
        device=context.device, dtype=torch.long
    )
    count_sum = F.cross_entropy(
        model.count_head(context), count_targets, reduction="sum"
    )
    delta_sum = context.sum() * 0.0
    coordinate_sum = context.sum() * 0.0
    action_atoms = other_atoms = 0
    vocabulary_index = {
        int(delta): index for index, delta in enumerate(exact_vocabulary)
    }
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
        delta_values = [
            _canonical_signed_line_delta(
                actions[row][rank], int(base_array[row])
            )
            for row in active_np
        ]
        classes_np = np.asarray([
            vocabulary_index.get(delta, model.other_delta_class)
            for delta in delta_values
        ], dtype=np.int64)
        classes = torch.from_numpy(classes_np).to(context.device)
        logits = model.delta_class_head(ranked)
        delta_sum = delta_sum + F.cross_entropy(
            logits, classes, reduction="sum"
        )
        action_atoms += len(active_np)

        other_mask_np = classes_np == model.other_delta_class
        if bool(other_mask_np.any()):
            other_mask = torch.from_numpy(other_mask_np).to(context.device)
            predictions = model.delta_coordinate_head(ranked).squeeze(1)
            truth = torch.as_tensor(
                [_signed_log(value) for value in delta_values],
                dtype=context.dtype, device=context.device,
            )
            coordinate_sum = coordinate_sum + F.smooth_l1_loss(
                predictions[other_mask], truth[other_mask], reduction="sum"
            )
            other_atoms += int(other_mask_np.sum())

    decision_atoms = len(counts)
    list_nll = (count_sum + delta_sum) / float(decision_atoms)
    auxiliary = (
        coordinate_sum / float(other_atoms)
        if other_atoms else context.sum() * 0.0
    )
    objective = list_nll + auxiliary
    return objective, {
        "count_nll_sum": float(count_sum.detach()),
        "delta_nll_sum": float(delta_sum.detach()),
        "other_aux_sum": float(coordinate_sum.detach()),
        "decision_atoms": decision_atoms,
        "action_atoms": action_atoms,
        "other_atoms": other_atoms,
        "list_nll_per_callback": float(list_nll.detach()),
        "normalized_objective": float(objective.detach()),
        "objective_chunks": 1,
    }


def score_suffix(model, rows, runtime, device, chunk_len, output_start):
    if not 0 <= output_start <= len(rows):
        raise RuntimeError("invalid scored suffix")
    output = np.empty(
        (len(rows) - output_start, model.hidden_size), dtype=np.float32
    )
    pcs = np.asarray([pc for pc, _, _ in rows], dtype=np.uint64)
    state_map = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), chunk_len):
            stop = min(start + chunk_len, len(rows))
            features = torch.from_numpy(runtime[start:stop]).to(
                device=device, dtype=torch.float32
            )
            context = _encode_chunk(
                model, features, pcs[start:stop], state_map
            )
            copy_start = max(start, output_start)
            if copy_start < stop:
                output[copy_start - output_start:stop - output_start] = (
                    context[copy_start - start:].cpu().numpy()
                )
    return output, {"rows": len(rows), "unique_pc_states": len(state_map)}


def validation_nll(
    model, context_numpy, rows, actions, exact_vocabulary, device,
    chunk_len=4096,
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
                actions[start:stop], exact_vocabulary,
            )
            totals.update(components)
    categorical = (
        totals["count_nll_sum"] + totals["delta_nll_sum"]
    ) / max(1, totals["decision_atoms"])
    auxiliary = totals["other_aux_sum"] / max(1, totals["other_atoms"])
    return {
        "natural_action_list_nll_per_callback": float(categorical),
        "count_nll_per_callback": (
            totals["count_nll_sum"] / max(1, totals["decision_atoms"])
        ),
        "delta_nll_per_callback": (
            totals["delta_nll_sum"] / max(1, totals["decision_atoms"])
        ),
        "other_auxiliary_per_other_action": float(auxiliary),
        "decision_atoms": int(totals["decision_atoms"]),
        "action_atoms": int(totals["action_atoms"]),
        "other_atoms": int(totals["other_atoms"]),
    }


def decode(
    model, context_numpy, base_lines, exact_vocabulary, device,
    count_override=None, role="eval", chunk_len=4096,
):
    if len(context_numpy) != len(base_lines):
        raise RuntimeError("decoder row counts differ")
    if model.other_delta_class != len(exact_vocabulary):
        raise RuntimeError("decoder vocabulary differs from model")
    count_logits_parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(base_lines), chunk_len):
            stop = min(start + chunk_len, len(base_lines))
            context = torch.from_numpy(context_numpy[start:stop]).to(device)
            logits = model.count_head(context)
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("categorical count logits are non-finite")
            count_logits_parts.append(logits.cpu())
    count_logits = torch.cat(count_logits_parts, dim=0)
    probabilities = torch.softmax(count_logits.to(torch.float64), dim=1)
    entropy = -(
        probabilities * torch.log(probabilities.clamp_min(1e-300))
    ).sum(dim=1)
    natural_counts = count_logits.argmax(dim=1).numpy().astype(np.int64)
    if count_override is None:
        counts = natural_counts
    else:
        counts = np.asarray(count_override, dtype=np.int64)
        if len(counts) != len(base_lines) or bool((counts < 0).any()):
            raise RuntimeError("oracle count override is invalid")

    predicted_lines = [[] for _ in base_lines]
    predicted_fills = [[] for _ in base_lines]
    predicted_classes = [[] for _ in base_lines]
    delta_entropy_sum = 0.0
    delta_atoms = 0
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
                logits = model.delta_class_head(ranked)
                coordinates = model.delta_coordinate_head(
                    ranked
                ).squeeze(1)
                if (
                    not bool(torch.isfinite(logits).all())
                    or not bool(torch.isfinite(coordinates).all())
                ):
                    raise RuntimeError("rank action output is non-finite")
                delta_probabilities = torch.softmax(
                    logits.to(torch.float64), dim=1
                )
                delta_entropy_sum += float(
                    (-(delta_probabilities * torch.log(
                        delta_probabilities.clamp_min(1e-300)
                    )).sum(dim=1)).sum().item()
                )
                delta_atoms += len(active_np)
                choices = logits.argmax(dim=1).cpu().tolist()
                values = coordinates.cpu().tolist()
                for local, choice, coordinate in zip(
                    active_np, choices, values
                ):
                    choice = int(choice)
                    if choice == model.other_delta_class:
                        delta = _coordinate_to_delta(coordinate)
                    elif 0 <= choice < len(exact_vocabulary):
                        delta = int(exact_vocabulary[choice])
                    else:
                        raise RuntimeError("invalid delta class")
                    row = start + int(local)
                    predicted_lines[row].append(
                        int(apply_signed_line_delta(
                            int(base_lines[row]), delta
                        ))
                    )
                    predicted_fills[row].append(-1)
                    predicted_classes[row].append(choice)
    if any(
        len(items) != int(count)
        for items, count in zip(predicted_lines, counts)
    ):
        raise RuntimeError("rank decoder did not realize selected count")
    count_width = model.count_output_classes
    delta_width = model.delta_output_classes
    diagnostics = {
        "role": role,
        "count_override_used": count_override is not None,
        "count_output_classes": count_width,
        "count_support": list(range(count_width)),
        "decoded_count_distribution": {
            str(key): int(value)
            for key, value in sorted(Counter(counts.tolist()).items())
        },
        "decoded_positive_callbacks": int((counts > 0).sum()),
        "decoded_total_actions": int(counts.sum()),
        "decoded_max_actions_per_callback": (
            int(counts.max()) if len(counts) else 0
        ),
        "mean_count_entropy": float(entropy.mean().item()),
        "mean_count_entropy_normalized": (
            float(entropy.mean().item()) / math.log(count_width)
            if count_width > 1 else 0.0
        ),
        "mean_delta_entropy": (
            delta_entropy_sum / delta_atoms if delta_atoms else None
        ),
        "mean_delta_entropy_normalized": (
            delta_entropy_sum / delta_atoms / math.log(delta_width)
            if delta_atoms and delta_width > 1 else None
        ),
        "delta_class_histogram": {
            str(key): int(value)
            for key, value in sorted(Counter(
                choice for items in predicted_classes for choice in items
            ).items())
        },
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


def train_model(
    model, fit_rows, fit_actions, full_train_rows, blocked_rows,
    blocked_actions, exact_vocabulary, device, args,
):
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate
    )
    runtime_fit = runtime_features(fit_rows)
    runtime_full_train = runtime_features(full_train_rows)
    pcs = np.asarray([pc for pc, _, _ in fit_rows], dtype=np.uint64)
    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        model.train()
        state_map = {}
        totals = Counter()
        optimizer.zero_grad(set_to_none=True)
        pending = optimizer_steps = 0
        for start in range(0, len(fit_rows), args.chunk_len):
            stop = min(start + args.chunk_len, len(fit_rows))
            features = torch.from_numpy(runtime_fit[start:stop]).to(
                device=device, dtype=torch.float32
            )
            context = _encode_chunk(
                model, features, pcs[start:stop], state_map
            )
            objective, components = _chunk_objective(
                model, context,
                [line for _, line, _ in fit_rows[start:stop]],
                fit_actions[start:stop], exact_vocabulary,
            )
            if not torch.isfinite(objective):
                raise RuntimeError("non-finite Stride v24 objective")
            objective.backward()
            pending += 1
            totals.update(components)
            if (
                pending == args.accumulate_chunks
                or stop == len(fit_rows)
            ):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(float(pending))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                optimizer_steps += 1

        blocked_context, _ = score_suffix(
            model, full_train_rows, runtime_full_train, device,
            args.chunk_len, len(fit_rows),
        )
        validation = validation_nll(
            model, blocked_context, blocked_rows, blocked_actions,
            exact_vocabulary, device,
        )
        selection = (
            -validation["natural_action_list_nll_per_callback"],
            -epoch,
        )
        selected = best is None or selection > best["selection_key"]
        if selected:
            best = {
                "epoch": epoch,
                "selection_key": selection,
                "validation": validation,
                "state_dict": copy.deepcopy({
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                }),
            }
        train_list_nll = (
            totals["count_nll_sum"] + totals["delta_nll_sum"]
        ) / max(1, totals["decision_atoms"])
        row = {
            "epoch": epoch,
            "train_natural_action_list_nll_per_callback": train_list_nll,
            "train_count_nll_per_callback": (
                totals["count_nll_sum"] / max(1, totals["decision_atoms"])
            ),
            "train_delta_nll_per_callback": (
                totals["delta_nll_sum"] / max(1, totals["decision_atoms"])
            ),
            "train_other_auxiliary_per_other_action": (
                totals["other_aux_sum"] / max(1, totals["other_atoms"])
            ),
            "blocked_validation_natural_action_list_nll_per_callback": (
                validation["natural_action_list_nll_per_callback"]
            ),
            "blocked_validation_count_nll_per_callback": (
                validation["count_nll_per_callback"]
            ),
            "blocked_validation_delta_nll_per_callback": (
                validation["delta_nll_per_callback"]
            ),
            "blocked_validation_other_auxiliary_per_other_action": (
                validation["other_auxiliary_per_other_action"]
            ),
            "fit_callbacks": len(fit_rows),
            "blocked_validation_callbacks": len(blocked_rows),
            "optimizer_steps": optimizer_steps,
            "observed_pc_states": len(state_map),
            "selected_checkpoint": selected,
        }
        history.append(row)
        print(
            "[train:stride-v24] epoch={} train_list_nll={:.8f} "
            "blocked_list_nll={:.8f} selected={}".format(
                epoch, train_list_nll,
                validation["natural_action_list_nll_per_callback"],
                selected,
            ), flush=True,
        )
    if best is None:
        raise RuntimeError("blocked validation selected no checkpoint")
    model.load_state_dict(best["state_dict"])
    return history, best


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
    count_prior = [0.4, 0.4, 0.2]
    delta_prior = [0.5, 0.3, 0.2]
    model = NaturalCardinalityStrideLSTM(
        hidden_size, count_prior, delta_prior, 0.0
    )
    observed = sum(parameter.numel() for parameter in model.parameters())
    expected = expected_parameter_count(hidden_size, 3, 3)
    if observed != expected:
        raise RuntimeError(
            "Stride v24 parameter formula mismatch {} != {}".format(
                observed, expected
            )
        )
    names = [name for name, _ in model.named_parameters()]
    if any(
        token in name
        for name in names
        for token in ("hurdle", "log_count", "stop")
    ):
        raise RuntimeError("v23 decoder mechanism leaked into v24 model")
    context = torch.zeros((2, hidden_size), dtype=torch.float32)
    with torch.no_grad():
        model.count_head.weight.zero_()
        model.count_head.bias[:] = torch.tensor([-5.0, -5.0, 5.0])
        model.delta_class_head.weight.zero_()
        model.delta_class_head.bias[:] = torch.tensor([5.0, -5.0, -5.0])
    decoded = decode(
        model, context.numpy(), [10, 20], [1, 2],
        torch.device("cpu"), role="self-test",
    )
    if decoded[0].tolist() != [2, 2] or any(
        len(items) != 2 for items in decoded[1]
    ):
        raise RuntimeError("categorical count did not schedule exactly K actions")


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
        raise RuntimeError("model size/pair is not a configured v24 point")
    pinned = model_points_description()["training_config"]
    observed = {
        "seed": args.seed,
        "epochs": args.epochs,
        "chunk_len": args.chunk_len,
        "accumulate_chunks": args.accumulate_chunks,
        "learning_rate": args.learning_rate,
    }
    if observed != pinned:
        raise RuntimeError(
            "RUN_ID pins training config: observed={} expected={}".format(
                observed, pinned
            )
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not hasattr(torch, "set_float32_matmul_precision"):
        raise RuntimeError("v24 requires torch matmul precision control")
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
            "the pinned v24 run requires an A100; observed {}".format(
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

    validation_length = len(rows["guard"])
    if validation_length < 1 or len(rows["train"]) <= validation_length:
        raise RuntimeError(
            "TRAIN cannot supply a guard-length blocked validation suffix"
        )
    fit_stop = len(rows["train"]) - validation_length
    fit_rows = rows["train"][:fit_stop]
    fit_actions = actions["train"][:fit_stop]
    blocked_rows = rows["train"][fit_stop:]
    blocked_actions = actions["train"][fit_stop:]

    count_stats = count_statistics([
        len(items) for items in actions["train"]
    ])
    fit_count_stats = count_statistics([
        len(items) for items in fit_actions
    ])
    count_classes = count_stats["count_output_classes"]
    fit_frequencies = list(fit_count_stats["class_frequencies"])
    fit_frequencies.extend([0] * (count_classes - len(fit_frequencies)))
    count_prior = [
        (value + 1.0) / float(len(fit_actions) + count_classes)
        for value in fit_frequencies
    ]

    exact_vocabulary, train_delta_frequencies = build_delta_vocabulary(
        fit_rows, fit_actions
    )
    delta_prior = delta_class_prior(
        exact_vocabulary, train_delta_frequencies
    )
    coordinate_bias = delta_coordinate_initial_bias(
        fit_rows, fit_actions
    )
    model = NaturalCardinalityStrideLSTM(
        args.model_size, count_prior, delta_prior, coordinate_bias
    )
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    expected_parameters = expected_parameter_count(
        args.model_size, count_classes, len(exact_vocabulary) + 1
    )
    if parameter_count != expected_parameters:
        raise RuntimeError("realized Stride v24 parameter count changed")

    history, best = train_model(
        model, fit_rows, fit_actions, rows["train"], blocked_rows,
        blocked_actions, exact_vocabulary, device, args,
    )

    full_train_runtime = runtime_features(rows["train"])
    blocked_context, _ = score_suffix(
        model, rows["train"], full_train_runtime, device,
        args.chunk_len, fit_stop,
    )
    blocked_decode = decode(
        model, blocked_context,
        [line for _, line, _ in blocked_rows],
        exact_vocabulary, device, role="blocked-validation-audit",
    )
    blocked_metrics = complete_metrics(
        blocked_decode[0], blocked_decode[1], blocked_decode[2],
        blocked_actions,
    )

    train_guard_rows = rows["train"] + rows["guard"]
    train_guard_runtime = runtime_features(train_guard_rows)
    guard_context, guard_encoder = score_suffix(
        model, train_guard_rows, train_guard_runtime, device,
        args.chunk_len, len(rows["train"]),
    )
    guard_decode = decode(
        model, guard_context,
        [line for _, line, _ in rows["guard"]],
        exact_vocabulary, device, role="phase-shift-guard-audit",
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
        model, eval_context, eval_bases, exact_vocabulary,
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
        model, eval_context, eval_bases, exact_vocabulary,
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
        "count_support": list(range(count_classes)),
        "count_prior": count_prior,
        "exact_delta_vocabulary": [int(v) for v in exact_vocabulary],
        "other_delta_class": model.other_delta_class,
        "selected_epoch": best["epoch"],
        "selected_blocked_validation": best["validation"],
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
        "count_training_objective": COUNT_OBJECTIVE,
        "categorical_count_head_used": True,
        "count_head_used": True,
        "count_regression_used": False,
        "log_count_used": False,
        "hurdle_head_used": False,
        "stop_token_used": False,
        "separate_global_gate_used": False,
        "separate_count_head_used": False,
        "stop_padding_used": False,
        "loss_class_reweighting_used": False,
        "decode_prior_correction_used": False,
        "manual_loss_weights_used": False,
        "count_zero_is_implicit_hurdle": True,
        "count_support": list(range(count_classes)),
        "count_support_source": (
            "zero_through_maximum_complete_original_TRAIN_teacher_count"
        ),
        "count_support_is_dataset_derived": True,
        "count_support_is_normal_request_budget": False,
        "count_support_is_tuned_degree": False,
        "count_train_statistics": count_stats,
        "count_fit_train_class_frequencies": fit_frequencies,
        "count_fit_train_add_one_natural_priors": count_prior,
        "delta_training_objective": DELTA_OBJECTIVE,
        "delta_vocabulary_source": (
            "FIT_TRAIN_labels_only_top_frequency_then_signed_value"
        ),
        "delta_vocabulary_max_exact": MAX_EXACT_DELTA_CLASSES,
        "exact_delta_vocabulary": [int(v) for v in exact_vocabulary],
        "exact_delta_vocabulary_size": len(exact_vocabulary),
        "other_delta_class": model.other_delta_class,
        "delta_class_empirical_prior": delta_prior,
        "delta_coordinate_initial_bias": coordinate_bias,
        "delta_other_escape": (
            "signed_log_continuous_bounded_approximation"
        ),
        "delta_coordinate_auxiliary_scope": "OTHER_teacher_actions_only",
        "all_deltas_relative_to_current_demand": True,
        "stride_fill_level": "FILL_L2_only_no_learned_fill_head",
        "fill_level": "FILL_L2_only_no_fill_head",
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_previous_predicted_action_used_as_input": False,
        "decoder_previous_sampled_action_used_as_input": False,
        "action_loss_scope": "teacher_action_ranks_only",
        "blocked_validation_source": (
            "chronological_suffix_of_original_TRAIN"
        ),
        "blocked_validation_length_source": (
            BLOCKED_VALIDATION_LENGTH_SOURCE
        ),
        "fit_train_callbacks": len(fit_rows),
        "blocked_validation_callbacks": len(blocked_rows),
        "blocked_validation_selected_checkpoint": True,
        "selected_epoch": best["epoch"],
        "selected_blocked_validation": best["validation"],
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "checkpoint_selection_roles": [
            "blocked_TRAIN_validation_NLL", "earlier_epoch_tiebreak"
        ],
        "original_guard_role": ORIGINAL_GUARD_ROLE,
        "original_guard_used_for_checkpoint_selection": False,
        "original_guard_used_for_selection": False,
        "original_guard_phase_shift_metrics": guard_metrics,
        "blocked_validation_behavior_metrics": blocked_metrics,
        "evaluation_used_for_checkpoint_selection": False,
        "evaluation_used_for_selection": False,
        "evaluation_policy_decode_count": 1,
        "diagnostic_eval_decode_count": 1,
        "oracle_diagnostics": oracle_diagnostics,
        "oracle_diagnostics_replayed": False,
        "oracle_diagnostics_excluded_from_fair_claims": True,
        "training_chunks_shuffled": False,
        "training_state_mode": "exact_pc_keyed_stateful_tbptt",
        "training_state_carried_across_chunks": True,
        "training_state_detached_between_chunks": True,
        "training_state_routing": "one_lstm_state_per_exact_observed_PC",
        "inference_state_routing": "one_lstm_state_per_exact_observed_PC",
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
        "decoder_blocked_validation_diagnostics": blocked_decode[3],
        "decoder_original_guard_diagnostics": guard_decode[3],
        "decoder_eval_diagnostics": eval_decode[3],
        "encoder_original_guard_diagnostics": guard_encoder,
        "encoder_eval_diagnostics": eval_encoder,
        "train_action_summary": count_summary(actions["train"]),
        "guard_action_summary": count_summary(actions["guard"]),
        "eval_action_summary": count_summary(actions["eval"]),
        "delta_vocabulary_statistics": {
            role: vocabulary_statistics(
                rows[role], actions[role], exact_vocabulary
            ) for role in roles
        },
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
        "selected_epoch": best["epoch"],
        "blocked_validation_list_nll": best["validation"][
            "natural_action_list_nll_per_callback"
        ],
        "offline_normal_entries": normal_entries,
        "offline_nn_entries": nn_entries,
    }, indent=2))


if __name__ == "__main__":
    main()
