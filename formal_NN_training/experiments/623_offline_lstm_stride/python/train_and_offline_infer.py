#!/usr/bin/env python3
"""Train and decode the independent matched-input 623 Stride v22 model.

The runtime boundary is deliberately small and auditable: only the current
``pc`` and aligned ``addr`` are external inputs, encoded losslessly as raw
``pc64+line58`` bits.  Captured Stride actions are labels and the offline-normal
replay; they are never an encoder input, decoder prefix, candidate list,
request budget, engineered feature, or action template.

The model has one exact-PC keyed single-layer LSTM.  A callback-level learned
two-class hurdle predicts zero versus positive actions.  On positive TRAIN
rows, a learned log-count predicts cardinality.  A shared rank-conditioned
head then predicts each current-demand-relative delta independently.  Up to
255 frequent TRAIN deltas have exact categorical symbols; one dynamic OTHER
row carries a bounded continuous signed-log approximation.  No teacher or
predicted action is fed to a later rank.  Inference uses deterministic argmax,
a finite rounded-exp count mode, and rankwise delta MAP, with no probability
threshold, source template, page rule, normal budget, degree cap, or action
feedback.
"""
import argparse
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
    ADDRESS_BITS, CACHE_LINE_BYTES, CACHE_LINE_OFFSET_BITS,
    CAUSAL_RUNTIME_FEATURES, CHECKPOINT_SELECTION, COUNT_OBJECTIVE,
    DECODE_PER_CALLBACK_WATCHDOG, DECODE_PER_ROLE_WATCHDOG,
    DECODER_REVISION, DECODER_TRAINING_MODE, DECODING_RULE,
    DELTA_OBJECTIVE, EXPERIMENT_REVISION, HURDLE_OBJECTIVE,
    LINE_NUMBER_BITS, MAX_DELTA_OUTPUT_CLASSES, MAX_EXACT_DELTA_CLASSES,
    MAX_HOST_ACTION_COUNT, MODEL_POINTS, MODEL_REVISION, OPERATION, POLICY,
    POSITIVE_CLASS, RANK_CODE_FEATURES, RAW_RUNTIME_FEATURES, RUN_ID,
    RUNTIME_FEATURES, SOURCE_INPUTS, TRACE, ZERO_CLASS,
    TRAINING_ACCUMULATE_CHUNKS, TRAINING_CHUNK_LEN, TRAINING_EPOCHS,
    TRAINING_LEARNING_RATE, TRAINING_SEED,
    expected_parameter_count, hurdle_statistics_from_counts,
    model_points_description, model_tag, parse_exact_integer,
    positive_count_mode,
)

# This path must remain usable on a CPU-only server that validates a Colab
# archive.  In particular, it exits before importing torch or numpy.
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

TRAINER_SOURCE_PATH = Path(__file__).resolve()
MODEL_CONTRACT_SOURCE_PATH = Path(model_contract_module.__file__).resolve()
THRESHOLD_FREE_POLICY_SOURCE_PATH = (
    ROOT / "formal_NN_training" / "common" / "threshold_free_policy.py"
)

if (COMMON_ADDRESS_BITS, COMMON_CACHE_LINE_BYTES) != (
    ADDRESS_BITS, CACHE_LINE_BYTES
):
    raise RuntimeError("shared address contract differs from v22 contract")


EVENT_LOGGER_SCHEMA = "623_causal_trigger_v5"
CANDIDATE_ATTACHMENT_MODE = "explicit_trigger_event_id"
SOURCE_INPUT_LIST = list(SOURCE_INPUTS)
LINE_MODULUS = 1 << LINE_NUMBER_BITS
LINE_MASK = LINE_MODULUS - 1
SIGNED_LINE_MIN = -(1 << (LINE_NUMBER_BITS - 1))
SIGNED_LINE_MAX = (1 << (LINE_NUMBER_BITS - 1)) - 1


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
    """Load labels without exposing any label field to the runtime encoder."""
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
    """Losslessly encode raw PC64 and current aligned line58 only."""
    pcs = [pc for pc, _, _ in rows]
    lines = [line for _, line, _ in rows]
    encoded = np.concatenate([
        _unsigned_bits(pcs, ADDRESS_BITS),
        _unsigned_bits(lines, LINE_NUMBER_BITS),
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
    hidden = []
    cell = []
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
    """Route each event through only the recurrent state keyed by exact PC."""
    groups = _pc_groups(pcs)
    lengths = [len(indices) for _, indices in groups]
    padded = torch.zeros(
        len(groups), max(lengths), RUNTIME_FEATURES,
        dtype=features.dtype, device=features.device,
    )
    for row, (_, indices) in enumerate(groups):
        positions = torch.as_tensor(indices, dtype=torch.long, device=features.device)
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
        positions = torch.as_tensor(indices, dtype=torch.long, device=features.device)
        context = context.index_copy(0, positions, padded_output[row, :len(indices)])
        state_map[pc] = (final[0][0, row].detach(), final[1][0, row].detach())
    return context


def state_router_sha256():
    payload = (
        inspect.getsource(_pc_groups)
        + inspect.getsource(_initial_state)
        + inspect.getsource(_encode_chunk)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _rank_code(rank, count, device, dtype):
    """Unbounded standard sinusoidal rank code; no learned rank template."""
    positions = torch.full(
        (int(count), 1), float(int(rank) + 1), device=device, dtype=dtype
    )
    scales = torch.pow(
        torch.tensor(10000.0, device=device, dtype=dtype),
        torch.arange(0, RANK_CODE_FEATURES, 2, device=device, dtype=dtype)
        / float(RANK_CODE_FEATURES),
    ).reshape(1, -1)
    angles = positions / scales
    return torch.stack((torch.sin(angles), torch.cos(angles)), dim=2).reshape(
        int(count), RANK_CODE_FEATURES
    )


def _rank_context(model, context, rank):
    code = _rank_code(rank, len(context), context.device, context.dtype)
    return torch.tanh(context + model.rank_projection(code))


class RawHurdleCountStrideLSTM(nn.Module):
    def __init__(
        self, hidden_size, delta_class_prior, hurdle_initial_bias,
        positive_log_count_initial_bias, delta_coordinate_initial_bias,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.delta_output_classes = int(len(delta_class_prior))
        self.other_delta_class = self.delta_output_classes - 1
        if not 2 <= self.delta_output_classes <= MAX_DELTA_OUTPUT_CLASSES:
            raise ValueError("realized delta alphabet must contain 2..256 rows")
        self.input_projection = nn.Linear(RUNTIME_FEATURES, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.rank_projection = nn.Linear(RANK_CODE_FEATURES, hidden_size)
        self.hurdle_head = nn.Linear(hidden_size, 2)
        self.positive_log_count_head = nn.Linear(hidden_size, 1)
        self.delta_class_head = nn.Linear(
            hidden_size, self.delta_output_classes
        )
        # This auxiliary coordinate is trained at every teacher rank.  Decode
        # consults it only when the categorical head chooses OTHER.
        self.delta_coordinate_head = nn.Linear(hidden_size, 1)
        delta_prior = torch.as_tensor(
            delta_class_prior, dtype=self.delta_class_head.bias.dtype
        )
        if len(delta_prior) != self.delta_output_classes or torch.any(delta_prior <= 0):
            raise ValueError("delta prior must cover every output class")
        hurdle_bias = torch.as_tensor(
            hurdle_initial_bias, dtype=self.hurdle_head.bias.dtype
        )
        scalar_biases = (
            float(positive_log_count_initial_bias),
            float(delta_coordinate_initial_bias),
        )
        if (
            tuple(hurdle_bias.shape) != (2,)
            or not torch.all(torch.isfinite(hurdle_bias))
            or not all(math.isfinite(value) for value in scalar_biases)
        ):
            raise ValueError("TRAIN-derived decoder biases must be finite")
        with torch.no_grad():
            self.hurdle_head.bias.copy_(hurdle_bias)
            self.positive_log_count_head.bias.fill_(scalar_biases[0])
            self.delta_class_head.bias.copy_(torch.log(delta_prior))
            self.delta_coordinate_head.bias.fill_(scalar_biases[1])


def _signed_log(value):
    value = int(value)
    return math.copysign(math.log1p(abs(value)), value) if value else 0.0


def _coordinate_to_delta(value):
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError("OTHER delta coordinate is not finite")
    try:
        magnitude = math.expm1(abs(value))
    except OverflowError as exc:
        raise RuntimeError("OTHER delta exceeds the signed line domain") from exc
    if not math.isfinite(magnitude) or magnitude > abs(SIGNED_LINE_MIN):
        raise RuntimeError("OTHER delta exceeds the signed line domain")
    integer = int(math.floor(magnitude + 0.5))
    integer = -integer if value < 0 else integer
    if integer < SIGNED_LINE_MIN or integer > SIGNED_LINE_MAX:
        raise RuntimeError("rounded OTHER delta exceeds the signed line domain")
    return integer


def _teacher_deltas(rows, actions):
    values = []
    for (_, base, _), targets in zip(rows, actions):
        values.extend(_canonical_signed_line_delta(target, base) for target in targets)
    return values


def build_delta_vocabulary(rows, actions):
    frequencies = Counter(_teacher_deltas(rows, actions))
    if not frequencies:
        raise RuntimeError("cannot build an empty delta vocabulary")
    ranked = sorted(frequencies, key=lambda value: (-frequencies[value], value))
    exact = ranked[:MAX_EXACT_DELTA_CLASSES]
    return exact, frequencies


def delta_class_prior(exact_vocabulary, frequencies):
    """Add-one TRAIN prior over realized exact rows and dynamic OTHER."""
    counts = [0] * (len(exact_vocabulary) + 1)
    exact_total = 0
    for index, delta in enumerate(exact_vocabulary):
        count = int(frequencies[int(delta)])
        counts[index] = count
        exact_total += count
    total = int(sum(frequencies.values()))
    counts[-1] = total - exact_total
    denominator = float(total + len(counts))
    return [(count + 1.0) / denominator for count in counts]


def delta_coordinate_initial_bias(rows, actions):
    values = _teacher_deltas(rows, actions)
    if not values:
        raise RuntimeError("delta coordinate initialization requires TRAIN actions")
    bias = sum(_signed_log(value) for value in values) / float(len(values))
    if not math.isfinite(bias):
        raise RuntimeError("non-finite TRAIN delta-coordinate initialization")
    return bias


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


def _chunk_objective(
    model, context, base_lines, actions, exact_vocabulary,
    hurdle_class_weights,
):
    counts_numpy = np.asarray([len(items) for items in actions], dtype=np.int64)
    if model.other_delta_class != len(exact_vocabulary):
        raise RuntimeError("dynamic OTHER row differs from TRAIN vocabulary")
    hurdle_weights = torch.as_tensor(
        hurdle_class_weights,
        dtype=context.dtype, device=context.device,
    )
    if hurdle_weights.shape != (2,) or torch.any(hurdle_weights <= 0):
        raise RuntimeError("hurdle weights must be two positive values")
    components = Counter()
    hurdle_targets = torch.from_numpy(
        (counts_numpy > 0).astype(np.int64)
    ).to(context.device)
    hurdle_sum = F.cross_entropy(
        model.hurdle_head(context), hurdle_targets,
        weight=hurdle_weights, reduction="sum",
    )
    decision_atoms = len(counts_numpy)
    positive_numpy = np.flatnonzero(counts_numpy > 0).astype(np.int64)
    positive_atoms = len(positive_numpy)
    count_sum = context.sum() * 0.0
    if positive_atoms:
        positive = torch.from_numpy(positive_numpy).to(
            device=context.device, dtype=torch.long
        )
        log_count = model.positive_log_count_head(
            context.index_select(0, positive)
        ).squeeze(1)
        target_log_count = torch.as_tensor(
            np.log(counts_numpy[positive_numpy].astype(np.float64)),
            dtype=context.dtype, device=context.device,
        )
        count_sum = F.smooth_l1_loss(
            log_count, target_log_count, reduction="sum"
        )

    delta_class_sum = context.sum() * 0.0
    coordinate_sum = context.sum() * 0.0
    action_atoms = 0
    vocabulary_index = {
        int(delta): index for index, delta in enumerate(exact_vocabulary)
    }
    max_rank = int(counts_numpy.max()) if len(counts_numpy) else 0
    base_lines = np.asarray(base_lines, dtype=np.uint64)
    # The teacher count only schedules which direct target ranks receive loss.
    # Neither the count nor any teacher/predicted earlier action is a rank-head
    # input.  Training and inference use the same generic rank code.
    for rank in range(max_rank):
        active_numpy = np.flatnonzero(counts_numpy > rank).astype(np.int64)
        if not len(active_numpy):
            continue
        active = torch.from_numpy(active_numpy).to(
            device=context.device, dtype=torch.long
        )
        rank_context = _rank_context(model, context.index_select(0, active), rank)
        delta_values = [
            _canonical_signed_line_delta(
                actions[row][rank], int(base_lines[row])
            )
            for row in active_numpy
        ]
        target_classes_numpy = np.asarray([
            vocabulary_index.get(delta, model.other_delta_class)
            for delta in delta_values
        ], dtype=np.int64)
        target_classes = torch.from_numpy(target_classes_numpy).to(context.device)
        logits = model.delta_class_head(rank_context)
        delta_class_sum = delta_class_sum + F.cross_entropy(
            logits, target_classes, reduction="sum"
        )
        action_atoms += len(active_numpy)

        predicted_coordinate = model.delta_coordinate_head(
            rank_context
        ).squeeze(1)
        target_coordinate = torch.as_tensor(
            [_signed_log(delta) for delta in delta_values],
            dtype=context.dtype, device=context.device,
        )
        coordinate_sum = coordinate_sum + F.smooth_l1_loss(
            predicted_coordinate, target_coordinate, reduction="sum"
        )

    if decision_atoms <= 0:
        raise RuntimeError("TRAIN chunk contains no callback decisions")
    # Each task is reduced by its own natural atoms.  Their unit sum has no
    # tuned coefficient and prevents numerous target ranks from drowning the
    # sparse hurdle/count tasks.
    objective = hurdle_sum / float(decision_atoms)
    if positive_atoms:
        objective = objective + count_sum / float(positive_atoms)
    if action_atoms:
        objective = (
            objective
            + delta_class_sum / float(action_atoms)
            + coordinate_sum / float(action_atoms)
        )
    components.update({
        "hurdle_loss_sum": float(hurdle_sum.detach().item()),
        "positive_count_loss_sum": float(count_sum.detach().item()),
        "delta_class_loss_sum": float(delta_class_sum.detach().item()),
        "coordinate_loss_sum": float(coordinate_sum.detach().item()),
        "decision_atoms": decision_atoms,
        "positive_count_atoms": positive_atoms,
        "action_atoms": action_atoms,
        "normalized_objective": float(objective.detach().item()),
        "objective_chunks": 1,
    })
    return objective, components


def decode(
    model, context_numpy, base_lines, exact_vocabulary, device,
    per_callback_watchdog, per_role_watchdog, role, chunk_len=4096,
):
    if len(context_numpy) != len(base_lines):
        raise RuntimeError("decoder row counts differ")
    if model.other_delta_class != len(exact_vocabulary):
        raise RuntimeError("decoder vocabulary differs from dynamic model head")
    if (
        int(per_callback_watchdog) < 1
        or int(per_role_watchdog) < 1
        or int(per_callback_watchdog) > MAX_HOST_ACTION_COUNT
        or int(per_role_watchdog) > MAX_HOST_ACTION_COUNT
    ):
        raise RuntimeError("invalid decoder resource watchdog domain")
    decoded_counts = []
    positive_probability_sum = 0.0
    positive_log_values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(base_lines), chunk_len):
            stop = min(start + chunk_len, len(base_lines))
            context = torch.from_numpy(context_numpy[start:stop]).to(device)
            hurdle_logits = model.hurdle_head(context)
            positive_probability_sum += float(
                torch.softmax(hurdle_logits, dim=1)[:, POSITIVE_CLASS].sum().item()
            )
            decisions = torch.argmax(hurdle_logits, dim=1).cpu().tolist()
            log_counts = model.positive_log_count_head(
                context
            ).squeeze(1).cpu().tolist()
            for local, (decision, log_count) in enumerate(
                zip(decisions, log_counts)
            ):
                if int(decision) == ZERO_CLASS:
                    decoded_counts.append(0)
                    continue
                if int(decision) != POSITIVE_CLASS:
                    raise RuntimeError("hurdle selected an invalid class")
                try:
                    count = positive_count_mode(log_count)
                except ValueError as exc:
                    raise RuntimeError(
                        "{} callback {} positive count is outside the host "
                        "domain: {}".format(role, start + local, exc)
                    ) from exc
                if count > int(per_callback_watchdog):
                    raise RuntimeError(
                        "{} callback {} count {} hit the per-callback resource "
                        "watchdog; no replay will be produced".format(
                            role, start + local, count
                        )
                    )
                decoded_counts.append(count)
                positive_log_values.append(float(log_count))

    decoded_role_actions = sum(decoded_counts)
    if decoded_role_actions > int(per_role_watchdog):
        raise RuntimeError(
            "{} decoded {} actions and hit the per-role resource watchdog; "
            "no replay will be produced".format(role, decoded_role_actions)
        )
    if decoded_role_actions > MAX_HOST_ACTION_COUNT:
        raise RuntimeError("decoded role action total exceeds host integer domain")
    counts = np.asarray(decoded_counts, dtype=np.int64)
    predicted_lines = [[] for _ in base_lines]
    predicted_fills = [[] for _ in base_lines]
    class_counts = Counter()
    with torch.no_grad():
        for start in range(0, len(base_lines), chunk_len):
            stop = min(start + chunk_len, len(base_lines))
            context = torch.from_numpy(context_numpy[start:stop]).to(device)
            local_counts = counts[start:stop]
            steps = int(local_counts.max()) if len(local_counts) else 0
            for rank in range(steps):
                active_numpy = np.flatnonzero(local_counts > rank).astype(np.int64)
                if not len(active_numpy):
                    break
                active = torch.from_numpy(active_numpy).to(
                    device=device, dtype=torch.long
                )
                rank_context = _rank_context(
                    model, context.index_select(0, active), rank
                )
                choices = torch.argmax(
                    model.delta_class_head(rank_context), dim=1
                ).cpu().numpy()
                coordinates = model.delta_coordinate_head(
                    rank_context
                ).squeeze(1).cpu().numpy()
                for local, choice, coordinate in zip(
                    active_numpy, choices, coordinates
                ):
                    choice = int(choice)
                    if choice == model.other_delta_class:
                        delta = _coordinate_to_delta(coordinate)
                        class_counts["OTHER"] += 1
                    elif 0 <= choice < len(exact_vocabulary):
                        delta = int(exact_vocabulary[choice])
                        class_counts[str(choice)] += 1
                    else:
                        raise RuntimeError("decoder selected an invalid delta class")
                    target = apply_signed_line_delta(
                        int(base_lines[start + int(local)]), delta
                    )
                    predicted_lines[start + int(local)].append(int(target))
                    predicted_fills[start + int(local)].append(-1)

    if any(
        len(items) != int(count)
        for items, count in zip(predicted_lines, counts)
    ):
        raise RuntimeError("rank decoder did not realize the learned count exactly")
    diagnostics = {
        "callbacks": len(base_lines),
        "decoded_positive_callbacks": int(np.count_nonzero(counts)),
        "decoded_total_actions": int(counts.sum()),
        "decoded_mean_actions_per_callback": float(counts.mean()),
        "decoded_mean_actions_per_positive_callback": (
            float(counts[counts > 0].mean()) if np.any(counts > 0) else 0.0
        ),
        "decoded_max_actions_per_callback": int(counts.max()) if len(counts) else 0,
        "decoded_count_distribution": {
            str(key): int(value)
            for key, value in sorted(Counter(counts.tolist()).items())
        },
        "hurdle_decision_count": len(base_lines),
        "decoded_zero_callbacks": int(np.count_nonzero(counts == 0)),
        "mean_positive_probability_over_callbacks": (
            positive_probability_sum / max(1, len(base_lines))
        ),
        "minimum_decoded_positive_log_count": (
            min(positive_log_values) if positive_log_values else None
        ),
        "maximum_decoded_positive_log_count": (
            max(positive_log_values) if positive_log_values else None
        ),
        "decoded_delta_class_counts": dict(sorted(class_counts.items())),
        "probability_threshold_applied": False,
        "degree_cap_applied": False,
        "source_action_template_applied": False,
        "separate_global_gate_used": True,
        "separate_count_head_used": True,
        "positive_log_count_used": True,
        "finite_count_mode_realized_exactly": True,
        "host_integer_domain_checked_before_rank_decode": True,
        "per_callback_resource_watchdog": int(per_callback_watchdog),
        "per_role_resource_watchdog": int(per_role_watchdog),
        "resource_watchdog_hit": False,
        "resource_watchdog_behavior": (
            "fail_closed_raise_before_replay_never_truncate_or_change_actions"
        ),
        "resource_watchdog_is_neural_degree_cap": False,
    }
    return counts, predicted_lines, predicted_fills, diagnostics


def trigger_metrics(predicted_counts, target_actions):
    predicted_positive = np.asarray(predicted_counts) > 0
    target_positive = np.asarray(
        [bool(items) for items in target_actions], dtype=np.bool_
    )
    true_positive = int(np.count_nonzero(predicted_positive & target_positive))
    false_positive = int(np.count_nonzero(predicted_positive & ~target_positive))
    false_negative = int(np.count_nonzero(~predicted_positive & target_positive))
    precision = (
        true_positive / float(true_positive + false_positive)
        if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / float(true_positive + false_negative)
        if true_positive + false_negative else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
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


def score_suffix(model, rows, runtime, device, chunk_len, output_start):
    """Rebuild causal state from row zero but retain only the requested suffix."""
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


def _guard_result(
    model, rows, runtime, actions, exact_vocabulary, device, chunk_len,
    per_callback_watchdog, per_role_watchdog,
):
    train_count = len(rows["train"])
    history_rows = rows["train"] + rows["guard"]
    context, diagnostics = score_suffix(
        model, history_rows, runtime, device, chunk_len, train_count
    )
    counts, lines, fills, decoder = decode(
        model, context, [line for _, line, _ in rows["guard"]],
        exact_vocabulary, device, per_callback_watchdog, per_role_watchdog,
        "guard",
    )
    behavior = behavior_metrics(counts, lines, fills, actions["guard"])
    triggers = trigger_metrics(counts, actions["guard"])
    normal_actions = behavior["normal_actions"]
    ratio = (
        behavior["predicted_actions"] / float(normal_actions)
        if normal_actions else 0.0
    )
    result = {}
    result.update(behavior)
    result.update(triggers)
    result["request_ratio_vs_teacher"] = ratio
    result["absolute_request_ratio_error"] = abs(ratio - 1.0)
    result["encoder_diagnostics"] = diagnostics
    result["decoder_diagnostics"] = decoder
    return result


def _selection_key(guard, train_loss, epoch):
    return (
        float(guard["target_f1"]),
        float(guard["trigger_f1"]),
        float(guard["count_exact_match_rate"]),
        -float(guard["absolute_request_ratio_error"]),
        -float(train_loss),
        -int(epoch),
    )


def train_model(
    model, rows, runtime_train, runtime_train_guard, actions,
    exact_vocabulary, hurdle_class_weights, device, epochs,
    chunk_len, accumulate_chunks,
    learning_rate, per_callback_watchdog, per_role_watchdog,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    pcs = np.asarray([pc for pc, _, _ in rows["train"]], dtype=np.uint64)
    history = []
    best_state = None
    best_epoch = None
    best_key = None
    for epoch in range(1, epochs + 1):
        model.train()
        state_map = {}
        totals = Counter()
        optimizer.zero_grad(set_to_none=True)
        pending_chunks = 0
        optimizer_steps = 0
        for start in range(0, len(rows["train"]), chunk_len):
            stop = min(start + chunk_len, len(rows["train"]))
            features = torch.from_numpy(runtime_train[start:stop]).to(
                device=device, dtype=torch.float32
            )
            context = _encode_chunk(
                model, features, pcs[start:stop], state_map
            )
            objective, components = _chunk_objective(
                model, context,
                [line for _, line, _ in rows["train"][start:stop]],
                actions["train"][start:stop], exact_vocabulary,
                hurdle_class_weights,
            )
            if not torch.isfinite(objective):
                raise RuntimeError("non-finite v22 training objective")
            objective.backward()
            pending_chunks += 1
            for key, value in components.items():
                totals[key] += value
            if pending_chunks == accumulate_chunks or stop == len(rows["train"]):
                if pending_chunks <= 0:
                    raise RuntimeError("gradient window contains no chunks")
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(float(pending_chunks))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending_chunks = 0
                optimizer_steps += 1

        train_loss = (
            totals["normalized_objective"] / max(1, totals["objective_chunks"])
        )
        guard = _guard_result(
            model, rows, runtime_train_guard, actions, exact_vocabulary,
            device, chunk_len, per_callback_watchdog, per_role_watchdog,
        )
        selection = _selection_key(guard, train_loss, epoch)
        selected = best_key is None or selection > best_key
        if selected:
            best_key = selection
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        record = {
            "epoch": epoch,
            "mean_unit_sum_objective_per_chunk": train_loss,
            "weighted_hurdle_cross_entropy_per_callback": (
                totals["hurdle_loss_sum"]
                / max(1, totals["decision_atoms"])
            ),
            "positive_log_count_smooth_l1_per_positive_callback": (
                totals["positive_count_loss_sum"]
                / max(1, totals["positive_count_atoms"])
            ),
            "delta_class_cross_entropy": (
                totals["delta_class_loss_sum"]
                / max(1, totals["action_atoms"])
            ),
            "all_rank_signed_log_auxiliary_smooth_l1": (
                totals["coordinate_loss_sum"]
                / max(1, totals["action_atoms"])
            ),
            "decision_atoms": int(totals["decision_atoms"]),
            "positive_count_atoms": int(totals["positive_count_atoms"]),
            "action_atoms": int(totals["action_atoms"]),
            "objective_chunks": int(totals["objective_chunks"]),
            "optimizer_steps": optimizer_steps,
            "observed_pc_states": len(state_map),
            "guard_target_f1": guard["target_f1"],
            "guard_trigger_f1": guard["trigger_f1"],
            "guard_count_exact_match_rate": guard[
                "count_exact_match_rate"
            ],
            "guard_request_ratio_vs_teacher": guard["request_ratio_vs_teacher"],
            "guard_absolute_request_ratio_error": guard[
                "absolute_request_ratio_error"
            ],
            "selected_checkpoint": selected,
        }
        history.append(record)
        print(
            "[train:pc-keyed-hurdle-count-rank-delta] epoch={} loss={:.8f} "
            "guard_target_f1={:.6f} guard_trigger_f1={:.6f} "
            "guard_count_exact={:.6f} request_ratio_error={:.6f} "
            "selected={}".format(
                epoch, record["mean_unit_sum_objective_per_chunk"],
                record["guard_target_f1"], record["guard_trigger_f1"],
                record["guard_count_exact_match_rate"],
                record["guard_absolute_request_ratio_error"], selected,
            ), flush=True,
        )
    if best_state is None:
        raise RuntimeError("guard selection produced no checkpoint")
    model.load_state_dict(best_state)
    return history, best_epoch, best_key


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
    return {
        "rows": len(counts),
        "actions": int(sum(counts)),
        "trigger_rows": int(sum(value > 0 for value in counts)),
        "mean_actions_per_row": float(sum(counts)) / len(counts) if counts else 0.0,
        "count_distribution": {
            str(key): int(value)
            for key, value in sorted(Counter(counts).items())
        },
    }


def self_test_exact_integer_parser():
    examples = {"12": 12, "12.0": 12, "1.2e1": 12, "0xc": 12}
    for text, expected in examples.items():
        if as_int(text) != expected:
            raise RuntimeError("exact integer parser self-test failed")
    for text in ("", "1.25", "nan", "inf"):
        try:
            as_int(text)
        except ValueError:
            continue
        raise RuntimeError("exact integer parser accepted {}".format(text))


def self_test_raw_encoder():
    rows = [
        (1, 10, 0), (2, 20, 0), (3, 30, 0), (2, 21, 0),
        (1, 12, 0), (1, 15, 0),
    ]
    encoded = runtime_features(rows)
    if encoded.shape != (len(rows), RUNTIME_FEATURES):
        raise RuntimeError("raw encoder self-test width failed")
    decoded_pc = sum(
        int(bit) << index
        for index, bit in enumerate(encoded[4, :ADDRESS_BITS])
    )
    decoded_line = sum(
        int(bit) << index
        for index, bit in enumerate(encoded[4, ADDRESS_BITS:])
    )
    if (decoded_pc, decoded_line) != rows[4][:2]:
        raise RuntimeError("raw PC/line lossless round-trip failed")
    changed = list(rows)
    changed[-1] = (1, 99, 0)
    if not np.array_equal(
        encoded[:-1], runtime_features(changed)[:-1]
    ):
        raise RuntimeError("future row changed an earlier raw feature")
    if CAUSAL_RUNTIME_FEATURES != 0 or RUNTIME_FEATURES != 122:
        raise RuntimeError("engineered runtime features re-entered v22")


def self_test_vocabulary():
    rows = [(1, 100, 0), (1, 100, 1), (1, 100, 2)]
    actions = [[101, 99], [101], [102]]
    vocabulary, frequencies = build_delta_vocabulary(rows, actions)
    if vocabulary[:3] != [1, -1, 2] or frequencies[1] != 2:
        raise RuntimeError("train-frequency vocabulary ordering changed")
    prior = delta_class_prior(vocabulary, frequencies)
    if len(prior) != len(vocabulary) + 1:
        raise RuntimeError("dynamic OTHER class is not last")
    hurdle = hurdle_statistics_from_counts([0, 0, 0, 2, 1])
    if not math.isclose(
        hurdle["weighted_zero_mass"], hurdle["weighted_positive_mass"]
    ):
        raise RuntimeError("ZERO/POSITIVE TRAIN mass is not balanced")
    for value in (-100000, -1, 0, 1, 100000):
        if _coordinate_to_delta(_signed_log(value)) != value:
            raise RuntimeError("signed-log OTHER round-trip failed")


def self_test_parameter_count(hidden_size):
    realized_classes = 4
    delta_prior = [1.0 / realized_classes] * realized_classes
    model = RawHurdleCountStrideLSTM(
        hidden_size, delta_prior, [0.0, 0.0], math.log(2.0), 0.25
    )
    observed = sum(parameter.numel() for parameter in model.parameters())
    expected = expected_parameter_count(hidden_size, realized_classes)
    if observed != expected:
        raise RuntimeError(
            "parameter formula mismatch: {} != {}".format(
                observed, expected
            )
        )
    if observed > expected_parameter_count(hidden_size):
        raise RuntimeError("realized parameters exceed contract maximum")
    if torch.count_nonzero(model.hurdle_head.bias.detach()).item() != 0:
        raise RuntimeError("balanced hurdle bias is not neutral")
    if not torch.allclose(
        model.positive_log_count_head.bias.detach(),
        torch.tensor([math.log(2.0)]),
    ):
        raise RuntimeError("positive log-count bias initialization failed")
    if not torch.allclose(
        model.delta_coordinate_head.bias.detach(), torch.tensor([0.25])
    ):
        raise RuntimeError("delta coordinate bias initialization failed")
    if not torch.allclose(
        model.delta_class_head.bias.detach(),
        torch.log(torch.tensor(delta_prior)),
    ):
        raise RuntimeError("delta class prior bias initialization failed")


def self_test_rank_independence(hidden_size):
    delta_prior = [0.25] * 4
    model = RawHurdleCountStrideLSTM(
        hidden_size, delta_prior, [0.0, 0.0], 0.0, 0.0
    ).eval()
    context = torch.randn(3, hidden_size)
    first = _rank_context(model, context, 1)
    second = _rank_context(model, context, 1)
    if not torch.equal(first, second):
        raise RuntimeError("rank decoder is not deterministic")
    signature = inspect.signature(_rank_context)
    if list(signature.parameters) != ["model", "context", "rank"]:
        raise RuntimeError("rank decoder acquired action feedback")
    for forbidden in ("action_cell", "decoder_rnn", "previous_action"):
        if hasattr(model, forbidden):
            raise RuntimeError("rank decoder acquired action feedback state")


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
        "--decode-per-callback-watchdog", type=int,
        default=DECODE_PER_CALLBACK_WATCHDOG,
    )
    parser.add_argument(
        "--decode-per-role-watchdog", type=int,
        default=DECODE_PER_ROLE_WATCHDOG,
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    return parser


def main():
    args = build_parser().parse_args()
    source_hashes = {
        "trainer_source_sha256": sha256(TRAINER_SOURCE_PATH),
        "model_contract_source_sha256": sha256(MODEL_CONTRACT_SOURCE_PATH),
        "threshold_free_policy_source_sha256": sha256(
            THRESHOLD_FREE_POLICY_SOURCE_PATH
        ),
    }
    expected_pair = MODEL_POINTS["lstm"].get(args.model_size)
    if expected_pair is None or args.pair_id != expected_pair:
        raise RuntimeError("model size/pair is not a configured v22 LSTM point")
    pinned_training = {
        "seed": TRAINING_SEED,
        "epochs": TRAINING_EPOCHS,
        "chunk_len": TRAINING_CHUNK_LEN,
        "accumulate_chunks": TRAINING_ACCUMULATE_CHUNKS,
        "learning_rate": TRAINING_LEARNING_RATE,
    }
    observed_training = {
        "seed": args.seed,
        "epochs": args.epochs,
        "chunk_len": args.chunk_len,
        "accumulate_chunks": args.accumulate_chunks,
        "learning_rate": args.learning_rate,
    }
    if observed_training != pinned_training:
        raise RuntimeError(
            "RUN_ID pins training config: observed={} expected={}".format(
                observed_training, pinned_training
            )
        )
    if (
        args.epochs < 1 or args.chunk_len < 1 or args.accumulate_chunks < 1
        or args.learning_rate <= 0 or args.decode_per_callback_watchdog < 1
        or args.decode_per_role_watchdog < 1
        or args.decode_per_callback_watchdog > MAX_HOST_ACTION_COUNT
        or args.decode_per_role_watchdog > MAX_HOST_ACTION_COUNT
    ):
        raise RuntimeError("model/training/resource dimensions are out of domain")

    self_test_exact_integer_parser()
    self_test_raw_encoder()
    self_test_vocabulary()
    self_test_parameter_count(args.model_size)
    self_test_rank_independence(args.model_size)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not hasattr(torch, "set_float32_matmul_precision"):
        raise RuntimeError("pinned v22 requires torch matmul precision control")
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
            "the pinned v22 run requires an A100 CUDA device; observed {}"
            .format(device_name)
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    roles = ("train", "guard", "eval")
    stream_paths = {role: getattr(args, role + "_stream") for role in roles}
    action_paths = {role: getattr(args, role + "_candidates") for role in roles}
    rows = {role: load_stream(stream_paths[role]) for role in roles}
    actions = {
        role: load_teacher_actions(action_paths[role], rows[role])
        for role in roles
    }

    exact_vocabulary, train_delta_frequencies = build_delta_vocabulary(
        rows["train"], actions["train"]
    )
    counts_train = np.asarray(
        [len(items) for items in actions["train"]], dtype=np.int64
    )
    hurdle_stats = hurdle_statistics_from_counts(counts_train.tolist())
    hurdle_class_weights = hurdle_stats[
        "class_weights_ZERO_POSITIVE"
    ]
    delta_prior = delta_class_prior(
        exact_vocabulary, train_delta_frequencies
    )
    delta_initial_bias = [math.log(value) for value in delta_prior]
    coordinate_initial_bias = delta_coordinate_initial_bias(
        rows["train"], actions["train"]
    )

    runtime_train = runtime_features(rows["train"])
    train_guard_rows = rows["train"] + rows["guard"]
    runtime_train_guard = runtime_features(train_guard_rows)
    if not np.array_equal(
        runtime_train_guard[:len(rows["train"])], runtime_train
    ):
        raise RuntimeError("longer chronology changed the train feature prefix")

    model = RawHurdleCountStrideLSTM(
        args.model_size,
        delta_prior,
        hurdle_stats["hurdle_initial_bias_ZERO_POSITIVE"],
        hurdle_stats["positive_log_count_initial_bias"],
        coordinate_initial_bias,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    realized_delta_output_classes = len(exact_vocabulary) + 1
    expected_parameters = expected_parameter_count(
        args.model_size, realized_delta_output_classes
    )
    maximum_parameters = expected_parameter_count(args.model_size)
    if parameter_count != expected_parameters:
        raise RuntimeError("v22 realized parameter count changed")
    if parameter_count > maximum_parameters:
        raise RuntimeError("v22 realized parameters exceed contract maximum")
    history, best_epoch, best_selection_key = train_model(
        model, rows, runtime_train, runtime_train_guard, actions,
        exact_vocabulary, hurdle_class_weights, device, args.epochs,
        args.chunk_len, args.accumulate_chunks, args.learning_rate,
        args.decode_per_callback_watchdog, args.decode_per_role_watchdog,
    )
    selected_train_loss = float(
        history[best_epoch - 1]["mean_unit_sum_objective_per_chunk"]
    )
    del runtime_train

    # Evaluation is decoded exactly once, after guard-only checkpoint choice.
    complete_rows = train_guard_rows + rows["eval"]
    complete_runtime = runtime_features(complete_rows)
    if not np.array_equal(
        complete_runtime[:len(train_guard_rows)], runtime_train_guard
    ):
        raise RuntimeError("evaluation chronology changed an earlier feature")
    del runtime_train_guard
    eval_start = len(train_guard_rows)
    eval_context, encoder_diagnostics = score_suffix(
        model, complete_rows, complete_runtime, device, args.chunk_len, eval_start
    )
    del complete_runtime
    predicted_counts, predicted_lines, predicted_fills, decoder_diagnostics = decode(
        model, eval_context, [line for _, line, _ in rows["eval"]],
        exact_vocabulary, device, args.decode_per_callback_watchdog,
        args.decode_per_role_watchdog, "eval",
    )
    heldout = behavior_metrics(
        predicted_counts, predicted_lines, predicted_fills, actions["eval"]
    )
    heldout.update(trigger_metrics(predicted_counts, actions["eval"]))
    heldout["request_ratio_vs_teacher"] = (
        heldout["predicted_actions"] / float(heldout["normal_actions"])
        if heldout["normal_actions"] else 0.0
    )

    normal_path = args.out_dir / "offline_stride.replay.csv"
    nn_path = args.out_dir / "offline_nn.replay.csv"
    normal_entries, normal_triggers = write_replay(
        normal_path, rows["eval"], actions["eval"]
    )
    nn_entries, nn_triggers = write_replay(
        nn_path, rows["eval"], predicted_lines
    )
    write_table(args.out_dir / "training_history.csv", history)

    tag = model_tag(args.model_size)
    checkpoint = {
        "state_dict": model.state_dict(),
        "run_id": RUN_ID,
        "operation": OPERATION,
        "model_family": "lstm",
        "model_size": args.model_size,
        "exact_delta_vocabulary": [int(value) for value in exact_vocabulary],
        "realized_exact_delta_classes": len(exact_vocabulary),
        "realized_delta_output_classes": realized_delta_output_classes,
        "other_delta_class": model.other_delta_class,
        "hurdle_class_weights_ZERO_POSITIVE": hurdle_class_weights,
        "hurdle_class_indices": {
            "ZERO": ZERO_CLASS, "POSITIVE": POSITIVE_CLASS
        },
        "hurdle_training_statistics": hurdle_stats,
        "positive_log_count_initial_bias": hurdle_stats[
            "positive_log_count_initial_bias"
        ],
        "delta_class_empirical_prior": delta_prior,
        "delta_coordinate_initial_bias": coordinate_initial_bias,
        "realized_parameter_count": parameter_count,
        "expected_realized_parameter_count": expected_parameters,
        "maximum_parameter_count": maximum_parameters,
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": MODEL_REVISION,
        "decoder_revision": DECODER_REVISION,
        "selected_guard_epoch": best_epoch,
        "selected_train_objective_loss": selected_train_loss,
        "training_config": dict(model_points_description()["training_config"]),
        "trainer_source_sha256": source_hashes["trainer_source_sha256"],
        "model_contract_source_sha256": source_hashes[
            "model_contract_source_sha256"
        ],
        "threshold_free_policy_source_sha256": source_hashes[
            "threshold_free_policy_source_sha256"
        ],
    }
    torch.save(checkpoint, args.out_dir / "model.pt")

    encoder_hash = runtime_encoder_sha256()
    router_hash = state_router_sha256()
    role_vocabulary_stats = {
        role: vocabulary_statistics(rows[role], actions[role], exact_vocabulary)
        for role in roles
    }
    train_unique_pc_count = len({pc for pc, _, _ in rows["train"]})
    complete_unique_pc_count = len({pc for pc, _, _ in complete_rows})
    metadata = {
        "run_id": RUN_ID,
        "operation": OPERATION,
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
        "expected_parameter_count": expected_parameters,
        "expected_realized_parameter_count": expected_parameters,
        "maximum_parameter_count": maximum_parameters,
        "realized_parameter_count_matches_formula": True,
        "realized_parameter_count_within_maximum": True,
        "parameter_formula": model_points_description()["parameter_formula"],
        "model_point_contract": model_points_description(),
        "parameter_bytes_float32": parameter_count * 4,
        "maximum_parameter_bytes_float32": maximum_parameters * 4,
        "seed": args.seed,
        "epochs": args.epochs,
        "chunk_len": args.chunk_len,
        "accumulate_chunks": args.accumulate_chunks,
        "learning_rate": args.learning_rate,
        "training_device": str(device),
        "training_device_name": device_name,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "training_config": dict(model_points_description()["training_config"]),
        "training_config_pinned_by_run_id": True,
        "trainer_source_sha256": source_hashes["trainer_source_sha256"],
        "model_contract_source_sha256": source_hashes[
            "model_contract_source_sha256"
        ],
        "threshold_free_policy_source_sha256": source_hashes[
            "threshold_free_policy_source_sha256"
        ],
        "source_decision_effective_external_input": SOURCE_INPUT_LIST,
        "same_external_input_contract": True,
        "training_runtime_fields": SOURCE_INPUT_LIST,
        "inference_runtime_fields": SOURCE_INPUT_LIST,
        "training_inference_input_encoder_identical": True,
        "runtime_feature_count": RUNTIME_FEATURES,
        "raw_runtime_feature_count": RAW_RUNTIME_FEATURES,
        "causal_runtime_feature_count": CAUSAL_RUNTIME_FEATURES,
        "runtime_feature_breakdown": model_points_description()[
            "runtime_feature_breakdown"
        ],
        "runtime_encoding": "lossless_raw_pc64_plus_line58_only",
        "causal_derived_features_from_same_external_input": [],
        "engineered_runtime_features": [],
        "raw_runtime_input_only": True,
        "derived_features_use_teacher_or_future": False,
        "runtime_encoder_sha256": encoder_hash,
        "training_runtime_encoder_sha256": encoder_hash,
        "inference_runtime_encoder_sha256": encoder_hash,
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "teacher_actions_are_model_inputs": False,
        "normal_policy_request_rate_used_as_budget": False,
        "normal_policy_constants_used_by_neural_inference": False,
        "normal_policy_templates_used_by_neural_inference": False,
        "probability_threshold_used": False,
        "threshold_related_hardcodes_used": False,
        "inference_policy_hardcodes_used": False,
        "neural_degree_cap": None,
        "fixed_page_offset_classes": None,
        "same_page_rule_used_by_neural_inference": False,
        "future_label_window_used": False,
        "manual_loss_weights_used": False,
        "data_derived_hurdle_class_weights_used": True,
        "training_regularization_used": False,
        "nn_generates_own_target_addresses": True,
        "full_signed_line_delta_range_reachable": False,
        "every_signed_line_delta_exactly_representable": False,
        "exact_delta_representability_scope": "train_vocabulary_only",
        "all_deltas_relative_to_current_demand": True,
        "delta_target_origin": "current_demand_line_at_same_callback",
        "delta_vocabulary_source": "train_labels_only",
        "delta_vocabulary_order": "descending_train_frequency_then_signed_integer",
        "delta_vocabulary_max_exact": MAX_EXACT_DELTA_CLASSES,
        "delta_vocabulary_exact": [int(value) for value in exact_vocabulary],
        "delta_vocabulary_exact_size": len(exact_vocabulary),
        "realized_exact_delta_classes": len(exact_vocabulary),
        "realized_delta_output_classes": realized_delta_output_classes,
        "maximum_delta_output_classes": MAX_DELTA_OUTPUT_CLASSES,
        "delta_vocabulary_train_frequencies": [
            int(train_delta_frequencies[value]) for value in exact_vocabulary
        ],
        "delta_class_prior_smoothing": (
            "add_one_over_realized_exact_plus_OTHER_output_classes"
        ),
        "delta_class_empirical_prior": delta_prior,
        "delta_class_initial_bias": delta_initial_bias,
        "delta_class_bias_initialization": (
            "log_add_one_smoothed_TRAIN_exact_plus_OTHER_frequency"
        ),
        "delta_coordinate_bias_initialization": (
            "mean_TRAIN_signed_log_delta"
        ),
        "delta_coordinate_initial_bias": coordinate_initial_bias,
        "delta_other_class": model.other_delta_class,
        "delta_other_escape": "signed_log_continuous_bounded_approximation",
        "delta_other_decode_precision": (
            "rounded_float32_approximate_except_exact_vocabulary"
        ),
        "delta_coordinate_auxiliary_trained_on_all_teacher_actions": True,
        "delta_coordinate_used_for_decode_only_on_other": True,
        "delta_vocabulary_statistics": role_vocabulary_stats,
        "decoder_training_mode": DECODER_TRAINING_MODE,
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_previous_predicted_action_used_as_input": False,
        "decoder_previous_sampled_action_used_as_input": False,
        "decoder_rank_conditioning": "fixed_generic_sinusoidal_code",
        "decoder_rank_code_features": RANK_CODE_FEATURES,
        "all_teacher_ranks_supervised": True,
        "terminal_stop_supervised_for_every_teacher_sequence": False,
        "hurdle_training_objective": HURDLE_OBJECTIVE,
        "hurdle_classes": ["ZERO", "POSITIVE"],
        "hurdle_class_indices": {
            "ZERO": ZERO_CLASS, "POSITIVE": POSITIVE_CLASS
        },
        "hurdle_class_weighting": (
            "TRAIN_inverse_frequency_N_over_2N_class"
        ),
        "hurdle_equal_aggregate_train_mass": True,
        "hurdle_class_weights_ZERO_POSITIVE": hurdle_class_weights,
        "hurdle_training_statistics": hurdle_stats,
        "hurdle_bias_initialization": (
            "centered_log_effective_weighted_TRAIN_class_mass"
        ),
        "hurdle_initial_bias_ZERO_POSITIVE": hurdle_stats[
            "hurdle_initial_bias_ZERO_POSITIVE"
        ],
        "hurdle_decoding_rule": "deterministic_raw_two_class_argmax",
        "separate_global_gate_used": True,
        "separate_count_head_used": True,
        "log_count_used": True,
        "positive_count_training_objective": COUNT_OBJECTIVE,
        "positive_log_count_bias_initialization": (
            "mean_log_positive_TRAIN_count"
        ),
        "positive_log_count_initial_bias": hurdle_stats[
            "positive_log_count_initial_bias"
        ],
        "positive_count_support": "mathematically_unbounded_positive_integers",
        "positive_count_host_behavior": "fail_closed_no_clip_or_wrap",
        "positive_count_decoding_rule": "max_1_round_exp_log_count",
        "decoded_count_definition": (
            "zero_on_hurdle_ZERO_else_finite_mode_of_positive_log_count"
        ),
        "teacher_sequence_training_label_statistics": {
            "teacher_sequences": int(len(counts_train)),
            "teacher_actions": int(counts_train.sum()),
            "maximum_teacher_count": int(counts_train.max()),
            "count_distribution": {
                str(key): int(value)
                for key, value in sorted(Counter(counts_train.tolist()).items())
            },
        },
        "poisson_objective_used": False,
        "poisson_decoder_used": False,
        "delta_training_objective": DELTA_OBJECTIVE,
        "delta_decoding_rule": (
            "rank_conditioned_exact_class_MAP_or_rounded_signed_log_OTHER"
        ),
        "delta_mixture_components": 0,
        "gmm_objective_used": False,
        "gmm_decoder_used": False,
        "decision_rule": DECODING_RULE,
        "deterministic_decoding": True,
        "deterministic_decoding_reproducible": True,
        "stochastic_decoding": False,
        "decoder_sampling_roles": [],
        "decoder_train_sampling_performed": False,
        "decoder_guard_sampling_performed": False,
        "decoder_eval_sampling_performed": False,
        "sampled_outputs_used_as_decoder_feedback": False,
        "decode_per_callback_resource_watchdog": (
            args.decode_per_callback_watchdog
        ),
        "decode_per_role_resource_watchdog": args.decode_per_role_watchdog,
        "maximum_host_action_count": MAX_HOST_ACTION_COUNT,
        "decode_resource_watchdog_behavior": (
            "fail_closed_raise_before_replay_never_truncate_or_change_actions"
        ),
        "decode_resource_watchdog_is_neural_degree_cap": False,
        "successful_run_hit_decode_resource_watchdog": False,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "checkpoint_selection_roles": [
            "guard_metrics", "TRAIN_loss_tiebreak_only"
        ],
        "checkpoint_selection_primary_role": "guard",
        "training_loss_used_as_lexicographic_tiebreak": True,
        "guard_selection_composite_or_mean_used": False,
        "checkpoint_selection_metrics": [
            "maximize_target_f1", "maximize_trigger_f1",
            "maximize_count_exact_match_rate",
            "minimize_absolute_request_ratio_error", "minimize_train_loss",
            "prefer_earlier_epoch",
        ],
        "selected_guard_epoch": best_epoch,
        "selected_train_objective_loss": selected_train_loss,
        "selected_guard_key": [float(value) for value in best_selection_key],
        "guard_role": "checkpoint_selection_only_no_threshold_calibration",
        "evaluation_used_for_checkpoint_selection": False,
        "evaluation_decode_passes": 1,
        "training_chunks_shuffled": False,
        "training_state_mode": "exact_pc_keyed_stateful_tbptt",
        "training_state_carried_across_chunks": True,
        "training_state_detached_between_chunks": True,
        "training_state_reset": "only_at_epoch_start",
        "inference_history_mode": (
            "fresh_exact_PC_state_then_complete_train_guard_eval_chronology"
        ),
        "training_state_routing": "one_lstm_state_per_exact_observed_PC",
        "inference_state_routing": "one_lstm_state_per_exact_observed_PC",
        "standard_lstm_forget_gates_learn_staleness": True,
        "train_unique_pc_count": train_unique_pc_count,
        "history_unique_pc_count": complete_unique_pc_count,
        "local_recurrent_state_bytes_per_observed_pc_float32": (
            2 * args.model_size * 4
        ),
        "peak_training_recurrent_state_bytes_float32": (
            train_unique_pc_count * 2 * args.model_size * 4
        ),
        "peak_inference_recurrent_state_bytes_float32": (
            complete_unique_pc_count * 2 * args.model_size * 4
        ),
        "peak_persistent_recurrent_state_bytes": (
            complete_unique_pc_count * 2 * args.model_size * 4
        ),
        "training_state_router_sha256": router_hash,
        "inference_state_router_sha256": router_hash,
        "raw_pc64_line58_lossless_self_test": "PASS",
        "engineered_runtime_features_absent_self_test": "PASS",
        "exact_pc_state_routing_self_test": "PASS",
        "hurdle_equal_mass_self_test": "PASS",
        "data_derived_stable_bias_initialization_self_test": "PASS",
        "finite_positive_count_mode_self_test": "PASS",
        "host_domain_count_rejection_self_test": "PASS",
        "separate_hurdle_and_count_heads_self_test": "PASS",
        "terminal_stop_supervision_self_test": "NOT_APPLICABLE",
        "delta_class_prior_bias_initialization_self_test": "PASS",
        "train_only_delta_vocabulary_self_test": "PASS",
        "rank_no_action_feedback_self_test": "PASS",
        "signed_log_other_escape_self_test": "PASS",
        "exact_integer_parser_self_test": "PASS",
        "dynamic_realized_parameter_self_test": "PASS",
        "cnn_architecture_self_test": "NOT_APPLICABLE",
        "cnn_temporal_layers": 0,
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
        "hurdle_count_decoder_diagnostics": decoder_diagnostics,
        "encoder_diagnostics": encoder_diagnostics,
        "train_action_summary": _count_summary(actions["train"]),
        "guard_action_summary": _count_summary(actions["guard"]),
        "eval_action_summary": _count_summary(actions["eval"]),
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
        "selected_guard_epoch": best_epoch,
        "decision_rule": metadata["decision_rule"],
        "offline_normal_entries": normal_entries,
        "offline_nn_entries": nn_entries,
    }, indent=2))


if __name__ == "__main__":
    main()
