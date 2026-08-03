#!/usr/bin/env python3
"""PC-keyed, source-input-fair LSTM student for 623 Stride.

Training and inference receive exactly the source-visible PC and cache-line
address.  Teacher requests supervise the model but are never inference inputs.

This v16 revision applies the 602-proven deterministic heads to the 623 input
and state organization without adding capacity, a threshold, a degree limit,
a candidate table, a page rule, or normal-policy state:

* recurrent state is dynamically routed by the observed PC, with no fixed
  tracker capacity;
* a data-derived frequency-balanced two-class hurdle prevents sparse-trigger
  collapse without a manually selected probability cutoff;
* a deterministic rounded positive log-count preserves variable, unbounded
  count support while recovering the teacher's nearly always-two degree;
* one shared single-layer LSTM supplies both decision and action context;
* a deterministic scalar signed-log delta head removes the v15 mismatch where
  scale was trained but discarded and a sampled component mean was replayed;
* the lossless input encoder removes the six address-offset bits that are
  identically zero for aligned cache lines: PC64 + line-number58 = 122 bits.
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
from collections import Counter, OrderedDict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formal_NN_training.common.threshold_free_policy import (
    ADDRESS_BITS, CACHE_LINE_BYTES, apply_signed_line_delta, behavior_metrics,
    targets_from_actions,
)
TRACE = "623.xalancbmk_s-700B"
POLICY = "stride"
EXPERIMENT_REVISION = "stride_source_input_variable_delta_free_running_v9"
MODEL_REVISION = "compact_pc_keyed_balanced_deterministic_scalar_v16"
EVENT_LOGGER_SCHEMA = "623_causal_trigger_v5"
CANDIDATE_ATTACHMENT_MODE = "explicit_trigger_event_id"
SOURCE_INPUTS = ["pc", "addr"]
MODEL_POINTS = {
    "lstm": {8: "p0", 16: "p1", 32: "p2", 64: "p3", 128: "p4"},
}
CACHE_LINE_OFFSET_BITS = int(math.log(CACHE_LINE_BYTES, 2))
LINE_NUMBER_BITS = ADDRESS_BITS - CACHE_LINE_OFFSET_BITS
RUNTIME_FEATURES = ADDRESS_BITS + LINE_NUMBER_BITS
EXPECTED_PARAMETER_COUNTS = {
    8: 1860, 16: 5124, 32: 15876, 64: 54276, 128: 198660,
}


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
    text = str(value).strip()
    return int(text, 16) if text.lower().startswith("0x") else int(float(text))


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
            ):
                raise RuntimeError(
                    "stream identity/ordering failure at row {}".format(index)
                )
            rows.append((pc, line, occurrence))
    if not rows:
        raise RuntimeError("empty stream: {}".format(path))
    return rows


def load_teacher_actions(path, rows):
    """Audit captured Stride actions as labels, never neural inputs."""
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
            if (
                pf_event <= last_pf_event or trigger >= pf_event
                or distance != pf_event - trigger
            ):
                raise RuntimeError(
                    "invalid trigger transport at {}".format(index)
                )
            if (
                as_int(row["fill_level"]) != 2
                or as_int(row["accepted"]) not in (0, 1)
                or as_int(row["duplicate"]) not in (0, 1)
            ):
                raise RuntimeError(
                    "invalid captured Stride action at {}".format(index)
                )
            actions[index].append(as_int(row["pf_line"]))
            last_pf_event = pf_event
    if not any(actions):
        raise RuntimeError("empty Stride teacher action stream {}".format(path))
    return actions


def _unsigned_bits(values, width):
    """Encode bounded unsigned integers as deterministic LSB-first bits."""
    integers = [int(value) for value in values]
    limit = 1 << int(width)
    if any(value < 0 or value >= limit for value in integers):
        raise RuntimeError("runtime integer exceeds its lossless bit width")
    array = np.asarray(integers, dtype=np.uint64)
    shifts = np.arange(width, dtype=np.uint64)
    return (
        (array[:, None] >> shifts[None, :]) & np.uint64(1)
    ).astype(np.float32)


def runtime_features(rows):
    """Losslessly encode PC64 and aligned-address line-number58."""
    if CACHE_LINE_BYTES != (1 << CACHE_LINE_OFFSET_BITS):
        raise RuntimeError("cache-line bytes must be a power of two")
    return np.concatenate([
        _unsigned_bits([pc for pc, _, _ in rows], ADDRESS_BITS),
        _unsigned_bits([line for _, line, _ in rows], LINE_NUMBER_BITS),
    ], axis=1)


def runtime_encoder_sha256():
    payload = {
        "entrypoint_source": inspect.getsource(runtime_features),
        "primitive_source": inspect.getsource(_unsigned_bits),
        "fields": SOURCE_INPUTS,
        "use_pc": True,
        "address_bits": ADDRESS_BITS,
        "cache_line_bytes": CACHE_LINE_BYTES,
        "cache_line_offset_bits_removed": CACHE_LINE_OFFSET_BITS,
        "line_number_bits": LINE_NUMBER_BITS,
        "feature_count": RUNTIME_FEATURES,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def _group_indices_by_pc(rows):
    """Return causal same-PC subsequences in first-observation order."""
    grouped = OrderedDict()
    for index, (pc, _, _) in enumerate(rows):
        grouped.setdefault(pc, []).append(index)
    return grouped


def _chunk_batches(grouped, chunk_len, pc_batch_size):
    """Yield deterministic, per-PC chronological TBPTT batches."""
    positions = {pc: 0 for pc in grouped}
    pending = deque(grouped.keys())
    while pending:
        take = min(pc_batch_size, len(pending))
        selected = [pending.popleft() for _ in range(take)]
        batch = []
        requeue = []
        for pc in selected:
            start = positions[pc]
            stop = min(start + chunk_len, len(grouped[pc]))
            indices = grouped[pc][start:stop]
            positions[pc] = stop
            batch.append((pc, indices))
            if stop < len(grouped[pc]):
                requeue.append(pc)
        pending.extend(requeue)
        # PackedSequence is explicit about order; independent PC streams may be
        # length-sorted without changing any causal dependency.
        batch.sort(key=lambda item: (-len(item[1]), item[1][0]))
        yield batch


def state_router_sha256():
    payload = (
        inspect.getsource(_group_indices_by_pc)
        + inspect.getsource(_chunk_batches)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _delta_to_integer(coordinate):
    coordinate = float(coordinate)
    if not math.isfinite(coordinate):
        raise RuntimeError("neural delta coordinate is not finite")
    try:
        magnitude = math.expm1(abs(coordinate))
    except OverflowError as exc:
        raise RuntimeError("neural delta exceeds address domain") from exc
    if not math.isfinite(magnitude):
        raise RuntimeError("neural delta exceeds address domain")
    integer = int(round(magnitude))
    return -integer if coordinate < 0 else integer


def _positive_counts_from_log_mean(log_means):
    """Decode a learned positive count without a policy degree cap."""
    values = np.asarray(log_means, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise RuntimeError("positive-count prediction is not a finite vector")
    limit = math.log(float(np.iinfo(np.int64).max))
    if np.any(values > limit):
        raise RuntimeError("positive-count prediction exceeds host domain")
    counts = np.rint(np.exp(values)).astype(np.int64)
    counts[counts < 1] = 1
    return counts


def _data_derived_gate_class_weights(counts):
    """Give the two observed hurdle classes equal aggregate loss mass."""
    labels = (np.asarray(counts, dtype=np.int64) > 0).astype(np.int64)
    frequencies = np.bincount(labels, minlength=2).astype(np.float64)
    if np.any(frequencies == 0):
        raise RuntimeError(
            "hurdle training requires observed zero and positive rows"
        )
    weights = float(len(labels)) / (2.0 * frequencies)
    if not np.all(np.isfinite(weights)):
        raise RuntimeError("non-finite data-derived hurdle weights")
    return weights.astype(np.float32)


class CompactDirectDeltaDecoder(nn.Module):
    """Lightweight deterministic free-running signed-delta regressor."""

    def __init__(self, hidden_size):
        super().__init__()
        if hidden_size < 1:
            raise ValueError("decoder hidden size must be positive")
        self.action_cell = nn.GRUCell(1, hidden_size)
        self.delta_head = nn.Linear(hidden_size, 1)

    def begin(self, context):
        return context

    def coordinate(self, state):
        return self.delta_head(state).squeeze(-1)

    def advance(self, state, predicted_coordinate):
        return self.action_cell(predicted_coordinate.reshape(-1, 1), state)


class CompactPCKeyedDeterministicStrideLSTM(nn.Module):
    """One shared single-layer PC-keyed LSTM with lightweight heads."""

    def __init__(self, feature_count, hidden_size):
        super().__init__()
        self.feature_count = feature_count
        self.hidden_size = hidden_size
        self.input_projection = nn.Linear(feature_count, hidden_size)
        self.encoder_lstm = nn.LSTM(
            hidden_size, hidden_size, batch_first=True
        )
        self.emit_head = nn.Linear(hidden_size, 2)
        self.log_positive_count = nn.Linear(hidden_size, 1)
        self.action_decoder = CompactDirectDeltaDecoder(hidden_size)


def _initial_state(state_map, keys, hidden_size, device):
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


def _encode_shared(model, padded, lengths, state_map, keys):
    initial = _initial_state(
        state_map, keys, model.hidden_size, padded.device
    )
    projected = torch.tanh(model.input_projection(padded))
    packed = pack_padded_sequence(
        projected, lengths, batch_first=True, enforce_sorted=True
    )
    packed_output, final = model.encoder_lstm(packed, initial)
    output, _ = pad_packed_sequence(
        packed_output, batch_first=True, total_length=padded.shape[1]
    )
    for position, key in enumerate(keys):
        state_map[key] = (
            final[0][0, position].detach(),
            final[1][0, position].detach(),
        )
    return output


def _make_padded_batch(runtime, counts, deltas, batch, device):
    lengths = [len(indices) for _, indices in batch]
    width = deltas.shape[1]
    padded_runtime = torch.zeros(
        len(batch), max(lengths), runtime.shape[1],
        dtype=torch.float32, device=device,
    )
    padded_counts = torch.full(
        (len(batch), max(lengths)), -1,
        dtype=torch.long, device=device,
    )
    padded_deltas = torch.zeros(
        len(batch), max(lengths), width,
        dtype=torch.float32, device=device,
    )
    for position, (_, indices) in enumerate(batch):
        length = len(indices)
        padded_runtime[position, :length] = torch.from_numpy(
            runtime[indices]
        ).to(device)
        padded_counts[position, :length] = torch.from_numpy(
            counts[indices]
        ).to(device)
        if width:
            padded_deltas[position, :length] = torch.from_numpy(
                deltas[indices]
            ).to(device)
    return padded_runtime, padded_counts, padded_deltas, lengths


def _compact_balanced_deterministic_loss(
    model, context, counts, deltas, gate_class_weights,
):
    flat_context = context.reshape(-1, context.shape[-1])
    flat_counts = counts.reshape(-1)
    flat_deltas = deltas.reshape(-1, deltas.shape[-1])
    valid = flat_counts >= 0
    decision_atoms = int(valid.sum().detach().item())
    if not decision_atoms:
        raise RuntimeError("training batch has no valid decision")

    context = flat_context[valid]
    targets = flat_counts[valid]
    target_deltas = flat_deltas[valid]
    emit_targets = (targets > 0).to(torch.long)
    gate_loss = F.cross_entropy(
        model.emit_head(context), emit_targets,
        weight=gate_class_weights, reduction="sum",
    )

    positive = targets > 0
    positive_atoms = int(positive.sum().detach().item())
    count_loss = context.new_zeros(())
    if positive_atoms:
        predicted_log_count = model.log_positive_count(
            context[positive]
        ).squeeze(-1)
        target_log_count = torch.log(
            targets[positive].to(predicted_log_count.dtype)
        )
        count_loss = F.smooth_l1_loss(
            predicted_log_count, target_log_count, reduction="sum",
        )

    action_delta_loss = context.new_zeros(())
    action_atoms = 0
    state = model.action_decoder.begin(context)
    for step in range(target_deltas.shape[1]):
        active = targets > step
        active_atoms = int(active.sum().detach().item())
        if not active_atoms:
            break
        indices = torch.nonzero(active, as_tuple=False).squeeze(1)
        active_state = state.index_select(0, indices)
        predicted_coordinate = model.action_decoder.coordinate(active_state)
        target = target_deltas[active, step]
        action_delta_loss = action_delta_loss + F.smooth_l1_loss(
            predicted_coordinate, target, reduction="sum",
        )
        action_atoms += active_atoms

        # The teacher target above is loss-only.  State feedback uses the
        # model's own predicted coordinate, exactly as inference does.
        advanced = model.action_decoder.advance(
            active_state, predicted_coordinate
        )
        state = state.index_copy(0, indices, advanced)

    # Each objective is reduced to its own observation mean.  The class
    # weights are derived only from the two observed training frequencies;
    # there is no hand-tuned coefficient or inference threshold.
    mean_gate_loss = gate_loss / float(decision_atoms)
    mean_count_loss = (
        count_loss / float(positive_atoms)
        if positive_atoms else context.new_zeros(())
    )
    mean_action_loss = (
        action_delta_loss / float(action_atoms)
        if action_atoms else context.new_zeros(())
    )
    return mean_gate_loss + mean_count_loss + mean_action_loss, {
        "gate_loss_sum": float(gate_loss.detach().item()),
        "positive_count_loss_sum": float(count_loss.detach().item()),
        "action_delta_loss_sum": float(action_delta_loss.detach().item()),
        "decision_atoms": decision_atoms,
        "positive_count_atoms": positive_atoms,
        "action_atoms": action_atoms,
    }


def train_model(
    model, rows, runtime, counts, deltas, device, epochs, chunk_len,
    pc_batch_size, learning_rate,
):
    grouped = _group_indices_by_pc(rows)
    gate_class_weights_numpy = _data_derived_gate_class_weights(counts)
    gate_class_weights = torch.from_numpy(
        gate_class_weights_numpy
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        recurrent_states = {}
        totals = {
            "gate_loss_sum": 0.0,
            "positive_count_loss_sum": 0.0,
            "action_delta_loss_sum": 0.0,
            "decision_atoms": 0,
            "positive_count_atoms": 0,
            "action_atoms": 0,
        }
        optimizer_steps = 0
        for batch in _chunk_batches(grouped, chunk_len, pc_batch_size):
            keys = [pc for pc, _ in batch]
            padded, count_batch, delta_batch, lengths = _make_padded_batch(
                runtime, counts, deltas, batch, device
            )
            optimizer.zero_grad(set_to_none=True)
            context = _encode_shared(
                model, padded, lengths, recurrent_states, keys,
            )
            loss, components = _compact_balanced_deterministic_loss(
                model, context, count_batch, delta_batch,
                gate_class_weights,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            for key, value in components.items():
                totals[key] += value
        row = {
            "epoch": epoch,
            "gate_loss_per_callback": (
                totals["gate_loss_sum"]
                / max(1, totals["decision_atoms"])
            ),
            "positive_count_loss_per_positive_callback": (
                totals["positive_count_loss_sum"]
                / max(1, totals["positive_count_atoms"])
            ),
            "action_delta_loss_per_action": (
                totals["action_delta_loss_sum"]
                / max(1, totals["action_atoms"])
            ),
            "gate_loss_sum": totals["gate_loss_sum"],
            "positive_count_loss_sum": totals["positive_count_loss_sum"],
            "action_delta_loss_sum": totals["action_delta_loss_sum"],
            "pc_sequences": len(grouped),
            "optimizer_steps": optimizer_steps,
        }
        history.append(row)
        print(
            "[train:balanced-deterministic-lstm] epoch={} gate={:.8f} "
            "count={:.8f} action={:.8f}".format(
                epoch, row["gate_loss_per_callback"],
                row["positive_count_loss_per_positive_callback"],
                row["action_delta_loss_per_action"],
            ),
            flush=True,
        )
    return history, gate_class_weights_numpy


def score_model(
    model, rows, runtime, device, chunk_len, pc_batch_size,
):
    grouped = _group_indices_by_pc(rows)
    output = np.empty(
        (len(rows), model.hidden_size), dtype=np.float32
    )
    recurrent_states = {}
    model.eval()
    with torch.no_grad():
        for batch in _chunk_batches(grouped, chunk_len, pc_batch_size):
            keys = [pc for pc, _ in batch]
            lengths = [len(indices) for _, indices in batch]
            padded = torch.zeros(
                len(batch), max(lengths), runtime.shape[1],
                dtype=torch.float32, device=device,
            )
            for position, (_, indices) in enumerate(batch):
                padded[position, :len(indices)] = torch.from_numpy(
                    runtime[indices]
                ).to(device)
            context = _encode_shared(
                model, padded, lengths, recurrent_states, keys,
            )
            for position, (_, indices) in enumerate(batch):
                length = len(indices)
                output[indices] = (
                    context[position, :length].cpu().numpy()
                )
    return output


def decode(model, context_numpy, base_lines, device, chunk_len=8192):
    if len(context_numpy) != len(base_lines):
        raise RuntimeError("decoder row counts differ")
    counts = np.zeros(len(base_lines), dtype=np.int64)
    gate_scores = np.empty((len(base_lines), 2), dtype=np.float32)
    positive_log_counts = np.empty(len(base_lines), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(base_lines), chunk_len):
            stop = min(start + chunk_len, len(base_lines))
            context = torch.from_numpy(context_numpy[start:stop]).to(device)
            scores = model.emit_head(context).cpu().numpy()
            log_counts = model.log_positive_count(
                context
            ).squeeze(-1).cpu().numpy()
            emit = scores.argmax(axis=1)
            counts[start:stop] = np.where(
                emit == 1,
                _positive_counts_from_log_mean(log_counts),
                0,
            )
            gate_scores[start:stop] = scores
            positive_log_counts[start:stop] = log_counts

    predicted_lines = [[] for _ in range(len(counts))]
    predicted_fills = [[] for _ in range(len(counts))]
    with torch.no_grad():
        for start in range(0, len(counts), chunk_len):
            stop = min(start + chunk_len, len(counts))
            chunk_context = torch.from_numpy(
                context_numpy[start:stop]
            ).to(device)
            state = model.action_decoder.begin(chunk_context)
            local_counts = counts[start:stop]
            steps = int(local_counts.max()) if len(local_counts) else 0
            for step in range(steps):
                active_numpy = np.flatnonzero(local_counts > step)
                if not len(active_numpy):
                    break
                active = torch.from_numpy(active_numpy).to(
                    device=device, dtype=torch.long
                )
                active_state = state.index_select(0, active)
                coordinate = model.action_decoder.coordinate(active_state)
                for local_position, value in zip(
                    active_numpy, coordinate.cpu().numpy()
                ):
                    global_position = start + int(local_position)
                    delta = _delta_to_integer(value)
                    predicted_lines[global_position].append(
                        apply_signed_line_delta(
                            base_lines[global_position], delta
                        )
                    )
                    predicted_fills[global_position].append(-1)
                advanced = model.action_decoder.advance(
                    active_state, coordinate
                )
                state = state.index_copy(0, active, advanced)
    return (
        counts, predicted_lines, predicted_fills,
        gate_scores, positive_log_counts,
    )


def trigger_metrics(predicted_counts, target_actions):
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
        "predicted_positive_callbacks": int(
            np.count_nonzero(predicted_positive)
        ),
        "normal_positive_callbacks": int(
            np.count_nonzero(target_positive)
        ),
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


def count_decoder_diagnostics(
    gate_scores, positive_log_counts, decoded_counts, gate_class_weights,
):
    """Report the deterministic hurdle and positive-count operating point."""
    gate_scores = np.asarray(gate_scores, dtype=np.float64)
    positive_log_counts = np.asarray(
        positive_log_counts, dtype=np.float64
    )
    decoded_counts = np.asarray(decoded_counts, dtype=np.int64)
    gate_class_weights = np.asarray(gate_class_weights, dtype=np.float64)
    if not (
        gate_scores.shape == (len(decoded_counts), 2)
        and len(positive_log_counts) == len(decoded_counts)
        and gate_class_weights.shape == (2,)
    ):
        raise RuntimeError("count diagnostic vector lengths differ")
    if np.any(decoded_counts < 0) or not np.all(np.isfinite(gate_scores)):
        raise RuntimeError("invalid deterministic count diagnostics")
    positive = decoded_counts[decoded_counts > 0]
    distribution = Counter(int(value) for value in decoded_counts)
    margins = gate_scores[:, 1] - gate_scores[:, 0]
    return {
        "callbacks": int(len(decoded_counts)),
        "gate_class_weights": [float(value) for value in gate_class_weights],
        "mean_emit_logit_margin": float(margins.mean()),
        "decoded_positive_callbacks": int(np.count_nonzero(decoded_counts)),
        "decoded_total_actions": int(decoded_counts.sum()),
        "decoded_mean_actions_per_callback": float(decoded_counts.mean()),
        "decoded_mean_actions_per_positive_callback": (
            float(positive.mean()) if len(positive) else 0.0
        ),
        "decoded_max_actions_per_callback": (
            int(decoded_counts.max()) if len(decoded_counts) else 0
        ),
        "decoded_count_distribution": {
            str(key): int(value)
            for key, value in sorted(distribution.items())
        },
        "positive_log_count_mean": float(positive_log_counts.mean()),
    }


def self_test_free_running_decoder(feature_count, hidden_size):
    torch.manual_seed(2718)
    model = CompactPCKeyedDeterministicStrideLSTM(feature_count, hidden_size)
    state = model.action_decoder.begin(torch.randn(3, hidden_size))
    predicted = model.action_decoder.coordinate(state)
    advanced_a = model.action_decoder.advance(state, predicted)
    # Teacher values are intentionally different but cannot enter advance().
    teacher_a = torch.tensor([0.0, 3.0, -7.0])
    teacher_b = torch.tensor([19.0, -2.0, 1.0])
    if torch.equal(teacher_a, teacher_b):
        raise RuntimeError("decoder self-test teacher setup failed")
    advanced_b = model.action_decoder.advance(state, predicted)
    if not torch.equal(advanced_a, advanced_b):
        raise RuntimeError("teacher delta leaked into decoder feedback")


def expected_parameter_count(feature_count, hidden_size):
    """Exact parameter count for the compact shared architecture."""
    return (
        11 * hidden_size * hidden_size
        + (feature_count + 22) * hidden_size
        + 4
    )


def self_test_compact_parameter_count(feature_count, hidden_size):
    model = CompactPCKeyedDeterministicStrideLSTM(feature_count, hidden_size)
    observed = sum(parameter.numel() for parameter in model.parameters())
    expected = expected_parameter_count(feature_count, hidden_size)
    if observed != expected:
        raise RuntimeError(
            "compact parameter formula mismatch: {} != {}".format(
                observed, expected
            )
        )
    configured = EXPECTED_PARAMETER_COUNTS.get(hidden_size)
    if feature_count == RUNTIME_FEATURES and configured != expected:
        raise RuntimeError(
            "configured parameter count mismatch: {} != {}".format(
                configured, expected
            )
        )


def self_test_deterministic_count_and_balance():
    values = _positive_counts_from_log_mean(
        np.log(np.asarray([1.0, 2.0, 9.0, 257.0]))
    )
    if values.tolist() != [1, 2, 9, 257]:
        raise RuntimeError("positive-count decoder is capped or non-variable")
    counts = np.asarray([0, 0, 0, 2], dtype=np.int64)
    weights = _data_derived_gate_class_weights(counts)
    labels = counts > 0
    negative_mass = float(weights[0] * np.count_nonzero(~labels))
    positive_mass = float(weights[1] * np.count_nonzero(labels))
    if not math.isclose(negative_mass, positive_mass):
        raise RuntimeError("data-derived hurdle classes are not balanced")


def self_test_pc_router():
    rows = [
        (11, 100, 0), (22, 200, 0), (11, 101, 0),
        (33, 300, 0), (22, 201, 0), (11, 102, 0),
    ]
    grouped = _group_indices_by_pc(rows)
    if list(grouped.values()) != [[0, 2, 5], [1, 4], [3]]:
        raise RuntimeError("PC-keyed state router broke causal order")


def self_test_no_future(feature_count, hidden_size):
    """Changing future rows must not change an already encoded prefix."""
    torch.manual_seed(31415)
    model = CompactPCKeyedDeterministicStrideLSTM(feature_count, hidden_size)
    rows = [(7, 100 + index, 0) for index in range(8)]
    original = np.zeros((len(rows), feature_count), dtype=np.float32)
    changed = original.copy()
    changed[5:, :] = 1.0
    first = score_model(
        model, rows, original, torch.device("cpu"), 8, 1
    )
    second = score_model(
        model, rows, changed, torch.device("cpu"), 8, 1
    )
    if not np.array_equal(first[:5], second[:5]):
        raise RuntimeError("future Stride rows changed a prior neural state")


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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--chunk-len", type=int, default=256)
    parser.add_argument("--pc-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    return parser


def main():
    args = build_parser().parse_args()
    expected_pair = MODEL_POINTS["lstm"].get(args.model_size)
    if expected_pair is None or args.pair_id != expected_pair:
        raise RuntimeError("model size/pair is not a configured LSTM point")
    if (
        args.model_size < 1 or args.epochs < 1 or args.chunk_len < 1
        or args.pc_batch_size < 1
    ):
        raise RuntimeError("model and batching dimensions must be positive")

    self_test_deterministic_count_and_balance()
    self_test_pc_router()
    self_test_free_running_decoder(
        RUNTIME_FEATURES, max(2, args.model_size)
    )
    self_test_compact_parameter_count(
        RUNTIME_FEATURES, args.model_size
    )
    self_test_no_future(
        RUNTIME_FEATURES, max(2, args.model_size)
    )

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
    runtime = {role: runtime_features(rows[role]) for role in roles}
    if any(value.shape[1] != RUNTIME_FEATURES for value in runtime.values()):
        raise RuntimeError("lossless PC/line-number feature count mismatch")
    for role in roles:
        if not np.array_equal(runtime[role], runtime_features(rows[role])):
            raise RuntimeError(
                "{} training/inference encoder differs".format(role)
            )

    train_counts, train_deltas, _ = targets_from_actions(
        [line for _, line, _ in rows["train"]], teacher["train"]
    )
    model = CompactPCKeyedDeterministicStrideLSTM(
        runtime["train"].shape[1], args.model_size
    )
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    expected_parameters = expected_parameter_count(
        runtime["train"].shape[1], args.model_size
    )
    if parameter_count != expected_parameters:
        raise RuntimeError(
            "compact parameter count mismatch: {} != {}".format(
                parameter_count, expected_parameters
            )
        )

    history, gate_class_weights = train_model(
        model, rows["train"], runtime["train"],
        train_counts, train_deltas, device, args.epochs,
        args.chunk_len, args.pc_batch_size, args.learning_rate,
    )

    # Re-run the complete causal train -> guard -> eval history with fixed
    # weights and fresh state.  Only the evaluation suffix is decoded.
    history_rows = rows["train"] + rows["guard"] + rows["eval"]
    history_runtime = np.concatenate(
        [runtime["train"], runtime["guard"], runtime["eval"]], axis=0
    )
    encoded_history = score_model(
        model, history_rows, history_runtime, device,
        args.chunk_len, args.pc_batch_size,
    )
    eval_start = len(rows["train"]) + len(rows["guard"])
    eval_context = encoded_history[eval_start:]
    (
        predicted_counts, predicted_lines, predicted_fills,
        eval_gate_scores, eval_positive_log_counts,
    ) = decode(
        model, eval_context,
        [line for _, line, _ in rows["eval"]], device,
    )
    count_diagnostics = count_decoder_diagnostics(
        eval_gate_scores, eval_positive_log_counts, predicted_counts,
        gate_class_weights,
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

    tag = "independent_delta_stride_lstm_h{}".format(args.model_size)
    torch.save({
        "state_dict": model.state_dict(),
        "model_family": "lstm",
        "model_size": args.model_size,
        "runtime_features": runtime["train"].shape[1],
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": MODEL_REVISION,
    }, args.out_dir / "model.pt")

    train_positive = int(np.count_nonzero(train_counts > 0))
    train_zero = int(len(train_counts) - train_positive)
    train_unique_pc_count = len(_group_indices_by_pc(rows["train"]))
    history_unique_pc_count = len(_group_indices_by_pc(history_rows))
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
        "parameter_bytes_float32": parameter_count * 4,
        "seed": args.seed,
        "decoder_seed": None,
        "deterministic_decoding": True,
        "deterministic_decoding_reproducible": True,
        "stochastic_decoding": False,
        "stochastic_decoding_reproducible": False,
        "common_random_numbers_across_capacities": False,
        "strict_common_random_numbers_across_capacities": False,
        "cross_event_rng_state_used": False,
        "sampler_revision": None,
        "decoder_sampling_roles": [],
        "decoder_train_sampling_performed": False,
        "decoder_guard_sampling_performed": False,
        "decoder_event_key_definition": None,
        "decoder_event_key_uses_teacher_information": False,
        "decoder_action_rank_origin": 0,
        "decoder_key_fields": [],
        "decoder_key_includes_sampler_revision": False,
        "decoder_sampler": None,
        "epochs": args.epochs,
        "chunk_len": args.chunk_len,
        "pc_batch_size": args.pc_batch_size,
        "learning_rate": args.learning_rate,
        "runtime_feature_count": runtime["train"].shape[1],
        "runtime_encoding": (
            "lossless uint64 PC plus lossless 58-bit cache-line number"
        ),
        "runtime_encoding_note": (
            "six constant-zero aligned-address byte-offset bits removed"
        ),
        "runtime_pc_bits": ADDRESS_BITS,
        "runtime_line_number_bits": LINE_NUMBER_BITS,
        "runtime_constant_offset_bits_removed": CACHE_LINE_OFFSET_BITS,
        "source_decision_effective_external_input": SOURCE_INPUTS,
        "same_external_input_contract": True,
        "training_inference_input_encoder_identical": True,
        "decoder_training_mode": (
            "free_running_autoregressive_same_as_inference"
        ),
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_free_running_self_test": "PASS",
        "runtime_encoder_entrypoint": (
            "623_offline_lstm_stride.train_and_offline_infer."
            "runtime_features"
        ),
        "runtime_encoder_sha256": runtime_encoder_sha256(),
        "training_runtime_encoder_sha256": runtime_encoder_sha256(),
        "inference_runtime_encoder_sha256": runtime_encoder_sha256(),
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
        "handcrafted_semantic_features_used": False,
        "manual_loss_weights_used": False,
        "data_derived_gate_class_weights_used": True,
        "gate_class_weighting_used": True,
        "training_regularization_used": False,
        "inference_policy_hardcodes_used": False,
        "learned_request_count": True,
        "nn_generates_own_target_addresses": True,
        "complete_action_space": (
            "learned zero or unbounded positive count plus direct "
            "signed cache-line deltas"
        ),
        "decision_rule": (
            "balanced_two_class_argmax_then_deterministic_rounded_"
            "positive_log_count_then_scalar_signed_log_delta"
        ),
        "gate_training_objective": (
            "data_derived_frequency_balanced_two_class_cross_entropy"
        ),
        "gate_decoding_rule": "deterministic_two_class_argmax",
        "gate_class_weights_source": (
            "train_zero_positive_frequencies_equal_aggregate_loss_mass"
        ),
        "gate_class_weights": [
            float(value) for value in gate_class_weights
        ],
        "request_count_training_objective": (
            "balanced_two_class_hurdle_plus_positive_log_count_smooth_l1"
        ),
        "request_count_decoding_rule": (
            "deterministic_gate_argmax_plus_rounded_exp_positive_log_count"
        ),
        "request_count_residual_scope": "none_event_local",
        "request_count_decoder_diagnostics": count_diagnostics,
        "request_count_training_label_statistics": {
            "decision_callbacks": int(len(train_counts)),
            "positive_callbacks": train_positive,
            "zero_callbacks": train_zero,
            "positive_callback_rate": (
                float(train_positive) / float(len(train_counts))
            ),
        },
        "request_count_model": (
            "learned balanced zero/positive categorical plus deterministic "
            "positive log-count with unbounded positive integer support"
        ),
        "loss_design": (
            "data-derived balanced gate CE mean plus positive log-count "
            "smooth-L1 mean plus scalar signed-log delta smooth-L1 mean; "
            "unit sum without manually tuned coefficients"
        ),
        "training_labels": (
            "captured Stride actions; supervision and comparator replay only"
        ),
        "forbidden_inputs": [
            "normal_actions_at_inference", "Stride_tracker_table",
            "last_stride", "normal_degree", "cycle", "cache_hit",
            "queue_state", "future_rows",
        ],
        "training_chunks_shuffled": False,
        "training_state_mode": "chronological_stateful_tbptt",
        "training_state_carried_across_chunks": True,
        "training_state_detached_between_chunks": True,
        "training_state_routing": (
            "dynamic exact-PC key with no fixed tracker capacity"
        ),
        "training_state_reset": "only_at_epoch_start",
        "inference_history_mode": (
            "fresh_state_then_complete_train_guard_eval_chronology"
        ),
        "inference_state_routing": (
            "dynamic exact-PC key with no fixed tracker capacity"
        ),
        "delta_mixture_decoding_rule": None,
        "delta_mixture_component_canonicalization": None,
        "delta_training_objective": "scalar_signed_log_delta_smooth_l1",
        "delta_decoding_rule": "deterministic_rounded_scalar_signed_log_delta",
        "delta_decoder_feedback_rule": (
            "emitted_scalar_coordinate_same_in_training_and_inference"
        ),
        "guard_role": "causal_input_history_warmup_and_audit_only",
        "train_unique_pc_count": train_unique_pc_count,
        "history_unique_pc_count": history_unique_pc_count,
        "recurrent_state_bytes_per_observed_pc": (
            2 * args.model_size * 4
        ),
        "training_state_router_sha256": state_router_sha256(),
        "inference_state_router_sha256": state_router_sha256(),
        "peak_training_recurrent_state_bytes_float32": (
            train_unique_pc_count * 2 * args.model_size * 4
        ),
        "peak_inference_recurrent_state_bytes_float32": (
            history_unique_pc_count * 2 * args.model_size * 4
        ),
        "peak_persistent_recurrent_state_bytes": (
            history_unique_pc_count * 2 * args.model_size * 4
        ),
        "causal_no_future_self_test": "PASS",
        "pc_keyed_causality_self_test": "PASS",
        "event_keyed_crn_self_test": "NOT_APPLICABLE",
        "event_keyed_hurdle_count_self_test": "NOT_APPLICABLE",
        "canonicalized_mixture_sampling_self_test": "NOT_APPLICABLE",
        "deterministic_count_and_balance_self_test": "PASS",
        "decoder_probability_mass_carries_train_guard_history": False,
        "cross_event_probability_credit_used": False,
        "sampled_outputs_used_as_decoder_feedback": False,
        "delta_mixture_components": 0,
        "compact_parameter_self_test": "PASS",
        "cnn_architecture_self_test": "NOT_APPLICABLE",
        "cnn_temporal_layers": 0,
        "cnn_kernel_size": 0,
        "cnn_stride": 0,
        "cnn_dilations": [],
        "cnn_receptive_field_events": 0,
        "training_left_context_overlap": 0,
        "cnn_processes_complete_stream_in_order": False,
        "cnn_chunking_changes_visible_history": False,
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
        metadata[role + "_stream_gzip_sha256"] = sha256(
            stream_paths[role]
        )
        metadata[role + "_stream_content_sha256"] = gzip_content_sha256(
            stream_paths[role]
        )
        metadata[role + "_candidate_gzip_sha256"] = sha256(
            action_paths[role]
        )
        metadata[role + "_candidate_content_sha256"] = (
            gzip_content_sha256(action_paths[role])
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
