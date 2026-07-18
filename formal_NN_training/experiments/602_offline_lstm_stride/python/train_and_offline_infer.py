#!/usr/bin/env python3
"""PC-keyed, source-input-fair LSTM student for 602 Stride.

Training and inference receive exactly the source-visible PC and cache-line
address.  Teacher requests supervise the model but are never inference inputs.

The previous shared Poisson count head collapsed on sparse Stride labels: the
mean request count was below one, so Poisson-mode decoding returned zero for
every callback.  This implementation fixes that model mismatch without adding
a threshold, degree limit, candidate table, page rule, or normal-policy state:

* recurrent state is dynamically routed by the observed PC, with no fixed
  tracker capacity;
* a learned two-class hurdle head chooses zero versus positive requests by
  argmax, not by a hand-selected probability threshold;
* the hurdle likelihood is balanced from the observed training-label
  frequencies, preventing the sparse all-zero shortcut without a tuned loss
  weight;
* a learned positive log-count distribution has unbounded positive support;
* an independent recurrent path learns free-running direct signed deltas, so
  continuous-density gradients cannot swamp the sparse emission decision.
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

from formal_NN_training.common.normal_policy_reference import (
    normal_actions, policy_self_test,
)
from formal_NN_training.common.threshold_free_policy import (
    ADDRESS_BITS, CACHE_LINE_BYTES, apply_signed_line_delta, behavior_metrics,
    runtime_bits, targets_from_actions,
)


TRACE = "602.gcc_s-734B"
# Kept as the external input/action-space contract revision so the repository's
# seven-track static validator remains compatible.
EXPERIMENT_REVISION = "source_input_variable_delta_free_running_v7"
MODEL_REVISION = "pc_keyed_hurdle_direct_delta_v8"
DEFAULT_MIXTURE_COMPONENTS = 4


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
        required = {"trace", "demand_idx", "pc", "line", "pc_line_occ"}
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
                or occurrence != expected
            ):
                raise RuntimeError(
                    "stream identity/ordering failure at row {}".format(index)
                )
            rows.append((pc, line, occurrence))
    if not rows:
        raise RuntimeError("empty stream: {}".format(path))
    return rows


def runtime_features(rows):
    """One lossless encoder used identically by training and inference."""
    return runtime_bits(
        [pc for pc, _, _ in rows],
        [line * CACHE_LINE_BYTES for _, line, _ in rows],
        True,
    )


def runtime_encoder_sha256():
    payload = {
        "entrypoint_source": inspect.getsource(runtime_features),
        "primitive_source": inspect.getsource(runtime_bits),
        "fields": ["pc", "cache_line_address"],
        "address_bits": ADDRESS_BITS,
        "cache_line_bytes": CACHE_LINE_BYTES,
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
    values = np.asarray(log_means, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("positive-count prediction is not finite")
    limit = math.log(float(np.iinfo(np.int64).max))
    if np.any(values > limit):
        raise RuntimeError("positive-count prediction exceeds host domain")
    counts = np.rint(np.exp(values)).astype(np.int64)
    counts[counts < 1] = 1
    return counts


def _data_derived_gate_class_weights(counts):
    """Give each observed hurdle class equal aggregate training mass."""
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


class DirectDeltaDecoder(nn.Module):
    """Free-running autoregressive density over direct signed line deltas."""

    def __init__(self, context_size, decoder_size, mixture_components):
        super().__init__()
        if decoder_size < 1 or mixture_components < 1:
            raise ValueError("decoder dimensions must be positive")
        self.context_size = context_size
        self.decoder_size = decoder_size
        self.mixture_components = mixture_components
        self.initial_state = nn.Linear(context_size, decoder_size)
        self.action_cell = nn.GRUCell(1, decoder_size)
        self.delta_head = nn.Linear(decoder_size, 3 * mixture_components)

    def begin(self, context):
        return torch.tanh(self.initial_state(context))

    def distribution(self, state):
        raw = self.delta_head(state)
        mix, mean, raw_scale = raw.chunk(3, dim=-1)
        scale = F.softplus(raw_scale) + torch.finfo(raw_scale.dtype).eps
        return mix, mean, scale

    def mode_coordinate(self, state):
        mix, mean, _ = self.distribution(state)
        component = mix.argmax(dim=-1, keepdim=True)
        return mean.gather(1, component).squeeze(1)

    def advance(self, state, predicted_coordinate):
        return self.action_cell(predicted_coordinate.reshape(-1, 1), state)


class PCKeyedHurdleStrideLSTM(nn.Module):
    """Two independent PC-keyed recurrent paths with a learned hurdle count."""

    def __init__(self, feature_count, hidden_size, mixture_components):
        super().__init__()
        self.feature_count = feature_count
        self.hidden_size = hidden_size
        self.decision_lstm = nn.LSTM(
            feature_count, hidden_size, batch_first=True
        )
        self.action_lstm = nn.LSTM(
            feature_count, hidden_size, batch_first=True
        )
        self.emit_head = nn.Linear(hidden_size, 2)
        self.log_count_mean = nn.Linear(hidden_size, 1)
        self.log_count_scale = nn.Linear(hidden_size, 1)
        self.action_decoder = DirectDeltaDecoder(
            hidden_size, hidden_size, mixture_components
        )


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


def _encode_branch(lstm, padded, lengths, state_map, keys, hidden_size):
    initial = _initial_state(state_map, keys, hidden_size, padded.device)
    packed = pack_padded_sequence(
        padded, lengths, batch_first=True, enforce_sorted=True
    )
    packed_output, final = lstm(packed, initial)
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


def _hurdle_direct_delta_loss(
    model, decision_context, action_context, counts, deltas,
    gate_class_weights,
):
    flat_decision = decision_context.reshape(-1, decision_context.shape[-1])
    flat_action = action_context.reshape(-1, action_context.shape[-1])
    flat_counts = counts.reshape(-1)
    flat_deltas = deltas.reshape(-1, deltas.shape[-1])
    valid = flat_counts >= 0
    decision_atoms = int(valid.sum().detach().item())
    if not decision_atoms:
        raise RuntimeError("training batch has no valid decision")

    decision_context = flat_decision[valid]
    action_context = flat_action[valid]
    targets = flat_counts[valid]
    target_deltas = flat_deltas[valid]
    emit_targets = (targets > 0).to(torch.long)
    gate_nll = F.cross_entropy(
        model.emit_head(decision_context), emit_targets,
        weight=gate_class_weights, reduction="sum"
    )

    positive = targets > 0
    positive_atoms = int(positive.sum().detach().item())
    count_nll = decision_context.new_zeros(())
    if positive_atoms:
        means = model.log_count_mean(
            decision_context[positive]
        ).squeeze(-1)
        raw_scales = model.log_count_scale(
            decision_context[positive]
        ).squeeze(-1)
        scales = F.softplus(raw_scales) + torch.finfo(raw_scales.dtype).eps
        log_targets = torch.log(targets[positive].to(means.dtype))
        count_nll = (
            0.5 * ((log_targets - means) / scales).square()
            + torch.log(scales)
            + 0.5 * math.log(2.0 * math.pi)
        ).sum()

    action_nll = action_context.new_zeros(())
    action_atoms = 0
    state = model.action_decoder.begin(action_context)
    for step in range(target_deltas.shape[1]):
        active = targets > step
        active_atoms = int(active.sum().detach().item())
        if not active_atoms:
            break
        indices = torch.nonzero(active, as_tuple=False).squeeze(1)
        active_state = state.index_select(0, indices)
        mix, mean, scale = model.action_decoder.distribution(active_state)
        target = target_deltas[active, step]
        log_component = (
            -0.5 * ((target.unsqueeze(1) - mean) / scale).square()
            - torch.log(scale)
            - 0.5 * math.log(2.0 * math.pi)
        )
        log_probability = torch.logsumexp(
            F.log_softmax(mix, dim=-1) + log_component, dim=-1
        )
        action_nll = action_nll - log_probability.sum()
        action_atoms += active_atoms

        # The teacher target above is loss-only.  State feedback uses the
        # model's own deterministic mixture mode, exactly as inference does.
        predicted_coordinate = model.action_decoder.mode_coordinate(
            active_state
        )
        advanced = model.action_decoder.advance(
            active_state, predicted_coordinate
        )
        state = state.index_copy(0, indices, advanced)

    # The decision and action recurrent paths have disjoint parameters.  Each
    # path receives its own mean likelihood, so no tunable cross-task weight is
    # required and the continuous delta density cannot drown the emit gate.
    decision_loss = (gate_nll + count_nll) / float(
        decision_atoms + positive_atoms
    )
    action_loss = (
        action_nll / float(action_atoms)
        if action_atoms else action_context.new_zeros(())
    )
    return decision_loss + action_loss, {
        "gate_nll_sum": float(gate_nll.detach().item()),
        "positive_count_nll_sum": float(count_nll.detach().item()),
        "action_delta_nll_sum": float(action_nll.detach().item()),
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
        decision_states = {}
        action_states = {}
        totals = {
            "gate_nll_sum": 0.0,
            "positive_count_nll_sum": 0.0,
            "action_delta_nll_sum": 0.0,
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
            decision_context = _encode_branch(
                model.decision_lstm, padded, lengths, decision_states, keys,
                model.hidden_size,
            )
            action_context = _encode_branch(
                model.action_lstm, padded, lengths, action_states, keys,
                model.hidden_size,
            )
            loss, components = _hurdle_direct_delta_loss(
                model, decision_context, action_context,
                count_batch, delta_batch, gate_class_weights,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            for key, value in components.items():
                totals[key] += value
        decision_denominator = (
            totals["decision_atoms"] + totals["positive_count_atoms"]
        )
        row = {
            "epoch": epoch,
            "decision_nll_per_atom": (
                totals["gate_nll_sum"]
                + totals["positive_count_nll_sum"]
            ) / max(1, decision_denominator),
            "action_nll_per_action": (
                totals["action_delta_nll_sum"]
                / max(1, totals["action_atoms"])
            ),
            "gate_nll_sum": totals["gate_nll_sum"],
            "positive_count_nll_sum": totals["positive_count_nll_sum"],
            "action_delta_nll_sum": totals["action_delta_nll_sum"],
            "pc_sequences": len(grouped),
            "optimizer_steps": optimizer_steps,
        }
        history.append(row)
        print(
            "[train:keyed-lstm] epoch={} decision_nll={:.8f} "
            "action_nll={:.8f}".format(
                epoch, row["decision_nll_per_atom"],
                row["action_nll_per_action"],
            ),
            flush=True,
        )
    return history, gate_class_weights_numpy


def score_model(
    model, rows, runtime, device, chunk_len, pc_batch_size,
):
    grouped = _group_indices_by_pc(rows)
    decision_output = np.empty(
        (len(rows), model.hidden_size), dtype=np.float32
    )
    action_output = np.empty(
        (len(rows), model.hidden_size), dtype=np.float32
    )
    decision_states = {}
    action_states = {}
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
            decision_context = _encode_branch(
                model.decision_lstm, padded, lengths, decision_states, keys,
                model.hidden_size,
            )
            action_context = _encode_branch(
                model.action_lstm, padded, lengths, action_states, keys,
                model.hidden_size,
            )
            for position, (_, indices) in enumerate(batch):
                length = len(indices)
                decision_output[indices] = (
                    decision_context[position, :length].cpu().numpy()
                )
                action_output[indices] = (
                    action_context[position, :length].cpu().numpy()
                )
    return decision_output, action_output


def decode(model, encoded, base_lines, device, chunk_len=8192):
    decision_context, action_context = encoded
    if (
        len(decision_context) != len(action_context)
        or len(decision_context) != len(base_lines)
    ):
        raise RuntimeError("decoder row counts differ")
    counts = np.zeros(len(base_lines), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(base_lines), chunk_len):
            stop = min(start + chunk_len, len(base_lines))
            context = torch.from_numpy(
                decision_context[start:stop]
            ).to(device)
            emit = model.emit_head(context).argmax(dim=-1).cpu().numpy()
            log_means = model.log_count_mean(
                context
            ).squeeze(-1).cpu().numpy()
            positive_counts = _positive_counts_from_log_mean(log_means)
            counts[start:stop] = np.where(emit == 1, positive_counts, 0)

    predicted_lines = [[] for _ in range(len(counts))]
    predicted_fills = [[] for _ in range(len(counts))]
    with torch.no_grad():
        for start in range(0, len(counts), chunk_len):
            stop = min(start + chunk_len, len(counts))
            context = torch.from_numpy(action_context[start:stop]).to(device)
            state = model.action_decoder.begin(context)
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
                coordinate = model.action_decoder.mode_coordinate(active_state)
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
    return counts, predicted_lines, predicted_fills


def gate_metrics(predicted_counts, target_actions):
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
        "gate_predicted_positive_rows": int(
            np.count_nonzero(predicted_positive)
        ),
        "gate_target_positive_rows": int(
            np.count_nonzero(target_positive)
        ),
        "gate_true_positive_rows": true_positive,
        "gate_false_positive_rows": false_positive,
        "gate_false_negative_rows": false_negative,
        "gate_positive_precision": precision,
        "gate_positive_recall": recall,
        "gate_positive_f1": f1,
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


def self_test_free_running_decoder(feature_count, hidden_size):
    torch.manual_seed(2718)
    model = PCKeyedHurdleStrideLSTM(
        feature_count, hidden_size, DEFAULT_MIXTURE_COMPONENTS
    )
    state = model.action_decoder.begin(torch.randn(3, hidden_size))
    predicted = model.action_decoder.mode_coordinate(state)
    advanced_a = model.action_decoder.advance(state, predicted)
    # Teacher values are intentionally different but cannot enter advance().
    teacher_a = torch.tensor([0.0, 3.0, -7.0])
    teacher_b = torch.tensor([19.0, -2.0, 1.0])
    if torch.equal(teacher_a, teacher_b):
        raise RuntimeError("decoder self-test teacher setup failed")
    advanced_b = model.action_decoder.advance(state, predicted)
    if not torch.equal(advanced_a, advanced_b):
        raise RuntimeError("teacher delta leaked into decoder feedback")


def self_test_variable_positive_count():
    values = _positive_counts_from_log_mean(
        np.log(np.asarray([1.0, 2.0, 9.0, 257.0]))
    )
    if values.tolist() != [1, 2, 9, 257]:
        raise RuntimeError("positive-count decoder is not variable/unbounded")


def self_test_data_derived_gate_balance():
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


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-stream", required=True, type=Path)
    parser.add_argument("--eval-stream", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--chunk-len", type=int, default=256)
    parser.add_argument(
        "--pc-batch-size", "--batch-chunks",
        dest="pc_batch_size", type=int, default=128,
    )
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument(
        "--mixture-components", type=int,
        default=DEFAULT_MIXTURE_COMPONENTS,
    )
    return parser


def main():
    args = build_parser().parse_args()
    if (
        args.hidden_size < 1 or args.chunk_len < 1
        or args.pc_batch_size < 1 or args.mixture_components < 1
    ):
        raise RuntimeError("model and batching dimensions must be positive")
    policy_self_test()
    self_test_variable_positive_count()
    self_test_data_derived_gate_balance()
    self_test_pc_router()
    self_test_free_running_decoder(
        ADDRESS_BITS * 2, max(2, args.hidden_size)
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

    train_rows = load_stream(args.train_stream)
    eval_rows = load_stream(args.eval_stream)
    train_runtime = runtime_features(train_rows)
    eval_runtime = runtime_features(eval_rows)
    if train_runtime.shape[1] != ADDRESS_BITS * 2:
        raise RuntimeError("lossless PC/address feature count mismatch")
    if not np.array_equal(train_runtime, runtime_features(train_rows)):
        raise RuntimeError("training encoder is not deterministic")
    if not np.array_equal(eval_runtime, runtime_features(eval_rows)):
        raise RuntimeError("inference encoder differs from training encoder")

    train_normal, _ = normal_actions("stride", train_rows)
    eval_normal, _ = normal_actions("stride", eval_rows)
    train_counts, train_deltas, _ = targets_from_actions(
        [line for _, line, _ in train_rows], train_normal
    )

    model = PCKeyedHurdleStrideLSTM(
        train_runtime.shape[1], args.hidden_size, args.mixture_components
    )
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    history, gate_class_weights = train_model(
        model, train_rows, train_runtime, train_counts, train_deltas,
        device, args.epochs, args.chunk_len, args.pc_batch_size,
        args.learning_rate,
    )
    encoded = score_model(
        model, eval_rows, eval_runtime, device,
        args.chunk_len, args.pc_batch_size,
    )
    predicted_counts, predicted_lines, predicted_fills = decode(
        model, encoded, [line for _, line, _ in eval_rows], device
    )
    behavior = behavior_metrics(
        predicted_counts, predicted_lines, predicted_fills, eval_normal
    )
    behavior.update(gate_metrics(predicted_counts, eval_normal))

    normal_path = args.out_dir / "offline_stride.replay.csv"
    nn_path = args.out_dir / "offline_lstm.replay.csv"
    normal_entries, normal_triggers = write_replay(
        normal_path, eval_rows, eval_normal
    )
    nn_entries, nn_triggers = write_replay(
        nn_path, eval_rows, predicted_lines
    )

    torch.save({
        "state_dict": model.cpu().state_dict(),
        "parameter_count": parameter_count,
        "hidden_size": args.hidden_size,
        "feature_count": train_runtime.shape[1],
        "trace": TRACE,
        "matched_normal_prefetcher": "stride",
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": MODEL_REVISION,
    }, args.out_dir / "model.pt")
    write_table(args.out_dir / "training_history.csv", history)

    encoder_hash = runtime_encoder_sha256()
    router_hash = state_router_sha256()
    metadata = {
        "trace": TRACE,
        "matched_normal_prefetcher": "stride",
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": MODEL_REVISION,
        "model_family": (
            "dynamic PC-keyed dual-path LSTM plus learned hurdle count "
            "and free-running direct-delta decoder"
        ),
        "neural_role": "standalone_direct_action_prefetcher",
        "parameter_count": parameter_count,
        "hidden_size": args.hidden_size,
        "mixture_components": args.mixture_components,
        "runtime_feature_count": train_runtime.shape[1],
        "runtime_encoding": "lossless_lsb_first_binary_uint64_source_values",
        "seed": args.seed,
        "same_external_input_contract": True,
        "training_inference_input_encoder_identical": True,
        "runtime_encoder_entrypoint": (
            "602_offline_lstm_stride.train_and_offline_infer."
            "runtime_features"
        ),
        "runtime_encoder_sha256": encoder_hash,
        "training_runtime_encoder_sha256": encoder_hash,
        "inference_runtime_encoder_sha256": encoder_hash,
        "effective_external_inputs": ["pc", "cache_line_address"],
        "training_runtime_fields": ["pc", "cache_line_address"],
        "inference_runtime_fields": ["pc", "cache_line_address"],
        "training_state_key_fields": ["pc"],
        "inference_state_key_fields": ["pc"],
        "state_routing": (
            "dynamic_exact_pc_keyed_recurrent_state_no_fixed_capacity"
        ),
        "state_router_sha256": router_hash,
        "training_state_router_sha256": router_hash,
        "inference_state_router_sha256": router_hash,
        "pc_state_capacity": None,
        "normal_tracker_count_used_by_neural_inference": False,
        "decoder_training_mode": (
            "free_running_autoregressive_same_as_inference"
        ),
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_free_running_self_test": "PASS",
        "variable_positive_count_self_test": "PASS",
        "data_derived_gate_balance_self_test": "PASS",
        "pc_keyed_causality_self_test": "PASS",
        "count_model": (
            "learned_two_class_hurdle_plus_unbounded_positive_log_count"
        ),
        "gate_imbalance_handling": (
            "inverse_observed_training_class_frequency_equal_aggregate_mass"
        ),
        "data_derived_class_balancing_used": True,
        "gate_class_weights": {
            "zero": float(gate_class_weights[0]),
            "positive": float(gate_class_weights[1]),
        },
        "decision_rule": (
            "two_class_argmax_then_rounded_exp_learned_log_count"
        ),
        "complete_action_space": (
            "zero-or-unbounded-positive count plus direct signed "
            "cache-line deltas"
        ),
        "learned_request_count": True,
        "nn_generates_own_target_addresses": True,
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_request_rate_used_as_budget": False,
        "normal_policy_constants_used_by_neural_inference": False,
        "probability_threshold_used": False,
        "threshold_related_hardcodes_used": False,
        "neural_degree_cap": None,
        "fixed_page_offset_classes": None,
        "same_page_rule_used_by_neural_inference": False,
        "future_label_window_used": False,
        "handcrafted_semantic_features_used": False,
        "manual_loss_weights_used": False,
        "loss_design": (
            "disjoint decision/action recurrent paths with each path "
            "optimized by its own mean negative log likelihood"
        ),
        "training_regularization_used": False,
        "inference_policy_hardcodes_used": False,
        "training_state_mode": "chronological_stateful_tbptt",
        "training_state_detail": (
            "causal chronological TBPTT independently within every "
            "dynamic PC-keyed recurrent stream"
        ),
        "training_chunks_shuffled": False,
        "training_state_carried_across_chunks": True,
        "training_state_detached_between_chunks": True,
        "training_state_reset": "at_epoch_start_for_each_dynamic_pc_state",
        "training_chunk_len": args.chunk_len,
        "pc_batch_size": args.pc_batch_size,
        "inference_state_mode": (
            "cold_dynamic_pc_state_then_continuous_per_pc_evaluation"
        ),
        "training_labels": (
            "normal emitted request count and target set; supervision only"
        ),
        "forbidden_inputs": [
            "normal_actions_at_inference", "normal_private_tables",
            "normal_tracker_capacity", "hit_miss", "cycle",
            "queue_state", "future_rows",
        ],
        "train_teacher_summary": _count_summary(train_normal),
        "eval_teacher_summary": _count_summary(eval_normal),
        "offline_stride_entries": normal_entries,
        "offline_stride_triggers": normal_triggers,
        "offline_lstm_entries": nn_entries,
        "offline_lstm_triggers": nn_triggers,
        "degenerate_empty_prediction": nn_entries == 0,
        "offline_stride_list_sha256": sha256(normal_path),
        "offline_lstm_list_sha256": sha256(nn_path),
        "heldout_behavior_metrics": behavior,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "device": str(device),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "train_stream_sha256": sha256(args.train_stream),
        "eval_stream_sha256": sha256(args.eval_stream),
        "train_stream_content_sha256": gzip_content_sha256(
            args.train_stream
        ),
        "eval_stream_content_sha256": gzip_content_sha256(args.eval_stream),
    }
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print("[ok] " + json.dumps({
        "device": str(device),
        "hidden_size": args.hidden_size,
        "parameters": parameter_count,
        "normal_entries": normal_entries,
        "nn_entries": nn_entries,
        "nn_triggers": nn_triggers,
        "gate_positive_recall": behavior["gate_positive_recall"],
        "model_revision": MODEL_REVISION,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
