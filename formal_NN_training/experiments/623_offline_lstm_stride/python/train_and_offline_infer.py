#!/usr/bin/env python3
"""PC-keyed, source-input-fair LSTM student for 623 Stride.

Training and inference receive exactly the source-visible PC and cache-line
address.  Teacher requests supervise the model but are never inference inputs.

This revision models and samples the complete local action distribution at
each callback without adding capacity, a threshold, a degree limit, a
candidate table, a page rule, or normal-policy state:

* recurrent state is dynamically routed by the observed PC, with no fixed
  tracker capacity;
* an unweighted Bernoulli hurdle and positive-count Poisson are fit by ordinary
  maximum likelihood;
* SHA-256-keyed inverse-CDF Bernoulli and Poisson samples realize the learned
  hurdle and unbounded positive excess count.  Stateless event keys give all
  capacity points strict common random numbers without carrying RNG state;
* one shared single-layer LSTM supplies both decision and action context;
* a lightweight autoregressive three-component mixture learns direct signed
  deltas.  Components are canonicalized by learned mean before keyed sampling,
  while recurrent feedback uses the complete mixture expectation in both
  training and inference;
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
from formal_NN_training.common.keyed_sampling import (
    SAMPLER_REVISION, canonical_component_order, categorical_icdf,
    event_keyed_hurdle_counts, key_schedule_sha256, key_stream_sha256,
    keyed_uniform, sampler_metadata, sampler_source_sha256,
    sampling_schedule_sha256, self_test_keyed_crn,
)


TRACE = "623.xalancbmk_s-700B"
POLICY = "stride"
EXPERIMENT_REVISION = "stride_source_input_variable_delta_free_running_v9"
MODEL_REVISION = "compact_pc_keyed_crn_event_sampled_mixture_v15"
EVENT_LOGGER_SCHEMA = "623_causal_trigger_v5"
CANDIDATE_ATTACHMENT_MODE = "explicit_trigger_event_id"
SOURCE_INPUTS = ["pc", "addr"]
MODEL_POINTS = {
    "lstm": {8: "p0", 16: "p1", 32: "p2", 64: "p3", 128: "p4"},
}
MIXTURE_COMPONENTS = 3
CACHE_LINE_OFFSET_BITS = int(math.log(CACHE_LINE_BYTES, 2))
LINE_NUMBER_BITS = ADDRESS_BITS - CACHE_LINE_OFFSET_BITS
RUNTIME_FEATURES = ADDRESS_BITS + LINE_NUMBER_BITS
EXPECTED_PARAMETER_COUNTS = {
    8: 1923, 16: 5243, 32: 16107, 64: 54731, 128: 199563,
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


def _poisson_means(log_rates):
    """Convert learned log means without clipping or a policy count cap."""
    values = np.asarray(log_rates, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise RuntimeError("positive-count log mean is not a finite vector")
    limit = math.log(float(np.iinfo(np.int64).max))
    if np.any(values > limit):
        raise RuntimeError("positive-count mean exceeds host count domain")
    means = np.exp(values)
    if not np.all(np.isfinite(means)) or np.any(means < 0.0):
        raise RuntimeError("positive-count mean is outside its numeric domain")
    return means


def _sigmoid_probabilities(logits):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise RuntimeError("trigger logits are not a finite vector")
    probabilities = np.empty_like(values)
    positive = values >= 0.0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    probabilities[~positive] = exp_values / (1.0 + exp_values)
    return probabilities


class CompactDirectDeltaDecoder(nn.Module):
    """Lightweight free-running direct signed-delta mixture."""

    def __init__(self, hidden_size):
        super().__init__()
        if hidden_size < 1:
            raise ValueError("decoder hidden size must be positive")
        self.action_cell = nn.GRUCell(1, hidden_size)
        self.delta_head = nn.Linear(hidden_size, 3 * MIXTURE_COMPONENTS)

    def begin(self, context):
        return context

    def distribution(self, state):
        raw = self.delta_head(state)
        mix, mean, raw_scale = raw.chunk(3, dim=-1)
        scale = F.softplus(raw_scale) + torch.finfo(raw_scale.dtype).tiny
        return mix, mean, scale

    def feedback_coordinate(self, state):
        mix, mean, _ = self.distribution(state)
        return (F.softmax(mix, dim=-1) * mean).sum(dim=-1)

    def advance(self, state, predicted_coordinate):
        return self.action_cell(predicted_coordinate.reshape(-1, 1), state)


class CompactPCKeyedSampledStrideLSTM(nn.Module):
    """One shared single-layer PC-keyed LSTM with lightweight heads."""

    def __init__(self, feature_count, hidden_size):
        super().__init__()
        self.feature_count = feature_count
        self.hidden_size = hidden_size
        self.input_projection = nn.Linear(feature_count, hidden_size)
        self.encoder_lstm = nn.LSTM(
            hidden_size, hidden_size, batch_first=True
        )
        self.trigger_logit = nn.Linear(hidden_size, 1)
        self.log_positive_excess_mean = nn.Linear(hidden_size, 1)
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


def _compact_event_distribution_loss(model, context, counts, deltas):
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
    trigger_targets = (targets > 0).to(context.dtype)
    trigger_logits = model.trigger_logit(context).squeeze(-1)
    trigger_loss = F.binary_cross_entropy_with_logits(
        trigger_logits, trigger_targets, reduction="sum",
    )

    positive = targets > 0
    positive_atoms = int(positive.sum().detach().item())
    excess_loss = context.new_zeros(())
    if positive_atoms:
        log_excess = model.log_positive_excess_mean(
            context[positive]
        ).squeeze(-1)
        excess_targets = targets[positive] - 1
        excess_loss = F.poisson_nll_loss(
            log_excess, excess_targets.to(log_excess.dtype), log_input=True,
            full=False, reduction="sum",
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
        mix, mean, scale = model.action_decoder.distribution(active_state)
        predicted_coordinate = (
            F.softmax(mix, dim=-1) * mean
        ).sum(dim=-1)
        target = target_deltas[active, step]
        log_component = (
            -0.5 * ((target.unsqueeze(1) - mean) / scale).square()
            - torch.log(scale)
            - 0.5 * math.log(2.0 * math.pi)
        )
        action_delta_loss = action_delta_loss - torch.logsumexp(
            F.log_softmax(mix, dim=-1) + log_component, dim=-1
        ).sum()
        action_atoms += active_atoms

        # The teacher target above is loss-only.  State feedback uses the
        # model's own predicted coordinate, exactly as inference does.
        advanced = model.action_decoder.advance(
            active_state, predicted_coordinate
        )
        state = state.index_copy(0, indices, advanced)

    # Every likelihood is reduced to its own observation mean. Their unit sum
    # has no tuned coefficient or class-frequency reweighting.
    mean_trigger_loss = trigger_loss / float(decision_atoms)
    mean_excess_loss = (
        excess_loss / float(positive_atoms)
        if positive_atoms else context.new_zeros(())
    )
    mean_action_loss = (
        action_delta_loss / float(action_atoms)
        if action_atoms else context.new_zeros(())
    )
    return mean_trigger_loss + mean_excess_loss + mean_action_loss, {
        "trigger_nll_sum": float(trigger_loss.detach().item()),
        "positive_excess_nll_sum": float(excess_loss.detach().item()),
        "action_delta_nll_sum": float(action_delta_loss.detach().item()),
        "decision_atoms": decision_atoms,
        "positive_atoms": positive_atoms,
        "action_atoms": action_atoms,
    }


def train_model(
    model, rows, runtime, counts, deltas, device, epochs, chunk_len,
    pc_batch_size, learning_rate,
):
    grouped = _group_indices_by_pc(rows)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        recurrent_states = {}
        totals = {
            "trigger_nll_sum": 0.0,
            "positive_excess_nll_sum": 0.0,
            "action_delta_nll_sum": 0.0,
            "decision_atoms": 0,
            "positive_atoms": 0,
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
            loss, components = _compact_event_distribution_loss(
                model, context, count_batch, delta_batch,
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
            "trigger_nll_per_callback": (
                totals["trigger_nll_sum"]
                / max(1, totals["decision_atoms"])
            ),
            "positive_excess_nll_per_positive_callback": (
                totals["positive_excess_nll_sum"]
                / max(1, totals["positive_atoms"])
            ),
            "action_delta_nll_per_action": (
                totals["action_delta_nll_sum"]
                / max(1, totals["action_atoms"])
            ),
            "trigger_nll_sum": totals["trigger_nll_sum"],
            "positive_excess_nll_sum": totals["positive_excess_nll_sum"],
            "action_delta_nll_sum": totals["action_delta_nll_sum"],
            "pc_sequences": len(grouped),
            "optimizer_steps": optimizer_steps,
        }
        history.append(row)
        print(
            "[train:compact-keyed-lstm] epoch={} trigger={:.8f} "
            "excess={:.8f} action={:.8f}".format(
                epoch, row["trigger_nll_per_callback"],
                row["positive_excess_nll_per_positive_callback"],
                row["action_delta_nll_per_action"],
            ),
            flush=True,
        )
    return history


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


def score_count_distribution(
    model, context_numpy, device, chunk_len=8192,
):
    trigger_parts = []
    excess_parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(context_numpy), chunk_len):
            stop = min(start + chunk_len, len(context_numpy))
            context = torch.from_numpy(context_numpy[start:stop]).to(device)
            trigger_parts.append(
                model.trigger_logit(context)
                .squeeze(-1).cpu().numpy()
            )
            excess_parts.append(
                model.log_positive_excess_mean(context)
                .squeeze(-1).cpu().numpy()
            )
    if not trigger_parts:
        return np.empty(0), np.empty(0)
    return (
        np.concatenate(trigger_parts, axis=0),
        np.concatenate(excess_parts, axis=0),
    )


def decode(
    model, context_numpy, counts, base_lines, event_keys, device,
    decoder_seed, role="eval", chunk_len=8192,
):
    counts = np.asarray(counts, dtype=np.int64)
    if not (
        len(context_numpy) == len(counts) == len(base_lines) == len(event_keys)
    ):
        raise RuntimeError("decoder row counts differ")
    if np.any(counts < 0):
        raise RuntimeError("negative decoded Stride request count")

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
            local_distributions = [[] for _ in range(len(local_counts))]
            steps = int(local_counts.max()) if len(local_counts) else 0
            for step in range(steps):
                active_numpy = np.flatnonzero(local_counts > step)
                if not len(active_numpy):
                    break
                active = torch.from_numpy(active_numpy).to(
                    device=device, dtype=torch.long
                )
                active_state = state.index_select(0, active)
                mix, mean, _ = model.action_decoder.distribution(active_state)
                probabilities = F.softmax(mix, dim=-1)
                feedback_coordinate = (probabilities * mean).sum(dim=-1)
                for local_position, probability_row, mean_row in zip(
                    active_numpy,
                    probabilities.cpu().numpy(), mean.cpu().numpy(),
                ):
                    local_distributions[int(local_position)].append(
                        (probability_row, mean_row)
                    )
                advanced = model.action_decoder.advance(
                    active_state, feedback_coordinate
                )
                state = state.index_copy(0, active, advanced)
            # Stateless key coordinates make callback/action traversal order
            # irrelevant.  Mean-major canonicalization also makes component
            # label permutations share the same inverse-CDF uniform.
            for local_position, distributions in enumerate(local_distributions):
                global_position = start + local_position
                for action_rank, (probabilities, means) in enumerate(
                    distributions
                ):
                    order = canonical_component_order(means)
                    uniform = keyed_uniform(
                        decoder_seed, TRACE, POLICY, role,
                        event_keys[global_position], "delta_component",
                        action_rank,
                    )
                    canonical_index = categorical_icdf(
                        probabilities[order], uniform,
                    )
                    component = int(order[canonical_index])
                    delta = _delta_to_integer(means[component])
                    predicted_lines[global_position].append(
                        apply_signed_line_delta(
                            base_lines[global_position], delta
                        )
                    )
                    predicted_fills[global_position].append(-1)
    return counts, predicted_lines, predicted_fills


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
    trigger_probabilities, excess_means, sampled_counts,
):
    """Report learned count marginals and realized count dispersion."""
    trigger_probabilities = np.asarray(
        trigger_probabilities, dtype=np.float64,
    )
    excess_means = np.asarray(excess_means, dtype=np.float64)
    sampled_counts = np.asarray(sampled_counts, dtype=np.int64)
    if not (
        len(trigger_probabilities) == len(excess_means)
        == len(sampled_counts)
    ):
        raise RuntimeError("count diagnostic vector lengths differ")
    expected_counts = trigger_probabilities * (1.0 + excess_means)
    second_moments = trigger_probabilities * (
        excess_means + np.square(1.0 + excess_means)
    )
    model_variances = second_moments - np.square(expected_counts)
    sampled_mean = float(sampled_counts.mean())
    sampled_variance = float(sampled_counts.var())
    positive = sampled_counts[sampled_counts > 0]
    distribution = Counter(int(value) for value in sampled_counts)
    return {
        "callbacks": int(len(sampled_counts)),
        "mean_trigger_probability": float(trigger_probabilities.mean()),
        "expected_positive_callbacks": float(trigger_probabilities.sum()),
        "mean_conditional_excess": float(excess_means.mean()),
        "expected_total_actions": float(expected_counts.sum()),
        "expected_mean_actions_per_callback": float(expected_counts.mean()),
        "mean_model_count_variance": float(model_variances.mean()),
        "sampled_positive_callbacks": int(np.count_nonzero(sampled_counts)),
        "sampled_total_actions": int(sampled_counts.sum()),
        "sampled_mean_actions_per_callback": sampled_mean,
        "sampled_count_variance": sampled_variance,
        "sampled_count_fano_factor": (
            sampled_variance / sampled_mean if sampled_mean else 0.0
        ),
        "sampled_mean_actions_per_positive_callback": (
            float(positive.mean()) if len(positive) else 0.0
        ),
        "sampled_max_actions_per_callback": (
            int(sampled_counts.max()) if len(sampled_counts) else 0
        ),
        "sampled_count_distribution": {
            str(key): int(value)
            for key, value in sorted(distribution.items())
        },
    }


def decoder_schedule_coordinates(event_keys, counts):
    """Yield every stateless inverse-CDF coordinate used by evaluation."""
    for event_key, count in zip(event_keys, counts):
        yield event_key, "request_trigger", 0
        yield event_key, "request_excess", 0
        for action_rank in range(int(count)):
            yield event_key, "delta_component", action_rank


def self_test_free_running_decoder(feature_count, hidden_size):
    torch.manual_seed(2718)
    model = CompactPCKeyedSampledStrideLSTM(feature_count, hidden_size)
    state = model.action_decoder.begin(torch.randn(3, hidden_size))
    predicted = model.action_decoder.feedback_coordinate(state)
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
        + (feature_count + 29) * hidden_size
        + 11
    )


def self_test_compact_parameter_count(feature_count, hidden_size):
    model = CompactPCKeyedSampledStrideLSTM(feature_count, hidden_size)
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


def self_test_event_keyed_count_and_mixture():
    self_test_keyed_crn()
    low_logits = np.full(4096, -4.0, dtype=np.float64)
    high_logits = np.full(4096, 4.0, dtype=np.float64)
    probabilities = _sigmoid_probabilities(
        np.concatenate([low_logits, high_logits])
    )
    means = np.ones(len(probabilities), dtype=np.float64)
    event_keys = list(range(len(probabilities)))
    first = event_keyed_hurdle_counts(
        probabilities, means, event_keys,
        1701, "self-test", POLICY, "eval",
    )
    second = event_keyed_hurdle_counts(
        probabilities, means, event_keys,
        1701, "self-test", POLICY, "eval",
    )
    if not np.array_equal(first, second):
        raise RuntimeError("event-keyed count sampling is not reproducible")
    if np.count_nonzero(first[4096:]) <= np.count_nonzero(first[:4096]):
        raise RuntimeError("event-keyed sampling ignored learned trigger mass")
    large = event_keyed_hurdle_counts(
        np.asarray([1.0]), np.asarray([256.0]), [0],
        2718, "self-test", POLICY, "eval",
    )
    if large[0] <= 2:
        raise RuntimeError("positive count support appears degree capped")

    mixture_probabilities = np.asarray([0.1, 0.6, 0.3])
    mixture_means = np.asarray([-2.0, 4.0, 0.5])
    permutation = np.asarray([2, 0, 1])
    uniform = keyed_uniform(
        1701, "self-test", POLICY, "eval", 4,
        "delta_component", 0,
    )
    first_order = canonical_component_order(mixture_means)
    first_choice = first_order[categorical_icdf(
        mixture_probabilities[first_order], uniform,
    )]
    permuted_probabilities = mixture_probabilities[permutation]
    permuted_means = mixture_means[permutation]
    second_order = canonical_component_order(permuted_means)
    second_choice = second_order[categorical_icdf(
        permuted_probabilities[second_order], uniform,
    )]
    if mixture_means[first_choice] != permuted_means[second_choice]:
        raise RuntimeError("canonical mixture sampling changed under relabeling")


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
    model = CompactPCKeyedSampledStrideLSTM(feature_count, hidden_size)
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
    parser.add_argument(
        "--decoder-seed", type=int, default=None,
        help="stateless CRN decoder seed (defaults to --seed)",
    )
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
    decoder_seed = args.seed if args.decoder_seed is None else args.decoder_seed
    expected_pair = MODEL_POINTS["lstm"].get(args.model_size)
    if expected_pair is None or args.pair_id != expected_pair:
        raise RuntimeError("model size/pair is not a configured LSTM point")
    if (
        args.model_size < 1 or args.epochs < 1 or args.chunk_len < 1
        or args.pc_batch_size < 1
    ):
        raise RuntimeError("model and batching dimensions must be positive")

    self_test_event_keyed_count_and_mixture()
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
    model = CompactPCKeyedSampledStrideLSTM(
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

    history = train_model(
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
    eval_trigger_logits, eval_log_excess = score_count_distribution(
        model, eval_context, device
    )
    eval_trigger_probabilities = _sigmoid_probabilities(eval_trigger_logits)
    eval_excess_means = _poisson_means(eval_log_excess)
    # The zero-based evaluation decision index is source chronology, not a
    # teacher-action identifier.  Only the evaluation suffix is sampled; train
    # and guard rows establish causal recurrent history but never burn RNG.
    eval_event_keys = range(len(rows["eval"]))
    predicted_counts = event_keyed_hurdle_counts(
        eval_trigger_probabilities, eval_excess_means, eval_event_keys,
        decoder_seed, TRACE, POLICY, "eval",
    )
    predicted_counts, predicted_lines, predicted_fills = decode(
        model, eval_context, predicted_counts,
        [line for _, line, _ in rows["eval"]], eval_event_keys,
        device, decoder_seed, role="eval",
    )
    count_diagnostics = count_decoder_diagnostics(
        eval_trigger_probabilities, eval_excess_means, predicted_counts,
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
        "decoder_seed": decoder_seed,
        "sampler_revision": SAMPLER_REVISION,
    }, args.out_dir / "model.pt")

    train_positive = int(np.count_nonzero(train_counts > 0))
    train_zero = int(len(train_counts) - train_positive)
    train_unique_pc_count = len(_group_indices_by_pc(rows["train"]))
    history_unique_pc_count = len(_group_indices_by_pc(history_rows))
    sampler_info = sampler_metadata()
    sampler_schedule_hash = sampling_schedule_sha256(
        decoder_seed, TRACE, POLICY, "eval",
        decoder_schedule_coordinates(eval_event_keys, predicted_counts),
    )
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
        "decoder_seed": decoder_seed,
        "stochastic_decoding_reproducible": True,
        "common_random_numbers_across_capacities": True,
        "strict_common_random_numbers_across_capacities": True,
        "cross_event_rng_state_used": False,
        "sampler_revision": SAMPLER_REVISION,
        "decoder_sampling_roles": ["eval"],
        "decoder_train_sampling_performed": False,
        "decoder_guard_sampling_performed": False,
        "decoder_event_key_definition": "zero_based_eval_demand_idx",
        "decoder_event_key_uses_teacher_information": False,
        "decoder_event_key_stream_sha256": key_stream_sha256(
            eval_event_keys
        ),
        "decoder_action_rank_origin": 0,
        "decoder_key_fields": sampler_info["key_fields"],
        "decoder_key_includes_sampler_revision": True,
        "decoder_sampler": sampler_info,
        "decoder_sampler_source_sha256": sampler_source_sha256(),
        "decoder_sampler_key_schedule_sha256": key_schedule_sha256(),
        "decoder_sampling_schedule_sha256": sampler_schedule_hash,
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
        "data_derived_gate_class_weights_used": False,
        "gate_class_weighting_used": False,
        "training_regularization_used": False,
        "inference_policy_hardcodes_used": False,
        "learned_request_count": True,
        "nn_generates_own_target_addresses": True,
        "complete_action_space": (
            "learned zero or unbounded positive count plus direct "
            "signed cache-line deltas"
        ),
        "decision_rule": (
            "stateless_sha256_keyed_bernoulli_and_poisson_inverse_cdf_"
            "then_canonicalized_categorical_delta_inverse_cdf"
        ),
        "gate_training_objective": "unweighted_bernoulli_nll",
        "gate_decoding_rule": "event_keyed_bernoulli_inverse_cdf",
        "request_count_training_objective": (
            "unweighted_bernoulli_hurdle_plus_positive_poisson_excess_nll"
        ),
        "request_count_decoding_rule": (
            "event_keyed_bernoulli_plus_common_quantile_poisson_inverse_cdf"
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
            "learned unweighted zero/positive Bernoulli plus conditional "
            "Poisson excess with unbounded nonnegative integer support"
        ),
        "loss_design": (
            "unweighted trigger NLL mean plus positive-excess Poisson NLL "
            "mean plus direct-delta mixture NLL mean; unit sum without "
            "manually tuned coefficients"
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
        "delta_mixture_decoding_rule": (
            "event_keyed_mean_sorted_categorical_inverse_cdf_then_"
            "component_mean"
        ),
        "delta_mixture_component_canonicalization": (
            "ascending_delta_mean_then_original_component_index"
        ),
        "delta_decoder_feedback_rule": (
            "complete_mixture_expectation_same_in_training_and_inference"
        ),
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
        "event_keyed_crn_self_test": "PASS",
        "event_keyed_hurdle_count_self_test": "PASS",
        "canonicalized_mixture_sampling_self_test": "PASS",
        "decoder_probability_mass_carries_train_guard_history": False,
        "cross_event_probability_credit_used": False,
        "sampled_outputs_used_as_decoder_feedback": False,
        "delta_mixture_components": MIXTURE_COMPONENTS,
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
