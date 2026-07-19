#!/usr/bin/env python3
"""Source-input-fair compact LSTM student for 602 SPP.

The model consumes exactly the chronological external callbacks used by source
SPP: DEMAND(addr) and CACHE_FILL(evicted_addr).  Source SPP actions are
supervised labels and an offline comparator only; they never enter inference.
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
from collections import defaultdict
from pathlib import Path

# Make direct execution robust in Colab and on Sacramento.  Python otherwise
# puts only this nested script directory on sys.path, not the repository root.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from formal_NN_training.common.threshold_free_policy import (
    ADDRESS_BITS, CACHE_LINE_BYTES, CACHE_LINE_SHIFT,
    apply_signed_line_delta, behavior_metrics, expand_targets,
    runtime_bits, targets_from_actions,
)


TRACE = "602.gcc_s-734B"
POLICY = "spp"
FILL_LEVELS = (2, 4)
MIXTURE_COMPONENTS = 4
RUNTIME_FEATURES = ADDRESS_BITS + 1
EXPERIMENT_REVISION = "spp_source_input_compact_empirical_prior_hurdle_delta_fill_free_running_v2"
MODEL_REVISION = "compact_empirical_prior_hurdle_autoregressive_gmm_fill_v2"
EVENT_LOGGER_SCHEMA = "602_spp_causal_trigger_fill_v1"
ACTION_ATTACHMENT_MODE = "explicit_trigger_event_id"
CANONICALIZATION_MODE = "per_target_min_fill_queue_effect"
SOURCE_INPUTS = [
    "callback_kind", "invoke_prefetcher.addr", "cache_fill.evicted_addr",
]
MODEL_POINTS = {
    "lstm": {8: "p0", 16: "p1", 32: "p2", 64: "p3", 128: "p4"},
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
    try:
        return int(text, 0)
    except ValueError:
        return int(float(text))


def load_stream(path):
    context = []
    demands = []
    demand_positions = []
    occurrences = defaultdict(int)
    last_raw_event_id = -1
    last_cycle = -1
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
                    or occurrence != expected or hit not in (0, 1)
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
            last_raw_event_id = raw_event_id
            last_cycle = cycle
    if not context or not demands:
        raise RuntimeError("empty stream {}".format(path))
    if len(context) == len(demands):
        raise RuntimeError("SPP stream contains no cache-fill feedback")
    return {
        "context": context,
        "demands": demands,
        "demand_positions": np.asarray(demand_positions, dtype=np.int64),
    }


def load_teacher_actions(path, rows):
    """Audit source SPP outputs without leaking their topology to the NN."""
    actions = [[] for _ in rows]
    last_pf_event = -1
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "trace", "policy", "demand_idx", "pc", "line", "pc_line_occ",
            "action_rank", "pf_line", "fill_level",
            "accepted", "duplicate", "trigger_event_id", "pf_event_id",
            "event_distance", "raw_action_count", "source_first_pf_event_id",
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
                row["trace"] != TRACE or row["policy"] != POLICY
                or (
                    as_int(row["pc"]), as_int(row["line"]),
                    as_int(row["pc_line_occ"]),
                ) != (pc, line, occurrence)
                or row["logger_schema"] != EVENT_LOGGER_SCHEMA
                or row["match_mode"] != ACTION_ATTACHMENT_MODE
            ):
                raise RuntimeError(
                    "teacher action identity failure at {}".format(index)
                )
            if as_int(row["action_rank"]) != len(actions[index]) + 1:
                raise RuntimeError(
                    "noncontiguous action rank at {}".format(index)
                )
            pf_event = as_int(row["pf_event_id"])
            trigger = as_int(row["trigger_event_id"])
            distance = as_int(row["event_distance"])
            if (
                pf_event <= last_pf_event or trigger >= pf_event
                or distance != pf_event - trigger
            ):
                raise RuntimeError("invalid action attachment at {}".format(index))
            pf_line = as_int(row["pf_line"])
            fill = as_int(row["fill_level"])
            if (
                fill not in FILL_LEVELS
                or as_int(row["accepted"]) != 1
                or as_int(row["duplicate"]) not in (0, 1)
                or as_int(row["raw_action_count"]) < 1
                or as_int(row["source_first_pf_event_id"]) != pf_event
                or as_int(row["source_last_pf_event_id"]) < pf_event
                or as_int(row["is_self_target"]) != int(pf_line == line)
                or row["canonicalization"] != CANONICALIZATION_MODE
            ):
                raise RuntimeError(
                    "invalid captured SPP action at {}".format(index)
                )
            if any(existing_line == pf_line for existing_line, _ in actions[index]):
                raise RuntimeError(
                    "two fill choices for one target at {}".format(index)
                )
            actions[index].append((pf_line, fill))
            last_pf_event = pf_event
    if not any(actions):
        raise RuntimeError("empty teacher action stream {}".format(path))
    return actions


def runtime_array(stream):
    context = stream["context"]
    addresses = runtime_bits(
        [0 for _ in context], [address for _, address, _, _ in context], False
    )
    kinds = np.asarray([
        [1.0 if kind == "DEMAND" else 0.0]
        for kind, _, _, _ in context
    ], dtype=np.float32)
    return np.concatenate([addresses, kinds], axis=1)


def runtime_encoder_sha256():
    payload = {
        "entrypoint_source": inspect.getsource(runtime_array),
        "primitive_source": inspect.getsource(runtime_bits),
        "fields": SOURCE_INPUTS,
        "use_pc": False,
        "address_bits": ADDRESS_BITS,
        "cache_line_bytes": CACHE_LINE_BYTES,
        "callback_kind_encoding": {"DEMAND": 1.0, "FILL": 0.0},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def write_table(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_teacher_replay(path, rows, actions):
    entries = 0
    triggers = 0
    fill_counts = {"FILL_L2": 0, "FILL_LLC": 0}
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pc", "line", "occ", "prefetch_addr", "fill_level"])
        for (pc, _, line, occurrence), items in zip(rows, actions):
            if items:
                triggers += 1
            for pf_line, fill in items:
                writer.writerow([
                    pc, line, occurrence,
                    hex(pf_line << CACHE_LINE_SHIFT), fill,
                ])
                fill_counts["FILL_L2" if fill == 2 else "FILL_LLC"] += 1
                entries += 1
    return entries, triggers, fill_counts


def write_prediction_replay(path, rows, predicted_lines, predicted_fills):
    entries = 0
    triggers = 0
    fill_counts = {"FILL_L2": 0, "FILL_LLC": 0}
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pc", "line", "occ", "prefetch_addr", "fill_level"])
        for (pc, _, line, occurrence), targets, fills in zip(
            rows, predicted_lines, predicted_fills
        ):
            if targets:
                triggers += 1
            for pf_line, fill_index in zip(targets, fills):
                if fill_index < 0 or fill_index >= len(FILL_LEVELS):
                    raise RuntimeError("neural fill class is out of range")
                fill = FILL_LEVELS[fill_index]
                writer.writerow([
                    pc, line, occurrence,
                    hex(pf_line << CACHE_LINE_SHIFT), fill,
                ])
                fill_counts["FILL_L2" if fill == 2 else "FILL_LLC"] += 1
                entries += 1
    return entries, triggers, fill_counts



def _coordinate_to_delta(coordinate):
    coordinate = float(coordinate)
    if not math.isfinite(coordinate):
        raise RuntimeError("neural delta coordinate is not finite")
    try:
        magnitude = math.expm1(abs(coordinate))
    except OverflowError as exc:
        raise RuntimeError("neural delta exceeds uint64 address domain") from exc
    half_range = 1 << (ADDRESS_BITS - CACHE_LINE_SHIFT - 1)
    if not math.isfinite(magnitude) or magnitude > half_range:
        raise RuntimeError("neural delta exceeds uint64 address domain")
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


class CompactSPPActionDecoder(nn.Module):
    """Threshold-free variable-cardinality direct-delta/fill decoder."""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.emit_head = nn.Linear(hidden_size, 2)
        self.positive_log_count = nn.Linear(hidden_size, 1)
        self.action_cell = nn.GRUCell(1 + len(FILL_LEVELS), hidden_size)
        self.delta_head = nn.Linear(
            hidden_size, 3 * MIXTURE_COMPONENTS
        )
        self.fill_head = nn.Linear(hidden_size, len(FILL_LEVELS))

    def distribution(self, state):
        raw = self.delta_head(state)
        mix, mean, raw_scale = raw.chunk(3, dim=-1)
        scale = F.softplus(raw_scale) + torch.finfo(raw_scale.dtype).tiny
        return mix, mean, scale, self.fill_head(state)

    def advance(self, state, predicted_coordinate, predicted_fill):
        feedback = torch.cat([
            predicted_coordinate.reshape(-1, 1),
            F.one_hot(
                predicted_fill.to(torch.long), len(FILL_LEVELS)
            ).to(predicted_coordinate.dtype),
        ], dim=-1)
        return self.action_cell(feedback, state)


class CompactSPPLSTM(nn.Module):
    """One global chronological LSTM; raw eviction callbacks update its state."""

    def __init__(self, feature_count, hidden_size):
        super().__init__()
        if feature_count < 1 or hidden_size < 1:
            raise ValueError("model dimensions must be positive")
        self.feature_count = feature_count
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(feature_count, hidden_size, batch_first=True)
        self.decoder = CompactSPPActionDecoder(hidden_size)

    def encode(self, runtime, state=None):
        return self.lstm(runtime, state)


def expected_parameter_count(hidden_size):
    # LSTM(65,H), two-class gate, positive-count head, GRU(H,3),
    # four-component delta mixture, and two-class fill head.
    return 7 * hidden_size * hidden_size + 300 * hidden_size + 17


def _detach_state(state):
    return tuple(value.detach() for value in state)


def _iter_chunks(length, width):
    for start in range(0, length, width):
        yield start, min(length, start + width)


def _structured_loss(model, context, counts, deltas, fills):
    flat_context = context.reshape(-1, context.shape[-1])
    flat_counts = counts.reshape(-1)
    flat_deltas = deltas.reshape(-1, deltas.shape[-1])
    flat_fills = fills.reshape(-1, fills.shape[-1])
    valid = flat_counts >= 0
    decision_atoms = int(valid.sum().detach().item())
    if not decision_atoms:
        raise RuntimeError("training chunk has no SPP decision callbacks")

    decision_context = flat_context[valid]
    decision_counts = flat_counts[valid]
    decision_deltas = flat_deltas[valid]
    decision_fills = flat_fills[valid]

    emit_targets = (decision_counts > 0).to(torch.long)
    # Unweighted maximum likelihood preserves the observed SPP trigger prior.
    # Inverse-frequency balancing upweighted rare zero labels and moved raw
    # categorical argmax to the severe under-prefetch operating point observed
    # in v1.  v2 also records the exact training split for direct verification.
    gate_sum = F.cross_entropy(
        model.decoder.emit_head(decision_context), emit_targets,
        reduction="sum",
    )

    positive = decision_counts > 0
    positive_atoms = int(positive.sum().detach().item())
    count_sum = context.new_zeros(())
    if positive_atoms:
        predicted_log_count = model.decoder.positive_log_count(
            decision_context[positive]
        ).squeeze(-1)
        target_log_count = torch.log(
            decision_counts[positive].to(predicted_log_count.dtype)
        )
        count_sum = F.smooth_l1_loss(
            predicted_log_count, target_log_count, reduction="sum"
        )

    delta_sum = context.new_zeros(())
    fill_sum = context.new_zeros(())
    action_atoms = 0
    state = decision_context
    for step in range(decision_deltas.shape[1]):
        active = decision_counts > step
        active_atoms = int(active.sum().detach().item())
        if not active_atoms:
            break
        indices = torch.nonzero(active, as_tuple=False).squeeze(1)
        active_state = state.index_select(0, indices)
        mix, mean, scale, fill_logits = model.decoder.distribution(
            active_state
        )
        target = decision_deltas[active, step]
        log_component = (
            -0.5 * ((target.unsqueeze(1) - mean) / scale).square()
            - torch.log(scale)
            - 0.5 * math.log(2.0 * math.pi)
        )
        delta_sum = delta_sum - torch.logsumexp(
            F.log_softmax(mix, dim=-1) + log_component, dim=-1
        ).sum()
        target_fill = decision_fills[active, step]
        fill_sum = fill_sum + F.cross_entropy(
            fill_logits, target_fill, reduction="sum"
        )
        action_atoms += active_atoms

        # Teacher targets above are loss-only.  Recurrent feedback is the
        # model's own modal delta and fill, exactly as in offline inference.
        component = mix.argmax(dim=-1, keepdim=True)
        predicted_coordinate = mean.gather(1, component).squeeze(1)
        predicted_fill = fill_logits.argmax(dim=-1)
        advanced = model.decoder.advance(
            active_state, predicted_coordinate, predicted_fill
        )
        state = state.index_copy(0, indices, advanced)

    mean_gate = gate_sum / float(decision_atoms)
    mean_count = (
        count_sum / float(positive_atoms)
        if positive_atoms else context.new_zeros(())
    )
    mean_delta = (
        delta_sum / float(action_atoms)
        if action_atoms else context.new_zeros(())
    )
    mean_fill = (
        fill_sum / float(action_atoms)
        if action_atoms else context.new_zeros(())
    )
    return mean_gate + mean_count + mean_delta + mean_fill, {
        "gate_loss_sum": float(gate_sum.detach().item()),
        "positive_count_loss_sum": float(count_sum.detach().item()),
        "delta_nll_sum": float(delta_sum.detach().item()),
        "fill_nll_sum": float(fill_sum.detach().item()),
        "decision_atoms": decision_atoms,
        "positive_count_atoms": positive_atoms,
        "action_atoms": action_atoms,
    }


def train_model(
    model, runtime, counts, deltas, fills, fit_end, device, epochs,
    chunk_len, accumulate_chunks, learning_rate,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    x = torch.from_numpy(runtime)
    count_tensor = torch.from_numpy(counts).to(torch.long)
    delta_tensor = torch.from_numpy(deltas).to(torch.float32)
    fill_tensor = torch.from_numpy(fills).to(torch.long)
    chunks = list(_iter_chunks(fit_end, chunk_len))
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        state = None
        totals = {
            "gate_loss_sum": 0.0,
            "positive_count_loss_sum": 0.0,
            "delta_nll_sum": 0.0,
            "fill_nll_sum": 0.0,
            "decision_atoms": 0,
            "positive_count_atoms": 0,
            "action_atoms": 0,
        }
        steps = 0
        for group_start in range(0, len(chunks), accumulate_chunks):
            group = chunks[group_start:group_start + accumulate_chunks]
            decision_chunks = sum(
                int(np.any(counts[start:stop] >= 0))
                for start, stop in group
            )
            optimizer.zero_grad(set_to_none=True)
            for start, stop in group:
                xb = x[start:stop].unsqueeze(0).to(device)
                cb = count_tensor[start:stop].unsqueeze(0).to(device)
                db = delta_tensor[start:stop].unsqueeze(0).to(device)
                fb = fill_tensor[start:stop].unsqueeze(0).to(device)
                context, state = model.encode(xb, state)
                state = _detach_state(state)
                if not np.any(counts[start:stop] >= 0):
                    # Fill-only chunks still advance causal recurrent state,
                    # but they carry no SPP decision loss.
                    continue
                loss, components = _structured_loss(
                    model, context, cb, db, fb
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite SPP training loss")
                (loss / float(max(1, decision_chunks))).backward()
                for key, value in components.items():
                    totals[key] += value
            optimizer.step()
            steps += 1
        row = {
            "epoch": epoch,
            "gate_loss_per_decision": (
                totals["gate_loss_sum"]
                / max(1, totals["decision_atoms"])
            ),
            "positive_count_loss_per_positive": (
                totals["positive_count_loss_sum"]
                / max(1, totals["positive_count_atoms"])
            ),
            "delta_nll_per_action": (
                totals["delta_nll_sum"]
                / max(1, totals["action_atoms"])
            ),
            "fill_nll_per_action": (
                totals["fill_nll_sum"]
                / max(1, totals["action_atoms"])
            ),
            "chronological_chunks": len(chunks),
            "optimizer_steps": steps,
        }
        history.append(row)
        print("[train:spp-lstm] epoch={} gate={:.8f} count={:.8f} delta={:.8f} fill={:.8f}".format(
            epoch, row["gate_loss_per_decision"],
            row["positive_count_loss_per_positive"],
            row["delta_nll_per_action"], row["fill_nll_per_action"],
        ))
    return history


def score_lstm(model, runtime, device, initial_state=None, chunk_len=8192):
    model.eval()
    emit_parts = []
    count_parts = []
    context_parts = []
    state = initial_state
    with torch.no_grad():
        for start, stop in _iter_chunks(len(runtime), chunk_len):
            xb = torch.from_numpy(runtime[start:stop]).unsqueeze(0).to(device)
            context, state = model.encode(xb, state)
            flat = context.squeeze(0)
            emit_parts.append(
                model.decoder.emit_head(flat).cpu().numpy()
            )
            count_parts.append(
                model.decoder.positive_log_count(flat)
                .squeeze(-1).cpu().numpy()
            )
            context_parts.append(flat.cpu().numpy())
    return (
        np.concatenate(emit_parts, axis=0),
        np.concatenate(count_parts, axis=0),
        np.concatenate(context_parts, axis=0),
    ), state


def advance_lstm_state(model, runtime, device, initial_state=None):
    _, state = score_lstm(
        model, runtime, device, initial_state=initial_state
    )
    return state


def decode_actions(
    model, emit_logits, positive_log_counts, contexts, base_lines, device,
    chunk_len=8192,
):
    if not (
        len(emit_logits) == len(positive_log_counts)
        == len(contexts) == len(base_lines)
    ):
        raise RuntimeError("SPP decoder row counts differ")
    positive_counts = _positive_counts_from_log_mean(positive_log_counts)
    gates = np.asarray(emit_logits).argmax(axis=1)
    counts = np.where(gates == 1, positive_counts, 0).astype(np.int64)
    predicted_lines = [[] for _ in range(len(counts))]
    predicted_fills = [[] for _ in range(len(counts))]
    model.eval()
    with torch.no_grad():
        for start, stop in _iter_chunks(len(counts), chunk_len):
            state = torch.from_numpy(contexts[start:stop]).to(device)
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
                mix, mean, _, fill_logits = model.decoder.distribution(
                    active_state
                )
                component = mix.argmax(dim=-1, keepdim=True)
                coordinate = mean.gather(1, component).squeeze(1)
                fill_choice = fill_logits.argmax(dim=-1)
                coordinate_cpu = coordinate.cpu().numpy()
                fill_cpu = fill_choice.cpu().numpy()
                for local_position, value, fill in zip(
                    active_numpy, coordinate_cpu, fill_cpu
                ):
                    global_position = start + int(local_position)
                    predicted_lines[global_position].append(
                        apply_signed_line_delta(
                            base_lines[global_position],
                            _coordinate_to_delta(value),
                        )
                    )
                    predicted_fills[global_position].append(int(fill))
                advanced = model.decoder.advance(
                    active_state, coordinate, fill_choice
                )
                state = state.index_copy(0, active, advanced)
    return counts, predicted_lines, predicted_fills



def trigger_behavior_metrics(predicted_counts, teacher_actions):
    """Report trigger calibration separately from target-address quality."""
    predicted = np.asarray(predicted_counts, dtype=np.int64)
    normal = np.asarray(
        [len(items) for items in teacher_actions], dtype=np.int64
    )
    if predicted.shape != normal.shape or predicted.ndim != 1:
        raise RuntimeError("SPP trigger behavior rows differ")
    if np.any(predicted < 0):
        raise RuntimeError("negative predicted SPP request count")
    predicted_positive = predicted > 0
    normal_positive = normal > 0
    true_positive = int(np.logical_and(predicted_positive, normal_positive).sum())
    false_positive = int(
        np.logical_and(predicted_positive, np.logical_not(normal_positive)).sum()
    )
    false_negative = int(
        np.logical_and(np.logical_not(predicted_positive), normal_positive).sum()
    )
    normal_positive_count = int(normal_positive.sum())
    predicted_positive_count = int(predicted_positive.sum())
    callbacks = int(len(normal))

    def ratio(numerator, denominator):
        return float(numerator) / float(denominator) if denominator else 0.0

    precision = ratio(true_positive, true_positive + false_positive)
    recall = ratio(true_positive, true_positive + false_negative)
    return {
        "normal_positive_callbacks": normal_positive_count,
        "normal_zero_callbacks": callbacks - normal_positive_count,
        "predicted_positive_callbacks": predicted_positive_count,
        "true_positive_trigger_callbacks": true_positive,
        "false_positive_trigger_callbacks": false_positive,
        "false_negative_trigger_callbacks": false_negative,
        "normal_positive_callback_rate": ratio(normal_positive_count, callbacks),
        "predicted_positive_callback_rate": ratio(
            predicted_positive_count, callbacks
        ),
        "trigger_precision": precision,
        "trigger_recall": recall,
        "trigger_f1": ratio(2.0 * precision * recall, precision + recall),
        "mean_normal_actions_per_positive_callback": ratio(
            int(normal.sum()), normal_positive_count
        ),
        "mean_predicted_actions_per_positive_callback": ratio(
            int(predicted.sum()), predicted_positive_count
        ),
        "predicted_to_normal_action_ratio": ratio(
            int(predicted.sum()), int(normal.sum())
        ),
    }


def self_test_model(hidden_size):
    model = CompactSPPLSTM(RUNTIME_FEATURES, hidden_size)
    observed = sum(parameter.numel() for parameter in model.parameters())
    expected = expected_parameter_count(hidden_size)
    if observed != expected:
        raise RuntimeError(
            "compact SPP parameter formula mismatch: {} != {}".format(
                observed, expected
            )
        )
    test_counts = _positive_counts_from_log_mean(
        np.log(np.asarray([1.0, 2.0, 9.0, 257.0]))
    )
    if test_counts.tolist() != [1, 2, 9, 257]:
        raise RuntimeError("positive SPP count support is not variable")
    loss_source = inspect.getsource(_structured_loss)
    forbidden = (
        "advance(active_state, target",
        "advance(active_state, target_fill",
        "gate_" + "class_weights",
        "weight" + "=",
    )
    if any(token in loss_source for token in forbidden):
        raise RuntimeError(
            "structured SPP loss contains forbidden feedback or weighting"
        )
    required = (
        "active_state, predicted_coordinate, predicted_fill",
    )
    if any(token not in loss_source for token in required):
        raise RuntimeError("free-running decoder feedback evidence missing")

    model.eval()
    prefix = np.zeros((1, 5, RUNTIME_FEATURES), dtype=np.float32)
    changed = prefix.copy()
    changed[0, 4, 0] = 1.0
    with torch.no_grad():
        first, _ = model.encode(torch.from_numpy(prefix))
        second, _ = model.encode(torch.from_numpy(changed))
    if not torch.equal(first[:, :4], second[:, :4]):
        raise RuntimeError("future callback changed a prior LSTM state")


def model_tag(family, size):
    if family != "lstm":
        raise RuntimeError("602 SPP track is LSTM-only")
    return "independent_delta_spp_lstm_h{}".format(size)


def run_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, choices=[POLICY])
    for role in ("train", "guard", "eval"):
        parser.add_argument("--{}-stream".format(role), required=True, type=Path)
        parser.add_argument(
            "--{}-teacher-actions".format(role), required=True, type=Path
        )
    parser.add_argument("--source-contract", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-family", choices=["lstm"], required=True)
    parser.add_argument("--model-size", type=int, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--chunk-len", type=int, default=1024)
    parser.add_argument("--accumulate-chunks", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    source_contract = json.loads(args.source_contract.read_text())
    if source_contract.get("decision_effective_external_input") != SOURCE_INPUTS:
        raise RuntimeError("unexpected SPP source input contract")
    expected_pair = MODEL_POINTS["lstm"].get(args.model_size)
    if expected_pair is None or args.pair_id != expected_pair:
        raise RuntimeError("model size/pair is not a configured point")
    if (
        args.model_size < 1 or args.epochs < 1 or args.chunk_len < 1
        or args.accumulate_chunks < 1
    ):
        raise RuntimeError("model/training dimensions must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    self_test_model(args.model_size)

    roles = ("train", "guard", "eval")
    stream_paths = {role: getattr(args, role + "_stream") for role in roles}
    action_paths = {
        role: getattr(args, role + "_teacher_actions") for role in roles
    }
    streams = {role: load_stream(stream_paths[role]) for role in roles}
    normal = {
        role: load_teacher_actions(action_paths[role], streams[role]["demands"])
        for role in roles
    }
    runtime = {role: runtime_array(streams[role]) for role in roles}
    if any(value.shape[1] != RUNTIME_FEATURES for value in runtime.values()):
        raise RuntimeError("lossless callback-kind/address encoding changed")
    for role in roles:
        if not np.array_equal(runtime[role], runtime_array(streams[role])):
            raise RuntimeError(
                "{} training/inference runtime encoder differs".format(role)
            )

    decision_targets = {
        role: targets_from_actions(
            [line for _, _, line, _ in streams[role]["demands"]],
            normal[role], fill_levels=FILL_LEVELS,
        )
        for role in roles
    }
    context_targets = {
        role: expand_targets(
            decision_targets[role], streams[role]["demand_positions"],
            len(streams[role]["context"]),
        )
        for role in roles
    }

    model = CompactSPPLSTM(RUNTIME_FEATURES, args.model_size)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    if parameter_count != expected_parameter_count(args.model_size):
        raise RuntimeError("measured compact SPP parameter count changed")

    train_counts, train_deltas, train_fills = context_targets["train"]
    train_decision_counts = train_counts[train_counts >= 0]
    train_positive_callbacks = int((train_decision_counts > 0).sum())
    gate_training_label_statistics = {
        "decision_callbacks": int(len(train_decision_counts)),
        "positive_callbacks": train_positive_callbacks,
        "zero_callbacks": (
            int(len(train_decision_counts)) - train_positive_callbacks
        ),
        "positive_callback_rate": (
            float(train_positive_callbacks) / float(len(train_decision_counts))
        ),
    }
    history = train_model(
        model, runtime["train"], train_counts, train_deltas, train_fills,
        len(streams["train"]["context"]), device, args.epochs,
        args.chunk_len, args.accumulate_chunks, args.learning_rate,
    )

    history_state = advance_lstm_state(
        model, runtime["train"], device
    )
    guard_state = advance_lstm_state(
        model, runtime["guard"], device, initial_state=history_state
    )
    eval_encoded, _ = score_lstm(
        model, runtime["eval"], device, initial_state=guard_state
    )
    demand_positions = streams["eval"]["demand_positions"]
    emit_logits, positive_log_counts, contexts = (
        value[demand_positions] for value in eval_encoded
    )
    base_lines = [
        line for _, _, line, _ in streams["eval"]["demands"]
    ]
    predicted_counts, predicted_lines, predicted_fills = decode_actions(
        model, emit_logits, positive_log_counts, contexts,
        base_lines, device,
    )
    behavior = behavior_metrics(
        predicted_counts, predicted_lines, predicted_fills,
        normal["eval"], fill_levels=FILL_LEVELS,
    )
    behavior.update(trigger_behavior_metrics(
        predicted_counts, normal["eval"]
    ))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    normal_path = args.out_dir / "offline_spp.replay.csv"
    nn_path = args.out_dir / "offline_nn.replay.csv"
    normal_entries, normal_triggers, normal_fill_counts = write_teacher_replay(
        normal_path, streams["eval"]["demands"], normal["eval"]
    )
    nn_entries, nn_triggers, nn_fill_counts = write_prediction_replay(
        nn_path, streams["eval"]["demands"],
        predicted_lines, predicted_fills,
    )
    write_table(args.out_dir / "training_history.csv", history)
    torch.save({
        "state_dict": model.state_dict(),
        "model_family": "lstm",
        "model_size": args.model_size,
        "runtime_features": RUNTIME_FEATURES,
        "fill_levels": FILL_LEVELS,
        "mixture_components": MIXTURE_COMPONENTS,
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": MODEL_REVISION,
    }, args.out_dir / "model.pt")

    tag = model_tag("lstm", args.model_size)
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
        "seed": args.seed,
        "epochs": args.epochs,
        "chunk_len": args.chunk_len,
        "accumulate_chunks": args.accumulate_chunks,
        "learning_rate": args.learning_rate,
        "guard_rows": len(streams["guard"]["context"]),
        "eval_rows": len(streams["eval"]["context"]),
        "guard_demand_callbacks": len(streams["guard"]["demands"]),
        "eval_demand_callbacks": len(streams["eval"]["demands"]),
        "guard_cache_fill_callbacks": (
            len(streams["guard"]["context"])
            - len(streams["guard"]["demands"])
        ),
        "eval_cache_fill_callbacks": (
            len(streams["eval"]["context"])
            - len(streams["eval"]["demands"])
        ),
        "runtime_feature_count": RUNTIME_FEATURES,
        "runtime_encoding": (
            "lossless callback-address uint64 bits plus callback-kind bit"
        ),
        "source_decision_effective_external_input": SOURCE_INPUTS,
        "same_external_input_contract": True,
        "training_inference_input_encoder_identical": True,
        "decoder_training_mode": (
            "free_running_autoregressive_same_as_inference"
        ),
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_free_running_self_test": "PASS",
        "runtime_encoder_entrypoint": (
            "602_offline_lstm_spp.train_and_offline_infer.runtime_array"
        ),
        "runtime_encoder_sha256": runtime_encoder_sha256(),
        "training_runtime_encoder_sha256": runtime_encoder_sha256(),
        "inference_runtime_encoder_sha256": runtime_encoder_sha256(),
        "training_runtime_fields": SOURCE_INPUTS,
        "inference_runtime_fields": SOURCE_INPUTS,
        "model_does_not_use_pc": True,
        "pc_is_replay_transport_only": True,
        "model_input_is_causal_external_event_sequence_only": True,
        "cache_fill_feedback_used_as_raw_external_input": True,
        "cache_fill_private_state_used_as_model_input": False,
        "cache_hit_and_type_are_audit_only": True,
        "teacher_actions_are_model_inputs": False,
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_request_rate_used_as_budget": False,
        "normal_policy_constants_used_by_neural_inference": False,
        "teacher_same_page_property_used_only_for_source_output_audit": True,
        "nn_generates_own_target_addresses_and_fill_levels": True,
        "complete_action_space": (
            "zero-or-unbounded-positive count plus direct signed "
            "cache-line deltas and learned fill"
        ),
        "decision_rule": (
            "two_class_argmax_then_rounded_exp_positive_count_then_"
            "autoregressive_delta_mixture_and_fill_modes"
        ),
        "probability_threshold_used": False,
        "threshold_related_hardcodes_used": False,
        "neural_degree_cap": None,
        "fixed_page_offset_classes": None,
        "same_page_rule_used_by_neural_inference": False,
        "future_label_window_used": False,
        "fill_lead_cutoff_used": False,
        "handcrafted_semantic_features_used": False,
        "manual_loss_weights_used": False,
        "gate_class_weighting_used": False,
        "gate_training_objective": (
            "empirical_prior_unweighted_categorical_nll"
        ),
        "gate_decoding_rule": "two_class_categorical_argmax",
        "gate_operating_point_learned_from_empirical_prior": True,
        "gate_training_label_statistics": gate_training_label_statistics,
        "loss_design": (
            "empirical-prior unweighted hurdle NLL mean plus positive "
            "log-count mean plus direct-delta mixture NLL mean plus fill "
            "NLL mean; unit sum with no manually tuned coefficients"
        ),
        "training_regularization_used": False,
        "inference_policy_hardcodes_used": False,
        "learned_request_count": True,
        "address_interface_bits": ADDRESS_BITS,
        "cache_line_bytes": CACHE_LINE_BYTES,
        "decoder_mixture_components": MIXTURE_COMPONENTS,
        "eviction_feedback_role": (
            "raw chronological input event only; no private SPP state "
            "and no separate eviction prediction target"
        ),
        "training_labels": (
            "canonicalized source-SPP actions and fill; supervision only"
        ),
        "teacher_action_files_role": (
            "normal replay, supervised labels, and audit only"
        ),
        "forbidden_inputs": [
            "normal_actions_at_inference", "SPP_signature_tables",
            "pattern_tables", "normal_thresholds", "normal_degree",
            "global_history_register_contents", "prefetch_filter_contents",
            "cycle", "cache_hit", "access_type", "queue_state",
            "future_rows",
        ],
        "training_chunks_shuffled": False,
        "training_state_mode": "chronological_stateful_tbptt",
        "training_state_carried_across_chunks": True,
        "training_state_detached_between_chunks": True,
        "inference_history_mode": (
            "fresh_state_then_complete_train_guard_eval_chronology"
        ),
        "cnn_architecture_self_test": "NOT_APPLICABLE",
        "causal_no_future_self_test": "PASS",
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
        "action_attachment_mode": ACTION_ATTACHMENT_MODE,
        "canonicalization_mode": CANONICALIZATION_MODE,
        "teacher_action_canonicalization": CANONICALIZATION_MODE,
        "replay_preserves_explicit_fill_level": True,
        "source_contract_sha256": sha256(args.source_contract),
        "offline_normal_entries": normal_entries,
        "offline_normal_triggers": normal_triggers,
        "offline_normal_fill_counts": normal_fill_counts,
        "offline_normal_fill_level_counts": normal_fill_counts,
        "offline_nn_entries": nn_entries,
        "offline_nn_triggers": nn_triggers,
        "offline_nn_fill_counts": nn_fill_counts,
        "offline_nn_fill_level_counts": nn_fill_counts,
        "normal_list_sha256": sha256(normal_path),
        "nn_list_sha256": sha256(nn_path),
        "heldout_behavior_metrics": behavior,
        "train_history": history,
        "source_contract": source_contract,
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
        metadata[role + "_teacher_actions_gzip_sha256"] = sha256(
            action_paths[role]
        )
        metadata[
            role + "_teacher_actions_content_sha256"
        ] = gzip_content_sha256(action_paths[role])
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
    run_cli()
