#!/usr/bin/env python3
"""Source-input-fair compact LSTM student for 623 SPP.

The model consumes exactly the chronological external callbacks used by source
SPP: DEMAND(addr) and CACHE_FILL(evicted_addr). Source SPP actions are
supervised labels and an offline comparator only; they never enter inference.

The v17 model keeps the 623-proven keyed hurdle/Poisson request count, but
factorizes target-address and fill placement.  A four-component direct-delta
mixture emits its modal component mean, while a dedicated learned fill head
uses a stateless keyed categorical draw so the rare L2 class cannot disappear
under joint MAP.  Free-running feedback uses the model's expected delta and
fill probabilities in both training and inference.  No threshold, degree cap,
candidate table, page class, source-SPP private state, or teacher action is an
inference input.
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
    targets_from_actions,
)
from formal_NN_training.common.keyed_sampling import (
    KEY_FIELDS, SAMPLER_REVISION,
    categorical_icdf, event_keyed_hurdle_counts, key_stream_sha256,
    key_schedule_sha256, keyed_uniform, sampler_metadata, sampler_source_sha256,
    sampling_schedule_sha256, self_test_keyed_crn,
)


TRACE = "623.xalancbmk_s-700B"
POLICY = "spp"
FILL_LEVELS = (2, 4)
MIXTURE_COMPONENTS = 4
LINE_ADDRESS_BITS = ADDRESS_BITS - CACHE_LINE_SHIFT
RUNTIME_FEATURES = LINE_ADDRESS_BITS + 1
EXPERIMENT_REVISION = "spp_source_input_variable_delta_fill_feedback_free_running_v11"
MODEL_REVISION = "compact_crn_factorized_delta_keyed_fill_v17"
OPERATION = "train-v17"
EVENT_LOGGER_SCHEMA = "623_causal_trigger_fill_v6"
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
                or line < 0 or line >= 1 << LINE_ADDRESS_BITS
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


def _unsigned_bits(values, width):
    """Return a lossless least-significant-bit-first unsigned bit matrix."""
    integers = [int(value) for value in values]
    if width < 1 or any(value < 0 or value >= 1 << width for value in integers):
        raise RuntimeError("runtime line number is outside the encoded domain")
    array = np.asarray(integers, dtype=np.uint64)
    shifts = np.arange(width, dtype=np.uint64)
    return ((array[:, None] >> shifts[None, :]) & 1).astype(np.float32)


def runtime_array(stream):
    context = stream["context"]
    line_bits = _unsigned_bits(
        [line for _, _, line, _ in context], LINE_ADDRESS_BITS
    )
    kinds = np.asarray([
        [1.0 if kind == "DEMAND" else 0.0]
        for kind, _, _, _ in context
    ], dtype=np.float32)
    return np.concatenate([line_bits, kinds], axis=1)


def runtime_encoder_sha256():
    payload = {
        "entrypoint_source": inspect.getsource(runtime_array),
        "primitive_source": inspect.getsource(_unsigned_bits),
        "fields": SOURCE_INPUTS,
        "use_pc": False,
        "address_bits": ADDRESS_BITS,
        "line_address_bits": LINE_ADDRESS_BITS,
        "cache_line_bytes": CACHE_LINE_BYTES,
        "bit_order": "least_significant_first",
        "removed_constant_alignment_bits": CACHE_LINE_SHIFT,
        "callback_kind_encoding": {"DEMAND": 1.0, "FILL": 0.0},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def sampling_event_keys(stream):
    """Build keys only from stable chronology and source-visible demand line."""
    keys = []
    for decision_idx, (position, demand) in enumerate(zip(
        stream["demand_positions"], stream["demands"]
    )):
        kind, _, context_line, routed_idx = stream["context"][int(position)]
        _, _, demand_line, _ = demand
        if (
            kind != "DEMAND" or routed_idx != decision_idx
            or context_line != demand_line
        ):
            raise RuntimeError("SPP decision router changed")
        keys.append("decision_idx={}|kind=DEMAND|line={}".format(
            decision_idx, demand_line
        ))
    return keys


def decision_router_sha256(stream):
    """Hash the chronological context-to-demand routing used by the model."""
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
    return hashlib.sha256(inspect.getsource(
        decision_router_sha256
    ).encode()).hexdigest()


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


def _poisson_means(log_rates):
    values = np.asarray(log_rates, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise RuntimeError("SPP positive-count log mean is not finite")
    limit = math.log(float(np.iinfo(np.int64).max))
    if np.any(values > limit):
        raise RuntimeError("SPP positive-count mean exceeds host domain")
    means = np.exp(values)
    if not np.all(np.isfinite(means)) or np.any(means < 0.0):
        raise RuntimeError("SPP positive-count mean is outside numeric domain")
    return means


def _sigmoid_probabilities(logits):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise RuntimeError("SPP trigger logits are not a finite vector")
    probabilities = np.empty_like(values)
    positive = values >= 0.0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    probabilities[~positive] = exp_values / (1.0 + exp_values)
    return probabilities


class CompactSPPActionDecoder(nn.Module):
    """Threshold-free factorized direct-delta and rare-fill decoder."""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.trigger_logit = nn.Linear(hidden_size, 1)
        self.log_positive_excess_mean = nn.Linear(hidden_size, 1)
        self.action_cell = nn.GRUCell(1 + len(FILL_LEVELS), hidden_size)
        self.delta_head = nn.Linear(hidden_size, 3 * MIXTURE_COMPONENTS)
        self.fill_head = nn.Linear(hidden_size, len(FILL_LEVELS))

    def distribution(self, state):
        raw = self.delta_head(state)
        mix, mean, raw_scale = raw.chunk(3, dim=-1)
        scale = F.softplus(raw_scale) + torch.finfo(raw_scale.dtype).tiny
        return mix, mean, scale, self.fill_head(state)

    @staticmethod
    def marginals(mix_logits, mean, fill_logits):
        component_probabilities = F.softmax(mix_logits, dim=-1)
        fill_probabilities = F.softmax(fill_logits, dim=-1)
        expected_coordinate = (
            component_probabilities * mean
        ).sum(dim=-1)
        return expected_coordinate, fill_probabilities

    def advance(
        self, state, predicted_coordinate, predicted_fill_probabilities,
    ):
        if predicted_fill_probabilities.shape[-1] != len(FILL_LEVELS):
            raise RuntimeError("predicted fill probability width changed")
        feedback = torch.cat([
            predicted_coordinate.reshape(-1, 1),
            predicted_fill_probabilities.to(predicted_coordinate.dtype),
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
    # LSTM(59,H), hurdle/excess heads, GRU(H,3), four-component delta
    # mixture, and a separate two-class fill head.
    return 7 * hidden_size * hidden_size + 275 * hidden_size + 16


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

    trigger_targets = (decision_counts > 0).to(decision_context.dtype)
    trigger_logits = model.decoder.trigger_logit(
        decision_context
    ).squeeze(-1)
    trigger_sum = F.binary_cross_entropy_with_logits(
        trigger_logits, trigger_targets, reduction="sum",
    )
    positive = decision_counts > 0
    positive_atoms = int(positive.sum().detach().item())
    excess_sum = context.new_zeros(())
    if positive_atoms:
        log_excess = model.decoder.log_positive_excess_mean(
            decision_context[positive]
        ).squeeze(-1)
        excess_targets = decision_counts[positive] - 1
        excess_sum = F.poisson_nll_loss(
            log_excess, excess_targets.to(log_excess.dtype), log_input=True,
            full=False, reduction="sum",
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
        mix_logits, mean, scale, fill_logits = model.decoder.distribution(
            active_state
        )
        target = decision_deltas[active, step]
        log_component = (
            -0.5 * ((target.unsqueeze(1) - mean) / scale).square()
            - torch.log(scale)
            - 0.5 * math.log(2.0 * math.pi)
        )
        delta_sum = delta_sum - torch.logsumexp(
            F.log_softmax(mix_logits, dim=-1) + log_component, dim=-1
        ).sum()
        target_fill = decision_fills[active, step]
        fill_sum = fill_sum + F.cross_entropy(
            fill_logits, target_fill, reduction="sum",
        )
        action_atoms += active_atoms

        # The loss uses teacher delta/fill labels, but recurrent feedback uses
        # the two learned marginals in both training and inference.  Neither
        # sampled outputs nor teacher actions are fed back.
        predicted_coordinate, predicted_fill_probabilities = (
            model.decoder.marginals(mix_logits, mean, fill_logits)
        )
        advanced = model.decoder.advance(
            active_state, predicted_coordinate,
            predicted_fill_probabilities,
        )
        state = state.index_copy(0, indices, advanced)

    mean_trigger = trigger_sum / float(decision_atoms)
    mean_excess = (
        excess_sum / float(positive_atoms)
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
    return mean_trigger + mean_excess + mean_delta + mean_fill, {
        "trigger_nll_sum": float(trigger_sum.detach().item()),
        "positive_excess_nll_sum": float(excess_sum.detach().item()),
        "delta_nll_sum": float(delta_sum.detach().item()),
        "fill_nll_sum": float(fill_sum.detach().item()),
        "decision_atoms": decision_atoms,
        "positive_atoms": positive_atoms,
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
            "trigger_nll_sum": 0.0,
            "positive_excess_nll_sum": 0.0,
            "delta_nll_sum": 0.0,
            "fill_nll_sum": 0.0,
            "decision_atoms": 0,
            "positive_atoms": 0,
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
            "trigger_nll_per_decision": (
                totals["trigger_nll_sum"]
                / max(1, totals["decision_atoms"])
            ),
            "positive_excess_nll_per_positive_decision": (
                totals["positive_excess_nll_sum"]
                / max(1, totals["positive_atoms"])
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
        print((
            "[train:spp-lstm] epoch={} trigger={:.8f} "
            "excess={:.8f} delta={:.8f} fill={:.8f}"
        ).format(
            epoch, row["trigger_nll_per_decision"],
            row["positive_excess_nll_per_positive_decision"],
            row["delta_nll_per_action"], row["fill_nll_per_action"],
        ))
    return history


def score_lstm(model, runtime, device, initial_state=None, chunk_len=8192):
    model.eval()
    trigger_parts = []
    excess_parts = []
    context_parts = []
    state = initial_state
    with torch.no_grad():
        for start, stop in _iter_chunks(len(runtime), chunk_len):
            xb = torch.from_numpy(runtime[start:stop]).unsqueeze(0).to(device)
            context, state = model.encode(xb, state)
            flat = context.squeeze(0)
            trigger_parts.append(
                model.decoder.trigger_logit(flat)
                .squeeze(-1).cpu().numpy()
            )
            excess_parts.append(
                model.decoder.log_positive_excess_mean(flat)
                .squeeze(-1).cpu().numpy()
            )
            context_parts.append(flat.cpu().numpy())
    return (
        np.concatenate(trigger_parts, axis=0),
        np.concatenate(excess_parts, axis=0),
        np.concatenate(context_parts, axis=0),
    ), state


def advance_lstm_state(model, runtime, device, initial_state=None):
    _, state = score_lstm(
        model, runtime, device, initial_state=initial_state
    )
    return state


def decode_actions(
    model, trigger_logits, log_excess_means, contexts, base_lines, device,
    event_keys, decoder_seed, role, materialize=True,
    chunk_len=8192,
):
    if not (
        len(trigger_logits) == len(log_excess_means)
        == len(contexts) == len(base_lines) == len(event_keys)
    ):
        raise RuntimeError("SPP decoder row counts differ")
    counts = event_keyed_hurdle_counts(
        _sigmoid_probabilities(trigger_logits),
        _poisson_means(log_excess_means),
        event_keys, decoder_seed, TRACE, POLICY, role,
    )
    predicted_lines = (
        [[] for _ in range(len(counts))] if materialize else None
    )
    predicted_fills = (
        [[] for _ in range(len(counts))] if materialize else None
    )
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
                mix_logits, mean, _, fill_logits = model.decoder.distribution(
                    active_state
                )
                feedback_coordinate, fill_probabilities = (
                    model.decoder.marginals(mix_logits, mean, fill_logits)
                )
                if materialize:
                    for local_position, mix_row, mean_row, fill_row in zip(
                        active_numpy, mix_logits.cpu().numpy(),
                        mean.cpu().numpy(), fill_probabilities.cpu().numpy(),
                    ):
                        global_position = start + int(local_position)
                        component = int(np.argmax(mix_row))
                        fill_choice = categorical_icdf(
                            np.asarray(fill_row, dtype=np.float64),
                            keyed_uniform(
                                decoder_seed, TRACE, POLICY, role,
                                event_keys[global_position],
                                "fill_level", step,
                            ),
                        )
                        delta = _coordinate_to_delta(mean_row[component])
                        predicted_lines[global_position].append(
                            apply_signed_line_delta(
                                base_lines[global_position], delta,
                            )
                        )
                        predicted_fills[global_position].append(fill_choice)
                advanced = model.decoder.advance(
                    active_state, feedback_coordinate, fill_probabilities
                )
                state = state.index_copy(0, active, advanced)
    return counts, predicted_lines, predicted_fills


def decoder_sampling_coordinates(event_keys, counts):
    """Enumerate the exact stateless coordinates consumed by this replay."""
    if len(event_keys) != len(counts):
        raise RuntimeError("SPP schedule row counts differ")
    coordinates = []
    for event_key, count in zip(event_keys, counts):
        count = int(count)
        if count < 0:
            raise RuntimeError("SPP schedule contains a negative action count")
        coordinates.append((event_key, "request_trigger", 0))
        coordinates.append((event_key, "request_excess", 0))
        for action_rank in range(count):
            coordinates.append((event_key, "fill_level", action_rank))
    return coordinates



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


def joint_action_metrics(predicted_lines, predicted_fills, teacher_actions):
    """Measure address/fill pairs, including the rare L2 class, exactly."""
    if not (
        len(predicted_lines) == len(predicted_fills) == len(teacher_actions)
    ):
        raise RuntimeError("SPP joint metric row counts differ")
    true_positive = 0
    predicted_total = 0
    teacher_total = 0
    l2_predicted = 0
    l2_teacher = 0
    l2_true_positive = 0
    for lines, fill_classes, truth_items in zip(
        predicted_lines, predicted_fills, teacher_actions
    ):
        if len(lines) != len(fill_classes):
            raise RuntimeError("SPP predicted line/fill counts differ")
        predicted = Counter()
        teacher = Counter()
        for line, fill_class in zip(lines, fill_classes):
            fill_class = int(fill_class)
            if fill_class < 0 or fill_class >= len(FILL_LEVELS):
                raise RuntimeError("SPP prediction uses unknown fill class")
            predicted[(int(line), FILL_LEVELS[fill_class])] += 1
        for line, fill_level in truth_items:
            if fill_level not in FILL_LEVELS:
                raise RuntimeError("SPP teacher uses unknown fill level")
            teacher[(int(line), int(fill_level))] += 1
        intersection = predicted & teacher
        true_positive += int(sum(intersection.values()))
        predicted_total += int(sum(predicted.values()))
        teacher_total += int(sum(teacher.values()))
        l2_predicted += int(sum(
            count for (_, fill), count in predicted.items() if fill == 2
        ))
        l2_teacher += int(sum(
            count for (_, fill), count in teacher.items() if fill == 2
        ))
        l2_true_positive += int(sum(
            count for (_, fill), count in intersection.items() if fill == 2
        ))

    def ratio(numerator, denominator):
        return float(numerator) / float(denominator) if denominator else 0.0

    precision = ratio(true_positive, predicted_total)
    recall = ratio(true_positive, teacher_total)
    l2_precision = ratio(l2_true_positive, l2_predicted)
    l2_recall = ratio(l2_true_positive, l2_teacher)
    return {
        "joint_true_positive_actions": true_positive,
        "joint_action_precision": precision,
        "joint_action_recall": recall,
        "joint_action_f1": ratio(
            2.0 * precision * recall, precision + recall
        ),
        "predicted_l2_actions": l2_predicted,
        "teacher_l2_actions": l2_teacher,
        "l2_joint_true_positive_actions": l2_true_positive,
        "l2_joint_precision": l2_precision,
        "l2_joint_recall": l2_recall,
        "l2_joint_f1": ratio(
            2.0 * l2_precision * l2_recall,
            l2_precision + l2_recall,
        ),
        "predicted_l2_fraction": ratio(l2_predicted, predicted_total),
        "teacher_l2_fraction": ratio(l2_teacher, teacher_total),
    }


def complete_behavior_metrics(
    predicted_counts, predicted_lines, predicted_fills, teacher_actions,
):
    metrics = behavior_metrics(
        predicted_counts, predicted_lines, predicted_fills,
        teacher_actions, fill_levels=FILL_LEVELS,
    )
    metrics.update(trigger_behavior_metrics(predicted_counts, teacher_actions))
    metrics.update(joint_action_metrics(
        predicted_lines, predicted_fills, teacher_actions
    ))
    return metrics


def joint_label_diagnostics(targets):
    """Summarize whether target delta coordinates depend on fill class."""
    counts, deltas, fills = targets
    coordinates = []
    fill_classes = []
    for row, count in enumerate(np.asarray(counts, dtype=np.int64)):
        if count < 0:
            continue
        for action_rank in range(int(count)):
            coordinate = float(deltas[row, action_rank])
            fill_class = int(fills[row, action_rank])
            if not math.isfinite(coordinate) or fill_class not in (0, 1):
                raise RuntimeError("invalid SPP joint training label")
            coordinates.append(coordinate)
            fill_classes.append(fill_class)
    coordinate_array = np.asarray(coordinates, dtype=np.float64)
    fill_array = np.asarray(fill_classes, dtype=np.int64)
    by_fill = {}
    for fill_class, fill_level in enumerate(FILL_LEVELS):
        selected = coordinate_array[fill_array == fill_class]
        by_fill[str(fill_level)] = {
            "actions": int(len(selected)),
            "mean_delta_coordinate": (
                float(selected.mean()) if len(selected) else None
            ),
            "std_delta_coordinate": (
                float(selected.std()) if len(selected) else None
            ),
        }
    correlation = None
    if (
        len(coordinate_array) > 1 and len(np.unique(fill_array)) == 2
        and float(coordinate_array.std()) > 0.0
    ):
        candidate = float(np.corrcoef(coordinate_array, fill_array)[0, 1])
        correlation = candidate if math.isfinite(candidate) else None
    return {
        "actions": int(len(coordinate_array)),
        "by_fill_level": by_fill,
        "delta_coordinate_fill_point_biserial_correlation": correlation,
    }


def self_test_model(hidden_size):
    expected_points = {
        8: 2664, 16: 6208, 32: 15984, 64: 46288, 128: 149904,
    }
    for size, expected in expected_points.items():
        point = CompactSPPLSTM(RUNTIME_FEATURES, size)
        observed = sum(parameter.numel() for parameter in point.parameters())
        if observed != expected or observed != expected_parameter_count(size):
            raise RuntimeError(
                "compact SPP parameter formula mismatch at h{}: {} != {}".format(
                    size, observed, expected
                )
            )

    model = CompactSPPLSTM(RUNTIME_FEATURES, hidden_size)
    parameter_names = {name for name, _ in model.named_parameters()}
    for required in (
        "decoder.delta_head.weight", "decoder.fill_head.weight",
        "decoder.trigger_logit.weight",
        "decoder.log_positive_excess_mean.weight",
    ):
        if required not in parameter_names:
            raise RuntimeError("missing factorized SPP head {}".format(required))

    self_test_keyed_crn()
    low_logits = np.full(4096, -4.0, dtype=np.float64)
    high_logits = np.full(4096, 4.0, dtype=np.float64)
    log_excess = np.zeros(4096, dtype=np.float64)
    test_keys = ["self_test={}".format(index) for index in range(8192)]
    probabilities = _sigmoid_probabilities(
        np.concatenate([low_logits, high_logits])
    )
    excess_means = _poisson_means(
        np.concatenate([log_excess, log_excess])
    )
    test_counts = event_keyed_hurdle_counts(
        probabilities, excess_means, test_keys, 1701,
        TRACE, POLICY, "self_test",
    )
    repeated_counts = event_keyed_hurdle_counts(
        probabilities, excess_means, test_keys, 1701,
        TRACE, POLICY, "self_test",
    )
    if not np.array_equal(test_counts, repeated_counts):
        raise RuntimeError("SPP keyed count sampling is not reproducible")
    if np.count_nonzero(test_counts[4096:]) <= np.count_nonzero(
        test_counts[:4096]
    ):
        raise RuntimeError("SPP sampling ignored learned trigger mass")
    large_counts = event_keyed_hurdle_counts(
        np.asarray([1.0]), np.asarray([256.0]), ["large_count"],
        2718, TRACE, POLICY, "self_test",
    )
    if large_counts[0] <= 2:
        raise RuntimeError("SPP positive count support appears degree capped")

    if categorical_icdf(np.asarray([0.02, 0.98]), 0.01) != 0:
        raise RuntimeError("SPP keyed fill draw lost the rare L2 class")
    if categorical_icdf(np.asarray([0.02, 0.98]), 0.50) != 1:
        raise RuntimeError("SPP keyed fill draw lost the LLC class")

    loss_source = inspect.getsource(_structured_loss)
    for forbidden in (
        "advance(active_state, target",
        "gate_" + "class_weights",
        "weight" + "=",
        "joint_action_head",
    ):
        if forbidden in loss_source:
            raise RuntimeError(
                "structured SPP loss contains forbidden weighting/feedback"
            )
    for required in (
        "F.cross_entropy", "torch.logsumexp",
        "model.decoder.marginals",
    ):
        if required not in loss_source:
            raise RuntimeError("factorized SPP training evidence missing")

    decoder_source = inspect.getsource(decode_actions)
    for required in (
        "categorical_icdf", "keyed_uniform", '"fill_level"',
        "np.argmax(mix_row)",
    ):
        if required not in decoder_source:
            raise RuntimeError("factorized keyed SPP decoder evidence missing")
    if (
        "np.argmax(fill" in decoder_source
        or "raw_event_id" in decoder_source
        or "RandomState" in decoder_source
    ):
        raise RuntimeError("SPP fill decoder uses forbidden MAP/key/RNG state")

    callback_metrics = joint_action_metrics(
        [[11], []], [[0], []], [[], [(11, 2)]]
    )
    if callback_metrics["joint_true_positive_actions"] != 0:
        raise RuntimeError(
            "SPP behavior metric matched across callback identities"
        )

    encoded = _unsigned_bits(
        [0, 1, (1 << LINE_ADDRESS_BITS) - 1], LINE_ADDRESS_BITS
    )
    if encoded.shape != (3, LINE_ADDRESS_BITS):
        raise RuntimeError("compact SPP line encoder width changed")
    if (
        encoded[0].any() or encoded[1, 0] != 1.0
        or encoded[1, 1:].any() or not encoded[2].all()
    ):
        raise RuntimeError("compact SPP line encoder is not lossless")

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
    if family != "lstm" or size not in MODEL_POINTS["lstm"]:
        raise RuntimeError("unsupported SPP model point")
    return "factorized_delta_fill_spp_lstm_h{}".format(size)


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
    parser.add_argument(
        "--decoder-seed", type=int, default=None,
        help="stateless decoder key seed; defaults to --seed",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--chunk-len", type=int, default=1024)
    parser.add_argument("--accumulate-chunks", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.decoder_seed is None:
        args.decoder_seed = args.seed

    source_contract = json.loads(args.source_contract.read_text())
    if (
        source_contract.get("decision_effective_external_input")
        != SOURCE_INPUTS
    ):
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
        role: load_teacher_actions(
            action_paths[role], streams[role]["demands"]
        )
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
    event_keys = {
        role: sampling_event_keys(streams[role]) for role in roles
    }
    if any(
        len(event_keys[role]) != len(streams[role]["demands"])
        for role in roles
    ):
        raise RuntimeError("SPP keyed sampler router row count changed")

    decision_targets = {
        role: targets_from_actions(
            [line for _, _, line, _ in streams[role]["demands"]],
            normal[role], fill_levels=FILL_LEVELS,
        )
        for role in roles
    }
    training_joint_label_diagnostics = joint_label_diagnostics(
        decision_targets["train"]
    )
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
    model.to(device)

    train_counts, train_deltas, train_fills = context_targets["train"]
    train_decision_counts = train_counts[train_counts >= 0]
    train_positive_callbacks = int((train_decision_counts > 0).sum())
    request_count_training_label_statistics = {
        "decision_callbacks": int(len(train_decision_counts)),
        "positive_callbacks": train_positive_callbacks,
        "zero_callbacks": (
            int(len(train_decision_counts)) - train_positive_callbacks
        ),
        "positive_callback_rate": (
            float(train_positive_callbacks)
            / float(len(train_decision_counts))
        ),
    }
    history = train_model(
        model, runtime["train"], train_counts, train_deltas, train_fills,
        len(streams["train"]["context"]), device, args.epochs,
        args.chunk_len, args.accumulate_chunks, args.learning_rate,
    )

    # Re-run the exact train -> guard -> eval chronology from a fresh state.
    # Guard warms causal history and remains an audit split; it selects no
    # threshold, budget, fill rule, decoder mode, or checkpoint.
    encoded = {}
    recurrent_state = None
    for role in roles:
        encoded[role], recurrent_state = score_lstm(
            model, runtime[role], device, initial_state=recurrent_state
        )

    demand_positions = streams["eval"]["demand_positions"]
    trigger_logits, log_excess_means, contexts = (
        value[demand_positions] for value in encoded["eval"]
    )
    base_lines = [
        line for _, _, line, _ in streams["eval"]["demands"]
    ]
    predicted_counts, predicted_lines, predicted_fills = decode_actions(
        model, trigger_logits, log_excess_means, contexts, base_lines,
        device, event_keys["eval"], args.decoder_seed, "eval",
        materialize=True,
    )
    decoder_coordinates = decoder_sampling_coordinates(
        event_keys["eval"], predicted_counts
    )
    behavior = complete_behavior_metrics(
        predicted_counts, predicted_lines, predicted_fills,
        normal["eval"],
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    normal_path = args.out_dir / "offline_spp.replay.csv"
    nn_path = args.out_dir / "offline_nn.replay.csv"
    normal_entries, normal_triggers, normal_fill_counts = (
        write_teacher_replay(
            normal_path, streams["eval"]["demands"], normal["eval"]
        )
    )
    nn_entries, nn_triggers, nn_fill_counts = write_prediction_replay(
        nn_path, streams["eval"]["demands"],
        predicted_lines, predicted_fills,
    )
    history_path = args.out_dir / "training_history.csv"
    model_path = args.out_dir / "model.pt"
    write_table(history_path, history)
    torch.save({
        "state_dict": model.state_dict(),
        "model_family": "lstm",
        "model_size": args.model_size,
        "runtime_features": RUNTIME_FEATURES,
        "fill_levels": FILL_LEVELS,
        "mixture_components": MIXTURE_COMPONENTS,
        "factorized_delta_fill_heads": True,
        "decoder_seed": args.decoder_seed,
        "sampler_revision": SAMPLER_REVISION,
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": MODEL_REVISION,
    }, model_path)

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
        "parameter_formula": "7*H^2 + 275*H + 16",
        "configured_parameter_counts": {
            "8": 2664, "16": 6208, "32": 15984,
            "64": 46288, "128": 149904,
        },
        "parameter_storage_bytes_float32": parameter_count * 4,
        "peak_recurrent_state_bytes": args.model_size * 2 * 4,
        "peak_persistent_recurrent_state_bytes": args.model_size * 2 * 4,
        "peak_training_recurrent_state_bytes_float32": (
            args.model_size * 2 * 4
        ),
        "peak_inference_recurrent_state_bytes_float32": (
            args.model_size * 2 * 4
        ),
        "recurrent_state_bytes_per_global_stream": args.model_size * 2 * 4,
        "recurrent_state_dtype": "float32",
        "persistent_recurrent_state": (
            "one global float32 LSTM hidden/cell pair"
        ),
        "seed": args.seed,
        "decoder_seed": args.decoder_seed,
        "operation": OPERATION,
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": MODEL_REVISION,
        "decoder_revision": "factorized_delta_keyed_fill_v17",
        "weights_retrained": True,
        "checkpoint_reused": False,
        "decoder_only_change": False,
        "guard_selected_decoder": False,
        "joint_map_used": False,
        "selected_decoder_mode": (
            "deterministic_modal_delta_component_plus_keyed_fill_draw"
        ),
        "decoder_candidate_modes": [],
        "decoder_sampler": sampler_metadata(),
        "sampler_revision": SAMPLER_REVISION,
        "decoder_sampler_revision": SAMPLER_REVISION,
        "decoder_sampler_source_sha256": sampler_source_sha256(),
        "decoder_sampler_key_schedule_sha256": key_schedule_sha256(),
        "decoder_sampler_key_fields": list(KEY_FIELDS),
        "decoder_key_fields": list(KEY_FIELDS),
        "decoder_event_key_fields": [
            "decision_index", "constant_DEMAND_kind", "source_line",
        ],
        "decoder_event_key_definition": (
            "zero_based_role_decision_idx_plus_source_line"
        ),
        "decoder_event_key_uses_teacher_information": False,
        "decoder_key_includes_sampler_revision": True,
        "decoder_forbidden_key_fields": ["pc", "raw_teacher_event_id"],
        "decoder_additional_forbidden_key_fields": [
            "raw_event_id", "pf_event_id", "teacher_action_gap",
        ],
        "decoder_action_rank_origin": 0,
        "decoder_eval_event_key_stream_sha256": key_stream_sha256(
            event_keys["eval"]
        ),
        "decoder_eval_sampling_schedule_sha256": sampling_schedule_sha256(
            args.decoder_seed, TRACE, POLICY, "eval", decoder_coordinates
        ),
        "decoder_eval_sampling_coordinates": len(decoder_coordinates),
        "decoder_guard_sampling_schedule_sha256": None,
        "decoder_guard_sampling_coordinates": 0,
        "common_random_numbers_across_capacities": True,
        "strict_common_random_numbers_across_capacities": True,
        "cross_event_rng_state_used": False,
        "train_guard_decoder_rng_burn_used": False,
        "decoder_sampling_roles": ["eval"],
        "decoder_roles_sampled": ["eval"],
        "decoder_train_sampling_performed": False,
        "decoder_guard_sampling_performed": False,
        "decoder_action_sampling_performed": True,
        "decoder_count_sampling_performed": True,
        "keyed_sampling_self_test": "PASS",
        "stochastic_decoding": (
            "stateless SHA-256 event-keyed hurdle/count/fill inverse-CDF "
            "sampling; deterministic modal delta component mean"
        ),
        "stochastic_decoding_reproducible": True,
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
        "guard_role": "causal_input_history_warmup_and_audit_only",
        "runtime_feature_count": RUNTIME_FEATURES,
        "runtime_encoding": (
            "lossless 58-bit cache-line number plus one DEMAND/FILL kind bit"
        ),
        "runtime_address_alignment_bits_removed": CACHE_LINE_SHIFT,
        "runtime_address_alignment_bits_were_constant_zero": True,
        "source_decision_effective_external_input": SOURCE_INPUTS,
        "same_external_input_contract": True,
        "training_inference_input_encoder_identical": True,
        "decoder_training_mode": (
            "free_running_autoregressive_same_as_inference"
        ),
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_free_running_self_test": "PASS",
        "runtime_encoder_entrypoint": (
            "623_offline_lstm_spp.train_and_offline_infer.runtime_array"
        ),
        "runtime_encoder_sha256": runtime_encoder_sha256(),
        "training_runtime_encoder_sha256": runtime_encoder_sha256(),
        "inference_runtime_encoder_sha256": runtime_encoder_sha256(),
        "training_runtime_fields": SOURCE_INPUTS,
        "inference_runtime_fields": SOURCE_INPUTS,
        "decision_router_source_sha256": decision_router_source_sha256(),
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
            "event_keyed_bernoulli_then_conditional_poisson_count; "
            "deterministic_modal_delta_component_mean; "
            "event_keyed_fill_categorical_inverse_cdf"
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
        "gate_training_objective": "unweighted_bernoulli_nll",
        "gate_decoding_rule": "event_keyed_bernoulli_inverse_cdf",
        "gate_operating_point_learned_from_empirical_prior": False,
        "request_count_training_objective": (
            "unweighted_bernoulli_hurdle_plus_positive_poisson_excess_nll"
        ),
        "request_count_decoding_rule": (
            "event_keyed_bernoulli_plus_common_quantile_poisson_inverse_cdf"
        ),
        "request_count_residual_scope": "none_event_local",
        "request_count_training_label_statistics": (
            request_count_training_label_statistics
        ),
        "joint_delta_fill_dependency_modeled": False,
        "joint_delta_fill_class_count": 0,
        "joint_pair_classes": 0,
        "joint_delta_fill_training_objective": None,
        "joint_delta_fill_decoding_rule": None,
        "joint_component_canonicalization": None,
        "delta_mixture_components": MIXTURE_COMPONENTS,
        "decoder_mixture_components": MIXTURE_COMPONENTS,
        "delta_training_objective": (
            "four_component_signed_log_delta_mixture_nll"
        ),
        "delta_mixture_decoding_rule": (
            "deterministic_modal_component_then_component_mean"
        ),
        "delta_decoding_rule": (
            "deterministic_modal_component_then_component_mean"
        ),
        "fill_training_objective": (
            "unweighted_two_class_cross_entropy"
        ),
        "fill_decoding_rule": (
            "event_keyed_categorical_inverse_cdf"
        ),
        "fill_argmax_used": False,
        "fill_probability_feedback_used": True,
        "decoder_probability_mass_carries_train_guard_history": False,
        "cross_event_probability_credit_used": False,
        "sampled_outputs_used_as_decoder_feedback": False,
        "delta_decoder_feedback_rule": (
            "factorized_distribution_expectation_same_in_training_and_inference"
        ),
        "loss_design": (
            "unweighted trigger NLL mean plus positive-excess Poisson NLL "
            "mean plus delta-mixture NLL mean plus fill CE mean; "
            "unit sum with no manually tuned coefficients"
        ),
        "training_regularization_used": False,
        "inference_policy_hardcodes_used": False,
        "learned_request_count": True,
        "address_interface_bits": ADDRESS_BITS,
        "factorized_delta_fill_heads": True,
        "training_joint_label_diagnostics": (
            training_joint_label_diagnostics
        ),
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
        "causal_no_future_self_test": "PASS",
        "event_local_hurdle_count_self_test": "PASS",
        "factorized_delta_fill_sampling_self_test": "PASS",
        "joint_delta_fill_sampling_self_test": "NOT_APPLICABLE",
        "cnn_architecture_self_test": "NOT_APPLICABLE",
        "cnn_temporal_layers": 0,
        "cnn_kernel_size": 0,
        "cnn_stride": 0,
        "cnn_dilations": [],
        "cnn_receptive_field_events": 0,
        "training_left_context_overlap": 0,
        "cnn_processes_complete_stream_in_order": False,
        "cnn_chunking_changes_visible_history": False,
        "event_logger_schema": EVENT_LOGGER_SCHEMA,
        "action_attachment_mode": ACTION_ATTACHMENT_MODE,
        "canonicalization_mode": CANONICALIZATION_MODE,
        "teacher_action_canonicalization": CANONICALIZATION_MODE,
        "replay_preserves_explicit_fill_level": True,
        "same_source_input_offline_claim_allowed": True,
        "closed_loop_live_claim_allowed": False,
        "offline_input_feedback_origin": (
            "recorded cache-fill callbacks produced by the source SPP run"
        ),
        "comparison_claim_boundary": (
            "matched-input open-loop offline comparison only; live NN actions "
            "were not used to regenerate cache-fill feedback"
        ),
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
        "model_checkpoint_sha256": sha256(model_path),
        "training_history_sha256": sha256(history_path),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    for role in roles:
        metadata[role + "_decision_router_sha256"] = (
            decision_router_sha256(streams[role])
        )
        metadata[role + "_decoder_event_key_stream_sha256"] = (
            key_stream_sha256(event_keys[role])
        )
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
        "offline_nn_fill_level_counts": nn_fill_counts,
    }, indent=2))


if __name__ == "__main__":
    run_cli()
