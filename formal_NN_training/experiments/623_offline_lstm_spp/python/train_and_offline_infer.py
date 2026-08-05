#!/usr/bin/env python3
"""Train and decode the rankwise STOP/EMIT 623 SPP v21 prefetcher.

Runtime input is only the chronological source-visible callback stream:
DEMAND(addr) and CACHE_FILL(evicted_addr).  PC is replay transport.  Normal
SPP actions and fill levels are labels and comparator data only.

The model deliberately has no normal-SPP template: no page rule, candidate
bank, SPP threshold, signature table, action feedback, or teacher-forced
decoder state.  A single global LSTM learns the sequence.  At every rank an
independent binary head chooses STOP or EMIT; each teacher callback includes
one supervised terminal STOP.  Delta and fill are trained only at EMIT ranks.
Inference is deterministic categorical argmax for STOP/EMIT, delta, and fill.
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

# CUDA deterministic GEMM configuration must exist before importing torch.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import model_contract as model_contract_module
from model_contract import (
    ACCUMULATE_CHUNKS, ADDRESS_BITS, CACHE_LINE_BYTES, CACHE_LINE_SHIFT,
    CHUNK_LEN, DECODER_REVISION, EPOCHS,
    DECODER_TRAINING_MODE, DELTA_OBJECTIVE, EXPERIMENT_REVISION,
    EMIT_TOKEN, EXTERNAL_INPUT_FIELDS, FILL_LEVELS, FILL_OBJECTIVE,
    LINE_ADDRESS_BITS, LINE_ADDRESS_MODULUS, MAX_EXACT_DELTAS, MODEL_POINTS,
    LEARNING_RATE, MODEL_REVISION, OPERATION, POLICY, RANK_CODE_SIZE,
    RUNTIME_FEATURE_COUNT, RUN_ID, SEED, STOP_TOKEN, TOKEN_OBJECTIVE, TRACE,
    delta_embed_size,
    describe_model_points, exact_int as as_int,
    expected_parameter_count, model_tag, self_test_exact_int,
)

# The replay server must be able to inspect and validate a Colab archive on a
# CPU-only machine.  Keep these two contract-only paths ahead of numpy/torch.
if __name__ == "__main__" and sys.argv[1:] == ["--describe-model-points"]:
    print(json.dumps(describe_model_points(), indent=2, sort_keys=True))
    raise SystemExit(0)
if __name__ == "__main__" and sys.argv[1:] == ["--self-test"]:
    self_test_exact_int()
    expected_sizes = [8, 16, 32, 64, 128]
    if sorted(MODEL_POINTS["lstm"]) != expected_sizes:
        raise RuntimeError("SPP v21 capacity contract changed")
    for hidden_size in expected_sizes:
        if expected_parameter_count(hidden_size, MAX_EXACT_DELTAS) < 1:
            raise RuntimeError("SPP v21 parameter contract is invalid")
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
OTHER_NAME = "OTHER"

if RUNTIME_FEATURES != RUNTIME_FEATURE_COUNT:
    raise RuntimeError("SPP v21 runtime feature contract changed")


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
            if pf_event <= last_pf_event or trigger >= pf_event or as_int(row["event_distance"]) != pf_event - trigger:
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
    return {"features": features, "lines": lines, "demand_kind": kinds}


def runtime_encoder_sha256():
    payload = {
        "entrypoint_source": inspect.getsource(runtime_bundle),
        "primitive_source": inspect.getsource(_unsigned_bits),
        "fields": SOURCE_INPUTS, "use_pc": False,
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
        "demand_positions": [int(value) for value in stream["demand_positions"]],
        "decision_indices": [int(stream["context"][int(pos)][3]) for pos in stream["demand_positions"]],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def decision_router_source_sha256():
    return hashlib.sha256(inspect.getsource(decision_router_sha256).encode()).hexdigest()


def canonical_signed_delta(base, target):
    difference = (int(target) - int(base)) % LINE_ADDRESS_MODULUS
    return difference - LINE_ADDRESS_MODULUS if difference >= LINE_ADDRESS_HALF_RANGE else difference


def signed_log(delta):
    value = int(delta)
    return math.copysign(math.log1p(abs(value)), value) if value else 0.0


def inverse_signed_log(value):
    scalar = float(value)
    maximum = math.log1p(LINE_ADDRESS_HALF_RANGE)
    scalar = max(-maximum, min(maximum, scalar))
    magnitude = int(round(math.expm1(abs(scalar))))
    delta = -magnitude if scalar < 0 else magnitude
    return max(-LINE_ADDRESS_HALF_RANGE, min(LINE_ADDRESS_HALF_RANGE - 1, delta))


def build_delta_vocabulary(train_stream, train_actions):
    require_equal_lengths("TRAIN vocabulary", train_stream["demands"], train_actions)
    frequencies = Counter()
    for demand, items in zip(train_stream["demands"], train_actions):
        base = demand[2]
        frequencies.update(canonical_signed_delta(base, target) for target, _ in items)
    ordered = sorted(frequencies, key=lambda value: (-frequencies[value], value))
    exact = ordered[:MAX_EXACT_DELTAS]
    if not exact:
        raise RuntimeError("TRAIN delta vocabulary is empty")
    return exact, frequencies


def vocabulary_statistics(stream, actions, exact_vocabulary):
    exact = set(exact_vocabulary)
    frequencies = Counter()
    for demand, items in zip(stream["demands"], actions):
        frequencies.update(canonical_signed_delta(demand[2], target) for target, _ in items)
    total = sum(frequencies.values())
    in_vocab = sum(count for value, count in frequencies.items() if value in exact)
    return {
        "action_count": total, "unique_signed_deltas": len(frequencies),
        "in_vocabulary_actions": in_vocab, "other_actions": total - in_vocab,
        "in_vocabulary_fraction": in_vocab / float(total) if total else 0.0,
        "other_fraction": (total - in_vocab) / float(total) if total else 0.0,
    }


def build_context_targets(stream, actions, vocabulary):
    require_equal_lengths("teacher decision targets", stream["demand_positions"], stream["demands"], actions)
    counts = np.full(len(stream["context"]), -1, dtype=np.int64)
    width = max(1, max(len(items) for items in actions))
    classes = np.full((len(counts), width), -1, dtype=np.int64)
    signed_logs = np.zeros((len(counts), width), dtype=np.float32)
    target_lines = np.full((len(counts), width), -1, dtype=np.int64)
    fills = np.full((len(counts), width), -1, dtype=np.int64)
    class_by_delta = {value: index for index, value in enumerate(vocabulary)}
    other_class = len(vocabulary)
    fill_to_index = {value: index for index, value in enumerate(FILL_LEVELS)}
    for decision, position in enumerate(stream["demand_positions"]):
        items, base = actions[decision], stream["demands"][decision][2]
        counts[position] = len(items)
        for rank, (target, fill) in enumerate(items):
            delta = canonical_signed_delta(base, target)
            classes[position, rank] = class_by_delta.get(delta, other_class)
            signed_logs[position, rank] = signed_log(delta)
            target_lines[position, rank] = target
            fills[position, rank] = fill_to_index[fill]
    return counts, classes, signed_logs, target_lines, fills


def write_table(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


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


def _iter_chunks(length, width):
    for start in range(0, length, width):
        yield start, min(length, start + width)


def rank_code(ranks, dtype):
    ranks = ranks.to(dtype)
    frequencies = ranks.new_tensor([1.0, 0.01])
    phase = ranks.unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)


class GlobalSPPLSTM(nn.Module):
    def __init__(self, hidden_size, vocabulary_size):
        super().__init__()
        if hidden_size not in MODEL_POINTS["lstm"] or not 0 < vocabulary_size <= MAX_EXACT_DELTAS:
            raise ValueError("unsupported independent SPP dimensions")
        self.hidden_size = int(hidden_size)
        self.vocabulary_size = int(vocabulary_size)
        self.class_count = self.vocabulary_size + 1
        self.embed_size = delta_embed_size(hidden_size)
        self.input_projection = nn.Linear(RUNTIME_FEATURES, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.rank_fusion = nn.Linear(hidden_size + RANK_CODE_SIZE, hidden_size)
        self.stop_emit = nn.Linear(hidden_size, 2)
        self.delta_class = nn.Linear(hidden_size, self.class_count)
        self.delta_signed_log = nn.Linear(hidden_size, 1)
        self.delta_embedding = nn.Embedding(self.class_count, self.embed_size)
        self.fill = nn.Linear(hidden_size + RANK_CODE_SIZE + self.embed_size + 1, 2)

    def initialize_label_priors(self, token_counts, delta_class_counts):
        with torch.no_grad():
            tokens = torch.as_tensor(token_counts, dtype=self.stop_emit.bias.dtype)
            if bool((tokens <= 0).any()):
                raise RuntimeError("both natural STOP/EMIT classes require TRAIN support")
            self.stop_emit.bias.copy_(torch.log(tokens / tokens.sum()))
            classes = torch.as_tensor(delta_class_counts, dtype=self.delta_class.bias.dtype)
            # Add-one is a probability estimator, not a decision threshold; it
            # keeps the TRAIN-unseen OTHER escape representable.
            self.delta_class.bias.copy_(torch.log((classes + 1.0) / (classes.sum() + len(classes))))

    def encode(self, features, state=None):
        embedded = torch.tanh(self.input_projection(features))
        output, state = self.lstm(embedded.unsqueeze(0), state)
        return output.squeeze(0), state

    def ranked_state(self, contexts, ranks):
        code = rank_code(ranks, contexts.dtype)
        state = torch.tanh(self.rank_fusion(torch.cat([contexts, code], dim=1)))
        return state, code

    def action_heads(self, contexts, ranks):
        state, code = self.ranked_state(contexts, ranks)
        return self.delta_class(state), self.delta_signed_log(state).squeeze(1), state, code

    def token_logits(self, contexts, ranks):
        state, _ = self.ranked_state(contexts, ranks)
        return self.stop_emit(state)

    def fill_logits(self, ranked_state, code, delta_classes, delta_signed_logs):
        embedded = self.delta_embedding(delta_classes)
        return self.fill(torch.cat([
            ranked_state, code, embedded, delta_signed_logs.unsqueeze(1),
        ], dim=1))


def detach_state(state):
    if state is None:
        return None
    return tuple(value.detach() for value in state)


def training_priors(targets, class_count):
    counts, classes, _, _, fills = targets
    decisions = counts >= 0
    # Every decision contributes exactly one terminal STOP and one EMIT for
    # each teacher action.  The natural token prior is dense enough that no
    # class weighting is warranted.
    token_counts = np.asarray([
        int(decisions.sum()), int(counts[decisions].sum())
    ], dtype=np.int64)
    delta_counts = np.zeros(class_count, dtype=np.int64)
    fill_counts = np.zeros(2, dtype=np.int64)
    for row in np.flatnonzero(decisions):
        for rank in range(int(counts[row])):
            delta_counts[int(classes[row, rank])] += 1
            fill_counts[int(fills[row, rank])] += 1
    if not bool((token_counts > 0).all()):
        raise RuntimeError("both STOP and EMIT require TRAIN support")
    if not bool((fill_counts > 0).all()):
        raise RuntimeError("both fill classes require TRAIN support")
    fill_priors = fill_counts.astype(np.float64) / float(fill_counts.sum())
    fill_weights = 0.5 / fill_priors
    return {
        "stop_emit_counts": token_counts,
        "delta_class_counts": delta_counts, "fill_counts": fill_counts,
        "fill_priors": fill_priors, "fill_weights": fill_weights,
    }


def chunk_loss(model, contexts, targets, fill_weights, device):
    counts_np, classes_np, signed_logs_np, _, fills_np = targets
    counts = torch.from_numpy(counts_np).to(device=device, dtype=torch.long)
    decision = counts >= 0
    if not bool(decision.any()):
        return None, None
    decision_context = contexts[decision]
    decision_rows = torch.nonzero(decision, as_tuple=False).squeeze(1).cpu().tolist()
    token_rows, token_ranks, token_truth = [], [], []
    action_rows, action_ranks = [], []
    for local_row, original_row in enumerate(decision_rows):
        action_count = int(counts_np[original_row])
        for rank in range(action_count + 1):
            token_rows.append(local_row)
            token_ranks.append(rank)
            token_truth.append(EMIT_TOKEN if rank < action_count else STOP_TOKEN)
        for rank in range(action_count):
            action_rows.append(local_row); action_ranks.append(rank)

    token_row_tensor = torch.as_tensor(token_rows, dtype=torch.long, device=device)
    token_rank_tensor = torch.as_tensor(token_ranks, dtype=torch.long, device=device)
    stop_emit_sum = F.cross_entropy(
        model.token_logits(
            decision_context.index_select(0, token_row_tensor), token_rank_tensor
        ),
        torch.as_tensor(token_truth, dtype=torch.long, device=device),
        reduction="sum",
    )
    atoms = len(token_rows)
    loss_sum = stop_emit_sum
    components = {
        "stop_emit_sum": float(stop_emit_sum.detach()),
        "stop_emit_atoms": len(token_rows),
    }

    if action_rows:
        row_tensor = torch.as_tensor(action_rows, dtype=torch.long, device=device)
        rank_tensor = torch.as_tensor(action_ranks, dtype=torch.long, device=device)
        action_context = decision_context.index_select(0, row_tensor)
        class_logits, value_predictions, ranked, code = model.action_heads(action_context, rank_tensor)
        class_truth = torch.as_tensor([
            classes_np[decision_rows[row], rank] for row, rank in zip(action_rows, action_ranks)
        ], dtype=torch.long, device=device)
        value_truth = torch.as_tensor([
            signed_logs_np[decision_rows[row], rank] for row, rank in zip(action_rows, action_ranks)
        ], dtype=value_predictions.dtype, device=device)
        fill_truth = torch.as_tensor([
            fills_np[decision_rows[row], rank] for row, rank in zip(action_rows, action_ranks)
        ], dtype=torch.long, device=device)
        delta_class_sum = F.cross_entropy(class_logits, class_truth, reduction="sum")
        delta_value_sum = F.smooth_l1_loss(value_predictions, value_truth, reduction="sum")
        weights = torch.as_tensor(fill_weights, dtype=contexts.dtype, device=device)
        fill_sum = F.cross_entropy(
            model.fill_logits(ranked, code, class_truth, value_truth), fill_truth,
            weight=weights, reduction="sum",
        )
        loss_sum = loss_sum + delta_class_sum + delta_value_sum + fill_sum
        atoms += 3 * len(action_rows)
    else:
        delta_class_sum = delta_value_sum = fill_sum = decision_context.new_zeros(())
    components.update({
        "delta_class_sum": float(delta_class_sum.detach()),
        "delta_value_sum": float(delta_value_sum.detach()),
        "fill_sum": float(fill_sum.detach()), "action_atoms": len(action_rows),
        "total_atoms": atoms,
    })
    return loss_sum, components


def score_context(model, bundle, device, initial_state=None, chunk_len=8192):
    model.eval(); parts, state = [], initial_state
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
        contexts[role], state = score_context(model, bundles[role], device, state)
    return contexts


def deterministic_fill_map(logits, train_priors):
    """Undo inverse-frequency training reweighting, then take exact MAP."""
    priors = torch.as_tensor(
        train_priors, dtype=torch.float64, device=logits.device
    )
    if priors.shape != (2,) or bool((priors <= 0).any()):
        raise RuntimeError("deterministic fill MAP requires two positive TRAIN priors")
    corrected = logits.detach().to(torch.float64) + torch.log(priors).unsqueeze(0)
    return corrected.argmax(dim=1)


def decode_actions(
    model, contexts, base_lines, vocabulary, fill_priors, device,
    materialization_watchdog_per_callback,
    materialization_watchdog_total, chunk_len=8192,
):
    """Decode rankwise MAP tokens; abort rather than truncate on runaway EMIT."""
    require_equal_lengths("decode", contexts, base_lines)
    counts = np.zeros(len(contexts), dtype=np.int64)
    predicted_lines = [[] for _ in contexts]
    predicted_fills = [[] for _ in contexts]
    predicted_classes = [[] for _ in contexts]
    materialized_so_far = 0
    token_evaluations = terminal_stops = 0
    maximum_emitted_rank = -1
    model.eval()
    with torch.no_grad():
        for start, stop in _iter_chunks(len(contexts), chunk_len):
            callback_contexts = torch.from_numpy(contexts[start:stop]).to(device)
            active = torch.arange(stop - start, dtype=torch.long, device=device)
            rank = 0
            while int(active.numel()) > 0:
                active_context = callback_contexts.index_select(0, active)
                rank_tensor = torch.full(
                    (len(active),), rank, dtype=torch.long, device=device
                )
                ranked, code = model.ranked_state(active_context, rank_tensor)
                tokens = model.stop_emit(ranked).argmax(dim=1)
                token_evaluations += len(active)
                emit_indices = torch.nonzero(
                    tokens == EMIT_TOKEN, as_tuple=False
                ).squeeze(1)
                terminal_stops += len(active) - len(emit_indices)
                if int(emit_indices.numel()) == 0:
                    break
                if rank >= materialization_watchdog_per_callback:
                    raise RuntimeError(
                        "learned STOP/EMIT sequence exceeds the fail-closed "
                        "per-callback materialization watchdog; no replay will be emitted"
                    )
                if materialized_so_far + len(emit_indices) > materialization_watchdog_total:
                    raise RuntimeError(
                        "learned STOP/EMIT sequence exceeds the fail-closed total "
                        "materialization watchdog; no replay will be emitted"
                    )
                emit_rows = active.index_select(0, emit_indices)
                emit_ranked = ranked.index_select(0, emit_indices)
                emit_code = code.index_select(0, emit_indices)
                class_logits = model.delta_class(emit_ranked)
                value_predictions = model.delta_signed_log(emit_ranked).squeeze(1)
                classes = class_logits.argmax(dim=1)
                deltas = []
                for cls, scalar in zip(classes.cpu().tolist(), value_predictions.cpu().tolist()):
                    deltas.append(vocabulary[cls] if cls < len(vocabulary) else inverse_signed_log(scalar))
                global_rows = [start + int(index) for index in emit_rows.cpu().tolist()]
                actual_signed_logs = torch.as_tensor(
                    [signed_log(delta) for delta in deltas],
                    dtype=emit_ranked.dtype, device=device,
                )
                fill_logits = model.fill_logits(
                    emit_ranked, emit_code, classes, actual_signed_logs,
                )
                fill_predictions = deterministic_fill_map(
                    fill_logits, fill_priors
                ).cpu().tolist()
                for row, delta, cls, fill in zip(
                    global_rows, deltas, classes.cpu().tolist(), fill_predictions
                ):
                    target = (int(base_lines[row]) + int(delta)) % LINE_ADDRESS_MODULUS
                    predicted_lines[row].append(target)
                    predicted_fills[row].append(fill)
                    predicted_classes[row].append(int(cls))
                    counts[row] += 1
                materialized_so_far += len(emit_indices)
                maximum_emitted_rank = max(maximum_emitted_rank, rank)
                active = emit_rows
                rank += 1
    if int(counts.sum()) != sum(map(len, predicted_lines)):
        raise RuntimeError("decoded count/action materialization mismatch")
    if terminal_stops != len(contexts):
        raise RuntimeError("successful decode did not observe one terminal STOP per callback")
    decoder_diagnostics = {
        "stop_emit_token_evaluations": token_evaluations,
        "terminal_stop_tokens": terminal_stops,
        "emitted_action_tokens": materialized_so_far,
        "maximum_emitted_rank": maximum_emitted_rank,
        "deterministic_argmax": True,
    }
    return counts, predicted_lines, predicted_fills, predicted_classes, decoder_diagnostics


def ratio(numerator, denominator):
    return numerator / float(denominator) if denominator else 0.0


def trigger_behavior_metrics(predicted_counts, teacher_actions):
    predicted = np.asarray(predicted_counts) > 0
    teacher = np.asarray([bool(items) for items in teacher_actions])
    tp = int(np.logical_and(predicted, teacher).sum())
    precision, recall = ratio(tp, int(predicted.sum())), ratio(tp, int(teacher.sum()))
    return {
        "trigger_true_positive": tp, "trigger_precision": precision,
        "trigger_recall": recall,
        "trigger_f1": ratio(2 * precision * recall, precision + recall),
    }


def joint_action_metrics(predicted_lines, predicted_fills, teacher_actions):
    tp = pred_total = teacher_total = l2_tp = l2_pred = l2_teacher = 0
    for lines, fills, items in zip(predicted_lines, predicted_fills, teacher_actions):
        predicted = Counter(zip(map(int, lines), map(int, fills)))
        teacher = Counter((int(line), FILL_LEVELS.index(fill)) for line, fill in items)
        pred_total += sum(predicted.values()); teacher_total += sum(teacher.values())
        tp += sum((predicted & teacher).values())
        predicted_l2 = Counter(line for (line, fill), count in predicted.items() for _ in range(count) if fill == 0)
        teacher_l2_rows = Counter(line for (line, fill), count in teacher.items() for _ in range(count) if fill == 0)
        l2_pred += sum(predicted_l2.values()); l2_teacher += sum(teacher_l2_rows.values())
        l2_tp += sum((predicted_l2 & teacher_l2_rows).values())
    precision, recall = ratio(tp, pred_total), ratio(tp, teacher_total)
    l2_precision, l2_recall = ratio(l2_tp, l2_pred), ratio(l2_tp, l2_teacher)
    return {
        "joint_true_positive_actions": tp, "joint_action_precision": precision,
        "joint_action_recall": recall,
        "joint_action_f1": ratio(2 * precision * recall, precision + recall),
        "predicted_l2_actions": l2_pred, "teacher_l2_actions": l2_teacher,
        "l2_joint_true_positive_actions": l2_tp, "l2_joint_precision": l2_precision,
        "l2_joint_recall": l2_recall,
        "l2_joint_f1": ratio(2 * l2_precision * l2_recall, l2_precision + l2_recall),
        "predicted_l2_fraction": ratio(l2_pred, pred_total),
        "teacher_l2_fraction": ratio(l2_teacher, teacher_total),
    }


def complete_behavior_metrics(counts, lines, fills, teacher):
    result = behavior_metrics(counts, lines, fills, teacher, fill_levels=FILL_LEVELS)
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


def output_diagnostics(base_lines, counts, predicted_lines, predicted_classes, vocabulary_size):
    duplicate_targets = self_targets = other_actions = 0
    for base, lines, classes in zip(base_lines, predicted_lines, predicted_classes):
        duplicate_targets += len(lines) - len(set(lines))
        self_targets += sum(int(int(line) == int(base)) for line in lines)
        other_actions += sum(int(value == vocabulary_size) for value in classes)
    total = sum(map(len, predicted_lines))
    if int(np.asarray(counts).sum()) != total:
        raise RuntimeError("output accounting differs from learned count")
    return {
        "raw_predicted_action_count": total,
        "materialized_action_count": total,
        "raw_positive_callback_count": int((np.asarray(counts) > 0).sum()),
        "materialized_positive_callback_count": sum(bool(items) for items in predicted_lines),
        "self_target_actions": self_targets,
        "duplicate_target_actions": duplicate_targets,
        "other_escape_actions": other_actions,
        "duplicate_outputs_are_preserved_for_replay": True,
        "delta_legality_fallback": None,
    }


def train_model(model, bundles, targets, streams, teachers, vocabulary, priors, device, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    features = torch.from_numpy(bundles["train"]["features"])
    chunks = list(_iter_chunks(len(features), args.chunk_len))
    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        model.train(); recurrent = None
        totals = Counter(); optimizer_steps = 0
        for group_start in range(0, len(chunks), args.accumulate_chunks):
            optimizer.zero_grad(set_to_none=True)
            group_losses, group_atoms = [], 0
            for start, stop in chunks[group_start:group_start + args.accumulate_chunks]:
                xb = features[start:stop].to(device)
                context, recurrent = model.encode(xb, recurrent)
                recurrent = detach_state(recurrent)
                sliced = tuple(value[start:stop] for value in targets["train"])
                loss, components = chunk_loss(model, context, sliced, priors["fill_weights"], device)
                if loss is None:
                    continue
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite SPP v21 training loss")
                group_losses.append(loss); group_atoms += components["total_atoms"]
                totals.update(components)
            if not group_losses:
                continue
            (torch.stack(group_losses).sum() / group_atoms).backward()
            optimizer.step(); optimizer_steps += 1
        normalized = (
            totals["stop_emit_sum"] + totals["delta_class_sum"]
            + totals["delta_value_sum"] + totals["fill_sum"]
        ) / max(1, totals["total_atoms"])
        guard_contexts = score_role_history(model, bundles, ("train", "guard"), device)["guard"]
        guard_positions = streams["guard"]["demand_positions"]
        guard_bases = np.asarray([row[2] for row in streams["guard"]["demands"]], dtype=np.int64)
        decoded = decode_actions(
            model, guard_contexts[guard_positions], guard_bases, vocabulary,
            priors["fill_priors"], device,
            args.materialization_watchdog_per_callback,
            args.materialization_watchdog_total,
        )
        metrics = complete_behavior_metrics(decoded[0], decoded[1], decoded[2], teachers["guard"])
        selection = guard_selection_key(metrics, normalized, epoch)
        row = {
            "epoch": epoch, "normalized_train_loss": normalized,
            "stop_emit_nll": (
                totals["stop_emit_sum"] / max(1, totals["stop_emit_atoms"])
            ),
            "delta_class_nll": totals["delta_class_sum"] / max(1, totals["action_atoms"]),
            "delta_signed_log_loss": totals["delta_value_sum"] / max(1, totals["action_atoms"]),
            "weighted_fill_nll": totals["fill_sum"] / max(1, totals["action_atoms"]),
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
            "guard_selection_key": json.dumps(selection),
        }
        history.append(row)
        if best is None or selection > best["selection_key"]:
            best = {
                "epoch": epoch, "selection_key": selection,
                "guard_metrics": metrics,
                "state_dict": copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()}),
            }
        print("[train:spp-v21] epoch={} loss={:.8f} joint_f1={:.8f} target_f1={:.8f}".format(
            epoch, normalized, metrics["joint_action_f1"], metrics["target_f1"]
        ))
    if best is None:
        raise RuntimeError("SPP v21 produced no checkpoint")
    model.load_state_dict(best["state_dict"])
    return history, best


def self_test_model(hidden_size):
    self_test_exact_int()
    for size in MODEL_POINTS["lstm"]:
        model = GlobalSPPLSTM(size, 7)
        observed = sum(parameter.numel() for parameter in model.parameters())
        expected = expected_parameter_count(size, 7)
        if observed != expected:
            raise RuntimeError("SPP v21 parameter formula mismatch: {} != {}".format(observed, expected))
    if inverse_signed_log(signed_log(-12345)) != -12345 or inverse_signed_log(signed_log(6789)) != 6789:
        raise RuntimeError("signed-log OTHER codec round trip failed")
    sample = GlobalSPPLSTM(hidden_size, 7)
    sample.eval(); features = torch.zeros((5, RUNTIME_FEATURES)); changed = features.clone(); changed[-1, 0] = 1.0
    with torch.no_grad():
        first, _ = sample.encode(features); second, _ = sample.encode(changed)
    if not torch.equal(first[:-1], second[:-1]):
        raise RuntimeError("future callback changed a prior global LSTM state")
    forbidden = ("page", "candidate", "action_cell", "byte", "gate", "log_count")
    if any(any(token in name for token in forbidden) for name, _ in sample.named_parameters()):
        raise RuntimeError("normal template/gate-count state leaked into v21")
    with torch.no_grad():
        sample.stop_emit.weight.zero_()
        sample.stop_emit.bias.copy_(torch.tensor([1.0, 0.0]))
    stopped = decode_actions(
        sample, np.zeros((2, hidden_size), dtype=np.float32),
        np.asarray([0, 1], dtype=np.int64), list(range(7)),
        np.asarray([0.02, 0.98]), torch.device("cpu"), 2, 10,
    )
    if stopped[0].tolist() != [0, 0] or stopped[4]["terminal_stop_tokens"] != 2:
        raise RuntimeError("terminal STOP decoder self-test failed")
    with torch.no_grad():
        sample.stop_emit.bias.copy_(torch.tensor([0.0, 1.0]))
    try:
        decode_actions(
            sample, np.zeros((1, hidden_size), dtype=np.float32),
            np.asarray([0], dtype=np.int64), list(range(7)),
            np.asarray([0.02, 0.98]), torch.device("cpu"), 2, 10,
        )
    except RuntimeError as error:
        if "watchdog" not in str(error):
            raise
    else:
        raise RuntimeError("runaway EMIT decoder did not fail closed")
    logits = torch.tensor([[0.0, 0.0], [5.0, 0.0]])
    one = deterministic_fill_map(logits, np.asarray([0.02, 0.98]))
    two = deterministic_fill_map(logits, np.asarray([0.02, 0.98]))
    if one.tolist() != [1, 0] or not torch.equal(one, two):
        raise RuntimeError("prior-corrected deterministic fill MAP self-test failed")


def main():
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
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--chunk-len", type=int, default=CHUNK_LEN)
    parser.add_argument("--accumulate-chunks", type=int, default=ACCUMULATE_CHUNKS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--materialization-watchdog-per-callback", type=int, default=4096)
    parser.add_argument("--materialization-watchdog-total", type=int, default=10000000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    pinned_training_config = {
        "seed": SEED, "epochs": EPOCHS,
        "chunk_len": CHUNK_LEN, "accumulate_chunks": ACCUMULATE_CHUNKS,
        "learning_rate": LEARNING_RATE,
    }
    actual_training_config = {
        "seed": args.seed, "epochs": args.epochs, "chunk_len": args.chunk_len,
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
        raise RuntimeError("model size/pair is not a configured v21 point")
    if min(
        args.epochs, args.chunk_len, args.accumulate_chunks,
        args.materialization_watchdog_per_callback,
        args.materialization_watchdog_total,
    ) < 1:
        raise RuntimeError("model/training dimensions must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cuda_device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else None
    )
    if device.type != "cuda" or "A100" not in cuda_device_name:
        raise RuntimeError(
            "pinned v21 run requires an NVIDIA A100; observed {!r}".format(
                cuda_device_name
            )
        )
    if not hasattr(torch, "use_deterministic_algorithms"):
        raise RuntimeError("this torch build cannot enforce deterministic algorithms")
    if not hasattr(torch, "set_float32_matmul_precision") or not hasattr(
        torch, "get_float32_matmul_precision"
    ):
        raise RuntimeError("this torch build cannot pin float32 matmul precision")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    self_test_model(args.model_size)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    roles = ("train", "guard", "eval")
    stream_paths = {role: getattr(args, role + "_stream") for role in roles}
    action_paths = {role: getattr(args, role + "_teacher_actions") for role in roles}
    streams = {role: load_stream(stream_paths[role]) for role in roles}
    teachers = {role: load_teacher_actions(action_paths[role], streams[role]["demands"]) for role in roles}
    bundles = {role: runtime_bundle(streams[role]) for role in roles}
    for role in roles:
        if not np.array_equal(bundles[role]["features"], runtime_bundle(streams[role])["features"]):
            raise RuntimeError("{} runtime encoder is not reproducible".format(role))
    vocabulary, train_delta_frequencies = build_delta_vocabulary(streams["train"], teachers["train"])
    targets = {
        "train": build_context_targets(
            streams["train"], teachers["train"], vocabulary
        )
    }
    priors = training_priors(targets["train"], len(vocabulary) + 1)
    vocabulary_stats = {
        role: vocabulary_statistics(streams[role], teachers[role], vocabulary)
        for role in ("train", "guard")
    }

    model = GlobalSPPLSTM(args.model_size, len(vocabulary)).to(device)
    model.initialize_label_priors(
        priors["stop_emit_counts"], priors["delta_class_counts"]
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != expected_parameter_count(args.model_size, len(vocabulary)):
        raise RuntimeError("measured SPP v21 parameter count changed")
    history, best = train_model(
        model, bundles, targets, streams, teachers, vocabulary, priors, device,
        args,
    )

    # Reproduce the selected guard audit, then touch evaluation exactly once.
    selected_contexts = score_role_history(model, bundles, ("train", "guard"), device)
    guard_bases = np.asarray([row[2] for row in streams["guard"]["demands"]], dtype=np.int64)
    guard_decode = decode_actions(
        model, selected_contexts["guard"][streams["guard"]["demand_positions"]],
        guard_bases, vocabulary, priors["fill_priors"], device,
        args.materialization_watchdog_per_callback,
        args.materialization_watchdog_total,
    )
    selected_guard_metrics = complete_behavior_metrics(
        guard_decode[0], guard_decode[1], guard_decode[2], teachers["guard"]
    )
    if selected_guard_metrics != best["guard_metrics"]:
        raise RuntimeError("selected guard checkpoint did not reproduce")

    eval_contexts = score_role_history(model, bundles, roles, device)["eval"]
    eval_bases = np.asarray([row[2] for row in streams["eval"]["demands"]], dtype=np.int64)
    eval_decode = decode_actions(
        model, eval_contexts[streams["eval"]["demand_positions"]], eval_bases,
        vocabulary, priors["fill_priors"], device,
        args.materialization_watchdog_per_callback,
        args.materialization_watchdog_total,
    )
    behavior = complete_behavior_metrics(eval_decode[0], eval_decode[1], eval_decode[2], teachers["eval"])
    vocabulary_stats["eval"] = vocabulary_statistics(
        streams["eval"], teachers["eval"], vocabulary
    )
    diagnostics = output_diagnostics(eval_bases, eval_decode[0], eval_decode[1], eval_decode[3], len(vocabulary))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    normal_path = args.out_dir / "offline_spp.replay.csv"
    nn_path = args.out_dir / "offline_nn.replay.csv"
    normal_entries, normal_triggers, normal_fill_counts = write_teacher_replay(normal_path, streams["eval"]["demands"], teachers["eval"])
    nn_entries, nn_triggers, nn_fill_counts = write_prediction_replay(nn_path, streams["eval"]["demands"], eval_decode[1], eval_decode[2])
    history_path, model_path = args.out_dir / "training_history.csv", args.out_dir / "model.pt"
    write_table(history_path, history)
    torch.save({
        "state_dict": model.state_dict(), "model_family": "lstm",
        "model_size": args.model_size, "runtime_features": RUNTIME_FEATURES,
        "exact_delta_vocabulary": vocabulary, "other_class": len(vocabulary),
        "fill_levels": FILL_LEVELS, "fill_priors": priors["fill_priors"].tolist(),
        "selected_epoch": best["epoch"], "stochastic_decoding": False,
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": MODEL_REVISION, "decoder_revision": DECODER_REVISION,
    }, model_path)

    tag = model_tag("lstm", args.model_size)
    state_bytes = 2 * args.model_size * 4
    metadata = {
        "run_id": RUN_ID, "trace": TRACE, "model_tag": tag,
        "matched_normal_prefetcher": POLICY, "neural_role": "standalone_direct_action_prefetcher",
        "model_family": "lstm", "track_model_family": "lstm", "model_size": args.model_size,
        "architecture_pair_id": args.pair_id, "parameter_count": parameter_count,
        "realized_parameter_count": parameter_count,
        "maximum_parameter_count": expected_parameter_count(
            args.model_size, MAX_EXACT_DELTAS
        ),
        "maximum_parameter_count_at_255_exact_deltas": expected_parameter_count(
            args.model_size, MAX_EXACT_DELTAS
        ),
        "parameter_count_is_dataset_dependent": True,
        "parameter_formula": describe_model_points()["parameter_formula"],
        "parameter_storage_bytes_float32": parameter_count * 4,
        "peak_persistent_recurrent_state_bytes": state_bytes,
        "persistent_recurrent_state": "one bounded global chronological LSTM hidden/cell pair",
        "dynamic_page_state_pages": 0, "recurrent_state_dtype": "float32",
        "model_point_contract": describe_model_points(),
        "seed": args.seed, "operation": OPERATION,
        "experiment_revision": EXPERIMENT_REVISION, "model_revision": MODEL_REVISION,
        "decoder_revision": DECODER_REVISION, "weights_retrained": True,
        "checkpoint_reused": False, "decoder_only_change": False,
        "guard_selected_checkpoint": True, "guard_selected_decoder": False,
        "selected_epoch": best["epoch"], "guard_selection_key": list(best["selection_key"]),
        "guard_selection_metrics": selected_guard_metrics,
        "guard_selection_rule": describe_model_points()["guard_selection_rule"],
        "guard_selection_key_fields": [
            "joint_action_f1", "target_f1", "l2_joint_f1", "trigger_f1",
            "count_exact_match_rate", "fill_accuracy_on_matched_targets",
            "negative_normalized_train_loss", "negative_epoch",
        ],
        "guard_selection_composite_or_mean_used": False,
        "evaluation_decode_count": 1, "evaluation_used_for_selection": False,
        "training_config": describe_model_points()["training_config"],
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "determinism_fail_closed": True,
        "cuda_device_name": cuda_device_name,
        "epochs": args.epochs, "chunk_len": args.chunk_len,
        "accumulate_chunks": args.accumulate_chunks, "learning_rate": args.learning_rate,
        "runtime_feature_count": RUNTIME_FEATURES,
        "runtime_encoding": "lossless 58-bit cache-line number plus one DEMAND/FILL kind bit",
        "source_decision_effective_external_input": SOURCE_INPUTS,
        "training_runtime_fields": SOURCE_INPUTS, "inference_runtime_fields": SOURCE_INPUTS,
        "same_external_input_contract": True, "training_inference_input_encoder_identical": True,
        "runtime_encoder_entrypoint": "train_and_offline_infer.runtime_bundle",
        "runtime_encoder_sha256": runtime_encoder_sha256(),
        "training_runtime_encoder_sha256": runtime_encoder_sha256(),
        "inference_runtime_encoder_sha256": runtime_encoder_sha256(),
        "model_does_not_use_pc": True, "pc_is_replay_transport_only": True,
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
        "probability_threshold_used": False, "threshold_related_hardcodes_used": False,
        "inference_policy_hardcodes_used": False, "neural_degree_cap": None,
        "same_page_rule_used_by_neural_inference": False, "fixed_page_offset_classes": None,
        "normal_policy_templates_used_by_neural_inference": False,
        "future_label_window_used": False, "fill_lead_cutoff_used": False,
        "normal_candidate_bank_is_fixed": False, "nn_generates_own_target_addresses_and_fill_levels": True,
        "decoder_training_mode": DECODER_TRAINING_MODE,
        "decoding_rule": describe_model_points()["decoding_rule"],
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_previous_predicted_action_used_as_input": False,
        "decoder_previous_sampled_action_used_as_input": False,
        "teacher_action_values_used_as_decoder_feedback": False,
        "teacher_target_used_for_loss_local_fill_conditioning": True,
        "teacher_target_conditions_loss_only_fill_factor": True,
        "teacher_target_used_as_recurrent_feedback": False,
        "sampled_outputs_used_as_decoder_feedback": False,
        "rank_conditioning": "generic_four_component_sinusoidal_position_code",
        "stop_emit_training_objective": TOKEN_OBJECTIVE,
        "stop_emit_train_class_counts": priors["stop_emit_counts"].tolist(),
        "stop_emit_train_class_priors": (
            priors["stop_emit_counts"] / priors["stop_emit_counts"].sum()
        ).tolist(),
        "stop_emit_train_class_order": ["STOP", "EMIT"],
        "stop_emit_prior_initialization": "TRAIN_natural_class_log_priors",
        "stop_emit_class_weighting_used": False,
        "stop_emit_prior_extremely_sparse": False,
        "terminal_stop_supervised": True,
        "terminal_stop_count_train": int(priors["stop_emit_counts"][STOP_TOKEN]),
        "stop_emit_decoding_rule": "rank_conditioned_two_class_MAP_until_STOP",
        "separate_gate_head_used": False,
        "request_count_head_used": False,
        "request_count_regression_used": False,
        "action_or_byte_grammar_used": False,
        "delta_training_objective": DELTA_OBJECTIVE,
        "delta_decoding_rule": "class_MAP_exact_TRAIN_delta_or_signed_log_OTHER_relative_to_callback_line",
        "delta_vocabulary_source": "TRAIN_labels_only_top_frequency_then_signed_value_tie_break",
        "delta_vocabulary_architecture_budget": MAX_EXACT_DELTAS,
        "delta_vocabulary_max_exact": MAX_EXACT_DELTAS,
        "exact_delta_vocabulary": vocabulary, "exact_delta_vocabulary_size": len(vocabulary),
        "realized_exact_delta_vocabulary_size": len(vocabulary),
        "other_delta_class": len(vocabulary), "delta_vocabulary_statistics": vocabulary_stats,
        "delta_other_escape": "signed_log_continuous_bounded_approximation",
        "delta_other_decode_precision": (
            "rounded_float32_approximate_except_exact_vocabulary"
        ),
        "full_signed_line_delta_range_reachable": False,
        "every_signed_line_delta_exactly_representable": False,
        "exact_delta_representability_scope": "train_vocabulary_only",
        "train_delta_frequency_histogram": {str(key): value for key, value in sorted(train_delta_frequencies.items())},
        "delta_zero_allowed": True, "self_target_actions_allowed": True,
        "delta_legality_constraints": [], "delta_legality_fallback": None,
        "duplicate_target_handling": "preserve_all_learned_outputs_for_replay",
        "fill_training_objective": FILL_OBJECTIVE,
        "fill_train_class_counts": priors["fill_counts"].tolist(),
        "fill_train_priors": priors["fill_priors"].tolist(),
        "fill_train_inverse_frequency_weights": priors["fill_weights"].tolist(),
        "fill_prior_correction_at_decode_used": True,
        "fill_prior_correction_rule": "balanced_logits_plus_log_TRAIN_natural_prior",
        "fill_decoding_rule": describe_model_points()["fill_decoding_rule"],
        "fill_conditioned_on_actual_emitted_target": True, "fill_argmax_used": True,
        "fill_target_conditioning_features": "decoded_delta_class_plus_actual_decoded_signed_log_delta_plus_rank",
        "fill_probability_threshold": None,
        "stochastic_decoding": False,
        "keyed_sampling_used": False,
        "decoder_sampling_roles": [],
        "decoder_guard_diagnostics": guard_decode[4],
        "decoder_eval_diagnostics": eval_decode[4],
        "training_chunks_shuffled": False, "training_state_mode": "chronological_stateful_tbptt",
        "training_state_carried_across_chunks": True, "training_state_detached_between_chunks": True,
        "inference_history_mode": "fresh_state_then_complete_train_guard_eval_chronology",
        "global_chronological_lstm": True, "routed_demand_fill_recurrent_paths": False,
        "page_local_causal_state": False, "handcrafted_semantic_features_used": False,
        "causal_derived_features": [],
        "manual_head_loss_weights_used": False,
        "data_derived_fill_class_weights_used": True,
        "teacher_max_actions_per_callback": {
            role: max(map(len, teachers[role])) for role in roles
        },
        "train_terminal_stop_max_rank": max(map(len, teachers["train"])),
        "guard_decoded_max_actions_per_callback": int(guard_decode[0].max()),
        "eval_decoded_max_actions_per_callback": int(eval_decode[0].max()),
        "maximum_action_count_is_learned_not_fixed": True,
        "output_materialization_watchdog_actions_per_callback": args.materialization_watchdog_per_callback,
        "output_materialization_watchdog_actions_per_role": args.materialization_watchdog_total,
        "output_materialization_watchdog_role": "fail_closed_resource_guard_no_truncation_or_forced_count",
        "output_materialization_watchdog_is_neural_degree_cap": False,
        "causal_no_future_self_test": "PASS", "independent_rank_decoder_self_test": "PASS",
        "terminal_stop_self_test": "PASS", "fail_closed_watchdog_self_test": "PASS",
        "signed_log_other_codec_self_test": "PASS",
        "fill_prior_corrected_argmax_self_test": "PASS",
        "integer_csv_exactness_self_test": "PASS",
        "event_logger_schema": EVENT_LOGGER_SCHEMA, "action_attachment_mode": ACTION_ATTACHMENT_MODE,
        "teacher_action_canonicalization": CANONICALIZATION_MODE,
        "replay_preserves_explicit_fill_level": True,
        "same_source_input_offline_claim_allowed": True, "closed_loop_live_claim_allowed": False,
        "offline_input_feedback_origin": "recorded cache-fill callbacks produced by the source SPP run",
        "comparison_claim_boundary": "matched-input open-loop offline comparison only",
        "collection_manifest_role": "historical_input_package_provenance_only",
        "collection_manifest_decoder_fields_are_current_contract": False,
        "input_reuse": "v18 input package reused byte-for-byte",
        "source_contract_sha256": sha256(args.source_contract),
        "trainer_source_sha256": sha256(Path(__file__)),
        "model_contract_source_sha256": sha256(Path(__file__).with_name("model_contract.py")),
        "threshold_free_policy_source_sha256": sha256(
            REPO_ROOT / "formal_NN_training" / "common" / "threshold_free_policy.py"
        ),
        "offline_normal_entries": normal_entries, "offline_normal_triggers": normal_triggers,
        "offline_normal_fill_counts": normal_fill_counts,
        "offline_normal_fill_level_counts": normal_fill_counts,
        "offline_nn_entries": nn_entries, "offline_nn_triggers": nn_triggers,
        "offline_nn_fill_counts": nn_fill_counts, "offline_nn_fill_level_counts": nn_fill_counts,
        "action_output_diagnostics": diagnostics,
        "raw_predicted_action_count": diagnostics["raw_predicted_action_count"],
        "materialized_action_count": diagnostics["materialized_action_count"],
        "normal_list_sha256": sha256(normal_path), "nn_list_sha256": sha256(nn_path),
        "heldout_behavior_metrics": behavior, "train_history": history,
        "source_contract": source_contract, "model_checkpoint_sha256": sha256(model_path),
        "training_history_sha256": sha256(history_path), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__,
        "decision_router_source_sha256": decision_router_source_sha256(),
    }
    for role in roles:
        metadata[role + "_decision_router_sha256"] = decision_router_sha256(streams[role])
        metadata[role + "_stream_gzip_sha256"] = sha256(stream_paths[role])
        metadata[role + "_stream_content_sha256"] = gzip_content_sha256(stream_paths[role])
        metadata[role + "_teacher_actions_gzip_sha256"] = sha256(action_paths[role])
        metadata[role + "_teacher_actions_content_sha256"] = gzip_content_sha256(action_paths[role])
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "PASS", "model_tag": tag, "parameters": parameter_count,
        "selected_epoch": best["epoch"], "exact_delta_vocabulary_size": len(vocabulary),
        "offline_normal_entries": normal_entries, "offline_nn_entries": nn_entries,
        "offline_nn_fill_level_counts": nn_fill_counts,
    }, indent=2))


if __name__ == "__main__":
    main()
