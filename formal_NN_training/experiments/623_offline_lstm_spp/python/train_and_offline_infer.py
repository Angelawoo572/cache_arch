#!/usr/bin/env python3
"""Train and decode the matched-input 623 SPP v25 model.

The model sees only the unchanged source chronology: a DEMAND/FILL kind bit
and a lossless 58-bit line number.  It predicts a natural categorical action
count K, then exactly K rank-conditioned fill choices and exact modular 58-bit
delta payloads.  Every capacity uses the same global chronological one-layer
LSTM.  Fill and payload losses are present only for real teacher ranks.  Decode
uses exact k-best Bernoulli subset enumeration to select the most probable
feasible unique target without using an earlier action as a neural input.
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

import model_contract as model_contract_module
from model_contract import (
    ACCUMULATE_CHUNKS, ADDRESS_BITS,
    CACHE_LINE_BYTES, CACHE_LINE_SHIFT, CHECKPOINT_SELECTION, CHUNK_LEN,
    CORE_TYPE, COUNT_OBJECTIVE, DECODER_REVISION,
    DECODER_TRAINING_MODE, DECODING_RULE, EPOCHS, EXPERIMENT_REVISION,
    EXTERNAL_INPUT_FIELDS, FILL_LEVELS, FILL_OBJECTIVE, LEARNING_RATE,
    LINE_ADDRESS_BITS, LINE_ADDRESS_MODULUS, MODEL_POINTS,
    MODEL_REVISION, OPERATION, DELTA_BIT_OBJECTIVE, PARENT_INPUT_RUN_ID,
    POLICY, RANK_CODE_SIZE, RUNTIME_FEATURE_COUNT, RUN_ID, SEED, TRACE,
    bits_to_modular_delta, count_statistics, describe_model_points,
    exact_int as as_int, expected_parameter_count, k_best_flip_subsets,
    modal_bernoulli_payload, model_tag, modular_delta, modular_delta_bits,
)

# Contract inspection must work on the CPU-only replay host without torch.
if __name__ == "__main__" and sys.argv[1:] == ["--describe-model-points"]:
    print(json.dumps(describe_model_points(), indent=2, sort_keys=True))
    raise SystemExit(0)
if __name__ == "__main__" and sys.argv[1:] == ["--self-test"]:
    model_contract_module.self_test_contract()
    print("PASS")
    raise SystemExit(0)

# CUDA deterministic GEMM configuration must precede torch import.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from formal_NN_training.common.threshold_free_policy import behavior_metrics


EVENT_LOGGER_SCHEMA = "623_causal_trigger_fill_v6"
ACTION_ATTACHMENT_MODE = "explicit_trigger_event_id"
CANONICALIZATION_MODE = "per_target_min_fill_queue_effect"
SOURCE_INPUTS = list(EXTERNAL_INPUT_FIELDS)
LINE_ADDRESS_HALF_RANGE = 1 << (LINE_ADDRESS_BITS - 1)
RUNTIME_FEATURES = LINE_ADDRESS_BITS + 1

if RUNTIME_FEATURES != RUNTIME_FEATURE_COUNT:
    raise RuntimeError("SPP v25 raw runtime feature contract changed")


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
            pf_line = as_int(row["pf_line"])
            fill = as_int(row["fill_level"])
            if (
                pf_event <= last_pf_event
                or trigger >= pf_event
                or as_int(row["event_distance"]) != pf_event - trigger
                or pf_line < 0 or pf_line >= LINE_ADDRESS_MODULUS
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
        raise RuntimeError("runtime line number is outside encoded domain")
    array = np.asarray(integers, dtype=np.uint64)
    shifts = np.arange(width, dtype=np.uint64)
    return ((array[:, None] >> shifts[None, :]) & 1).astype(np.float32)


def runtime_bundle(stream):
    context = stream["context"]
    lines = np.asarray([line for _, _, line, _ in context], dtype=np.int64)
    demand_kind = np.asarray(
        [kind == "DEMAND" for kind, _, _, _ in context], dtype=np.bool_
    )
    features = np.concatenate([
        _unsigned_bits(lines, LINE_ADDRESS_BITS),
        demand_kind.astype(np.float32)[:, None],
    ], axis=1)
    return {
        "features": features,
        "lines": lines,
        "demand_kind": demand_kind,
    }


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
        "demand_positions": [int(value) for value in stream["demand_positions"]],
        "decision_indices": [
            int(stream["context"][int(position)][3])
            for position in stream["demand_positions"]
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def decision_router_source_sha256():
    return hashlib.sha256(inspect.getsource(decision_router_sha256).encode()).hexdigest()


def canonical_signed_delta(base, target):
    difference = (int(target) - int(base)) % LINE_ADDRESS_MODULUS
    if difference >= LINE_ADDRESS_HALF_RANGE:
        difference -= LINE_ADDRESS_MODULUS
    return difference


def teacher_action_pairs(stream, actions):
    require_equal_lengths("teacher action pairs", stream["demands"], actions)
    values = []
    for demand, items in zip(stream["demands"], actions):
        values.extend(
            (canonical_signed_delta(demand[2], target), int(fill))
            for target, fill in items
        )
    return values


def fill_and_bit_statistics(train_stream, train_actions):
    require_equal_lengths(
        "TRAIN fill/bit statistics", train_stream["demands"], train_actions
    )
    counts = [0, 0]
    one_counts = [[0] * LINE_ADDRESS_BITS for _ in FILL_LEVELS]
    for demand, items in zip(train_stream["demands"], train_actions):
        base = demand[2]
        for target, fill in items:
            fill_index = FILL_LEVELS.index(int(fill))
            counts[fill_index] += 1
            for bit, value in enumerate(modular_delta_bits(modular_delta(base, target))):
                one_counts[fill_index][bit] += int(value)
    if any(value <= 0 for value in counts):
        raise RuntimeError(
            "both fill-specific delta-bit heads require actual TRAIN labels; "
            "observed L2/LLC counts {}".format(counts)
        )
    denominator = float(sum(counts) + len(counts))
    fill_prior = [(value + 1.0) / denominator for value in counts]
    bit_priors = [
        [(ones + 1.0) / float(counts[fill_index] + 2)
         for ones in one_counts[fill_index]]
        for fill_index in range(len(FILL_LEVELS))
    ]
    return counts, fill_prior, one_counts, bit_priors


def payload_statistics(stream, actions):
    pairs = teacher_action_pairs(stream, actions)
    return {
        "teacher_actions": len(pairs),
        "unique_teacher_joint_pairs": len(set(pairs)),
        "fill_level_counts": {
            "FILL_L2": sum(fill == 2 for _, fill in pairs),
            "FILL_LLC": sum(fill == 4 for _, fill in pairs),
        },
        "every_action_has_exact_58_bit_payload": True,
    }


def build_context_targets(stream, actions, count_classes):
    require_equal_lengths(
        "target decisions", stream["demands"], stream["demand_positions"], actions
    )
    maximum_count = count_classes - 1
    counts = np.full(len(stream["context"]), -1, dtype=np.int64)
    fills = np.full(
        (len(stream["context"]), maximum_count), -1, dtype=np.int64
    )
    delta_bits = np.zeros(
        fills.shape + (LINE_ADDRESS_BITS,), dtype=np.float32
    )
    for decision, position in enumerate(stream["demand_positions"]):
        items = actions[decision]
        if len(items) > maximum_count:
            raise RuntimeError("teacher count exceeds TRAIN-derived support")
        counts[position] = len(items)
        base = stream["demands"][decision][2]
        for rank, (target, fill) in enumerate(items):
            fills[position, rank] = FILL_LEVELS.index(int(fill))
            delta_bits[position, rank] = np.asarray(
                modular_delta_bits(modular_delta(base, target)),
                dtype=np.float32,
            )
    return counts, fills, delta_bits


def write_table(path, rows):
    if not rows:
        raise RuntimeError("cannot write an empty table")
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
            for pf_line, fill in zip(targets, fills):
                if int(fill) not in FILL_LEVELS:
                    raise RuntimeError("prediction has an invalid fill level")
                writer.writerow([
                    pc, line, occurrence, hex(int(pf_line) << CACHE_LINE_SHIFT),
                    int(fill),
                ])
                fill_counts["FILL_L2" if int(fill) == 2 else "FILL_LLC"] += 1
                entries += 1
    return entries, triggers, fill_counts


def build_modal_llc_control(base_lines, modal_delta):
    lines = [[(int(base) + int(modal_delta)) % LINE_ADDRESS_MODULUS]
             for base in base_lines]
    fills = [[4] for _ in base_lines]
    return lines, fills


def _iter_chunks(length, width):
    for start in range(0, length, width):
        yield start, min(length, start + width)


def rank_code(ranks, dtype):
    ranks = ranks.to(dtype) + 1.0
    frequencies = ranks.new_tensor([1.0, 0.01])
    phase = ranks.unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)


class NaturalCardinalitySPPLSTM(nn.Module):
    def __init__(
        self, hidden_size, count_prior, fill_prior, delta_bit_priors,
    ):
        super().__init__()
        self.core_type = CORE_TYPE
        self.hidden_size = int(hidden_size)
        self.count_output_classes = len(count_prior)
        self.fill_output_classes = len(fill_prior)
        if (
            self.hidden_size not in MODEL_POINTS["lstm"]
            or self.count_output_classes < 1
            or self.fill_output_classes != len(FILL_LEVELS)
        ):
            raise ValueError("unsupported realized SPP v25 dimensions")
        self.input_projection = nn.Linear(RUNTIME_FEATURES, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.rank_fusion = nn.Linear(hidden_size + RANK_CODE_SIZE, hidden_size)
        self.count_head = nn.Linear(hidden_size, self.count_output_classes)
        self.fill_head = nn.Linear(hidden_size, self.fill_output_classes)
        self.l2_delta_bit_head = nn.Linear(hidden_size, LINE_ADDRESS_BITS)
        self.llc_delta_bit_head = nn.Linear(hidden_size, LINE_ADDRESS_BITS)

        count_tensor = torch.as_tensor(count_prior, dtype=self.count_head.bias.dtype)
        fill_tensor = torch.as_tensor(
            fill_prior, dtype=self.fill_head.bias.dtype
        )
        bit_tensor = torch.as_tensor(
            delta_bit_priors, dtype=self.l2_delta_bit_head.bias.dtype
        )
        if (
            bool((count_tensor <= 0).any())
            or bool((fill_tensor <= 0).any())
            or not bool(torch.isfinite(count_tensor).all())
            or not bool(torch.isfinite(fill_tensor).all())
            or bit_tensor.shape != (len(FILL_LEVELS), LINE_ADDRESS_BITS)
            or bool((bit_tensor <= 0).any())
            or bool((bit_tensor >= 1).any())
            or not bool(torch.isfinite(bit_tensor).all())
        ):
            raise ValueError("TRAIN-derived initialization is invalid")
        with torch.no_grad():
            self.count_head.weight.zero_()
            self.count_head.bias.copy_(torch.log(count_tensor))
            self.fill_head.weight.zero_()
            self.fill_head.bias.copy_(torch.log(fill_tensor))
            self.l2_delta_bit_head.weight.zero_()
            self.llc_delta_bit_head.weight.zero_()
            bit_logits = torch.log(bit_tensor) - torch.log1p(-bit_tensor)
            self.l2_delta_bit_head.bias.copy_(bit_logits[0])
            self.llc_delta_bit_head.bias.copy_(bit_logits[1])

    def encode(self, features, demand_kind, state=None):
        embedded = torch.tanh(self.input_projection(features))
        output, state = self.lstm(embedded.unsqueeze(0), state)
        return output.squeeze(0), state

    def ranked_heads(self, contexts, ranks):
        ranked = torch.tanh(
            self.rank_fusion(torch.cat([
                contexts, rank_code(ranks, contexts.dtype)
            ], dim=1))
        )
        return {
            "fill_logits": self.fill_head(ranked),
            "l2_delta_bit_logits": self.l2_delta_bit_head(ranked),
            "llc_delta_bit_logits": self.llc_delta_bit_head(ranked),
        }


def detach_state(state):
    if state is None:
        return None
    return tuple(value.detach() for value in state)


def chunk_loss_parts(model, contexts, targets):
    counts_np, fills_np, delta_bits_np = targets
    decision_rows = np.flatnonzero(counts_np >= 0).astype(np.int64)
    if not len(decision_rows):
        return None
    rows = torch.from_numpy(decision_rows).to(
        device=contexts.device, dtype=torch.long
    )
    decisions = contexts.index_select(0, rows)
    count_truth = torch.from_numpy(counts_np[decision_rows]).to(
        device=contexts.device, dtype=torch.long
    )
    if int(count_truth.max()) >= model.count_output_classes:
        raise RuntimeError("count label exceeds TRAIN-derived support")
    count_sum = F.cross_entropy(
        model.count_head(decisions), count_truth, reduction="sum"
    )
    fill_sum = contexts.sum() * 0.0
    delta_bit_sum = contexts.sum() * 0.0
    action_atoms = delta_bit_atoms = 0
    fill_action_atoms = [0, 0]
    maximum_rank = int(count_truth.max().item()) if len(count_truth) else 0
    for rank in range(maximum_rank):
        active_np = np.flatnonzero(counts_np[decision_rows] > rank).astype(np.int64)
        if not len(active_np):
            continue
        active = torch.from_numpy(active_np).to(
            device=contexts.device, dtype=torch.long
        )
        ranked_contexts = decisions.index_select(0, active)
        ranks = torch.full(
            (len(active_np),), rank, device=contexts.device, dtype=torch.long
        )
        heads = model.ranked_heads(ranked_contexts, ranks)
        logits = heads["fill_logits"]
        truth_np = fills_np[decision_rows[active_np], rank]
        if bool((truth_np < 0).any()):
            raise RuntimeError("real teacher rank is missing a fill label")
        truth = torch.from_numpy(truth_np).to(
            device=contexts.device, dtype=torch.long
        )
        fill_sum = fill_sum + F.cross_entropy(
            logits, truth, reduction="sum"
        )
        action_atoms += len(active_np)
        bit_truth = torch.from_numpy(
            delta_bits_np[decision_rows[active_np], rank]
        ).to(device=contexts.device, dtype=contexts.dtype)
        for fill_index, head_name in enumerate((
            "l2_delta_bit_logits", "llc_delta_bit_logits",
        )):
            selected_np = truth_np == fill_index
            if not bool(selected_np.any()):
                continue
            selected = torch.from_numpy(selected_np).to(contexts.device)
            delta_bit_sum = delta_bit_sum + F.binary_cross_entropy_with_logits(
                heads[head_name][selected], bit_truth[selected], reduction="sum"
            )
            selected_count = int(selected_np.sum())
            fill_action_atoms[fill_index] += selected_count
            delta_bit_atoms += selected_count * LINE_ADDRESS_BITS
    return {
        "count_sum": count_sum,
        "fill_sum": fill_sum,
        "delta_bit_sum": delta_bit_sum,
        "decision_atoms": len(decision_rows),
        "action_atoms": action_atoms,
        "delta_bit_atoms": delta_bit_atoms,
        "l2_action_atoms": fill_action_atoms[0],
        "llc_action_atoms": fill_action_atoms[1],
    }


def score_context(model, bundle, device, initial_state=None, chunk_len=8192):
    model.eval()
    parts, state = [], initial_state
    with torch.no_grad():
        for start, stop in _iter_chunks(len(bundle["features"]), chunk_len):
            features = torch.from_numpy(bundle["features"][start:stop]).to(device)
            kinds = torch.from_numpy(bundle["demand_kind"][start:stop]).to(device)
            context, state = model.encode(features, kinds, state)
            state = detach_state(state)
            parts.append(context.cpu().numpy())
    return np.concatenate(parts, axis=0), state


def score_role_history(model, bundles, roles, device, chunk_len=8192):
    contexts, state = {}, None
    for role in roles:
        contexts[role], state = score_context(
            model, bundles[role], device, state, chunk_len
        )
    return contexts


def natural_list_nll(model, contexts, targets, device, chunk_len=4096):
    totals = Counter()
    model.eval()
    with torch.no_grad():
        for start, stop in _iter_chunks(len(contexts), chunk_len):
            context = torch.from_numpy(contexts[start:stop]).to(device)
            parts = chunk_loss_parts(
                model, context, tuple(value[start:stop] for value in targets)
            )
            if parts is None:
                continue
            for key in ("count_sum", "fill_sum", "delta_bit_sum"):
                totals[key] += float(parts[key].detach())
            for key in (
                "decision_atoms", "action_atoms", "delta_bit_atoms",
                "l2_action_atoms", "llc_action_atoms",
            ):
                totals[key] += int(parts[key])
    complete = (
        totals["count_sum"] + totals["fill_sum"] + totals["delta_bit_sum"]
    ) / max(1, totals["decision_atoms"])
    return {
        "natural_action_list_nll_per_callback": float(complete),
        "count_nll_per_callback": (
            totals["count_sum"] / max(1, totals["decision_atoms"])
        ),
        "fill_nll_per_callback": (
            totals["fill_sum"] / max(1, totals["decision_atoms"])
        ),
        "fill_nll_per_action": (
            totals["fill_sum"] / max(1, totals["action_atoms"])
        ),
        "delta_bit_nll_per_callback": (
            totals["delta_bit_sum"] / max(1, totals["decision_atoms"])
        ),
        "delta_bit_nll_per_action": (
            totals["delta_bit_sum"] / max(1, totals["action_atoms"])
        ),
        "delta_bit_nll_per_bit": (
            totals["delta_bit_sum"] / max(1, totals["delta_bit_atoms"])
        ),
        "decision_atoms": int(totals["decision_atoms"]),
        "action_atoms": int(totals["action_atoms"]),
        "delta_bit_atoms": int(totals["delta_bit_atoms"]),
        "l2_action_atoms": int(totals["l2_action_atoms"]),
        "llc_action_atoms": int(totals["llc_action_atoms"]),
        "training_guard_objective_terms": [
            "count_sum", "fill_sum", "delta_bit_sum",
        ],
    }


def decode_actions(
    model, contexts, base_lines, device,
    count_override=None, role="eval", chunk_len=4096,
):
    require_equal_lengths("decode", contexts, base_lines)
    count_logits_parts = []
    model.eval()
    with torch.no_grad():
        for start, stop in _iter_chunks(len(contexts), chunk_len):
            values = torch.from_numpy(contexts[start:stop]).to(device)
            logits = model.count_head(values)
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("categorical count logits are non-finite")
            count_logits_parts.append(logits.cpu())
    count_logits = torch.cat(count_logits_parts, dim=0)
    probabilities = torch.softmax(count_logits.to(torch.float64), dim=1)
    count_entropy = -(
        probabilities * torch.log(probabilities.clamp_min(1e-300))
    ).sum(dim=1)
    natural_counts = count_logits.argmax(dim=1).numpy().astype(np.int64)
    if count_override is None:
        counts = natural_counts
    else:
        counts = np.asarray(count_override, dtype=np.int64)
        if (
            len(counts) != len(base_lines)
            or bool((counts < 0).any())
            or bool((counts >= model.count_output_classes).any())
        ):
            raise RuntimeError("oracle count override is outside TRAIN support")

    predicted_lines = [[] for _ in base_lines]
    predicted_fills = [[] for _ in base_lines]
    fill_entropy_sum = 0.0
    action_atoms = 0
    enumerated_payloads = 0
    masked_payloads = 0
    selected_joint_log_probability_sum = 0.0
    callbacks_requiring_mask = set()

    def best_feasible_payload(bit_logits, base, used_targets):
        modal_bits, modal_log_probability, flip_costs = (
            modal_bernoulli_payload(bit_logits)
        )
        # At most len(used_targets) payloads can be forbidden, so the first
        # len(used_targets)+1 distinct k-best payloads contain a feasible one.
        attempts = 0
        for flip_cost, flipped_bits in k_best_flip_subsets(
            flip_costs, len(used_targets) + 1
        ):
            attempts += 1
            bits = list(modal_bits)
            for bit in flipped_bits:
                bits[bit] ^= 1
            delta = bits_to_modular_delta(bits)
            target = (int(base) + delta) % LINE_ADDRESS_MODULUS
            if target not in used_targets:
                return target, modal_log_probability - flip_cost, attempts
        raise RuntimeError(
            "exact k-best payload enumeration exhausted the mathematical "
            "used-target bound"
        )

    with torch.no_grad():
        for start, stop in _iter_chunks(len(contexts), chunk_len):
            values = torch.from_numpy(contexts[start:stop]).to(device)
            local_counts = counts[start:stop]
            maximum = int(local_counts.max()) if len(local_counts) else 0
            for rank in range(maximum):
                active_np = np.flatnonzero(local_counts > rank).astype(np.int64)
                if not len(active_np):
                    continue
                active = torch.from_numpy(active_np).to(
                    device=device, dtype=torch.long
                )
                ranks = torch.full(
                    (len(active_np),), rank, device=device, dtype=torch.long
                )
                heads = model.ranked_heads(
                    values.index_select(0, active), ranks
                )
                fill_logits = heads["fill_logits"]
                if (
                    not bool(torch.isfinite(fill_logits).all())
                    or not bool(torch.isfinite(
                        heads["l2_delta_bit_logits"]
                    ).all())
                    or not bool(torch.isfinite(
                        heads["llc_delta_bit_logits"]
                    ).all())
                ):
                    raise RuntimeError("fill/delta-bit decoder is non-finite")
                fill_probabilities = torch.softmax(
                    fill_logits.to(torch.float64), dim=1
                )
                fill_entropy_sum += float((- (
                    fill_probabilities * torch.log(
                        fill_probabilities.clamp_min(1e-300)
                    )
                ).sum(dim=1)).sum().item())
                action_atoms += len(active_np)
                fill_log_probability_rows = torch.log_softmax(
                    fill_logits.to(torch.float64), dim=1
                ).cpu().tolist()
                bit_rows = (
                    heads["l2_delta_bit_logits"].cpu().tolist(),
                    heads["llc_delta_bit_logits"].cpu().tolist(),
                )
                for head_row, local in enumerate(active_np):
                    row = start + int(local)
                    used_targets = set(predicted_lines[row])
                    candidates = []
                    callback_attempts = 0
                    for fill_index, fill in enumerate(FILL_LEVELS):
                        target, payload_log_probability, attempts = (
                            best_feasible_payload(
                                bit_rows[fill_index][head_row],
                                base_lines[row], used_targets,
                            )
                        )
                        callback_attempts += attempts
                        candidates.append((
                            fill_log_probability_rows[head_row][fill_index]
                            + payload_log_probability,
                            fill_index, int(target), int(fill), attempts,
                        ))
                    if not candidates:
                        raise RuntimeError(
                            "unique-target decode infeasible for role={} callback={} "
                            "rank={} selected_K={}; no count reduction or target "
                            "mutation is permitted".format(
                                role, row, rank, int(counts[row])
                            )
                        )
                    # Highest combined fill+payload log-probability wins.  An
                    # exact score tie chooses L2 before LLC, then lower target.
                    selected = min(
                        candidates,
                        key=lambda item: (-item[0], item[1], item[2]),
                    )
                    joint_log_probability, _, target, fill, _ = selected
                    predicted_lines[row].append(int(target))
                    predicted_fills[row].append(int(fill))
                    selected_joint_log_probability_sum += joint_log_probability
                    enumerated_payloads += callback_attempts
                    masked_payloads += callback_attempts - len(FILL_LEVELS)
                    if callback_attempts > len(FILL_LEVELS):
                        callbacks_requiring_mask.add(row)
    if any(
        len(items) != int(count)
        for items, count in zip(predicted_lines, counts)
    ):
        raise RuntimeError("rank decoder did not realize selected count")
    count_width = model.count_output_classes
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
        "decoded_max_actions_per_callback": int(counts.max()) if len(counts) else 0,
        "decoded_fill_histogram": {
            str(key): int(value) for key, value in sorted(Counter(
                fill for items in predicted_fills for fill in items
            ).items())
        },
        "mean_count_entropy": float(count_entropy.mean().item()),
        "mean_count_entropy_normalized": (
            float(count_entropy.mean().item()) / math.log(count_width)
            if count_width > 1 else 0.0
        ),
        "mean_fill_entropy": (
            fill_entropy_sum / action_atoms if action_atoms else None
        ),
        "mean_fill_entropy_normalized": (
            fill_entropy_sum / action_atoms / math.log(len(FILL_LEVELS))
            if action_atoms else None
        ),
        "mean_selected_fill_plus_payload_log_probability": (
            selected_joint_log_probability_sum / action_atoms
            if action_atoms else None
        ),
        "probability_threshold_used": False,
        "class_reweighting_used": False,
        "decode_prior_correction_used": False,
        "action_feedback_used": False,
        "normal_request_budget_used": False,
        "rank_action_logits_use_previous_action": False,
        "rank_logits_conditionally_independent_of_previous_actions": True,
        "target_uniqueness_feasibility_mask_used": True,
        "target_uniqueness_ignores_fill_level": True,
        "target_mutation_fallback_used": False,
        "count_reduction_fallback_used": False,
        "infeasible_unique_decode_behavior": "fail_closed",
        "kbest_payload_enumeration_exact": True,
        "fill_and_payload_log_probability_combined": True,
        "enumerated_payload_count": int(enumerated_payloads),
        "uniqueness_masked_payload_count": int(masked_payloads),
        "callbacks_requiring_feasibility_mask": len(
            callbacks_requiring_mask
        ),
        "duplicate_decoded_targets": sum(
            len(items) - len(set(items)) for items in predicted_lines
        ),
    }
    if diagnostics["duplicate_decoded_targets"] != 0:
        raise RuntimeError("unique-target feasibility mask emitted a duplicate target")
    return counts, predicted_lines, predicted_fills, diagnostics


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
        teacher = Counter((int(line), int(fill)) for line, fill in items)
        predicted_total += sum(predicted.values())
        teacher_total += sum(teacher.values())
        true_positive += sum((predicted & teacher).values())
        predicted_l2 = Counter({
            line: count for (line, fill), count in predicted.items() if fill == 2
        })
        teacher_l2_rows = Counter({
            line: count for (line, fill), count in teacher.items() if fill == 2
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
    fill_indices = [
        [FILL_LEVELS.index(int(fill)) for fill in callback]
        for callback in fills
    ]
    result = behavior_metrics(
        counts, lines, fill_indices, teacher, fill_levels=FILL_LEVELS
    )
    result.update(trigger_behavior_metrics(counts, teacher))
    result.update(joint_action_metrics(lines, fills, teacher))
    teacher_counts = np.asarray([len(items) for items in teacher], dtype=np.int64)
    confusion = Counter(
        (int(truth), int(prediction))
        for truth, prediction in zip(teacher_counts, counts)
    )
    result["count_confusion"] = {
        "{}->{}".format(truth, prediction): int(value)
        for (truth, prediction), value in sorted(confusion.items())
    }
    result["count_mae"] = (
        float(np.abs(np.asarray(counts) - teacher_counts).mean())
        if len(teacher_counts) else 0.0
    )
    result["request_ratio_vs_teacher"] = ratio(
        int(np.asarray(counts).sum()), int(teacher_counts.sum())
    )
    return result


def count_oracle_upper_bound(predicted_counts, teacher_actions):
    predicted = np.asarray(predicted_counts, dtype=np.int64)
    teacher = np.asarray([len(items) for items in teacher_actions], dtype=np.int64)
    true_positive = int(np.minimum(predicted, teacher).sum())
    predicted_total = int(predicted.sum())
    teacher_total = int(teacher.sum())
    precision = ratio(true_positive, predicted_total)
    recall = ratio(true_positive, teacher_total)
    return {
        "diagnostic_only": True,
        "replayed": False,
        "true_positive_actions_with_oracle_targets": true_positive,
        "target_precision_upper_bound": precision,
        "target_recall_upper_bound": recall,
        "target_f1_upper_bound": ratio(2 * precision * recall, precision + recall),
    }


def output_diagnostics(base_lines, counts, predicted_lines):
    duplicate_targets = self_targets = 0
    for base, lines in zip(base_lines, predicted_lines):
        duplicate_targets += len(lines) - len(set(lines))
        self_targets += sum(int(int(line) == int(base)) for line in lines)
    total = sum(map(len, predicted_lines))
    if int(np.asarray(counts).sum()) != total:
        raise RuntimeError("output accounting differs from emitted actions")
    return {
        "raw_predicted_action_count": total,
        "materialized_action_count": total,
        "raw_positive_callback_count": int((np.asarray(counts) > 0).sum()),
        "materialized_positive_callback_count": sum(bool(x) for x in predicted_lines),
        "self_target_actions": self_targets,
        "duplicate_target_actions": duplicate_targets,
        "direct_exact_delta_bit_actions": total,
        "action_vocabulary_used": False,
        "other_token_used": False,
        "duplicate_outputs_are_preserved_for_replay": False,
        "unique_targets_per_callback_enforced": True,
        "target_uniqueness_ignores_fill_level": True,
        "target_mutation_fallback_used": False,
        "count_reduction_fallback_used": False,
        "delta_legality_fallback": None,
    }


def train_model(model, bundles, targets, device, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    chunks = list(_iter_chunks(len(bundles["train"]["features"]), args.chunk_len))
    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        model.train()
        recurrent = None
        totals = Counter()
        optimizer_steps = 0
        for group_start in range(0, len(chunks), args.accumulate_chunks):
            optimizer.zero_grad(set_to_none=True)
            group = {
                "count": None, "fill": None, "delta_bit": None,
                "decisions": 0, "actions": 0, "delta_bits": 0,
            }
            for start, stop in chunks[
                group_start:group_start + args.accumulate_chunks
            ]:
                features = torch.from_numpy(
                    bundles["train"]["features"][start:stop]
                ).to(device)
                kinds = torch.from_numpy(
                    bundles["train"]["demand_kind"][start:stop]
                ).to(device)
                context, recurrent = model.encode(features, kinds, recurrent)
                recurrent = detach_state(recurrent)
                parts = chunk_loss_parts(
                    model, context,
                    tuple(value[start:stop] for value in targets["train"]),
                )
                if parts is None:
                    continue
                for source, target in (
                    ("count_sum", "count"),
                    ("fill_sum", "fill"),
                    ("delta_bit_sum", "delta_bit"),
                ):
                    group[target] = (
                        parts[source] if group[target] is None
                        else group[target] + parts[source]
                    )
                    totals[source] += float(parts[source].detach())
                group["decisions"] += parts["decision_atoms"]
                group["actions"] += parts["action_atoms"]
                group["delta_bits"] += parts["delta_bit_atoms"]
                totals["decision_atoms"] += parts["decision_atoms"]
                totals["action_atoms"] += parts["action_atoms"]
                totals["delta_bit_atoms"] += parts["delta_bit_atoms"]
                totals["l2_action_atoms"] += parts["l2_action_atoms"]
                totals["llc_action_atoms"] += parts["llc_action_atoms"]
            if group["decisions"] == 0:
                continue
            objective = (
                group["count"] + group["fill"] + group["delta_bit"]
            ) / float(group["decisions"])
            if not torch.isfinite(objective):
                raise RuntimeError("non-finite SPP v25 training objective")
            objective.backward()
            optimizer.step()
            optimizer_steps += 1

        guard_context = score_role_history(
            model, bundles, ("train", "guard"), device, args.chunk_len
        )["guard"]
        guard_nll = natural_list_nll(
            model, guard_context, targets["guard"], device
        )
        selection = (
            -guard_nll["natural_action_list_nll_per_callback"], -epoch
        )
        train_list_nll = (
            totals["count_sum"] + totals["fill_sum"]
            + totals["delta_bit_sum"]
        ) / max(1, totals["decision_atoms"])
        row = {
            "epoch": epoch,
            "train_natural_action_list_nll_per_callback": train_list_nll,
            "train_count_nll_per_callback": (
                totals["count_sum"] / max(1, totals["decision_atoms"])
            ),
            "train_fill_nll_per_callback": (
                totals["fill_sum"] / max(1, totals["decision_atoms"])
            ),
            "train_delta_bit_nll_per_callback": (
                totals["delta_bit_sum"] / max(1, totals["decision_atoms"])
            ),
            "train_delta_bit_nll_per_action": (
                totals["delta_bit_sum"] / max(1, totals["action_atoms"])
            ),
            "train_delta_bit_nll_per_bit": (
                totals["delta_bit_sum"] / max(1, totals["delta_bit_atoms"])
            ),
            "guard_natural_action_list_nll_per_callback": guard_nll[
                "natural_action_list_nll_per_callback"
            ],
            "guard_count_nll_per_callback": guard_nll["count_nll_per_callback"],
            "guard_fill_nll_per_callback": guard_nll[
                "fill_nll_per_callback"
            ],
            "guard_delta_bit_nll_per_callback": guard_nll[
                "delta_bit_nll_per_callback"
            ],
            "guard_delta_bit_nll_per_action": guard_nll[
                "delta_bit_nll_per_action"
            ],
            "guard_delta_bit_nll_per_bit": guard_nll[
                "delta_bit_nll_per_bit"
            ],
            "optimizer_steps": optimizer_steps,
            "selection_key": json.dumps(selection),
        }
        history.append(row)
        if best is None or selection > best["selection_key"]:
            best = {
                "epoch": epoch,
                "selection_key": selection,
                "guard_nll": guard_nll,
                "state_dict": copy.deepcopy({
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                }),
            }
        print(
            "[train:spp-v25:global] epoch={} train_list_nll={:.8f} "
            "guard_list_nll={:.8f}".format(
                epoch, train_list_nll,
                guard_nll["natural_action_list_nll_per_callback"],
            )
        )
    if best is None:
        raise RuntimeError("SPP v25 produced no checkpoint")
    model.load_state_dict(best["state_dict"])
    return history, best


def self_test_model(hidden_size):
    bit_priors = [[0.5] * LINE_ADDRESS_BITS for _ in FILL_LEVELS]
    for size in MODEL_POINTS["lstm"]:
        model = NaturalCardinalitySPPLSTM(
            size, [0.5, 0.5], [0.5, 0.5], bit_priors
        )
        observed = sum(parameter.numel() for parameter in model.parameters())
        expected = expected_parameter_count(size, 2)
        if observed != expected:
            raise RuntimeError(
                "SPP v25 parameter formula mismatch: {} != {}".format(
                    observed, expected
                )
            )
    model = NaturalCardinalitySPPLSTM(
        hidden_size, [0.5, 0.5], [0.5, 0.5], bit_priors
    )
    features = torch.zeros((5, RUNTIME_FEATURES))
    kinds = torch.tensor([True, False, True, False, True])
    changed = features.clone()
    changed[-1, 0] = 1.0
    model.eval()
    with torch.no_grad():
        first, _ = model.encode(features, kinds)
        second, _ = model.encode(changed, kinds)
    if not torch.equal(first[:-1], second[:-1]):
        raise RuntimeError("future callback changed a prior recurrent state")
    sample = NaturalCardinalitySPPLSTM(
        hidden_size, [0.5, 0.25, 0.25], [0.5, 0.5], bit_priors
    )
    with torch.no_grad():
        sample.count_head.weight.zero_()
        sample.count_head.bias[:] = torch.tensor([-5.0, -5.0, 5.0])
    decoded = decode_actions(
        sample, np.zeros((2, hidden_size), dtype=np.float32), [10, 20],
        torch.device("cpu"), role="self-test",
    )
    if decoded[0].tolist() != [2, 2] or any(
        len(items) != 2 for items in decoded[1]
    ):
        raise RuntimeError("categorical count did not schedule exactly K actions")
    if any(len(items) != len(set(items)) for items in decoded[1]):
        raise RuntimeError("target uniqueness feasibility mask did not remove duplicates")
    if decoded[3]["target_mutation_fallback_used"]:
        raise RuntimeError("unique decode mutated a target")
    fill_metric = complete_behavior_metrics(
        np.asarray([1]), [[11]], [[2]], [[(11, 2)]]
    )
    if fill_metric["fill_accuracy_on_matched_targets"] != 1.0:
        raise RuntimeError("explicit fill-level metric codec changed")
    toy_stream = {
        "context": [("DEMAND", 0, 0, 0), ("DEMAND", 64, 1, 1)],
        "demands": [(1, 0, 0, 0), (2, 64, 1, 0)],
        "demand_positions": np.asarray([0, 1], dtype=np.int64),
    }
    targets = build_context_targets(
        toy_stream, [[(1, 2)], []], 2
    )
    if targets[0].tolist() != [1, 0] or targets[1].tolist() != [[0], [-1]]:
        raise RuntimeError("natural no-STOP target construction changed")


def build_parser():
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
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--chunk-len", type=int, default=CHUNK_LEN)
    parser.add_argument("--accumulate-chunks", type=int, default=ACCUMULATE_CHUNKS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main():
    args = build_parser().parse_args()
    if MODEL_POINTS["lstm"].get(args.model_size) != args.pair_id:
        raise RuntimeError("model size/pair is not a configured v25 point")

    pinned = describe_model_points()["training_config"]
    actual = {
        "seed": args.seed,
        "epochs": args.epochs,
        "chunk_len": args.chunk_len,
        "accumulate_chunks": args.accumulate_chunks,
        "learning_rate": args.learning_rate,
    }
    if actual != pinned:
        raise RuntimeError(
            "RUN_ID pins training config: observed={} expected={}".format(
                actual, pinned
            )
        )
    source_contract = json.loads(args.source_contract.read_text())
    if source_contract.get("decision_effective_external_input") != SOURCE_INPUTS:
        raise RuntimeError("unexpected SPP source input contract")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
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
            "the pinned v25 run requires an A100; observed {}".format(device_name)
        )
    model_contract_module.self_test_contract()
    self_test_model(args.model_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # EVAL is deliberately absent until the GUARD-selected checkpoint has
    # reproduced exactly.  There is no architecture/core selection stage.
    roles = ["train", "guard"]
    stream_paths = {role: getattr(args, role + "_stream") for role in roles}
    action_paths = {
        role: getattr(args, role + "_teacher_actions") for role in roles
    }
    streams = {role: load_stream(stream_paths[role]) for role in roles}
    teachers = {
        role: load_teacher_actions(action_paths[role], streams[role]["demands"])
        for role in roles
    }
    bundles = {role: runtime_bundle(streams[role]) for role in roles}
    for role in roles:
        reproduced = runtime_bundle(streams[role])
        if (
            not np.array_equal(bundles[role]["features"], reproduced["features"])
            or not np.array_equal(
                bundles[role]["demand_kind"], reproduced["demand_kind"]
            )
        ):
            raise RuntimeError("{} runtime encoder is not reproducible".format(role))

    count_stats = count_statistics([len(items) for items in teachers["train"]])
    count_classes = count_stats["count_output_classes"]
    count_prior = count_stats["add_one_smoothed_natural_priors"]
    train_pair_frequencies = Counter(teacher_action_pairs(
        streams["train"], teachers["train"]
    ))
    fill_counts, fill_prior, delta_bit_one_counts, delta_bit_priors = (
        fill_and_bit_statistics(streams["train"], teachers["train"])
    )
    targets = {
        role: build_context_targets(
            streams[role], teachers[role], count_classes
        ) for role in roles
    }
    model = NaturalCardinalitySPPLSTM(
        args.model_size, count_prior, fill_prior, delta_bit_priors,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    expected_parameters = expected_parameter_count(
        args.model_size, count_classes,
    )
    if parameter_count != expected_parameters:
        raise RuntimeError("realized SPP v25 parameter count changed")

    history, best = train_model(model, bundles, targets, device, args)
    guard_context = score_role_history(
        model, bundles, ("train", "guard"), device, args.chunk_len
    )["guard"]
    reproduced_guard_nll = natural_list_nll(
        model, guard_context, targets["guard"], device
    )
    if reproduced_guard_nll != best["guard_nll"]:
        raise RuntimeError("selected GUARD checkpoint did not reproduce")

    roles.append("eval")
    stream_paths["eval"] = args.eval_stream
    action_paths["eval"] = args.eval_teacher_actions
    streams["eval"] = load_stream(stream_paths["eval"])
    teachers["eval"] = load_teacher_actions(
        action_paths["eval"], streams["eval"]["demands"]
    )
    bundles["eval"] = runtime_bundle(streams["eval"])
    reproduced_eval = runtime_bundle(streams["eval"])
    if (
        not np.array_equal(
            bundles["eval"]["features"], reproduced_eval["features"]
        )
        or not np.array_equal(
            bundles["eval"]["demand_kind"],
            reproduced_eval["demand_kind"],
        )
    ):
        raise RuntimeError("eval runtime encoder is not reproducible")
    targets["eval"] = build_context_targets(
        streams["eval"], teachers["eval"], count_classes
    )

    history_path = args.out_dir / "training_history.csv"
    model_path = args.out_dir / "model.pt"
    write_table(history_path, history)
    torch.save({
        "state_dict": model.state_dict(),
        "run_id": RUN_ID,
        "operation": OPERATION,
        "run_mode": "final",
        "core_type": CORE_TYPE,
        "model_size": args.model_size,
        "count_support": list(range(count_classes)),
        "count_prior": count_prior,
        "fill_prior": fill_prior,
        "delta_bit_priors": delta_bit_priors,
        "selected_epoch": best["epoch"],
        "selected_guard_nll": best["guard_nll"],
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": MODEL_REVISION,
        "decoder_revision": DECODER_REVISION,
    }, model_path)

    contract = describe_model_points()
    tag = model_tag("lstm", args.model_size)
    encoder_hash = runtime_encoder_sha256()
    metadata = {
        "run_id": RUN_ID,
        "operation": OPERATION,
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": MODEL_REVISION,
        "decoder_revision": DECODER_REVISION,
        "run_mode": "final",
        "parent_input_run_id": PARENT_INPUT_RUN_ID,
        "input_reuse": "v23 input package reused byte-for-byte",
        "trace": TRACE,
        "model_tag": tag,
        "matched_normal_prefetcher": POLICY,
        "neural_role": "standalone_direct_action_prefetcher",
        "model_family": "lstm",
        "track_model_family": "lstm",
        "model_size": args.model_size,
        "architecture_pair_id": args.pair_id,
        "core_type": CORE_TYPE,
        "global_core_fixed_for_all_capacities": True,
        "core_selection_used": False,
        "event_routed_core_used": False,
        "parameter_count": parameter_count,
        "realized_parameter_count": parameter_count,
        "expected_parameter_count": expected_parameters,
        "parameter_count_is_dataset_dependent": True,
        "parameter_formula": contract["parameter_formula"],
        "model_point_contract": contract,
        "parameter_storage_bytes_float32": parameter_count * 4,
        "peak_persistent_recurrent_state_bytes": 2 * args.model_size * 4,
        "persistent_recurrent_state": (
            "one global chronological one-layer LSTM hidden/cell pair"
        ),
        "dynamic_page_state_pages": 0,
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
        "source_decision_effective_external_input": SOURCE_INPUTS,
        "training_runtime_fields": SOURCE_INPUTS,
        "inference_runtime_fields": SOURCE_INPUTS,
        "same_external_input_contract": True,
        "training_inference_input_encoder_identical": True,
        "runtime_feature_count": RUNTIME_FEATURES,
        "runtime_encoding": contract["runtime_encoding"],
        "runtime_encoder_sha256": encoder_hash,
        "training_runtime_encoder_sha256": encoder_hash,
        "inference_runtime_encoder_sha256": encoder_hash,
        "model_does_not_use_pc": True,
        "pc_is_replay_transport_only": True,
        "model_input_is_causal_external_event_sequence_only": True,
        "cache_fill_feedback_used_as_raw_external_input": True,
        "cache_fill_private_state_used_as_model_input": False,
        "cache_hit_and_type_are_audit_only": True,
        "teacher_actions_are_model_inputs": False,
        "teacher_actions_are_model_inputs_scope": "labels_comparator_and_diagnosis_only",
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
        "decoder_training_mode": DECODER_TRAINING_MODE,
        "decoding_rule": DECODING_RULE,
        "decision_rule": DECODING_RULE,
        "count_training_objective": COUNT_OBJECTIVE,
        "fill_training_objective": FILL_OBJECTIVE,
        "delta_bit_training_objective": DELTA_BIT_OBJECTIVE,
        "categorical_count_head_used": True,
        "count_head_used": True,
        "fill_head_used": True,
        "fill_specific_delta_bit_heads_used": True,
        "both_fill_bit_heads_require_train_supervision": True,
        "joint_action_token_head_used": False,
        "action_vocabulary_used": False,
        "other_token_used": False,
        "count_regression_used": False,
        "hurdle_head_used": False,
        "stop_token_used": False,
        "stop_padding_used": False,
        "loss_class_reweighting_used": False,
        "decode_prior_correction_used": False,
        "manual_loss_weights_used": False,
        "count_zero_is_implicit_hurdle": True,
        "count_support": list(range(count_classes)),
        "count_support_source": "zero_through_maximum_TRAIN_teacher_count",
        "count_support_is_dataset_derived": True,
        "count_support_is_normal_request_budget": False,
        "count_support_is_tuned_degree": False,
        "count_train_statistics": count_stats,
        "count_class_weights": None,
        "action_loss_scope": "teacher_action_ranks_only",
        "fill_levels": list(FILL_LEVELS),
        "fill_train_class_counts": fill_counts,
        "fill_add_one_natural_priors": fill_prior,
        "fill_class_weights": None,
        "fill_head_initialization_source": (
            "add_one_smoothed_natural_TRAIN_fill_marginal"
        ),
        "delta_payload_bits": LINE_ADDRESS_BITS,
        "delta_payload_encoding": "exact_unsigned_modular_line_delta_bits",
        "delta_payload_float_or_clip_used": False,
        "delta_bit_train_one_counts_by_fill": delta_bit_one_counts,
        "delta_bit_add_one_priors_by_fill": delta_bit_priors,
        "delta_bit_head_initialization_source": (
            "teacher_fill_specific_add_one_smoothed_TRAIN_bit_marginals"
        ),
        "training_and_guard_objective_identical": True,
        "per_callback_objective_terms": [
            "count_cross_entropy", "real_rank_fill_cross_entropy",
            "all_real_rank_58_bit_Bernoulli_negative_log_likelihood",
        ],
        "all_actions_relative_to_current_demand": True,
        "delta_legality_constraints": [],
        "delta_legality_fallback": None,
        "duplicate_target_handling": (
            "rank_ordered_exact_kbest_payload_feasibility_selection_"
            "combining_fill_and_payload_log_probability_across_fills"
        ),
        "target_uniqueness_feasibility_mask_used": True,
        "target_uniqueness_ignores_fill_level": True,
        "target_mutation_fallback_used": False,
        "count_reduction_fallback_used": False,
        "infeasible_unique_decode_behavior": "fail_closed",
        "kbest_payload_enumeration_exact": True,
        "fill_and_payload_log_probability_combined": True,
        "source_action_order_preserved": True,
        "payload_training_statistics": {
            role: payload_statistics(streams[role], teachers[role])
            for role in roles
        },
        "separate_fill_head_used": True,
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_previous_predicted_action_used_as_input": False,
        "decoder_previous_sampled_action_used_as_input": False,
        "rank_action_logits_use_previous_action": False,
        "rank_logits_conditionally_independent_of_previous_actions": True,
        "rank_conditioning": "generic_four_component_sinusoidal_position_code",
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "guard_selected_checkpoint": True,
        "selected_epoch": best["epoch"],
        "selected_guard_natural_action_list_nll": best["guard_nll"],
        "guard_selection_composite_or_mean_used": False,
        "evaluation_used_for_selection": False,
        "evaluation_loaded_after_checkpoint_selection": True,
        "training_chunks_shuffled": False,
        "training_state_mode": "chronological_stateful_tbptt",
        "training_state_carried_across_chunks": True,
        "training_state_detached_between_chunks": True,
        "global_chronological_state_count": 1,
        "global_chronological_lstm": True,
        "routed_demand_fill_recurrent_paths": False,
        "page_local_causal_state": False,
        "handcrafted_semantic_features_used": False,
        "causal_derived_features": [],
        "causal_no_future_self_test": "PASS",
        "natural_no_stop_target_self_test": "PASS",
        "categorical_count_exact_K_self_test": "PASS",
        "rank_no_action_feedback_self_test": "PASS",
        "exact_modular_delta_bit_codec_self_test": "PASS",
        "mixed_sign_bernoulli_log_probability_self_test": "PASS",
        "exact_kbest_subset_enumeration_self_test": "PASS",
        "unique_target_decode_self_test": "PASS",
        "event_logger_schema": EVENT_LOGGER_SCHEMA,
        "action_attachment_mode": ACTION_ATTACHMENT_MODE,
        "teacher_action_canonicalization": CANONICALIZATION_MODE,
        "replay_preserves_explicit_fill_level": True,
        "same_source_input_offline_claim_allowed": True,
        "closed_loop_live_claim_allowed": False,
        "offline_input_feedback_origin": (
            "recorded cache-fill callbacks produced by the source SPP run"
        ),
        "comparison_claim_boundary": "matched-input open-loop offline comparison only",
        "input_archive_reused_byte_for_byte": True,
        "train_history": history,
        "model_checkpoint_sha256": sha256(model_path),
        "training_history_sha256": sha256(history_path),
        "source_contract_sha256": sha256(args.source_contract),
        "trainer_source_sha256": sha256(Path(__file__)),
        "model_contract_source_sha256": sha256(
            Path(__file__).with_name("model_contract.py")
        ),
        "threshold_free_policy_source_sha256": sha256(
            REPO_ROOT / "formal_NN_training" / "common" / "threshold_free_policy.py"
        ),
        "decision_router_source_sha256": decision_router_source_sha256(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
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
        metadata[role + "_teacher_actions_content_sha256"] = gzip_content_sha256(
            action_paths[role]
        )

    eval_context = score_role_history(
        model, bundles, ("train", "guard", "eval"), device, args.chunk_len
    )["eval"]
    eval_positions = streams["eval"]["demand_positions"]
    eval_decisions = eval_context[eval_positions]
    eval_bases = np.asarray(
        [row[2] for row in streams["eval"]["demands"]], dtype=np.int64
    )
    eval_decode = decode_actions(
        model, eval_decisions, eval_bases, device, role="eval"
    )
    heldout = complete_behavior_metrics(
        eval_decode[0], eval_decode[1], eval_decode[2], teachers["eval"]
    )
    teacher_counts = np.asarray(
        [len(items) for items in teachers["eval"]], dtype=np.int64
    )
    oracle_decode = decode_actions(
        model, eval_decisions, eval_bases, device,
        count_override=teacher_counts, role="diagnostic-oracle-count",
    )
    oracle_metrics = complete_behavior_metrics(
        oracle_decode[0], oracle_decode[1], oracle_decode[2], teachers["eval"]
    )
    oracle_diagnostics = {
        "diagnosis_only": True,
        "excluded_from_fair_replay_claims": True,
        "oracle_count_plus_nn_action": {
            "replayed": False,
            "behavior_metrics": oracle_metrics,
            "decoder_diagnostics": oracle_decode[3],
        },
        "nn_count_plus_oracle_action": count_oracle_upper_bound(
            eval_decode[0], teachers["eval"]
        ),
    }
    modal_delta_counts = Counter()
    for (delta, _), frequency in train_pair_frequencies.items():
        modal_delta_counts[int(delta)] += int(frequency)
    modal_delta = min(
        modal_delta_counts,
        key=lambda value: (-modal_delta_counts[value], value),
    )
    control_lines, control_fills = build_modal_llc_control(
        eval_bases, modal_delta
    )
    normal_path = args.out_dir / "offline_spp.replay.csv"
    nn_path = args.out_dir / "offline_nn.replay.csv"
    control_path = args.out_dir / "offline_modal_llc_control.replay.csv"
    normal = write_teacher_replay(
        normal_path, streams["eval"]["demands"], teachers["eval"]
    )
    neural = write_prediction_replay(
        nn_path, streams["eval"]["demands"], eval_decode[1], eval_decode[2]
    )
    control = write_prediction_replay(
        control_path, streams["eval"]["demands"], control_lines, control_fills
    )
    diagnostics = output_diagnostics(
        eval_bases, eval_decode[0], eval_decode[1]
    )
    metadata.update({
            "evaluation_policy_decode_count": 1,
            "diagnostic_eval_decode_count": 1,
            "oracle_diagnostics": oracle_diagnostics,
            "oracle_diagnostics_replayed": False,
            "oracle_diagnostics_excluded_from_fair_claims": True,
            "heldout_behavior_metrics": heldout,
            "decoder_eval_diagnostics": eval_decode[3],
            "action_output_diagnostics": diagnostics,
            "raw_predicted_action_count": diagnostics["raw_predicted_action_count"],
            "materialized_action_count": diagnostics["materialized_action_count"],
            "offline_normal_entries": normal[0],
            "offline_normal_triggers": normal[1],
            "offline_normal_fill_counts": normal[2],
            "offline_normal_fill_level_counts": normal[2],
            "offline_nn_entries": neural[0],
            "offline_nn_triggers": neural[1],
            "offline_nn_fill_counts": neural[2],
            "offline_nn_fill_level_counts": neural[2],
            "normal_list_sha256": sha256(normal_path),
            "nn_list_sha256": sha256(nn_path),
            "non_neural_control_name": "every_callback_TRAIN_modal_delta_FILL_LLC",
            "non_neural_control_uses_model": False,
            "non_neural_control_excluded_from_neural_claims": True,
            "non_neural_control_actions_per_callback": 1,
            "non_neural_control_fill_level": "FILL_LLC",
            "non_neural_control_delta_source": "TRAIN_teacher_action_frequency_only",
            "non_neural_control_modal_delta": int(modal_delta),
            "non_neural_control_modal_delta_train_frequency": int(
                modal_delta_counts[modal_delta]
            ),
            "non_neural_control_entries": control[0],
            "non_neural_control_triggers": control[1],
            "non_neural_control_fill_counts": control[2],
            "non_neural_control_list_sha256": sha256(control_path),
        })

    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "status": "PASS",
        "run_mode": "final",
        "model_tag": tag,
        "core_type": CORE_TYPE,
        "parameters": parameter_count,
        "selected_epoch": best["epoch"],
        "guard_natural_action_list_nll_per_callback": best["guard_nll"][
            "natural_action_list_nll_per_callback"
        ],
        "offline_nn_entries": metadata.get("offline_nn_entries"),
    }, indent=2))


if __name__ == "__main__":
    main()
