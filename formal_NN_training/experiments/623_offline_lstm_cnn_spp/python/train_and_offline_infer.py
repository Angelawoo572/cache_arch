#!/usr/bin/env python3
"""Train direct SPP-interface LSTM/CNN students and export replay actions.

The models consume only the causal line-aligned address sequence.  Captured
normal-SPP actions are training labels and an evaluation audit reference; they
are never passed to either model during scoring.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TRACE = "623.xalancbmk_s-700B"
POLICY = "spp"
PAGE_LINES = 64
FILL_LEVELS = (2, 4)
RUNTIME_FEATURES = 9
ACTION_CLASSES = PAGE_LINES * len(FILL_LEVELS)
MAX_ACTIONS_PER_CALLBACK = 32
LINE_DELTA_CLIP = 256
PAGE_DELTA_CLIP = 64
CNN_KERNEL_SIZE = 3
CNN_STRIDE = 1
CNN_DILATION = 1
EXPERIMENT_REVISION = "spp_direct_io_sliding_cnn_v3"
EVENT_LOGGER_SCHEMA = "623_causal_trigger_v5"
ACTION_ATTACHMENT_MODE = "explicit_trigger_event_id"
CANONICALIZATION_MODE = "per_target_min_fill_queue_effect"
PAIR_SPECS = {
    ("lstm", 4): ("p0", 880),
    ("cnn", 5): ("p0", 908),
    ("lstm", 8): ("p1", 1760),
    ("cnn", 10): ("p1", 1688),
    ("lstm", 16): ("p2", 3904),
    ("cnn", 24): ("p2", 3872),
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
    occurrences = defaultdict(int)
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "trace", "demand_idx", "pc", "address", "line", "cache_hit",
            "access_type", "pc_line_occ", "logger_schema",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError("{} missing {}".format(path, sorted(missing)))
        for index, row in enumerate(reader):
            pc = as_int(row["pc"])
            address = as_int(row["address"])
            line = as_int(row["line"])
            occurrence = as_int(row["pc_line_occ"])
            expected_occurrence = occurrences[(pc, line)]
            occurrences[(pc, line)] += 1
            if (
                row["trace"] != TRACE
                or as_int(row["demand_idx"]) != index
                or row["logger_schema"] != EVENT_LOGGER_SCHEMA
                or address != line << 6
                or occurrence != expected_occurrence
                or as_int(row["cache_hit"]) not in (0, 1)
            ):
                raise RuntimeError("stream identity/input failure at row {}".format(index))
            # Hit/type are intentionally validated but discarded: the audited
            # SPP prediction body does not read them and the restricted neural
            # input contract forbids them.
            rows.append((pc, address, line, occurrence))
    if not rows:
        raise RuntimeError("empty stream {}".format(path))
    return rows


def load_teacher_actions(path, rows):
    actions = [[] for _ in rows]
    last_pf_event = -1
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "trace", "policy", "demand_idx", "pc", "line", "pc_line_occ",
            "action_rank", "pf_line", "target_page_offset", "fill_level",
            "accepted", "duplicate", "trigger_event_id", "pf_event_id",
            "event_distance", "raw_action_count",
            "source_first_pf_event_id", "source_last_pf_event_id",
            "is_self_target", "canonicalization", "match_mode",
            "logger_schema",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError("{} missing {}".format(path, sorted(missing)))
        for row in reader:
            index = as_int(row["demand_idx"])
            if index < 0 or index >= len(rows):
                raise RuntimeError("teacher action demand_idx out of range")
            pc, _, line, occurrence = rows[index]
            identity = (
                as_int(row["pc"]), as_int(row["line"]),
                as_int(row["pc_line_occ"]),
            )
            if (
                row["trace"] != TRACE
                or row["policy"] != POLICY
                or identity != (pc, line, occurrence)
                or row["logger_schema"] != EVENT_LOGGER_SCHEMA
                or row["match_mode"] != ACTION_ATTACHMENT_MODE
            ):
                raise RuntimeError("teacher action identity failure at {}".format(index))
            if as_int(row["action_rank"]) != len(actions[index]) + 1:
                raise RuntimeError("noncontiguous action rank at {}".format(index))
            pf_event = as_int(row["pf_event_id"])
            trigger = as_int(row["trigger_event_id"])
            distance = as_int(row["event_distance"])
            if (
                pf_event <= last_pf_event
                or trigger >= pf_event
                or distance != pf_event - trigger
                or distance < 1
                or distance > 256
            ):
                raise RuntimeError("invalid action attachment at {}".format(index))
            pf_line = as_int(row["pf_line"])
            offset = as_int(row["target_page_offset"])
            fill = as_int(row["fill_level"])
            raw_action_count = as_int(row["raw_action_count"])
            source_first = as_int(row["source_first_pf_event_id"])
            source_last = as_int(row["source_last_pf_event_id"])
            is_self_target = as_int(row["is_self_target"])
            if (
                pf_line // PAGE_LINES != line // PAGE_LINES
                or offset != pf_line % PAGE_LINES
                or fill not in FILL_LEVELS
                or as_int(row["accepted"]) != 1
                or as_int(row["duplicate"]) not in (0, 1)
                or raw_action_count < 1
                or source_first != pf_event
                or source_last < source_first
                or is_self_target != int(pf_line == line)
                or row["canonicalization"] != CANONICALIZATION_MODE
            ):
                raise RuntimeError("invalid direct SPP action at {}".format(index))
            action = (pf_line, fill)
            if any(existing_line == pf_line for existing_line, _ in actions[index]):
                raise RuntimeError("two fill choices for one direct target at {}".format(index))
            actions[index].append(action)
            if len(actions[index]) > MAX_ACTIONS_PER_CALLBACK:
                raise RuntimeError("SPP action count exceeds source bound")
            last_pf_event = pf_event
    if not any(actions):
        raise RuntimeError("empty teacher action stream {}".format(path))
    return actions


def runtime_array(rows, previous_line=None):
    runtime = np.zeros((len(rows), RUNTIME_FEATURES), dtype=np.float32)
    previous = previous_line
    log_scale = math.log1p(4096.0)
    for index, (_, _, line, _) in enumerate(rows):
        page = line // PAGE_LINES
        offset = line % PAGE_LINES
        if previous is None:
            delta = 0
            page_delta = 0
            same_page = 0.0
        else:
            delta = line - previous
            page_delta = page - previous // PAGE_LINES
            same_page = float(page == previous // PAGE_LINES)
        page_chunks = [
            ((page >> shift) & 0xFFFF) / 65535.0
            for shift in (0, 16, 32, 48)
        ]
        runtime[index] = [
            offset / float(PAGE_LINES - 1),
            *page_chunks,
            np.clip(delta, -LINE_DELTA_CLIP, LINE_DELTA_CLIP)
            / float(LINE_DELTA_CLIP),
            min(1.0, math.log1p(abs(delta)) / log_scale),
            same_page,
            np.clip(page_delta, -PAGE_DELTA_CLIP, PAGE_DELTA_CLIP)
            / float(PAGE_DELTA_CLIP),
        ]
        previous = line
    return runtime, previous


def labels_from_actions(actions):
    labels = np.zeros((len(actions), ACTION_CLASSES), dtype=np.uint8)
    for index, items in enumerate(actions):
        for pf_line, fill in items:
            offset = pf_line % PAGE_LINES
            fill_index = FILL_LEVELS.index(fill)
            labels[index, offset * len(FILL_LEVELS) + fill_index] = 1
    return labels


class DirectActionLSTM(nn.Module):
    family = "lstm"

    def __init__(self, hidden_size):
        super().__init__()
        self.model_size = hidden_size
        self.lstm = nn.LSTM(RUNTIME_FEATURES, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, ACTION_CLASSES)

    def forward(self, runtime, state=None):
        temporal, state = self.lstm(runtime, state)
        return self.head(temporal), state


class DirectActionCNN(nn.Module):
    """Exactly one causal three-event moving filter plus a linear action head."""
    family = "cnn"

    def __init__(self, channels):
        super().__init__()
        self.model_size = channels
        self.conv = nn.Conv1d(
            RUNTIME_FEATURES,
            channels,
            kernel_size=CNN_KERNEL_SIZE,
            stride=CNN_STRIDE,
            dilation=CNN_DILATION,
            padding=0,
        )
        self.head = nn.Linear(channels, ACTION_CLASSES)

    @property
    def receptive_field(self):
        return CNN_KERNEL_SIZE

    def forward(self, runtime):
        x = runtime.transpose(1, 2)
        x = F.pad(x, (CNN_KERNEL_SIZE - 1, 0))
        temporal = torch.tanh(self.conv(x)).transpose(1, 2)
        return self.head(temporal)


def expected_parameter_count(family, size):
    if family == "lstm":
        return 4 * size * size + 172 * size + ACTION_CLASSES
    if family == "cnn":
        return 156 * size + ACTION_CLASSES
    raise ValueError(family)


def build_model(family, size, pair_id):
    spec = PAIR_SPECS.get((family, size))
    if spec is None or spec[0] != pair_id:
        raise RuntimeError("model family/size/pair is not a pinned matched point")
    model = DirectActionLSTM(size) if family == "lstm" else DirectActionCNN(size)
    observed = sum(parameter.numel() for parameter in model.parameters())
    expected = expected_parameter_count(family, size)
    if observed != expected or observed != spec[1]:
        raise RuntimeError(
            "parameter-count mismatch: observed={} formula={} pinned={}".format(
                observed, expected, spec[1]
            )
        )
    return model, observed


def iter_chunks(end, chunk_len):
    for start in range(0, end, chunk_len):
        yield start, min(end, start + chunk_len)


def weighted_loss(logits, labels, pos_weight):
    return F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight,
        reduction="sum",
    )


def optimizer_step(model, optimizer, loss_sum, element_count):
    if loss_sum is None or element_count <= 0:
        return 0
    (loss_sum / float(element_count)).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return 1


def train_lstm(
    model, runtime, labels, fit_end, device, epochs, chunk_len,
    accumulate_chunks, learning_rate, pos_weight,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    model.to(device)
    history = []
    chunks = list(iter_chunks(fit_end, chunk_len))
    pos_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        state = None
        group_loss = None
        group_elements = 0
        group_chunks = 0
        total_loss = 0.0
        total_elements = 0
        steps = 0
        for chunk_index, (start, stop) in enumerate(chunks):
            x = torch.from_numpy(runtime[start:stop]).unsqueeze(0).to(device)
            y = torch.from_numpy(labels[start:stop].astype(np.float32)).unsqueeze(0).to(device)
            logits, state = model(x, state)
            state = tuple(value.detach() for value in state)
            loss = weighted_loss(logits, y, pos_tensor)
            elements = y.numel()
            group_loss = loss if group_loss is None else group_loss + loss
            group_elements += elements
            group_chunks += 1
            total_loss += float(loss.detach().item())
            total_elements += elements
            if group_chunks == accumulate_chunks or chunk_index + 1 == len(chunks):
                steps += optimizer_step(model, optimizer, group_loss, group_elements)
                optimizer.zero_grad(set_to_none=True)
                group_loss = None
                group_elements = 0
                group_chunks = 0
        row = {
            "epoch": epoch,
            "weighted_loss_per_action_class": total_loss / max(1, total_elements),
            "chronological_chunks": len(chunks),
            "optimizer_steps": steps,
        }
        history.append(row)
        print("[train:lstm] epoch={} loss={:.8f}".format(epoch, row["weighted_loss_per_action_class"]))
    return history


def train_cnn(
    model, runtime, labels, fit_end, device, epochs, chunk_len,
    accumulate_chunks, learning_rate, pos_weight,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    model.to(device)
    history = []
    chunks = list(iter_chunks(fit_end, chunk_len))
    context = CNN_KERNEL_SIZE - 1
    pos_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        group_loss = None
        group_elements = 0
        group_chunks = 0
        total_loss = 0.0
        total_elements = 0
        steps = 0
        for chunk_index, (start, stop) in enumerate(chunks):
            context_start = max(0, start - context)
            offset = start - context_start
            x = torch.from_numpy(runtime[context_start:stop]).unsqueeze(0).to(device)
            y = torch.from_numpy(labels[start:stop].astype(np.float32)).unsqueeze(0).to(device)
            logits = model(x)[:, offset:]
            loss = weighted_loss(logits, y, pos_tensor)
            elements = y.numel()
            group_loss = loss if group_loss is None else group_loss + loss
            group_elements += elements
            group_chunks += 1
            total_loss += float(loss.detach().item())
            total_elements += elements
            if group_chunks == accumulate_chunks or chunk_index + 1 == len(chunks):
                steps += optimizer_step(model, optimizer, group_loss, group_elements)
                optimizer.zero_grad(set_to_none=True)
                group_loss = None
                group_elements = 0
                group_chunks = 0
        row = {
            "epoch": epoch,
            "weighted_loss_per_action_class": total_loss / max(1, total_elements),
            "chronological_chunks": len(chunks),
            "optimizer_steps": steps,
        }
        history.append(row)
        print("[train:cnn] epoch={} loss={:.8f}".format(epoch, row["weighted_loss_per_action_class"]))
    return history


def score_lstm(model, runtime, device, initial_state=None, chunk_len=8192):
    model.eval()
    scores = np.zeros((len(runtime), ACTION_CLASSES), dtype=np.float32)
    state = initial_state
    with torch.no_grad():
        for start, stop in iter_chunks(len(runtime), chunk_len):
            x = torch.from_numpy(runtime[start:stop]).unsqueeze(0).to(device)
            logits, state = model(x, state)
            state = tuple(value.detach() for value in state)
            scores[start:stop] = torch.sigmoid(logits[0]).cpu().numpy()
    return scores, state


def score_cnn(model, runtime, device, prefix_runtime=None, chunk_len=8192):
    model.eval()
    context = CNN_KERNEL_SIZE - 1
    if prefix_runtime is None:
        prefix_count = 0
        all_runtime = runtime
    else:
        prefix_count = min(context, len(prefix_runtime))
        all_runtime = np.concatenate([prefix_runtime[-prefix_count:], runtime], axis=0)
    scores = np.zeros((len(runtime), ACTION_CLASSES), dtype=np.float32)
    with torch.no_grad():
        for start, stop in iter_chunks(len(runtime), chunk_len):
            global_start = prefix_count + start
            global_stop = prefix_count + stop
            context_start = max(0, global_start - context)
            offset = global_start - context_start
            x = torch.from_numpy(all_runtime[context_start:global_stop]).unsqueeze(0).to(device)
            logits = model(x)[:, offset:]
            scores[start:stop] = torch.sigmoid(logits[0]).cpu().numpy()
    return scores


def self_test_cnn_causality():
    torch.manual_seed(123)
    model = DirectActionCNN(5).eval()
    layers = [module for module in model.modules() if isinstance(module, nn.Conv1d)]
    if len(layers) != 1:
        raise RuntimeError("CNN must contain exactly one temporal Conv1d")
    conv = layers[0]
    if (
        conv.kernel_size != (3,)
        or conv.stride != (1,)
        or conv.dilation != (1,)
        or conv.padding != (0,)
    ):
        raise RuntimeError("CNN geometry differs from the three-step sketch")
    runtime = torch.randn(1, 17, RUNTIME_FEATURES)
    pivot = 7
    with torch.no_grad():
        original = model(runtime)
        future_changed = runtime.clone()
        future_changed[:, pivot + 1:] += 1000.0
        future_output = model(future_changed)
    if not torch.allclose(original[:, :pivot + 1], future_output[:, :pivot + 1], atol=1e-6, rtol=1e-6):
        raise RuntimeError("CNN future-input causality self-test failed")
    old_changed = runtime.clone()
    old_changed[:, pivot - 3] += 1000.0
    with torch.no_grad():
        old_output = model(old_changed)
    if not torch.allclose(original[:, pivot], old_output[:, pivot], atol=1e-6, rtol=1e-6):
        raise RuntimeError("CNN receptive field exceeds three callbacks")
    runtime_np = runtime[0].numpy().astype(np.float32)
    full = torch.sigmoid(original[0]).detach().numpy()
    chunked = score_cnn(model, runtime_np, torch.device("cpu"), chunk_len=4)
    if not np.allclose(full, chunked, atol=1e-6, rtol=1e-6):
        raise RuntimeError("CNN chunk-overlap equivalence failed")
    prefixed = score_cnn(
        model, runtime_np[5:], torch.device("cpu"),
        prefix_runtime=runtime_np[:5], chunk_len=4,
    )
    if not np.allclose(full[5:], prefixed, atol=1e-6, rtol=1e-6):
        raise RuntimeError("CNN guard-prefix equivalence failed")


def decode(scores, rows, threshold, max_actions=MAX_ACTIONS_PER_CALLBACK):
    if scores.shape != (len(rows), ACTION_CLASSES):
        raise RuntimeError("score shape does not match direct action space")
    paired = scores.reshape(len(rows), PAGE_LINES, len(FILL_LEVELS))
    best_fill = paired.argmax(axis=2)
    best_score = paired.max(axis=2)
    selected = np.zeros(scores.shape, dtype=np.bool_)
    for index, (_, _, line, _) in enumerate(rows):
        eligible = np.flatnonzero(best_score[index] >= threshold)
        if len(eligible) > max_actions:
            order = np.argsort(-best_score[index, eligible], kind="stable")
            eligible = eligible[order[:max_actions]]
        for offset in eligible:
            selected[index, offset * 2 + int(best_fill[index, offset])] = True
    return selected


def fidelity(predicted, teacher, rows):
    if len(rows) != len(predicted):
        raise RuntimeError("fidelity rows do not match action matrices")
    teacher_bool = teacher.astype(np.bool_)
    tp = int((predicted & teacher_bool).sum())
    fp = int((predicted & ~teacher_bool).sum())
    fn = int((~predicted & teacher_bool).sum())
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    jaccard = tp / float(tp + fp + fn) if tp + fp + fn else 1.0
    pred_lines = predicted.reshape(len(predicted), PAGE_LINES, 2).any(axis=2)
    teacher_lines = teacher_bool.reshape(len(teacher_bool), PAGE_LINES, 2).any(axis=2)
    line_tp = int((pred_lines & teacher_lines).sum())
    line_fp = int((pred_lines & ~teacher_lines).sum())
    line_fn = int((~pred_lines & teacher_lines).sum())
    line_precision = line_tp / float(line_tp + line_fp) if line_tp + line_fp else 0.0
    line_recall = line_tp / float(line_tp + line_fn) if line_tp + line_fn else 0.0
    line_f1 = (
        2 * line_precision * line_recall / (line_precision + line_recall)
        if line_precision + line_recall else 0.0
    )
    exact_events = int(np.all(predicted == teacher_bool, axis=1).sum())
    predicted_self = 0
    teacher_self = 0
    for index, (_, _, line, _) in enumerate(rows):
        start = (line % PAGE_LINES) * len(FILL_LEVELS)
        stop = start + len(FILL_LEVELS)
        predicted_self += int(predicted[index, start:stop].sum())
        teacher_self += int(teacher_bool[index, start:stop].sum())
    predicted_total = int(predicted.sum())
    teacher_total = int(teacher_bool.sum())
    return {
        "predicted_actions": predicted_total,
        "teacher_actions": teacher_total,
        "true_positive_actions": tp,
        "false_positive_actions": fp,
        "false_negative_actions": fn,
        "action_precision": precision,
        "action_recall": recall,
        "action_f1": f1,
        "action_jaccard": jaccard,
        "target_line_precision": line_precision,
        "target_line_recall": line_recall,
        "target_line_f1": line_f1,
        "fill_accuracy_given_matched_target_line": tp / float(line_tp) if line_tp else 0.0,
        "exact_callback_match_rate": exact_events / float(len(predicted)),
        "predicted_actions_per_callback": predicted_total / float(len(predicted)),
        "teacher_actions_per_callback": teacher_total / float(len(predicted)),
        "predicted_self_target_actions": predicted_self,
        "teacher_self_target_actions": teacher_self,
        "predicted_self_target_action_rate": (
            predicted_self / float(predicted_total) if predicted_total else 0.0
        ),
        "teacher_self_target_action_rate": (
            teacher_self / float(teacher_total) if teacher_total else 0.0
        ),
    }


def calibrate(scores, labels, rows, start):
    finite = scores[start:].reshape(-1)
    if finite.size == 0:
        raise RuntimeError("empty calibration split")
    thresholds = [0.0, 1.0]
    thresholds.extend(np.linspace(0.05, 0.95, 19).tolist())
    thresholds.extend(float(np.quantile(finite, q)) for q in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995))
    teacher_rate = float(labels[start:].sum()) / max(1, len(labels) - start)
    rows_out = []
    for threshold in sorted(set(round(value, 8) for value in thresholds)):
        selected = decode(scores[start:], rows[start:], threshold)
        result = fidelity(selected, labels[start:], rows[start:])
        result.update({
            "threshold": threshold,
            "teacher_action_budget_per_callback": teacher_rate,
            "within_teacher_action_budget": int(
                result["predicted_actions_per_callback"]
                <= teacher_rate + 1.0 / max(1, len(labels) - start)
            ),
        })
        rows_out.append(result)
    eligible = [row for row in rows_out if row["within_teacher_action_budget"]]
    if not eligible:
        raise RuntimeError("no calibration threshold satisfies teacher action budget")
    eligible.sort(key=lambda row: (
        -row["action_f1"], -row["action_jaccard"],
        -row["action_precision"], -row["action_recall"],
        abs(row["predicted_actions_per_callback"] - teacher_rate),
        -row["threshold"],
    ))
    return eligible[0], rows_out


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
                writer.writerow([pc, line, occurrence, "0x{:x}".format(pf_line << 6), fill])
                fill_counts["FILL_L2" if fill == 2 else "FILL_LLC"] += 1
                entries += 1
    return entries, triggers, fill_counts


def write_prediction_replay(path, rows, selected, scores):
    entries = 0
    triggers = 0
    fill_counts = {"FILL_L2": 0, "FILL_LLC": 0}
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pc", "line", "occ", "prefetch_addr", "fill_level"])
        for index, (pc, _, line, occurrence) in enumerate(rows):
            classes = np.flatnonzero(selected[index])
            if classes.size:
                triggers += 1
                classes = classes[np.argsort(-scores[index, classes], kind="stable")]
            page_base = (line // PAGE_LINES) * PAGE_LINES
            for action_class in classes:
                offset = int(action_class) // 2
                fill = FILL_LEVELS[int(action_class) % 2]
                pf_line = page_base + offset
                writer.writerow([pc, line, occurrence, "0x{:x}".format(pf_line << 6), fill])
                fill_counts["FILL_L2" if fill == 2 else "FILL_LLC"] += 1
                entries += 1
    return entries, triggers, fill_counts


def model_tag(family, size):
    suffix = "lstm_h{}".format(size) if family == "lstm" else "cnn_c{}".format(size)
    return "direct_spp_{}".format(suffix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, choices=[POLICY])
    for role in ("train", "guard", "eval"):
        parser.add_argument("--{}-stream".format(role), required=True, type=Path)
        parser.add_argument("--{}-teacher-actions".format(role), required=True, type=Path)
    parser.add_argument("--source-contract", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-family", choices=["lstm", "cnn"], required=True)
    parser.add_argument("--model-size", type=int, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--chunk-len", type=int, default=1024)
    parser.add_argument("--accumulate-chunks", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    source_contract = json.loads(args.source_contract.read_text())
    if (
        source_contract.get("decision_effective_external_input") != ["addr"]
        or source_contract.get("output_action") != ["same-page pf_addr", "FILL_L2 or FILL_LLC"]
        or source_contract.get("self_target_action_semantics")
        != "allowed_by_source_lookahead_and_replayed"
    ):
        raise RuntimeError("unexpected SPP source contract")
    if args.model_size <= 0 or args.epochs <= 0 or args.chunk_len <= 0:
        raise SystemExit("invalid model/training dimensions")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    self_test_cnn_causality()
    stream_paths = {role: getattr(args, role + "_stream") for role in ("train", "guard", "eval")}
    action_paths = {role: getattr(args, role + "_teacher_actions") for role in ("train", "guard", "eval")}
    rows = {role: load_stream(path) for role, path in stream_paths.items()}
    teacher_actions = {
        role: load_teacher_actions(action_paths[role], rows[role])
        for role in ("train", "guard", "eval")
    }
    train_runtime, _ = runtime_array(rows["train"])
    guard_runtime, guard_last_line = runtime_array(rows["guard"])
    eval_runtime, _ = runtime_array(rows["eval"], previous_line=guard_last_line)
    labels = labels_from_actions(teacher_actions["train"])
    eval_labels = labels_from_actions(teacher_actions["eval"])
    fit_end = int(len(rows["train"]) * 0.8)
    if fit_end <= 0 or fit_end >= len(rows["train"]):
        raise RuntimeError("training stream too short for fit/calibration split")
    positives = int(labels[:fit_end].sum())
    total = fit_end * ACTION_CLASSES
    if positives <= 0 or positives >= total:
        raise RuntimeError("degenerate direct-action training labels")
    positive_weight = min(100.0, max(1.0, (total - positives) / float(positives)))

    model, parameter_count = build_model(args.model_family, args.model_size, args.pair_id)
    train_args = (
        model, train_runtime, labels, fit_end, device, args.epochs,
        args.chunk_len, args.accumulate_chunks, args.learning_rate,
        positive_weight,
    )
    history = train_lstm(*train_args) if args.model_family == "lstm" else train_cnn(*train_args)

    if args.model_family == "lstm":
        train_scores, _ = score_lstm(model, train_runtime, device)
        _, guard_state = score_lstm(model, guard_runtime, device)
        eval_scores, _ = score_lstm(model, eval_runtime, device, initial_state=guard_state)
    else:
        train_scores = score_cnn(model, train_runtime, device)
        eval_scores = score_cnn(model, eval_runtime, device, prefix_runtime=guard_runtime)

    calibration, sweep = calibrate(train_scores, labels, rows["train"], fit_end)
    threshold = calibration["threshold"]
    selected_eval = decode(eval_scores, rows["eval"], threshold)
    eval_fidelity = fidelity(selected_eval, eval_labels, rows["eval"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    normal_path = args.out_dir / "offline_spp.replay.csv"
    nn_path = args.out_dir / "offline_nn.replay.csv"
    normal_entries, normal_triggers, normal_fill_counts = write_teacher_replay(
        normal_path, rows["eval"], teacher_actions["eval"]
    )
    nn_entries, nn_triggers, nn_fill_counts = write_prediction_replay(
        nn_path, rows["eval"], selected_eval, eval_scores
    )
    write_table(args.out_dir / "policy_sweep.csv", sweep)
    torch.save({
        "state_dict": model.state_dict(),
        "model_family": args.model_family,
        "model_size": args.model_size,
        "runtime_features": RUNTIME_FEATURES,
        "action_classes": ACTION_CLASSES,
        "experiment_revision": EXPERIMENT_REVISION,
    }, args.out_dir / "model.pt")

    tag = model_tag(args.model_family, args.model_size)
    metadata = {
        "trace": TRACE,
        "model_tag": tag,
        "matched_normal_prefetcher": POLICY,
        "neural_role": "direct_spp_action_predictor",
        "model_family": args.model_family,
        "model_size": args.model_size,
        "architecture_pair_id": args.pair_id,
        "parameter_count": parameter_count,
        "seed": args.seed,
        "epochs": args.epochs,
        "chunk_len": args.chunk_len,
        "accumulate_chunks": args.accumulate_chunks,
        "learning_rate": args.learning_rate,
        "positive_class_weight": positive_weight,
        "threshold": threshold,
        "max_actions_per_callback": MAX_ACTIONS_PER_CALLBACK,
        "self_target_actions_allowed": True,
        "teacher_action_canonicalization": CANONICALIZATION_MODE,
        "fit_rows": fit_end,
        "calibration_rows": len(rows["train"]) - fit_end,
        "guard_rows": len(rows["guard"]),
        "eval_rows": len(rows["eval"]),
        "runtime_feature_count": RUNTIME_FEATURES,
        "runtime_feature_contract": [
            "current_page_offset", "page_number_bits_0_15",
            "page_number_bits_16_31", "page_number_bits_32_47",
            "page_number_bits_48_63", "signed_line_delta_from_prior",
            "log_absolute_line_delta_from_prior", "same_page_as_prior",
            "signed_page_delta_from_prior",
        ],
        "source_decision_effective_external_input": ["addr"],
        "source_signature_audit_only": ["ip", "cache_hit", "type"],
        "model_input_is_causal_address_sequence_only": True,
        "model_does_not_use_pc": True,
        "pc_is_replay_transport_only": True,
        "cache_hit_and_type_are_audit_only": True,
        "teacher_actions_are_model_inputs": False,
        "evaluation_teacher_actions_role": "normal comparator and fidelity audit only",
        "normal_candidate_bank_is_fixed": False,
        "nn_can_generate_actions_not_emitted_by_teacher": True,
        "direct_action_output_classes": ACTION_CLASSES,
        "direct_action_encoding": "target_page_offset*2 + fill_index",
        "forbidden_inputs": [
            "teacher_actions", "normal_candidate_bank", "cycle", "pc",
            "cache_hit", "access_type", "queue_state", "accepted",
            "duplicate", "SPP_confidence", "SPP_ST", "SPP_PT",
            "SPP_GHR", "SPP_FILTER_feedback", "future_eval_rows",
        ],
        "normal_policy_private_state_is_not_nn_input": True,
        "replay_preserves_explicit_fill_level": True,
        "training_chunks_shuffled": False,
        "training_labels_are_direct_spp_actions": True,
        "training_labels_use_future_rows": False,
        "causal_no_future_self_test": "PASS",
        "cnn_architecture_self_test": "PASS",
        "experiment_revision": EXPERIMENT_REVISION,
        "event_logger_schema": EVENT_LOGGER_SCHEMA,
        "action_attachment_mode": ACTION_ATTACHMENT_MODE,
        "training_state_mode": (
            "chronological_stateful_tbptt"
            if args.model_family == "lstm"
            else "three_event_causal_sliding_window"
        ),
        "training_state_carried_across_chunks": args.model_family == "lstm",
        "training_state_detached_between_chunks": args.model_family == "lstm",
        "cnn_temporal_layers": 1 if args.model_family == "cnn" else 0,
        "cnn_kernel_size": CNN_KERNEL_SIZE if args.model_family == "cnn" else 0,
        "cnn_stride": CNN_STRIDE if args.model_family == "cnn" else 0,
        "cnn_dilation": CNN_DILATION if args.model_family == "cnn" else 0,
        "cnn_receptive_field_events": CNN_KERNEL_SIZE if args.model_family == "cnn" else 0,
        "training_left_context_overlap": CNN_KERNEL_SIZE - 1 if args.model_family == "cnn" else 0,
        "offline_normal_entries": normal_entries,
        "offline_normal_triggers": normal_triggers,
        "offline_nn_entries": nn_entries,
        "offline_nn_triggers": nn_triggers,
        "offline_normal_fill_level_counts": normal_fill_counts,
        "offline_nn_fill_level_counts": nn_fill_counts,
        "normal_list_sha256": sha256(normal_path),
        "nn_list_sha256": sha256(nn_path),
        "source_contract_sha256": sha256(args.source_contract),
        "train_history": history,
        "calibration_choice": calibration,
        "eval_action_fidelity": eval_fidelity,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    for role in ("train", "guard", "eval"):
        metadata[role + "_stream_gzip_sha256"] = sha256(stream_paths[role])
        metadata[role + "_stream_content_sha256"] = gzip_content_sha256(stream_paths[role])
        metadata[role + "_teacher_actions_gzip_sha256"] = sha256(action_paths[role])
        metadata[role + "_teacher_actions_content_sha256"] = gzip_content_sha256(action_paths[role])
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "status": "PASS",
        "model_tag": tag,
        "family": args.model_family,
        "parameters": parameter_count,
        "threshold": threshold,
        "offline_normal_entries": normal_entries,
        "offline_nn_entries": nn_entries,
        "eval_action_f1": eval_fidelity["action_f1"],
    }, indent=2))


if __name__ == "__main__":
    main()
