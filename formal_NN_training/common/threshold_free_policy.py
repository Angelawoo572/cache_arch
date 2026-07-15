#!/usr/bin/env python3
"""Threshold-free, source-input-only neural prefetch policies.

The neural policy learns a distribution over request count and a ranking over
all legal same-page targets.  Inference is entirely categorical: argmax picks
the count and top-k picks that many targets.  There is no probability
threshold, comparator request-rate budget, or comparator-specific degree cap.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ADDRESS_BITS = 64
PAGE_LINES = 64
COUNT_CLASSES = PAGE_LINES + 1
CNN_KERNEL_SIZE = 7
CNN_STRIDE = 1
CNN_DILATIONS = (1, 6, 36, 216)
CNN_RECEPTIVE_FIELD = 1 + (CNN_KERNEL_SIZE - 1) * sum(CNN_DILATIONS)


def _bits64(value):
    value = int(value) & ((1 << ADDRESS_BITS) - 1)
    return [(value >> bit) & 1 for bit in range(ADDRESS_BITS)]


def runtime_bits(pcs, lines, use_pc):
    """Lossless binary encoding of the allowed external source inputs."""
    if len(pcs) != len(lines):
        raise RuntimeError("PC and line streams have different lengths")
    feature_count = ADDRESS_BITS * (2 if use_pc else 1)
    runtime = np.empty((len(lines), feature_count), dtype=np.float32)
    for index, (pc, line) in enumerate(zip(pcs, lines)):
        values = _bits64(line)
        if use_pc:
            values = _bits64(pc) + values
        runtime[index] = values
    return runtime


class CountRankLSTM(nn.Module):
    family = "lstm"

    def __init__(self, feature_count, hidden_size, fill_classes=0):
        super().__init__()
        self.feature_count = feature_count
        self.model_size = hidden_size
        self.fill_classes = fill_classes
        self.lstm = nn.LSTM(feature_count, hidden_size, batch_first=True)
        self.count_head = nn.Linear(hidden_size, COUNT_CLASSES)
        self.target_head = nn.Linear(hidden_size, PAGE_LINES)
        self.fill_head = (
            nn.Linear(hidden_size, PAGE_LINES * fill_classes)
            if fill_classes else None
        )

    def _heads(self, temporal):
        fill = None
        if self.fill_head is not None:
            fill = self.fill_head(temporal).reshape(
                *temporal.shape[:-1], PAGE_LINES, self.fill_classes
            )
        return self.count_head(temporal), self.target_head(temporal), fill

    def forward(self, runtime, state=None):
        temporal, state = self.lstm(runtime, state)
        return self._heads(temporal), state


class CountRankCNN(nn.Module):
    """Causal residual TCN over the chronological callback stream.

    Four kernel-7 filters with base-6 dilation provide a contiguous
    1,555-callback receptive field without any inference-time threshold.
    Kernel width seven is a local operator, not a seven-callback input window.
    """
    family = "cnn"

    def __init__(self, feature_count, channels, fill_classes=0):
        super().__init__()
        self.feature_count = feature_count
        self.model_size = channels
        self.fill_classes = fill_classes
        self.input_projection = nn.Conv1d(feature_count, channels, 1)
        self.temporal_blocks = nn.ModuleList([
            nn.Conv1d(
                channels,
                channels,
                kernel_size=CNN_KERNEL_SIZE,
                stride=CNN_STRIDE,
                dilation=dilation,
                padding=0,
            )
            for dilation in CNN_DILATIONS
        ])
        self.count_head = nn.Linear(channels, COUNT_CLASSES)
        self.target_head = nn.Linear(channels, PAGE_LINES)
        self.fill_head = (
            nn.Linear(channels, PAGE_LINES * fill_classes)
            if fill_classes else None
        )

    def forward(self, runtime):
        x = runtime.transpose(1, 2)
        x = torch.tanh(self.input_projection(x))
        for dilation, convolution in zip(
            CNN_DILATIONS, self.temporal_blocks
        ):
            left_context = dilation * (CNN_KERNEL_SIZE - 1)
            residual = torch.tanh(convolution(F.pad(x, (left_context, 0))))
            x = x + residual
        temporal = x.transpose(1, 2)
        fill = None
        if self.fill_head is not None:
            fill = self.fill_head(temporal).reshape(
                *temporal.shape[:-1], PAGE_LINES, self.fill_classes
            )
        return self.count_head(temporal), self.target_head(temporal), fill


def expected_parameter_count(family, feature_count, size, fill_classes=0):
    head_outputs = COUNT_CLASSES + PAGE_LINES + PAGE_LINES * fill_classes
    if family == "lstm":
        recurrent = 4 * size * size + (4 * feature_count + 8) * size
        return recurrent + head_outputs * size + head_outputs
    if family == "cnn":
        input_projection = (feature_count + 1) * size
        temporal_blocks = len(CNN_DILATIONS) * (
            CNN_KERNEL_SIZE * size * size + size
        )
        return (
            input_projection + temporal_blocks
            + head_outputs * size + head_outputs
        )
    raise ValueError(family)


def build_model(family, feature_count, size, fill_classes=0):
    if family == "lstm":
        model = CountRankLSTM(feature_count, size, fill_classes)
    elif family == "cnn":
        model = CountRankCNN(feature_count, size, fill_classes)
    else:
        raise RuntimeError("unknown model family {}".format(family))
    observed = sum(parameter.numel() for parameter in model.parameters())
    expected = expected_parameter_count(
        family, feature_count, size, fill_classes
    )
    if observed != expected:
        raise RuntimeError(
            "parameter-count mismatch: observed={} expected={}".format(
                observed, expected
            )
        )
    return model, observed


def targets_from_actions(lines, actions, fill_levels=()):
    """Convert normal-policy requests into count/rank/fill supervision only."""
    if len(lines) != len(actions):
        raise RuntimeError("action rows do not match demand rows")
    counts = np.zeros(len(lines), dtype=np.int64)
    targets = np.zeros((len(lines), PAGE_LINES), dtype=np.float32)
    fills = np.full((len(lines), PAGE_LINES), -1, dtype=np.int64)
    fill_to_index = {value: index for index, value in enumerate(fill_levels)}
    for index, (line, items) in enumerate(zip(lines, actions)):
        page = int(line) // PAGE_LINES
        for item in items:
            if fill_levels:
                target_line, fill = item
                if fill not in fill_to_index:
                    raise RuntimeError("action uses an unknown fill level")
            else:
                target_line = item
                fill = None
            target_line = int(target_line)
            if target_line // PAGE_LINES != page:
                raise RuntimeError("training action crosses a page")
            offset = target_line % PAGE_LINES
            if targets[index, offset]:
                raise RuntimeError("duplicate target line in one callback")
            targets[index, offset] = 1.0
            if fill_levels:
                fills[index, offset] = fill_to_index[fill]
        counts[index] = int(targets[index].sum())
    if np.any(counts > PAGE_LINES):
        raise RuntimeError("action count exceeds complete page action space")
    return counts, targets, fills


def _structured_nll(outputs, counts, targets, fills, reduction="mean"):
    """Composite categorical NLL with no manually weighted loss terms.

    A negative count marks a context-only event (for example SPP cache-fill
    feedback).  Such an event updates temporal state but has no action loss.
    """
    count_logits, target_logits, fill_logits = outputs
    flat_counts = counts.reshape(-1)
    flat_targets = targets.reshape(-1, PAGE_LINES)
    flat_count_logits = count_logits.reshape(-1, COUNT_CLASSES)
    flat_target_logits = target_logits.reshape(-1, PAGE_LINES)
    decision_rows = flat_counts >= 0
    decision_atoms = int(decision_rows.sum().detach().item())
    count_nll = count_logits.new_zeros(())
    if decision_atoms:
        count_nll = F.cross_entropy(
            flat_count_logits[decision_rows],
            flat_counts[decision_rows],
            reduction="sum",
        )
    target_log_probability = F.log_softmax(flat_target_logits, dim=-1)
    target_nll = -(target_log_probability * flat_targets).sum()
    target_atoms = int(flat_targets.sum().detach().item())
    fill_nll = count_nll.new_zeros(())
    fill_atoms = 0
    if fill_logits is not None:
        flat_fill_logits = fill_logits.reshape(
            -1, PAGE_LINES, fill_logits.shape[-1]
        )
        flat_fills = fills.reshape(-1, PAGE_LINES)
        selected = flat_fills >= 0
        fill_atoms = int(selected.sum().detach().item())
        if fill_atoms:
            fill_nll = F.cross_entropy(
                flat_fill_logits[selected], flat_fills[selected], reduction="sum"
            )
    total = count_nll + target_nll + fill_nll
    atoms = decision_atoms + target_atoms + fill_atoms
    if reduction == "sum":
        return total, atoms, (
            float(count_nll.detach().item()),
            float(target_nll.detach().item()),
            float(fill_nll.detach().item()),
        )
    return total / float(max(1, atoms)), atoms, None


def _detach_state(state):
    return None if state is None else tuple(value.detach() for value in state)


def _iter_chunks(end, chunk_len):
    for start in range(0, end, chunk_len):
        yield start, min(end, start + chunk_len)


def _tensor_targets(counts, targets, fills):
    return (
        torch.from_numpy(counts),
        torch.from_numpy(targets),
        torch.from_numpy(fills),
    )


def train_lstm(
    model, runtime, counts, targets, fills, fit_end, device, epochs,
    chunk_len, accumulate_chunks, learning_rate,
):
    # No hand-written regularizer or class weighting is used.  Learning rate
    # and epoch count are training hyperparameters, never inference gates.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    x = torch.from_numpy(runtime)
    count_tensor, target_tensor, fill_tensor = _tensor_targets(
        counts, targets, fills
    )
    chunks = list(_iter_chunks(fit_end, chunk_len))
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        state = None
        totals = [0.0, 0.0, 0.0]
        total_nll = 0.0
        total_atoms = 0
        steps = 0
        for group_start in range(0, len(chunks), accumulate_chunks):
            group = chunks[group_start:group_start + accumulate_chunks]
            group_atoms = sum(
                int(np.count_nonzero(counts[start:stop] >= 0))
                + int(targets[start:stop].sum())
                * (2 if model.fill_classes else 1)
                for start, stop in group
            )
            optimizer.zero_grad(set_to_none=True)
            for start, stop in group:
                xb = x[start:stop].unsqueeze(0).to(device)
                cb = count_tensor[start:stop].unsqueeze(0).to(device)
                tb = target_tensor[start:stop].unsqueeze(0).to(device)
                fb = fill_tensor[start:stop].unsqueeze(0).to(device)
                outputs, state = model(xb, state)
                state = _detach_state(state)
                loss_sum, atoms, components = _structured_nll(
                    outputs, cb, tb, fb, reduction="sum"
                )
                (loss_sum / float(max(1, group_atoms))).backward()
                total_nll += float(loss_sum.detach().item())
                total_atoms += atoms
                for position, value in enumerate(components):
                    totals[position] += value
            optimizer.step()
            steps += 1
        row = {
            "epoch": epoch,
            "nll_per_categorical_decision": total_nll / max(1, total_atoms),
            "count_nll": totals[0],
            "target_nll": totals[1],
            "fill_nll": totals[2],
            "chronological_chunks": len(chunks),
            "optimizer_steps": steps,
        }
        history.append(row)
        print(
            "[train:lstm] epoch={} nll={:.8f}".format(
                epoch, row["nll_per_categorical_decision"]
            )
        )
    return history


def train_cnn(
    model, runtime, counts, targets, fills, fit_end, device, epochs,
    chunk_len, accumulate_chunks, learning_rate,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    x = torch.from_numpy(runtime)
    count_tensor, target_tensor, fill_tensor = _tensor_targets(
        counts, targets, fills
    )
    chunks = list(_iter_chunks(fit_end, chunk_len))
    context = CNN_RECEPTIVE_FIELD - 1
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals = [0.0, 0.0, 0.0]
        total_nll = 0.0
        total_atoms = 0
        steps = 0
        for group_start in range(0, len(chunks), accumulate_chunks):
            group = chunks[group_start:group_start + accumulate_chunks]
            group_atoms = sum(
                int(np.count_nonzero(counts[start:stop] >= 0))
                + int(targets[start:stop].sum())
                * (2 if model.fill_classes else 1)
                for start, stop in group
            )
            optimizer.zero_grad(set_to_none=True)
            for start, stop in group:
                context_start = max(0, start - context)
                offset = start - context_start
                xb = x[context_start:stop].unsqueeze(0).to(device)
                cb = count_tensor[start:stop].unsqueeze(0).to(device)
                tb = target_tensor[start:stop].unsqueeze(0).to(device)
                fb = fill_tensor[start:stop].unsqueeze(0).to(device)
                raw = model(xb)
                outputs = tuple(
                    None if value is None else value[:, offset:]
                    for value in raw
                )
                loss_sum, atoms, components = _structured_nll(
                    outputs, cb, tb, fb, reduction="sum"
                )
                (loss_sum / float(max(1, group_atoms))).backward()
                total_nll += float(loss_sum.detach().item())
                total_atoms += atoms
                for position, value in enumerate(components):
                    totals[position] += value
            optimizer.step()
            steps += 1
        row = {
            "epoch": epoch,
            "nll_per_categorical_decision": total_nll / max(1, total_atoms),
            "count_nll": totals[0],
            "target_nll": totals[1],
            "fill_nll": totals[2],
            "chronological_chunks": len(chunks),
            "optimizer_steps": steps,
        }
        history.append(row)
        print(
            "[train:cnn] epoch={} nll={:.8f}".format(
                epoch, row["nll_per_categorical_decision"]
            )
        )
    return history


def score_lstm(model, runtime, device, initial_state=None, chunk_len=8192):
    model.eval()
    count = np.empty((len(runtime), COUNT_CLASSES), dtype=np.float32)
    target = np.empty((len(runtime), PAGE_LINES), dtype=np.float32)
    fill = (
        np.empty((len(runtime), PAGE_LINES, model.fill_classes), dtype=np.float32)
        if model.fill_classes else None
    )
    state = initial_state
    with torch.no_grad():
        for start, stop in _iter_chunks(len(runtime), chunk_len):
            xb = torch.from_numpy(runtime[start:stop]).unsqueeze(0).to(device)
            outputs, state = model(xb, state)
            state = _detach_state(state)
            count[start:stop] = outputs[0][0].cpu().numpy()
            target[start:stop] = outputs[1][0].cpu().numpy()
            if fill is not None:
                fill[start:stop] = outputs[2][0].cpu().numpy()
    return (count, target, fill), state


def advance_lstm_state(
    model, runtime, device, initial_state=None, chunk_len=8192,
):
    """Advance recurrent state through causal context without storing logits."""
    model.eval()
    state = initial_state
    with torch.no_grad():
        for start, stop in _iter_chunks(len(runtime), chunk_len):
            xb = torch.from_numpy(runtime[start:stop]).unsqueeze(0).to(device)
            _, state = model(xb, state)
            state = _detach_state(state)
    return state


def score_cnn(
    model, runtime, device, prefix_runtime=None, chunk_len=8192,
):
    model.eval()
    context = CNN_RECEPTIVE_FIELD - 1
    prefix_count = (
        0 if prefix_runtime is None else min(context, len(prefix_runtime))
    )
    all_runtime = (
        runtime if prefix_count == 0
        else np.concatenate([prefix_runtime[-prefix_count:], runtime], axis=0)
    )
    count = np.empty((len(runtime), COUNT_CLASSES), dtype=np.float32)
    target = np.empty((len(runtime), PAGE_LINES), dtype=np.float32)
    fill = (
        np.empty((len(runtime), PAGE_LINES, model.fill_classes), dtype=np.float32)
        if model.fill_classes else None
    )
    with torch.no_grad():
        for start, stop in _iter_chunks(len(runtime), chunk_len):
            global_start = prefix_count + start
            global_stop = prefix_count + stop
            context_start = max(0, global_start - context)
            offset = global_start - context_start
            xb = torch.from_numpy(
                all_runtime[context_start:global_stop]
            ).unsqueeze(0).to(device)
            outputs = model(xb)
            count[start:stop] = outputs[0][0, offset:].cpu().numpy()
            target[start:stop] = outputs[1][0, offset:].cpu().numpy()
            if fill is not None:
                fill[start:stop] = outputs[2][0, offset:].cpu().numpy()
    return count, target, fill


def decode(logits):
    """Categorical count argmax plus deterministic top-count target ranking."""
    count_logits, target_logits, fill_logits = logits
    if count_logits.shape[0] != target_logits.shape[0]:
        raise RuntimeError("count and target logits have different row counts")
    counts = count_logits.argmax(axis=1).astype(np.int64)
    selected = np.zeros(target_logits.shape, dtype=np.bool_)
    fill_choice = np.full(target_logits.shape, -1, dtype=np.int64)
    for index, count in enumerate(counts):
        if count:
            order = np.argsort(-target_logits[index], kind="stable")
            offsets = order[:int(count)]
            selected[index, offsets] = True
            if fill_logits is not None:
                fill_choice[index, offsets] = fill_logits[
                    index, offsets
                ].argmax(axis=1)
    return counts, selected, fill_choice


def behavior_metrics(
    predicted_counts, predicted_targets, predicted_fills,
    target_counts, target_targets, target_fills,
):
    truth = target_targets.astype(np.bool_)
    predicted = predicted_targets.astype(np.bool_)
    tp = int(np.logical_and(predicted, truth).sum())
    fp = int(np.logical_and(predicted, ~truth).sum())
    fn = int(np.logical_and(~predicted, truth).sum())
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    fill_matches = 0
    fill_total = 0
    if target_fills is not None and np.any(target_fills >= 0):
        jointly_selected = np.logical_and(predicted, truth)
        fill_total = int(jointly_selected.sum())
        fill_matches = int(np.logical_and(
            jointly_selected, predicted_fills == target_fills
        ).sum())
    return {
        "callbacks": int(len(target_counts)),
        "predicted_actions": int(predicted.sum()),
        "normal_actions": int(truth.sum()),
        "count_exact_match_rate": float(
            np.mean(predicted_counts == target_counts)
        ),
        "target_precision": precision,
        "target_recall": recall,
        "target_f1": f1,
        "fill_accuracy_on_matched_targets": (
            fill_matches / float(fill_total) if fill_total else None
        ),
    }


def self_test_cnn(feature_count, fill_classes=0):
    torch.manual_seed(123)
    model = CountRankCNN(feature_count, 5, fill_classes).eval()
    layers = [m for m in model.modules() if isinstance(m, nn.Conv1d)]
    if len(layers) != 1 + len(CNN_DILATIONS):
        raise RuntimeError(
            "CNN must contain one projection plus four temporal blocks"
        )
    projection, temporal_layers = layers[0], layers[1:]
    if projection.kernel_size != (1,) or projection.padding != (0,):
        raise RuntimeError("CNN input projection geometry changed")
    for dilation, convolution in zip(CNN_DILATIONS, temporal_layers):
        if (
            convolution.kernel_size != (CNN_KERNEL_SIZE,)
            or convolution.stride != (CNN_STRIDE,)
            or convolution.dilation != (dilation,)
            or convolution.padding != (0,)
        ):
            raise RuntimeError("CNN temporal geometry differs from contract")

    # Kernel positions 0..6 at each base-6 dilation form a complete mixed-radix
    # representation of every causal lag 0..1554.  This proves that the large
    # receptive field has no blind temporal holes; it is a contiguous sliding
    # history, not four sparsely sampled snapshots.
    reachable_lags = {0}
    for dilation in CNN_DILATIONS:
        reachable_lags = {
            previous + tap * dilation
            for previous in reachable_lags
            for tap in range(CNN_KERNEL_SIZE)
        }
    if reachable_lags != set(range(CNN_RECEPTIVE_FIELD)):
        raise RuntimeError("CNN receptive field is not contiguous")
    runtime = torch.randn(1, CNN_RECEPTIVE_FIELD + 20, feature_count)
    pivot = CNN_RECEPTIVE_FIELD + 5
    with torch.no_grad():
        original = model(runtime)
        future = runtime.clone()
        future[:, pivot + 1:] += 1000.0
        changed = model(future)
    for left, right in zip(original, changed):
        if left is not None and not torch.allclose(
            left[:, :pivot + 1], right[:, :pivot + 1],
            atol=1e-6, rtol=1e-6,
        ):
            raise RuntimeError("CNN future-input causality self-test failed")
    old = runtime.clone()
    old[:, pivot - CNN_RECEPTIVE_FIELD] += 1000.0
    with torch.no_grad():
        old_output = model(old)
    for baseline, changed_old in zip(original, old_output):
        if baseline is not None and not torch.allclose(
            baseline[:, pivot], changed_old[:, pivot],
            atol=1e-6, rtol=1e-6,
        ):
            raise RuntimeError("CNN receptive field exceeds its contract")


__all__ = [
    "ADDRESS_BITS", "CNN_DILATIONS", "CNN_KERNEL_SIZE",
    "CNN_RECEPTIVE_FIELD", "CNN_STRIDE", "COUNT_CLASSES", "CountRankCNN",
    "CountRankLSTM", "PAGE_LINES",
    "advance_lstm_state", "behavior_metrics", "build_model", "decode",
    "expected_parameter_count",
    "runtime_bits", "score_cnn", "score_lstm", "self_test_cnn",
    "targets_from_actions", "train_cnn", "train_lstm",
]
