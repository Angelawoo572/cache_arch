#!/usr/bin/env python3
"""Shared causal LSTM / one-filter CNN training primitives."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CNN_KERNEL_SIZE = 3
CNN_STRIDE = 1
CNN_DILATION = 1


class DirectActionLSTM(nn.Module):
    family = "lstm"

    def __init__(self, feature_count, action_classes, hidden_size):
        super().__init__()
        self.feature_count = feature_count
        self.action_classes = action_classes
        self.model_size = hidden_size
        self.lstm = nn.LSTM(feature_count, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, action_classes)

    def forward(self, runtime, state=None):
        temporal, state = self.lstm(runtime, state)
        return self.head(temporal), state


class DirectActionCNN(nn.Module):
    """Exactly one causal filter over the current and prior two callbacks."""
    family = "cnn"

    def __init__(self, feature_count, action_classes, channels):
        super().__init__()
        self.feature_count = feature_count
        self.action_classes = action_classes
        self.model_size = channels
        self.conv = nn.Conv1d(
            feature_count, channels, kernel_size=CNN_KERNEL_SIZE,
            stride=CNN_STRIDE, dilation=CNN_DILATION, padding=0,
        )
        self.head = nn.Linear(channels, action_classes)

    def forward(self, runtime):
        x = runtime.transpose(1, 2)
        x = F.pad(x, (CNN_KERNEL_SIZE - 1, 0))
        temporal = torch.tanh(self.conv(x)).transpose(1, 2)
        return self.head(temporal)


def expected_parameter_count(family, feature_count, action_classes, size):
    if family == "lstm":
        return 4 * size * size + (4 * feature_count + 8 + action_classes) * size + action_classes
    if family == "cnn":
        return (CNN_KERNEL_SIZE * feature_count + 1 + action_classes) * size + action_classes
    raise ValueError(family)


def build_model(family, feature_count, action_classes, size, pair_id, pair_specs):
    spec = pair_specs.get((family, size))
    if spec is None or spec[0] != pair_id:
        raise RuntimeError("model family/size/pair is not a pinned matched point")
    if family == "lstm":
        model = DirectActionLSTM(feature_count, action_classes, size)
    else:
        model = DirectActionCNN(feature_count, action_classes, size)
    observed = sum(parameter.numel() for parameter in model.parameters())
    expected = expected_parameter_count(family, feature_count, action_classes, size)
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


def _optimizer_step(model, optimizer):
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return 1


def train_lstm(model, runtime, labels, fit_end, device, epochs, chunk_len, accumulate_chunks, learning_rate, pos_weight):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    model.to(device)
    history = []
    chunks = list(iter_chunks(fit_end, chunk_len))
    pos_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    for epoch in range(1, epochs + 1):
        model.train()
        state = None
        total_loss = 0.0
        total_elements = 0
        steps = 0
        for group_start in range(0, len(chunks), accumulate_chunks):
            group = chunks[group_start:group_start + accumulate_chunks]
            group_elements = sum((stop - start) * model.action_classes for start, stop in group)
            optimizer.zero_grad(set_to_none=True)
            for start, stop in group:
                x = torch.from_numpy(runtime[start:stop]).unsqueeze(0).to(device)
                y = torch.from_numpy(labels[start:stop].astype(np.float32)).unsqueeze(0).to(device)
                logits, state = model(x, state)
                state = tuple(value.detach() for value in state)
                loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_tensor, reduction="sum")
                # Hidden/cell values cross the boundary, but the graph does
                # not.  Immediate backward avoids retaining many chunk graphs.
                (loss / float(group_elements)).backward()
                total_loss += float(loss.detach().item())
                total_elements += y.numel()
            steps += _optimizer_step(model, optimizer)
        row = {
            "epoch": epoch,
            "weighted_loss_per_action_class": total_loss / max(1, total_elements),
            "chronological_chunks": len(chunks),
            "optimizer_steps": steps,
        }
        history.append(row)
        print("[train:lstm] epoch={} loss={:.8f}".format(epoch, row["weighted_loss_per_action_class"]))
    return history


def train_cnn(model, runtime, labels, fit_end, device, epochs, chunk_len, accumulate_chunks, learning_rate, pos_weight):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    model.to(device)
    history = []
    chunks = list(iter_chunks(fit_end, chunk_len))
    context = CNN_KERNEL_SIZE - 1
    pos_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_elements = 0
        steps = 0
        for group_start in range(0, len(chunks), accumulate_chunks):
            group = chunks[group_start:group_start + accumulate_chunks]
            group_elements = sum((stop - start) * model.action_classes for start, stop in group)
            optimizer.zero_grad(set_to_none=True)
            for start, stop in group:
                context_start = max(0, start - context)
                offset = start - context_start
                x = torch.from_numpy(runtime[context_start:stop]).unsqueeze(0).to(device)
                y = torch.from_numpy(labels[start:stop].astype(np.float32)).unsqueeze(0).to(device)
                logits = model(x)[:, offset:]
                loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_tensor, reduction="sum")
                (loss / float(group_elements)).backward()
                total_loss += float(loss.detach().item())
                total_elements += y.numel()
            steps += _optimizer_step(model, optimizer)
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
    scores = np.zeros((len(runtime), model.action_classes), dtype=np.float32)
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
    prefix_count = 0 if prefix_runtime is None else min(context, len(prefix_runtime))
    all_runtime = runtime if prefix_count == 0 else np.concatenate([prefix_runtime[-prefix_count:], runtime], axis=0)
    scores = np.zeros((len(runtime), model.action_classes), dtype=np.float32)
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


def self_test_cnn(feature_count, action_classes):
    torch.manual_seed(123)
    model = DirectActionCNN(feature_count, action_classes, 5).eval()
    layers = [module for module in model.modules() if isinstance(module, nn.Conv1d)]
    if len(layers) != 1:
        raise RuntimeError("CNN must contain exactly one temporal Conv1d")
    conv = layers[0]
    if conv.kernel_size != (3,) or conv.stride != (1,) or conv.dilation != (1,) or conv.padding != (0,):
        raise RuntimeError("CNN geometry differs from the three-event moving filter")
    runtime = torch.randn(1, 17, feature_count)
    pivot = 7
    with torch.no_grad():
        original = model(runtime)
        future = runtime.clone()
        future[:, pivot + 1:] += 1000.0
        changed = model(future)
    if not torch.allclose(original[:, :pivot + 1], changed[:, :pivot + 1], atol=1e-6, rtol=1e-6):
        raise RuntimeError("CNN future-input causality self-test failed")
    old = runtime.clone()
    old[:, pivot - 3] += 1000.0
    with torch.no_grad():
        old_output = model(old)
    if not torch.allclose(original[:, pivot], old_output[:, pivot], atol=1e-6, rtol=1e-6):
        raise RuntimeError("CNN receptive field exceeds three callbacks")
    full = torch.sigmoid(original[0]).detach().numpy()
    chunked = score_cnn(model, runtime[0].numpy().astype(np.float32), torch.device("cpu"), chunk_len=4)
    if not np.allclose(full, chunked, atol=1e-6, rtol=1e-6):
        raise RuntimeError("CNN chunk-overlap equivalence failed")


__all__ = [
    "CNN_DILATION", "CNN_KERNEL_SIZE", "CNN_STRIDE", "DirectActionCNN",
    "DirectActionLSTM", "build_model", "expected_parameter_count",
    "score_cnn", "score_lstm", "self_test_cnn", "train_cnn", "train_lstm",
]
