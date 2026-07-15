#!/usr/bin/env python3
"""Train a matched LSTM or shallow sliding CNN gate for stride/SPP on 623.

Each policy is an independent matched track.  The normal replay and its neural
gate use the exact same live-policy candidate stream and PC-line-occ replay
transport.  The model sees only current/prior cache-line-derived features plus
candidate address/rank features; PC, cycle, hit/miss, queue state, and future
evaluation rows are forbidden.
"""
import argparse
import bisect
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
POLICIES = {"stride", "spp"}
PAGE_LINES = 64
RUNTIME_FEATURES = 5
CANDIDATE_FEATURES = 3
LINE_DELTA_CLIP = 256
PAGE_DELTA_CLIP = 64
CANDIDATE_DELTA_CLIP = 512
CNN_KERNEL_SIZE = 3
CNN_STRIDE = 1
CNN_DILATION = 1
EXPERIMENT_REVISION = "stride_spp_sliding_cnn_v2"


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
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"trace", "demand_idx", "pc", "line", "pc_line_occ"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError("{} missing {}".format(path, sorted(missing)))
        for index, row in enumerate(reader):
            if row["trace"] != TRACE or as_int(row["demand_idx"]) != index:
                raise RuntimeError("stream identity/ordering failure at row {}".format(index))
            rows.append((
                as_int(row["pc"]),
                as_int(row["line"]),
                as_int(row["pc_line_occ"]),
            ))
    if not rows:
        raise RuntimeError("empty stream: {}".format(path))
    return rows


def load_candidate_bank(path, policy, rows):
    bank = [[] for _ in rows]
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "trace", "policy", "demand_idx", "pc", "line", "pc_line_occ",
            "candidate_rank", "pf_line",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError("{} missing {}".format(path, sorted(missing)))
        for row in reader:
            index = as_int(row["demand_idx"])
            if index < 0 or index >= len(rows):
                raise RuntimeError("candidate demand_idx out of range")
            pc, line, occ = rows[index]
            observed = (
                as_int(row["pc"]),
                as_int(row["line"]),
                as_int(row["pc_line_occ"]),
            )
            if row["trace"] != TRACE or row["policy"] != policy or observed != (pc, line, occ):
                raise RuntimeError("candidate transport identity mismatch at {}".format(index))
            rank = as_int(row["candidate_rank"])
            if rank != len(bank[index]) + 1:
                raise RuntimeError("non-contiguous candidate rank at demand {}".format(index))
            bank[index].append(as_int(row["pf_line"]))
    if not any(bank):
        raise RuntimeError("empty {} candidate bank: {}".format(policy, path))
    return bank


def runtime_arrays(rows, previous_line=None):
    runtime = np.zeros((len(rows), RUNTIME_FEATURES), dtype=np.float32)
    prev = previous_line
    log_scale = math.log1p(4096.0)
    for index, (_, line, _) in enumerate(rows):
        offset = line % PAGE_LINES
        if prev is None:
            delta = 0
            page_delta = 0
            same_page = 0.0
        else:
            delta = line - prev
            page_delta = line // PAGE_LINES - prev // PAGE_LINES
            same_page = float(line // PAGE_LINES == prev // PAGE_LINES)
        runtime[index] = [
            offset / float(PAGE_LINES - 1),
            np.clip(delta, -LINE_DELTA_CLIP, LINE_DELTA_CLIP) / float(LINE_DELTA_CLIP),
            min(1.0, math.log1p(abs(delta)) / log_scale),
            same_page,
            np.clip(page_delta, -PAGE_DELTA_CLIP, PAGE_DELTA_CLIP) / float(PAGE_DELTA_CLIP),
        ]
        prev = line
    return runtime, prev


def candidate_arrays(rows, bank, max_candidates):
    candidate = np.zeros(
        (len(rows), max_candidates, CANDIDATE_FEATURES), dtype=np.float32
    )
    valid = np.zeros((len(rows), max_candidates), dtype=np.bool_)
    target = np.zeros((len(rows), max_candidates), dtype=np.int64)
    for index, (_, line, _) in enumerate(rows):
        for slot, pf_line in enumerate(bank[index]):
            if slot >= max_candidates:
                raise RuntimeError("candidate count exceeds pinned maximum")
            delta = pf_line - line
            valid[index, slot] = True
            target[index, slot] = pf_line
            candidate[index, slot] = [
                np.clip(
                    delta, -CANDIDATE_DELTA_CLIP, CANDIDATE_DELTA_CLIP
                ) / float(CANDIDATE_DELTA_CLIP),
                (pf_line % PAGE_LINES) / float(PAGE_LINES - 1),
                (slot + 1) / float(max_candidates),
            ]
    return candidate, valid, target


def training_labels(rows, valid, target, min_lead, max_lead, fit_end):
    positions = defaultdict(list)
    for index, (_, line, _) in enumerate(rows):
        positions[line].append(index)
    labels = np.zeros(valid.shape, dtype=np.float32)
    for index in range(len(rows)):
        split_end = fit_end if index < fit_end else len(rows)
        if index + min_lead >= split_end:
            continue
        latest = min(index + max_lead, split_end - 1)
        for slot in np.flatnonzero(valid[index]):
            occurrences = positions[int(target[index, slot])]
            position = bisect.bisect_left(occurrences, index + min_lead)
            if position < len(occurrences) and occurrences[position] <= latest:
                labels[index, slot] = 1.0
    return labels


class CandidateHead(nn.Module):
    def __init__(self, temporal_size):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(temporal_size + CANDIDATE_FEATURES, temporal_size),
            nn.Tanh(),
        )
        self.head = nn.Linear(temporal_size, 1)

    def forward(self, temporal, candidate):
        expanded = temporal.unsqueeze(2).expand(
            -1, -1, candidate.shape[2], -1
        )
        joined = torch.cat([expanded, candidate], dim=-1)
        return self.head(self.projection(joined)).squeeze(-1)


class CandidateGateLSTM(nn.Module):
    family = "lstm"

    def __init__(self, hidden_size):
        super().__init__()
        self.model_size = hidden_size
        self.lstm = nn.LSTM(
            RUNTIME_FEATURES, hidden_size, batch_first=True
        )
        self.candidate_head = CandidateHead(hidden_size)

    def forward(self, runtime, candidate, state=None):
        temporal, state = self.lstm(runtime, state)
        return self.candidate_head(temporal, candidate), state


class CausalSlidingCNN(nn.Module):
    """One short filter slides one event at a time, matching the professor sketch."""
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
        self.candidate_head = CandidateHead(channels)

    @property
    def receptive_field(self):
        return CNN_KERNEL_SIZE

    def forward(self, runtime, candidate):
        # Left-only padding makes output t depend on [t-2, t-1, t], never future rows.
        x = runtime.transpose(1, 2)
        temporal = torch.tanh(self.conv(F.pad(x, (CNN_KERNEL_SIZE - 1, 0))))
        return self.candidate_head(temporal.transpose(1, 2), candidate)


def expected_parameter_count(family, size):
    if family == "lstm":
        return 5 * size * size + 33 * size + 1
    if family == "cnn":
        return size * size + 21 * size + 1
    raise ValueError(family)


def build_model(family, size):
    model = CandidateGateLSTM(size) if family == "lstm" else CausalSlidingCNN(size)
    observed = sum(parameter.numel() for parameter in model.parameters())
    expected = expected_parameter_count(family, size)
    if observed != expected:
        raise RuntimeError(
            "parameter-count mismatch: {} != {}".format(observed, expected)
        )
    return model, observed


def iter_chunks(end, chunk_len):
    for start in range(0, end, chunk_len):
        yield start, min(end, start + chunk_len)


def optimizer_step(model, optimizer, loss_sum, valid_count):
    if loss_sum is None:
        return 0
    (loss_sum / float(valid_count)).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return 1


def train_lstm(
    model, runtime, candidate, labels, valid, fit_end, device,
    epochs, chunk_len, accumulate_chunks, learning_rate,
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    history = []
    model.to(device)
    chunks = list(iter_chunks(fit_end, chunk_len))
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        state = None
        group_loss = None
        group_valid = 0
        group_chunks = 0
        total_loss = 0.0
        total_valid = 0
        steps = 0
        for chunk_index, (start, stop) in enumerate(chunks):
            x = torch.from_numpy(runtime[start:stop]).unsqueeze(0).to(device)
            c = torch.from_numpy(candidate[start:stop]).unsqueeze(0).to(device)
            y = torch.from_numpy(labels[start:stop]).unsqueeze(0).to(device)
            mask = torch.from_numpy(valid[start:stop]).unsqueeze(0).to(device)
            logits, state = model(x, c, state)
            state = tuple(value.detach() for value in state)
            if bool(mask.any()):
                loss = F.binary_cross_entropy_with_logits(
                    logits[mask], y[mask], reduction="sum"
                )
                group_loss = loss if group_loss is None else group_loss + loss
                selected = int(mask.sum().item())
                group_valid += selected
                total_valid += selected
                total_loss += float(loss.detach().item())
            group_chunks += 1
            if group_chunks == accumulate_chunks or chunk_index + 1 == len(chunks):
                steps += optimizer_step(
                    model, optimizer, group_loss, group_valid
                )
                optimizer.zero_grad(set_to_none=True)
                group_loss = None
                group_valid = 0
                group_chunks = 0
        row = {
            "epoch": epoch,
            "loss": total_loss / max(1, total_valid),
            "valid_candidates": total_valid,
            "chronological_chunks": len(chunks),
            "optimizer_steps": steps,
        }
        history.append(row)
        print(
            "[train:lstm] epoch={epoch} loss={loss:.6f} "
            "valid={valid_candidates}".format(**row)
        )
    return history


def train_cnn(
    model, runtime, candidate, labels, valid, fit_end, device,
    epochs, chunk_len, accumulate_chunks, learning_rate,
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    history = []
    model.to(device)
    chunks = list(iter_chunks(fit_end, chunk_len))
    left_context = model.receptive_field - 1
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        group_loss = None
        group_valid = 0
        group_chunks = 0
        total_loss = 0.0
        total_valid = 0
        steps = 0
        for chunk_index, (start, stop) in enumerate(chunks):
            context_start = max(0, start - left_context)
            offset = start - context_start
            x = torch.from_numpy(runtime[context_start:stop]).unsqueeze(0).to(device)
            c = torch.from_numpy(candidate[context_start:stop]).unsqueeze(0).to(device)
            logits = model(x, c)[:, offset:]
            y = torch.from_numpy(labels[start:stop]).unsqueeze(0).to(device)
            mask = torch.from_numpy(valid[start:stop]).unsqueeze(0).to(device)
            if bool(mask.any()):
                loss = F.binary_cross_entropy_with_logits(
                    logits[mask], y[mask], reduction="sum"
                )
                group_loss = loss if group_loss is None else group_loss + loss
                selected = int(mask.sum().item())
                group_valid += selected
                total_valid += selected
                total_loss += float(loss.detach().item())
            group_chunks += 1
            if group_chunks == accumulate_chunks or chunk_index + 1 == len(chunks):
                steps += optimizer_step(
                    model, optimizer, group_loss, group_valid
                )
                optimizer.zero_grad(set_to_none=True)
                group_loss = None
                group_valid = 0
                group_chunks = 0
        row = {
            "epoch": epoch,
            "loss": total_loss / max(1, total_valid),
            "valid_candidates": total_valid,
            "chronological_chunks": len(chunks),
            "optimizer_steps": steps,
        }
        history.append(row)
        print(
            "[train:cnn] epoch={epoch} loss={loss:.6f} "
            "valid={valid_candidates}".format(**row)
        )
    return history


def train_model(model, family, *args):
    return (
        train_lstm(model, *args)
        if family == "lstm"
        else train_cnn(model, *args)
    )


def score_lstm(
    model, runtime, candidate, device, initial_state=None, chunk_len=8192
):
    model.eval()
    scores = np.zeros(candidate.shape[:2], dtype=np.float32)
    state = initial_state
    with torch.no_grad():
        for start, stop in iter_chunks(len(runtime), chunk_len):
            x = torch.from_numpy(runtime[start:stop]).unsqueeze(0).to(device)
            c = torch.from_numpy(candidate[start:stop]).unsqueeze(0).to(device)
            logits, state = model(x, c, state)
            state = tuple(value.detach() for value in state)
            scores[start:stop] = torch.sigmoid(logits[0]).cpu().numpy()
    return scores, state


def score_cnn(
    model, runtime, candidate, device,
    prefix_runtime=None, prefix_candidate=None, chunk_len=8192,
):
    model.eval()
    context = model.receptive_field - 1
    if prefix_runtime is None:
        prefix_count = 0
        all_runtime = runtime
        all_candidate = candidate
    else:
        if prefix_candidate is None or len(prefix_runtime) != len(prefix_candidate):
            raise RuntimeError("CNN prefix runtime/candidate mismatch")
        prefix_count = min(context, len(prefix_runtime))
        all_runtime = np.concatenate(
            [prefix_runtime[-prefix_count:], runtime], axis=0
        )
        all_candidate = np.concatenate(
            [prefix_candidate[-prefix_count:], candidate], axis=0
        )
    scores = np.zeros(candidate.shape[:2], dtype=np.float32)
    with torch.no_grad():
        for start, stop in iter_chunks(len(runtime), chunk_len):
            global_start = prefix_count + start
            global_stop = prefix_count + stop
            context_start = max(0, global_start - context)
            offset = global_start - context_start
            x = torch.from_numpy(
                all_runtime[context_start:global_stop]
            ).unsqueeze(0).to(device)
            c = torch.from_numpy(
                all_candidate[context_start:global_stop]
            ).unsqueeze(0).to(device)
            logits = model(x, c)[:, offset:]
            scores[start:stop] = torch.sigmoid(logits[0]).cpu().numpy()
    return scores


def self_test_cnn_causality():
    torch.manual_seed(123)
    model = CausalSlidingCNN(4).eval()
    runtime = torch.randn(1, 17, RUNTIME_FEATURES)
    candidate = torch.randn(1, 17, 5, CANDIDATE_FEATURES)
    pivot = 7
    with torch.no_grad():
        original = model(runtime, candidate)
        changed_runtime = runtime.clone()
        changed_candidate = candidate.clone()
        changed_runtime[:, pivot + 1:] += 1000.0
        changed_candidate[:, pivot + 1:] -= 1000.0
        changed = model(changed_runtime, changed_candidate)
    if not torch.allclose(
        original[:, :pivot + 1],
        changed[:, :pivot + 1],
        atol=1e-6,
        rtol=1e-6,
    ):
        raise RuntimeError("CNN future-input causality self-test failed")

    runtime_np = runtime[0].numpy().astype(np.float32)
    candidate_np = candidate[0].numpy().astype(np.float32)
    with torch.no_grad():
        full = torch.sigmoid(original[0]).numpy()
    chunked = score_cnn(
        model, runtime_np, candidate_np, torch.device("cpu"), chunk_len=4
    )
    if not np.allclose(full, chunked, atol=1e-6, rtol=1e-6):
        raise RuntimeError("CNN overlap/chunk equivalence self-test failed")


def metrics(selected, labels, valid):
    chosen = selected & valid
    issued = int(chosen.sum())
    useful = int((chosen & (labels > 0.5)).sum())
    positives = int(((labels > 0.5) & valid).sum())
    precision = useful / float(issued) if issued else 0.0
    recall = useful / float(positives) if positives else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return issued, useful, precision, recall, f1


def calibrate(scores, labels, valid, start, policy):
    finite = scores[start:][valid[start:]]
    if finite.size == 0:
        raise RuntimeError("calibration split has no {} candidates".format(policy))
    thresholds = [0.0, 1.0]
    thresholds.extend(np.linspace(0.05, 0.95, 19).tolist())
    thresholds.extend(
        float(np.quantile(finite, q))
        for q in (0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)
    )
    events = max(1, len(valid) - start)
    budget = float(valid[start:].sum()) / events
    rows = []
    for threshold in sorted(set(round(value, 8) for value in thresholds)):
        selected = scores[start:] >= threshold
        issued, useful, precision, recall, f1 = metrics(
            selected, labels[start:], valid[start:]
        )
        rate = issued / float(events)
        rows.append({
            "threshold": threshold,
            "issued": issued,
            "useful": useful,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "candidates_per_event": rate,
            "offline_{}_candidates_per_event_budget".format(policy): budget,
            "within_budget": int(rate <= budget + 1.0 / events),
        })
    eligible = [row for row in rows if row["within_budget"]]
    eligible.sort(
        key=lambda row: (
            -row["f1"], -row["precision"], -row["recall"],
            row["candidates_per_event"],
        )
    )
    return eligible[0], rows


def write_table(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_replay(path, rows, selected, valid, target):
    entries = 0
    triggers = 0
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pc", "line", "occ", "prefetch_addr"])
        for index, (pc, line, occurrence) in enumerate(rows):
            slots = np.flatnonzero(selected[index] & valid[index])
            if slots.size:
                triggers += 1
            for slot in slots:
                writer.writerow([
                    pc,
                    line,
                    occurrence,
                    "0x{:x}".format(int(target[index, slot]) * 64),
                ])
                entries += 1
    return entries, triggers


def model_tag(policy, family, size):
    suffix = "lstm_h{}".format(size) if family == "lstm" else "cnn_c{}".format(size)
    return "{}_{}".format(policy, suffix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, choices=sorted(POLICIES))
    for role in ("train", "guard", "eval"):
        parser.add_argument(
            "--{}-stream".format(role), required=True, type=Path
        )
        parser.add_argument(
            "--{}-candidates".format(role), required=True, type=Path
        )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-family", choices=["lstm", "cnn"], required=True)
    parser.add_argument("--model-size", type=int, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--chunk-len", type=int, default=1024)
    parser.add_argument("--accumulate-chunks", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--min-lead", type=int, default=4)
    parser.add_argument("--max-lead", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.min_lead < 1 or args.max_lead < args.min_lead:
        raise SystemExit("invalid lead window")
    if args.model_size <= 0:
        raise SystemExit("model size must be positive")

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
    stream_paths = {
        role: getattr(args, role + "_stream")
        for role in ("train", "guard", "eval")
    }
    candidate_paths = {
        role: getattr(args, role + "_candidates")
        for role in ("train", "guard", "eval")
    }
    rows = {role: load_stream(path) for role, path in stream_paths.items()}
    banks = {
        role: load_candidate_bank(
            candidate_paths[role], args.policy, rows[role]
        )
        for role in ("train", "guard", "eval")
    }
    max_candidates = max(
        len(candidates)
        for role in ("train", "guard", "eval")
        for candidates in banks[role]
    )
    if max_candidates <= 0:
        raise RuntimeError("candidate bank has no entries")

    train_runtime, _ = runtime_arrays(rows["train"])
    guard_runtime, guard_last_line = runtime_arrays(rows["guard"])
    eval_runtime, _ = runtime_arrays(
        rows["eval"], previous_line=guard_last_line
    )
    runtime = {
        "train": train_runtime,
        "guard": guard_runtime,
        "eval": eval_runtime,
    }
    candidate = {}
    valid = {}
    target = {}
    for role in ("train", "guard", "eval"):
        candidate[role], valid[role], target[role] = candidate_arrays(
            rows[role], banks[role], max_candidates
        )

    fit_end = int(len(rows["train"]) * 0.8)
    if fit_end <= 0 or fit_end >= len(rows["train"]):
        raise RuntimeError("training stream is too short for fit/calibration split")
    labels = training_labels(
        rows["train"],
        valid["train"],
        target["train"],
        args.min_lead,
        args.max_lead,
        fit_end,
    )

    model, parameter_count = build_model(
        args.model_family, args.model_size
    )
    history = train_model(
        model,
        args.model_family,
        runtime["train"],
        candidate["train"],
        labels,
        valid["train"],
        fit_end,
        device,
        args.epochs,
        args.chunk_len,
        args.accumulate_chunks,
        args.learning_rate,
    )

    if args.model_family == "lstm":
        train_scores, _ = score_lstm(
            model, runtime["train"], candidate["train"], device
        )
        _, guard_state = score_lstm(
            model, runtime["guard"], candidate["guard"], device
        )
        eval_scores, _ = score_lstm(
            model,
            runtime["eval"],
            candidate["eval"],
            device,
            initial_state=guard_state,
        )
    else:
        train_scores = score_cnn(
            model, runtime["train"], candidate["train"], device
        )
        eval_scores = score_cnn(
            model,
            runtime["eval"],
            candidate["eval"],
            device,
            prefix_runtime=runtime["guard"],
            prefix_candidate=candidate["guard"],
        )

    best, sweep = calibrate(
        train_scores, labels, valid["train"], fit_end, args.policy
    )
    threshold = best["threshold"]
    selected_eval = eval_scores >= threshold
    normal_selected = np.ones(valid["eval"].shape, dtype=np.bool_)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    normal_name = "offline_{}.replay.csv".format(args.policy)
    normal_entries, normal_triggers = write_replay(
        args.out_dir / normal_name,
        rows["eval"],
        normal_selected,
        valid["eval"],
        target["eval"],
    )
    nn_entries, nn_triggers = write_replay(
        args.out_dir / "offline_nn.replay.csv",
        rows["eval"],
        selected_eval,
        valid["eval"],
        target["eval"],
    )
    if nn_entries > normal_entries:
        raise RuntimeError("suppress-only gate emitted more than normal bank")
    write_table(args.out_dir / "policy_sweep.csv", sweep)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_family": args.model_family,
            "model_size": args.model_size,
            "policy": args.policy,
            "runtime_features": RUNTIME_FEATURES,
            "candidate_features": CANDIDATE_FEATURES,
            "max_candidates_per_event": max_candidates,
        },
        args.out_dir / "model.pt",
    )

    tag = model_tag(args.policy, args.model_family, args.model_size)
    metadata = {
        "trace": TRACE,
        "model_tag": tag,
        "matched_normal_prefetcher": args.policy,
        "model_family": args.model_family,
        "model_size": args.model_size,
        "architecture_pair_id": args.pair_id,
        "parameter_count": parameter_count,
        "seed": args.seed,
        "epochs": args.epochs,
        "chunk_len": args.chunk_len,
        "accumulate_chunks": args.accumulate_chunks,
        "learning_rate": args.learning_rate,
        "threshold": threshold,
        "fit_rows": fit_end,
        "calibration_rows": len(rows["train"]) - fit_end,
        "guard_rows": len(rows["guard"]),
        "eval_rows": len(rows["eval"]),
        "runtime_feature_count": RUNTIME_FEATURES,
        "runtime_feature_contract": [
            "current_page_offset",
            "signed_delta_from_previous_line",
            "log_absolute_delta_from_previous_line",
            "same_page_as_previous",
            "signed_page_delta_from_previous_line",
        ],
        "candidate_feature_contract": [
            "signed_candidate_line_delta",
            "candidate_page_offset",
            "candidate_rank",
        ],
        "model_does_not_use_pc": True,
        "pc_is_replay_transport_only": True,
        "forbidden_inputs": [
            "cycle", "hit_miss", "queue_state", "accepted",
            "duplicate", "future_eval_rows",
        ],
        "normal_candidate_bank_is_fixed": True,
        "normal_candidate_bank_source": "live_policy_PF_requests",
        "nn_can_only_suppress_normal_candidates": True,
        "training_chunks_shuffled": False,
        "training_labels_use_future_only_offline": True,
        "causal_no_future_self_test": "PASS",
        "experiment_revision": EXPERIMENT_REVISION,
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
        "cnn_receptive_field_events": (
            CNN_KERNEL_SIZE if args.model_family == "cnn" else 0
        ),
        "training_left_context_overlap": (
            CNN_KERNEL_SIZE - 1 if args.model_family == "cnn" else 0
        ),
        "max_candidates_per_event": max_candidates,
        "offline_normal_entries": normal_entries,
        "offline_normal_triggers": normal_triggers,
        "offline_nn_entries": nn_entries,
        "offline_nn_triggers": nn_triggers,
        "normal_list_sha256": sha256(args.out_dir / normal_name),
        "nn_list_sha256": sha256(args.out_dir / "offline_nn.replay.csv"),
        "train_history": history,
        "calibration_choice": best,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    for role in ("train", "guard", "eval"):
        metadata[role + "_stream_gzip_sha256"] = sha256(
            stream_paths[role]
        )
        metadata[role + "_stream_content_sha256"] = gzip_content_sha256(
            stream_paths[role]
        )
        metadata[role + "_candidate_gzip_sha256"] = sha256(
            candidate_paths[role]
        )
        metadata[role + "_candidate_content_sha256"] = gzip_content_sha256(
            candidate_paths[role]
        )
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "status": "PASS",
        "model_tag": tag,
        "policy": args.policy,
        "family": args.model_family,
        "parameters": parameter_count,
        "threshold": threshold,
        "offline_normal_entries": normal_entries,
        "offline_nn_entries": nn_entries,
    }, indent=2))


if __name__ == "__main__":
    main()
