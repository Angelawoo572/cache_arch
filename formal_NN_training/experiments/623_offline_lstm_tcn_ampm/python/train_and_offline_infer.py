#!/usr/bin/env python3
"""Train matched LSTM/causal-TCN AMPM gates for 623.xalancbmk_s-700B.

The normal policy, candidate bank, labels, calibration objective, raw inputs,
and keyed replay transport are fixed.  The only experimental variable is the
temporal model family.  Both neural models can suppress AMPM candidates but
cannot invent a candidate outside the pinned AMPM bank.
"""
import argparse
import bisect
import csv
import gzip
import hashlib
import json
import platform
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TRACE = "623.xalancbmk_s-700B"
TRACKERS = 64
PAGE_LINES = 64
PRED_DEGREE = 4
MAX_DELTA = 16
RUNTIME_FEATURES = PAGE_LINES + 2
CANDIDATE_FEATURES = 3
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = (1, 2, 4, 8, 16, 32)
EXPERIMENT_REVISION = "architecture_ablation_v1"


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
            rows.append((as_int(row["pc"]), as_int(row["line"]), as_int(row["pc_line_occ"])))
    if not rows:
        raise RuntimeError("empty stream: {}".format(path))
    return rows


def ampm_arrays(rows, state=None):
    """Mirror pinned AMPM: 64 LRU pages, 64-bit bitmap, degree 4."""
    count = len(rows)
    runtime = np.zeros((count, RUNTIME_FEATURES), dtype=np.float32)
    candidate = np.zeros((count, PRED_DEGREE, CANDIDATE_FEATURES), dtype=np.float32)
    valid = np.zeros((count, PRED_DEGREE), dtype=np.bool_)
    target = np.zeros((count, PRED_DEGREE), dtype=np.int64)
    pages = OrderedDict() if state is None else state

    for index, (unused_pc, line, unused_occ) in enumerate(rows):
        del unused_pc, unused_occ
        page = line // PAGE_LINES
        offset = line % PAGE_LINES
        was_tracked = page in pages
        if was_tracked:
            bitmap = pages.pop(page)
        else:
            if len(pages) >= TRACKERS:
                pages.popitem(last=False)
            bitmap = np.zeros(PAGE_LINES, dtype=np.bool_)
        bitmap[offset] = True
        pages[page] = bitmap

        runtime[index, :PAGE_LINES] = bitmap
        runtime[index, PAGE_LINES] = offset / float(PAGE_LINES - 1)
        runtime[index, PAGE_LINES + 1] = float(was_tracked)

        selected = []
        for delta in range(MAX_DELTA, 0, -1):
            if len(selected) >= PRED_DEGREE:
                break
            one_hop = offset - delta
            two_hop = offset - 2 * delta
            if one_hop >= 0 and two_hop >= 0 and bitmap[one_hop] and bitmap[two_hop]:
                target_offset = offset + delta
                if target_offset < PAGE_LINES:
                    selected.append((delta, target_offset))
        for delta in range(MAX_DELTA, 0, -1):
            if len(selected) >= PRED_DEGREE:
                break
            one_hop = offset + delta
            two_hop = offset + 2 * delta
            if one_hop < PAGE_LINES and two_hop < PAGE_LINES and bitmap[one_hop] and bitmap[two_hop]:
                target_offset = offset - delta
                if target_offset >= 0:
                    selected.append((-delta, target_offset))

        for slot, (signed_delta, target_offset) in enumerate(selected):
            valid[index, slot] = True
            target[index, slot] = page * PAGE_LINES + target_offset
            candidate[index, slot] = [
                signed_delta / float(MAX_DELTA),
                target_offset / float(PAGE_LINES - 1),
                (slot + 1) / float(PRED_DEGREE),
            ]
    return runtime, candidate, valid, target, pages


def self_test_ampm_policy():
    page = 11 * PAGE_LINES
    positive_rows = [(1, page + x, 0) for x in (1, 2, 3)]
    _, _, valid, target, _ = ampm_arrays(positive_rows)
    assert not bool(valid[0].any())
    assert not bool(valid[1].any())
    assert target[2][valid[2]].tolist() == [page + 4]

    negative_rows = [(1, page + x, 0) for x in (10, 9, 8)]
    _, _, valid, target, _ = ampm_arrays(negative_rows)
    assert target[2][valid[2]].tolist() == [page + 7]

    boundary_rows = [(1, page + x, 0) for x in (18, 34, 50)]
    _, _, valid, _, _ = ampm_arrays(boundary_rows)
    assert not bool(valid[2].any())

    _, _, _, _, state = ampm_arrays(positive_rows)
    _, _, valid, target, _ = ampm_arrays([(1, page + 4, 0)], state)
    assert target[0][valid[0]].tolist() == [page + 5]


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
        expanded = temporal.unsqueeze(2).expand(-1, -1, candidate.shape[2], -1)
        joined = torch.cat([expanded, candidate], dim=-1)
        return self.head(self.projection(joined)).squeeze(-1)


class AMPMGateLSTM(nn.Module):
    family = "lstm"

    def __init__(self, hidden_size):
        super().__init__()
        self.model_size = hidden_size
        self.lstm = nn.LSTM(RUNTIME_FEATURES, hidden_size, batch_first=True)
        self.candidate_head = CandidateHead(hidden_size)

    def forward(self, runtime, candidate, state=None):
        temporal, state = self.lstm(runtime, state)
        return self.candidate_head(temporal, candidate), state


class CausalResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.left_padding = (TCN_KERNEL_SIZE - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=TCN_KERNEL_SIZE,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x):
        update = self.conv(F.pad(x, (self.left_padding, 0)))
        return x + torch.tanh(update)


class AMPMGateTCN(nn.Module):
    family = "tcn"

    def __init__(self, channels):
        super().__init__()
        self.model_size = channels
        self.input_projection = nn.Linear(RUNTIME_FEATURES, channels)
        self.blocks = nn.ModuleList(
            CausalResidualBlock(channels, dilation) for dilation in TCN_DILATIONS
        )
        self.candidate_head = CandidateHead(channels)

    @property
    def receptive_field(self):
        return 1 + (TCN_KERNEL_SIZE - 1) * sum(TCN_DILATIONS)

    def forward(self, runtime, candidate):
        temporal = torch.tanh(self.input_projection(runtime)).transpose(1, 2)
        for block in self.blocks:
            temporal = block(temporal)
        temporal = temporal.transpose(1, 2)
        return self.candidate_head(temporal, candidate)


def expected_parameter_count(family, size):
    if family == "lstm":
        return 5 * size * size + 277 * size + 1
    if family == "tcn":
        return 19 * size * size + 78 * size + 1
    raise ValueError(family)


def build_model(family, size):
    model = AMPMGateLSTM(size) if family == "lstm" else AMPMGateTCN(size)
    observed = sum(parameter.numel() for parameter in model.parameters())
    expected = expected_parameter_count(family, size)
    if observed != expected:
        raise RuntimeError("parameter-count mismatch: {} != {}".format(observed, expected))
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


def train_lstm(model, runtime, candidate, labels, valid, fit_end, device, epochs, chunk_len, accumulate_chunks, learning_rate):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
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
                loss = F.binary_cross_entropy_with_logits(logits[mask], y[mask], reduction="sum")
                group_loss = loss if group_loss is None else group_loss + loss
                selected = int(mask.sum().item())
                group_valid += selected
                total_valid += selected
                total_loss += float(loss.detach().item())
            group_chunks += 1
            if group_chunks == accumulate_chunks or chunk_index + 1 == len(chunks):
                steps += optimizer_step(model, optimizer, group_loss, group_valid)
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
        print("[train:lstm] epoch={epoch} loss={loss:.6f} valid={valid_candidates}".format(**row))
    return history


def train_tcn(model, runtime, candidate, labels, valid, fit_end, device, epochs, chunk_len, accumulate_chunks, learning_rate):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
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
                loss = F.binary_cross_entropy_with_logits(logits[mask], y[mask], reduction="sum")
                group_loss = loss if group_loss is None else group_loss + loss
                selected = int(mask.sum().item())
                group_valid += selected
                total_valid += selected
                total_loss += float(loss.detach().item())
            group_chunks += 1
            if group_chunks == accumulate_chunks or chunk_index + 1 == len(chunks):
                steps += optimizer_step(model, optimizer, group_loss, group_valid)
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
        print("[train:tcn] epoch={epoch} loss={loss:.6f} valid={valid_candidates}".format(**row))
    return history


def train_model(model, family, *args):
    return train_lstm(model, *args) if family == "lstm" else train_tcn(model, *args)


def score_lstm(model, runtime, candidate, device, initial_state=None, chunk_len=8192):
    model.eval()
    scores = np.zeros((len(runtime), PRED_DEGREE), dtype=np.float32)
    state = initial_state
    with torch.no_grad():
        for start, stop in iter_chunks(len(runtime), chunk_len):
            x = torch.from_numpy(runtime[start:stop]).unsqueeze(0).to(device)
            c = torch.from_numpy(candidate[start:stop]).unsqueeze(0).to(device)
            logits, state = model(x, c, state)
            state = tuple(value.detach() for value in state)
            scores[start:stop] = torch.sigmoid(logits[0]).cpu().numpy()
    return scores, state


def score_tcn(model, runtime, candidate, device, prefix_runtime=None, prefix_candidate=None, chunk_len=8192):
    model.eval()
    context = model.receptive_field - 1
    if prefix_runtime is None:
        prefix_count = 0
        all_runtime = runtime
        all_candidate = candidate
    else:
        if prefix_candidate is None or len(prefix_runtime) != len(prefix_candidate):
            raise RuntimeError("TCN prefix runtime/candidate mismatch")
        prefix_count = min(context, len(prefix_runtime))
        all_runtime = np.concatenate([prefix_runtime[-prefix_count:], runtime], axis=0)
        all_candidate = np.concatenate([prefix_candidate[-prefix_count:], candidate], axis=0)
    scores = np.zeros((len(runtime), PRED_DEGREE), dtype=np.float32)
    with torch.no_grad():
        for start, stop in iter_chunks(len(runtime), chunk_len):
            global_start = prefix_count + start
            global_stop = prefix_count + stop
            context_start = max(0, global_start - context)
            offset = global_start - context_start
            x = torch.from_numpy(all_runtime[context_start:global_stop]).unsqueeze(0).to(device)
            c = torch.from_numpy(all_candidate[context_start:global_stop]).unsqueeze(0).to(device)
            logits = model(x, c)[:, offset:]
            scores[start:stop] = torch.sigmoid(logits[0]).cpu().numpy()
    return scores


def self_test_tcn_causality():
    torch.manual_seed(123)
    model = AMPMGateTCN(4).eval()
    runtime = torch.randn(1, 48, RUNTIME_FEATURES)
    candidate = torch.randn(1, 48, PRED_DEGREE, CANDIDATE_FEATURES)
    pivot = 19
    with torch.no_grad():
        original = model(runtime, candidate)
        changed_runtime = runtime.clone()
        changed_candidate = candidate.clone()
        changed_runtime[:, pivot + 1:] += 1000.0
        changed_candidate[:, pivot + 1:] -= 1000.0
        changed = model(changed_runtime, changed_candidate)
    if not torch.allclose(original[:, : pivot + 1], changed[:, : pivot + 1], atol=1e-6, rtol=1e-6):
        raise RuntimeError("TCN future-input causality self-test failed")

    runtime_np = runtime[0].numpy().astype(np.float32)
    candidate_np = candidate[0].numpy().astype(np.float32)
    with torch.no_grad():
        full = torch.sigmoid(original[0]).numpy()
    chunked = score_tcn(model, runtime_np, candidate_np, torch.device("cpu"), chunk_len=7)
    if not np.allclose(full, chunked, atol=1e-6, rtol=1e-6):
        raise RuntimeError("TCN overlap/chunk equivalence self-test failed")


def metrics(selected, labels, valid):
    chosen = selected & valid
    issued = int(chosen.sum())
    useful = int((chosen & (labels > 0.5)).sum())
    positives = int(((labels > 0.5) & valid).sum())
    precision = useful / float(issued) if issued else 0.0
    recall = useful / float(positives) if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return issued, useful, precision, recall, f1


def calibrate(scores, labels, valid, start):
    finite = scores[start:][valid[start:]]
    if finite.size == 0:
        raise RuntimeError("calibration split has no AMPM candidates")
    thresholds = [0.0, 1.0]
    thresholds.extend(np.linspace(0.05, 0.95, 19).tolist())
    thresholds.extend(float(np.quantile(finite, q)) for q in (0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99))
    events = max(1, len(valid) - start)
    budget = float(valid[start:].sum()) / events
    rows = []
    for threshold in sorted(set(round(value, 8) for value in thresholds)):
        selected = scores[start:] >= threshold
        issued, useful, precision, recall, f1 = metrics(selected, labels[start:], valid[start:])
        rate = issued / float(events)
        rows.append({
            "threshold": threshold,
            "issued": issued,
            "useful": useful,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "candidates_per_event": rate,
            "offline_ampm_candidates_per_event_budget": budget,
            "within_budget": int(rate <= budget + 1.0 / events),
        })
    eligible = [row for row in rows if row["within_budget"]]
    if not eligible:
        raise RuntimeError("no threshold satisfies the offline AMPM budget")
    eligible.sort(key=lambda row: (-row["f1"], -row["precision"], -row["recall"], row["candidates_per_event"]))
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
                writer.writerow([pc, line, occurrence, "0x{:x}".format(int(target[index, slot]) * 64)])
                entries += 1
    return entries, triggers


def model_tag(family, size):
    return "lstm_h{}".format(size) if family == "lstm" else "tcn_c{}".format(size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-stream", required=True, type=Path)
    parser.add_argument("--guard-stream", required=True, type=Path)
    parser.add_argument("--eval-stream", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-family", choices=["lstm", "tcn"], required=True)
    parser.add_argument("--model-size", type=int, required=True)
    parser.add_argument("--pair-id", default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--chunk-len", type=int, default=1024)
    parser.add_argument("--accumulate-chunks", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--min-lead", type=int, default=4)
    parser.add_argument("--max-lead", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    if args.model_size < 1 or args.chunk_len < 1 or args.accumulate_chunks < 1:
        raise RuntimeError("model size and chunk settings must be positive")

    self_test_ampm_policy()
    self_test_tcn_causality()
    if args.self_test_only:
        print("[PASS] AMPM mirror and causal-TCN self-tests")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = load_stream(args.train_stream)
    guard_rows = load_stream(args.guard_stream)
    eval_rows = load_stream(args.eval_stream)

    train_runtime, train_candidate, train_valid, train_target, _ = ampm_arrays(train_rows)
    fit_end = int(0.8 * len(train_rows))
    labels = training_labels(
        train_rows,
        train_valid,
        train_target,
        args.min_lead,
        args.max_lead,
        fit_end,
    )
    model, parameters = build_model(args.model_family, args.model_size)
    history = train_model(
        model,
        args.model_family,
        train_runtime,
        train_candidate,
        labels,
        train_valid,
        fit_end,
        device,
        args.epochs,
        args.chunk_len,
        args.accumulate_chunks,
        args.learning_rate,
    )

    if args.model_family == "lstm":
        train_scores, _ = score_lstm(model, train_runtime, train_candidate, device)
    else:
        train_scores = score_tcn(model, train_runtime, train_candidate, device)
    policy, sweep = calibrate(train_scores, labels, train_valid, fit_end)

    guard_runtime, guard_candidate, _, _, page_state = ampm_arrays(guard_rows)
    eval_runtime, eval_candidate, eval_valid, eval_target, _ = ampm_arrays(eval_rows, page_state)
    if args.model_family == "lstm":
        _, guard_state = score_lstm(model, guard_runtime, guard_candidate, device)
        eval_scores, _ = score_lstm(model, eval_runtime, eval_candidate, device, guard_state)
    else:
        eval_scores = score_tcn(
            model,
            eval_runtime,
            eval_candidate,
            device,
            prefix_runtime=guard_runtime,
            prefix_candidate=guard_candidate,
        )

    nn_selected = eval_scores >= policy["threshold"]
    ampm_selected = eval_valid.copy()
    ampm_entries, ampm_triggers = write_replay(
        args.out_dir / "offline_ampm.replay.csv",
        eval_rows,
        ampm_selected,
        eval_valid,
        eval_target,
    )
    nn_entries, nn_triggers = write_replay(
        args.out_dir / "offline_nn.replay.csv",
        eval_rows,
        nn_selected,
        eval_valid,
        eval_target,
    )
    tag = model_tag(args.model_family, args.model_size)
    checkpoint = {
        "state_dict": model.cpu().state_dict(),
        "model_family": args.model_family,
        "model_size": args.model_size,
        "model_tag": tag,
        "parameter_count": parameters,
        "trace": TRACE,
        "matched_normal_prefetcher": "ampm",
    }
    torch.save(checkpoint, args.out_dir / "model.pt")
    write_table(args.out_dir / "training_history.csv", history)
    write_table(args.out_dir / "policy_sweep.csv", sweep)

    receptive_field = model.receptive_field if args.model_family == "tcn" else None
    metadata = {
        "trace": TRACE,
        "model_family": args.model_family,
        "model_tag": tag,
        "model_size": args.model_size,
        "architecture_pair_id": args.pair_id,
        "parameter_count": parameters,
        "parameter_count_formula": "5*s^2 + 277*s + 1" if args.model_family == "lstm" else "19*s^2 + 78*s + 1",
        "matched_normal_prefetcher": "ampm",
        "normal_candidate_bank_is_fixed": True,
        "nn_can_only_suppress_ampm_candidates": True,
        "seed": args.seed,
        "ampm_pb_size": TRACKERS,
        "ampm_pred_degree": PRED_DEGREE,
        "ampm_pref_degree": PRED_DEGREE,
        "ampm_pref_buffer_enabled": False,
        "ampm_max_delta": MAX_DELTA,
        "shared_eval_raw_input": ["current_cache_line_address", "causal_prior_address_history"],
        "runtime_features_derived_from_shared_address_stream": ["64_page_offset_bitmap_after_current_access", "current_page_offset", "page_tracker_hit", "64_page_lru_tracker_state"],
        "candidate_features_derived_from_ampm_candidates": ["signed_delta_within_16_lines", "candidate_page_offset", "AMPM_candidate_rank_1_to_4"],
        "model_does_not_use_pc": True,
        "pc_line_occ_role": "replay_transport_identity_only",
        "forbidden_inputs": ["hit_miss", "cycle", "queue_state", "metadata", "future_evaluation_rows"],
        "training_labels": "future addresses only inside each chronological training/calibration split of the disjoint 0_to_20M training stream",
        "training_chunks_shuffled": False,
        "training_chunk_len": args.chunk_len,
        "optimizer_step_every_chunks": args.accumulate_chunks,
        "training_state_mode": "chronological_stateful_tbptt" if args.model_family == "lstm" else "finite_causal_left_context",
        "training_state_carried_across_chunks": args.model_family == "lstm",
        "training_state_detached_between_chunks": args.model_family == "lstm",
        "training_left_context_overlap": receptive_field - 1 if receptive_field else 0,
        "inference_state_mode": "guard_initialized_recurrent_state" if args.model_family == "lstm" else "guard_initialized_finite_left_context",
        "tcn_kernel_size": TCN_KERNEL_SIZE if args.model_family == "tcn" else None,
        "tcn_dilations": list(TCN_DILATIONS) if args.model_family == "tcn" else None,
        "tcn_receptive_field_events": receptive_field,
        "causal_no_future_self_test": "PASS",
        "experiment_revision": EXPERIMENT_REVISION,
        "guard_stream_role": "causal 20M_to_25M state/context initialization only; never fitted, calibrated, or exported",
        "evaluation_stream_role": "causal inference only; never used for fitting or threshold calibration",
        "transport": "same keyed PC-line-occ ListReplayer for offline AMPM, LSTM, and TCN",
        "train_stream_sha256": sha256(args.train_stream),
        "guard_stream_sha256": sha256(args.guard_stream),
        "eval_stream_sha256": sha256(args.eval_stream),
        "train_stream_content_sha256": gzip_content_sha256(args.train_stream),
        "guard_stream_content_sha256": gzip_content_sha256(args.guard_stream),
        "eval_stream_content_sha256": gzip_content_sha256(args.eval_stream),
        "train_rows": len(train_rows),
        "fit_rows": fit_end,
        "calibration_rows": len(train_rows) - fit_end,
        "guard_rows": len(guard_rows),
        "eval_rows": len(eval_rows),
        "threshold": policy["threshold"],
        "offline_ampm_entries": ampm_entries,
        "offline_ampm_triggers": ampm_triggers,
        "offline_nn_entries": nn_entries,
        "offline_nn_triggers": nn_triggers,
        "offline_ampm_list_sha256": sha256(args.out_dir / "offline_ampm.replay.csv"),
        "offline_nn_list_sha256": sha256(args.out_dir / "offline_nn.replay.csv"),
        "device": str(device),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "ampm_policy_self_test": "PASS",
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print("[ok] " + json.dumps({
        "device": str(device),
        "model_tag": tag,
        "parameters": parameters,
        "threshold": policy["threshold"],
        "ampm_entries": ampm_entries,
        "nn_entries": nn_entries,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
