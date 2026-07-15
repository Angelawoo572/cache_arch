#!/usr/bin/env python3
"""Independent direct-action LSTM used by the matched 602 experiments.

The normal policy and neural policy share only the effective external inputs
read by the audited ChampSim source.  The normal policy is mirrored solely to
build its baseline replay list and to define an equal request-rate budget.  Its
candidates and private tables are never tensors consumed by the LSTM.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import json
import math
import platform
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TRACE = "602.gcc_s-734B"
PAGE_LINES = 64
TRACKERS = 64
MAX_DELTA = 16
EXPERIMENT_REVISION = "direct_action_independent_v3"
POLICY_DEGREE = {"stride": 2, "streamer": 5, "ampm": 4}
POLICY_USES_PC = {"stride": True, "streamer": False, "ampm": False}
POLICY_ALLOW_SELF = {"stride": True, "streamer": False, "ampm": False}
ADDRESS_FEATURES = 9
STRIDE_PC_FEATURES = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gzip_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_int(value) -> int:
    text = str(value).strip()
    return int(text, 16) if text.lower().startswith("0x") else int(float(text))


def load_stream(path: Path):
    rows = []
    occurrences = defaultdict(int)
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"trace", "demand_idx", "pc", "line", "pc_line_occ"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError("{} missing {}".format(path, sorted(missing)))
        for index, row in enumerate(reader):
            pc = as_int(row["pc"])
            line = as_int(row["line"])
            occurrence = as_int(row["pc_line_occ"])
            expected_occurrence = occurrences[(pc, line)]
            occurrences[(pc, line)] += 1
            if (
                row["trace"] != TRACE
                or as_int(row["demand_idx"]) != index
                or occurrence != expected_occurrence
            ):
                raise RuntimeError("stream identity/ordering failure at row {}".format(index))
            rows.append((pc, line, occurrence))
    if not rows:
        raise RuntimeError("empty stream: {}".format(path))
    return rows


def _chunks16(value: int):
    return [((value >> shift) & 0xFFFF) / 65535.0 for shift in (0, 16, 32, 48)]


def runtime_features(policy: str, rows, previous_line=None, previous_pc=None):
    feature_count = ADDRESS_FEATURES + (STRIDE_PC_FEATURES if POLICY_USES_PC[policy] else 0)
    runtime = np.zeros((len(rows), feature_count), dtype=np.float32)
    prior_line = previous_line
    prior_pc = previous_pc
    log_scale = math.log1p(4096.0)
    for index, (pc, line, unused_occurrence) in enumerate(rows):
        del unused_occurrence
        page = line // PAGE_LINES
        offset = line % PAGE_LINES
        if prior_line is None:
            delta = 0
            page_delta = 0
            same_page = 0.0
        else:
            delta = line - prior_line
            page_delta = page - prior_line // PAGE_LINES
            same_page = float(page == prior_line // PAGE_LINES)
        values = [
            offset / float(PAGE_LINES - 1),
            *_chunks16(page),
            np.clip(delta, -256, 256) / 256.0,
            min(1.0, math.log1p(abs(delta)) / log_scale),
            same_page,
            np.clip(page_delta, -64, 64) / 64.0,
        ]
        if POLICY_USES_PC[policy]:
            values.extend([*_chunks16(pc), float(prior_pc is not None and pc == prior_pc)])
        runtime[index] = values
        prior_line = line
        prior_pc = pc
    return runtime, prior_line, prior_pc


def future_use_labels(rows, min_lead: int, max_lead: int):
    """Label every same-page target independently from future demand reuse."""
    if min_lead < 1 or max_lead < min_lead:
        raise RuntimeError("invalid future-use lead window")
    positions = defaultdict(list)
    for index, (_, line, _) in enumerate(rows):
        positions[line].append(index)
    labels = np.zeros((len(rows), PAGE_LINES), dtype=np.uint8)
    first_lead = np.zeros((len(rows), PAGE_LINES), dtype=np.uint16)
    for index, (_, line, _) in enumerate(rows):
        page_base = (line // PAGE_LINES) * PAGE_LINES
        lower = index + min_lead
        upper = index + max_lead
        for offset in range(PAGE_LINES):
            values = positions.get(page_base + offset)
            if not values:
                continue
            at = bisect.bisect_left(values, lower)
            if at < len(values) and values[at] <= upper:
                labels[index, offset] = 1
                first_lead[index, offset] = values[at] - index
    return labels, first_lead


def stride_actions(rows, state=None):
    trackers = OrderedDict() if state is None else state
    actions = [[] for _ in rows]
    for index, (pc, line, _) in enumerate(rows):
        if pc not in trackers:
            if len(trackers) >= TRACKERS:
                trackers.popitem(last=True)
            trackers[pc] = (line, 0)
            trackers.move_to_end(pc, last=False)
            continue
        last_line, last_stride = trackers[pc]
        stride = line - last_line
        if stride == 0:
            # Pinned stride.cc returns before updating state or LRU recency.
            continue
        if stride == last_stride:
            page = line // PAGE_LINES
            offset = line % PAGE_LINES
            for degree in range(POLICY_DEGREE["stride"]):
                target_offset = offset + stride * degree
                if target_offset < 0 or target_offset >= PAGE_LINES:
                    break
                actions[index].append(page * PAGE_LINES + target_offset)
        trackers[pc] = (line, stride)
        trackers.move_to_end(pc, last=False)
    return actions, trackers


def streamer_actions(rows, state=None):
    trackers = OrderedDict() if state is None else state
    actions = [[] for _ in rows]
    for index, (_, line, _) in enumerate(rows):
        page = line // PAGE_LINES
        offset = line % PAGE_LINES
        if page not in trackers:
            if len(trackers) >= TRACKERS:
                trackers.popitem(last=False)
            trackers[page] = (offset, 0)
            continue
        last_offset, last_direction = trackers[page]
        if offset == last_offset:
            # Pinned streamer.cc returns before updating state or recency.
            continue
        direction = 1 if offset > last_offset else -1
        direction_match = direction == last_direction
        trackers.pop(page)
        trackers[page] = (offset, direction)
        if direction_match:
            for distance in range(1, POLICY_DEGREE["streamer"] + 1):
                target_offset = offset + direction * distance
                if target_offset < 0 or target_offset >= PAGE_LINES:
                    break
                actions[index].append(page * PAGE_LINES + target_offset)
    return actions, trackers


def ampm_actions(rows, state=None):
    pages = OrderedDict() if state is None else state
    actions = [[] for _ in rows]
    for index, (_, line, _) in enumerate(rows):
        page = line // PAGE_LINES
        offset = line % PAGE_LINES
        if page in pages:
            bitmap = pages.pop(page)
        else:
            if len(pages) >= TRACKERS:
                pages.popitem(last=False)
            bitmap = np.zeros(PAGE_LINES, dtype=np.bool_)
        bitmap[offset] = True
        pages[page] = bitmap
        selected = []
        for delta in range(MAX_DELTA, 0, -1):
            one_hop = offset - delta
            two_hop = offset - 2 * delta
            if one_hop >= 0 and two_hop >= 0 and bitmap[one_hop] and bitmap[two_hop]:
                target_offset = offset + delta
                if target_offset < PAGE_LINES:
                    selected.append(target_offset)
            if len(selected) >= POLICY_DEGREE["ampm"]:
                break
        if len(selected) < POLICY_DEGREE["ampm"]:
            for delta in range(MAX_DELTA, 0, -1):
                one_hop = offset + delta
                two_hop = offset + 2 * delta
                if one_hop < PAGE_LINES and two_hop < PAGE_LINES and bitmap[one_hop] and bitmap[two_hop]:
                    target_offset = offset - delta
                    if target_offset >= 0:
                        selected.append(target_offset)
                if len(selected) >= POLICY_DEGREE["ampm"]:
                    break
        actions[index] = [page * PAGE_LINES + target_offset for target_offset in selected]
    return actions, pages


def normal_actions(policy: str, rows, state=None):
    return {"stride": stride_actions, "streamer": streamer_actions, "ampm": ampm_actions}[policy](rows, state)


def policy_self_test():
    base = 7 * PAGE_LINES
    stride_rows = [(1, base + 10, 0), (1, base + 12, 0), (1, base + 14, 0)]
    actions, _ = stride_actions(stride_rows)
    assert actions[2] == [base + 14, base + 16]
    stream_rows = [(1, base + 10, 0), (2, base + 12, 0), (3, base + 13, 0)]
    actions, _ = streamer_actions(stream_rows)
    assert actions[2] == [base + x for x in (14, 15, 16, 17, 18)]
    ampm_rows = [(1, base + 0, 0), (2, base + 2, 0), (3, base + 4, 0)]
    actions, _ = ampm_actions(ampm_rows)
    assert base + 6 in actions[2]


class DirectActionLSTM(nn.Module):
    def __init__(self, feature_count: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(feature_count, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, PAGE_LINES)

    def forward(self, runtime, state=None):
        temporal, state = self.lstm(runtime, state)
        return self.head(temporal), state


def detach_state(state):
    if state is None:
        return None
    return tuple(value.detach() for value in state)


def train_model(model, runtime, labels, fit_end, device, epochs, chunk_len, accumulate_chunks, learning_rate):
    train_end = fit_end
    if train_end < chunk_len:
        raise RuntimeError("training split is shorter than one chunk")
    x = torch.from_numpy(runtime)
    y = torch.from_numpy(labels.astype(np.float32))
    positives = float(labels[:train_end].sum())
    negatives = float(train_end * PAGE_LINES - positives)
    positive_weight = min(64.0, max(1.0, negatives / max(1.0, positives)))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.full((PAGE_LINES,), positive_weight, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    model.to(device)
    history = []
    chunks = [
        (start, min(train_end, start + chunk_len))
        for start in range(0, train_end, chunk_len)
    ]
    for epoch in range(epochs):
        model.train()
        state = None
        total_loss = 0.0
        total_rows = 0
        for group_start in range(0, len(chunks), accumulate_chunks):
            group = chunks[group_start:group_start + accumulate_chunks]
            group_rows = sum(stop - start for start, stop in group)
            optimizer.zero_grad(set_to_none=True)
            for start, stop in group:
                xb = x[start:stop].unsqueeze(0).to(device)
                yb = y[start:stop].unsqueeze(0).to(device)
                logits, state = model(xb, state)
                state = detach_state(state)
                loss = criterion(logits, yb)
                # Backpropagate immediately so only one chunk graph is live;
                # row weighting makes a short final chunk/group exact.
                (loss * ((stop - start) / float(group_rows))).backward()
                total_loss += float(loss.detach().cpu()) * (stop - start)
                total_rows += stop - start
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        row = {"epoch": epoch + 1, "loss": total_loss / max(1, total_rows), "rows": total_rows}
        history.append(row)
        print("[train] epoch={epoch} loss={loss:.6f} rows={rows}".format(**row))
    return history


def score_continuous(model, runtime, device, initial_state=None, chunk_len=8192):
    model.eval()
    scores = np.zeros((len(runtime), PAGE_LINES), dtype=np.float32)
    state = initial_state
    with torch.no_grad():
        for start in range(0, len(runtime), chunk_len):
            stop = min(len(runtime), start + chunk_len)
            x = torch.from_numpy(runtime[start:stop]).unsqueeze(0).to(device)
            logits, state = model(x, state)
            scores[start:stop] = torch.sigmoid(logits[0]).cpu().numpy()
            state = detach_state(state)
    return scores, state


def decode(scores, rows, threshold, degree, allow_self):
    selected = np.zeros(scores.shape, dtype=np.bool_)
    for index, (_, line, _) in enumerate(rows):
        eligible = np.flatnonzero(scores[index] >= threshold)
        if not allow_self:
            eligible = eligible[eligible != line % PAGE_LINES]
        if eligible.size:
            order = np.argsort(-scores[index, eligible], kind="stable")
            selected[index, eligible[order[:degree]]] = True
    return selected


def selection_stats(selected, labels):
    tp = int(np.logical_and(selected, labels != 0).sum())
    fp = int(np.logical_and(selected, labels == 0).sum())
    fn = int(np.logical_and(~selected, labels != 0).sum())
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def calibrate(policy, scores, labels, rows, normal, start):
    degree = POLICY_DEGREE[policy]
    allow_self = POLICY_ALLOW_SELF[policy]
    normal_rate = sum(len(items) for items in normal[start:]) / float(max(1, len(rows) - start))
    finite = scores[start:].reshape(-1)
    thresholds = [0.0, 1.0]
    thresholds.extend(np.linspace(0.05, 0.95, 19).tolist())
    thresholds.extend(float(np.quantile(finite, q)) for q in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995))
    sweep = []
    for threshold in sorted(set(round(value, 8) for value in thresholds)):
        selected = decode(scores[start:], rows[start:], threshold, degree, allow_self)
        stats = selection_stats(selected, labels[start:])
        rate = float(selected.sum()) / float(max(1, len(rows) - start))
        stats.update({"threshold": threshold, "actions_per_event": rate, "normal_actions_per_event_budget": normal_rate})
        sweep.append(stats)
    eligible = [row for row in sweep if row["actions_per_event"] <= normal_rate + 1e-12]
    if not eligible:
        raise RuntimeError("no calibration threshold satisfies normal-policy action budget")
    eligible.sort(key=lambda row: (-row["f1"], -row["recall"], -row["precision"], row["actions_per_event"], -row["threshold"]))
    return eligible[0], sweep


def write_table(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_normal_replay(path, rows, actions):
    entries = 0
    triggers = 0
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pc", "line", "occ", "prefetch_addr"])
        for (pc, line, occurrence), targets in zip(rows, actions):
            if targets:
                triggers += 1
            for target in targets:
                writer.writerow([pc, line, occurrence, "0x{:x}".format(int(target) * 64)])
                entries += 1
    return entries, triggers


def write_nn_replay(path, rows, selected, scores):
    entries = 0
    triggers = 0
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pc", "line", "occ", "prefetch_addr"])
        for index, (pc, line, occurrence) in enumerate(rows):
            offsets = np.flatnonzero(selected[index])
            if offsets.size:
                triggers += 1
                offsets = offsets[np.argsort(-scores[index, offsets], kind="stable")]
            page_base = (line // PAGE_LINES) * PAGE_LINES
            for offset in offsets:
                writer.writerow([pc, line, occurrence, "0x{:x}".format((page_base + int(offset)) * 64)])
                entries += 1
    return entries, triggers


def build_parser(policy):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-stream", required=True, type=Path)
    if policy == "ampm":
        parser.add_argument("--guard-stream", required=True, type=Path)
    parser.add_argument("--eval-stream", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--chunk-len", type=int, default=1024)
    parser.add_argument("--batch-chunks", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--min-lead", type=int, default=4)
    parser.add_argument("--max-lead", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-size", type=int, default=8)
    return parser


def run_cli(policy: str):
    if policy not in POLICY_DEGREE:
        raise RuntimeError("unsupported policy {}".format(policy))
    args = build_parser(policy).parse_args()
    if args.hidden_size < 1 or args.chunk_len < 1 or args.batch_chunks < 1:
        raise RuntimeError("model/chunk sizes must be positive")
    policy_self_test()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_stream(args.train_stream)
    eval_rows = load_stream(args.eval_stream)
    guard_rows = load_stream(args.guard_stream) if policy == "ampm" else []
    train_runtime, _, _ = runtime_features(policy, train_rows)
    train_labels, _ = future_use_labels(train_rows, args.min_lead, args.max_lead)
    if not POLICY_ALLOW_SELF[policy]:
        for index, (_, line, _) in enumerate(train_rows):
            train_labels[index, line % PAGE_LINES] = 0
    train_normal, _ = normal_actions(policy, train_rows)
    fit_end = int(0.8 * len(train_rows))
    train_end = fit_end - args.max_lead
    if train_end <= 0:
        raise RuntimeError("training stream too short for leakage-free fit/calibration split")

    feature_count = train_runtime.shape[1]
    model = DirectActionLSTM(feature_count, args.hidden_size)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    history = train_model(
        model, train_runtime, train_labels, train_end, device, args.epochs,
        args.chunk_len, args.batch_chunks, args.learning_rate,
    )
    train_scores, _ = score_continuous(model, train_runtime, device)
    calibration, sweep = calibrate(policy, train_scores, train_labels, train_rows, train_normal, fit_end)

    normal_state = None
    prior_line = None
    prior_pc = None
    nn_state = None
    if guard_rows:
        guard_runtime, prior_line, prior_pc = runtime_features(policy, guard_rows)
        _, nn_state = score_continuous(model, guard_runtime, device)
        _, normal_state = normal_actions(policy, guard_rows)
    eval_runtime, _, _ = runtime_features(policy, eval_rows, prior_line, prior_pc)
    eval_scores, _ = score_continuous(model, eval_runtime, device, nn_state)
    eval_normal, _ = normal_actions(policy, eval_rows, normal_state)
    selected_eval = decode(
        eval_scores, eval_rows, calibration["threshold"],
        POLICY_DEGREE[policy], POLICY_ALLOW_SELF[policy],
    )

    normal_path = args.out_dir / ("offline_{}.replay.csv".format(policy))
    nn_path = args.out_dir / "offline_lstm.replay.csv"
    normal_entries, normal_triggers = write_normal_replay(normal_path, eval_rows, eval_normal)
    nn_entries, nn_triggers = write_nn_replay(nn_path, eval_rows, selected_eval, eval_scores)
    if nn_entries > POLICY_DEGREE[policy] * len(eval_rows):
        raise RuntimeError("direct neural action degree bound violated")

    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "parameters": parameters,
            "hidden_size": args.hidden_size,
            "feature_count": feature_count,
            "trace": TRACE,
            "matched_normal_prefetcher": policy,
            "neural_role": "standalone_direct_action_prefetcher",
        },
        args.out_dir / "model.pt",
    )
    write_table(args.out_dir / "training_history.csv", history)
    write_table(args.out_dir / "policy_sweep.csv", sweep)

    paths = {"train": args.train_stream, "eval": args.eval_stream}
    if guard_rows:
        paths["guard"] = args.guard_stream
    metadata = {
        "trace": TRACE,
        "matched_normal_prefetcher": policy,
        "model_family": "LSTM direct-action",
        "neural_role": "standalone_direct_action_prefetcher",
        "parameter_count": parameters,
        "hidden_size": args.hidden_size,
        "runtime_feature_count": feature_count,
        "seed": args.seed,
        "same_external_input_contract": True,
        "effective_external_inputs": ["pc", "cache_line_address"] if POLICY_USES_PC[policy] else ["cache_line_address"],
        "model_does_not_use_pc": not POLICY_USES_PC[policy],
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "nn_generates_own_target_addresses": True,
        "action_space": "64 same-page cache-line offsets",
        "action_degree_cap": POLICY_DEGREE[policy],
        "self_target_legal": POLICY_ALLOW_SELF[policy],
        "training_labels": "future demand reuse in the training stream; no normal-policy candidate or action labels",
        "forbidden_inputs": ["normal_candidates", "normal_private_tables", "hit_miss", "cycle", "queue_state", "future_evaluation_rows"],
        "training_state_mode": "chronological_stateful_tbptt",
        "training_chunks_shuffled": False,
        "training_state_carried_across_chunks": True,
        "training_state_detached_between_chunks": True,
        "training_state_reset": "only_at_epoch_start",
        "training_chunk_len": args.chunk_len,
        "optimizer_step_every_chunks": args.batch_chunks,
        "inference_state_mode": "guard_then_continuous_evaluation" if guard_rows else "continuous_within_independent_evaluation_stream",
        "experiment_revision": EXPERIMENT_REVISION,
        "evaluation_stream_role": "causal inference only; never used for fitting or threshold calibration",
        "calibration_split_start": fit_end,
        "leakage_free_training_end": train_end,
        "min_lead": args.min_lead,
        "max_lead": args.max_lead,
        "threshold": calibration["threshold"],
        "calibration_choice": calibration,
        "offline_{}_entries".format(policy): normal_entries,
        "offline_{}_triggers".format(policy): normal_triggers,
        "offline_lstm_entries": nn_entries,
        "offline_lstm_triggers": nn_triggers,
        "offline_{}_list_sha256".format(policy): sha256(normal_path),
        "offline_lstm_list_sha256": sha256(nn_path),
        "train_rows": len(train_rows),
        "guard_rows": len(guard_rows),
        "eval_rows": len(eval_rows),
        "device": str(device),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    for role, path in paths.items():
        metadata[role + "_stream_sha256"] = sha256(path)
        metadata[role + "_stream_content_sha256"] = gzip_content_sha256(path)
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print("[ok] " + json.dumps({
        "policy": policy,
        "device": str(device),
        "hidden_size": args.hidden_size,
        "parameters": parameters,
        "threshold": calibration["threshold"],
        "normal_entries": normal_entries,
        "nn_entries": nn_entries,
    }, sort_keys=True))


__all__ = [
    "DirectActionLSTM", "ampm_actions", "future_use_labels", "normal_actions",
    "run_cli", "runtime_features", "streamer_actions", "stride_actions",
]
