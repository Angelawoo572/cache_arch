#!/usr/bin/env python3
"""Train a 602 address-only LSTM gate and export offline streamer/LSTM lists.

The matched normal policy and the LSTM consume the same causal L2 LOAD address
stream. PC is retained only as a replay transport key and is never a numerical
model input. Neither policy reads hit/miss, cycle, queue state, metadata, or
future evaluation rows.
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


TRACE = "602.gcc_s-734B"
TRACKERS = 64
PAGE_LINES = 64
DEGREE = 5
RUNTIME_FEATURES = 4
CANDIDATE_FEATURES = 2


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
            rows.append(
                (
                    as_int(row["pc"]),
                    as_int(row["line"]),
                    as_int(row["pc_line_occ"]),
                )
            )
    if not rows:
        raise RuntimeError("empty stream: {}".format(path))
    return rows


def streamer_arrays(rows):
    """Mirror streamer.cc: 64 LRU page trackers and degree-5 same-page targets."""
    count = len(rows)
    runtime = np.zeros((count, RUNTIME_FEATURES), dtype=np.float32)
    candidate = np.zeros((count, DEGREE, CANDIDATE_FEATURES), dtype=np.float32)
    valid = np.zeros((count, DEGREE), dtype=np.bool_)
    target = np.zeros((count, DEGREE), dtype=np.int64)
    trackers = OrderedDict()

    for index, (unused_pc, line, unused_occ) in enumerate(rows):
        del unused_pc, unused_occ
        page = line // PAGE_LINES
        offset = line % PAGE_LINES
        if page not in trackers:
            runtime[index] = [offset / 63.0, 0.0, 0.0, 0.0]
            if len(trackers) >= TRACKERS:
                trackers.popitem(last=False)
            trackers[page] = (offset, 0)
            continue

        last_offset, last_direction = trackers[page]
        if offset == last_offset:
            runtime[index] = [offset / 63.0, 0.0, float(last_direction), 1.0]
            continue

        direction = 1 if offset > last_offset else -1
        direction_match = direction == last_direction
        runtime[index] = [
            offset / 63.0,
            float(direction),
            float(last_direction),
            1.0,
        ]

        if direction_match:
            for slot in range(DEGREE):
                distance = slot + 1
                target_offset = offset + direction * distance
                if target_offset < 0 or target_offset >= PAGE_LINES:
                    break
                valid[index, slot] = True
                target[index, slot] = page * PAGE_LINES + target_offset
                candidate[index, slot] = [
                    distance / float(DEGREE),
                    target_offset / 63.0,
                ]

        trackers.pop(page)
        trackers[page] = (offset, direction)

    return runtime, candidate, valid, target


def self_test_streamer_policy():
    base = 7 * PAGE_LINES
    rows = [
        (1, base + 10, 0),
        (2, base + 12, 0),
        (3, base + 13, 0),
        (4, base + 13, 0),
        (5, base + 11, 0),
        (6, base + 9, 0),
    ]
    _, _, valid, target = streamer_arrays(rows)
    assert not bool(valid[0].any())
    assert not bool(valid[1].any())
    assert target[2][valid[2]].tolist() == [base + x for x in (14, 15, 16, 17, 18)]
    assert not bool(valid[3].any())
    assert not bool(valid[4].any())
    assert target[5][valid[5]].tolist() == [base + x for x in (8, 7, 6, 5, 4)]


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
            values = positions[int(target[index, slot])]
            pos = bisect.bisect_left(values, index + min_lead)
            if pos < len(values) and values[pos] <= latest:
                labels[index, slot] = 1.0
    return labels


class StreamerGateLSTM(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(RUNTIME_FEATURES, hidden_size, batch_first=True)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size + CANDIDATE_FEATURES, hidden_size),
            nn.Tanh(),
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, runtime, candidate, state=None):
        hidden, state = self.lstm(runtime, state)
        expanded = hidden.unsqueeze(2).expand(-1, -1, candidate.shape[2], -1)
        joined = torch.cat([expanded, candidate], dim=-1)
        logits = self.head(self.projection(joined)).squeeze(-1)
        return logits, state


def parameter_count(hidden_size):
    return 5 * hidden_size * hidden_size + 28 * hidden_size + 1


def chunk_tensors(runtime, candidate, labels, valid, end, chunk_len):
    usable = (end // chunk_len) * chunk_len
    count = usable // chunk_len
    if count == 0:
        raise RuntimeError("training stream is shorter than one chunk")
    return (
        torch.from_numpy(runtime[:usable]).reshape(count, chunk_len, RUNTIME_FEATURES),
        torch.from_numpy(candidate[:usable]).reshape(
            count, chunk_len, DEGREE, CANDIDATE_FEATURES
        ),
        torch.from_numpy(labels[:usable]).reshape(count, chunk_len, DEGREE),
        torch.from_numpy(valid[:usable]).reshape(count, chunk_len, DEGREE),
    )


def train_model(
    model,
    runtime,
    candidate,
    labels,
    valid,
    fit_end,
    device,
    epochs,
    chunk_len,
    batch_chunks,
    learning_rate,
):
    x, c, y, mask = chunk_tensors(
        runtime, candidate, labels, valid, fit_end, chunk_len
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    history = []
    model.to(device)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_valid = 0
        optimizer_steps = 0
        model.train()

        # Stateful truncated BPTT. Chunks stay chronological and recurrent
        # values cross every chunk boundary. detach() truncates gradients only;
        # it does not clear the LSTM hidden or cell state.
        state = None
        group_loss_sum = None
        group_valid = 0
        group_chunks = 0
        optimizer.zero_grad(set_to_none=True)
        for chunk_index in range(len(x)):
            xb = x[chunk_index : chunk_index + 1].to(device)
            cb = c[chunk_index : chunk_index + 1].to(device)
            yb = y[chunk_index : chunk_index + 1].to(device)
            mb = mask[chunk_index : chunk_index + 1].to(device)
            logits, state = model(xb, cb, state)
            state = tuple(value.detach() for value in state)
            if bool(mb.any()):
                loss_sum = F.binary_cross_entropy_with_logits(
                    logits[mb], yb[mb], reduction="sum"
                )
                group_loss_sum = (
                    loss_sum if group_loss_sum is None else group_loss_sum + loss_sum
                )
                selected = int(mb.sum().item())
                group_valid += selected
                total_valid += selected
                total_loss += float(loss_sum.detach().item())
            group_chunks += 1

            if group_chunks == batch_chunks or chunk_index + 1 == len(x):
                if group_loss_sum is not None:
                    (group_loss_sum / float(group_valid)).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer_steps += 1
                optimizer.zero_grad(set_to_none=True)
                group_loss_sum = None
                group_valid = 0
                group_chunks = 0

        row = {
            "epoch": epoch,
            "loss": total_loss / max(1, total_valid),
            "valid_candidates": total_valid,
            "chronological_chunks": len(x),
            "optimizer_steps": optimizer_steps,
        }
        history.append(row)
        print(
            "[train] epoch={epoch} loss={loss:.6f} valid={valid_candidates}".format(
                **row
            )
        )
    return history


def score_continuous(model, runtime, candidate, device, chunk_len=8192):
    model.eval()
    scores = np.zeros((len(runtime), DEGREE), dtype=np.float32)
    state = None
    with torch.no_grad():
        for start in range(0, len(runtime), chunk_len):
            stop = min(len(runtime), start + chunk_len)
            x = torch.from_numpy(runtime[start:stop]).unsqueeze(0).to(device)
            c = torch.from_numpy(candidate[start:stop]).unsqueeze(0).to(device)
            logits, state = model(x, c, state)
            state = tuple(value.detach() for value in state)
            scores[start:stop] = torch.sigmoid(logits[0]).cpu().numpy()
    return scores


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


def calibrate(scores, labels, valid, start):
    finite = scores[start:][valid[start:]]
    if finite.size == 0:
        raise RuntimeError("calibration split has no streamer candidates")

    thresholds = list(np.linspace(0.05, 0.95, 19))
    thresholds += [
        float(np.quantile(finite, q))
        for q in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)
    ]
    events = max(1, len(valid) - start)
    budget = float(valid[start:].sum()) / events
    rows = []

    for threshold in sorted(set(round(value, 8) for value in thresholds)):
        selected = scores[start:] >= threshold
        issued, useful, precision, recall, f1 = metrics(
            selected, labels[start:], valid[start:]
        )
        rate = issued / float(events)
        rows.append(
            {
                "threshold": threshold,
                "issued": issued,
                "useful": useful,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "candidates_per_event": rate,
                "offline_streamer_candidates_per_event_budget": budget,
                "within_budget": int(rate <= budget + 1.0 / events),
            }
        )

    eligible = [row for row in rows if row["within_budget"]]
    if not eligible:
        raise RuntimeError("no threshold satisfies the offline streamer budget")
    eligible.sort(
        key=lambda row: (
            -row["f1"],
            -row["precision"],
            -row["recall"],
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
        for index, (pc, line, occ) in enumerate(rows):
            slots = np.flatnonzero(selected[index] & valid[index])
            if slots.size:
                triggers += 1
            for slot in slots:
                writer.writerow(
                    [
                        pc,
                        line,
                        occ,
                        "0x{:x}".format(int(target[index, slot]) * 64),
                    ]
                )
                entries += 1
    return entries, triggers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-stream", required=True, type=Path)
    parser.add_argument("--eval-stream", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--chunk-len", type=int, default=1024)
    parser.add_argument("--batch-chunks", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--min-lead", type=int, default=4)
    parser.add_argument("--max-lead", type=int, default=64)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    parser.add_argument("--hidden-size", type=int, default=16)
    args = parser.parse_args()

    if args.hidden_size < 1:
        raise RuntimeError("--hidden-size must be positive")
    self_test_streamer_policy()

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
    eval_rows = load_stream(args.eval_stream)

    train_runtime, train_candidate, train_valid, train_target = streamer_arrays(
        train_rows
    )
    fit_end = int(0.8 * len(train_rows))
    labels = training_labels(
        train_rows,
        train_valid,
        train_target,
        args.min_lead,
        args.max_lead,
        fit_end,
    )

    model = StreamerGateLSTM(args.hidden_size)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != parameter_count(args.hidden_size):
        raise RuntimeError(
            "parameter-count formula mismatch: {} != {}".format(
                parameters, parameter_count(args.hidden_size)
            )
        )

    history = train_model(
        model,
        train_runtime,
        train_candidate,
        labels,
        train_valid,
        fit_end,
        device,
        args.epochs,
        args.chunk_len,
        args.batch_chunks,
        args.learning_rate,
    )
    train_scores = score_continuous(
        model, train_runtime, train_candidate, device
    )
    policy, sweep = calibrate(train_scores, labels, train_valid, fit_end)

    eval_runtime, eval_candidate, eval_valid, eval_target = streamer_arrays(
        eval_rows
    )
    eval_scores = score_continuous(
        model, eval_runtime, eval_candidate, device
    )
    lstm_selected = eval_scores >= policy["threshold"]
    streamer_selected = eval_valid.copy()

    streamer_entries, streamer_triggers = write_replay(
        args.out_dir / "offline_streamer.replay.csv",
        eval_rows,
        streamer_selected,
        eval_valid,
        eval_target,
    )
    lstm_entries, lstm_triggers = write_replay(
        args.out_dir / "offline_lstm.replay.csv",
        eval_rows,
        lstm_selected,
        eval_valid,
        eval_target,
    )

    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "parameters": parameters,
            "hidden_size": args.hidden_size,
            "trace": TRACE,
            "matched_normal_prefetcher": "streamer",
        },
        args.out_dir / "model.pt",
    )
    write_table(args.out_dir / "training_history.csv", history)
    write_table(args.out_dir / "policy_sweep.csv", sweep)

    metadata = {
        "trace": TRACE,
        "model_family": "LSTM candidate gate",
        "matched_normal_prefetcher": "streamer",
        "parameter_count": parameters,
        "hidden_size": args.hidden_size,
        "parameter_count_formula": "5*h^2 + 28*h + 1",
        "seed": args.seed,
        "streamer_num_trackers": TRACKERS,
        "streamer_pref_degree": DEGREE,
        "training_and_inference_location": "Colab_or_any_PyTorch_host",
        "inference_mode": "offline_causal_list_generation",
        "primary_methods": ["offline_streamer", "offline_lstm_streamer_gate"],
        "shared_eval_raw_input": ["current_cache_line_address", "causal_prior_address_history"],
        "runtime_features_derived_from_shared_address_stream": [
            "current_page_offset",
            "current_direction_for_page",
            "prior_direction_for_page",
            "page_tracker_hit",
        ],
        "candidate_features_derived_from_streamer_candidates": [
            "candidate_distance_1_to_5",
            "candidate_page_offset",
        ],
        "model_does_not_use_pc": True,
        "pc_line_occ_role": "replay_transport_identity_only",
        "forbidden_inputs": [
            "hit_miss",
            "cycle",
            "queue_state",
            "metadata",
            "future_evaluation_rows",
        ],
        "training_labels": (
            "future addresses only inside each chronological training/calibration "
            "split of the disjoint 0_to_20M training stream"
        ),
        "training_state_mode": "chronological_stateful_tbptt",
        "training_chunks_shuffled": False,
        "training_state_carried_across_chunks": True,
        "training_state_detached_between_chunks": True,
        "training_state_reset": "only_at_epoch_start",
        "training_chunk_len": args.chunk_len,
        "optimizer_step_every_chunks": args.batch_chunks,
        "inference_state_mode": "continuous_within_each_independent_stream",
        "experiment_revision": "stateful_tbptt_v2",
        "evaluation_stream_role": (
            "causal inference only; never used for fitting or threshold calibration"
        ),
        "transport": (
            "same keyed PC-line-occ ListReplayer for offline streamer and LSTM"
        ),
        "train_stream_sha256": sha256(args.train_stream),
        "eval_stream_sha256": sha256(args.eval_stream),
        "train_stream_content_sha256": gzip_content_sha256(args.train_stream),
        "eval_stream_content_sha256": gzip_content_sha256(args.eval_stream),
        "train_rows": len(train_rows),
        "fit_rows": fit_end,
        "calibration_rows": len(train_rows) - fit_end,
        "eval_rows": len(eval_rows),
        "threshold": policy["threshold"],
        "offline_streamer_entries": streamer_entries,
        "offline_streamer_triggers": streamer_triggers,
        "offline_lstm_entries": lstm_entries,
        "offline_lstm_triggers": lstm_triggers,
        "offline_streamer_list_sha256": sha256(
            args.out_dir / "offline_streamer.replay.csv"
        ),
        "offline_lstm_list_sha256": sha256(
            args.out_dir / "offline_lstm.replay.csv"
        ),
        "device": str(device),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "streamer_policy_self_test": "PASS",
    }
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(
        "[ok] "
        + json.dumps(
            {
                "device": str(device),
                "hidden_size": args.hidden_size,
                "parameters": parameters,
                "threshold": policy["threshold"],
                "streamer_entries": streamer_entries,
                "lstm_entries": lstm_entries,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

