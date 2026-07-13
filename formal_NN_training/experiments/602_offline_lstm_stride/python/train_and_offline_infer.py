#!/usr/bin/env python3
"""Train the tiny 602 LSTM and export offline LSTM and stride replay lists.

Both policies consume exactly the same evaluation PC/address stream.  Neither
policy reads hit/miss, cycle, queue state, metadata, or future evaluation rows.
Future rows are used only to construct labels inside the disjoint training
prefix.
"""
import argparse
import bisect
import csv
import gzip
import hashlib
import json
import math
import os
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

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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


def clip(value, bound):
    return max(-bound, min(bound, value)) / float(bound)


def pc_unit(pc):
    mixed = pc ^ (pc >> 12) ^ (pc >> 24)
    return float(mixed & 4095) / 4095.0


def causal_arrays(rows):
    n = len(rows)
    runtime = np.zeros((n, 4), dtype=np.float32)
    candidate = np.zeros((n, 2), dtype=np.float32)
    valid = np.zeros(n, dtype=np.bool_)
    repeat = np.zeros(n, dtype=np.bool_)
    target = np.zeros(n, dtype=np.int64)
    trackers = OrderedDict()
    for index, (pc, line, unused_occ) in enumerate(rows):
        del unused_occ
        stride = 0
        previous_stride = 0
        if pc in trackers:
            last_line, previous_stride = trackers[pc]
            stride = line - last_line
        runtime[index] = [pc_unit(pc), clip(stride, 256), clip(previous_stride, 256), (line & 63) / 63.0]
        candidate[index] = [clip(stride, 64), 1.0 if stride == previous_stride and stride != 0 else 0.0]
        if stride != 0:
            candidate_line = line + stride
            if candidate_line > 0 and candidate_line // PAGE_LINES == line // PAGE_LINES:
                valid[index] = True
                target[index] = candidate_line
                repeat[index] = stride == previous_stride
        # Match the implementation's zero-stride behavior: it does not update.
        if pc not in trackers:
            if len(trackers) >= TRACKERS:
                trackers.popitem(last=False)
            trackers[pc] = (line, stride)
        elif stride != 0:
            trackers.pop(pc)
            trackers[pc] = (line, stride)
    return runtime, candidate, valid, repeat, target


def training_labels(rows, valid, target, min_lead, max_lead):
    positions = defaultdict(list)
    for index, (_, line, _) in enumerate(rows):
        positions[line].append(index)
    labels = np.zeros(len(rows), dtype=np.float32)
    for index in np.flatnonzero(valid):
        values = positions[int(target[index])]
        pos = bisect.bisect_left(values, index + min_lead)
        if pos < len(values) and values[pos] <= index + max_lead:
            labels[index] = 1.0
    return labels


class TinyStrideLSTM(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(4, hidden_size, batch_first=True)
        self.projection = nn.Sequential(nn.Linear(hidden_size + 2, hidden_size), nn.Tanh())
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, runtime, candidate, state=None):
        hidden, state = self.lstm(runtime, state)
        joined = torch.cat([hidden, candidate], dim=-1)
        return self.head(self.projection(joined)).squeeze(-1), state


def parameter_count(hidden_size):
    # LSTM(4,h): 4h*4 + 4h*h + two 4h biases; projection(h+2,h);
    # scalar head.  This is 5h^2 + 28h + 1 and is checked against PyTorch below.
    return 5 * hidden_size * hidden_size + 28 * hidden_size + 1


def chunk_tensors(runtime, candidate, labels, valid, end, chunk_len):
    usable = (end // chunk_len) * chunk_len
    count = usable // chunk_len
    return (
        torch.from_numpy(runtime[:usable]).reshape(count, chunk_len, 4),
        torch.from_numpy(candidate[:usable]).reshape(count, chunk_len, 2),
        torch.from_numpy(labels[:usable]).reshape(count, chunk_len),
        torch.from_numpy(valid[:usable]).reshape(count, chunk_len),
    )


def train_model(model, runtime, candidate, labels, valid, fit_end, device, epochs, chunk_len, batch_chunks, lr):
    x, c, y, mask = chunk_tensors(runtime, candidate, labels, valid, fit_end, chunk_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    history = []
    model.to(device)
    for epoch in range(1, epochs + 1):
        generator = torch.Generator().manual_seed(1000 + epoch)
        order = torch.randperm(len(x), generator=generator)
        total_loss = 0.0
        total_valid = 0
        model.train()
        for start in range(0, len(order), batch_chunks):
            indices = order[start : start + batch_chunks]
            xb = x[indices].to(device)
            cb = c[indices].to(device)
            yb = y[indices].to(device)
            mb = mask[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(xb, cb)
            if not bool(mb.any()):
                continue
            loss = F.binary_cross_entropy_with_logits(logits[mb], yb[mb])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(mb.sum().item())
            total_loss += float(loss.item()) * count
            total_valid += count
        row = {"epoch": epoch, "loss": total_loss / max(1, total_valid), "valid_candidates": total_valid}
        history.append(row)
        print("[train] epoch={epoch} loss={loss:.6f} valid={valid_candidates}".format(**row))
    return history


def score_continuous(model, runtime, candidate, device, chunk_len=8192):
    model.eval()
    scores = np.zeros(len(runtime), dtype=np.float32)
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
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return issued, useful, precision, recall, f1


def calibrate(scores, labels, valid, repeat, start):
    finite = scores[start:][valid[start:]]
    thresholds = list(np.linspace(0.05, 0.95, 19))
    thresholds += [float(np.quantile(finite, q)) for q in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)]
    budget = float(repeat[start:].sum()) / max(1, len(repeat) - start)
    rows = []
    for threshold in sorted(set(round(x, 8) for x in thresholds)):
        selected = scores[start:] >= threshold
        issued, useful, precision, recall, f1 = metrics(selected, labels[start:], valid[start:])
        rate = issued / float(max(1, len(selected)))
        rows.append(
            {
                "threshold": threshold,
                "issued": issued,
                "useful": useful,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "issue_rate": rate,
                "offline_stride_issue_rate_budget": budget,
                "within_budget": int(rate <= budget + 1.0 / max(1, len(selected))),
            }
        )
    eligible = [row for row in rows if row["within_budget"]]
    if not eligible:
        raise RuntimeError("no calibration threshold satisfies the offline stride issue budget")
    eligible.sort(key=lambda row: (-row["f1"], -row["precision"], -row["recall"], row["issue_rate"]))
    return eligible[0], rows


def write_table(path, rows):
    path = Path(path)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_replay(path, rows, selected, valid, target):
    count = 0
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pc", "line", "occ", "prefetch_addr"])
        for index in np.flatnonzero(selected & valid):
            pc, line, occ = rows[int(index)]
            writer.writerow([pc, line, occ, "0x{:x}".format(int(target[index]) * 64)])
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-stream", required=True, type=Path)
    ap.add_argument("--eval-stream", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--chunk-len", type=int, default=1024)
    ap.add_argument("--batch-chunks", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=0.002)
    ap.add_argument("--min-lead", type=int, default=4)
    ap.add_argument("--max-lead", type=int, default=64)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--hidden-size", type=int, default=8,
                    help="LSTM hidden width; each width is one independent capacity-sweep point")
    args = ap.parse_args()
    if args.hidden_size < 1:
        raise RuntimeError("--hidden-size must be positive")

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
    train_runtime, train_candidate, train_valid, train_repeat, train_target = causal_arrays(train_rows)
    train_labels_array = training_labels(train_rows, train_valid, train_target, args.min_lead, args.max_lead)
    fit_end = int(0.8 * len(train_rows))
    model = TinyStrideLSTM(args.hidden_size)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != parameter_count(args.hidden_size):
        raise RuntimeError("parameter-count formula mismatch: {} != {}".format(parameters, parameter_count(args.hidden_size)))
    history = train_model(
        model, train_runtime, train_candidate, train_labels_array, train_valid,
        fit_end, device, args.epochs, args.chunk_len, args.batch_chunks, args.learning_rate,
    )
    train_scores = score_continuous(model, train_runtime, train_candidate, device)
    policy, sweep = calibrate(train_scores, train_labels_array, train_valid, train_repeat, fit_end)

    eval_runtime, eval_candidate, eval_valid, eval_repeat, eval_target = causal_arrays(eval_rows)
    eval_scores = score_continuous(model, eval_runtime, eval_candidate, device)
    lstm_selected = eval_scores >= policy["threshold"]
    stride_selected = eval_repeat.copy()
    stride_count = write_replay(args.out_dir / "offline_stride.replay.csv", eval_rows, stride_selected, eval_valid, eval_target)
    lstm_count = write_replay(args.out_dir / "offline_lstm.replay.csv", eval_rows, lstm_selected, eval_valid, eval_target)
    torch.save({"state_dict": model.cpu().state_dict(), "parameters": parameters, "hidden_size": args.hidden_size, "trace": TRACE}, args.out_dir / "model.pt")
    write_table(args.out_dir / "training_history.csv", history)
    write_table(args.out_dir / "policy_sweep.csv", sweep)
    metadata = {
        "trace": TRACE,
        "model_family": "LSTM",
        "parameter_count": parameters,
        "hidden_size": args.hidden_size,
        "parameter_count_formula": "5*h^2 + 28*h + 1",
        "seed": args.seed,
        "training_and_inference_location": "Colab_or_any_PyTorch_host",
        "inference_mode": "offline_causal_list_generation",
        "primary_methods": ["offline_stride", "offline_lstm"],
        "shared_eval_inputs": ["current_pc", "current_cache_line", "causal_prior_pc_address_history"],
        "forbidden_inputs": ["hit_miss", "cycle", "queue_state", "metadata", "future_evaluation_rows"],
        "training_labels": "future addresses only inside the disjoint 0_to_20M training stream",
        "evaluation_stream_role": "causal inference only; never used for fitting or threshold calibration",
        "transport": "same keyed PC-line-occ ListReplayer for both primary methods",
        "train_stream_sha256": sha256(args.train_stream),
        "eval_stream_sha256": sha256(args.eval_stream),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "threshold": policy["threshold"],
        "offline_stride_entries": stride_count,
        "offline_lstm_entries": lstm_count,
        "offline_stride_list_sha256": sha256(args.out_dir / "offline_stride.replay.csv"),
        "offline_lstm_list_sha256": sha256(args.out_dir / "offline_lstm.replay.csv"),
        "device": str(device),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print("[ok] " + json.dumps({"device": str(device), "hidden_size": args.hidden_size, "parameters": parameters, "threshold": policy["threshold"], "stride_entries": stride_count, "lstm_entries": lstm_count}, sort_keys=True))


if __name__ == "__main__":
    main()
