#!/usr/bin/env python3
"""Train the live 602 matched-input LSTM and export its C++ runtime state.

The model consumes only the PC and address delivered to each L2 prefetcher
callback, plus causal state derived from earlier callbacks.  Later addresses
inside the earlier 20M-instruction training window are targets; hit/miss and
all other oracle columns are ignored.  Evaluation data is not an input to this
program.  The trained model is executed live inside ChampSim, so its recurrent
state follows the LSTM run's real callback order rather than a no-pref replay.

No pandas import is used.  NumPy and PyTorch are the only third-party training
dependencies; all Sacramento-side orchestration and analysis remain standard
library/shell code.
"""
from __future__ import print_function

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
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit(
        "[dependency error] NumPy and PyTorch are required; pandas is not: {}".format(exc)
    )


TRACE = "602.gcc_s-734B"
LINE_BYTES = 64
PAGE_LINES = 64
TRACKER_CAPACITY = 64
MODEL_FORMAT = "matched_stride_lstm_runtime_v2"
MODEL_NAME = "matched_stride_lstm_545p_v2"
RUNTIME_INPUTS = [
    "current_l2_load_pc",
    "current_l2_load_cache_line_address",
    "causal_64_entry_lru_per_pc_previous_line",
    "causal_64_entry_lru_per_pc_previous_stride",
    "causal_callback_order_for_live_lstm_state",
]
FORBIDDEN_RUNTIME_INPUTS = [
    "cycle",
    "cache_hit_or_no_pref_miss",
    "was_prefetch_or_late",
    "pq_or_mshr_occupancy",
    "access_metadata",
    "future_event_or_label",
    "normal_prefetcher_output",
]


def parse_int(value, name):
    if value is None or str(value).strip() == "":
        raise ValueError("missing {}".format(name))
    text = str(value).strip()
    return int(text, 16) if text.lower().startswith("0x") else int(float(text))


def open_csv(path):
    path = Path(path)
    if str(path).endswith(".gz"):
        return gzip.open(str(path), "rt", newline="")
    return path.open("r", newline="")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(path))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_training_stream(path):
    records = []
    required = {"trace", "demand_idx", "pc", "line"}
    with open_csv(path) as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError("training stream missing columns {}".format(sorted(missing)))
        for expected_index, row in enumerate(reader):
            if row.get("trace") != TRACE:
                raise ValueError("training stream contains a non-602 row")
            demand_idx = parse_int(row.get("demand_idx"), "demand_idx")
            if demand_idx != expected_index:
                raise ValueError("demand_idx must be contiguous from zero")
            records.append({
                "pc": parse_int(row.get("pc"), "pc"),
                "line": parse_int(row.get("line"), "line"),
            })
    if len(records) < 4096:
        raise RuntimeError("training stream has too few L2 LOAD rows: {}".format(len(records)))
    return records


def pc_unit(pc):
    mixed = int(pc) ^ (int(pc) >> 12) ^ (int(pc) >> 24)
    return float(mixed & 4095) / 4095.0


def clip_unit(value, bound):
    return float(max(-bound, min(bound, int(value)))) / float(bound)


def build_runtime_features(records):
    """Mirror stride's live 64-entry tracker using PC/address only."""
    trackers = OrderedDict()
    features = np.zeros((len(records), 4), dtype=np.float32)
    strides = np.zeros(len(records), dtype=np.int64)
    previous_strides = np.zeros(len(records), dtype=np.int64)
    evictions = 0
    for index, row in enumerate(records):
        pc, line = int(row["pc"]), int(row["line"])
        previous = trackers.get(pc)
        stride = 0 if previous is None else line - int(previous[0])
        previous_stride = 0 if previous is None else int(previous[1])
        features[index] = (
            pc_unit(pc),
            clip_unit(stride, 256),
            clip_unit(previous_stride, 256),
            float(line % PAGE_LINES) / 63.0,
        )
        strides[index] = stride
        previous_strides[index] = previous_stride
        # The audited stride implementation inserts a new PC, but returns
        # without updating or touching its LRU on a zero stride.
        if previous is None:
            trackers[pc] = (line, 0)
            trackers.move_to_end(pc, last=False)
            if len(trackers) > TRACKER_CAPACITY:
                trackers.popitem(last=True)
                evictions += 1
        elif stride != 0:
            trackers[pc] = (line, stride)
            trackers.move_to_end(pc, last=False)
    return features, strides, previous_strides, evictions


def build_candidates(records, strides, previous_strides):
    total = len(records)
    deltas = np.zeros((total, 1), dtype=np.int64)
    valid = np.zeros((total, 1), dtype=np.uint8)
    candidate_features = np.zeros((total, 1, 2), dtype=np.float32)
    stride_repeat = np.zeros(total, dtype=np.uint8)
    for index, row in enumerate(records):
        line = int(row["line"])
        page = line // PAGE_LINES
        stride = int(strides[index])
        previous_stride = int(previous_strides[index])
        if stride and line + stride > 0 and (line + stride) // PAGE_LINES == page:
            deltas[index, 0] = stride
            valid[index, 0] = 1
            candidate_features[index, 0] = (
                clip_unit(stride, 64),
                1.0 if stride == previous_stride else 0.5,
            )
            if stride == previous_stride:
                stride_repeat[index] = 1
    return deltas, valid, candidate_features, stride_repeat


def make_labels(records, deltas, valid, start, end, target_end, min_lead, max_lead):
    labels = np.zeros_like(valid, dtype=np.float32)
    accesses_by_line = defaultdict(list)
    for index in range(int(target_end)):
        accesses_by_line[int(records[index]["line"])].append(index)
    for index in range(int(start), int(end)):
        lower = index + int(min_lead)
        upper = min(int(target_end) - 1, index + int(max_lead))
        if lower > upper:
            continue
        base = int(records[index]["line"])
        for slot in range(valid.shape[1]):
            if not int(valid[index, slot]):
                continue
            positions = accesses_by_line.get(base + int(deltas[index, slot]), [])
            cursor = bisect.bisect_left(positions, lower)
            if cursor < len(positions) and positions[cursor] <= upper:
                labels[index, slot] = 1.0
    return labels


class MatchedStrideLSTM(nn.Module):
    def __init__(self):
        super(MatchedStrideLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=8, num_layers=1, batch_first=True)
        self.candidate_projection = nn.Sequential(nn.Linear(10, 8), nn.Tanh())
        self.utility_head = nn.Linear(8, 1)

    def forward(self, runtime_features, candidate_features, state=None):
        hidden, state = self.lstm(runtime_features, state)
        expanded = hidden.unsqueeze(2).expand(
            hidden.size(0), hidden.size(1), candidate_features.size(2), hidden.size(2)
        )
        joined = torch.cat([expanded, candidate_features], dim=-1)
        scores = self.utility_head(self.candidate_projection(joined)).squeeze(-1)
        return scores, state


def parameter_count(model):
    return int(sum(value.numel() for value in model.parameters() if value.requires_grad))


def seed_all(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def score_sequence(model, runtime, candidates, device, chunk_len):
    model.eval()
    output = np.zeros(candidates.shape[:2], dtype=np.float32)
    state = None
    with torch.no_grad():
        for start in range(0, len(runtime), int(chunk_len)):
            end = min(len(runtime), start + int(chunk_len))
            x = torch.from_numpy(runtime[start:end]).to(device).unsqueeze(0)
            c = torch.from_numpy(candidates[start:end]).to(device).unsqueeze(0)
            logits, state = model(x, c, state)
            state = tuple(value.detach() for value in state)
            output[start:end] = torch.sigmoid(logits[0]).cpu().numpy()
    return output


def validation_bce(scores, labels, valid, start):
    mask = valid[int(start):].astype(bool)
    if not np.any(mask):
        raise RuntimeError("calibration suffix has no legal stride candidates")
    probability = np.clip(scores[int(start):][mask], 1e-7, 1.0 - 1e-7)
    target = labels[int(start):][mask]
    return float(np.mean(-(target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability))))


def train_model(model, runtime, candidates, labels, valid, fit_end, device, epochs, chunk_len, learning_rate):
    model.to(device)
    if parameter_count(model) != 545:
        raise RuntimeError("expected 545 parameters, got {}".format(parameter_count(model)))
    mask = valid[:int(fit_end)].astype(bool)
    positive = float(np.sum(labels[:int(fit_end)][mask]))
    total = float(np.sum(mask))
    if positive <= 0 or total <= positive:
        raise RuntimeError("fit candidates require both positive and negative labels")
    pos_weight = min(20.0, max(1.0, (total - positive) / positive))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=1e-5)
    history, best_state, best_bce, stale = [], None, None, 0
    for epoch in range(int(epochs)):
        model.train()
        state, total_loss, batches = None, 0.0, 0
        for start in range(0, int(fit_end), int(chunk_len)):
            end = min(int(fit_end), start + int(chunk_len))
            x = torch.from_numpy(runtime[start:end]).to(device).unsqueeze(0)
            c = torch.from_numpy(candidates[start:end]).to(device).unsqueeze(0)
            y = torch.from_numpy(labels[start:end]).to(device).unsqueeze(0)
            valid_mask = torch.from_numpy(valid[start:end].astype(np.float32)).to(device).unsqueeze(0)
            logits, state = model(x, c, state)
            state = tuple(value.detach() for value in state)
            losses = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=torch.tensor(pos_weight, device=device), reduction="none"
            )
            loss = torch.sum(losses * valid_mask) / torch.clamp(valid_mask.sum(), min=1.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        scores = score_sequence(model, runtime, candidates, device, chunk_len)
        current_bce = validation_bce(scores, labels, valid, fit_end)
        history.append({
            "epoch": epoch + 1,
            "fit_loss": total_loss / float(max(1, batches)),
            "calibration_bce": current_bce,
            "fit_positive_candidates": int(positive),
            "fit_valid_candidates": int(total),
            "positive_weight": pos_weight,
        })
        if best_bce is None or current_bce < best_bce - 1e-6:
            best_bce = current_bce
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= 2:
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return history


def select_top1(scores, valid, threshold):
    selected = np.zeros_like(valid, dtype=np.uint8)
    for index in range(scores.shape[0]):
        choices = [
            slot for slot in range(valid.shape[1])
            if int(valid[index, slot]) and float(scores[index, slot]) >= float(threshold)
        ]
        if choices:
            choices.sort(key=lambda slot: (-float(scores[index, slot]), slot))
            selected[index, choices[0]] = 1
    return selected


def policy_metrics(selected, labels, valid):
    chosen = selected.astype(bool)
    issued = int(chosen.sum())
    useful = int(np.sum(chosen & (labels > 0.5)))
    reachable = int(np.sum(np.any((labels > 0.5) & valid.astype(bool), axis=1)))
    covered = int(np.sum(np.any((labels > 0.5) & chosen, axis=1)))
    precision = float(useful) / float(issued) if issued else 0.0
    recall = float(covered) / float(reachable) if reachable else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "issued": issued,
        "useful_label_matches": useful,
        "reachable_positive_events": reachable,
        "covered_positive_events": covered,
        "precision": precision,
        "reachable_event_recall": recall,
        "f1": f1,
        "issue_per_event": float(issued) / float(max(1, selected.shape[0])),
    }


def calibrate_policy(scores, labels, valid, stride_repeat):
    finite = scores[valid.astype(bool)]
    if not len(finite):
        raise RuntimeError("calibration suffix has no scores")
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.000001]
    thresholds.extend(
        float(np.quantile(finite, quantile))
        for quantile in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99)
    )
    stride_rate = float(np.sum(stride_repeat)) / float(max(1, len(stride_repeat)))
    rows = []
    for threshold in sorted(set(round(value, 8) for value in thresholds)):
        metrics = policy_metrics(select_top1(scores, valid, threshold), labels, valid)
        metrics.update({
            "threshold": threshold,
            "max_degree": 1,
            "stride_repeat_event_rate_budget": stride_rate,
            "within_stride_event_rate_budget": int(
                metrics["issue_per_event"] <= stride_rate + 1.0 / float(max(1, len(stride_repeat)))
            ),
        })
        rows.append(metrics)
    eligible = [row for row in rows if row["within_stride_event_rate_budget"]]
    eligible.sort(
        key=lambda row: (-row["f1"], -row["precision"], -row["reachable_event_recall"],
                         row["issue_per_event"], -row["threshold"])
    )
    if not eligible:
        raise RuntimeError("no threshold satisfies the stride event-rate budget")
    return eligible[0], rows


def flat_values(tensor):
    return tensor.detach().cpu().numpy().astype(np.float32).reshape(-1)


def write_vector(handle, name, values):
    handle.write(name)
    for value in values:
        handle.write(" {:.9g}".format(float(value)))
    handle.write("\n")


def export_runtime_model(path, model, policy):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    linear = model.candidate_projection[0]
    with path.open("w") as handle:
        handle.write("format {}\n".format(MODEL_FORMAT))
        handle.write("input_size 4\n")
        handle.write("hidden_size 8\n")
        handle.write("candidate_size 2\n")
        handle.write("threshold {:.9g}\n".format(float(policy["threshold"])))
        write_vector(handle, "weight_ih", flat_values(model.lstm.weight_ih_l0))
        write_vector(handle, "weight_hh", flat_values(model.lstm.weight_hh_l0))
        write_vector(handle, "bias_ih", flat_values(model.lstm.bias_ih_l0))
        write_vector(handle, "bias_hh", flat_values(model.lstm.bias_hh_l0))
        write_vector(handle, "projection_weight", flat_values(linear.weight))
        write_vector(handle, "projection_bias", flat_values(linear.bias))
        write_vector(handle, "utility_weight", flat_values(model.utility_head.weight))
        handle.write("utility_bias {:.9g}\n".format(float(model.utility_head.bias.detach().cpu()[0])))


def scalar_export_parity(model, runtime, candidates, count=64):
    """Check the exact gate/projection math implemented by the C++ runtime."""
    state = model.state_dict()
    wih = flat_values(state["lstm.weight_ih_l0"]).reshape(32, 4)
    whh = flat_values(state["lstm.weight_hh_l0"]).reshape(32, 8)
    bih = flat_values(state["lstm.bias_ih_l0"])
    bhh = flat_values(state["lstm.bias_hh_l0"])
    pw = flat_values(state["candidate_projection.0.weight"]).reshape(8, 10)
    pb = flat_values(state["candidate_projection.0.bias"])
    uw = flat_values(state["utility_head.weight"]).reshape(8)
    ub = float(flat_values(state["utility_head.bias"])[0])
    hidden = np.zeros(8, dtype=np.float32)
    cell = np.zeros(8, dtype=np.float32)
    scalar_scores = np.zeros((count, candidates.shape[1]), dtype=np.float32)
    for index in range(count):
        gates = wih.dot(runtime[index]) + whh.dot(hidden) + bih + bhh
        ingate = 1.0 / (1.0 + np.exp(-gates[0:8]))
        forget = 1.0 / (1.0 + np.exp(-gates[8:16]))
        candidate = np.tanh(gates[16:24])
        output = 1.0 / (1.0 + np.exp(-gates[24:32]))
        cell = forget * cell + ingate * candidate
        hidden = output * np.tanh(cell)
        for slot in range(candidates.shape[1]):
            projected = np.tanh(pw.dot(np.concatenate([hidden, candidates[index, slot]])) + pb)
            scalar_scores[index, slot] = 1.0 / (1.0 + math.exp(-(float(uw.dot(projected)) + ub)))
    model.eval()
    with torch.no_grad():
        logits, unused_state = model(
            torch.from_numpy(runtime[:count]).unsqueeze(0),
            torch.from_numpy(candidates[:count]).unsqueeze(0),
        )
        del unused_state
        torch_scores = torch.sigmoid(logits[0]).cpu().numpy()
    error = float(np.max(np.abs(scalar_scores - torch_scores)))
    if error > 1e-5:
        raise RuntimeError("C++-math export parity failed: max error {}".format(error))
    return error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-stream", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--run-id", default="602_matched_stride_lstm_seed7")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fit-fraction", type=float, default=0.80)
    parser.add_argument("--min-lead", type=int, default=4)
    parser.add_argument("--max-lead", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--chunk-len", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if not args.train_stream.is_file():
        parser.error("training stream does not exist")
    if not 0.50 <= args.fit_fraction < 0.95:
        parser.error("fit fraction must be in [0.50, 0.95)")
    if args.min_lead < 1 or args.max_lead < args.min_lead:
        parser.error("invalid lead window")

    seed_all(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    records = read_training_stream(args.train_stream)
    fit_end = int(len(records) * args.fit_fraction)
    if min(fit_end, len(records) - fit_end) <= args.max_lead:
        raise RuntimeError("fit/calibration split is too small")
    runtime, strides, previous_strides, tracker_evictions = build_runtime_features(records)
    deltas, valid, candidates, stride_repeat = build_candidates(
        records, strides, previous_strides
    )
    labels = make_labels(
        records, deltas, valid, 0, fit_end, fit_end, args.min_lead, args.max_lead
    )
    labels += make_labels(
        records, deltas, valid, fit_end, len(records), len(records),
        args.min_lead, args.max_lead,
    )
    labels = np.minimum(labels, 1.0)

    model = MatchedStrideLSTM()
    history = train_model(
        model, runtime, candidates, labels, valid, fit_end, device,
        args.epochs, args.chunk_len, args.learning_rate,
    )
    scores = score_sequence(model, runtime, candidates, device, args.chunk_len)
    policy, sweep = calibrate_policy(
        scores[fit_end:], labels[fit_end:], valid[fit_end:], stride_repeat[fit_end:]
    )
    parity_error = scalar_export_parity(model.cpu(), runtime, candidates)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = args.artifact_dir / "matched_stride_lstm.runtime.txt"
    checkpoint_path = args.artifact_dir / "matched_stride_lstm.pt"
    export_runtime_model(runtime_path, model, policy)
    torch.save({
        "model_name": MODEL_NAME,
        "state_dict": model.state_dict(),
        "parameter_count": parameter_count(model),
        "trace": TRACE,
    }, str(checkpoint_path))
    write_csv(args.artifact_dir / "training_history.csv", history, list(history[0].keys()))
    write_csv(args.artifact_dir / "policy_sweep_training_calibration_only.csv", sweep, list(sweep[0].keys()))
    metadata = {
        "trace": TRACE,
        "run_id": args.run_id,
        "model_name": MODEL_NAME,
        "model_family": "LSTM",
        "parameter_count": parameter_count(model),
        "hidden_units": 8,
        "candidate_slots": 1,
        "max_prefetches_per_event": 1,
        "tracker_capacity": TRACKER_CAPACITY,
        "runtime_inputs": RUNTIME_INPUTS,
        "forbidden_runtime_inputs": FORBIDDEN_RUNTIME_INPUTS,
        "input_contract": "PC and current cache-line address at each live L2 callback, plus causal state derived from those fields only",
        "training_columns_consumed": ["trace", "demand_idx", "pc", "line"],
        "label_contract": "later addresses in the earlier PC/address training stream are targets; hit/miss and evaluation rows are not consumed",
        "live_in_simulator_inference": True,
        "keyed_offline_replay_used_for_primary_result": False,
        "evaluation_data_used_for_training_or_policy": False,
        "train_stream": str(args.train_stream),
        "train_stream_sha256": sha256_file(args.train_stream),
        "train_rows": len(records),
        "fit_rows": fit_end,
        "calibration_rows": len(records) - fit_end,
        "fit_fraction": args.fit_fraction,
        "lead_window_events": [args.min_lead, args.max_lead],
        "policy": policy,
        "tracker_evictions_in_training_stream": tracker_evictions,
        "export_math_max_abs_error": parity_error,
        "runtime_model": str(runtime_path),
        "runtime_model_sha256": sha256_file(runtime_path),
        "checkpoint": str(checkpoint_path),
        "seed": args.seed,
        "device": str(device),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "pandas_dependency": "none",
    }
    write_json(args.artifact_dir / "run_metadata.json", metadata)
    print("[ok] {}".format(json.dumps({
        "model": MODEL_NAME,
        "parameters": 545,
        "threshold": policy["threshold"],
        "runtime_model": str(runtime_path),
        "parity_error": parity_error,
    }, sort_keys=True)))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("[error] {}".format(exc), file=sys.stderr)
        raise
