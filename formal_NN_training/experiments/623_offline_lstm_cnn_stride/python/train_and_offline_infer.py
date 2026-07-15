#!/usr/bin/env python3
"""Train standalone direct-action LSTM/CNN students against Stride on 623.

Both methods receive the effective Stride source inputs (PC and cache-line
address).  Captured Stride requests are used only to write the normal replay
and set a training-split request budget.  They never enter either neural model.
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
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formal_NN_training.common.direct_action_pair import (
    CNN_DILATION, CNN_KERNEL_SIZE, CNN_STRIDE, build_model, score_cnn,
    score_lstm, self_test_cnn, train_cnn, train_lstm,
)


TRACE = "623.xalancbmk_s-700B"
POLICY = "stride"
PAGE_LINES = 64
RUNTIME_FEATURES = 14
ACTION_CLASSES = 64
MAX_ACTIONS_PER_CALLBACK = 2
EVENT_LOGGER_SCHEMA = "623_causal_trigger_v5"
CANDIDATE_ATTACHMENT_MODE = "explicit_trigger_event_id"
EXPERIMENT_REVISION = "stride_direct_io_sliding_cnn_v3"
PAIR_SPECS = {
    ("lstm", 4): ("p0", 640),
    ("cnn", 5): ("p0", 599),
    ("lstm", 8): ("p1", 1344),
    ("cnn", 12): ("p1", 1348),
    ("lstm", 16): ("p2", 3136),
    ("cnn", 29): ("p2", 3167),
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
        required = {"trace", "demand_idx", "pc", "line", "pc_line_occ", "logger_schema"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError("{} missing {}".format(path, sorted(missing)))
        for index, row in enumerate(reader):
            pc = as_int(row["pc"])
            line = as_int(row["line"])
            occurrence = as_int(row["pc_line_occ"])
            expected = occurrences[(pc, line)]
            occurrences[(pc, line)] += 1
            if (
                row["trace"] != TRACE
                or as_int(row["demand_idx"]) != index
                or row["logger_schema"] != EVENT_LOGGER_SCHEMA
                or occurrence != expected
            ):
                raise RuntimeError("stream identity/ordering failure at row {}".format(index))
            rows.append((pc, line, occurrence))
    if not rows:
        raise RuntimeError("empty stream: {}".format(path))
    return rows


def load_normal_actions(path, rows):
    actions = [[] for _ in rows]
    last_pf_event = -1
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "trace", "policy", "demand_idx", "pc", "line", "pc_line_occ",
            "candidate_rank", "pf_line", "fill_level", "accepted", "duplicate",
            "trigger_event_id", "pf_event_id", "event_distance", "match_mode",
            "logger_schema",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError("{} missing {}".format(path, sorted(missing)))
        for row in reader:
            index = as_int(row["demand_idx"])
            if index < 0 or index >= len(rows):
                raise RuntimeError("normal action demand_idx out of range")
            pc, line, occurrence = rows[index]
            if (
                row["trace"] != TRACE
                or row["policy"] != POLICY
                or (as_int(row["pc"]), as_int(row["line"]), as_int(row["pc_line_occ"]))
                != (pc, line, occurrence)
                or row["logger_schema"] != EVENT_LOGGER_SCHEMA
                or row["match_mode"] != CANDIDATE_ATTACHMENT_MODE
            ):
                raise RuntimeError("normal action identity failure at {}".format(index))
            if as_int(row["candidate_rank"]) != len(actions[index]) + 1:
                raise RuntimeError("noncontiguous normal action rank at {}".format(index))
            trigger = as_int(row["trigger_event_id"])
            pf_event = as_int(row["pf_event_id"])
            distance = as_int(row["event_distance"])
            if (
                pf_event <= last_pf_event or trigger >= pf_event
                or distance != pf_event - trigger or not 1 <= distance <= 256
            ):
                raise RuntimeError("invalid trigger transport at {}".format(index))
            pf_line = as_int(row["pf_line"])
            if (
                as_int(row["fill_level"]) != 2
                or as_int(row["accepted"]) not in (0, 1)
                or as_int(row["duplicate"]) not in (0, 1)
                or pf_line // PAGE_LINES != line // PAGE_LINES
            ):
                raise RuntimeError("invalid source-legal Stride action at {}".format(index))
            actions[index].append(pf_line)
            if len(actions[index]) > MAX_ACTIONS_PER_CALLBACK:
                raise RuntimeError("Stride action count exceeds degree two")
            last_pf_event = pf_event
    if not any(actions):
        raise RuntimeError("empty Stride action stream {}".format(path))
    return actions


def _chunks16(value):
    return [((value >> shift) & 0xFFFF) / 65535.0 for shift in (0, 16, 32, 48)]


def runtime_array(rows, previous_line=None, previous_pc=None):
    runtime = np.zeros((len(rows), RUNTIME_FEATURES), dtype=np.float32)
    prior_line = previous_line
    prior_pc = previous_pc
    log_scale = math.log1p(4096.0)
    for index, (pc, line, _) in enumerate(rows):
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
        runtime[index] = [
            offset / 63.0,
            *_chunks16(page),
            np.clip(delta, -256, 256) / 256.0,
            min(1.0, math.log1p(abs(delta)) / log_scale),
            same_page,
            np.clip(page_delta, -64, 64) / 64.0,
            *_chunks16(pc),
            float(prior_pc is not None and pc == prior_pc),
        ]
        prior_line = line
        prior_pc = pc
    return runtime, prior_line, prior_pc


def future_use_labels(rows, min_lead, max_lead):
    positions = defaultdict(list)
    for index, (_, line, _) in enumerate(rows):
        positions[line].append(index)
    labels = np.zeros((len(rows), ACTION_CLASSES), dtype=np.uint8)
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
    return labels


def decode(scores, rows, threshold):
    selected = np.zeros(scores.shape, dtype=np.bool_)
    for index in range(len(rows)):
        eligible = np.flatnonzero(scores[index] >= threshold)
        if eligible.size:
            order = np.argsort(-scores[index, eligible], kind="stable")
            selected[index, eligible[order[:MAX_ACTIONS_PER_CALLBACK]]] = True
    return selected


def action_metrics(selected, labels):
    truth = labels.astype(np.bool_)
    tp = int(np.logical_and(selected, truth).sum())
    fp = int(np.logical_and(selected, ~truth).sum())
    fn = int(np.logical_and(~selected, truth).sum())
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "predicted_actions": int(selected.sum()), "useful_labels": int(truth.sum()),
        "true_positive_actions": tp, "false_positive_actions": fp,
        "false_negative_actions": fn, "precision": precision,
        "recall": recall, "f1": f1,
    }


def calibrate(scores, labels, rows, normal_actions, start):
    finite = scores[start:].reshape(-1)
    thresholds = [0.0, 1.0]
    thresholds.extend(np.linspace(0.05, 0.95, 19).tolist())
    thresholds.extend(float(np.quantile(finite, q)) for q in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995))
    normal_rate = sum(len(items) for items in normal_actions[start:]) / float(max(1, len(rows) - start))
    sweep = []
    for threshold in sorted(set(round(value, 8) for value in thresholds)):
        selected = decode(scores[start:], rows[start:], threshold)
        metrics = action_metrics(selected, labels[start:])
        rate = metrics["predicted_actions"] / float(max(1, len(rows) - start))
        metrics.update({
            "threshold": threshold, "actions_per_callback": rate,
            "normal_action_budget_per_callback": normal_rate,
            "within_normal_action_budget": int(rate <= normal_rate + 1.0 / max(1, len(rows) - start)),
        })
        sweep.append(metrics)
    eligible = [row for row in sweep if row["within_normal_action_budget"]]
    if not eligible:
        raise RuntimeError("no threshold satisfies the Stride request budget")
    eligible.sort(key=lambda row: (-row["f1"], -row["recall"], -row["precision"], row["actions_per_callback"], -row["threshold"]))
    return eligible[0], sweep


def write_table(path, rows):
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
                writer.writerow([pc, line, occurrence, "0x{:x}".format(target << 6)])
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
                writer.writerow([pc, line, occurrence, "0x{:x}".format((page_base + int(offset)) << 6)])
                entries += 1
    return entries, triggers


def model_tag(family, size):
    return "direct_stride_{}".format("lstm_h{}".format(size) if family == "lstm" else "cnn_c{}".format(size))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, choices=[POLICY])
    for role in ("train", "guard", "eval"):
        parser.add_argument("--{}-stream".format(role), required=True, type=Path)
        parser.add_argument("--{}-candidates".format(role), required=True, type=Path)
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
        raise RuntimeError("invalid lead window")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    self_test_cnn(RUNTIME_FEATURES, ACTION_CLASSES)

    roles = ("train", "guard", "eval")
    stream_paths = {role: getattr(args, role + "_stream") for role in roles}
    action_paths = {role: getattr(args, role + "_candidates") for role in roles}
    rows = {role: load_stream(stream_paths[role]) for role in roles}
    normal = {role: load_normal_actions(action_paths[role], rows[role]) for role in roles}
    train_runtime, _, _ = runtime_array(rows["train"])
    guard_runtime, guard_line, guard_pc = runtime_array(rows["guard"])
    eval_runtime, _, _ = runtime_array(rows["eval"], guard_line, guard_pc)
    labels = future_use_labels(rows["train"], args.min_lead, args.max_lead)
    eval_labels = future_use_labels(rows["eval"], args.min_lead, args.max_lead)
    split = int(len(rows["train"]) * 0.8)
    fit_end = split - args.max_lead
    if fit_end <= 0 or split >= len(rows["train"]):
        raise RuntimeError("training stream too short for leakage-free fit/calibration")
    positives = int(labels[:fit_end].sum())
    total = fit_end * ACTION_CLASSES
    if positives <= 0 or positives >= total:
        raise RuntimeError("degenerate future-use labels")
    positive_weight = min(100.0, max(1.0, (total - positives) / float(positives)))

    model, parameter_count = build_model(
        args.model_family, RUNTIME_FEATURES, ACTION_CLASSES,
        args.model_size, args.pair_id, PAIR_SPECS,
    )
    train_args = (
        model, train_runtime, labels, fit_end, device, args.epochs,
        args.chunk_len, args.accumulate_chunks, args.learning_rate,
        positive_weight,
    )
    history = train_lstm(*train_args) if args.model_family == "lstm" else train_cnn(*train_args)
    if args.model_family == "lstm":
        train_scores, _ = score_lstm(model, train_runtime, device)
        _, guard_state = score_lstm(model, guard_runtime, device)
        eval_scores, _ = score_lstm(model, eval_runtime, device, guard_state)
    else:
        train_scores = score_cnn(model, train_runtime, device)
        eval_scores = score_cnn(model, eval_runtime, device, guard_runtime)
    calibration, sweep = calibrate(train_scores, labels, rows["train"], normal["train"], split)
    selected_eval = decode(eval_scores, rows["eval"], calibration["threshold"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    normal_path = args.out_dir / "offline_stride.replay.csv"
    nn_path = args.out_dir / "offline_nn.replay.csv"
    normal_entries, normal_triggers = write_normal_replay(normal_path, rows["eval"], normal["eval"])
    nn_entries, nn_triggers = write_nn_replay(nn_path, rows["eval"], selected_eval, eval_scores)
    write_table(args.out_dir / "policy_sweep.csv", sweep)
    torch.save({
        "state_dict": model.state_dict(), "model_family": args.model_family,
        "model_size": args.model_size, "runtime_features": RUNTIME_FEATURES,
        "action_classes": ACTION_CLASSES, "experiment_revision": EXPERIMENT_REVISION,
    }, args.out_dir / "model.pt")

    tag = model_tag(args.model_family, args.model_size)
    metadata = {
        "trace": TRACE, "model_tag": tag, "matched_normal_prefetcher": POLICY,
        "neural_role": "standalone_direct_action_prefetcher",
        "model_family": args.model_family, "model_size": args.model_size,
        "architecture_pair_id": args.pair_id, "parameter_count": parameter_count,
        "seed": args.seed, "epochs": args.epochs, "chunk_len": args.chunk_len,
        "accumulate_chunks": args.accumulate_chunks,
        "learning_rate": args.learning_rate, "positive_class_weight": positive_weight,
        "threshold": calibration["threshold"], "fit_rows": fit_end,
        "calibration_start": split, "calibration_rows": len(rows["train"]) - split,
        "guard_rows": len(rows["guard"]), "eval_rows": len(rows["eval"]),
        "runtime_feature_count": RUNTIME_FEATURES,
        "runtime_feature_contract": [
            "current_page_offset", "current_page_number_four_16bit_chunks",
            "signed_line_delta_from_prior", "log_absolute_line_delta_from_prior",
            "same_page_as_prior", "signed_page_delta_from_prior",
            "current_pc_four_16bit_chunks", "same_pc_as_prior",
        ],
        "source_decision_effective_external_input": ["pc", "addr"],
        "same_external_input_contract": True,
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_candidate_files_role": "normal replay and training-split action budget only",
        "nn_generates_own_target_addresses": True,
        "direct_action_output_classes": ACTION_CLASSES,
        "direct_action_encoding": "same-page target offset; fixed FILL_L2",
        "max_actions_per_callback": MAX_ACTIONS_PER_CALLBACK,
        "self_target_actions_allowed": True,
        "training_labels": "future same-page demand reuse from training stream; independent of Stride actions",
        "training_labels_use_future_rows": True,
        "evaluation_future_rows_role": "post-inference utility audit only",
        "forbidden_inputs": [
            "normal_candidates", "Stride_tracker_table", "last_stride", "cycle",
            "cache_hit", "access_type", "queue_state", "future_eval_rows_at_inference",
        ],
        "training_chunks_shuffled": False,
        "training_state_mode": "chronological_stateful_tbptt" if args.model_family == "lstm" else "three_event_causal_sliding_window",
        "training_state_carried_across_chunks": args.model_family == "lstm",
        "training_state_detached_between_chunks": args.model_family == "lstm",
        "cnn_architecture_self_test": "PASS", "causal_no_future_self_test": "PASS",
        "cnn_temporal_layers": 1 if args.model_family == "cnn" else 0,
        "cnn_kernel_size": CNN_KERNEL_SIZE if args.model_family == "cnn" else 0,
        "cnn_stride": CNN_STRIDE if args.model_family == "cnn" else 0,
        "cnn_dilation": CNN_DILATION if args.model_family == "cnn" else 0,
        "cnn_receptive_field_events": CNN_KERNEL_SIZE if args.model_family == "cnn" else 0,
        "experiment_revision": EXPERIMENT_REVISION,
        "event_logger_schema": EVENT_LOGGER_SCHEMA,
        "candidate_attachment_mode": CANDIDATE_ATTACHMENT_MODE,
        "offline_normal_entries": normal_entries, "offline_normal_triggers": normal_triggers,
        "offline_nn_entries": nn_entries, "offline_nn_triggers": nn_triggers,
        "normal_list_sha256": sha256(normal_path), "nn_list_sha256": sha256(nn_path),
        "train_history": history, "calibration_choice": calibration,
        "eval_future_use_metrics": action_metrics(selected_eval, eval_labels),
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
    }
    for role in roles:
        metadata[role + "_stream_gzip_sha256"] = sha256(stream_paths[role])
        metadata[role + "_stream_content_sha256"] = gzip_content_sha256(stream_paths[role])
        metadata[role + "_candidate_gzip_sha256"] = sha256(action_paths[role])
        metadata[role + "_candidate_content_sha256"] = gzip_content_sha256(action_paths[role])
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "PASS", "model_tag": tag, "family": args.model_family,
        "parameters": parameter_count, "threshold": calibration["threshold"],
        "offline_normal_entries": normal_entries, "offline_nn_entries": nn_entries,
    }, indent=2))


if __name__ == "__main__":
    main()
