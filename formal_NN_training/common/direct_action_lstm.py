#!/usr/bin/env python3
"""Threshold-free matched-input LSTM experiments for the 602 trace.

The normal prefetcher and neural policy receive the same effective external
inputs.  Normal-policy requests are supervised targets and the normal replay
baseline; they are never neural inputs, gates, thresholds, budgets, or private
state.  The LSTM learns request count and target ranking directly.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from formal_NN_training.common.threshold_free_policy import (
    ADDRESS_BITS, CountRankLSTM, behavior_metrics, build_model, decode,
    runtime_bits, score_lstm, targets_from_actions, train_lstm,
)
from formal_NN_training.common.normal_policy_reference import (
    POLICY_USES_PC, ampm_actions, normal_actions, policy_self_test,
    streamer_actions, stride_actions,
)


TRACE = "602.gcc_s-734B"
PAGE_LINES = 64
EXPERIMENT_REVISION = "threshold_free_count_rank_v5"


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
            expected = occurrences[(pc, line)]
            occurrences[(pc, line)] += 1
            if (
                row["trace"] != TRACE
                or as_int(row["demand_idx"]) != index
                or occurrence != expected
            ):
                raise RuntimeError(
                    "stream identity/ordering failure at row {}".format(index)
                )
            rows.append((pc, line, occurrence))
    if not rows:
        raise RuntimeError("empty stream: {}".format(path))
    return rows


def runtime_features(policy, rows):
    return runtime_bits(
        [pc for pc, _, _ in rows],
        [line for _, line, _ in rows],
        POLICY_USES_PC[policy],
    )


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
                writer.writerow(
                    [pc, line, occurrence, "0x{:x}".format(int(target) * 64)]
                )
                entries += 1
    return entries, triggers


def write_nn_replay(path, rows, selected, target_logits):
    entries = 0
    triggers = 0
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pc", "line", "occ", "prefetch_addr"])
        for index, (pc, line, occurrence) in enumerate(rows):
            offsets = np.flatnonzero(selected[index])
            if offsets.size:
                triggers += 1
                offsets = offsets[
                    np.argsort(-target_logits[index, offsets], kind="stable")
                ]
            page_base = (line // PAGE_LINES) * PAGE_LINES
            for offset in offsets:
                writer.writerow([
                    pc, line, occurrence,
                    "0x{:x}".format((page_base + int(offset)) * 64),
                ])
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
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-size", type=int, default=8)
    return parser


def run_cli(policy: str):
    if policy not in POLICY_USES_PC:
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
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_stream(args.train_stream)
    eval_rows = load_stream(args.eval_stream)
    guard_rows = load_stream(args.guard_stream) if policy == "ampm" else []
    train_runtime = runtime_features(policy, train_rows)
    train_normal, _ = normal_actions(policy, train_rows)
    train_counts, train_targets, train_fills = targets_from_actions(
        [line for _, line, _ in train_rows], train_normal
    )

    feature_count = train_runtime.shape[1]
    expected_features = ADDRESS_BITS * (2 if POLICY_USES_PC[policy] else 1)
    if feature_count != expected_features:
        raise RuntimeError("lossless source-input feature count mismatch")
    model, parameters = build_model(
        "lstm", feature_count, args.hidden_size, fill_classes=0
    )
    history = train_lstm(
        model, train_runtime, train_counts, train_targets, train_fills,
        len(train_rows), device, args.epochs, args.chunk_len,
        args.batch_chunks, args.learning_rate,
    )

    normal_state = None
    nn_state = None
    if guard_rows:
        guard_runtime = runtime_features(policy, guard_rows)
        _, nn_state = score_lstm(model, guard_runtime, device)
        _, normal_state = normal_actions(policy, guard_rows)
    eval_runtime = runtime_features(policy, eval_rows)
    eval_logits, _ = score_lstm(
        model, eval_runtime, device, initial_state=nn_state
    )
    eval_normal, _ = normal_actions(policy, eval_rows, normal_state)
    eval_counts, eval_targets, eval_fills = targets_from_actions(
        [line for _, line, _ in eval_rows], eval_normal
    )
    predicted_counts, selected_eval, predicted_fills = decode(eval_logits)
    imitation = behavior_metrics(
        predicted_counts, selected_eval, predicted_fills,
        eval_counts, eval_targets, eval_fills,
    )

    normal_path = args.out_dir / "offline_{}.replay.csv".format(policy)
    nn_path = args.out_dir / "offline_lstm.replay.csv"
    normal_entries, normal_triggers = write_normal_replay(
        normal_path, eval_rows, eval_normal
    )
    nn_entries, nn_triggers = write_nn_replay(
        nn_path, eval_rows, selected_eval, eval_logits[1]
    )

    torch.save({
        "state_dict": model.cpu().state_dict(),
        "parameters": parameters,
        "hidden_size": args.hidden_size,
        "feature_count": feature_count,
        "trace": TRACE,
        "matched_normal_prefetcher": policy,
        "experiment_revision": EXPERIMENT_REVISION,
    }, args.out_dir / "model.pt")
    write_table(args.out_dir / "training_history.csv", history)

    paths = {"train": args.train_stream, "eval": args.eval_stream}
    if guard_rows:
        paths["guard"] = args.guard_stream
    metadata = {
        "trace": TRACE,
        "matched_normal_prefetcher": policy,
        "model_family": "LSTM count-rank direct-action",
        "neural_role": "standalone_direct_action_prefetcher",
        "parameter_count": parameters,
        "hidden_size": args.hidden_size,
        "runtime_feature_count": feature_count,
        "runtime_encoding": "lossless_lsb_first_binary_u64",
        "seed": args.seed,
        "same_external_input_contract": True,
        "effective_external_inputs": (
            ["pc", "cache_line_address"]
            if POLICY_USES_PC[policy] else ["cache_line_address"]
        ),
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_request_rate_used_as_budget": False,
        "normal_policy_constants_used_by_neural_inference": False,
        "nn_generates_own_target_addresses": True,
        "complete_action_space": "count 0..64 plus ranking of 64 same-page offsets",
        "decision_rule": "count_argmax_then_target_top_count",
        "probability_threshold_used": False,
        "neural_degree_cap": None,
        "future_label_window_used": False,
        "handcrafted_semantic_features_used": False,
        "manual_loss_weights_used": False,
        "training_regularization_used": False,
        "inference_policy_hardcodes_used": False,
        "threshold_related_hardcodes_used": False,
        "hardware_action_space_constants": {
            "cache_line_bytes": 64,
            "page_lines": PAGE_LINES,
        },
        "learned_request_count": True,
        "training_labels": "normal emitted request count and target set; supervision only",
        "forbidden_inputs": [
            "normal_actions_at_inference", "normal_private_tables", "hit_miss",
            "cycle", "queue_state", "future_rows",
        ],
        "training_state_mode": "chronological_stateful_tbptt",
        "training_chunks_shuffled": False,
        "training_state_carried_across_chunks": True,
        "training_state_detached_between_chunks": True,
        "training_state_reset": "only_at_epoch_start",
        "training_chunk_len": args.chunk_len,
        "optimizer_step_every_chunks": args.batch_chunks,
        "inference_state_mode": (
            "guard_then_continuous_evaluation"
            if guard_rows else "continuous_within_independent_evaluation_stream"
        ),
        "experiment_revision": EXPERIMENT_REVISION,
        "evaluation_stream_role": "causal inference and post-inference behavior audit",
        "offline_{}_entries".format(policy): normal_entries,
        "offline_{}_triggers".format(policy): normal_triggers,
        "offline_lstm_entries": nn_entries,
        "offline_lstm_triggers": nn_triggers,
        "offline_{}_list_sha256".format(policy): sha256(normal_path),
        "offline_lstm_list_sha256": sha256(nn_path),
        "heldout_behavior_metrics": imitation,
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
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print("[ok] " + json.dumps({
        "policy": policy,
        "device": str(device),
        "hidden_size": args.hidden_size,
        "parameters": parameters,
        "decision_rule": metadata["decision_rule"],
        "normal_entries": normal_entries,
        "nn_entries": nn_entries,
    }, sort_keys=True))


DirectActionLSTM = CountRankLSTM

__all__ = [
    "DirectActionLSTM", "ampm_actions", "normal_actions", "run_cli",
    "runtime_features", "streamer_actions", "stride_actions",
]
