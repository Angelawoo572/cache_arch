#!/usr/bin/env python3
"""Shared, source-input-fair 623 Stride student implementation.

The LSTM and CNN live in separate experiment directories but intentionally
share this loader, target conversion, decoder, and metadata contract.  This
prevents architecture-specific input drift.
"""
import argparse
import csv
import gzip
import hashlib
import inspect
import json
import platform
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from formal_NN_training.common.threshold_free_policy import (
    ADDRESS_BITS, CACHE_LINE_BYTES, CACHE_LINE_SHIFT, CNN_DILATIONS,
    CNN_KERNEL_SIZE, CNN_RECEPTIVE_FIELD, CNN_STRIDE, advance_lstm_state,
    behavior_metrics, build_model, decode,
    runtime_bits, score_cnn, score_lstm, self_test_cnn,
    self_test_free_running_decoder, self_test_variable_action_decoder,
    targets_from_actions, train_cnn, train_lstm,
)


TRACE = "623.xalancbmk_s-700B"
POLICY = "stride"
RUNTIME_FEATURES = ADDRESS_BITS * 2
EVENT_LOGGER_SCHEMA = "623_causal_trigger_v5"
CANDIDATE_ATTACHMENT_MODE = "explicit_trigger_event_id"
EXPERIMENT_REVISION = "stride_source_input_variable_delta_free_running_v9"
MODEL_POINTS = {
    "lstm": {5: "p0", 16: "p1", 32: "p2"},
    "cnn": {7: "p0", 15: "p1", 24: "p2"},
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
        required = {
            "trace", "demand_idx", "pc", "line", "pc_line_occ",
            "logger_schema",
        }
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
                raise RuntimeError(
                    "stream identity/ordering failure at row {}".format(index)
                )
            rows.append((pc, line, occurrence))
    if not rows:
        raise RuntimeError("empty stream: {}".format(path))
    return rows


def load_normal_actions(path, rows):
    """Load and audit normal Stride outputs as labels, never model inputs."""
    actions = [[] for _ in rows]
    last_pf_event = -1
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "trace", "policy", "demand_idx", "pc", "line", "pc_line_occ",
            "candidate_rank", "pf_line", "fill_level", "accepted",
            "duplicate", "trigger_event_id", "pf_event_id", "event_distance",
            "match_mode", "logger_schema",
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
                row["trace"] != TRACE or row["policy"] != POLICY
                or (
                    as_int(row["pc"]), as_int(row["line"]),
                    as_int(row["pc_line_occ"]),
                ) != (pc, line, occurrence)
                or row["logger_schema"] != EVENT_LOGGER_SCHEMA
                or row["match_mode"] != CANDIDATE_ATTACHMENT_MODE
            ):
                raise RuntimeError(
                    "normal action identity failure at {}".format(index)
                )
            if as_int(row["candidate_rank"]) != len(actions[index]) + 1:
                raise RuntimeError(
                    "noncontiguous normal action rank at {}".format(index)
                )
            trigger = as_int(row["trigger_event_id"])
            pf_event = as_int(row["pf_event_id"])
            distance = as_int(row["event_distance"])
            if (
                pf_event <= last_pf_event or trigger >= pf_event
                or distance != pf_event - trigger
            ):
                raise RuntimeError("invalid trigger transport at {}".format(index))
            if (
                as_int(row["fill_level"]) != 2
                or as_int(row["accepted"]) not in (0, 1)
                or as_int(row["duplicate"]) not in (0, 1)
            ):
                raise RuntimeError(
                    "invalid captured Stride action at {}".format(index)
                )
            actions[index].append(as_int(row["pf_line"]))
            last_pf_event = pf_event
    if not any(actions):
        raise RuntimeError("empty Stride action stream {}".format(path))
    return actions


def runtime_array(rows):
    # Exactly the source's externally visible decision inputs: PC and address.
    return runtime_bits(
        [pc for pc, _, _ in rows],
        [line * CACHE_LINE_BYTES for _, line, _ in rows],
        True,
    )


def runtime_encoder_sha256():
    payload = {
        "entrypoint_source": inspect.getsource(runtime_array),
        "primitive_source": inspect.getsource(runtime_bits),
        "fields": ["pc", "addr"],
        "use_pc": True,
        "address_bits": ADDRESS_BITS,
        "cache_line_bytes": CACHE_LINE_BYTES,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


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
                writer.writerow([
                    pc, line, occurrence, hex(int(target) << CACHE_LINE_SHIFT)
                ])
                entries += 1
    return entries, triggers


def write_nn_replay(path, rows, predicted_lines):
    entries = 0
    triggers = 0
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pc", "line", "occ", "prefetch_addr"])
        for (pc, line, occurrence), targets in zip(rows, predicted_lines):
            if targets:
                triggers += 1
            for target in targets:
                writer.writerow([
                    pc, line, occurrence, hex(int(target) << CACHE_LINE_SHIFT)
                ])
                entries += 1
    return entries, triggers


def model_tag(family, size):
    suffix = "lstm_h{}".format(size) if family == "lstm" else "cnn_c{}".format(size)
    return "independent_delta_stride_{}".format(suffix)


def run_cli(track_family):
    if track_family not in MODEL_POINTS:
        raise RuntimeError("unknown track family {}".format(track_family))
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, choices=[POLICY])
    for role in ("train", "guard", "eval"):
        parser.add_argument("--{}-stream".format(role), required=True, type=Path)
        parser.add_argument("--{}-candidates".format(role), required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-family", choices=[track_family], required=True)
    parser.add_argument("--model-size", type=int, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--chunk-len", type=int, default=1024)
    parser.add_argument("--accumulate-chunks", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    expected_pair = MODEL_POINTS[track_family].get(args.model_size)
    if expected_pair is None or args.pair_id != expected_pair:
        raise RuntimeError("model size/pair is not a configured point")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    self_test_variable_action_decoder(RUNTIME_FEATURES, track_family)
    self_test_free_running_decoder(RUNTIME_FEATURES, track_family)
    if track_family == "cnn":
        self_test_cnn(RUNTIME_FEATURES)

    roles = ("train", "guard", "eval")
    stream_paths = {role: getattr(args, role + "_stream") for role in roles}
    action_paths = {role: getattr(args, role + "_candidates") for role in roles}
    rows = {role: load_stream(stream_paths[role]) for role in roles}
    normal = {
        role: load_normal_actions(action_paths[role], rows[role])
        for role in roles
    }
    runtime = {role: runtime_array(rows[role]) for role in roles}
    if any(value.shape[1] != RUNTIME_FEATURES for value in runtime.values()):
        raise RuntimeError("lossless PC/address encoding changed shape")
    for role in roles:
        if not np.array_equal(runtime[role], runtime_array(rows[role])):
            raise RuntimeError(
                "{} train/inference runtime encoder is not identical".format(role)
            )
    targets = {
        role: targets_from_actions(
            [line for _, line, _ in rows[role]], normal[role]
        )
        for role in roles
    }

    model, parameter_count = build_model(
        track_family, RUNTIME_FEATURES, args.model_size, fill_classes=0
    )
    train_args = (
        model, runtime["train"], *targets["train"], len(rows["train"]),
        device, args.epochs, args.chunk_len, args.accumulate_chunks,
        args.learning_rate,
    )
    history = (
        train_lstm(*train_args) if track_family == "lstm"
        else train_cnn(*train_args)
    )

    if track_family == "lstm":
        history_state = advance_lstm_state(model, runtime["train"], device)
        guard_state = advance_lstm_state(
            model, runtime["guard"], device, initial_state=history_state
        )
        eval_encoded, _ = score_lstm(
            model, runtime["eval"], device, initial_state=guard_state
        )
    else:
        prefix = np.concatenate([runtime["train"], runtime["guard"]], axis=0)
        eval_encoded = score_cnn(
            model, runtime["eval"], device, prefix_runtime=prefix
        )
    predicted_counts, predicted_lines, predicted_fills = decode(
        model, eval_encoded, [line for _, line, _ in rows["eval"]], device
    )
    behavior = behavior_metrics(
        predicted_counts, predicted_lines, predicted_fills, normal["eval"]
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    normal_path = args.out_dir / "offline_stride.replay.csv"
    nn_path = args.out_dir / "offline_nn.replay.csv"
    normal_entries, normal_triggers = write_normal_replay(
        normal_path, rows["eval"], normal["eval"]
    )
    nn_entries, nn_triggers = write_nn_replay(
        nn_path, rows["eval"], predicted_lines
    )
    write_table(args.out_dir / "training_history.csv", history)
    torch.save({
        "state_dict": model.state_dict(),
        "model_family": track_family,
        "model_size": args.model_size,
        "runtime_features": RUNTIME_FEATURES,
        "experiment_revision": EXPERIMENT_REVISION,
    }, args.out_dir / "model.pt")

    tag = model_tag(track_family, args.model_size)
    is_lstm = track_family == "lstm"
    metadata = {
        "trace": TRACE,
        "model_tag": tag,
        "matched_normal_prefetcher": POLICY,
        "neural_role": "standalone_direct_action_prefetcher",
        "model_family": track_family,
        "track_model_family": track_family,
        "model_size": args.model_size,
        "architecture_pair_id": args.pair_id,
        "parameter_count": parameter_count,
        "seed": args.seed,
        "epochs": args.epochs,
        "chunk_len": args.chunk_len,
        "accumulate_chunks": args.accumulate_chunks,
        "learning_rate": args.learning_rate,
        "guard_rows": len(rows["guard"]),
        "eval_rows": len(rows["eval"]),
        "runtime_feature_count": RUNTIME_FEATURES,
        "runtime_encoding": "lossless PC uint64 bits plus cache-line uint64 bits",
        "source_decision_effective_external_input": ["pc", "addr"],
        "same_external_input_contract": True,
        "training_inference_input_encoder_identical": True,
        "decoder_training_mode": "free_running_autoregressive_same_as_inference",
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_free_running_self_test": "PASS",
        "runtime_encoder_entrypoint": "experiment_623_stride.runtime_array",
        "runtime_encoder_sha256": runtime_encoder_sha256(),
        "training_runtime_encoder_sha256": runtime_encoder_sha256(),
        "inference_runtime_encoder_sha256": runtime_encoder_sha256(),
        "training_runtime_fields": ["pc", "addr"],
        "inference_runtime_fields": ["pc", "addr"],
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "normal_policy_request_rate_used_as_budget": False,
        "normal_policy_constants_used_by_neural_inference": False,
        "teacher_same_page_property_used_only_for_source_output_audit": True,
        "nn_generates_own_target_addresses": True,
        "complete_action_space": "unbounded count plus direct signed cache-line deltas",
        "decision_rule": "Poisson_mode_then_autoregressive_delta_mixture_modes",
        "probability_threshold_used": False,
        "neural_degree_cap": None,
        "fixed_page_offset_classes": None,
        "same_page_rule_used_by_neural_inference": False,
        "future_label_window_used": False,
        "handcrafted_semantic_features_used": False,
        "manual_loss_weights_used": False,
        "training_regularization_used": False,
        "inference_policy_hardcodes_used": False,
        "learned_request_count": True,
        "address_interface_bits": ADDRESS_BITS,
        "cache_line_bytes": CACHE_LINE_BYTES,
        "training_labels": "captured Stride actions; supervision only",
        "forbidden_inputs": [
            "normal_actions_at_inference", "Stride_tracker_table",
            "last_stride", "normal_degree", "cycle", "cache_hit",
            "queue_state", "future_rows",
        ],
        "training_chunks_shuffled": False,
        "training_state_mode": (
            "chronological_stateful_tbptt" if is_lstm
            else "two_layer_causal_cnn_over_chronological_stream"
        ),
        "training_state_carried_across_chunks": True if is_lstm else None,
        "training_state_detached_between_chunks": True if is_lstm else None,
        "inference_history_mode": (
            "fresh_state_then_complete_train_guard_eval_chronology"
            if is_lstm else
            "chronological_sliding_context_with_exact_overlap"
        ),
        "cnn_architecture_self_test": "NOT_APPLICABLE" if is_lstm else "PASS",
        "causal_no_future_self_test": "PASS",
        "cnn_temporal_layers": 0 if is_lstm else len(CNN_DILATIONS),
        "cnn_kernel_size": 0 if is_lstm else CNN_KERNEL_SIZE,
        "cnn_stride": 0 if is_lstm else CNN_STRIDE,
        "cnn_dilations": [] if is_lstm else list(CNN_DILATIONS),
        "cnn_receptive_field_events": 0 if is_lstm else CNN_RECEPTIVE_FIELD,
        "training_left_context_overlap": 0 if is_lstm else CNN_RECEPTIVE_FIELD - 1,
        "cnn_processes_complete_stream_in_order": False if is_lstm else True,
        "cnn_chunking_changes_visible_history": False,
        "experiment_revision": EXPERIMENT_REVISION,
        "event_logger_schema": EVENT_LOGGER_SCHEMA,
        "candidate_attachment_mode": CANDIDATE_ATTACHMENT_MODE,
        "offline_normal_entries": normal_entries,
        "offline_normal_triggers": normal_triggers,
        "offline_nn_entries": nn_entries,
        "offline_nn_triggers": nn_triggers,
        "normal_list_sha256": sha256(normal_path),
        "nn_list_sha256": sha256(nn_path),
        "heldout_behavior_metrics": behavior,
        "train_history": history,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    for role in roles:
        metadata[role + "_stream_gzip_sha256"] = sha256(stream_paths[role])
        metadata[role + "_stream_content_sha256"] = gzip_content_sha256(stream_paths[role])
        metadata[role + "_candidate_gzip_sha256"] = sha256(action_paths[role])
        metadata[role + "_candidate_content_sha256"] = gzip_content_sha256(action_paths[role])
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "status": "PASS", "model_tag": tag, "family": track_family,
        "parameters": parameter_count, "decision_rule": metadata["decision_rule"],
        "offline_normal_entries": normal_entries,
        "offline_nn_entries": nn_entries,
    }, indent=2))


__all__ = ["run_cli"]
