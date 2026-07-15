#!/usr/bin/env python3
"""Threshold-free causal-CNN direct-action students for source SPP on 623.

The causal-CNN student consumes the same chronological external callback stream as source
SPP: line-aligned demand addresses and cache-fill evicted addresses, with the
callback kind preserved losslessly.  Canonicalized SPP requests are supervised
targets only; they are never inference inputs, gates, or request budgets.
"""
import argparse
import csv
import gzip
import hashlib
import json
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

from formal_NN_training.common.threshold_free_policy import (
    ADDRESS_BITS, CNN_DILATIONS, CNN_KERNEL_SIZE, CNN_RECEPTIVE_FIELD,
    CNN_STRIDE, behavior_metrics, build_model, decode, runtime_bits, score_cnn,
    self_test_cnn, targets_from_actions, train_cnn,
)


TRACE = "623.xalancbmk_s-700B"
POLICY = "spp"
PAGE_LINES = 64
FILL_LEVELS = (2, 4)
RUNTIME_FEATURES = ADDRESS_BITS + 1
EXPERIMENT_REVISION = "spp_threshold_free_fill_feedback_split_v9"
TRACK_MODEL_FAMILY = "cnn"
EVENT_LOGGER_SCHEMA = "623_causal_trigger_fill_v6"
ACTION_ATTACHMENT_MODE = "explicit_trigger_event_id"
CANONICALIZATION_MODE = "per_target_min_fill_queue_effect"
SOURCE_INPUTS = [
    "callback_kind", "invoke_prefetcher.addr", "cache_fill.evicted_addr",
]
PAIR_SPECS = {
    ("cnn", 8): ("p0", 4665),
    ("cnn", 13): ("p1", 9240),
    ("cnn", 22): ("p2", 21003),
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
    try:
        return int(text, 0)
    except ValueError:
        return int(float(text))


def load_stream(path):
    context = []
    demands = []
    demand_positions = []
    occurrences = defaultdict(int)
    last_raw_event_id = -1
    last_cycle = -1
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "trace", "event_idx", "raw_event_id", "cycle", "event_kind",
            "event_address", "event_line", "decision_idx", "pc",
            "cache_hit", "access_type", "pc_line_occ", "logger_schema",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError("{} missing {}".format(path, sorted(missing)))
        for index, row in enumerate(reader):
            raw_event_id = as_int(row["raw_event_id"])
            cycle = as_int(row["cycle"])
            kind = row["event_kind"]
            decision_idx = as_int(row["decision_idx"])
            pc = as_int(row["pc"])
            address = as_int(row["event_address"])
            line = as_int(row["event_line"])
            hit = as_int(row["cache_hit"])
            occurrence = as_int(row["pc_line_occ"])
            if (
                row["trace"] != TRACE
                or as_int(row["event_idx"]) != index
                or row["logger_schema"] != EVENT_LOGGER_SCHEMA
                or address != line << 6
                or raw_event_id <= last_raw_event_id
                or cycle < last_cycle
                or kind not in ("DEMAND", "FILL")
            ):
                raise RuntimeError(
                    "stream identity/input failure at row {}".format(index)
                )
            if kind == "DEMAND":
                expected = occurrences[(pc, line)]
                occurrences[(pc, line)] += 1
                if (
                    decision_idx != len(demands)
                    or occurrence != expected
                    or hit not in (0, 1)
                ):
                    raise RuntimeError(
                        "demand identity failure at row {}".format(index)
                    )
                demands.append((pc, address, line, occurrence))
                demand_positions.append(index)
            elif decision_idx != -1 or pc != 0 or hit != 0 or occurrence != -1:
                raise RuntimeError(
                    "cache-fill context row leaks transport fields at {}".format(index)
                )
            context.append((kind, address, line, decision_idx))
            last_raw_event_id = raw_event_id
            last_cycle = cycle
    if not context or not demands:
        raise RuntimeError("empty stream {}".format(path))
    if len(context) == len(demands):
        raise RuntimeError("SPP stream contains no cache-fill feedback")
    return {
        "context": context,
        "demands": demands,
        "demand_positions": np.asarray(demand_positions, dtype=np.int64),
    }


def load_teacher_actions(path, rows):
    actions = [[] for _ in rows]
    last_pf_event = -1
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "trace", "policy", "demand_idx", "pc", "line", "pc_line_occ",
            "action_rank", "pf_line", "target_page_offset", "fill_level",
            "accepted", "duplicate", "trigger_event_id", "pf_event_id",
            "event_distance", "raw_action_count", "source_first_pf_event_id",
            "source_last_pf_event_id", "is_self_target", "canonicalization",
            "match_mode", "logger_schema",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError("{} missing {}".format(path, sorted(missing)))
        for row in reader:
            index = as_int(row["demand_idx"])
            if index < 0 or index >= len(rows):
                raise RuntimeError("teacher action demand_idx out of range")
            pc, _, line, occurrence = rows[index]
            identity = (
                as_int(row["pc"]), as_int(row["line"]),
                as_int(row["pc_line_occ"]),
            )
            if (
                row["trace"] != TRACE
                or row["policy"] != POLICY
                or identity != (pc, line, occurrence)
                or row["logger_schema"] != EVENT_LOGGER_SCHEMA
                or row["match_mode"] != ACTION_ATTACHMENT_MODE
            ):
                raise RuntimeError(
                    "teacher action identity failure at {}".format(index)
                )
            if as_int(row["action_rank"]) != len(actions[index]) + 1:
                raise RuntimeError(
                    "noncontiguous action rank at {}".format(index)
                )
            pf_event = as_int(row["pf_event_id"])
            trigger = as_int(row["trigger_event_id"])
            distance = as_int(row["event_distance"])
            if (
                pf_event <= last_pf_event or trigger >= pf_event
                or distance != pf_event - trigger
            ):
                raise RuntimeError(
                    "invalid action attachment at {}".format(index)
                )
            pf_line = as_int(row["pf_line"])
            offset = as_int(row["target_page_offset"])
            fill = as_int(row["fill_level"])
            raw_count = as_int(row["raw_action_count"])
            source_first = as_int(row["source_first_pf_event_id"])
            source_last = as_int(row["source_last_pf_event_id"])
            is_self = as_int(row["is_self_target"])
            if (
                pf_line // PAGE_LINES != line // PAGE_LINES
                or offset != pf_line % PAGE_LINES
                or fill not in FILL_LEVELS
                or as_int(row["accepted"]) != 1
                or as_int(row["duplicate"]) not in (0, 1)
                or raw_count < 1
                or source_first != pf_event
                or source_last < source_first
                or is_self != int(pf_line == line)
                or row["canonicalization"] != CANONICALIZATION_MODE
            ):
                raise RuntimeError(
                    "invalid direct SPP action at {}".format(index)
                )
            if any(existing_line == pf_line for existing_line, _ in actions[index]):
                raise RuntimeError(
                    "two fill choices for one target at {}".format(index)
                )
            actions[index].append((pf_line, fill))
            last_pf_event = pf_event
    if not any(actions):
        raise RuntimeError("empty teacher action stream {}".format(path))
    return actions


def runtime_array(stream):
    context = stream["context"]
    addresses = runtime_bits(
        [0 for _ in context], [line for _, _, line, _ in context], False
    )
    # One bit losslessly identifies the two source callback entry points.
    kinds = np.asarray(
        [[1.0 if kind == "DEMAND" else 0.0] for kind, _, _, _ in context],
        dtype=np.float32,
    )
    return np.concatenate([addresses, kinds], axis=1)


def expand_targets_to_context(stream, decision_targets):
    counts, targets, fills = decision_targets
    positions = stream["demand_positions"]
    if len(positions) != len(counts):
        raise RuntimeError("demand positions and decision targets differ")
    context_count = len(stream["context"])
    expanded_counts = np.full(context_count, -1, dtype=np.int64)
    expanded_targets = np.zeros((context_count, PAGE_LINES), dtype=np.float32)
    expanded_fills = np.full((context_count, PAGE_LINES), -1, dtype=np.int64)
    expanded_counts[positions] = counts
    expanded_targets[positions] = targets
    expanded_fills[positions] = fills
    return expanded_counts, expanded_targets, expanded_fills


def write_table(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_teacher_replay(path, rows, actions):
    entries = 0
    triggers = 0
    fill_counts = {"FILL_L2": 0, "FILL_LLC": 0}
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pc", "line", "occ", "prefetch_addr", "fill_level"])
        for (pc, _, line, occurrence), items in zip(rows, actions):
            if items:
                triggers += 1
            for pf_line, fill in items:
                writer.writerow([
                    pc, line, occurrence, "0x{:x}".format(pf_line << 6), fill,
                ])
                fill_counts["FILL_L2" if fill == 2 else "FILL_LLC"] += 1
                entries += 1
    return entries, triggers, fill_counts


def write_prediction_replay(
    path, rows, selected, fill_choice, target_logits,
):
    entries = 0
    triggers = 0
    fill_counts = {"FILL_L2": 0, "FILL_LLC": 0}
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pc", "line", "occ", "prefetch_addr", "fill_level"])
        for index, (pc, _, line, occurrence) in enumerate(rows):
            offsets = np.flatnonzero(selected[index])
            if offsets.size:
                triggers += 1
                offsets = offsets[
                    np.argsort(-target_logits[index, offsets], kind="stable")
                ]
            page_base = (line // PAGE_LINES) * PAGE_LINES
            for offset in offsets:
                fill = FILL_LEVELS[int(fill_choice[index, offset])]
                pf_line = page_base + int(offset)
                writer.writerow([
                    pc, line, occurrence, "0x{:x}".format(pf_line << 6), fill,
                ])
                fill_counts["FILL_L2" if fill == 2 else "FILL_LLC"] += 1
                entries += 1
    return entries, triggers, fill_counts


def model_tag(family, size):
    suffix = "lstm_h{}".format(size) if family == "lstm" else "cnn_c{}".format(size)
    return "threshold_free_spp_{}".format(suffix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, choices=[POLICY])
    for role in ("train", "guard", "eval"):
        parser.add_argument("--{}-stream".format(role), required=True, type=Path)
        parser.add_argument(
            "--{}-teacher-actions".format(role), required=True, type=Path
        )
    parser.add_argument("--source-contract", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--model-family", choices=[TRACK_MODEL_FAMILY], required=True
    )
    parser.add_argument("--model-size", type=int, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--chunk-len", type=int, default=1024)
    parser.add_argument("--accumulate-chunks", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.model_family != TRACK_MODEL_FAMILY:
        raise RuntimeError("this directory trains causal-CNN points only")

    source_contract = json.loads(args.source_contract.read_text())
    if (
        source_contract.get("decision_effective_external_input") != SOURCE_INPUTS
        or source_contract.get("output_action")
        != ["same-page pf_addr", "FILL_L2 or FILL_LLC"]
        or source_contract.get("self_target_action_semantics")
        != "allowed_by_source_lookahead_and_replayed"
    ):
        raise RuntimeError("unexpected SPP source contract")
    pinned = PAIR_SPECS.get((args.model_family, args.model_size))
    if pinned is None or pinned[0] != args.pair_id:
        raise RuntimeError("model family/size/pair is not a pinned matched point")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    self_test_cnn(RUNTIME_FEATURES, fill_classes=len(FILL_LEVELS))

    roles = ("train", "guard", "eval")
    stream_paths = {role: getattr(args, role + "_stream") for role in roles}
    action_paths = {
        role: getattr(args, role + "_teacher_actions") for role in roles
    }
    streams = {role: load_stream(stream_paths[role]) for role in roles}
    normal = {
        role: load_teacher_actions(
            action_paths[role], streams[role]["demands"]
        )
        for role in roles
    }
    runtime = {role: runtime_array(streams[role]) for role in roles}
    if any(value.shape[1] != RUNTIME_FEATURES for value in runtime.values()):
        raise RuntimeError("lossless callback-kind/address encoding changed shape")
    decision_targets = {}
    context_targets = {}
    for role in roles:
        decision_targets[role] = targets_from_actions(
            [line for _, _, line, _ in streams[role]["demands"]],
            normal[role], fill_levels=FILL_LEVELS,
        )
        context_targets[role] = expand_targets_to_context(
            streams[role], decision_targets[role]
        )

    model, parameter_count = build_model(
        args.model_family, RUNTIME_FEATURES, args.model_size,
        fill_classes=len(FILL_LEVELS),
    )
    if parameter_count != pinned[1]:
        raise RuntimeError(
            "pinned parameter count {} != {}".format(parameter_count, pinned[1])
        )
    train_args = (
        model, runtime["train"], *context_targets["train"],
        len(streams["train"]["context"]),
        device, args.epochs, args.chunk_len, args.accumulate_chunks,
        args.learning_rate,
    )
    history = train_cnn(*train_args)

    eval_logits = score_cnn(
        model, runtime["eval"], device, prefix_runtime=runtime["guard"]
    )
    demand_positions = streams["eval"]["demand_positions"]
    eval_logits = tuple(
        None if value is None else value[demand_positions]
        for value in eval_logits
    )
    predicted_counts, selected_eval, predicted_fills = decode(eval_logits)
    behavior = behavior_metrics(
        predicted_counts, selected_eval, predicted_fills,
        *decision_targets["eval"],
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    normal_path = args.out_dir / "offline_spp.replay.csv"
    nn_path = args.out_dir / "offline_nn.replay.csv"
    normal_entries, normal_triggers, normal_fill_counts = write_teacher_replay(
        normal_path, streams["eval"]["demands"], normal["eval"]
    )
    nn_entries, nn_triggers, nn_fill_counts = write_prediction_replay(
        nn_path, streams["eval"]["demands"], selected_eval,
        predicted_fills, eval_logits[1]
    )
    write_table(args.out_dir / "training_history.csv", history)
    torch.save({
        "state_dict": model.state_dict(),
        "model_family": args.model_family,
        "track_model_family": TRACK_MODEL_FAMILY,
        "model_size": args.model_size,
        "runtime_features": RUNTIME_FEATURES,
        "fill_levels": FILL_LEVELS,
        "experiment_revision": EXPERIMENT_REVISION,
    }, args.out_dir / "model.pt")

    tag = model_tag(args.model_family, args.model_size)
    metadata = {
        "trace": TRACE,
        "model_tag": tag,
        "matched_normal_prefetcher": POLICY,
        "neural_role": "standalone_direct_action_prefetcher",
        "model_family": args.model_family,
        "track_model_family": TRACK_MODEL_FAMILY,
        "model_size": args.model_size,
        "architecture_pair_id": args.pair_id,
        "parameter_count": parameter_count,
        "seed": args.seed,
        "epochs": args.epochs,
        "chunk_len": args.chunk_len,
        "accumulate_chunks": args.accumulate_chunks,
        "learning_rate": args.learning_rate,
        "guard_rows": len(streams["guard"]["context"]),
        "eval_rows": len(streams["eval"]["context"]),
        "guard_demand_callbacks": len(streams["guard"]["demands"]),
        "eval_demand_callbacks": len(streams["eval"]["demands"]),
        "guard_cache_fill_callbacks": (
            len(streams["guard"]["context"])
            - len(streams["guard"]["demands"])
        ),
        "eval_cache_fill_callbacks": (
            len(streams["eval"]["context"])
            - len(streams["eval"]["demands"])
        ),
        "runtime_feature_count": RUNTIME_FEATURES,
        "runtime_encoding": (
            "64 line-aligned-address bits plus one DEMAND/FILL callback-kind bit"
        ),
        "source_decision_effective_external_input": SOURCE_INPUTS,
        "model_does_not_use_pc": True,
        "pc_is_replay_transport_only": True,
        "model_input_is_causal_external_event_sequence_only": True,
        "cache_fill_feedback_used_as_raw_external_input": True,
        "cache_fill_private_state_used_as_model_input": False,
        "cache_hit_and_type_are_audit_only": True,
        "teacher_actions_are_model_inputs": False,
        "same_external_input_contract": True,
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_request_rate_used_as_budget": False,
        "normal_policy_constants_used_by_neural_inference": False,
        "nn_generates_own_target_addresses_and_fill_levels": True,
        "complete_action_space": (
            "count 0..64, ranking of 64 same-page offsets, learned L2/LLC fill"
        ),
        "decision_rule": "count_argmax_then_target_top_count_and_fill_argmax",
        "probability_threshold_used": False,
        "neural_degree_cap": None,
        "future_label_window_used": False,
        "fill_lead_cutoff_used": False,
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
        "training_labels": (
            "canonicalized source-SPP request count, target set, and fill; supervision only"
        ),
        "teacher_action_files_role": "normal replay, supervised labels, and audit only",
        "forbidden_inputs": [
            "normal_actions_at_inference", "SPP_signature_tables", "pattern_tables",
            "global_history_register_contents", "prefetch_filter_contents",
            "cycle", "cache_hit", "access_type",
            "queue_state", "future_rows",
        ],
        "training_chunks_shuffled": False,
        "training_state_mode": "causal_dilated_tcn_over_chronological_stream",
        "training_state_carried_across_chunks": False,
        "training_state_detached_between_chunks": False,
        "inference_history_mode": (
            "chronological_sliding_context_with_exact_1554_event_overlap"
        ),
        "cnn_architecture_self_test": "PASS",
        "causal_no_future_self_test": "PASS",
        "cnn_temporal_layers": len(CNN_DILATIONS),
        "cnn_kernel_size": CNN_KERNEL_SIZE,
        "cnn_stride": CNN_STRIDE,
        "cnn_dilations": list(CNN_DILATIONS),
        "cnn_receptive_field_events": CNN_RECEPTIVE_FIELD,
        "training_left_context_overlap": CNN_RECEPTIVE_FIELD - 1,
        "cnn_processes_complete_stream_in_order": True,
        "cnn_chunking_changes_visible_history": False,
        "experiment_revision": EXPERIMENT_REVISION,
        "event_logger_schema": EVENT_LOGGER_SCHEMA,
        "action_attachment_mode": ACTION_ATTACHMENT_MODE,
        "canonicalization_mode": CANONICALIZATION_MODE,
        "teacher_action_canonicalization": CANONICALIZATION_MODE,
        "replay_preserves_explicit_fill_level": True,
        "source_contract_sha256": sha256(args.source_contract),
        "offline_normal_entries": normal_entries,
        "offline_normal_triggers": normal_triggers,
        "offline_normal_fill_counts": normal_fill_counts,
        "offline_normal_fill_level_counts": normal_fill_counts,
        "offline_nn_entries": nn_entries,
        "offline_nn_triggers": nn_triggers,
        "offline_nn_fill_counts": nn_fill_counts,
        "offline_nn_fill_level_counts": nn_fill_counts,
        "normal_list_sha256": sha256(normal_path),
        "nn_list_sha256": sha256(nn_path),
        "heldout_behavior_metrics": behavior,
        "train_history": history,
        "source_contract": source_contract,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    for role in roles:
        metadata[role + "_stream_gzip_sha256"] = sha256(stream_paths[role])
        metadata[role + "_stream_content_sha256"] = gzip_content_sha256(stream_paths[role])
        metadata[role + "_teacher_actions_gzip_sha256"] = sha256(action_paths[role])
        metadata[role + "_teacher_actions_content_sha256"] = gzip_content_sha256(action_paths[role])
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "status": "PASS",
        "model_tag": tag,
        "family": args.model_family,
        "parameters": parameter_count,
        "decision_rule": metadata["decision_rule"],
        "offline_normal_entries": normal_entries,
        "offline_nn_entries": nn_entries,
    }, indent=2))


if __name__ == "__main__":
    main()
