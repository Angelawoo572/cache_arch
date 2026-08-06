#!/usr/bin/env python3
"""Re-decode the five 623 Stride v22 checkpoints with natural-prior MAP.

This is a strict decoder-only ablation.  It loads the v22 checkpoint and
training-history bytes, proves that their TRAIN-derived vocabulary, class
weights, input hashes, architecture point, and raw replay are reproducible,
then changes only the hurdle decision logits from ``z`` to ``z-log(w)``.
Teacher actions remain labels/comparator data and never enter the encoder.
"""
import argparse
import csv
import json
import math
import os
import platform
import shutil
from collections import Counter
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch

import model_contract as model_contract_module
from model_contract import (
    CAUSAL_RUNTIME_FEATURES, CHECKPOINT_SELECTION, DECODER_REVISION,
    DECODER_TRAINING_MODE, DECODING_RULE, DELTA_OBJECTIVE,
    DECODE_PER_CALLBACK_WATCHDOG, DECODE_PER_ROLE_WATCHDOG,
    EXPERIMENT_REVISION, HURDLE_OBJECTIVE, MAX_DELTA_OUTPUT_CLASSES,
    MAX_EXACT_DELTA_CLASSES, MODEL_POINTS, MODEL_REVISION, OPERATION,
    PARENT_DECODER_REVISION, PARENT_MODEL_REVISION, PARENT_RUN_ID, POLICY,
    RAW_RUNTIME_FEATURES, RANK_CODE_FEATURES, RUN_ID, RUNTIME_FEATURES,
    SOURCE_INPUTS, TRACE, TRAINING_ACCUMULATE_CHUNKS, TRAINING_CHUNK_LEN,
    TRAINING_EPOCHS, TRAINING_LEARNING_RATE, TRAINING_SEED,
    expected_parameter_count, hurdle_statistics_from_counts,
    model_points_description, model_tag, parent_model_tag,
)
from train_and_offline_infer import (
    RawHurdleCountStrideLSTM, _count_summary, behavior_metrics,
    build_delta_vocabulary, decode, delta_class_prior,
    delta_coordinate_initial_bias, gzip_content_sha256, load_stream,
    load_teacher_actions, runtime_encoder_sha256, runtime_features,
    score_suffix, sha256, state_router_sha256, trigger_metrics,
    vocabulary_statistics, write_replay,
)


ROOT = Path(__file__).resolve().parents[4]
TRAINER_SOURCE = Path(__file__).with_name("train_and_offline_infer.py")
REDECODER_SOURCE = Path(__file__).resolve()
MODEL_CONTRACT_SOURCE = Path(model_contract_module.__file__).resolve()
THRESHOLD_FREE_POLICY_SOURCE = (
    ROOT / "formal_NN_training" / "common" / "threshold_free_policy.py"
)
SOURCE_INPUT_LIST = list(SOURCE_INPUTS)


def read_json_object(path, label):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError("cannot read {} {}: {}".format(label, path, error))
    if not isinstance(value, dict):
        raise RuntimeError("{} must contain one JSON object".format(path))
    return value


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_history(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("empty parent training history {}".format(path))
    return rows


def same_float_list(left, right, tolerance=1e-12):
    if not isinstance(left, (list, tuple)) or len(left) != len(right):
        return False
    return all(
        math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)
        for a, b in zip(left, right)
    )


def verify_parent_metadata(
    metadata, parent_tag, model_size, pair_id, stream_paths, action_paths,
):
    expected = {
        "run_id": PARENT_RUN_ID,
        "trace": TRACE,
        "model_tag": parent_tag,
        "model_family": "lstm",
        "model_size": model_size,
        "architecture_pair_id": pair_id,
        "matched_normal_prefetcher": POLICY,
        "model_revision": PARENT_MODEL_REVISION,
        "decoder_revision": PARENT_DECODER_REVISION,
        "same_external_input_contract": True,
        "training_inference_input_encoder_identical": True,
        "teacher_actions_are_model_inputs": False,
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "normal_policy_request_rate_used_as_budget": False,
    }
    failures = [
        "{}={!r}, expected {!r}".format(key, metadata.get(key), value)
        for key, value in expected.items() if metadata.get(key) != value
    ]
    for role in ("train", "guard", "eval"):
        expected_hashes = {
            role + "_stream_gzip_sha256": sha256(stream_paths[role]),
            role + "_stream_content_sha256": gzip_content_sha256(
                stream_paths[role]
            ),
            role + "_candidate_gzip_sha256": sha256(action_paths[role]),
            role + "_candidate_content_sha256": gzip_content_sha256(
                action_paths[role]
            ),
        }
        for key, value in expected_hashes.items():
            if metadata.get(key) != value:
                failures.append("{} input hash differs from parent".format(key))
    if failures:
        raise RuntimeError(
            "v22 parent metadata is not the matched-input parent:\n{}".format(
                "\n".join("- " + item for item in failures)
            )
        )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, choices=[POLICY])
    for role in ("train", "guard", "eval"):
        parser.add_argument(
            "--{}-stream".format(role), required=True, type=Path
        )
        parser.add_argument(
            "--{}-candidates".format(role), required=True, type=Path
        )
    parser.add_argument("--parent-output-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-family", choices=["lstm"], required=True)
    parser.add_argument("--model-size", type=int, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--seed", type=int, default=TRAINING_SEED)
    parser.add_argument("--epochs", type=int, default=TRAINING_EPOCHS)
    parser.add_argument("--chunk-len", type=int, default=TRAINING_CHUNK_LEN)
    parser.add_argument(
        "--accumulate-chunks", type=int,
        default=TRAINING_ACCUMULATE_CHUNKS,
    )
    parser.add_argument(
        "--learning-rate", type=float, default=TRAINING_LEARNING_RATE
    )
    parser.add_argument("--device", choices=["auto", "cuda"], default="auto")
    return parser


def main():
    args = build_parser().parse_args()
    expected_pair = MODEL_POINTS["lstm"].get(args.model_size)
    if expected_pair is None or args.pair_id != expected_pair:
        raise RuntimeError("model size/pair is not a configured v23 point")
    observed_training = {
        "seed": args.seed,
        "epochs": args.epochs,
        "chunk_len": args.chunk_len,
        "accumulate_chunks": args.accumulate_chunks,
        "learning_rate": args.learning_rate,
    }
    expected_training = dict(model_points_description()["training_config"])
    if observed_training != expected_training:
        raise RuntimeError(
            "parent training config {} differs from pinned {}".format(
                observed_training, expected_training
            )
        )

    if not torch.cuda.is_available():
        raise RuntimeError("Stride v23 parent-reproduction decode requires CUDA")
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(device)
    if "A100" not in device_name:
        raise RuntimeError(
            "Stride v23 parent-reproduction decode requires A100; observed {}"
            .format(device_name)
        )
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)

    roles = ("train", "guard", "eval")
    stream_paths = {role: getattr(args, role + "_stream") for role in roles}
    action_paths = {role: getattr(args, role + "_candidates") for role in roles}
    rows = {role: load_stream(stream_paths[role]) for role in roles}
    actions = {
        role: load_teacher_actions(action_paths[role], rows[role])
        for role in roles
    }
    exact_vocabulary, train_delta_frequencies = build_delta_vocabulary(
        rows["train"], actions["train"]
    )
    counts_train = np.asarray(
        [len(items) for items in actions["train"]], dtype=np.int64
    )
    hurdle_stats = hurdle_statistics_from_counts(counts_train.tolist())
    hurdle_weights = hurdle_stats["class_weights_ZERO_POSITIVE"]
    delta_prior = delta_class_prior(
        exact_vocabulary, train_delta_frequencies
    )
    coordinate_initial_bias = delta_coordinate_initial_bias(
        rows["train"], actions["train"]
    )

    parent_tag = parent_model_tag(args.model_size)
    parent_dir = args.parent_output_root / parent_tag
    parent_metadata_path = parent_dir / "run_metadata.json"
    parent_model_path = parent_dir / "model.pt"
    parent_history_path = parent_dir / "training_history.csv"
    parent_normal_path = parent_dir / "offline_stride.replay.csv"
    parent_nn_path = parent_dir / "offline_nn.replay.csv"
    for path in (
        parent_metadata_path, parent_model_path, parent_history_path,
        parent_normal_path, parent_nn_path,
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError("missing parent v22 artifact {}".format(path))
    parent_metadata = read_json_object(parent_metadata_path, "parent metadata")
    verify_parent_metadata(
        parent_metadata, parent_tag, args.model_size, args.pair_id,
        stream_paths, action_paths,
    )
    checkpoint = load_checkpoint(parent_model_path)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("state_dict"), dict
    ):
        raise RuntimeError("parent checkpoint is not a v22 state dictionary")
    if (
        checkpoint.get("run_id") != PARENT_RUN_ID
        or checkpoint.get("model_revision") != PARENT_MODEL_REVISION
        or checkpoint.get("decoder_revision") != PARENT_DECODER_REVISION
        or checkpoint.get("model_size") != args.model_size
        or checkpoint.get("exact_delta_vocabulary")
        != [int(value) for value in exact_vocabulary]
        or not same_float_list(
            checkpoint.get("hurdle_class_weights_ZERO_POSITIVE"),
            hurdle_weights,
        )
    ):
        raise RuntimeError("parent checkpoint contract differs from reused input")

    model = RawHurdleCountStrideLSTM(
        args.model_size,
        delta_prior,
        hurdle_stats["hurdle_initial_bias_ZERO_POSITIVE"],
        hurdle_stats["positive_log_count_initial_bias"],
        coordinate_initial_bias,
    )
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("parent checkpoint state keys changed")
    model = model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    realized_delta_output_classes = len(exact_vocabulary) + 1
    expected_parameters = expected_parameter_count(
        args.model_size, realized_delta_output_classes
    )
    maximum_parameters = expected_parameter_count(args.model_size)
    if (
        parameter_count != expected_parameters
        or parent_metadata.get("parameter_count") != parameter_count
        or checkpoint.get("realized_parameter_count") != parameter_count
        or parameter_count > maximum_parameters
    ):
        raise RuntimeError("parent v22 parameter accounting changed")

    complete_rows = rows["train"] + rows["guard"] + rows["eval"]
    complete_runtime = runtime_features(complete_rows)
    eval_start = len(rows["train"]) + len(rows["guard"])
    eval_context, encoder_diagnostics = score_suffix(
        model, complete_rows, complete_runtime, device, args.chunk_len,
        eval_start,
    )
    base_lines = [line for _, line, _ in rows["eval"]]
    raw = decode(
        model, eval_context, base_lines, exact_vocabulary, device,
        DECODE_PER_CALLBACK_WATCHDOG, DECODE_PER_ROLE_WATCHDOG, "eval-parent",
        hurdle_class_weights=hurdle_weights,
        apply_hurdle_prior_correction=False,
    )
    corrected = decode(
        model, eval_context, base_lines, exact_vocabulary, device,
        DECODE_PER_CALLBACK_WATCHDOG, DECODE_PER_ROLE_WATCHDOG, "eval-v23",
        hurdle_class_weights=hurdle_weights,
        apply_hurdle_prior_correction=True,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    normal_path = args.out_dir / "offline_stride.replay.csv"
    nn_path = args.out_dir / "offline_nn.replay.csv"
    raw_reproduction_path = args.out_dir / "parent_raw_reproduction.replay.csv"
    normal_entries, normal_triggers = write_replay(
        normal_path, rows["eval"], actions["eval"]
    )
    raw_entries, raw_triggers = write_replay(
        raw_reproduction_path, rows["eval"], raw[1]
    )
    nn_entries, nn_triggers = write_replay(
        nn_path, rows["eval"], corrected[1]
    )
    if sha256(normal_path) != sha256(parent_normal_path):
        raise RuntimeError("reused input did not reproduce parent normal replay")
    if sha256(raw_reproduction_path) != sha256(parent_nn_path):
        raise RuntimeError(
            "v22 checkpoint/input did not reproduce parent raw NN replay bytes"
        )
    if (
        raw_entries != parent_metadata.get("offline_nn_entries")
        or raw_triggers != parent_metadata.get("offline_nn_triggers")
    ):
        raise RuntimeError("parent raw replay count reproduction failed")

    model_path = args.out_dir / "model.pt"
    history_path = args.out_dir / "training_history.csv"
    shutil.copy2(parent_model_path, model_path)
    shutil.copy2(parent_history_path, history_path)
    if (
        sha256(model_path) != sha256(parent_model_path)
        or sha256(history_path) != sha256(parent_history_path)
    ):
        raise RuntimeError("parent checkpoint/history byte reuse failed")
    history = read_history(history_path)

    heldout = behavior_metrics(
        corrected[0], corrected[1], corrected[2], actions["eval"]
    )
    heldout.update(trigger_metrics(corrected[0], actions["eval"]))
    heldout["request_ratio_vs_teacher"] = (
        heldout["predicted_actions"] / float(heldout["normal_actions"])
        if heldout["normal_actions"] else 0.0
    )
    raw_behavior = behavior_metrics(raw[0], raw[1], raw[2], actions["eval"])
    raw_behavior.update(trigger_metrics(raw[0], actions["eval"]))
    raw_behavior["request_ratio_vs_teacher"] = (
        raw_behavior["predicted_actions"] / float(raw_behavior["normal_actions"])
        if raw_behavior["normal_actions"] else 0.0
    )

    encoder_hash = runtime_encoder_sha256()
    router_hash = state_router_sha256()
    active_contract = model_points_description()
    tag = model_tag(args.model_size)
    source_hashes = {
        "trainer_source_sha256": sha256(TRAINER_SOURCE),
        "redecoder_source_sha256": sha256(REDECODER_SOURCE),
        "model_contract_source_sha256": sha256(MODEL_CONTRACT_SOURCE),
        "threshold_free_policy_source_sha256": sha256(
            THRESHOLD_FREE_POLICY_SOURCE
        ),
    }
    role_vocabulary_stats = {
        role: vocabulary_statistics(rows[role], actions[role], exact_vocabulary)
        for role in roles
    }
    train_unique_pc_count = len({pc for pc, _, _ in rows["train"]})
    complete_unique_pc_count = len({pc for pc, _, _ in complete_rows})

    metadata = dict(parent_metadata)
    metadata.update({
        "run_id": RUN_ID,
        "operation": OPERATION,
        "model_tag": tag,
        "model_revision": MODEL_REVISION,
        "decoder_revision": DECODER_REVISION,
        "model_point_contract": active_contract,
        "weights_retrained": False,
        "checkpoint_reused": True,
        "training_history_reused": True,
        "decoder_only_change": True,
        "parent_run_id": PARENT_RUN_ID,
        "parent_model_tag": parent_tag,
        "parent_model_revision": PARENT_MODEL_REVISION,
        "parent_decoder_revision": PARENT_DECODER_REVISION,
        "parent_run_metadata_sha256": sha256(parent_metadata_path),
        "parent_model_checkpoint_sha256": sha256(parent_model_path),
        "parent_training_history_sha256": sha256(parent_history_path),
        "parent_normal_list_sha256": sha256(parent_normal_path),
        "parent_nn_list_sha256": sha256(parent_nn_path),
        "parent_raw_reproduction_sha256": sha256(raw_reproduction_path),
        "parent_raw_reproduction_matches": True,
        "parent_checkpoint_state_dict_reloaded": True,
        "parent_artifact_identity_required": True,
        "training_config": active_contract["training_config"],
        "training_config_pinned_by_run_id": True,
        "redecode_device": str(device),
        "redecode_device_name": device_name,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "parameter_count": parameter_count,
        "realized_parameter_count": parameter_count,
        "expected_parameter_count": expected_parameters,
        "expected_realized_parameter_count": expected_parameters,
        "maximum_parameter_count": maximum_parameters,
        "realized_parameter_count_matches_formula": True,
        "realized_parameter_count_within_maximum": True,
        "parameter_formula": active_contract["parameter_formula"],
        "runtime_feature_count": RUNTIME_FEATURES,
        "raw_runtime_feature_count": RAW_RUNTIME_FEATURES,
        "causal_runtime_feature_count": CAUSAL_RUNTIME_FEATURES,
        "runtime_encoder_sha256": encoder_hash,
        "training_runtime_encoder_sha256": encoder_hash,
        "inference_runtime_encoder_sha256": encoder_hash,
        "training_state_router_sha256": router_hash,
        "inference_state_router_sha256": router_hash,
        "train_unique_pc_count": train_unique_pc_count,
        "history_unique_pc_count": complete_unique_pc_count,
        "peak_inference_recurrent_state_bytes_float32": (
            complete_unique_pc_count * 2 * args.model_size * 4
        ),
        "peak_persistent_recurrent_state_bytes": (
            complete_unique_pc_count * 2 * args.model_size * 4
        ),
        "decoder_training_mode": DECODER_TRAINING_MODE,
        "hurdle_training_objective": HURDLE_OBJECTIVE,
        "hurdle_class_weights_ZERO_POSITIVE": hurdle_weights,
        "hurdle_training_statistics": hurdle_stats,
        "hurdle_decoding_rule": (
            "deterministic_prior_corrected_two_class_argmax"
        ),
        "hurdle_prior_correction_at_decode_used": True,
        "hurdle_prior_correction_rule": (
            "weighted_logits_minus_log_TRAIN_inverse_frequency_class_weight"
        ),
        "probability_threshold_used": False,
        "threshold_related_hardcodes_used": False,
        "inference_policy_hardcodes_used": False,
        "normal_policy_request_rate_used_as_budget": False,
        "normal_policy_templates_used_by_neural_inference": False,
        "neural_degree_cap": None,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "checkpoint_selection_roles": ["parent_v22_guard_selection"],
        "checkpoint_selection_primary_role": "parent_v22_guard",
        "guard_selection_composite_or_mean_used": False,
        "guard_role": "parent_v22_guard_selection_reused_no_v23_reselection",
        "evaluation_used_for_checkpoint_selection": False,
        "evaluation_decode_passes": 2,
        "parent_raw_reproduction_decode_passes": 1,
        "prior_corrected_evaluation_decode_passes": 1,
        "decision_rule": DECODING_RULE,
        "delta_training_objective": DELTA_OBJECTIVE,
        "delta_vocabulary_statistics": role_vocabulary_stats,
        "normal_list_sha256": sha256(normal_path),
        "nn_list_sha256": sha256(nn_path),
        "model_checkpoint_sha256": sha256(model_path),
        "training_history_sha256": sha256(history_path),
        "offline_normal_entries": normal_entries,
        "offline_normal_triggers": normal_triggers,
        "offline_nn_entries": nn_entries,
        "offline_nn_triggers": nn_triggers,
        "heldout_behavior_metrics": heldout,
        "parent_raw_heldout_behavior_metrics": raw_behavior,
        "hurdle_count_decoder_diagnostics": corrected[3],
        "parent_raw_decoder_diagnostics": raw[3],
        "encoder_diagnostics": encoder_diagnostics,
        "train_action_summary": _count_summary(actions["train"]),
        "guard_action_summary": _count_summary(actions["guard"]),
        "eval_action_summary": _count_summary(actions["eval"]),
        "train_history": parent_metadata.get("train_history", history),
        "experiment_revision": EXPERIMENT_REVISION,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        **source_hashes
    })
    for role in roles:
        metadata[role + "_stream_gzip_sha256"] = sha256(stream_paths[role])
        metadata[role + "_stream_content_sha256"] = gzip_content_sha256(
            stream_paths[role]
        )
        metadata[role + "_candidate_gzip_sha256"] = sha256(action_paths[role])
        metadata[role + "_candidate_content_sha256"] = gzip_content_sha256(
            action_paths[role]
        )

    metadata_path = args.out_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "PASS",
        "model_tag": tag,
        "parent_model_tag": parent_tag,
        "parameters": parameter_count,
        "checkpoint_reused_byte_for_byte": True,
        "training_history_reused_byte_for_byte": True,
        "parent_raw_replay_reproduced_byte_for_byte": True,
        "raw_parent_actions": raw_entries,
        "prior_corrected_actions": nn_entries,
    }, indent=2))


if __name__ == "__main__":
    main()
