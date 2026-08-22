#!/usr/bin/env python3
"""Export a complete frozen Stride LSTM checkpoint to model.bin."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
from live_model_format import (
    ENDIAN_MARKER,
    FLOAT_TYPE_FLOAT32,
    FORMAT_VERSION,
    MAGIC,
    TENSOR_ORDER,
    write_model,
)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            universal_newlines=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_json(path):
    return json.loads(Path(path).read_text()) if path else {}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--run-metadata", required=True, type=Path)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--evaluation-stream", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--hidden-size", required=True, type=int, choices=[8, 16])
    parser.add_argument("--instruction-budget", required=True, type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from formal_NN_training.common.stride_direct_action_model import (
        AUTHORITATIVE_SOURCE,
        EXPERIMENT_REVISION,
        MODEL_REVISION,
        authoritative_source_sha256,
        load_checkpoint,
        runtime_encoder_sha256,
        sha256,
    )
    model_path = args.out_dir / "model.bin"
    metadata_path = args.out_dir / "model_metadata.json"
    if not args.force and (model_path.exists() or metadata_path.exists()):
        raise RuntimeError(
            "export exists; validate it or pass --force explicitly"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_checkpoint(
        args.checkpoint, expected_hidden_size=args.hidden_size
    )
    run_metadata = load_json(args.run_metadata)
    training = load_json(args.training_manifest)
    parameter_count = write_model(
        model_path, model.state_dict(), model.hidden_size, model.feature_count
    )
    if parameter_count != checkpoint["parameter_count"]:
        raise RuntimeError("exported parameter count changed")
    tensor_shapes = {
        name: list(model.state_dict()[name].shape) for name in TENSOR_ORDER
    }
    metadata = {
        "schema_version": 1,
        "magic": MAGIC.decode("ascii"),
        "format_version": FORMAT_VERSION,
        "endianness": "little",
        "endianness_marker": ENDIAN_MARKER,
        "float_type": "float32",
        "float_type_code": FLOAT_TYPE_FLOAT32,
        "tensor_names": TENSOR_ORDER,
        "tensor_shapes": tensor_shapes,
        "tensor_order": TENSOR_ORDER,
        "hidden_size": args.hidden_size,
        "feature_width": int(model.feature_count),
        "parameter_count": int(parameter_count),
        "weight_bytes": int(parameter_count) * 4,
        "recurrent_state_bytes_per_pc": args.hidden_size * 2 * 4,
        "model_revision": MODEL_REVISION,
        "experiment_revision": EXPERIMENT_REVISION,
        "runtime_encoder_sha256": runtime_encoder_sha256(),
        "python_source": str(AUTHORITATIVE_SOURCE.relative_to(ROOT)),
        "python_source_sha256": authoritative_source_sha256(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "source_run_metadata_sha256": sha256(args.run_metadata),
        "training_stream_sha256": (
            training.get("training_stream_sha256")
            or run_metadata.get("train_stream_sha256")
        ),
        "evaluation_stream_sha256": sha256(args.evaluation_stream),
        "export_sha256": sha256(model_path),
        "git_commit_sha": git_commit(),
        "seed": args.seed,
        "instruction_budget": args.instruction_budget,
        "teacher_role": "offline_labels_only",
        "weights_frozen": True,
        "online_learning": False,
        "optimizer_in_live_runtime": False,
        "backpropagation_in_live_runtime": False,
        "heldout_behavior_metrics": run_metadata.get(
            "heldout_behavior_metrics"
        ),
        "training_runtime_fields": run_metadata.get(
            "training_runtime_fields"
        ),
        "inference_runtime_fields": run_metadata.get(
            "inference_runtime_fields"
        ),
        "normal_policy_outputs_used_as_training_targets": (
            run_metadata.get(
                "normal_policy_outputs_used_as_training_targets"
            )
        ),
        "normal_policy_outputs_used_as_model_inputs": (
            run_metadata.get(
                "normal_policy_outputs_used_as_model_inputs"
            )
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    point_metadata_path = args.out_dir.parent / "point_metadata.json"
    if point_metadata_path.is_file():
        point = json.loads(point_metadata_path.read_text())
        history = point.setdefault("status_history", [])
        if "export_complete" not in history:
            history.append("export_complete")
        point["status"] = "export_complete"
        point["failure_reason"] = None
        point_metadata_path.write_text(
            json.dumps(point, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps({
        "status": "export_complete",
        "model_bin": str(model_path),
        "model_metadata": str(metadata_path),
        "export_sha256": metadata["export_sha256"],
        "parameter_count": parameter_count,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
