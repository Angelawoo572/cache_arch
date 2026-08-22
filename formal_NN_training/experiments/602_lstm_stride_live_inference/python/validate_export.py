#!/usr/bin/env python3
"""Validate model.bin and optionally compare two checkpoints event by event."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
from live_model_format import FORMAT_VERSION, TENSOR_ORDER, read_model, sha256


DEFAULT_TOLERANCES = (
    ROOT
    / "formal_NN_training/experiments/602_lstm_stride_live_inference"
    / "config/regression_tolerances.json"
)


def dotted(payload, path):
    value = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def compare_metadata(candidate_path, reference_path, tolerances_path):
    candidate = json.loads(Path(candidate_path).read_text())
    reference = json.loads(Path(reference_path).read_text())
    config = json.loads(Path(tolerances_path).read_text())
    errors = []
    for field in config["exact_metadata_fields"]:
        left = dotted(candidate, field)
        right = dotted(reference, field)
        if left != right:
            errors.append(
                "exact metadata {}: candidate={!r} reference={!r}".format(
                    field, left, right
                )
            )
    for field, tolerance in config[
        "absolute_metric_tolerances"
    ].items():
        left = dotted(candidate, field)
        right = dotted(reference, field)
        if left is None or right is None:
            errors.append(
                "metric {} unavailable: candidate={!r} reference={!r}".format(
                    field, left, right
                )
            )
        elif abs(float(left) - float(right)) > float(tolerance):
            errors.append(
                "metric {} differs by {} > {}".format(
                    field, abs(float(left) - float(right)), tolerance
                )
            )
    if errors:
        raise RuntimeError("metadata regression mismatch\n" + "\n".join(errors))
    return {
        "exact_fields": len(config["exact_metadata_fields"]),
        "metric_fields": len(config["absolute_metric_tolerances"]),
    }


def compare_checkpoint_actions(
    first_path, second_path, stream, max_events,
    frozen_model_type, load_checkpoint, load_stream,
):
    first_model, _ = load_checkpoint(first_path)
    second_model, _ = load_checkpoint(second_path)
    if first_model.hidden_size != second_model.hidden_size:
        raise RuntimeError("checkpoint hidden sizes differ")
    first = frozen_model_type(first_model)
    second = frozen_model_type(second_model)
    rows = load_stream(stream)
    limit = min(len(rows), max_events or len(rows))
    for index, (pc, line, _) in enumerate(rows[:limit]):
        left = first.infer(pc, line)
        right = second.infer(pc, line)
        if (
            left["emit"], left["k"], left["deltas"], left["addresses"]
        ) != (
            right["emit"], right["k"], right["deltas"], right["addresses"]
        ):
            raise RuntimeError(
                "checkpoint action mismatch at event {}: {} != {}".format(
                    index, left, right
                )
            )
    return limit


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-bin", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--candidate-run-metadata", type=Path)
    parser.add_argument("--reference-run-metadata", type=Path)
    parser.add_argument(
        "--metric-tolerances", type=Path, default=DEFAULT_TOLERANCES
    )
    parser.add_argument("--evaluation-stream", type=Path)
    parser.add_argument("--max-events", type=int)
    return parser


def main():
    args = build_parser().parse_args()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from formal_NN_training.common.stride_direct_action_model import (
        FrozenStrideLiveModel,
        MODEL_REVISION,
        expected_parameter_count,
        load_checkpoint,
        load_stream,
        runtime_encoder_sha256,
    )
    binary = read_model(args.model_bin)
    metadata = json.loads(args.metadata.read_text())
    model, checkpoint = load_checkpoint(args.checkpoint)
    errors = []
    expected = {
        "format_version": FORMAT_VERSION,
        "hidden_size": model.hidden_size,
        "feature_width": model.feature_count,
        "parameter_count": expected_parameter_count(
            model.feature_count, model.hidden_size
        ),
        "model_revision": MODEL_REVISION,
        "runtime_encoder_sha256": runtime_encoder_sha256(),
        "export_sha256": sha256(args.model_bin),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            errors.append(
                "{}: metadata={!r} expected={!r}".format(
                    key, metadata.get(key), value
                )
            )
    for key in ("hidden_size", "feature_width", "parameter_count"):
        if binary[key] != expected[key]:
            errors.append(
                "{}: model.bin={!r} expected={!r}".format(
                    key, binary[key], expected[key]
                )
            )
    state = model.state_dict()
    if list(binary["tensors"]) != TENSOR_ORDER:
        errors.append("tensor order differs")
    for name in TENSOR_ORDER:
        expected_tensor = state[name].detach().cpu().numpy()
        observed = binary["tensors"].get(name)
        if observed is None or not np.array_equal(observed, expected_tensor):
            errors.append("tensor mismatch: {}".format(name))
    if int(checkpoint["parameter_count"]) != expected["parameter_count"]:
        errors.append("checkpoint parameter count differs")
    if errors:
        raise SystemExit("EXPORT FAIL\n" + "\n".join(errors))
    compared = 0
    metadata_comparison = None
    if bool(args.candidate_run_metadata) != bool(
        args.reference_run_metadata
    ):
        raise RuntimeError(
            "candidate/reference run metadata must be supplied together"
        )
    if args.candidate_run_metadata:
        metadata_comparison = compare_metadata(
            args.candidate_run_metadata,
            args.reference_run_metadata,
            args.metric_tolerances,
        )
    if args.reference_checkpoint:
        if not args.evaluation_stream:
            raise RuntimeError(
                "--evaluation-stream is required for checkpoint comparison"
            )
        compared = compare_checkpoint_actions(
            args.reference_checkpoint,
            args.checkpoint,
            args.evaluation_stream,
            args.max_events,
            FrozenStrideLiveModel,
            load_checkpoint,
            load_stream,
        )
    print(json.dumps({
        "status": "PASS",
        "exact_tensor_parity": True,
        "checkpoint_action_events_compared": compared,
        "parameter_count": expected["parameter_count"],
        "metadata_comparison": metadata_comparison,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
