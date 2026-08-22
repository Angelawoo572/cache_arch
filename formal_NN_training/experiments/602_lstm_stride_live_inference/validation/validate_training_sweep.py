#!/usr/bin/env python3
"""Validate collected supervision and per-point status without hiding probes."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "formal_NN_training/experiments/602_lstm_stride_live_inference"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--require-all", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    config = json.loads(
        (EXP / "config/training_prefix_sweep.json").read_text()
    )
    errors = []
    present = 0
    for item in config["instruction_budgets"]:
        base = args.run_dir / "training_prefixes" / item["tag"]
        manifest_path = base / "training_manifest.json"
        if not manifest_path.is_file():
            if args.require_all:
                errors.append("missing {}".format(manifest_path))
            continue
        present += 1
        row = json.loads(manifest_path.read_text())
        stream = base / Path(row["training_stream"]).name
        checks = {
            "instruction_budget": item["instructions"],
            "budget_tag": item["tag"],
            "decision_rows": row["silent_rows"] + row["positive_count_rows"],
            "action_atoms": sum(
                int(k) * int(v)
                for k, v in row["k_histogram"].items()
            ),
        }
        for field, expected in checks.items():
            if row.get(field) != expected:
                errors.append("{} mismatch {}".format(field, manifest_path))
        if not stream.is_file():
            errors.append("missing stream {}".format(stream))
        elif digest(stream) != row.get("training_stream_sha256"):
            errors.append("stream SHA mismatch {}".format(stream))
        if row.get("status") not in {
            "no_callbacks", "single_class_no_act",
            "single_class_no_silent", "insufficient_rows", "trainable",
        }:
            errors.append("unknown prefix status {}".format(manifest_path))
    point_paths = list(args.run_dir.glob(
        "points/h*/*/seed*/point_metadata.json"
    ))
    if args.require_all and point_paths and len(point_paths) != 34:
        errors.append("expected 34 point metadata files, found {}".format(
            len(point_paths)
        ))
    for path in point_paths:
        row = json.loads(path.read_text())
        hidden = row.get("hidden_size")
        expected = 1908 if hidden == 8 else 5220 if hidden == 16 else None
        if expected is None or row.get("parameter_count") != expected:
            errors.append("parameter mismatch {}".format(path))
        if row.get("status") is None:
            errors.append("missing status {}".format(path))
        if "failure_reason" not in row:
            errors.append("missing failure reason field {}".format(path))
    if errors:
        raise SystemExit("SWEEP VALIDATION FAIL\n" + "\n".join(errors))
    print("[PASS] {} of 17 prefix manifests; {} point statuses".format(
        present, len(point_paths)
    ))


if __name__ == "__main__":
    main()
