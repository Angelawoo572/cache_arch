#!/usr/bin/env python3
"""Run fresh authoritative offline training/inference for selected prefixes."""

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "formal_NN_training/experiments/602_lstm_stride_live_inference"
TRAINER = (
    ROOT
    / "formal_NN_training/experiments/602_offline_lstm_stride/python"
    / "train_and_offline_infer.py"
)
MODEL_REVISION = "compact_shared_pc_hurdle_delta_v9"


def expected_parameter_count(feature_count, hidden_size):
    return 11 * hidden_size * hidden_size + (
        feature_count + 22
    ) * hidden_size + 4


UNTRAINABLE = {
    "no_callbacks",
    "single_class_no_act",
    "single_class_no_silent",
    "insufficient_rows",
}
TERMINAL = {
    "offline_inference_complete",
    "offline_replay_complete",
    "export_complete",
    "parity_complete",
    "live_smoke_complete",
    "live_run_complete",
}


def parse_csv(text, cast=str):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def resolve_training_stream(manifest_path, manifest):
    recorded = Path(manifest["training_stream"])
    candidates = [
        recorded,
        manifest_path.parent / recorded.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "training stream missing for {}".format(manifest_path)
    )


def choose_points(config, hidden_sizes, budgets, seeds):
    allowed_h = set(config["hidden_sizes"])
    allowed_s = set(config["seeds"])
    budget_items = config["instruction_budgets"]
    by_tag = {item["tag"]: item for item in budget_items}
    by_value = {str(item["instructions"]): item for item in budget_items}
    if hidden_sizes == ["all"]:
        selected_h = config["hidden_sizes"]
    else:
        selected_h = [int(item) for item in hidden_sizes]
    if seeds == ["all"]:
        selected_s = config["seeds"]
    else:
        selected_s = [int(item) for item in seeds]
    if budgets == ["all"]:
        selected_b = budget_items
    else:
        selected_b = []
        seen = set()
        for token in budgets:
            item = by_tag.get(token) or by_value.get(token)
            if item is None:
                raise RuntimeError("unknown budget {}".format(token))
            if item["tag"] not in seen:
                selected_b.append(item)
                seen.add(item["tag"])
    if not set(selected_h).issubset(allowed_h):
        raise RuntimeError("only hidden sizes 8 and 16 are in scope")
    if not set(selected_s).issubset(allowed_s):
        raise RuntimeError("seed is not configured for this stage")
    return [
        (hidden, budget, seed)
        for hidden in selected_h
        for budget in selected_b
        for seed in selected_s
    ]


def latest_history(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}, 0
    row = rows[-1]
    optimizer_steps = sum(int(item["optimizer_steps"]) for item in rows)
    return row, optimizer_steps


def ratio(numerator, denominator):
    return numerator / float(denominator) if denominator else None


def point_metadata(
    hidden, budget, seed, manifest, controls, status, failure_reason=None
):
    return {
        "schema_version": 1,
        "run_identity": (
            "602.gcc_s-734B_h{}_{}_seed{}_{}_offline".format(
                hidden, budget["tag"], seed, MODEL_REVISION
            )
        ),
        "trace": "602.gcc_s-734B",
        "hidden_size": hidden,
        "parameter_count": expected_parameter_count(128, hidden),
        "weight_bytes": expected_parameter_count(128, hidden) * 4,
        "instruction_budget": budget["instructions"],
        "budget_tag": budget["tag"],
        "decision_rows": manifest.get("decision_rows"),
        "silent_rows": manifest.get("silent_rows"),
        "positive_count_rows": manifest.get("positive_count_rows"),
        "action_atoms": manifest.get("action_atoms"),
        "k_histogram": manifest.get("k_histogram"),
        "unique_pcs": manifest.get("unique_pcs"),
        "unique_positive_pcs": manifest.get("unique_positive_pcs"),
        "training_stream_sha256": manifest.get("training_stream_sha256"),
        "raw_event_log_sha256": manifest.get("raw_event_log_sha256"),
        "model_revision": MODEL_REVISION,
        "seed": seed,
        "epochs": controls["epochs"],
        "chunk_length": controls["chunk_length"],
        "optimizer": controls["optimizer"],
        "learning_rate": controls["learning_rate"],
        "status_history": [manifest["status"], status],
        "status": status,
        "failure_reason": failure_reason,
    }


def archive_point(point_dir, run_root):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = (
        run_root / "replaced" / timestamp
        / point_dir.relative_to(run_root)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(point_dir), str(destination))
    print("[archived] {} -> {}".format(point_dir, destination))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=EXP / "config/training_prefix_sweep.json",
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--evaluation-stream", required=True, type=Path)
    parser.add_argument("--hidden-sizes", default="8,16")
    parser.add_argument("--budgets", default="all")
    parser.add_argument("--seeds", default="7")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    config = load_json(args.config)
    controls = config["fixed_controls"]
    points = choose_points(
        config,
        parse_csv(args.hidden_sizes),
        parse_csv(args.budgets),
        parse_csv(args.seeds),
    )
    if not args.status_only and not args.evaluation_stream.is_file():
        raise RuntimeError(
            "fixed evaluation stream missing: {}".format(
                args.evaluation_stream
            )
        )
    summary = []
    for hidden, budget, seed in points:
        prefix_dir = (
            args.run_dir / "training_prefixes" / budget["tag"]
        )
        manifest_path = prefix_dir / "training_manifest.json"
        if not manifest_path.is_file():
            summary.append({
                "hidden_size": hidden,
                "budget_tag": budget["tag"],
                "seed": seed,
                "status": "training_failed",
                "failure_reason": "training manifest missing",
            })
            continue
        manifest = load_json(manifest_path)
        point_dir = (
            args.run_dir / "points"
            / "h{}".format(hidden) / budget["tag"]
            / "seed{}".format(seed)
        )
        status_path = point_dir / "point_metadata.json"
        if status_path.is_file():
            existing = load_json(status_path)
            if args.status_only or (
                (args.resume or not args.force)
                and existing.get("status") in TERMINAL.union(UNTRAINABLE)
            ):
                summary.append(existing)
                print("[skip] h{} {} {}".format(
                    hidden, budget["tag"], existing.get("status")
                ))
                continue
            if not args.force:
                raise RuntimeError(
                    "point exists but is incomplete; use --resume or --force: "
                    + str(point_dir)
                )
            archive_point(point_dir, args.run_dir)
        if args.status_only:
            summary.append(point_metadata(
                hidden, budget, seed, manifest, controls,
                manifest["status"], manifest.get("failure_reason"),
            ))
            continue
        if manifest["status"] in UNTRAINABLE:
            metadata = point_metadata(
                hidden, budget, seed, manifest, controls,
                manifest["status"], manifest.get("failure_reason"),
            )
            write_json(status_path, metadata)
            summary.append(metadata)
            print("[untrainable] h{} {} {}".format(
                hidden, budget["tag"], manifest["status"]
            ))
            continue
        train_stream = resolve_training_stream(manifest_path, manifest)
        offline_dir = point_dir / "offline"
        command = [
            sys.executable, str(TRAINER),
            "--train-stream", str(train_stream),
            "--eval-stream", str(args.evaluation_stream),
            "--out-dir", str(offline_dir),
            "--seed", str(seed),
            "--epochs", str(controls["epochs"]),
            "--chunk-len", str(controls["chunk_length"]),
            "--pc-batch-size", str(controls["pc_batch_size"]),
            "--learning-rate", str(controls["learning_rate"]),
            "--hidden-size", str(hidden),
            "--device", args.device,
        ]
        if args.dry_run:
            print("[dry-run] " + " ".join(command))
            continue
        point_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        status = point_metadata(
            hidden, budget, seed, manifest, controls, "trainable"
        )
        status["status_history"].append("training_started")
        write_json(status_path, status)
        log_path = point_dir / "training.log"
        try:
            with log_path.open("w") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT,
                    check=True,
                )
        except subprocess.CalledProcessError as exc:
            status["status"] = "training_failed"
            status["failure_reason"] = (
                "authoritative trainer exited {}".format(exc.returncode)
            )
            status["training_wall_clock_seconds"] = (
                time.monotonic() - started
            )
            status["status_history"].append("training_failed")
            write_json(status_path, status)
            summary.append(status)
            print("[failed] h{} {}".format(hidden, budget["tag"]))
            continue
        old_metadata = load_json(offline_dir / "run_metadata.json")
        history, optimizer_steps = latest_history(
            offline_dir / "training_history.csv"
        )
        behavior = old_metadata["heldout_behavior_metrics"]
        eval_rows = int(old_metadata["eval_rows"])
        predicted_positive = int(
            behavior["gate_predicted_positive_rows"]
        )
        teacher_positive = int(
            behavior["gate_target_positive_rows"]
        )
        status.update({
            "evaluation_stream_sha256": old_metadata["eval_stream_sha256"],
            "runtime_encoder_sha256": old_metadata[
                "runtime_encoder_sha256"
            ],
            "optimizer_steps": optimizer_steps,
            "training_wall_clock_seconds": old_metadata.get(
                "training_wall_clock_seconds"
            ),
            "training_and_offline_inference_wall_clock_seconds": (
                time.monotonic() - started
            ),
            "gate_loss": float(history["gate_loss_per_callback"]),
            "count_loss": float(
                history["positive_count_loss_per_positive_callback"]
            ),
            "delta_loss": float(
                history["action_delta_loss_per_action"]
            ),
            "student_heldout_silent_rate": 1.0 - ratio(
                predicted_positive, eval_rows
            ),
            "student_heldout_act_rate": ratio(
                predicted_positive, eval_rows
            ),
            "teacher_silent_to_student_act_rate": ratio(
                behavior["gate_false_positive_rows"],
                eval_rows - teacher_positive,
            ),
            "teacher_act_to_student_silent_rate": ratio(
                behavior["gate_false_negative_rows"],
                teacher_positive,
            ),
            "exact_k_accuracy": behavior["count_exact_match_rate"],
            "offline_replay_entry_count": old_metadata[
                "offline_lstm_entries"
            ],
            "heldout_behavior_metrics": behavior,
            "checkpoint": str(offline_dir / "model.pt"),
            "offline_replay": str(
                offline_dir / "offline_lstm.replay.csv"
            ),
            "status": "offline_inference_complete",
            "failure_reason": None,
        })
        status["status_history"].extend([
            "trained_nonempty"
            if old_metadata["offline_lstm_entries"] else "trained_all_silent",
            "offline_inference_complete",
        ])
        write_json(status_path, status)
        summary.append(status)
        print("[complete] h{} {}".format(hidden, budget["tag"]))
    summary_path = args.run_dir / "training_sweep_status.json"
    write_json(summary_path, {
        "schema_version": 1,
        "model_revision": MODEL_REVISION,
        "points": summary,
    })
    print("[summary] {}".format(summary_path))


if __name__ == "__main__":
    main()
