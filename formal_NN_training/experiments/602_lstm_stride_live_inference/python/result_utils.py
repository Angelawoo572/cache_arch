#!/usr/bin/env python3
"""Shared result parsing without duplicating the offline metric oracle."""

import csv
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OFFLINE_ANALYZER = (
    ROOT
    / "formal_NN_training/experiments/602_offline_lstm_stride/python"
    / "analyze_replay.py"
)
LIVE_KV = re.compile(
    r"^(stride_lstm_live_[A-Za-z0-9_]+)\s+([-+0-9.eE]+)\s*$"
)


def offline_analyzer():
    spec = importlib.util.spec_from_file_location(
        "offline_stride_metric_oracle", OFFLINE_ANALYZER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path):
    return json.loads(Path(path).read_text())


def parse_log(path):
    return offline_analyzer().parse_log(Path(path))


def parse_live_counters(path):
    counters = {}
    for line in Path(path).read_text(errors="ignore").splitlines():
        match = LIVE_KV.match(line.strip())
        if match:
            counters[match.group(1)] = float(match.group(2))
    return counters


def ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def base_point(metadata):
    fields = (
        "trace", "run_identity", "hidden_size", "parameter_count",
        "weight_bytes", "instruction_budget", "budget_tag", "decision_rows",
        "silent_rows", "positive_count_rows", "action_atoms", "k_histogram",
        "unique_pcs", "unique_positive_pcs",
        "model_revision", "seed", "epochs", "chunk_length", "optimizer",
        "learning_rate", "optimizer_steps", "training_wall_clock_seconds",
        "gate_loss", "count_loss", "delta_loss",
        "student_heldout_silent_rate", "student_heldout_act_rate",
        "teacher_silent_to_student_act_rate",
        "teacher_act_to_student_silent_rate", "exact_k_accuracy",
        "offline_replay_entry_count", "status", "failure_reason",
    )
    return {field: metadata.get(field) for field in fields}


def system_metrics(stats, baseline):
    if stats is None:
        return {
            key: None for key in (
                "requested", "issued", "nonmerged", "filled", "useful",
                "late", "unused_evicted", "request_drops", "pq_merges",
                "requests_per_l2_load", "raw_accuracy",
                "selected_accuracy", "coverage", "timeliness",
                "l2_load_miss_rate", "miss_reduction", "cycles", "ipc",
                "speedup_versus_no_pref",
            )
        }
    return {
        "requested": stats["pf_requested"],
        "issued": stats["pf_issued"],
        "nonmerged": stats["nodup_issued"],
        "filled": stats["pf_filled"],
        "useful": stats["pf_useful"],
        "late": stats["pf_late"],
        "unused_evicted": stats["pf_useless"],
        "request_drops": stats["pf_dropped"],
        "pq_merges": stats["pq_merged_duplicate_proxy"],
        "requests_per_l2_load": stats["request_per_l2_load"],
        "raw_accuracy": stats["accuracy"],
        "selected_accuracy": stats["selected_accuracy"],
        "coverage": (
            ratio(stats["pf_useful"], baseline["l2_load_miss"])
            if baseline else None
        ),
        "timeliness": stats["timeliness"],
        "l2_load_miss_rate": stats["l2_load_miss_rate"],
        "miss_reduction": (
            ratio(
                baseline["l2_load_miss"] - stats["l2_load_miss"],
                baseline["l2_load_miss"],
            )
            if baseline else None
        ),
        "cycles": stats["cycles"],
        "ipc": stats["ipc"],
        "speedup_versus_no_pref": (
            ratio(stats["ipc"], baseline["ipc"]) if baseline else None
        ),
    }


def write_outputs(rows, csv_path, json_path, references=None):
    csv_path = Path(csv_path)
    json_path = Path(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded = {}
            for key in fields:
                value = row.get(key)
                if isinstance(value, (dict, list)):
                    value = json.dumps(
                        value, sort_keys=True, separators=(",", ":")
                    )
                encoded[key] = "NA" if value is None else value
            writer.writerow(encoded)
    json_path.write_text(json.dumps({
        "schema_version": 1,
        "references": references or {},
        "points": rows,
    }, indent=2, sort_keys=True) + "\n")


def reference_logs(run_dir):
    logs = Path(run_dir) / "references/logs"
    result = {}
    for name in ("no_pref", "live_stride", "offline_stride"):
        path = logs / (name + ".log")
        if path.is_file():
            result[name] = parse_log(path)
    return result
