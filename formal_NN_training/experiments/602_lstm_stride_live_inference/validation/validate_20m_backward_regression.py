#!/usr/bin/env python3
"""Compare sweep i20m anchors with an earlier default 20M offline run.

Exact local artifacts are compared when both sides exist.  Otherwise the
report explicitly falls back to historical-metric regression and never calls
that exact artifact parity.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
ANALYZER = ROOT / "formal_NN_training/experiments/602_offline_lstm_stride/python/analyze_replay.py"


HISTORICAL_IPC = {8: 0.40924, 16: 0.40994}
METRICS = (
    "ipc", "cycles", "coverage", "l2_load_miss_rate",
    "requests_per_l2_load", "selected_accuracy", "raw_accuracy",
    "timeliness", "student_heldout_act_rate", "student_heldout_silent_rate",
    "offline_replay_entry_count",
)
ARTIFACTS = (
    "training_stream", "evaluation_stream", "offline_stride.replay.csv",
    "offline_lstm.replay.csv", "model.pt", "run_metadata.json",
    "no_pref.log", "offline_stride.log", "replay.log",
)
REQUIRED_EXACT_ARTIFACTS = {
    "training_stream", "evaluation_stream", "offline_stride.replay.csv",
}
MODEL_METADATA_FIELDS = (
    "hidden_size", "parameter_count", "model_revision", "experiment_revision",
    "runtime_encoder_sha256", "state_router_sha256", "effective_external_inputs",
    "training_runtime_fields", "inference_runtime_fields", "state_routing",
    "inference_state_mode", "gate_imbalance_handling",
    "data_derived_class_balancing_used", "pc_batch_size", "training_chunk_len",
)


def load(path):
    return json.loads(Path(path).read_text())


def parse_log(path):
    spec = importlib.util.spec_from_file_location("offline_metric_oracle", str(ANALYZER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_log(Path(path))


def compare_reference_logs(old_root, new_root):
    output = []
    for name in ("no_pref.log", "offline_stride.log", "live_stride.log"):
        old_path = find_first(old_root, ["**/" + name])
        new_path = find_first(Path(new_root) / "references/logs", [name])
        if not old_path or not new_path:
            output.append({"reference": name, "status": "UNAVAILABLE"})
            continue
        old, new = parse_log(old_path), parse_log(new_path)
        fields = ("ipc", "cycles", "l2_load_miss_rate", "request_per_l2_load",
                  "accuracy", "selected_accuracy", "timeliness")
        differences = {}
        mismatch = False
        for field in fields:
            old_value, new_value = old.get(field), new.get(field)
            same = close(new_value, old_value, 1e-9)
            differences[field] = {"old": old_value, "new": new_value,
                                  "status": "PASS" if same is True else
                                            "DIFFERENT" if same is False else "UNAVAILABLE"}
            mismatch = mismatch or same is False
        output.append({"reference": name,
                       "status": "DIFFERENT" if mismatch else "PASS",
                       "metrics": differences})
    return output


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_first(root, patterns):
    for pattern in patterns:
        matches = sorted(Path(root).glob(pattern))
        if matches:
            return matches[0]
    return None


def new_point(new_root, hidden):
    return Path(new_root) / "points" / "h{}".format(hidden) / "i20m/seed7"


def metric_row(root, hidden, new_layout=False):
    aggregate = Path(root) / "offline_sweep_results.json"
    if aggregate.is_file():
        for row in load(aggregate).get("points", []):
            if (row.get("hidden_size") == hidden and
                    row.get("instruction_budget") == 20000000 and
                    row.get("seed") == 7):
                return row
    base = new_point(root, hidden) if new_layout else Path(root)
    metadata = find_first(base, [
        "point_metadata.json", "**/h{}/*/run_metadata.json".format(hidden),
        "**/h{}*/run_metadata.json".format(hidden),
        "**/*h{}*/run_metadata.json".format(hidden), "**/run_metadata.json",
    ])
    return load(metadata) if metadata else {}


def artifact_candidates(root, hidden, name, new_layout=False):
    base = new_point(root, hidden) if new_layout else Path(root)
    if new_layout and name == "training_stream":
        return find_first(Path(root) / "training_prefixes/i20m", ["*.train_stream.csv.gz"])
    if new_layout and name == "evaluation_stream":
        return find_first(Path(root) / "evaluation", ["*.eval_stream.csv.gz"])
    if new_layout and name in ("no_pref.log", "offline_stride.log"):
        return find_first(Path(root) / "references/logs", [name])
    if name == "training_stream":
        patterns = ["**/*train*stream*.csv.gz"]
    elif name == "evaluation_stream":
        patterns = ["**/*eval*stream*.csv.gz"]
    elif name in ("no_pref.log", "offline_stride.log"):
        patterns = ["**/" + name]
    else:
        patterns = ["offline/" + name,
                    "**/h{}*/**/{}".format(hidden, name),
                    "**/*h{}*/**/{}".format(hidden, name),
                    "**/" + name]
    return find_first(base, patterns)


def close(actual, expected, relative_tolerance, absolute_tolerance=0.0):
    if actual is None or expected is None:
        return None
    return math.isclose(float(actual), float(expected),
                        rel_tol=relative_tolerance,
                        abs_tol=absolute_tolerance)


def compare_hidden(old_root, new_root, hidden, ipc_tolerance):
    old_metrics = metric_row(old_root, hidden, False)
    new_metrics = metric_row(new_root, hidden, True)
    artifacts = []
    exact_available = False
    exact_mismatch = False
    for name in ARTIFACTS:
        old_path = artifact_candidates(old_root, hidden, name, False)
        new_path = artifact_candidates(new_root, hidden, name, True)
        if old_path and new_path:
            old_hash, new_hash = sha256(old_path), sha256(new_path)
            status = "PASS" if old_hash == new_hash else "DIFFERENT"
            exact_mismatch = exact_mismatch or (
                status == "DIFFERENT" and name in REQUIRED_EXACT_ARTIFACTS
            )
        else:
            old_hash = new_hash = None
            status = "UNAVAILABLE"
        artifacts.append({
            "artifact": name, "status": status,
            "old_path": str(old_path) if old_path else None,
            "new_path": str(new_path) if new_path else None,
            "old_sha256": old_hash, "new_sha256": new_hash,
        })
    exact_available = all(
        item["status"] != "UNAVAILABLE"
        for item in artifacts if item["artifact"] in REQUIRED_EXACT_ARTIFACTS
    )
    old_metadata_path = artifact_candidates(old_root, hidden, "run_metadata.json", False)
    new_metadata_path = artifact_candidates(new_root, hidden, "run_metadata.json", True)
    old_metadata = load(old_metadata_path) if old_metadata_path else {}
    new_metadata = load(new_metadata_path) if new_metadata_path else {}
    metadata_comparisons = []
    metadata_mismatch = False
    for field in MODEL_METADATA_FIELDS:
        old_value, new_value = old_metadata.get(field), new_metadata.get(field)
        if old_value is None or new_value is None:
            status = "UNAVAILABLE"
        elif old_value == new_value:
            status = "PASS"
        else:
            status = "MISMATCH"
            metadata_mismatch = True
        metadata_comparisons.append({"field": field, "old": old_value,
                                     "new": new_value, "status": status})
    comparisons = []
    for field in METRICS:
        old_value = old_metrics.get(field)
        if old_value is None and field == "ipc":
            old_value = HISTORICAL_IPC[hidden]
        new_value = new_metrics.get(field)
        tolerance = ipc_tolerance if field == "ipc" else 1e-9
        passed = close(new_value, old_value, tolerance)
        comparisons.append({
            "metric": field, "old": old_value, "new": new_value,
            "relative_tolerance": tolerance,
            "status": "PASS" if passed is True else
                      "MISMATCH" if passed is False else "UNAVAILABLE",
        })
    mode = ("exact_artifact_and_metric_regression" if exact_available
            else "historical_metric_regression_only")
    metric_mismatch = any(item["status"] == "MISMATCH" for item in comparisons)
    unavailable_non_ipc = any(
        item["metric"] != "ipc" and item["status"] == "UNAVAILABLE"
        for item in comparisons
    )
    if exact_mismatch or metric_mismatch or metadata_mismatch:
        status = "FAIL"
    elif mode == "historical_metric_regression_only" or unavailable_non_ipc:
        status = "INCOMPLETE"
    else:
        status = "PASS"
    return {"hidden_size": hidden, "mode": mode, "status": status,
            "artifacts": artifacts, "model_input_metadata": metadata_comparisons,
            "metrics": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run-dir", required=True, type=Path)
    parser.add_argument("--new-run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ipc-relative-tolerance", type=float, default=0.005)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    results = [compare_hidden(args.old_run_dir, args.new_run_dir, hidden,
                              args.ipc_relative_tolerance)
               for hidden in (8, 16)]
    payload = {
        "schema_version": 1,
        "old_run_dir": str(args.old_run_dir),
        "new_run_dir": str(args.new_run_dir),
        "historical_anchors_are_not_sufficient_for_exact_parity": True,
        "results": results,
        "reference_log_comparisons": compare_reference_logs(
            args.old_run_dir, args.new_run_dir
        ),
    }
    output = args.output or args.new_run_dir / "report/20m_backward_regression.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tex = output.with_suffix(".tex")
    tex.write_text("\\begin{itemize}\n" + "\n".join(
        "\\item h{}: {} ({})".format(
            item["hidden_size"], item["status"],
            item["mode"].replace("_", "\\_")
        ) for item in results
    ) + "\n\\end{itemize}\n")
    for item in results:
        print("h{} {} {}".format(item["hidden_size"], item["status"], item["mode"]))
    if any(item["status"] == "FAIL" for item in results):
        raise SystemExit(2)
    if args.require_exact and any(item["status"] != "PASS" for item in results):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
