#!/usr/bin/env python3
"""Compare new h8/h16 i20m points with the original default 20M run.

The script never launches training or ChampSim.  When the original artifact
tree is incomplete it deliberately falls back to historical-metric regression
and refuses to claim exact artifact parity.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "formal_NN_training/experiments/602_lstm_stride_live_inference"
PYTHON_DIR = EXP / "python"
VALIDATION_DIR = Path(__file__).resolve().parent
for path in (PYTHON_DIR, VALIDATION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from provenance_utils import artifact, load_json  # noqa: E402
from result_utils import parse_log  # noqa: E402


TRACE = "602.gcc_s-734B"
HISTORICAL_IPC = {8: 0.40924, 16: 0.40994}


def nested(data, dotted):
    value = data
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def close(left, right, tolerance):
    if left is None or right is None:
        return False
    try:
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=tolerance
        )
    except (TypeError, ValueError):
        return False


def compare_artifact(name, old_path, new_path, content=False,
                     exact_required=False):
    old = artifact(old_path, gzip_content=content)
    new = artifact(new_path, gzip_content=content)
    key = "content_sha256" if content else "sha256"
    if not old["available"] or not new["available"]:
        status = "UNAVAILABLE"
        same = None
    else:
        same = old.get(key) == new.get(key)
        status = "PASS" if same else (
            "FAIL" if exact_required else "DIFFERENT"
        )
    return {
        "name": name,
        "status": status,
        "exact_required": bool(exact_required),
        "same_identity": same,
        "identity_kind": key,
        "old": old,
        "new": new,
    }


def metrics_from_log(path, baseline=None):
    if not Path(path).is_file():
        return None
    parsed = parse_log(path)
    coverage = None
    if baseline and baseline.get("l2_load_miss"):
        coverage = (
            parsed["pf_useful"] / float(baseline["l2_load_miss"])
        )
    return {
        "ipc": parsed["ipc"],
        "cycles": parsed["cycles"],
        "coverage": coverage,
        "l2_load_miss_rate": parsed["l2_load_miss_rate"],
        "requests_per_l2_load": parsed["request_per_l2_load"],
        "raw_accuracy": parsed["accuracy"],
        "selected_accuracy": parsed["selected_accuracy"],
        "timeliness": parsed["timeliness"],
        "requested": parsed["pf_requested"],
        "issued": parsed["pf_issued"],
        "useful": parsed["pf_useful"],
        "late": parsed["pf_late"],
    }


def behavior_from_metadata(path):
    if not Path(path).is_file():
        return None
    data = load_json(path)
    behavior = data.get("heldout_behavior_metrics") or {}
    eval_rows = data.get("eval_rows")
    predicted = behavior.get("gate_predicted_positive_rows")
    return {
        "student_act_rate": (
            predicted / float(eval_rows)
            if predicted is not None and eval_rows else None
        ),
        "student_silent_rate": (
            1.0 - predicted / float(eval_rows)
            if predicted is not None and eval_rows else None
        ),
        "replay_action_count": data.get("offline_lstm_entries"),
        "count_exact_match_rate": behavior.get("count_exact_match_rate"),
        "target_precision": behavior.get("target_precision"),
        "target_recall": behavior.get("target_recall"),
        "target_f1": behavior.get("target_f1"),
    }


def compare_mapping(old, new, tolerances, exact_fields=()):
    rows = []
    fields = sorted(set(tolerances).union(exact_fields))
    for field in fields:
        old_value = nested(old, field) if old else None
        new_value = nested(new, field) if new else None
        if old_value is None or new_value is None:
            status = "UNAVAILABLE"
            delta = None
        elif field in exact_fields:
            status = "PASS" if old_value == new_value else "FAIL"
            delta = None
        else:
            tolerance = tolerances[field]
            try:
                delta = float(new_value) - float(old_value)
            except (TypeError, ValueError):
                delta = None
            status = "PASS" if close(new_value, old_value, tolerance) else "FAIL"
        rows.append({
            "field": field,
            "old": old_value,
            "new": new_value,
            "delta_new_minus_old": delta,
            "absolute_tolerance": None if field in exact_fields
            else tolerances[field],
            "status": status,
        })
    return rows


def old_paths(old_run, hidden):
    return {
        "metadata": old_run / "colab_output/h{}/run_metadata.json".format(hidden),
        "lstm_list": old_run / "colab_output/h{}/offline_lstm.replay.csv".format(hidden),
        "stride_list": old_run / "colab_output/h{}/offline_stride.replay.csv".format(hidden),
        "log": old_run / "logs/{}.offline_lstm_h{}.log".format(TRACE, hidden),
    }


def new_paths(new_run, hidden):
    point = new_run / "points/h{}/i20m/seed7".format(hidden)
    return {
        "metadata": point / "offline/run_metadata.json",
        "lstm_list": point / "offline/offline_lstm.replay.csv",
        "stride_list": point / "offline/offline_stride.replay.csv",
        "log": point / "offline/replay.log",
    }


def write_csv(path, report):
    rows = []
    for item in report["artifact_comparisons"]:
        rows.append({
            "scope": "artifact", "hidden_size": "all",
            "field": item["name"], "old": item["old"].get(item["identity_kind"]),
            "new": item["new"].get(item["identity_kind"]),
            "delta": "", "tolerance": "", "status": item["status"],
        })
    for hidden in (8, 16):
        point = report["points"]["h{}".format(hidden)]
        for scope in ("metadata", "system_metrics", "behavior"):
            for item in point[scope]:
                rows.append({
                    "scope": scope, "hidden_size": hidden,
                    "field": item["field"], "old": item["old"],
                    "new": item["new"],
                    "delta": item["delta_new_minus_old"],
                    "tolerance": item["absolute_tolerance"],
                    "status": item["status"],
                })
        anchor = point["historical_ipc_anchor"]
        rows.append({
            "scope": "historical_anchor", "hidden_size": hidden,
            "field": "ipc", "old": anchor["expected"],
            "new": anchor["observed"], "delta": anchor["delta"],
            "tolerance": anchor["absolute_tolerance"],
            "status": anchor["status"],
        })
    for reference, comparisons in report.get("reference_metrics", {}).items():
        for item in comparisons:
            rows.append({
                "scope": "reference_metrics." + reference,
                "hidden_size": "all", "field": item["field"],
                "old": item["old"], "new": item["new"],
                "delta": item["delta_new_minus_old"],
                "tolerance": item["absolute_tolerance"],
                "status": item["status"],
            })
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(value):
    return str(value).replace("_", r"\_").replace("%", r"\%")


def write_tex(path, report):
    lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Model & Historical IPC & New IPC & $\Delta$IPC & Status\\",
        r"\midrule",
    ]
    for hidden in (8, 16):
        anchor = report["points"]["h{}".format(hidden)][
            "historical_ipc_anchor"
        ]
        observed = anchor["observed"]
        delta = anchor["delta"]
        lines.append(
            "h{} & {:.5f} & {} & {} & {}\\\\".format(
                hidden, anchor["expected"],
                "NA" if observed is None else "{:.5f}".format(observed),
                "NA" if delta is None else "{:+.5f}".format(delta),
                tex_escape(anchor["status"]),
            )
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\paragraph{Regression mode.} " + tex_escape(report["mode"]) + ".",
        r"\paragraph{Exact artifact parity.} "
        + tex_escape(report["exact_artifact_parity_status"]) + ".",
    ])
    if report["mode"] == "historical-metric regression only":
        lines.append(
            "Old local artifacts were incomplete, so this result must not "
            "be described as exact artifact parity."
        )
    Path(path).write_text("\n".join(lines) + "\n")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run-dir", required=True, type=Path)
    parser.add_argument("--new-run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main():
    args = build_parser().parse_args()
    old_run = args.old_run_dir
    new_run = args.new_run_dir
    output = args.output or new_run / "report/regression_20m_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    tolerances = load_json(EXP / "config/regression_tolerances.json")
    system_tolerances = tolerances[
        "system_metric_tolerances_for_retrained_anchors"
    ]
    system_tolerances = dict(system_tolerances)
    system_tolerances.update({
        "cycles": 400000.0,
        "raw_accuracy": 0.02,
        "selected_accuracy": 0.02,
        "timeliness": 0.01,
        "requested": 20000.0,
        "issued": 20000.0,
        "useful": 5000.0,
        "late": 100.0,
    })
    behavior_tolerances = {
        "student_act_rate": 0.02,
        "student_silent_rate": 0.02,
        "replay_action_count": 20000.0,
        "count_exact_match_rate": 0.0,
        "target_precision": 0.0,
        "target_recall": 0.0,
        "target_f1": 0.0,
    }
    exact_metadata_fields = tuple(tolerances["exact_metadata_fields"])

    old_train = old_run / "colab_input/{}.train_stream.csv.gz".format(TRACE)
    old_eval = old_run / "colab_input/{}.eval_stream.csv.gz".format(TRACE)
    new_train = (
        new_run / "training_prefixes/i20m/{}.i20m.train_stream.csv.gz".format(TRACE)
    )
    new_eval = new_run / "evaluation/{}.eval_stream.csv.gz".format(TRACE)
    artifacts = [
        compare_artifact(
            "training_stream_content_sha256", old_train, new_train,
            content=True, exact_required=True,
        ),
        compare_artifact(
            "evaluation_stream_content_sha256", old_eval, new_eval,
            content=True, exact_required=True,
        ),
    ]
    old_no_pref = old_run / "logs/{}.no_pref.log".format(TRACE)
    new_no_pref = new_run / "references/logs/no_pref.log"
    old_stride_log = old_run / "logs/{}.offline_stride.log".format(TRACE)
    new_stride_log = new_run / "references/logs/offline_stride.log"
    artifacts.extend([
        compare_artifact(
            "no_pref_log", old_no_pref, new_no_pref, exact_required=False
        ),
        compare_artifact(
            "offline_stride_log", old_stride_log, new_stride_log,
            exact_required=False,
        ),
    ])
    for hidden in (8, 16):
        old = old_paths(old_run, hidden)
        new = new_paths(new_run, hidden)
        artifacts.extend([
            compare_artifact(
                "h{}_offline_stride_replay_list".format(hidden),
                old["stride_list"], new["stride_list"], exact_required=True,
            ),
            compare_artifact(
                "h{}_lstm_replay_list".format(hidden),
                old["lstm_list"], new["lstm_list"], exact_required=False,
            ),
            compare_artifact(
                "h{}_replay_log".format(hidden), old["log"], new["log"],
                exact_required=False,
            ),
        ])

    complete_artifact_set_available = (
        all(
            item["old"]["available"] and item["new"]["available"]
            for item in artifacts
        )
        and all(
            old_paths(old_run, hidden)["metadata"].is_file()
            and new_paths(new_run, hidden)["metadata"].is_file()
            for hidden in (8, 16)
        )
    )
    mode = (
        "old-versus-new artifact and metric regression"
        if complete_artifact_set_available
        else "historical-metric regression only"
    )
    old_baseline = metrics_from_log(old_no_pref)
    new_baseline = metrics_from_log(new_no_pref)
    reference_tolerances = {
        "ipc": 0.0,
        "cycles": 0.0,
        "coverage": 0.0,
        "l2_load_miss_rate": 0.0,
        "requests_per_l2_load": 0.0,
        "raw_accuracy": 0.0,
        "selected_accuracy": 0.0,
        "timeliness": 0.0,
        "requested": 0.0,
        "issued": 0.0,
        "useful": 0.0,
        "late": 0.0,
    }
    offline_stride_reference_tolerances = dict(reference_tolerances)
    for field in (
        "ipc", "coverage", "l2_load_miss_rate", "requests_per_l2_load"
    ):
        offline_stride_reference_tolerances[field] = system_tolerances[field]
    reference_metrics = {}
    if complete_artifact_set_available:
        no_pref_tolerances = dict(reference_tolerances)
        # Coverage is defined against the no-prefetch miss count, so it is
        # not a meaningful metric for the no-prefetch reference itself.
        no_pref_tolerances.pop("coverage")
        reference_metrics["no_pref"] = compare_mapping(
            old_baseline, new_baseline, no_pref_tolerances
        )
        reference_metrics["offline_stride"] = compare_mapping(
            metrics_from_log(old_stride_log, old_baseline),
            metrics_from_log(new_stride_log, new_baseline),
            offline_stride_reference_tolerances,
        )
    points = {}
    failures = []
    for hidden in (8, 16):
        old = old_paths(old_run, hidden)
        new = new_paths(new_run, hidden)
        old_metadata = (
            load_json(old["metadata"]) if old["metadata"].is_file() else None
        )
        new_metadata = (
            load_json(new["metadata"]) if new["metadata"].is_file() else None
        )
        metadata_rows = compare_mapping(
            old_metadata, new_metadata, {}, exact_metadata_fields
        ) if complete_artifact_set_available else []
        old_metrics = metrics_from_log(old["log"], old_baseline)
        new_metrics = metrics_from_log(new["log"], new_baseline)
        metric_rows = compare_mapping(
            old_metrics, new_metrics, system_tolerances
        ) if complete_artifact_set_available else []
        old_behavior = behavior_from_metadata(old["metadata"])
        new_behavior = behavior_from_metadata(new["metadata"])
        behavior_rows = compare_mapping(
            old_behavior, new_behavior, behavior_tolerances
        ) if complete_artifact_set_available else []
        observed_ipc = new_metrics.get("ipc") if new_metrics else None
        expected_ipc = HISTORICAL_IPC[hidden]
        anchor_tolerance = system_tolerances["ipc"]
        anchor = {
            "expected": expected_ipc,
            "observed": observed_ipc,
            "delta": (
                observed_ipc - expected_ipc
                if observed_ipc is not None else None
            ),
            "absolute_tolerance": anchor_tolerance,
            "status": (
                "PASS" if close(observed_ipc, expected_ipc, anchor_tolerance)
                else "FAIL"
            ),
        }
        point_failures = [
            row for row in metadata_rows + metric_rows + behavior_rows
            if row["status"] == "FAIL"
        ]
        if anchor["status"] == "FAIL":
            point_failures.append({"field": "historical_ipc_anchor"})
        failures.extend(
            "h{} {}".format(hidden, row["field"])
            for row in point_failures
        )
        points["h{}".format(hidden)] = {
            "metadata": metadata_rows,
            "system_metrics": metric_rows,
            "behavior": behavior_rows,
            "historical_ipc_anchor": anchor,
            "old_metadata_path": str(old["metadata"]),
            "new_metadata_path": str(new["metadata"]),
        }

    exact_failures = [
        item for item in artifacts
        if item["exact_required"] and item["status"] == "FAIL"
    ]
    failures.extend(item["name"] for item in exact_failures)
    for name, comparisons in reference_metrics.items():
        failures.extend(
            "{} reference {}".format(name, item["field"])
            for item in comparisons if item["status"] != "PASS"
        )
    if mode == "historical-metric regression only":
        exact_status = "NOT CLAIMED"
    else:
        exact_items = list(artifacts)
        metadata_exact = all(
            item["status"] == "PASS"
            for hidden in (8, 16)
            for item in points["h{}".format(hidden)]["metadata"]
        )
        exact_status = (
            "PASS" if exact_items and metadata_exact and all(
                item["status"] == "PASS" for item in exact_items
            ) else "NOT ESTABLISHED"
        )
    status = "FAIL" if failures else "PASS"
    report = {
        "schema_version": 2,
        "status": status,
        "mode": mode,
        "exact_artifact_parity_status": exact_status,
        "exact_artifact_parity_claimed": exact_status == "PASS",
        "old_run_dir": str(old_run),
        "new_run_dir": str(new_run),
        "historical_anchors": HISTORICAL_IPC,
        "artifact_comparisons": artifacts,
        "reference_metrics": reference_metrics,
        "reference_metric_policy": {
            "no_pref": (
                "exact parsed counters/metrics; coverage is not defined"
            ),
            "offline_stride": (
                "IPC/cache outcomes use the declared retrained-anchor "
                "tolerances; remaining counters/behavior metrics are exact"
            ),
        },
        "points": points,
        "failures": failures,
        "interpretation": (
            "Artifact equality and metric regression are reported separately; "
            "a tolerance-based system-metric PASS never establishes exact "
            "artifact parity; "
            "matching approximate IPC anchors alone never establishes exact "
            "artifact parity."
        ),
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_csv(output.with_suffix(".csv"), report)
    write_tex(output.with_suffix(".tex"), report)
    print("[{}] {}".format(status, mode))
    print("[report] {}".format(output))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
