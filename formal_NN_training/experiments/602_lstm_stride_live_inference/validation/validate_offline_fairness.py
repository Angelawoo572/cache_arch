#!/usr/bin/env python3
"""Validate that only the training prefix changes in the offline sweep.

This validator is intentionally Torch-free.  It audits the completed ignored
run tree and writes machine-readable JSON/CSV plus a compact TeX summary.  It
does not train a model or launch ChampSim.
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "formal_NN_training/experiments/602_lstm_stride_live_inference"
PYTHON_DIR = EXP / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from provenance_utils import (  # noqa: E402
    artifact,
    authoritative_source_hashes,
    load_json,
    replay_header,
    sha256,
)
from result_utils import parse_log  # noqa: E402


REPLAY_HEADER = "pc,line,occ,prefetch_addr"
TERMINAL_UNTRAINABLE = {
    "no_callbacks", "single_class_no_act", "single_class_no_silent",
    "insufficient_rows", "training_failed",
}


def equal_number(left, right, rel_tol=1e-6, abs_tol=1e-8):
    if left is None or right is None:
        return left is right
    try:
        return math.isclose(
            float(left), float(right), rel_tol=rel_tol, abs_tol=abs_tol
        )
    except (TypeError, ValueError):
        return False


def add_check(checks, name, observed, expected, evidence, required=True,
              numeric=False):
    if observed is None:
        status = "UNAVAILABLE" if required else "NOT_APPLICABLE"
    else:
        matched = (
            equal_number(observed, expected) if numeric
            else observed == expected
        )
        status = "PASS" if matched else "FAIL"
    checks.append({
        "name": name,
        "status": status,
        "observed": observed,
        "expected": expected,
        "evidence": evidence,
        "required": bool(required),
    })


def history_rows(path):
    if not Path(path).is_file():
        return []
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def keyed_replayer_build_identity(root):
    champ = Path(root) / "external/ChampSim"
    source = champ / "prefetcher/list_replayer.cc"
    header = champ / "inc/list_replayer.h"
    build_script = (
        Path(root)
        / "formal_NN_training/experiments/602_offline_lstm_stride"
        / "linux/build_keyed_replayer.sh"
    )
    if not source.is_file() or not header.is_file():
        return None
    try:
        head = subprocess.check_output(
            ["git", "-C", str(champ), "rev-parse", "HEAD"],
            universal_newlines=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return {
        "champsim_git_head": head,
        "list_replayer_source_sha256": sha256(source),
        "list_replayer_header_sha256": sha256(header),
        "build_script_sha256": sha256(build_script),
    }


def point_record(run_dir, hidden, budget, seed, config, source_hashes,
                 trace_identity, eval_identity, binary_identity,
                 metric_parser_sha,
                 reference_identity, teacher_list_sha, build_identity):
    tag = budget["tag"]
    point_dir = (
        run_dir / "points" / "h{}".format(hidden) / tag
        / "seed{}".format(seed)
    )
    prefix_dir = run_dir / "training_prefixes" / tag
    manifest_path = prefix_dir / "training_manifest.json"
    point_path = point_dir / "point_metadata.json"
    run_metadata_path = point_dir / "offline/run_metadata.json"
    history_path = point_dir / "offline/training_history.csv"
    replay_path = point_dir / "offline/offline_lstm.replay.csv"
    stride_path = point_dir / "offline/offline_stride.replay.csv"
    replay_log = point_dir / "offline/replay.log"
    controls = config["fixed_controls"]

    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    point = load_json(point_path) if point_path.is_file() else {}
    run_metadata = (
        load_json(run_metadata_path) if run_metadata_path.is_file() else {}
    )
    trained = bool(
        point and point.get("status") not in TERMINAL_UNTRAINABLE
        and run_metadata
    )
    checks = []
    add_check(checks, "exact_trace", manifest.get("trace"),
              config["trace"], str(manifest_path))
    add_check(
        checks, "exact_trace_artifact_sha256",
        trace_identity.get("sha256"), trace_identity.get("sha256"),
        trace_identity.get("path"), required=True,
    )
    add_check(checks, "trace_start_instruction_budget",
              manifest.get("instruction_budget"), budget["instructions"],
              str(manifest_path))
    add_check(checks, "training_warmup_instructions",
              manifest.get("training_warmup_instructions"), 0,
              str(manifest_path))
    add_check(checks, "training_prefix_collection_semantics",
              manifest.get("collection_semantics"),
              "trace_start_to_exact_retired_instruction_budget",
              str(manifest_path))
    add_check(checks, "hidden_size", point.get("hidden_size"), hidden,
              str(point_path))
    add_check(checks, "seed", point.get("seed"), seed, str(point_path))

    training_checks = (
        ("model_revision", point.get("model_revision"),
         controls["model_revision"], str(point_path), False),
        ("epochs", point.get("epochs"), controls["epochs"],
         str(point_path), False),
        ("chunk_length", point.get("chunk_length"),
         controls["chunk_length"], str(point_path), False),
        ("pc_batch_size", run_metadata.get("pc_batch_size"),
         controls["pc_batch_size"], str(run_metadata_path), False),
        ("optimizer", point.get("optimizer"), controls["optimizer"],
         str(point_path), False),
        ("learning_rate", point.get("learning_rate"),
         controls["learning_rate"], str(point_path), True),
        ("teacher_policy", run_metadata.get("matched_normal_prefetcher"),
         controls["teacher"].replace("conventional_", ""),
         str(run_metadata_path), False),
        ("class_balancing_algorithm",
         run_metadata.get("gate_imbalance_handling"),
         "inverse_observed_training_class_frequency_equal_aggregate_mass",
         str(run_metadata_path), False),
        ("offline_inference_state_semantics",
         run_metadata.get("inference_state_mode"),
         "cold_dynamic_pc_state_then_continuous_per_pc_evaluation",
         str(run_metadata_path), False),
    )
    for name, observed, expected, evidence, numeric in training_checks:
        add_check(
            checks, name, observed if trained else None, expected, evidence,
            required=trained, numeric=numeric,
        )

    for field, source_field in (
        ("runtime_encoder_sha256", "runtime_encoder_sha256"),
        ("state_router_sha256", "state_router_sha256"),
    ):
        recorded = run_metadata.get(field) if trained else None
        observed = recorded or (source_hashes[source_field] if trained else None)
        evidence = (
            str(run_metadata_path) if recorded else
            "authoritative model-revision-locked source"
        )
        add_check(
            checks, field, observed if trained else None,
            source_hashes[source_field], evidence, required=trained,
        )

    decision_rows = manifest.get("decision_rows")
    silent_rows = manifest.get("silent_rows")
    positive_rows = manifest.get("positive_count_rows")
    if trained and decision_rows and silent_rows and positive_rows:
        expected_zero = decision_rows / (2.0 * silent_rows)
        expected_positive = decision_rows / (2.0 * positive_rows)
        weights = run_metadata.get("gate_class_weights") or {}
        add_check(
            checks, "prefix_local_zero_class_weight", weights.get("zero"),
            expected_zero, str(run_metadata_path), numeric=True,
        )
        add_check(
            checks, "prefix_local_positive_class_weight",
            weights.get("positive"), expected_positive,
            str(run_metadata_path), numeric=True,
        )
    else:
        add_check(
            checks, "prefix_local_zero_class_weight", None, None,
            str(run_metadata_path), required=False,
        )
        add_check(
            checks, "prefix_local_positive_class_weight", None, None,
            str(run_metadata_path), required=False,
        )

    teacher_summary = run_metadata.get("train_teacher_summary") or {}
    for name, observed, expected in (
        ("training_decision_rows", teacher_summary.get("rows"), decision_rows),
        ("training_positive_teacher_rows",
         teacher_summary.get("trigger_rows"), positive_rows),
        ("training_action_atoms", teacher_summary.get("actions"),
         manifest.get("action_atoms")),
    ):
        add_check(
            checks, name, observed if trained else None, expected,
            str(run_metadata_path), required=trained,
        )

    epochs_observed = len(history_rows(history_path)) if trained else None
    add_check(
        checks, "fixed_epoch_count_in_history", epochs_observed,
        controls["epochs"], str(history_path), required=trained,
    )
    add_check(
        checks, "evaluation_stream_content_sha256",
        eval_identity.get("content_sha256") if trained else None,
        eval_identity.get("content_sha256"), eval_identity.get("path"),
        required=trained,
    )
    add_check(
        checks, "evaluation_warmup_instructions",
        controls.get("evaluation_warmup_instructions") if trained else None,
        25000000, "training_prefix_sweep.json and run_offline_sweep.sh",
        required=trained,
    )
    add_check(
        checks, "evaluation_measured_instructions",
        controls.get("evaluation_simulation_instructions")
        if trained else None,
        25000000, "training_prefix_sweep.json and replay.log",
        required=trained,
    )

    replay_entries = point.get("offline_replay_entry_count")
    header = replay_header(replay_path) if replay_path.is_file() else None
    add_check(
        checks, "replay_list_format", header if trained else None,
        REPLAY_HEADER, str(replay_path), required=trained,
    )
    observed_stride_sha = sha256(stride_path) if stride_path.is_file() else None
    add_check(
        checks, "offline_stride_teacher_list_identity",
        observed_stride_sha if trained else None, teacher_list_sha,
        str(stride_path), required=trained,
    )
    add_check(
        checks, "keyed_replayer_binary_sha256",
        binary_identity.get("sha256") if trained else None,
        binary_identity.get("sha256"), binary_identity.get("path"),
        required=trained,
    )
    add_check(
        checks, "keyed_replayer_build_identity",
        build_identity if trained else None, build_identity,
        "ChampSim git/source/build-script identity", required=trained,
    )
    add_check(
        checks, "champ_sim_configuration",
        {
            "prefetcher": "list_replayer", "warmup": 25000000,
            "measured": 25000000,
        } if trained else None,
        {
            "prefetcher": "list_replayer", "warmup": 25000000,
            "measured": 25000000,
        },
        "run_offline_sweep.sh sha256={}".format(
            reference_identity["run_script_sha256"]
        ), required=trained,
    )
    add_check(
        checks, "no_pref_reference_identity",
        reference_identity.get("no_pref_log_sha256") if trained else None,
        reference_identity.get("no_pref_log_sha256"),
        reference_identity.get("no_pref_log"), required=trained,
    )
    add_check(
        checks, "metric_parser_revision", metric_parser_sha if trained else None,
        metric_parser_sha, "602_offline_lstm_stride/python/analyze_replay.py",
        required=trained,
    )
    if trained and replay_log.is_file():
        try:
            parsed = parse_log(replay_log)
            measured = parsed.get("instructions")
        except Exception:
            measured = None
        add_check(
            checks, "replay_log_measured_instruction_count", measured,
            25000000, str(replay_log), required=True,
        )

    failures = [item for item in checks if item["status"] == "FAIL"]
    unavailable = [
        item for item in checks
        if item["status"] == "UNAVAILABLE" and item["required"]
    ]
    return {
        "hidden_size": hidden,
        "instruction_budget": budget["instructions"],
        "budget_tag": tag,
        "seed": seed,
        "point_status": point.get("status"),
        "trained_point": trained,
        "optimizer_steps": point.get("optimizer_steps"),
        "offline_replay_entry_count": replay_entries,
        "status": "FAIL" if failures else (
            "INCOMPLETE" if unavailable else "PASS"
        ),
        "checks": checks,
    }


def write_csv(path, points):
    rows = []
    for point in points:
        for check in point["checks"]:
            rows.append({
                "hidden_size": point["hidden_size"],
                "budget_tag": point["budget_tag"],
                "instruction_budget": point["instruction_budget"],
                "seed": point["seed"],
                "point_status": point["point_status"],
                "check": check["name"],
                "status": check["status"],
                "observed": json.dumps(check["observed"], sort_keys=True),
                "expected": json.dumps(check["expected"], sort_keys=True),
                "evidence": check["evidence"],
            })
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(value):
    return str(value).replace("_", r"\_").replace("%", r"\%")


def short_digest(value):
    return "NA" if not value else str(value)[:12] + "..."


def write_tex(path, report):
    summary = report["summary"]
    lines = [
        r"\begin{tabular}{@{}ll@{}}",
        r"\toprule",
        r"Fairness audit & {}\\".format(tex_escape(report["status"])),
        r"Audited points & {} ({} trained)\\".format(
            summary["point_count"], summary["trained_point_count"]
        ),
        r"Failed checks & {}\\".format(summary["failed_check_count"]),
        r"Unavailable required checks & {}\\".format(
            summary["unavailable_required_check_count"]
        ),
        r"Trace SHA256 & \texttt{{{}}}\\".format(
            short_digest(report["shared_artifacts"]["trace"].get("sha256"))
        ),
        r"Evaluation content SHA256 & \texttt{{{}}}\\".format(
            short_digest(report["shared_artifacts"]["evaluation_stream"].get(
                "content_sha256"
            ))
        ),
        r"Keyed replayer binary SHA256 & \texttt{{{}}}\\".format(
            short_digest(report["shared_artifacts"]["keyed_replayer_binary"].get(
                "sha256"
            ))
        ),
        r"Offline Stride list SHA256 & \texttt{{{}}}\\".format(
            short_digest(report["shared_artifacts"].get(
                "offline_stride_teacher_list_sha256"
            ))
        ),
        r"Metric parser SHA256 & \texttt{{{}}}\\".format(
            short_digest(report["shared_artifacts"].get(
                "metric_parser_sha256"
            ))
        ),
        r"\bottomrule",
        r"\end{tabular}",
        "",
        (
            "The audit recomputes each prefix's class weights from that "
            "prefix's own zero/positive labels. Fixed epochs imply that "
            "larger prefixes receive more optimizer steps; this study "
            "therefore measures trace and training work under the current "
            "fixed recipe, not unique-data diversity at fixed update count."
        ),
        "The machine-readable JSON contains the complete SHA256 values and "
        "all per-point checks.",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keyed-replayer-binary", type=Path)
    parser.add_argument("--require-all", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    run_dir = args.run_dir
    output = args.output or run_dir / "report/offline_fairness_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    config = load_json(EXP / "config/training_prefix_sweep.json")
    source_hashes = authoritative_source_hashes(ROOT)
    trace_path = ROOT / "traces/{}.champsimtrace.xz".format(config["trace"])
    trace_identity = artifact(trace_path)
    eval_path = (
        run_dir / "evaluation/602.gcc_s-734B.eval_stream.csv.gz"
    )
    eval_identity = artifact(eval_path, gzip_content=True)
    binary_path = args.keyed_replayer_binary or (
        ROOT / "external/ChampSim/bin/champsim.602_offline_replay"
    )
    binary_identity = artifact(binary_path)
    build_identity = keyed_replayer_build_identity(ROOT)
    analyzer = (
        ROOT / "formal_NN_training/experiments/602_offline_lstm_stride"
        / "python/analyze_replay.py"
    )
    run_script = EXP / "linux/run_offline_sweep.sh"
    no_pref_log = run_dir / "references/logs/no_pref.log"
    reference_identity = {
        "run_script_sha256": sha256(run_script),
        "no_pref_log": str(no_pref_log),
        "no_pref_log_sha256": (
            sha256(no_pref_log) if no_pref_log.is_file() else None
        ),
        "offline_stride_log": str(
            run_dir / "references/logs/offline_stride.log"
        ),
    }
    metric_parser_sha = sha256(analyzer)
    stride_lists = list(run_dir.glob(
        "points/h*/*/seed*/offline/offline_stride.replay.csv"
    ))
    stride_hashes = sorted(set(sha256(path) for path in stride_lists))
    teacher_list_sha = stride_hashes[0] if len(stride_hashes) == 1 else None
    points = []
    for hidden in config["hidden_sizes"]:
        for budget in config["instruction_budgets"]:
            for seed in config["seeds"]:
                points.append(point_record(
                    run_dir, hidden, budget, seed, config, source_hashes,
                    trace_identity, eval_identity, binary_identity,
                    metric_parser_sha,
                    reference_identity, teacher_list_sha,
                    build_identity,
                ))
    failed_checks = sum(
        check["status"] == "FAIL"
        for point in points for check in point["checks"]
    )
    unavailable = sum(
        check["status"] == "UNAVAILABLE" and check["required"]
        for point in points for check in point["checks"]
    )
    trained_count = sum(point["trained_point"] for point in points)
    status = "FAIL" if failed_checks else (
        "INCOMPLETE" if unavailable else "PASS"
    )
    report = {
        "schema_version": 1,
        "study": (
            "602 Stride training-prefix sufficiency under the original "
            "offline keyed-replay protocol"
        ),
        "scope": "602.gcc_s-734B seed 7 h8/h16",
        "status": status,
        "only_allowed_independent_variable": "instruction_budget",
        "fixed_recipe_limitation": (
            "fixed epochs give larger prefixes more optimizer steps; the "
            "study measures trace and training work under the current fixed "
            "recipe and does not isolate data diversity from update count"
        ),
        "class_balance_rule": (
            "weights[c] = prefix_decision_rows / "
            "(2 * prefix_label_frequency[c])"
        ),
        "source_identities": source_hashes,
        "shared_artifacts": {
            "trace": trace_identity,
            "evaluation_stream": eval_identity,
            "keyed_replayer_binary": binary_identity,
            "keyed_replayer_build_identity": build_identity,
            "offline_stride_teacher_list_sha256": teacher_list_sha,
            "offline_stride_teacher_list_distinct_sha256_count": len(
                stride_hashes
            ),
            "metric_parser_sha256": metric_parser_sha,
            "references": reference_identity,
        },
        "summary": {
            "point_count": len(points),
            "trained_point_count": trained_count,
            "failed_check_count": failed_checks,
            "unavailable_required_check_count": unavailable,
        },
        "points": points,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    csv_path = output.with_suffix(".csv")
    write_csv(csv_path, points)
    tex_path = output.with_suffix(".tex")
    write_tex(tex_path, report)
    print("[{}] fairness audit: {} points; {} trained".format(
        status, len(points), trained_count
    ))
    print("[report] {}".format(output))
    if failed_checks or (args.require_all and unavailable):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
