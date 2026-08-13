#!/usr/bin/env python3
"""Report held-out silent/act rates for the four 602 direct-action tracks.

The report is deliberately callback-based: a callback is ``act`` when the
teacher or student emits at least one request, regardless of the learned count
K.  It therefore must not divide replay entry counts by callback counts.

By default the script reads the ``Default run`` named in each 602 experiment's
README and reports every model capacity found below its ``colab_output``
directory.  It supports the intentionally different metadata schemas used by
Stride, Streamer/AMPM, and SPP.  In particular, SPP uses demand callbacks as
the denominator; its ``eval_rows`` also includes cache-fill callbacks.
"""

from __future__ import print_function

import argparse
import csv
import json
import re
import sys
from pathlib import Path


TRACKS = {
    "ampm": "602_offline_lstm_ampm",
    "spp": "602_offline_lstm_spp",
    "streamer": "602_offline_lstm_streamer",
    "stride": "602_offline_lstm_stride",
}

OUTPUT_FIELDS = [
    "policy",
    "model",
    "parameters",
    "callbacks",
    "teacher_silent_rows",
    "teacher_silent_percent",
    "teacher_act_rows",
    "teacher_act_percent",
    "student_silent_rows",
    "student_silent_percent",
    "student_act_rows",
    "student_act_percent",
    "net_act_change_pp",
    "relative_act_change_percent",
    "newly_active_rows",
    "newly_active_percent",
    "suppressed_rows",
    "suppressed_percent",
    "metadata",
]


def repository_root():
    # <repo>/formal_NN_training/scripts/analysis/this_file.py
    return Path(__file__).resolve().parents[3]


def parse_default_run(readme_path):
    text = readme_path.read_text()
    match = re.search(r"Default run:\s*`([^`]+)`", text, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError("{} does not name a `Default run`".format(readme_path))
    return match.group(1)


def first_present(mapping, keys):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return int(mapping[key])
    return None


def extract_confusion(behavior):
    """Return TP/FP/FN when the metadata preserves callback confusion."""
    schemas = [
        (
            "gate_true_positive_rows",
            "gate_false_positive_rows",
            "gate_false_negative_rows",
        ),
        (
            "true_positive_trigger_callbacks",
            "false_positive_trigger_callbacks",
            "false_negative_trigger_callbacks",
        ),
    ]
    for tp_key, fp_key, fn_key in schemas:
        if all(key in behavior for key in (tp_key, fp_key, fn_key)):
            return (
                int(behavior[tp_key]),
                int(behavior[fp_key]),
                int(behavior[fn_key]),
            )
    return None


def extract_row(metadata, metadata_path):
    behavior = metadata.get("heldout_behavior_metrics", {})
    if not isinstance(behavior, dict):
        raise RuntimeError("heldout_behavior_metrics is not an object")

    policy = str(metadata.get("matched_normal_prefetcher", "")).lower()
    if policy not in TRACKS:
        raise RuntimeError("unsupported or missing matched_normal_prefetcher")

    # behavior_metrics.callbacks is the strongest cross-track denominator.
    # For SPP this is eval_demand_callbacks, not eval_rows (which includes
    # cache-fill callbacks).
    callbacks = first_present(
        behavior,
        ("callbacks",),
    )
    if callbacks is None:
        callbacks = first_present(
            metadata,
            ("eval_demand_callbacks", "eval_rows"),
        )
    if callbacks is None or callbacks <= 0:
        raise RuntimeError("missing or non-positive held-out callback count")

    teacher_act = first_present(
        behavior,
        ("gate_target_positive_rows", "normal_positive_callbacks"),
    )
    if teacher_act is None:
        summary = metadata.get("eval_teacher_summary", {})
        if isinstance(summary, dict):
            teacher_act = first_present(summary, ("trigger_rows",))
    if teacher_act is None:
        teacher_act = first_present(
            metadata,
            (
                "offline_{}_triggers".format(policy),
                "offline_normal_triggers",
            ),
        )

    student_act = first_present(
        behavior,
        ("gate_predicted_positive_rows", "predicted_positive_callbacks"),
    )
    if student_act is None:
        student_act = first_present(
            metadata,
            ("offline_lstm_triggers", "offline_nn_triggers"),
        )

    if teacher_act is None or student_act is None:
        raise RuntimeError("missing teacher/student positive callback count")
    if not 0 <= teacher_act <= callbacks:
        raise RuntimeError("teacher positive callbacks outside denominator")
    if not 0 <= student_act <= callbacks:
        raise RuntimeError("student positive callbacks outside denominator")

    confusion = extract_confusion(behavior)
    newly_active = None
    suppressed = None
    if confusion is not None:
        true_positive, newly_active, suppressed = confusion
        if teacher_act != true_positive + suppressed:
            raise RuntimeError("teacher trigger confusion identity failed")
        if student_act != true_positive + newly_active:
            raise RuntimeError("student trigger confusion identity failed")
        if student_act - teacher_act != newly_active - suppressed:
            raise RuntimeError("net trigger-change identity failed")

    def percent(value):
        return 100.0 * float(value) / float(callbacks)

    relative_change = None
    if teacher_act:
        relative_change = 100.0 * (
            float(student_act) / float(teacher_act) - 1.0
        )

    return {
        "policy": policy,
        "model": metadata_path.parent.name,
        "parameters": metadata.get(
            "parameter_count", metadata.get("parameters", "")
        ),
        "callbacks": callbacks,
        "teacher_silent_rows": callbacks - teacher_act,
        "teacher_silent_percent": percent(callbacks - teacher_act),
        "teacher_act_rows": teacher_act,
        "teacher_act_percent": percent(teacher_act),
        "student_silent_rows": callbacks - student_act,
        "student_silent_percent": percent(callbacks - student_act),
        "student_act_rows": student_act,
        "student_act_percent": percent(student_act),
        "net_act_change_pp": percent(student_act - teacher_act),
        "relative_act_change_percent": relative_change,
        "newly_active_rows": newly_active,
        "newly_active_percent": (
            percent(newly_active) if newly_active is not None else None
        ),
        "suppressed_rows": suppressed,
        "suppressed_percent": (
            percent(suppressed) if suppressed is not None else None
        ),
        "metadata": str(metadata_path),
    }


def hidden_width(model_name):
    matches = re.findall(r"(?:^|_)h(\d+)(?:_|$)", model_name)
    return int(matches[-1]) if matches else sys.maxsize


def collect_rows(repo_root, policies):
    experiments = repo_root / "formal_NN_training" / "experiments"
    rows = []
    for policy in policies:
        experiment = experiments / TRACKS[policy]
        run_name = parse_default_run(experiment / "README.md")
        output = experiment / "runs" / run_name / "colab_output"
        metadata_paths = sorted(
            output.glob("*/run_metadata.json"),
            key=lambda path: (hidden_width(path.parent.name), path.parent.name),
        )
        if not metadata_paths:
            raise RuntimeError(
                "no capacity metadata found below {}".format(output)
            )
        for metadata_path in metadata_paths:
            metadata = json.loads(metadata_path.read_text())
            row = extract_row(metadata, metadata_path)
            if row["policy"] != policy:
                raise RuntimeError(
                    "{}: policy {} does not match directory {}".format(
                        metadata_path, row["policy"], policy
                    )
                )
            rows.append(row)
    return rows


def display_number(value, digits=2):
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        return ("{:." + str(digits) + "f}").format(value)
    return str(value)


def print_table(rows):
    headers = [
        "policy",
        "model",
        "params",
        "callbacks",
        "teacher silent%",
        "teacher act%",
        "student silent%",
        "student act%",
        "act delta pp",
        "new act%",
        "suppressed%",
    ]
    table = []
    for row in rows:
        table.append([
            row["policy"],
            row["model"],
            display_number(row["parameters"], 0),
            display_number(row["callbacks"], 0),
            display_number(row["teacher_silent_percent"]),
            display_number(row["teacher_act_percent"]),
            display_number(row["student_silent_percent"]),
            display_number(row["student_act_percent"]),
            display_number(row["net_act_change_pp"]),
            display_number(row["newly_active_percent"]),
            display_number(row["suppressed_percent"]),
        ])
    widths = [len(header) for header in headers]
    for values in table:
        for index, value in enumerate(values):
            widths[index] = max(widths[index], len(value))
    print("  ".join(
        header.ljust(widths[index])
        for index, header in enumerate(headers)
    ))
    print("  ".join("-" * width for width in widths))
    for values in table:
        print("  ".join(
            value.ljust(widths[index])
            for index, value in enumerate(values)
        ))
    print()
    print("Rates use held-out decision callbacks; act means K > 0.")
    print("new act/suppressed require callback-confusion counters; n/a means")
    print("that the run stored only aggregate trigger counts.")


def print_csv(rows):
    writer = csv.DictWriter(sys.stdout, fieldnames=OUTPUT_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repository_root(),
        help="cache_arch checkout root (default: derived from this script)",
    )
    parser.add_argument(
        "--policy",
        action="append",
        choices=sorted(TRACKS),
        help="report one policy; repeat for several (default: all four)",
    )
    parser.add_argument(
        "--format",
        choices=("table", "csv", "json"),
        default="table",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    policies = args.policy or ["stride", "streamer", "ampm", "spp"]
    try:
        rows = collect_rows(args.repo_root.resolve(), policies)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        raise SystemExit("error: {}".format(error))

    if args.format == "table":
        print_table(rows)
    elif args.format == "csv":
        print_csv(rows)
    else:
        print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
