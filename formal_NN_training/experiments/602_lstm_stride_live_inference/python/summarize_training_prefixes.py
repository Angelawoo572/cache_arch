#!/usr/bin/env python3
"""Aggregate exact-prefix manifests without dropping untrainable probes."""

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "budget_tag", "instruction_budget", "decision_rows", "silent_rows",
    "positive_count_rows", "action_atoms", "unique_pcs",
    "unique_positive_pcs", "k_histogram", "training_stream_sha256",
    "raw_event_log_sha256", "status", "failure_reason",
]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main():
    args = build_parser().parse_args()
    rows = []
    paths = list(args.prefix_root.glob("*/training_manifest.json"))
    if not paths:
        paths = list(args.prefix_root.glob("*/manifest.json"))
    for path in sorted(paths):
        payload = json.loads(path.read_text())
        row = {field: payload.get(field) for field in FIELDS}
        row["manifest_path"] = str(path)
        row["k_histogram"] = json.dumps(
            row["k_histogram"], sort_keys=True, separators=(",", ":")
        )
        rows.append(row)
    if not rows:
        raise RuntimeError("no training-prefix manifests below {}".format(
            args.prefix_root
        ))
    rows.sort(key=lambda row: int(row["instruction_budget"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "training_prefix_data_summary.csv"
    json_path = args.out_dir / "training_prefix_data_summary.json"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_rows = []
    for row in rows:
        copy = dict(row)
        copy["k_histogram"] = json.loads(copy["k_histogram"])
        json_rows.append(copy)
    json_path.write_text(json.dumps(
        {"schema_version": 1, "points": json_rows},
        indent=2, sort_keys=True,
    ) + "\n")
    print("[ok] wrote {} and {}".format(csv_path, json_path))


if __name__ == "__main__":
    main()
