#!/usr/bin/env python3
"""Aggregate frozen live ChampSim results and host inference counters."""

import argparse
from pathlib import Path

from result_utils import (
    base_point,
    load_json,
    parse_live_counters,
    parse_log,
    reference_logs,
    system_metrics,
    write_outputs,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser


def main():
    args = build_parser().parse_args()
    references = reference_logs(args.run_dir)
    baseline = references.get("no_pref")
    rows = []
    for log in sorted(args.run_dir.glob(
        "points/h*/*/seed*/live/*/*/run.log"
    )):
        point_dir = log.parents[3]
        metadata_path = point_dir / "point_metadata.json"
        export_path = point_dir / "export/model_metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = load_json(metadata_path)
        export = load_json(export_path) if export_path.is_file() else {}
        row = base_point(metadata)
        row["state_mode"] = log.parents[1].name
        row["live_run_mode"] = log.parent.name
        try:
            stats = parse_log(log)
            row.update(system_metrics(stats, baseline))
            row["status"] = (
                "live_smoke_complete"
                if row["live_run_mode"] == "smoke"
                else "live_run_complete"
            )
            row["failure_reason"] = None
        except Exception as exc:
            row.update(system_metrics(None, baseline))
            row["status"] = "live_run_failed"
            row["failure_reason"] = str(exc)
        counters = parse_live_counters(log)
        row.update({
            "live_model_sha256": export.get("export_sha256"),
            "live_model_format_version": export.get("format_version"),
            "live_inference_calls": counters.get(
                "stride_lstm_live_inference_calls"
            ),
            "live_generated_addresses": counters.get(
                "stride_lstm_live_generated_addresses"
            ),
            "live_unique_pcs": counters.get(
                "stride_lstm_live_unique_pcs"
            ),
            "live_peak_recurrent_state_bytes": counters.get(
                "stride_lstm_live_peak_recurrent_state_bytes"
            ),
            "live_weight_bytes": counters.get(
                "stride_lstm_live_weight_bytes"
            ),
            "live_total_deployment_bytes": counters.get(
                "stride_lstm_live_total_deployment_bytes"
            ),
            "host_total_nanoseconds": counters.get(
                "stride_lstm_live_host_total_ns"
            ),
            "host_mean_nanoseconds": counters.get(
                "stride_lstm_live_host_mean_ns"
            ),
            "host_p50_nanoseconds": counters.get(
                "stride_lstm_live_host_p50_ns"
            ),
            "host_p95_nanoseconds": counters.get(
                "stride_lstm_live_host_p95_ns"
            ),
            "host_p99_nanoseconds": counters.get(
                "stride_lstm_live_host_p99_ns"
            ),
            "host_maximum_nanoseconds": counters.get(
                "stride_lstm_live_host_max_ns"
            ),
            "host_events_per_second": counters.get(
                "stride_lstm_live_host_events_per_second"
            ),
            "live_log": str(log),
        })
        rows.append(row)
    rows.sort(key=lambda row: (
        row.get("hidden_size") or 0,
        row.get("instruction_budget") or 0,
        row.get("seed") or 0,
        row.get("state_mode") or "",
        row.get("live_run_mode") or "",
    ))
    write_outputs(
        rows,
        args.run_dir / "live_sweep_results.csv",
        args.run_dir / "live_sweep_results.json",
        references,
    )
    print("[ok] aggregated {} live results".format(len(rows)))


if __name__ == "__main__":
    main()
