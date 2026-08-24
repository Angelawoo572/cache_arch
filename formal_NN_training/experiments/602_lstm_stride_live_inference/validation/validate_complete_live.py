#!/usr/bin/env python3
"""Verify that every exported h8/h16 checkpoint has a valid full live log."""

import argparse
import json
from pathlib import Path


REQUIRED_MARKERS = (
    "adding L2C_PREFETCHER: stride_lstm_live",
    "stride_lstm_live_weights frozen",
    "stride_lstm_live_measured_callbacks ",
)


def validate(run_dir, state_mode="parity", run_mode="live"):
    run_dir = Path(run_dir)
    points = []
    for metadata_path in sorted(
        run_dir.glob("points/h*/*/seed*/point_metadata.json")
    ):
        point_dir = metadata_path.parent
        model = point_dir / "export/model.bin"
        model_metadata = point_dir / "export/model_metadata.json"
        if not model.is_file():
            continue
        metadata = json.loads(metadata_path.read_text())
        hidden = metadata.get("hidden_size")
        if hidden not in (8, 16):
            continue
        log = point_dir / "live" / state_mode / run_mode / "run.log"
        missing_markers = list(REQUIRED_MARKERS)
        if log.is_file():
            text = log.read_text(errors="ignore")
            missing_markers = [
                marker for marker in REQUIRED_MARKERS if marker not in text
            ]
        points.append({
            "hidden_size": hidden,
            "budget_tag": metadata.get("budget_tag"),
            "seed": metadata.get("seed"),
            "model_bin": str(model),
            "model_metadata_available": model_metadata.is_file(),
            "live_log": str(log),
            "live_log_available": log.is_file(),
            "missing_required_markers": missing_markers,
            "status": (
                "PASS" if model_metadata.is_file() and not missing_markers
                else "FAIL"
            ),
        })
    failures = [point for point in points if point["status"] != "PASS"]
    by_hidden = {}
    for hidden in (8, 16):
        selected = [p for p in points if p["hidden_size"] == hidden]
        by_hidden["h{}".format(hidden)] = {
            "expected_exported_points": len(selected),
            "valid_live_points": sum(p["status"] == "PASS" for p in selected),
        }
    return {
        "schema_version": 1,
        "status": "PASS" if points and not failures else "FAIL",
        "state_mode": state_mode,
        "run_mode": run_mode,
        "expected_exported_point_count": len(points),
        "valid_live_point_count": len(points) - len(failures),
        "by_hidden_size": by_hidden,
        "failures": failures,
        "points": points,
        "missing_points_are_interpolated": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--state-mode", default="parity")
    parser.add_argument("--run-mode", default="live")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.run_dir, args.state_mode, args.run_mode)
    output = args.output or args.run_dir / "report/complete_live_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("[{}] complete live validation: {} of {} exported points".format(
        report["status"], report["valid_live_point_count"],
        report["expected_exported_point_count"],
    ))
    print("[report] {}".format(output))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
