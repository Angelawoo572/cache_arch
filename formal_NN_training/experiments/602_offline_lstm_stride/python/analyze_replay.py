#!/usr/bin/env python3
"""Verify and summarize the 602 offline-stride versus offline-LSTM replay."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


TRACE = "602.gcc_s-734B"
METHODS = ["no_pref", "live_stride_reference", "offline_stride", "offline_lstm"]
KV = re.compile(r"^([A-Za-z0-9_]+)\s+([-+0-9.eE]+)\s*$")
FINISHED = re.compile(
    r"Finished CPU\s+0\s+instructions:\s+(\d+)\s+cycles:\s+(\d+)\s+cumulative IPC:\s+([-+0-9.eE]+)"
)
REPLAYER = re.compile(
    r"emitted\s+(\d+)\s+candidates over\s+(\d+)\s+runtime ROI L2 LOAD accesses "
    r"\((\d+)\s+matched PC-line-occ triggers"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_log(path):
    result = {"ipc": 0.0, "instructions": 0, "cycles": 0, "emitted": 0, "callbacks": 0, "matched": 0}
    text = path.read_text(errors="ignore")
    for raw in text.splitlines():
        match = KV.match(raw.strip())
        if match:
            if match.group(1) == "Core_0_IPC":
                result["ipc"] = float(match.group(2))
            elif match.group(1) == "Core_0_instructions":
                result["instructions"] = int(float(match.group(2)))
            elif match.group(1) == "Core_0_cycles":
                result["cycles"] = int(float(match.group(2)))
        match = FINISHED.search(raw)
        if match:
            result["instructions"] = int(match.group(1))
            result["cycles"] = int(match.group(2))
            if not result["ipc"]:
                result["ipc"] = float(match.group(3))
        match = REPLAYER.search(raw)
        if match:
            result["emitted"] = int(match.group(1))
            result["callbacks"] = int(match.group(2))
            result["matched"] = int(match.group(3))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    args = ap.parse_args()
    logs = args.run_dir / "logs"
    lists = args.run_dir / "colab_output"
    rows = []
    failures = []
    for method in METHODS:
        path = logs / (TRACE + "." + method + ".log")
        if not path.is_file():
            failures.append("missing log {}".format(path))
            continue
        row = {"trace": TRACE, "method": method, "log": str(path)}
        row.update(parse_log(path))
        if row["ipc"] <= 0 or row["instructions"] <= 0:
            failures.append("{} lacks final simulator statistics".format(method))
        if method.startswith("offline_") and (row["matched"] <= 0 or row["emitted"] <= 0):
            failures.append("{} did not replay keyed entries".format(method))
        row["matched_primary_comparison"] = int(method in {"offline_stride", "offline_lstm"})
        rows.append(row)

    by_method = {row["method"]: row for row in rows}
    if len({row["instructions"] for row in rows}) != 1:
        failures.append("simulation instruction counts differ")
    for name in ("offline_stride.replay.csv", "offline_lstm.replay.csv", "run_metadata.json"):
        path = lists / name
        if not path.is_file():
            failures.append("missing Colab output {}".format(path))

    status = "FAIL" if failures else "PASS"
    fields = [
        "trace", "method", "matched_primary_comparison", "ipc", "instructions", "cycles",
        "matched", "emitted", "callbacks", "log",
    ]
    out_csv = args.run_dir / "matched_comparison.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "status": status,
        "fair_comparison_claim_allowed": not failures,
        "trace": TRACE,
        "primary_comparison": ["offline_stride", "offline_lstm"],
        "transport": "both lists produced causally from the same no-prefetch evaluation stream and replayed by the same PC-line-occ ListReplayer",
        "live_stride_role": "general reference only",
        "failures": failures,
        "rows": rows,
    }
    if not failures:
        stride = by_method["offline_stride"]["ipc"]
        lstm = by_method["offline_lstm"]["ipc"]
        payload.update(
            {
                "offline_stride_ipc": stride,
                "offline_lstm_ipc": lstm,
                "ipc_delta_lstm_minus_stride": lstm - stride,
                "lstm_beats_offline_stride": lstm > stride,
                "offline_stride_list_sha256": sha256(lists / "offline_stride.replay.csv"),
                "offline_lstm_list_sha256": sha256(lists / "offline_lstm.replay.csv"),
            }
        )
    out_json = args.run_dir / "matched_comparison.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("[{}] {}".format(status, out_json))
    if failures:
        raise SystemExit(" | ".join(failures))


if __name__ == "__main__":
    main()
