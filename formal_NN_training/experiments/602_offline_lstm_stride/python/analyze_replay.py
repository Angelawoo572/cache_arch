#!/usr/bin/env python3
"""Verify and summarize the isolated 602 offline stride/LSTM capacity sweep."""
import argparse
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path


TRACE = "602.gcc_s-734B"
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


def parse_events(path):
    """Read post-warmup L2 demand/PF records emitted by patch_demand_logger.sh."""
    result = {
        "demand_l2_loads": 0,
        "demand_l2_hits": 0,
        "demand_l2_misses": 0,
        "prefetch_useful_demand_hits": 0,
        "prefetch_late_demand_misses": 0,
        "prefetch_request_events": 0,
    }
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"event", "hit", "was_prefetch", "late"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError("{} missing event columns {}".format(path, sorted(missing)))
        for row in reader:
            if row["event"] == "PF":
                result["prefetch_request_events"] += 1
                continue
            if row["event"] != "DEMAND":
                continue
            result["demand_l2_loads"] += 1
            hit = int(row["hit"])
            if hit:
                result["demand_l2_hits"] += 1
            else:
                result["demand_l2_misses"] += 1
            if int(row["was_prefetch"]):
                result["prefetch_useful_demand_hits"] += 1
            if int(row["late"]):
                result["prefetch_late_demand_misses"] += 1
    return result


def model_tag_to_hidden(tag):
    if not tag.startswith("h") or not tag[1:].isdigit():
        raise ValueError("model tag must be h<positive integer>: {}".format(tag))
    return int(tag[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--model-tags", default="h8,h16,h32,h64,h128")
    ap.add_argument("--base-model-tag", default="h8")
    args = ap.parse_args()
    model_tags = [x.strip() for x in args.model_tags.split(",") if x.strip()]
    if not model_tags or args.base_model_tag not in model_tags:
        raise SystemExit("base model tag must be one of --model-tags")
    methods = ["no_pref", "live_stride_reference", "offline_stride"]
    methods += ["offline_lstm_" + tag for tag in model_tags]
    logs = args.run_dir / "logs"
    events = args.run_dir / "events"
    colab_root = args.run_dir / "colab_output"
    rows = []
    failures = []

    for method in methods:
        log_path = logs / (TRACE + "." + method + ".log")
        event_path = events / (TRACE + "." + method + ".events.csv.gz")
        if not log_path.is_file():
            failures.append("missing log {}".format(log_path))
            continue
        if not event_path.is_file():
            failures.append("missing event log {}".format(event_path))
            continue
        row = {"trace": TRACE, "method": method, "log": str(log_path), "event_log": str(event_path)}
        row.update(parse_log(log_path))
        try:
            row.update(parse_events(event_path))
        except Exception as exc:
            failures.append("{} event parse failed: {}".format(method, exc))
        if row["ipc"] <= 0 or row["instructions"] <= 0:
            failures.append("{} lacks final simulator statistics".format(method))
        if method == "live_stride_reference" and row["prefetch_request_events"] <= 0:
            failures.append("live_stride_reference emitted no PF requests")
        if method.startswith("offline_") and method != "offline_stride" and (row["matched"] <= 0 or row["emitted"] <= 0):
            failures.append("{} did not replay keyed entries".format(method))
        if method == "offline_stride" and (row["matched"] <= 0 or row["emitted"] <= 0):
            failures.append("offline_stride did not replay keyed entries")
        row["matched_primary_comparison"] = int(method == "offline_stride" or method.startswith("offline_lstm_"))
        row["model_tag"] = method[len("offline_lstm_"):] if method.startswith("offline_lstm_") else ""
        row["hidden_size"] = model_tag_to_hidden(row["model_tag"]) if row["model_tag"] else 0
        rows.append(row)

    by_method = {row["method"]: row for row in rows}
    if len(rows) != len(methods):
        failures.append("one or more methods are missing")
    if rows and len({row["instructions"] for row in rows}) != 1:
        failures.append("simulation instruction counts differ")
    expected_train_stream = args.run_dir / "colab_input" / (TRACE + ".train_stream.csv.gz")
    expected_eval_stream = args.run_dir / "colab_input" / (TRACE + ".eval_stream.csv.gz")
    expected_train_sha = sha256(expected_train_stream) if expected_train_stream.is_file() else None
    expected_eval_sha = sha256(expected_eval_stream) if expected_eval_stream.is_file() else None
    if expected_train_sha is None or expected_eval_sha is None:
        failures.append("missing normalized Colab input streams under {}".format(args.run_dir / "colab_input"))
    for tag in model_tags:
        for name in ("offline_lstm.replay.csv", "model.pt", "run_metadata.json"):
            path = colab_root / tag / name
            if not path.is_file():
                failures.append("missing Colab output {}".format(path))
        metadata_path = colab_root / tag / "run_metadata.json"
        if metadata_path.is_file() and expected_train_sha is not None and expected_eval_sha is not None:
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("train_stream_sha256") != expected_train_sha:
                failures.append("{} training-stream SHA256 does not match this run".format(tag))
            if metadata.get("eval_stream_sha256") != expected_eval_sha:
                failures.append("{} evaluation-stream SHA256 does not match this run".format(tag))

    if "no_pref" in by_method:
        no_pref_misses = by_method["no_pref"]["demand_l2_misses"]
        for row in rows:
            useful = row["prefetch_useful_demand_hits"]
            row["prefetch_coverage_vs_no_pref_misses"] = useful / float(no_pref_misses) if no_pref_misses else 0.0
            row["l2_miss_reduction_vs_no_pref"] = (
                (no_pref_misses - row["demand_l2_misses"]) / float(no_pref_misses) if no_pref_misses else 0.0
            )

    status = "FAIL" if failures else "PASS"
    fields = [
        "trace", "method", "model_tag", "hidden_size", "matched_primary_comparison",
        "ipc", "instructions", "cycles", "matched", "emitted", "callbacks",
        "demand_l2_loads", "demand_l2_hits", "demand_l2_misses",
        "prefetch_useful_demand_hits", "prefetch_late_demand_misses",
        "prefetch_request_events", "prefetch_coverage_vs_no_pref_misses",
        "l2_miss_reduction_vs_no_pref", "log", "event_log",
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
        "primary_comparison": ["offline_stride", "offline_lstm_<capacity>"],
        "transport": "both lists produced causally from the same no-prefetch evaluation stream and replayed by the same PC-line-occ ListReplayer",
        "live_stride_role": "validated general reference only; it is not the offline primary comparator",
        "metric_definitions": {
            "prefetch_useful_demand_hits": "post-warmup L2 load hits on a line marked as prefetched",
            "prefetch_late_demand_misses": "post-warmup L2 load misses merged into an in-flight prefetch MSHR",
            "prefetch_coverage_vs_no_pref_misses": "useful prefetch demand hits divided by no-prefetch L2 load misses",
            "l2_miss_reduction_vs_no_pref": "(no-prefetch L2 load misses - method L2 load misses) / no-prefetch L2 load misses",
        },
        "model_tags": model_tags,
        "failures": failures,
        "rows": rows,
    }
    if not failures:
        stride = by_method["offline_stride"]["ipc"]
        sweep = []
        first_better = None
        for tag in model_tags:
            row = by_method["offline_lstm_" + tag]
            metadata = json.loads((colab_root / tag / "run_metadata.json").read_text())
            point = {
                "model_tag": tag,
                "hidden_size": metadata["hidden_size"],
                "parameter_count": metadata["parameter_count"],
                "ipc": row["ipc"],
                "ipc_delta_lstm_minus_stride": row["ipc"] - stride,
                "lstm_beats_offline_stride": row["ipc"] > stride,
                "prefetch_coverage_vs_no_pref_misses": row["prefetch_coverage_vs_no_pref_misses"],
                "prefetch_useful_demand_hits": row["prefetch_useful_demand_hits"],
                "prefetch_late_demand_misses": row["prefetch_late_demand_misses"],
            }
            sweep.append(point)
            if first_better is None and point["lstm_beats_offline_stride"]:
                first_better = point
        payload.update(
            {
                "offline_stride_ipc": stride,
                "capacity_sweep": sweep,
                "first_measured_capacity_beating_offline_stride": first_better,
                "saturation_interpretation": (
                    "The first measured capacity above offline stride is exploratory because "
                    "the evaluation stream is reused across capacity points; confirm it on a "
                    "fresh held-out window before selecting a final capacity."
                ),
                "offline_stride_list_sha256": sha256(colab_root / args.base_model_tag / "offline_stride.replay.csv"),
            }
        )
    out_json = args.run_dir / "matched_comparison.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("[{}] {}".format(status, out_json))
    if failures:
        raise SystemExit(" | ".join(failures))


if __name__ == "__main__":
    main()
