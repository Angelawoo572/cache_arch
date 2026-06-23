#!/usr/bin/env python3
"""Summarize standalone keyed-LSTM replay logs."""
from __future__ import print_function
import argparse
import csv
import importlib.util
import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pythia_stats", str(SCRIPT_DIR / "01_parse_prefetch_behavior_audit.py"))
pythia_stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pythia_stats)

LOADED_RE = re.compile(r"\[list_replayer\] loaded\s+(\d+)\s+prefetch entries across\s+(\d+)\s+PC-line-occ triggers")
FINAL_RE = re.compile(r"\[list_replayer\] emitted\s+(\d+)\s+candidates over\s+(\d+)\s+runtime ROI L2 LOAD accesses\s+\((\d+)\s+matched PC-line-occ triggers;\s+(\d+)\s+loaded trigger keys;\s+key=pc_line_occ\)")

def to_float(x, default=0.0):
    try:
        return float(x) if x not in (None, "") else default
    except Exception:
        return default

def div(a, b):
    return float(a) / float(b) if float(b) else 0.0

def read_rows(path):
    if not path or not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def baseline_by_trace(rows):
    out = {}
    for row in rows:
        trace = row.get("trace", "")
        pf = row.get("prefetcher", "")
        ipc = to_float(row.get("ipc"))
        if not trace:
            continue
        cur = out.setdefault(trace, {"no_pref_ipc": 0.0, "best_normal": "", "best_normal_ipc": 0.0})
        if pf in {"no_pref", "none", "nopref"}:
            cur["no_pref_ipc"] = ipc
        elif ipc > cur["best_normal_ipc"]:
            cur["best_normal"] = pf
            cur["best_normal_ipc"] = ipc
    return out

def replayer_stats(log):
    out = {"list_replayer_instantiated": 0, "list_loaded_entries": 0, "list_loaded_trigger_keys": 0,
           "replayer_emitted_candidates": 0, "replayer_runtime_l2_loads": 0,
           "replayer_matched_trigger_keys": 0, "replayer_final_loaded_trigger_keys": 0}
    if not log.is_file():
        return out
    for raw in log.open(errors="ignore"):
        line = raw.strip()
        if "adding L2C_PREFETCHER: list_replayer" in line:
            out["list_replayer_instantiated"] = 1
        match = LOADED_RE.search(line)
        if match:
            out["list_loaded_entries"] = int(match.group(1))
            out["list_loaded_trigger_keys"] = int(match.group(2))
        match = FINAL_RE.search(line)
        if match:
            out["replayer_emitted_candidates"] = int(match.group(1))
            out["replayer_runtime_l2_loads"] = int(match.group(2))
            out["replayer_matched_trigger_keys"] = int(match.group(3))
            out["replayer_final_loaded_trigger_keys"] = int(match.group(4))
    return out

def load_meta(path):
    if not path.is_file():
        return {}
    return json.loads(path.read_text())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", required=True, type=Path)
    ap.add_argument("--replay-input-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--baseline-summary", required=True, type=Path)
    ap.add_argument("--same-binary-log-root", type=Path, default=None)
    args = ap.parse_args()

    baselines = baseline_by_trace(read_rows(args.baseline_summary))
    rows = []
    for trace in args.traces.split():
        log = args.log_root / (trace + ".standalone_lstm.log")
        base = pythia_stats.summarize_one(trace, "standalone_lstm", log, nodup=True)
        replay = replayer_stats(log)
        meta = load_meta(args.replay_input_root / (trace + ".pc_line_occ.csv.meta.json"))
        normal = baselines.get(trace, {})
        ipc = to_float(base.get("ipc"))
        row = dict(base)
        row.update(replay)
        row.update({"keyed_" + k: v for k, v in meta.items() if k in {"entries", "unique_trigger_keys", "unmatched_rows", "dropped_invalid_address"}})
        row["no_pref_ipc"] = normal.get("no_pref_ipc", 0.0)
        row["best_normal"] = normal.get("best_normal", "")
        row["best_normal_ipc"] = normal.get("best_normal_ipc", 0.0)
        row["speedup_vs_no_pref"] = div(ipc, row["no_pref_ipc"])
        row["speedup_vs_best_normal"] = div(ipc, row["best_normal_ipc"])
        row["keyed_trigger_coverage"] = div(replay["replayer_matched_trigger_keys"], meta.get("unique_trigger_keys", 0))
        row["replay_transport_ok"] = int(replay["list_replayer_instantiated"] == 1 and replay["list_loaded_entries"] == meta.get("entries", -1) and replay["list_loaded_trigger_keys"] == meta.get("unique_trigger_keys", -1) and replay["replayer_final_loaded_trigger_keys"] == meta.get("unique_trigger_keys", -1) and meta.get("unmatched_rows", -1) == 0 and not base.get("run_failed", 0))
        if args.same_binary_log_root is not None:
            same_log = args.same_binary_log_root / (trace + ".same_binary_no_pref.log")
            same = pythia_stats.summarize_one(trace, "same_binary_no_pref", same_log, nodup=True)
            same_ipc = to_float(same.get("ipc"))
            row["same_binary_no_pref_ipc"] = same_ipc
            row["speedup_vs_same_binary_no_pref"] = div(ipc, same_ipc)
            row["same_binary_no_pref_run_failed"] = int(same.get("run_failed", 0))
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("[write] {}".format(args.out))

if __name__ == "__main__":
    main()
