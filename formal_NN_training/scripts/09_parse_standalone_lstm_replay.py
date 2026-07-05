#!/usr/bin/env python3
"""Summarize keyed standalone replay logs.

Two layouts are supported:
  1. scripts/08_run_standalone_lstm_replay.sh:
       --log-root + --replay-input-root
  2. scripts/11_run_prefetch_event_attribution.sh:
       --event-root
     This mode reads event-logging replay logs and the corresponding keyed-input
     metadata so one experiment root produces both counter and event evidence.

Both modes consume the same canonical replay-plan contract.
"""
from __future__ import print_function

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPLAY_HELPER_DIR = SCRIPT_DIR / "replay"
if str(REPLAY_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(REPLAY_HELPER_DIR))
from resolve_replay_plan import read_plan

spec = importlib.util.spec_from_file_location(
    "pythia_stats", str(SCRIPT_DIR / "01_parse_prefetch_behavior_audit.py")
)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import normal-log parser")
pythia_stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pythia_stats)

LOADED_RE = re.compile(r"\[list_replayer\] loaded\s+(\d+)\s+prefetch entries across\s+(\d+)\s+PC-line-occ triggers")
FINAL_RE = re.compile(r"\[list_replayer\] emitted\s+(\d+)\s+candidates over\s+(\d+)\s+runtime ROI L2 LOAD accesses\s+\((\d+)\s+matched PC-line-occ triggers;\s+(\d+)\s+loaded trigger keys;\s+key=pc_line_occ\)")


def to_float(value, default=0.0):
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def div(numerator, denominator):
    denominator = float(denominator)
    return float(numerator) / denominator if denominator else 0.0


def read_rows(path):
    if path is None or not Path(path).is_file():
        return []
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def baseline_by_trace(rows):
    output = {}
    for row in rows:
        trace = row.get("trace", "")
        prefetcher = row.get("prefetcher", "")
        if not trace:
            continue
        entry = output.setdefault(trace, {
            "no_pref_ipc": 0.0, "best_normal": "", "best_normal_ipc": 0.0,
        })
        ipc = to_float(row.get("ipc"))
        if prefetcher in ("no_pref", "none", "nopref"):
            entry["no_pref_ipc"] = ipc
        elif not to_float(row.get("run_failed")) and ipc > entry["best_normal_ipc"]:
            entry["best_normal"] = prefetcher
            entry["best_normal_ipc"] = ipc
    return output


def replayer_stats(log):
    output = {
        "list_replayer_instantiated": 0,
        "list_loaded_entries": 0,
        "list_loaded_trigger_keys": 0,
        "replayer_emitted_candidates": 0,
        "replayer_runtime_l2_loads": 0,
        "replayer_matched_trigger_keys": 0,
        "replayer_final_loaded_trigger_keys": 0,
    }
    if not log.is_file():
        return output
    with log.open(errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if "adding L2C_PREFETCHER: list_replayer" in line:
                output["list_replayer_instantiated"] = 1
            match = LOADED_RE.search(line)
            if match:
                output["list_loaded_entries"] = int(match.group(1))
                output["list_loaded_trigger_keys"] = int(match.group(2))
            match = FINAL_RE.search(line)
            if match:
                output["replayer_emitted_candidates"] = int(match.group(1))
                output["replayer_runtime_l2_loads"] = int(match.group(2))
                output["replayer_matched_trigger_keys"] = int(match.group(3))
                output["replayer_final_loaded_trigger_keys"] = int(match.group(4))
    return output


def load_metadata(path):
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def paths_for_entry(entry, args):
    tag = entry["tag"]
    trace = entry["trace"]
    if args.event_root is not None:
        root = args.event_root
        return (
            root / "lstm" / tag / "logs" / (trace + ".standalone_lstm.log"),
            root / "replay_inputs" / tag / (trace + ".pc_line_occ.csv.meta.json"),
            "event_attribution_layout",
        )
    return (
        args.log_root / (tag + ".standalone_lstm.log"),
        args.replay_input_root / (tag + ".pc_line_occ.csv.meta.json"),
        "standalone_replay_layout",
    )


def enrich(base, replay, metadata, normal, same, entry, layout):
    ipc = to_float(base.get("ipc"))
    row = dict(base)
    row.update(replay)
    for name in (
        "entries", "unique_trigger_keys", "unmatched_rows", "dropped_invalid_address",
        "direct_index_rows", "direct_index_rows_verified_pc_line",
        "mapped_cycle_pc_line_rows",
    ):
        if name in metadata:
            row["keyed_" + name] = metadata[name]
    row.update(entry)
    row["candidate_tag"] = entry["tag"]
    row["standalone_variant"] = entry["tag"]
    row["rich_list"] = entry["rich_list"]
    row["replay_layout"] = layout
    row["replay_ipc"] = ipc
    row["no_pref_ipc"] = normal.get("no_pref_ipc", 0.0)
    row["best_normal"] = normal.get("best_normal", "")
    row["best_normal_ipc"] = normal.get("best_normal_ipc", 0.0)
    row["speedup_vs_no_pref"] = div(ipc, row["no_pref_ipc"])
    row["speedup_vs_best_normal"] = div(ipc, row["best_normal_ipc"])
    row["ipc_delta_vs_best_normal"] = ipc - to_float(row["best_normal_ipc"])
    row["keyed_trigger_coverage"] = div(
        replay["replayer_matched_trigger_keys"], metadata.get("unique_trigger_keys", 0)
    )
    row["replay_transport_ok"] = int(
        replay["list_replayer_instantiated"] == 1
        and replay["list_loaded_entries"] == metadata.get("entries", -1)
        and replay["list_loaded_trigger_keys"] == metadata.get("unique_trigger_keys", -1)
        and replay["replayer_final_loaded_trigger_keys"] == metadata.get("unique_trigger_keys", -1)
        and metadata.get("unmatched_rows", -1) == 0
        and int(to_float(base.get("run_failed"))) == 0
    )
    if same is not None:
        same_ipc = to_float(same.get("ipc"))
        row["same_binary_no_pref_ipc"] = same_ipc
        row["speedup_vs_same_binary_no_pref"] = div(ipc, same_ipc)
        row["ipc_delta_vs_same_binary_no_pref"] = ipc - same_ipc
        row["same_binary_no_pref_run_failed"] = int(to_float(same.get("run_failed")))
    return row


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({name for row in rows for name in row}) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--baseline-summary", type=Path, default=None)
    parser.add_argument("--same-binary-log-root", type=Path, default=None)
    parser.add_argument("--replay-plan", required=True, type=Path)
    parser.add_argument("--plan-root", type=Path, default=None)
    parser.add_argument("--winner-out", type=Path, default=None)
    parser.add_argument("--log-root", type=Path, default=None)
    parser.add_argument("--replay-input-root", type=Path, default=None)
    parser.add_argument("--event-root", type=Path, default=None)
    args = parser.parse_args()

    if args.event_root is None:
        if args.log_root is None or args.replay_input_root is None:
            parser.error("supply --event-root, or both --log-root and --replay-input-root")
    else:
        if not args.event_root.is_dir():
            parser.error("--event-root does not exist: {}".format(args.event_root))
        if args.baseline_summary is None:
            args.baseline_summary = args.event_root / "normal" / "summary.csv"
    if args.baseline_summary is None or not args.baseline_summary.is_file():
        parser.error("missing normal baseline summary: {}".format(args.baseline_summary))

    root = args.plan_root or args.replay_plan.parent
    entries = read_plan(args.replay_plan.resolve(), root.resolve())
    baselines = baseline_by_trace(read_rows(args.baseline_summary))
    rows = []
    for entry in entries:
        log, metadata_path, layout = paths_for_entry(entry, args)
        base = pythia_stats.summarize_one(
            entry["trace"], entry.get("model_family") or "standalone_lstm", log, nodup=True
        )
        replay = replayer_stats(log)
        metadata = load_metadata(metadata_path)
        same = None
        if args.same_binary_log_root is not None:
            same_log = args.same_binary_log_root / (entry["trace"] + ".same_binary_no_pref.log")
            same = pythia_stats.summarize_one(
                entry["trace"], "same_binary_no_pref", same_log, nodup=True
            )
        rows.append(enrich(base, replay, metadata, baselines.get(entry["trace"], {}), same, entry, layout))

    write_rows(args.out, rows)
    if args.winner_out is not None:
        winners = []
        for trace in sorted({row.get("trace", "") for row in rows if row.get("trace", "")}):
            candidates = [
                row for row in rows
                if row.get("trace") == trace
                and int(to_float(row.get("replay_transport_ok"))) == 1
                and int(to_float(row.get("run_failed"))) == 0
            ]
            if not candidates:
                continue
            candidates.sort(key=lambda row: (-to_float(row.get("replay_ipc")), row.get("candidate_tag", "")))
            winner = dict(candidates[0])
            winner["winner_rank_within_trace"] = 1
            winner["winner_selection"] = "max_replay_ipc_among_current_run_nn_candidates"
            winner["beats_best_normal"] = int(
                to_float(winner.get("replay_ipc")) > to_float(winner.get("best_normal_ipc"))
            )
            winners.append(winner)
        write_rows(args.winner_out, winners)
        print("[write] {}".format(args.winner_out))
    print("[write] {}".format(args.out))


if __name__ == "__main__":
    main()
