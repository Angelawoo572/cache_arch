#!/usr/bin/env python3
"""Summarize PC-line-occurrence keyed LSTM ListReplayer runs.

This script reuses the generic Pythia counter parser in
01_parse_prefetch_behavior_audit.py. It adds keyed-replay transport fields and
comparisons against no-prefetch / best-normal IPC.

The replay is an offline policy replay keyed by (pc,line,occ), not an
in-simulator neural inference result. `replay_transport_ok=1` means the keyed
list was fully loaded and the simulator produced parseable nonzero keyed replay
counters; it does not claim global callback order is invariant under prefetching.
"""

from __future__ import print_function

import argparse
import csv
import importlib.util
import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_BEHAVIOR_PATH = SCRIPT_DIR / "01_parse_prefetch_behavior_audit.py"
_spec = importlib.util.spec_from_file_location("behavior_audit_parser", str(_BEHAVIOR_PATH))
_behavior = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_behavior)

LOADED_RE = re.compile(
    r"\[list_replayer\] loaded\s+(\d+)\s+prefetch entries across\s+(\d+)\s+PC-line-occ triggers"
)
FINAL_RE = re.compile(
    r"\[list_replayer\] emitted\s+(\d+)\s+candidates over\s+(\d+)\s+runtime ROI L2 LOAD accesses\s+"
    r"\((\d+)\s+matched PC-line-occ triggers;\s+(\d+)\s+loaded trigger keys;\s+key=pc_line_occ\)"
)
VALIDATED_RE = re.compile(r"\[oracle-replay-validation\]\s+status=(keyed_transport_pass|keyed_transport_fail)")


def to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def to_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def div(a, b):
    return float(a) / float(b) if float(b) else 0.0


def read_csv_rows(path):
    if not path or not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def choose_number(row, keys, default=0.0):
    for key in keys:
        if key in row and row.get(key) not in (None, ""):
            return to_float(row.get(key), default)
    return default


def parse_replayer_log(path):
    out = {
        "list_replayer_instantiated": 0,
        "validation_marker": "",
        "list_loaded_entries": 0,
        "list_loaded_trigger_keys": 0,
        "replayer_emitted_candidates": 0,
        "replayer_runtime_l2_loads": 0,
        "replayer_matched_trigger_keys": 0,
        "replayer_final_loaded_trigger_keys": 0,
    }
    if not path.is_file():
        return out

    with path.open(errors="ignore") as f:
        for raw in f:
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

            match = VALIDATED_RE.search(line)
            if match:
                out["validation_marker"] = match.group(1)
    return out


def load_keyed_meta(path):
    out = {
        "keyed_entries": 0,
        "keyed_unique_trigger_keys": 0,
        "keyed_oracle_rows": 0,
        "keyed_unmatched_rows": 0,
        "keyed_dropped_invalid_address": 0,
    }
    if not path.is_file():
        return out
    try:
        data = json.loads(path.read_text())
    except Exception:
        return out
    out["keyed_entries"] = to_int(data.get("entries"))
    out["keyed_unique_trigger_keys"] = to_int(data.get("unique_trigger_keys"))
    out["keyed_oracle_rows"] = to_int(data.get("oracle_rows"))
    out["keyed_unmatched_rows"] = to_int(data.get("unmatched_rows"))
    out["keyed_dropped_invalid_address"] = to_int(data.get("dropped_invalid_address"))
    return out


def make_baselines(rows):
    by_trace = {}
    for row in rows:
        trace = row.get("trace", "")
        pf = row.get("prefetcher", "")
        ipc = choose_number(row, ["behavior_ipc", "ipc"])
        if not trace:
            continue
        ent = by_trace.setdefault(trace, {"no_pref_ipc": 0.0, "best_normal": "", "best_normal_ipc": 0.0})
        if pf in ("no_pref", "none", "nopref"):
            ent["no_pref_ipc"] = ipc
        elif ipc > ent["best_normal_ipc"]:
            ent["best_normal_ipc"] = ipc
            ent["best_normal"] = pf
    return by_trace


def offline_by_trace(rows):
    return {row.get("trace", ""): row for row in rows if row.get("trace", "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", required=True, type=Path)
    ap.add_argument("--replay-input-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--traces", required=True, help="space-separated trace names")
    ap.add_argument("--baseline-metrics", type=Path, default=None,
                    help="normal_prefetcher_metrics.csv; used only for IPC comparisons")
    ap.add_argument("--offline-summary", type=Path, default=None,
                    help="notebook sweep CSV; joined only as clearly prefixed offline_* fields")
    ap.add_argument("--log-suffix", default=".oracle_replacer.log")
    args = ap.parse_args()

    baselines = make_baselines(read_csv_rows(args.baseline_metrics))
    offline = offline_by_trace(read_csv_rows(args.offline_summary))
    rows = []

    for trace in args.traces.split():
        log = args.log_root / (trace + args.log_suffix)
        base = _behavior.summarize_one(trace, "oracle_lstm", log, nodup=True)
        replay = parse_replayer_log(log)
        meta = load_keyed_meta(args.replay_input_root / (trace + ".pc_line_occ.csv.meta.json"))
        normal = baselines.get(trace, {})
        off = offline.get(trace, {})

        no_pref_ipc = to_float(normal.get("no_pref_ipc"))
        best_normal_ipc = to_float(normal.get("best_normal_ipc"))
        ipc = to_float(base.get("ipc"))

        row = {}
        row.update(base)
        row.update(replay)
        row.update(meta)
        row["no_pref_ipc"] = no_pref_ipc
        row["best_normal"] = normal.get("best_normal", "")
        row["best_normal_ipc"] = best_normal_ipc
        row["speedup_vs_no_pref"] = div(ipc, no_pref_ipc)
        row["speedup_vs_best_normal"] = div(ipc, best_normal_ipc)
        row["keyed_trigger_coverage"] = div(
            row.get("replayer_matched_trigger_keys"), row.get("keyed_unique_trigger_keys")
        )

        row["replay_transport_ok"] = int(
            row.get("validation_marker") == "keyed_transport_pass"
            and row.get("list_replayer_instantiated") == 1
            and row.get("list_loaded_entries") == row.get("keyed_entries")
            and row.get("list_loaded_trigger_keys") == row.get("keyed_unique_trigger_keys")
            and row.get("replayer_final_loaded_trigger_keys") == row.get("keyed_unique_trigger_keys")
            and row.get("keyed_unmatched_rows") == 0
            and int(row.get("run_failed", 0)) == 0
        )

        for key in (
            "chunk_len", "emit_mode", "emit_rate", "addr_supervision_rate",
            "policy_emit", "policy_addr_correct", "policy_precision",
            "issue_head_emit", "issue_head_precision", "recall_all",
            "recall_all_top3", "recall_union_res", "recall_best_base_res",
            "page_delta_top1", "page_delta_top3", "offset_top1", "offset_top3",
            "exported_undedup", "exported_dedup", "dedup_dropped",
            "dedup_drop_rate", "dedup_policy", "dedup_capacity", "best_base",
        ):
            if key in off:
                row["offline_" + key] = off[key]

        row["keyed_meta"] = str(args.replay_input_root / (trace + ".pc_line_occ.csv.meta.json"))
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "trace", "replay_transport_ok", "run_failed", "fail_reason",
        "ipc", "speedup_vs_no_pref", "speedup_vs_best_normal",
        "no_pref_ipc", "best_normal", "best_normal_ipc",
        "instructions", "cycles", "l2_loads", "l2_load_miss", "l2_load_miss_rate",
        "pf_requested", "pf_dropped", "pf_issued", "nodup_issued", "pf_filled",
        "pf_useful", "pf_useless", "pf_late", "pq_merged_duplicate_proxy",
        "selected_accuracy", "timeliness", "drop_rate", "useless_per_issued",
        "list_replayer_instantiated", "validation_marker",
        "list_loaded_entries", "list_loaded_trigger_keys",
        "replayer_emitted_candidates", "replayer_runtime_l2_loads",
        "replayer_matched_trigger_keys", "replayer_final_loaded_trigger_keys",
        "keyed_entries", "keyed_unique_trigger_keys", "keyed_oracle_rows",
        "keyed_unmatched_rows", "keyed_dropped_invalid_address", "keyed_trigger_coverage",
        "offline_chunk_len", "offline_emit_mode", "offline_emit_rate",
        "offline_addr_supervision_rate", "offline_policy_emit",
        "offline_policy_addr_correct", "offline_policy_precision",
        "offline_issue_head_emit", "offline_issue_head_precision",
        "offline_recall_all", "offline_recall_all_top3",
        "offline_recall_union_res", "offline_recall_best_base_res",
        "offline_page_delta_top1", "offline_page_delta_top3",
        "offline_offset_top1", "offline_offset_top3",
        "offline_exported_undedup", "offline_exported_dedup",
        "offline_dedup_dropped", "offline_dedup_drop_rate",
        "offline_dedup_policy", "offline_dedup_capacity", "offline_best_base",
        "log", "keyed_meta",
    ]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("[write] {}".format(args.out))
    for row in rows:
        print(
            "[replay] {trace} transport_ok={ok} IPC={ipc:.6f} "
            "speedup(no_pref)={sp0:.4f} speedup(best={best})={spb:.4f} "
            "pf_issued={issued} useful={useful} acc={acc:.4f} "
            "keyed_match={matched}/{keys} ({coverage:.4f})".format(
                trace=row["trace"], ok=row["replay_transport_ok"],
                ipc=to_float(row["ipc"]), sp0=to_float(row["speedup_vs_no_pref"]),
                best=row.get("best_normal") or "<unknown>",
                spb=to_float(row["speedup_vs_best_normal"]),
                issued=to_int(row["pf_issued"]), useful=to_int(row["pf_useful"]),
                acc=to_float(row["selected_accuracy"]),
                matched=to_int(row["replayer_matched_trigger_keys"]),
                keys=to_int(row["keyed_unique_trigger_keys"]),
                coverage=to_float(row["keyed_trigger_coverage"]),
            )
        )


if __name__ == "__main__":
    main()
