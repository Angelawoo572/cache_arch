#!/usr/bin/env python3
"""Summarize signature-validated oracle-LSTM ListReplayer runs.

This script intentionally reuses the generic counter parser in
01_parse_prefetch_behavior_audit.py instead of duplicating Pythia log parsing.
It adds only oracle-replay-specific fields: strict-list metadata, signature
validation, and comparison against no-prefetch / best normal-prefetcher IPC.
Python 3.6 compatible.
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
    r"\[list_replayer\] loaded\s+(\d+)\s+prefetch entries across\s+(\d+)\s+ROI L2 LOAD indices"
)
REFERENCE_RE = re.compile(
    r"\[list_replayer\] loaded\s+(\d+)\s+dense ROI L2 LOAD signatures"
)
FINAL_RE = re.compile(
    r"\[list_replayer\] emitted\s+(\d+)\s+candidates over\s+(\d+)\s+ROI L2 LOAD accesses\s+"
    r"\((\d+)\s+matched access indices;\s+(\d+)\s+signature mismatches;\s+"
    r"(\d+)\s+post-reference tail accesses;\s+(reference enabled|reference DISABLED)\)"
)
VALIDATED_RE = re.compile(r"\[oracle-replay-validation\]\s+status=(pass|fail)")


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
        "reference_loaded": 0,
        "validation_marker": "",
        "list_loaded_entries": 0,
        "list_loaded_indices": 0,
        "reference_rows": 0,
        "replayer_emitted_candidates": 0,
        "replayer_observed_l2_loads": 0,
        "replayer_matched_indices": 0,
        "signature_mismatches": -1,
        "reference_tail_accesses": 0,
        "reference_enabled": 0,
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
                out["list_loaded_indices"] = int(match.group(2))

            match = REFERENCE_RE.search(line)
            if match:
                out["reference_loaded"] = 1
                out["reference_rows"] = int(match.group(1))

            match = FINAL_RE.search(line)
            if match:
                out["replayer_emitted_candidates"] = int(match.group(1))
                out["replayer_observed_l2_loads"] = int(match.group(2))
                out["replayer_matched_indices"] = int(match.group(3))
                out["signature_mismatches"] = int(match.group(4))
                out["reference_tail_accesses"] = int(match.group(5))
                out["reference_enabled"] = int(match.group(6) == "reference enabled")

            match = VALIDATED_RE.search(line)
            if match:
                out["validation_marker"] = match.group(1)

    return out


def load_strict_meta(path):
    out = {
        "strict_entries": 0,
        "strict_unique_indices": 0,
        "strict_reference_rows": 0,
        "strict_unmatched_rows": 0,
        "strict_dropped_invalid_address": 0,
    }
    if not path.is_file():
        return out

    try:
        data = json.loads(path.read_text())
    except Exception:
        return out

    out["strict_entries"] = to_int(data.get("entries"))
    out["strict_unique_indices"] = to_int(data.get("unique_indices"))
    out["strict_reference_rows"] = to_int(data.get("reference_rows"))
    out["strict_unmatched_rows"] = to_int(data.get("unmatched_rows"))
    out["strict_dropped_invalid_address"] = to_int(data.get("dropped_invalid_address"))
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
                    help="oracle_replacer_sweep.csv; joined as clearly prefixed offline_* fields")
    ap.add_argument("--log-suffix", default=".oracle_replacer.log")
    args = ap.parse_args()

    baselines = make_baselines(read_csv_rows(args.baseline_metrics))
    offline = offline_by_trace(read_csv_rows(args.offline_summary))
    rows = []

    for trace in args.traces.split():
        log = args.log_root / (trace + args.log_suffix)
        base = _behavior.summarize_one(trace, "oracle_lstm", log, nodup=True)
        replay = parse_replayer_log(log)
        meta = load_strict_meta(args.replay_input_root / (trace + ".l2roi.idx_addr.csv.meta.json"))
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

        # Script 09 writes the marker only after its strict-prefix validation passes.
        row["replay_validated"] = int(
            row.get("validation_marker") == "pass"
            and row.get("list_replayer_instantiated") == 1
            and row.get("reference_loaded") == 1
            and row.get("reference_enabled") == 1
            and row.get("signature_mismatches") == 0
            and int(row.get("run_failed", 0)) == 0
        )

        # Keep offline proxy quantities visibly separate from simulator measurements.
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

        row["strict_meta"] = str(args.replay_input_root / (trace + ".l2roi.idx_addr.csv.meta.json"))
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "trace", "replay_validated", "run_failed", "fail_reason",
        "ipc", "speedup_vs_no_pref", "speedup_vs_best_normal",
        "no_pref_ipc", "best_normal", "best_normal_ipc",
        "instructions", "cycles", "l2_loads", "l2_load_miss", "l2_load_miss_rate",
        "pf_requested", "pf_dropped", "pf_issued", "nodup_issued", "pf_filled",
        "pf_useful", "pf_useless", "pf_late", "pq_merged_duplicate_proxy",
        "selected_accuracy", "timeliness", "drop_rate", "useless_per_issued",
        "list_replayer_instantiated", "reference_loaded", "reference_enabled",
        "validation_marker", "list_loaded_entries", "list_loaded_indices",
        "reference_rows", "replayer_emitted_candidates", "replayer_observed_l2_loads",
        "replayer_matched_indices", "signature_mismatches", "reference_tail_accesses",
        "strict_entries", "strict_unique_indices", "strict_reference_rows",
        "strict_unmatched_rows", "strict_dropped_invalid_address",
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
        "log", "strict_meta",
    ]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("[write] {}".format(args.out))
    for row in rows:
        print(
            "[replay] {trace} valid={valid} IPC={ipc:.6f} "
            "speedup(no_pref)={sp0:.4f} speedup(best={best})={spb:.4f} "
            "pf_issued={issued} useful={useful} acc={acc:.4f} "
            "replayer={emit}/{obs} sig_mismatch={sig}".format(
                trace=row["trace"],
                valid=row["replay_validated"],
                ipc=to_float(row["ipc"]),
                sp0=to_float(row["speedup_vs_no_pref"]),
                best=row.get("best_normal") or "<unknown>",
                spb=to_float(row["speedup_vs_best_normal"]),
                issued=to_int(row["pf_issued"]),
                useful=to_int(row["pf_useful"]),
                acc=to_float(row["selected_accuracy"]),
                emit=to_int(row["replayer_emitted_candidates"]),
                obs=to_int(row["replayer_observed_l2_loads"]),
                sig=to_int(row["signature_mismatches"], -1),
            )
        )


if __name__ == "__main__":
    main()
