#!/usr/bin/env python3
"""Fail-closed analysis for the 623 AMPM LSTM-versus-causal-TCN study."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path


TRACE = "623.xalancbmk_s-700B"
EXPERIMENT_REVISION = "architecture_ablation_v1"
KV = re.compile(r"^([A-Za-z0-9_]+)\s+([-+0-9.eE]+)\s*$")
FINISHED = re.compile(
    r"Finished CPU\s+0\s+instructions:\s+(\d+)\s+cycles:\s+(\d+)\s+cumulative IPC:\s+([-+0-9.eE]+)"
)
REPLAYER = re.compile(
    r"emitted\s+(\d+)\s+candidates over\s+(\d+)\s+runtime ROI L2 LOAD accesses \((\d+)\s+matched PC-line-occ triggers"
)


def div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gzip_content_sha256(path):
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_hashes(path):
    return {"gzip_sha256": sha256(path), "content_sha256": gzip_content_sha256(path)}


def parse_log(path):
    stats = {}
    text = path.read_text(errors="ignore")
    emitted = callbacks = matched = 0
    for raw in text.splitlines():
        match = KV.match(raw.strip())
        if match:
            stats[match.group(1)] = float(match.group(2))
        match = FINISHED.search(raw)
        if match:
            stats["finished_instructions"] = float(match.group(1))
            stats["finished_cycles"] = float(match.group(2))
            stats["finished_ipc"] = float(match.group(3))
        match = REPLAYER.search(raw)
        if match:
            emitted = int(match.group(1))
            callbacks = int(match.group(2))
            matched = int(match.group(3))

    def value(key, fallback=0.0):
        return stats.get(key, fallback)

    ipc = value("Core_0_IPC", value("finished_ipc"))
    instructions = int(value("Core_0_instructions", value("finished_instructions")))
    cycles = int(value("Core_0_cycles", value("finished_cycles")))
    l2_loads = int(value("Core_0_L2C_loads"))
    l2_miss = int(value("Core_0_L2C_load_miss"))
    requested = int(value("Core_0_L2C_prefetch_requested"))
    dropped = int(value("Core_0_L2C_prefetch_dropped"))
    issued = int(value("Core_0_L2C_prefetch_issued"))
    filled = int(value("Core_0_L2C_prefetch_filled"))
    useful = int(value("Core_0_L2C_prefetch_useful"))
    useless = int(value("Core_0_L2C_prefetch_useless"))
    late = int(value("Core_0_L2C_prefetch_late"))
    merged = int(value("Core_0_L2C_pq_merged"))
    nodup_issued = max(0, issued - merged)
    return {
        "ipc": ipc,
        "instructions": instructions,
        "cycles": cycles,
        "emitted": emitted,
        "callbacks": callbacks,
        "matched": matched,
        "l2_loads": l2_loads,
        "l2_load_miss": l2_miss,
        "l2_load_miss_rate": div(l2_miss, l2_loads),
        "pf_requested": requested,
        "pf_dropped": dropped,
        "pf_issued": issued,
        "nodup_issued": nodup_issued,
        "pf_filled": filled,
        "pf_useful": useful,
        "pf_useless": useless,
        "pf_late": late,
        "pq_merged_duplicate_proxy": merged,
        "accuracy": div(useful, issued),
        "selected_accuracy": div(useful, nodup_issued),
        "timeliness": div(useful, useful + late),
        "late_per_issued": div(late, issued),
        "drop_rate": div(dropped, requested),
        "useless_per_issued": div(useless, issued),
        "pollution_risk_proxy": max(0.0, 1.0 - div(useful, nodup_issued)) if nodup_issued else 0.0,
    }


def parse_events(path):
    result = {
        "demand_l2_loads": 0,
        "demand_l2_hits": 0,
        "demand_l2_misses": 0,
        "prefetch_useful_demand_hits": 0,
        "prefetch_late_demand_misses": 0,
        "prefetch_request_events": 0,
        "prefetch_accepted_events": 0,
        "prefetch_duplicate_events": 0,
    }
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"event", "hit", "was_prefetch", "late", "accepted", "duplicate"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError("{} missing event columns {}".format(path, sorted(missing)))
        for row in reader:
            if row["event"] == "PF":
                result["prefetch_request_events"] += 1
                result["prefetch_accepted_events"] += int(row["accepted"])
                result["prefetch_duplicate_events"] += int(row["duplicate"])
                continue
            if row["event"] != "DEMAND":
                continue
            result["demand_l2_loads"] += 1
            if int(row["hit"]):
                result["demand_l2_hits"] += 1
            else:
                result["demand_l2_misses"] += 1
            if int(row["was_prefetch"]):
                result["prefetch_useful_demand_hits"] += 1
            if int(row["late"]):
                result["prefetch_late_demand_misses"] += 1
    result["event_selected_accuracy_proxy"] = div(
        result["prefetch_useful_demand_hits"], result["prefetch_request_events"]
    )
    result["event_timeliness_proxy"] = div(
        result["prefetch_useful_demand_hits"],
        result["prefetch_useful_demand_hits"] + result["prefetch_late_demand_misses"],
    )
    return result


def capped_ratio(value, baseline, lower_is_better=False):
    if value <= 0 or baseline <= 0:
        return 0.0
    ratio = baseline / value if lower_is_better else value / baseline
    return min(1.0, ratio)


def add_baseline_metrics(rows, failures):
    by_method = {row["method"]: row for row in rows}
    no_pref = by_method.get("no_pref")
    offline_ampm = by_method.get("offline_ampm")
    if no_pref is None or offline_ampm is None:
        failures.append("no-prefetch or offline-AMPM baseline is missing")
        return
    if no_pref["ipc"] <= 0 or no_pref["l2_load_miss"] <= 0:
        failures.append("invalid no-prefetch denominator")
        return
    for row in rows:
        row["speedup_vs_no_pref"] = div(row["ipc"], no_pref["ipc"])
        row["miss_reduction_vs_no_pref"] = div(
            no_pref["l2_load_miss"] - row["l2_load_miss"], no_pref["l2_load_miss"]
        )
        row["coverage_vs_no_pref_l2_miss"] = div(row["pf_useful"], no_pref["l2_load_miss"])
        row["event_coverage_vs_no_pref_l2_miss"] = div(
            row["prefetch_useful_demand_hits"], no_pref["demand_l2_misses"]
        )
        row["balanced_parity_index"] = ""
        row["parity_miss_rate"] = ""
        row["parity_selected_accuracy"] = ""
        row["parity_coverage"] = ""
        row["parity_timeliness"] = ""

    required = (
        offline_ampm["l2_load_miss_rate"],
        offline_ampm["selected_accuracy"],
        offline_ampm["coverage_vs_no_pref_l2_miss"],
        offline_ampm["timeliness"],
    )
    if any(value <= 0 for value in required):
        failures.append("offline AMPM has a zero denominator in balanced parity metrics")
        return
    for row in rows:
        if not row["matched_primary_comparison"]:
            continue
        q_miss = capped_ratio(
            row["l2_load_miss_rate"], offline_ampm["l2_load_miss_rate"], lower_is_better=True
        )
        q_accuracy = capped_ratio(row["selected_accuracy"], offline_ampm["selected_accuracy"])
        q_coverage = capped_ratio(
            row["coverage_vs_no_pref_l2_miss"], offline_ampm["coverage_vs_no_pref_l2_miss"]
        )
        q_timeliness = capped_ratio(row["timeliness"], offline_ampm["timeliness"])
        cache_block = math.sqrt(q_miss * q_coverage)
        bpi = 100.0 * (cache_block * q_accuracy * q_timeliness) ** (1.0 / 3.0)
        row["balanced_parity_index"] = bpi
        row["parity_miss_rate"] = q_miss
        row["parity_selected_accuracy"] = q_accuracy
        row["parity_coverage"] = q_coverage
        row["parity_timeliness"] = q_timeliness


def validate_metadata(metadata, tag, current_stream_info, failures):
    common = {
        "trace": TRACE,
        "model_tag": tag,
        "matched_normal_prefetcher": "ampm",
        "model_does_not_use_pc": True,
        "normal_candidate_bank_is_fixed": True,
        "nn_can_only_suppress_ampm_candidates": True,
        "training_chunks_shuffled": False,
        "causal_no_future_self_test": "PASS",
        "experiment_revision": EXPERIMENT_REVISION,
    }
    for key, expected in common.items():
        if metadata.get(key) != expected:
            failures.append("{} metadata {}={!r}; expected {!r}".format(tag, key, metadata.get(key), expected))
    family = metadata.get("model_family")
    if family == "lstm":
        expected = {
            "training_state_mode": "chronological_stateful_tbptt",
            "training_state_carried_across_chunks": True,
            "training_state_detached_between_chunks": True,
        }
    elif family == "tcn":
        expected = {
            "training_state_mode": "finite_causal_left_context",
            "training_state_carried_across_chunks": False,
            "training_state_detached_between_chunks": False,
            "tcn_receptive_field_events": 127,
            "training_left_context_overlap": 126,
        }
    else:
        failures.append("{} has unknown model family {!r}".format(tag, family))
        expected = {}
    for key, value in expected.items():
        if metadata.get(key) != value:
            failures.append("{} metadata {}={!r}; expected {!r}".format(tag, key, metadata.get(key), value))
    for role in ("train", "guard", "eval"):
        key = role + "_stream_content_sha256"
        if metadata.get(key) != current_stream_info.get(role, {}).get("content_sha256"):
            failures.append("{} {}-stream content SHA256 does not match this run".format(tag, role))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--model-tags",
        default="lstm_h8,lstm_h16,lstm_h32,tcn_c10,tcn_c16,tcn_c24",
    )
    parser.add_argument("--base-model-tag", default="lstm_h8")
    args = parser.parse_args()
    model_tags = [tag.strip() for tag in args.model_tags.split(",") if tag.strip()]
    if not model_tags or args.base_model_tag not in model_tags:
        raise SystemExit("base model tag must be one of --model-tags")

    methods = ["no_pref", "live_spp_context", "live_ampm_reference", "offline_ampm"]
    methods.extend("offline_" + tag for tag in model_tags)
    logs = args.run_dir / "logs"
    events = args.run_dir / "events"
    colab_root = args.run_dir / "colab_output"
    failures = []
    rows = []

    for method in methods:
        log_path = logs / (TRACE + "." + method + ".log")
        event_path = events / (TRACE + "." + method + ".events.csv.gz")
        if not log_path.is_file():
            failures.append("missing log {}".format(log_path))
            continue
        if not event_path.is_file():
            failures.append("missing event log {}".format(event_path))
            continue
        row = {
            "trace": TRACE,
            "method": method,
            "model_tag": method[len("offline_"):] if method.startswith(("offline_lstm_", "offline_tcn_")) else "",
            "matched_primary_comparison": int(method == "offline_ampm" or method.startswith(("offline_lstm_", "offline_tcn_"))),
            "log": str(log_path),
            "event_log": str(event_path),
        }
        row.update(parse_log(log_path))
        try:
            row.update(parse_events(event_path))
        except Exception as exc:
            failures.append("{} event parse failed: {}".format(method, exc))
        if row["ipc"] <= 0 or row["instructions"] <= 0:
            failures.append("{} lacks final simulator statistics".format(method))
        if row["l2_loads"] <= 0 or row["l2_load_miss"] <= 0:
            failures.append("{} lacks L2 load counters".format(method))
        if method == "live_ampm_reference" and row["pf_requested"] <= 0:
            failures.append("live AMPM emitted no requests")
        if method == "live_spp_context" and row["pf_requested"] <= 0:
            failures.append("live SPP context emitted no requests")
        if method.startswith("offline_") and method != "offline_ampm" and (row["matched"] <= 0 or row["emitted"] <= 0):
            failures.append("{} did not replay keyed entries".format(method))
        if method == "offline_ampm" and (row["matched"] <= 0 or row["emitted"] <= 0):
            failures.append("offline AMPM did not replay keyed entries")
        rows.append(row)

    if len(rows) != len(methods):
        failures.append("one or more methods are missing")
    if rows and len({row["instructions"] for row in rows}) != 1:
        failures.append("simulation instruction counts differ")

    input_dir = args.run_dir / "colab_input"
    current_stream_info = {}
    for role in ("train", "guard", "eval"):
        path = input_dir / (TRACE + "." + role + "_stream.csv.gz")
        if not path.is_file():
            failures.append("missing normalized {} stream {}".format(role, path))
            continue
        try:
            current_stream_info[role] = stream_hashes(path)
        except (OSError, gzip.BadGzipFile) as exc:
            failures.append("cannot hash {} stream: {}".format(role, exc))

    metadata_by_tag = {}
    ampm_hashes = {}
    pairs = defaultdict(list)
    for tag in model_tags:
        for name in ("offline_ampm.replay.csv", "offline_nn.replay.csv", "model.pt", "run_metadata.json"):
            path = colab_root / tag / name
            if not path.is_file():
                failures.append("missing Colab output {}".format(path))
        metadata_path = colab_root / tag / "run_metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            failures.append("{} metadata parse failed: {}".format(tag, exc))
            continue
        metadata_by_tag[tag] = metadata
        validate_metadata(metadata, tag, current_stream_info, failures)
        pair_id = metadata.get("architecture_pair_id")
        if not pair_id:
            failures.append("{} lacks architecture_pair_id".format(tag))
        else:
            pairs[pair_id].append(metadata)
        list_path = colab_root / tag / "offline_ampm.replay.csv"
        if list_path.is_file():
            ampm_hashes[tag] = sha256(list_path)

    if ampm_hashes and len(set(ampm_hashes.values())) != 1:
        failures.append("offline AMPM list differs across architecture points")
    for pair_id, members in pairs.items():
        families = {member.get("model_family") for member in members}
        if len(members) != 2 or families != {"lstm", "tcn"}:
            failures.append("pair {} must contain one LSTM and one TCN".format(pair_id))
            continue
        counts = [int(member["parameter_count"]) for member in members]
        if max(counts) / float(min(counts)) > 1.15:
            failures.append("pair {} parameter counts differ by more than 15%".format(pair_id))

    add_baseline_metrics(rows, failures)
    by_method = {row["method"]: row for row in rows}
    fields = [
        "trace", "method", "model_tag", "matched_primary_comparison", "ipc", "speedup_vs_no_pref",
        "instructions", "cycles", "l2_loads", "l2_load_miss", "l2_load_miss_rate", "miss_reduction_vs_no_pref",
        "pf_requested", "pf_dropped", "pf_issued", "nodup_issued", "pf_filled", "pf_useful", "pf_useless", "pf_late",
        "pq_merged_duplicate_proxy", "accuracy", "selected_accuracy", "coverage_vs_no_pref_l2_miss", "timeliness",
        "late_per_issued", "drop_rate", "useless_per_issued", "pollution_risk_proxy", "balanced_parity_index",
        "parity_miss_rate", "parity_selected_accuracy", "parity_coverage", "parity_timeliness",
        "demand_l2_loads", "demand_l2_hits", "demand_l2_misses", "prefetch_useful_demand_hits",
        "prefetch_late_demand_misses", "prefetch_request_events", "prefetch_accepted_events", "prefetch_duplicate_events",
        "event_selected_accuracy_proxy", "event_coverage_vs_no_pref_l2_miss", "event_timeliness_proxy",
        "matched", "emitted", "callbacks", "log", "event_log",
    ]
    with (args.run_dir / "matched_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    status = "FAIL" if failures else "PASS"
    payload = {
        "status": status,
        "fair_comparison_claim_allowed": not failures,
        "trace": TRACE,
        "trace_selection": {
            "reason": "pollution-stress trace: historical live AMPM is below no-prefetch and has low selected accuracy",
            "primary_question": "can a temporal gate suppress harmful AMPM candidates without losing the useful subset?",
            "follow_up_trace_if_time_allows": "619.lbm_s-4268B",
        },
        "primary_comparison": ["offline_ampm", "offline_lstm_<size>", "offline_tcn_<size>"],
        "context_reference_only": ["no_pref", "live_spp_context", "live_ampm_reference"],
        "context_guardrail": "SPP is an external live reference, not a matched neural comparator; no parity claim may use it.",
        "transport": "offline AMPM, LSTM, and TCN lists are generated from the same guard-plus-evaluation stream and replayed by the same PC-line-occ ListReplayer",
        "candidate_contract": "all neural points see the same AMPM candidates and can suppress but cannot invent candidates",
        "metric_definitions": {
            "l2_load_miss_rate": "Core_0_L2C_load_miss / Core_0_L2C_loads; final cache outcome, lower is better",
            "selected_accuracy": "Core_0_L2C_prefetch_useful / (Core_0_L2C_prefetch_issued - Core_0_L2C_pq_merged); same nodup convention as the baseline table",
            "coverage_vs_no_pref_l2_miss": "Core_0_L2C_prefetch_useful / no-prefetch Core_0_L2C_load_miss",
            "timeliness": "Core_0_L2C_prefetch_useful / (Core_0_L2C_prefetch_useful + Core_0_L2C_prefetch_late)",
            "pollution_risk_proxy": "1 - selected_accuracy; a risk proxy, not a direct harmful-eviction count",
            "balanced_parity_index": "100 * (sqrt(q_miss_rate*q_coverage) * q_selected_accuracy * q_timeliness)^(1/3), with each parity ratio capped at 1 against offline AMPM",
        },
        "balanced_parity_guardrail": "BPI summarizes matched-normal parity only; IPC and no-prefetch recovery remain separate outcomes.",
        "input_provenance": {"current_input_dir": str(input_dir), "current_streams": current_stream_info},
        "offline_ampm_list_hashes_by_model_tag": ampm_hashes,
        "architecture_pairs": {
            pair_id: [
                {
                    "model_tag": member.get("model_tag"),
                    "model_family": member.get("model_family"),
                    "model_size": member.get("model_size"),
                    "parameter_count": member.get("parameter_count"),
                }
                for member in sorted(members, key=lambda item: item.get("model_family", ""))
            ]
            for pair_id, members in sorted(pairs.items())
        },
        "failures": failures,
        "rows": rows,
    }

    if not failures:
        offline_ampm = by_method["offline_ampm"]
        no_pref = by_method["no_pref"]
        model_points = []
        for tag in model_tags:
            row = by_method["offline_" + tag]
            metadata = metadata_by_tag[tag]
            point = {
                "model_tag": tag,
                "model_family": metadata["model_family"],
                "model_size": metadata["model_size"],
                "architecture_pair_id": metadata["architecture_pair_id"],
                "parameter_count": metadata["parameter_count"],
                "ipc": row["ipc"],
                "ipc_delta_vs_offline_ampm": row["ipc"] - offline_ampm["ipc"],
                "ipc_delta_vs_no_pref": row["ipc"] - no_pref["ipc"],
                "beats_offline_ampm": row["ipc"] > offline_ampm["ipc"],
                "recovers_no_pref": row["ipc"] >= no_pref["ipc"],
                "l2_load_miss_rate": row["l2_load_miss_rate"],
                "selected_accuracy": row["selected_accuracy"],
                "coverage_vs_no_pref_l2_miss": row["coverage_vs_no_pref_l2_miss"],
                "timeliness": row["timeliness"],
                "balanced_parity_index": row["balanced_parity_index"],
            }
            model_points.append(point)
        best_ipc = max(model_points, key=lambda point: point["ipc"])
        best_bpi = max(model_points, key=lambda point: point["balanced_parity_index"])
        best_by_family = {
            family: max(
                (point for point in model_points if point["model_family"] == family),
                key=lambda point: point["ipc"],
            )
            for family in ("lstm", "tcn")
        }
        payload.update({
            "offline_ampm_ipc": offline_ampm["ipc"],
            "no_pref_ipc": no_pref["ipc"],
            "architecture_sweep": model_points,
            "best_model_by_ipc": best_ipc,
            "best_model_by_balanced_parity": best_bpi,
            "best_model_by_family": best_by_family,
            "any_model_beats_offline_ampm": any(point["beats_offline_ampm"] for point in model_points),
            "any_model_recovers_no_pref": any(point["recovers_no_pref"] for point in model_points),
            "presentation_decision_rule": {
                "tcn_wins": "at a matched parameter pair, TCN improves IPC and miss rate without a material BPI loss; local finite-window correlation is the stronger inductive bias",
                "lstm_wins": "at a matched parameter pair, LSTM improves IPC and BPI; longer recurrent state carries useful information beyond 127 events",
                "both_fail": "the fixed AMPM candidate bank or future-use label is the bottleneck; changing temporal architecture alone is insufficient",
            },
        })

    out_json = args.run_dir / "matched_comparison.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("[{}] {}".format(status, out_json))
    if failures:
        raise SystemExit(" | ".join(failures))


if __name__ == "__main__":
    main()
