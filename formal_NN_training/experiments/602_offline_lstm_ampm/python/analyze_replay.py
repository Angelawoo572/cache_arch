#!/usr/bin/env python3
"""Fail-closed summary for the isolated 602 offline AMPM/LSTM sweep."""
import argparse
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path


TRACE = "602.gcc_s-734B"
KV = re.compile(r"^([A-Za-z0-9_]+)\s+([-+0-9.eE]+)\s*$")
FINISHED = re.compile(r"Finished CPU\s+0\s+instructions:\s+(\d+)\s+cycles:\s+(\d+)\s+cumulative IPC:\s+([-+0-9.eE]+)")
REPLAYER = re.compile(r"emitted\s+(\d+)\s+candidates over\s+(\d+)\s+runtime ROI L2 LOAD accesses \((\d+)\s+matched PC-line-occ triggers")


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
    """Parse simulator-final counters without inferring semantics from names."""
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

    required = [
        "Core_0_L2C_loads",
        "Core_0_L2C_load_miss",
        "Core_0_L2C_prefetch_requested",
        "Core_0_L2C_prefetch_dropped",
        "Core_0_L2C_prefetch_issued",
        "Core_0_L2C_prefetch_filled",
        "Core_0_L2C_prefetch_useful",
        "Core_0_L2C_prefetch_useless",
        "Core_0_L2C_prefetch_late",
        "Core_0_L2C_pq_merged",
    ]
    missing = [key for key in required if key not in stats]
    if missing:
        raise RuntimeError(
            "{} missing ChampSim final counters {}".format(path, missing)
        )

    def value(key, fallback=0.0):
        return stats.get(key, fallback)

    ipc = value("Core_0_IPC", value("finished_ipc"))
    instructions = int(value(
        "Core_0_instructions", value("finished_instructions")
    ))
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
    nodup_issued = issued - merged
    if min(
        l2_loads, l2_miss, requested, dropped, issued, filled,
        useful, useless, late, merged, nodup_issued,
    ) < 0:
        raise RuntimeError("{} has a negative ChampSim counter".format(path))

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
        "nodup_accuracy": div(useful, nodup_issued),
        "selected_accuracy": div(useful, nodup_issued),
        "useful_per_l2_miss_self": div(useful, l2_miss),
        "timeliness": div(useful, useful + late),
        "request_per_l2_load": div(requested, l2_loads),
        "merge_per_issued": div(merged, issued),
        "late_per_issued": div(late, issued),
        "drop_rate": div(dropped, requested),
        "fill_per_nodup_issued": div(filled, nodup_issued),
        "useless_per_issued": div(useless, issued),
        "resolved_fill_utility": div(useful, useful + useless),
    }


def parse_events(path):
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
            if int(row["hit"]):
                result["demand_l2_hits"] += 1
            else:
                result["demand_l2_misses"] += 1
            if int(row["was_prefetch"]):
                result["prefetch_useful_demand_hits"] += 1
            if int(row["late"]):
                result["prefetch_late_demand_misses"] += 1
    return result


def annotate_counter_event_consistency(row, failures):
    """Audit overlap between analyzer event counts and ChampSim final counters."""
    row["counter_event_l2_load_delta"] = (
        row["l2_loads"] - row["demand_l2_loads"]
    )
    row["counter_event_l2_miss_delta"] = (
        row["l2_load_miss"] - row["demand_l2_misses"]
    )
    row["counter_event_request_delta"] = (
        row["pf_issued"] - row["prefetch_request_events"]
    )
    row["counter_event_useful_delta"] = (
        row["pf_useful"] - row["prefetch_useful_demand_hits"]
    )
    row["counter_event_late_delta"] = (
        row["pf_late"] - row["prefetch_late_demand_misses"]
    )

    if row["l2_load_miss"] > row["l2_loads"]:
        failures.append("{} has more L2 misses than loads".format(row["method"]))
    if row["pf_requested"] != row["pf_issued"] + row["pf_dropped"]:
        failures.append(
            "{} violates requested = issued + dropped".format(row["method"])
        )
    if row["pf_filled"] < row["pf_useful"] + row["pf_useless"]:
        failures.append(
            "{} has useful + useless greater than filled".format(row["method"])
        )
    if row["counter_event_l2_load_delta"] != 0:
        failures.append(
            "{} analyzer/ChampSim L2-load mismatch".format(row["method"])
        )
    if row["counter_event_l2_miss_delta"] != 0:
        failures.append(
            "{} analyzer/ChampSim L2-miss mismatch".format(row["method"])
        )

    if row["simulator_prefetch_counter_scope"] == "measurement_only":
        checks = (
            row["counter_event_request_delta"] == 0
            and abs(row["counter_event_useful_delta"]) <= 1
            and row["counter_event_late_delta"] == 0
        )
        row["counter_event_core_consistent"] = bool(checks)
        if not checks:
            failures.append(
                "{} analyzer/ChampSim prefetch-counter mismatch".format(
                    row["method"]
                )
            )
        if row["counter_event_useful_delta"] == 0:
            row["counter_event_consistency_note"] = "exact"
        else:
            row["counter_event_consistency_note"] = (
                "useful differs by one at the measurement boundary"
            )
    else:
        row["counter_event_core_consistent"] = bool(
            row["counter_event_l2_load_delta"] == 0
            and row["counter_event_l2_miss_delta"] == 0
        )
        row["counter_event_consistency_note"] = (
            "live context: final prefetch lifecycle counters include warmup; "
            "event counters are measurement-window only"
        )


def add_cache_metrics(rows, failures):
    """Add source-defined cache metrics while keeping dimensions separate."""
    by_method = {row["method"]: row for row in rows}
    no_pref = by_method.get("no_pref")
    if (
        no_pref is None
        or no_pref["ipc"] <= 0
        or no_pref["l2_load_miss"] <= 0
        or no_pref["demand_l2_misses"] <= 0
    ):
        failures.append("valid no-prefetch baseline is missing")
        return

    for row in rows:
        row["speedup_vs_no_pref"] = div(row["ipc"], no_pref["ipc"])
        row["miss_reduction_vs_no_pref"] = div(
            no_pref["l2_load_miss"] - row["l2_load_miss"],
            no_pref["l2_load_miss"],
        )
        row["l2_miss_reduction_vs_no_pref"] = (
            row["miss_reduction_vs_no_pref"]
        )
        row["event_useful_per_request"] = div(
            row["prefetch_useful_demand_hits"],
            row["prefetch_request_events"],
        )
        row["event_coverage_vs_no_pref_l2_miss"] = div(
            row["prefetch_useful_demand_hits"],
            no_pref["demand_l2_misses"],
        )
        row["event_timeliness"] = div(
            row["prefetch_useful_demand_hits"],
            row["prefetch_useful_demand_hits"]
            + row["prefetch_late_demand_misses"],
        )

        if row["simulator_prefetch_counter_scope"] == "measurement_only":
            row["coverage_vs_no_pref_l2_miss"] = div(
                row["pf_useful"], no_pref["l2_load_miss"]
            )
            row["coverage_metric_source"] = "ChampSim measurement counters"
        else:
            row["coverage_vs_no_pref_l2_miss"] = (
                row["event_coverage_vs_no_pref_l2_miss"]
            )
            row["coverage_metric_source"] = (
                "analyzer measurement-window events"
            )

        # Backward-compatible names retain their original event-window meaning.
        row["prefetch_coverage_vs_no_pref_misses"] = (
            row["event_coverage_vs_no_pref_l2_miss"]
        )
        row["primary_metric_eligible"] = bool(
            row["simulator_prefetch_counter_scope"] == "measurement_only"
        )


def model_tag_to_hidden(tag):
    if not tag.startswith("h") or not tag[1:].isdigit():
        raise ValueError("model tag must be h<positive integer>: {}".format(tag))
    return int(tag[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--model-tags", default="h8,h16,h32,h64,h128")
    ap.add_argument("--base-model-tag", default="h8")
    ap.add_argument("--source-input-dir", type=Path, default=None, help="Only for legacy Colab outputs that lack canonical decompressed-content hashes.")
    args = ap.parse_args()
    model_tags = [x.strip() for x in args.model_tags.split(",") if x.strip()]
    if not model_tags or args.base_model_tag not in model_tags:
        raise SystemExit("base model tag must be one of --model-tags")
    methods = ["no_pref", "live_ampm_reference", "offline_ampm"] + ["offline_lstm_" + tag for tag in model_tags]
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
        row["simulator_prefetch_counter_scope"] = (
            "warmup_plus_measurement"
            if method.startswith("live_")
            else "measurement_only"
        )
        annotate_counter_event_consistency(row, failures)
        if row["ipc"] <= 0 or row["instructions"] <= 0:
            failures.append("{} lacks final simulator statistics".format(method))
        if method == "live_ampm_reference" and row["prefetch_request_events"] <= 0:
            failures.append("live_ampm_reference emitted no PF requests")
        # A threshold-free neural policy may legitimately learn request count
        # zero.  An empty NN action list is therefore a measured outcome, not a
        # replay failure.  The normal comparator must remain nonempty, while
        # every offline replay must still process the full callback stream.
        if method == "offline_ampm" and (row["matched"] <= 0 or row["emitted"] <= 0):
            failures.append("offline_ampm did not replay keyed entries")
        if method.startswith("offline_") and row["callbacks"] <= 0:
            failures.append("{} reported zero replay callbacks".format(method))
        row["matched_primary_comparison"] = int(method == "offline_ampm" or method.startswith("offline_lstm_"))
        row["model_tag"] = method[len("offline_lstm_"):] if method.startswith("offline_lstm_") else ""
        row["hidden_size"] = model_tag_to_hidden(row["model_tag"]) if row["model_tag"] else 0
        rows.append(row)

    by_method = {row["method"]: row for row in rows}
    if len(rows) != len(methods):
        failures.append("one or more methods are missing")
    if rows and len({row["instructions"] for row in rows}) != 1:
        failures.append("simulation instruction counts differ")

    current_input_dir = args.run_dir / "colab_input"
    roles = ("train", "guard", "eval")
    current_streams = {role: current_input_dir / (TRACE + "." + role + "_stream.csv.gz") for role in roles}
    current_stream_info = {}
    for role, path in current_streams.items():
        if not path.is_file():
            failures.append("missing normalized {} stream {}".format(role, path))
            continue
        try:
            current_stream_info[role] = stream_hashes(path)
        except (OSError, gzip.BadGzipFile) as exc:
            failures.append("cannot hash current {} stream {}: {}".format(role, path, exc))

    source_stream_info = {}
    if args.source_input_dir is not None:
        for role in roles:
            path = args.source_input_dir / (TRACE + "." + role + "_stream.csv.gz")
            if not path.is_file():
                failures.append("missing source {} stream {}".format(role, path))
                continue
            try:
                source_stream_info[role] = stream_hashes(path)
            except (OSError, gzip.BadGzipFile) as exc:
                failures.append("cannot hash source {} stream {}: {}".format(role, path, exc))

    input_provenance = {"current_input_dir": str(current_input_dir), "current_streams": current_stream_info, "per_model_tag": {}}
    if args.source_input_dir is not None:
        input_provenance["legacy_source_input_dir"] = str(args.source_input_dir)
        input_provenance["legacy_source_streams"] = source_stream_info

    metadata_by_tag = {}
    for tag in model_tags:
        for name in ("offline_ampm.replay.csv", "offline_lstm.replay.csv", "model.pt", "run_metadata.json"):
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
        if metadata.get("matched_normal_prefetcher") != "ampm":
            failures.append("{} metadata is not matched to AMPM".format(tag))
        state_contract = {
            "training_state_mode": "chronological_stateful_tbptt",
            "training_chunks_shuffled": False,
            "training_state_carried_across_chunks": True,
            "training_state_detached_between_chunks": True,
            "experiment_revision": "source_input_variable_delta_free_running_v7",
            "neural_role": "standalone_direct_action_prefetcher",
            "same_external_input_contract": True,
            "training_inference_input_encoder_identical": True,
            "decoder_training_mode": "free_running_autoregressive_same_as_inference",
            "decoder_previous_teacher_action_used_as_input": False,
            "decoder_free_running_self_test": "PASS",
            "training_runtime_fields": ["cache_line_address"],
            "inference_runtime_fields": ["cache_line_address"],
            "normal_policy_candidates_used_as_model_inputs": False,
            "normal_policy_private_state_used_as_model_inputs": False,
            "normal_policy_outputs_used_as_training_targets": True,
            "normal_policy_request_rate_used_as_budget": False,
            "normal_policy_constants_used_by_neural_inference": False,
            "probability_threshold_used": False,
            "neural_degree_cap": None,
            "fixed_page_offset_classes": None,
            "same_page_rule_used_by_neural_inference": False,
            "future_label_window_used": False,
            "handcrafted_semantic_features_used": False,
            "manual_loss_weights_used": False,
            "training_regularization_used": False,
            "inference_policy_hardcodes_used": False,
            "learned_request_count": True,
            "nn_generates_own_target_addresses": True,
        }
        for key, expected in state_contract.items():
            if metadata.get(key) != expected:
                failures.append("{} metadata {}={!r}; expected {!r}".format(tag, key, metadata.get(key), expected))
        encoder_hashes = {metadata.get("runtime_encoder_sha256"), metadata.get("training_runtime_encoder_sha256"), metadata.get("inference_runtime_encoder_sha256")}
        encoder_hash = next(iter(encoder_hashes)) if len(encoder_hashes) == 1 else None
        if not isinstance(encoder_hash, str) or len(encoder_hash) != 64:
            failures.append("{} train/inference encoder hash mismatch".format(tag))
        if set(current_stream_info) != set(roles):
            continue
        canonical_keys = {role: role + "_stream_content_sha256" for role in roles}
        if all(metadata.get(key) for key in canonical_keys.values()):
            input_provenance["per_model_tag"][tag] = {"mode": "canonical_decompressed_content_sha256"}
            for role, key in canonical_keys.items():
                if metadata.get(key) != current_stream_info[role]["content_sha256"]:
                    failures.append("{} {}-stream content SHA256 does not match this run".format(tag, role))
            continue
        if args.source_input_dir is None:
            failures.append("{} has legacy gzip-byte hashes only; rerun analyze with --source-input-dir".format(tag))
            continue
        if set(source_stream_info) != set(roles):
            continue
        input_provenance["per_model_tag"][tag] = {"mode": "legacy_source_byte_hash_plus_decompressed_content_match"}
        for role in roles:
            metadata_key = role + "_stream_sha256"
            if metadata.get(metadata_key) != source_stream_info[role]["gzip_sha256"]:
                failures.append("{} {}-stream SHA256 does not match supplied source input".format(tag, role))
            if current_stream_info[role]["content_sha256"] != source_stream_info[role]["content_sha256"]:
                failures.append("{} {}-stream decompressed content does not match this run".format(tag, role))

    ampm_list_hashes = {}
    for tag in model_tags:
        path = colab_root / tag / "offline_ampm.replay.csv"
        if path.is_file():
            ampm_list_hashes[tag] = sha256(path)
    if ampm_list_hashes and len(set(ampm_list_hashes.values())) != 1:
        failures.append("offline AMPM list differs across capacity points")

    add_cache_metrics(rows, failures)

    status = "FAIL" if failures else "PASS"
    fields = [
        "trace", "method", "model_tag", "hidden_size",
        "matched_primary_comparison", "primary_metric_eligible",
        "simulator_prefetch_counter_scope", "ipc", "speedup_vs_no_pref",
        "instructions", "cycles", "l2_loads", "l2_load_miss",
        "l2_load_miss_rate", "miss_reduction_vs_no_pref",
        "pf_requested", "pf_dropped", "pf_issued", "nodup_issued",
        "pf_filled", "pf_useful", "pf_useless", "pf_late",
        "pq_merged_duplicate_proxy", "accuracy", "nodup_accuracy",
        "selected_accuracy", "coverage_vs_no_pref_l2_miss",
        "coverage_metric_source", "useful_per_l2_miss_self",
        "timeliness", "request_per_l2_load", "merge_per_issued",
        "drop_rate", "fill_per_nodup_issued", "late_per_issued",
        "useless_per_issued", "resolved_fill_utility",
        "demand_l2_loads", "demand_l2_hits", "demand_l2_misses",
        "prefetch_useful_demand_hits", "prefetch_late_demand_misses",
        "prefetch_request_events", "event_useful_per_request",
        "event_coverage_vs_no_pref_l2_miss", "event_timeliness",
        "prefetch_coverage_vs_no_pref_misses",
        "l2_miss_reduction_vs_no_pref",
        "counter_event_l2_load_delta", "counter_event_l2_miss_delta",
        "counter_event_request_delta", "counter_event_useful_delta",
        "counter_event_late_delta", "counter_event_core_consistent",
        "counter_event_consistency_note", "matched", "emitted",
        "callbacks", "log", "event_log",
    ]
    with (args.run_dir / "matched_comparison.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    insight_fields = [
        "method", "model_tag", "hidden_size", "ipc",
        "speedup_vs_no_pref", "l2_load_miss_rate",
        "miss_reduction_vs_no_pref", "accuracy", "selected_accuracy",
        "coverage_vs_no_pref_l2_miss", "useful_per_l2_miss_self",
        "timeliness", "request_per_l2_load", "merge_per_issued",
        "drop_rate", "fill_per_nodup_issued", "late_per_issued",
        "useless_per_issued", "resolved_fill_utility",
        "pf_requested", "pf_issued", "nodup_issued", "pf_filled",
        "pf_useful", "pf_useless", "pf_late",
        "pq_merged_duplicate_proxy",
    ]
    with (args.run_dir / "insight_summary.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=insight_fields)
        writer.writeheader()
        writer.writerows(
            {key: row.get(key, "") for key in insight_fields}
            for row in rows
        )

    payload = {
        "status": status,
        "fair_comparison_claim_allowed": not failures,
        "trace": TRACE,
        "primary_comparison": ["offline_ampm", "offline_lstm_<capacity>"],
        "transport": "both policies see the same guard-plus-evaluation address stream and use the same PC-line-occ ListReplayer; the LSTM generates targets independently",
        "live_ampm_role": "validated general reference only; it is not the offline primary comparator",
        "matched_input_contract": "effective AMPM input only: current cache-line address plus the LSTM's own guard-initialized causal state; PC is replay transport identity only",
        "neural_contract": "unbounded learned count plus autoregressive direct cache-line deltas; no AMPM candidates, page-offset table, degree, bitmap, page buffer, or LRU state enter inference",
        "metric_sources": {
            "analyzer_events": (
                "measurement-window demand loads/hits/misses, PF request "
                "events, useful-demand hits, and late-demand misses"
            ),
            "champsim_final_counters": (
                "Core_0_L2C loads/misses and requested/dropped/issued/"
                "pq_merged/filled/useful/useless/late"
            ),
            "scope": (
                "offline/no-prefetch rows have measurement-only prefetch "
                "counters; live-reference prefetch lifecycle counters include "
                "warmup and are context-only"
            ),
        },
        "metric_definitions": {
            "l2_load_miss_rate": (
                "Core_0_L2C_load_miss / Core_0_L2C_loads"
            ),
            "miss_reduction_vs_no_pref": (
                "(no-prefetch L2 load misses - method L2 load misses) / "
                "no-prefetch L2 load misses"
            ),
            "accuracy": (
                "Core_0_L2C_prefetch_useful / "
                "Core_0_L2C_prefetch_issued"
            ),
            "selected_accuracy": (
                "Core_0_L2C_prefetch_useful / "
                "(Core_0_L2C_prefetch_issued - Core_0_L2C_pq_merged)"
            ),
            "coverage_vs_no_pref_l2_miss": (
                "measurement-window useful prefetches / no-prefetch "
                "measurement-window L2 load misses"
            ),
            "useful_per_l2_miss_self": (
                "Core_0_L2C_prefetch_useful / Core_0_L2C_load_miss"
            ),
            "timeliness": (
                "Core_0_L2C_prefetch_useful / "
                "(Core_0_L2C_prefetch_useful + Core_0_L2C_prefetch_late)"
            ),
            "request_per_l2_load": (
                "Core_0_L2C_prefetch_requested / Core_0_L2C_loads"
            ),
            "merge_per_issued": (
                "Core_0_L2C_pq_merged / Core_0_L2C_prefetch_issued"
            ),
            "drop_rate": (
                "Core_0_L2C_prefetch_dropped / "
                "Core_0_L2C_prefetch_requested"
            ),
            "fill_per_nodup_issued": (
                "Core_0_L2C_prefetch_filled / "
                "(Core_0_L2C_prefetch_issued - Core_0_L2C_pq_merged)"
            ),
            "late_per_issued": (
                "Core_0_L2C_prefetch_late / "
                "Core_0_L2C_prefetch_issued"
            ),
            "useless_per_issued": (
                "Core_0_L2C_prefetch_useless / "
                "Core_0_L2C_prefetch_issued"
            ),
            "resolved_fill_utility": (
                "Core_0_L2C_prefetch_useful / "
                "(Core_0_L2C_prefetch_useful + "
                "Core_0_L2C_prefetch_useless)"
            ),
            "event_useful_per_request": (
                "measurement-window useful-demand hits / "
                "measurement-window PF request events"
            ),
            "event_coverage_vs_no_pref_l2_miss": (
                "measurement-window useful-demand hits / no-prefetch "
                "measurement-window demand L2 misses"
            ),
            "event_timeliness": (
                "measurement-window useful-demand hits / "
                "(useful-demand hits + late-demand misses)"
            ),
        },
        "model_tags": model_tags,
        "required_recurrent_state_contract": {
            "training": "chronological stateful TBPTT; hidden/cell values cross chunks; graph detached at chunk boundaries",
            "inference": "20M--25M guard initializes hidden/cell state, then evaluation is continuous",
        },
        "offline_ampm_list_hashes_by_model_tag": ampm_list_hashes,
        "input_provenance": input_provenance,
        "failures": failures,
        "rows": rows,
    }
    if not failures:
        ampm_ipc = by_method["offline_ampm"]["ipc"]
        sweep = []
        first_better = None
        for tag in model_tags:
            row = by_method["offline_lstm_" + tag]
            metadata = metadata_by_tag[tag]
            point = {
                "model_tag": tag,
                "hidden_size": metadata["hidden_size"],
                "parameter_count": metadata["parameter_count"],
                "ipc": row["ipc"],
                "ipc_delta_lstm_minus_ampm": row["ipc"] - ampm_ipc,
                "lstm_beats_offline_ampm": row["ipc"] > ampm_ipc,
                "prefetch_coverage_vs_no_pref_misses": row["prefetch_coverage_vs_no_pref_misses"],
                "prefetch_useful_demand_hits": row["prefetch_useful_demand_hits"],
                "prefetch_late_demand_misses": row["prefetch_late_demand_misses"],
                "l2_load_miss_rate": row["l2_load_miss_rate"],
                "miss_reduction_vs_no_pref": row["miss_reduction_vs_no_pref"],
                "accuracy": row["accuracy"],
                "selected_accuracy": row["selected_accuracy"],
                "coverage_vs_no_pref_l2_miss": row["coverage_vs_no_pref_l2_miss"],
                "timeliness": row["timeliness"],
                "pf_requested": row["pf_requested"],
                "pf_issued": row["pf_issued"],
                "nodup_issued": row["nodup_issued"],
                "pf_filled": row["pf_filled"],
                "pf_useful": row["pf_useful"],
                "pf_useless": row["pf_useless"],
                "pf_late": row["pf_late"],
                "pq_merged_duplicate_proxy": row["pq_merged_duplicate_proxy"],
            }
            sweep.append(point)
            if first_better is None and point["lstm_beats_offline_ampm"]:
                first_better = point
        payload.update({
            "offline_ampm_ipc": ampm_ipc,
            "capacity_sweep": sweep,
            "first_measured_capacity_beating_offline_ampm": first_better,
            "saturation_interpretation": "The first measured capacity above offline AMPM is exploratory because the evaluation stream is reused across capacity points; confirm it on a fresh held-out window before selecting a final capacity.",
            "offline_ampm_list_sha256": sha256(colab_root / args.base_model_tag / "offline_ampm.replay.csv"),
        })
    out_json = args.run_dir / "matched_comparison.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("[{}] {}".format(status, out_json))
    if failures:
        raise SystemExit(" | ".join(failures))


if __name__ == "__main__":
    main()


