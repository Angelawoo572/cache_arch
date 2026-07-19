#!/usr/bin/env python3
"""Fail-closed analysis for the standalone 602 SPP-interface LSTM track."""
import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


TRACE = "602.gcc_s-734B"
POLICY = "spp"
POLICIES = (POLICY,)
EXPERIMENT_REVISION = "spp_source_input_compact_empirical_prior_hurdle_delta_fill_free_running_v2"
TRACK_MODEL_FAMILY = "lstm"
DEFAULT_MODEL_TAGS = (
    "independent_delta_spp_lstm_h8,independent_delta_spp_lstm_h16,"
    "independent_delta_spp_lstm_h32,independent_delta_spp_lstm_h64,"
    "independent_delta_spp_lstm_h128"
)
EXPECTED_POINTS = {
    ("lstm", 8): "p0",
    ("lstm", 16): "p1",
    ("lstm", 32): "p2",
    ("lstm", 64): "p3",
    ("lstm", 128): "p4",
}
EXPECTED_PARAMETER_COUNTS = {
    8: 2865,
    16: 6609,
    32: 16785,
    64: 47889,
    128: 153105,
}
EVENT_LOGGER_SCHEMA = "602_spp_causal_trigger_fill_v1"
ACTION_ATTACHMENT_MODE = "explicit_trigger_event_id"
SOURCE_INPUTS = [
    "callback_kind", "invoke_prefetcher.addr", "cache_fill.evicted_addr",
]
KV = re.compile(r"^([A-Za-z0-9_]+)\s+([-+0-9.eE]+)\s*$")
FINISHED = re.compile(
    r"Finished CPU\s+0\s+instructions:\s+(\d+)\s+cycles:\s+(\d+)\s+"
    r"cumulative IPC:\s+([-+0-9.eE]+)"
)
REPLAYER = re.compile(
    r"emitted\s+(\d+)\s+(?:candidates|actions) over\s+(\d+)\s+runtime ROI L2 LOAD "
    r"accesses \((\d+)\s+matched PC-line-occ triggers"
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
    return {
        "gzip_sha256": sha256(path),
        "content_sha256": gzip_content_sha256(path),
    }


def replay_list_info(path, allow_empty=False):
    count = 0
    fill_counts = {"FILL_L2": 0, "FILL_LLC": 0}
    with Path(path).open(newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != [
            "pc", "line", "occ", "prefetch_addr", "fill_level"
        ]:
            raise RuntimeError("invalid five-column SPP replay header")
        for line_number, fields in enumerate(reader, 2):
            if len(fields) != 5:
                raise RuntimeError(
                    "invalid SPP replay row {}".format(line_number)
                )
            try:
                pc = int(fields[0], 0)
                line = int(fields[1], 0)
                occurrence = int(fields[2], 10)
                address = int(fields[3], 0)
                fill_level = int(fields[4], 0)
            except ValueError as exc:
                raise RuntimeError(
                    "invalid SPP replay integer at row {}: {}".format(
                        line_number, exc
                    )
                )
            if min(pc, line, occurrence, address) < 0 or address % 64:
                raise RuntimeError(
                    "unaligned/negative SPP replay row {}".format(line_number)
                )
            if fill_level not in (2, 4):
                raise RuntimeError(
                    "invalid SPP fill level at row {}".format(line_number)
                )
            fill_counts[
                "FILL_L2" if fill_level == 2 else "FILL_LLC"
            ] += 1
            count += 1
    if count <= 0 and not allow_empty:
        raise RuntimeError("empty SPP replay list")
    return {
        "entries": count,
        "sha256": sha256(path),
        "fill_counts": fill_counts,
    }


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
        "pollution_risk_proxy": (
            max(0.0, 1.0 - div(useful, nodup_issued))
            if nodup_issued else 0.0
        ),
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
        "prefetch_fill_l2_events": 0,
        "prefetch_fill_llc_events": 0,
        "cache_fill_feedback_events": 0,
    }
    last_event_id = -1
    latest_demand_event_id = None
    latest_demand_identity = None
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "event", "event_id", "cpu", "cycle", "cache", "op", "ip",
            "addr", "line",
            "hit", "was_prefetch", "late", "accepted", "duplicate",
            "base_addr", "fill_level", "trigger_event_id", "trigger_cpu", "trigger_ip",
            "trigger_line", "logger_schema",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                "{} missing event columns {}".format(path, sorted(missing))
            )
        for row in reader:
            event_id = int(row["event_id"])
            if event_id != last_event_id + 1:
                raise RuntimeError("noncontiguous event IDs at {}".format(event_id))
            last_event_id = event_id
            if row["cache"] != "L2C" or row["logger_schema"] != EVENT_LOGGER_SCHEMA:
                raise RuntimeError("stale/non-L2 602 event logger row")
            if row["event"] == "PF":
                trigger_event_id = int(row["trigger_event_id"])
                if (
                    latest_demand_event_id is None
                    or trigger_event_id != latest_demand_event_id
                    or trigger_event_id >= event_id
                ):
                    raise RuntimeError("PF has invalid explicit trigger event")
                trigger_identity = (
                    int(row["trigger_cpu"]), int(row["trigger_ip"]),
                    int(row["trigger_line"]), int(row["cycle"]),
                )
                if trigger_identity != latest_demand_identity:
                    raise RuntimeError("PF trigger identity/cycle mismatch")
                if (int(row["base_addr"]) >> 6) != int(row["trigger_line"]):
                    raise RuntimeError("PF base line differs from trigger line")
                fill_level = int(row["fill_level"])
                if fill_level not in (2, 4):
                    raise RuntimeError("SPP PF has invalid captured fill level")
                result["prefetch_request_events"] += 1
                result["prefetch_accepted_events"] += int(row["accepted"])
                result["prefetch_duplicate_events"] += int(row["duplicate"])
                result[
                    "prefetch_fill_l2_events"
                    if fill_level == 2 else "prefetch_fill_llc_events"
                ] += 1
                continue
            if row["event"] == "FILL":
                if (
                    row.get("op") != "cache_fill"
                    or int(row["trigger_event_id"]) != (1 << 64) - 1
                    or int(row["trigger_cpu"]) != 0
                    or int(row["trigger_ip"]) != 0
                    or int(row["trigger_line"]) != 0
                    or int(row["ip"]) != 0
                    or (int(row["addr"]) >> 6) != int(row["line"])
                ):
                    raise RuntimeError("invalid SPP cache-fill feedback event")
                latest_demand_event_id = None
                latest_demand_identity = None
                result["cache_fill_feedback_events"] += 1
                continue
            if row["event"] != "DEMAND":
                raise RuntimeError("unknown 602 event kind {}".format(row["event"]))
            demand_identity = (
                int(row["cpu"]), int(row["ip"]), int(row["line"]),
                int(row["cycle"]),
            )
            trigger_identity = (
                int(row["trigger_cpu"]), int(row["trigger_ip"]),
                int(row["trigger_line"]), int(row["cycle"]),
            )
            if (
                int(row["trigger_event_id"]) != event_id
                or trigger_identity != demand_identity
            ):
                raise RuntimeError("demand self-trigger event mismatch")
            latest_demand_event_id = event_id
            latest_demand_identity = demand_identity
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
        result["prefetch_useful_demand_hits"],
        result["prefetch_request_events"],
    )
    result["event_timeliness_proxy"] = div(
        result["prefetch_useful_demand_hits"],
        result["prefetch_useful_demand_hits"]
        + result["prefetch_late_demand_misses"],
    )
    return result


def policy_for_method(method):
    if method == "offline_" + POLICY or method.startswith("offline_independent_delta_spp_"):
        return POLICY
    return ""


def model_tag_for_method(method):
    if method.startswith("offline_independent_delta_spp_"):
        return method[len("offline_"):]
    return ""


def capped_ratio(value, baseline, lower_is_better=False):
    if value <= 0 or baseline <= 0:
        return 0.0
    ratio = baseline / value if lower_is_better else value / baseline
    return min(1.0, ratio)


def add_comparison_metrics(rows, failures):
    by_method = {row["method"]: row for row in rows}
    no_pref = by_method.get("no_pref")
    if no_pref is None or no_pref["ipc"] <= 0 or no_pref["l2_load_miss"] <= 0:
        failures.append("valid no-prefetch baseline is missing")
        return

    for row in rows:
        row["speedup_vs_no_pref"] = div(row["ipc"], no_pref["ipc"])
        row["miss_reduction_vs_no_pref"] = div(
            no_pref["l2_load_miss"] - row["l2_load_miss"],
            no_pref["l2_load_miss"],
        )
        row["coverage_vs_no_pref_l2_miss"] = div(
            row["pf_useful"], no_pref["l2_load_miss"]
        )
        row["event_coverage_vs_no_pref_l2_miss"] = div(
            row["prefetch_useful_demand_hits"],
            no_pref["demand_l2_misses"],
        )
        row["balanced_parity_index"] = ""
        row["parity_miss_rate"] = ""
        row["parity_selected_accuracy"] = ""
        row["parity_coverage"] = ""
        row["parity_timeliness"] = ""
        row["parity_bottleneck"] = ""
        row["ipc_delta_vs_offline_normal"] = ""
        row["ipc_pct_vs_offline_normal"] = ""
        row["l2_miss_rate_delta_vs_offline_normal"] = ""
        row["selected_accuracy_delta_vs_offline_normal"] = ""
        row["coverage_delta_vs_offline_normal"] = ""
        row["timeliness_delta_vs_offline_normal"] = ""
        row["prefetch_request_ratio_vs_offline_normal"] = ""
        row["prefetch_request_reduction_vs_offline_normal"] = ""
        row["pollution_proxy_delta_vs_offline_normal"] = ""

    for policy in POLICIES:
        baseline = by_method.get("offline_" + policy)
        if baseline is None:
            failures.append("offline {} baseline is missing".format(policy))
            continue
        required = (
            baseline["l2_load_miss_rate"],
            baseline["selected_accuracy"],
            baseline["coverage_vs_no_pref_l2_miss"],
            baseline["timeliness"],
        )
        if any(value <= 0 for value in required):
            failures.append(
                "offline {} has a zero BPI denominator".format(policy)
            )
            continue
        for row in rows:
            if row["comparison_policy"] != policy:
                continue
            q_miss = capped_ratio(
                row["l2_load_miss_rate"],
                baseline["l2_load_miss_rate"],
                lower_is_better=True,
            )
            q_accuracy = capped_ratio(
                row["selected_accuracy"], baseline["selected_accuracy"]
            )
            q_coverage = capped_ratio(
                row["coverage_vs_no_pref_l2_miss"],
                baseline["coverage_vs_no_pref_l2_miss"],
            )
            q_timeliness = capped_ratio(
                row["timeliness"], baseline["timeliness"]
            )
            bpi = 100.0 * (
                q_miss * q_accuracy * q_coverage * q_timeliness
            ) ** 0.25
            row["balanced_parity_index"] = bpi
            row["parity_miss_rate"] = q_miss
            row["parity_selected_accuracy"] = q_accuracy
            row["parity_coverage"] = q_coverage
            row["parity_timeliness"] = q_timeliness
            parity = {
                "miss_rate": q_miss,
                "selected_accuracy": q_accuracy,
                "coverage": q_coverage,
                "timeliness": q_timeliness,
            }
            row["parity_bottleneck"] = min(
                parity, key=lambda name: parity[name]
            )
            row["ipc_delta_vs_offline_normal"] = (
                row["ipc"] - baseline["ipc"]
            )
            row["ipc_pct_vs_offline_normal"] = div(
                row["ipc"] - baseline["ipc"], baseline["ipc"]
            )
            row["l2_miss_rate_delta_vs_offline_normal"] = (
                row["l2_load_miss_rate"]
                - baseline["l2_load_miss_rate"]
            )
            row["selected_accuracy_delta_vs_offline_normal"] = (
                row["selected_accuracy"]
                - baseline["selected_accuracy"]
            )
            row["coverage_delta_vs_offline_normal"] = (
                row["coverage_vs_no_pref_l2_miss"]
                - baseline["coverage_vs_no_pref_l2_miss"]
            )
            row["timeliness_delta_vs_offline_normal"] = (
                row["timeliness"] - baseline["timeliness"]
            )
            request_ratio = div(
                row["pf_requested"], baseline["pf_requested"]
            )
            row["prefetch_request_ratio_vs_offline_normal"] = request_ratio
            row["prefetch_request_reduction_vs_offline_normal"] = (
                1.0 - request_ratio
            )
            row["pollution_proxy_delta_vs_offline_normal"] = (
                row["pollution_risk_proxy"]
                - baseline["pollution_risk_proxy"]
            )


def validate_metadata(metadata, tag, inputs, source_contract_hash, failures):
    policy = POLICY
    common = {
        "trace": TRACE,
        "model_tag": tag,
        "matched_normal_prefetcher": policy,
        "model_does_not_use_pc": True,
        "pc_is_replay_transport_only": True,
        "teacher_actions_are_model_inputs": False,
        "same_external_input_contract": True,
        "training_inference_input_encoder_identical": True,
        "decoder_training_mode": "free_running_autoregressive_same_as_inference",
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_free_running_self_test": "PASS",
        "training_runtime_fields": SOURCE_INPUTS,
        "inference_runtime_fields": SOURCE_INPUTS,
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "model_input_is_causal_external_event_sequence_only": True,
        "cache_fill_feedback_used_as_raw_external_input": True,
        "cache_fill_private_state_used_as_model_input": False,
        "cache_hit_and_type_are_audit_only": True,
        "teacher_action_canonicalization": (
            "per_target_min_fill_queue_effect"
        ),
        "training_chunks_shuffled": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "normal_policy_request_rate_used_as_budget": False,
        "normal_policy_constants_used_by_neural_inference": False,
        "probability_threshold_used": False,
        "neural_degree_cap": None,
        "fixed_page_offset_classes": None,
        "same_page_rule_used_by_neural_inference": False,
        "future_label_window_used": False,
        "fill_lead_cutoff_used": False,
        "handcrafted_semantic_features_used": False,
        "manual_loss_weights_used": False,
        "training_regularization_used": False,
        "inference_policy_hardcodes_used": False,
        "threshold_related_hardcodes_used": False,
        "gate_class_weighting_used": False,
        "gate_training_objective": (
            "empirical_prior_unweighted_categorical_nll"
        ),
        "gate_decoding_rule": "two_class_categorical_argmax",
        "gate_operating_point_learned_from_empirical_prior": True,
        "model_revision": "compact_empirical_prior_hurdle_autoregressive_gmm_fill_v2",
        "learned_request_count": True,
        "causal_no_future_self_test": "PASS",
        "cnn_architecture_self_test": "NOT_APPLICABLE",
        "event_logger_schema": EVENT_LOGGER_SCHEMA,
        "action_attachment_mode": ACTION_ATTACHMENT_MODE,
        "experiment_revision": EXPERIMENT_REVISION,
        "neural_role": "standalone_direct_action_prefetcher",
        "replay_preserves_explicit_fill_level": True,
        "source_decision_effective_external_input": SOURCE_INPUTS,
        "runtime_feature_count": 65,
        "track_model_family": TRACK_MODEL_FAMILY,
    }
    for key, expected in common.items():
        if metadata.get(key) != expected:
            failures.append(
                "{} metadata {}={!r}; expected {!r}".format(
                    tag, key, metadata.get(key), expected
                )
            )
    if metadata.get("source_contract_sha256") != source_contract_hash:
        failures.append("{} SPP source-contract SHA256 mismatch".format(tag))
    behavior = metadata.get("heldout_behavior_metrics")
    required_behavior = (
        "count_exact_match_rate", "target_precision", "target_recall",
        "target_f1", "normal_positive_callback_rate",
        "predicted_positive_callback_rate", "trigger_precision",
        "trigger_recall", "trigger_f1",
    )
    if not isinstance(behavior, dict):
        failures.append("{} lacks held-out behavior audit".format(tag))
    else:
        for key in required_behavior:
            value = behavior.get(key)
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                failures.append("{} invalid behavior metric {}".format(tag, key))
        fill_accuracy = behavior.get("fill_accuracy_on_matched_targets")
        if fill_accuracy is not None and (
            not isinstance(fill_accuracy, (int, float))
            or fill_accuracy < 0 or fill_accuracy > 1
        ):
            failures.append("{} invalid fill behavior accuracy".format(tag))
    family = metadata.get("model_family")
    if family != TRACK_MODEL_FAMILY:
        failures.append(
            "{} model family {!r}; expected {!r}".format(
                tag, family, TRACK_MODEL_FAMILY
            )
        )
    point = EXPECTED_POINTS.get((family, metadata.get("model_size")))
    if point is None:
        failures.append("{} is not a pinned matched architecture point".format(tag))
    else:
        if metadata.get("architecture_pair_id") != point:
            failures.append("{} architecture pair mismatch".format(tag))
        expected_parameters = EXPECTED_PARAMETER_COUNTS[metadata.get("model_size")]
        if metadata.get("parameter_count") != expected_parameters:
            failures.append(
                "{} parameter count {!r}; expected {!r}".format(
                    tag, metadata.get("parameter_count"), expected_parameters
                )
            )
    encoder_hashes = {
        metadata.get("runtime_encoder_sha256"),
        metadata.get("training_runtime_encoder_sha256"),
        metadata.get("inference_runtime_encoder_sha256"),
    }
    encoder_hash = next(iter(encoder_hashes)) if len(encoder_hashes) == 1 else None
    if not isinstance(encoder_hash, str) or len(encoder_hash) != 64:
        failures.append("{} train/inference encoder hash mismatch".format(tag))
    expected = {
        "training_state_mode": "chronological_stateful_tbptt",
        "training_state_carried_across_chunks": True,
        "training_state_detached_between_chunks": True,
        "inference_history_mode": "fresh_state_then_complete_train_guard_eval_chronology",
        "cnn_temporal_layers": 0,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            failures.append(
                "{} metadata {}={!r}; expected {!r}".format(
                    tag, key, metadata.get(key), value
                )
            )
    for role in ("train", "guard", "eval"):
        for kind in ("stream", "teacher_actions"):
            key = role + "_" + kind + "_content_sha256"
            expected_hash = (
                inputs.get(policy, {})
                .get(role, {})
                .get(kind, {})
                .get("content_sha256")
            )
            if metadata.get(key) != expected_hash:
                failures.append(
                    "{} {} {} content SHA256 mismatch".format(
                        tag, role, kind
                    )
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--model-tags",
        default=DEFAULT_MODEL_TAGS,
    )
    args = parser.parse_args()
    model_tags = [
        tag.strip() for tag in args.model_tags.split(",") if tag.strip()
    ]
    if not model_tags:
        raise SystemExit("--model-tags is empty")
    for tag in model_tags:
        if not tag.startswith("independent_delta_spp_lstm_"):
            raise SystemExit("invalid model tag {}".format(tag))

    methods = [
        "no_pref",
        "live_spp_reference",
        "offline_spp",
    ]
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
        if method == "offline_spp" or method.startswith("offline_independent_delta_spp_"):
            replay_text = log_path.read_text(errors="ignore")
            if "list_replayer_action_metadata captured_fill_level" not in replay_text:
                failures.append(
                    "{} did not use the fill-preserving SPP replayer".format(method)
                )
        policy = policy_for_method(method)
        row = {
            "trace": TRACE,
            "method": method,
            "model_tag": model_tag_for_method(method),
            "comparison_policy": policy,
            "matched_primary_comparison": int(bool(policy)),
            "log": str(log_path),
            "event_log": str(event_path),
        }
        row.update(parse_log(log_path))
        try:
            row.update(parse_events(event_path))
        except Exception as exc:
            failures.append("{} event parse failed: {}".format(method, exc))
        if row["ipc"] <= 0 or row["instructions"] <= 0:
            failures.append(
                "{} lacks final simulator statistics".format(method)
            )
        if row["l2_loads"] <= 0 or row["l2_load_miss"] <= 0:
            failures.append("{} lacks L2 load counters".format(method))
        if row.get("demand_l2_loads") != row["l2_loads"]:
            failures.append(
                "{} logger demand callbacks {} != simulator L2 loads {}".format(
                    method, row.get("demand_l2_loads"), row["l2_loads"]
                )
            )
        if (
            method == "live_{}_reference".format(POLICY)
            and row["pf_requested"] <= 0
        ):
            failures.append("{} emitted no requests".format(method))
        if method == "offline_" + POLICY and (
            row["matched"] <= 0 or row["emitted"] <= 0
        ):
            failures.append("{} did not replay keyed entries".format(method))
        if policy and row["callbacks"] <= 0:
            failures.append("{} reported zero replay callbacks".format(method))
        rows.append(row)

    if len(rows) != len(methods):
        failures.append("one or more methods are missing")
    if rows and len({row["instructions"] for row in rows}) != 1:
        failures.append("simulation instruction counts differ")

    input_dir = args.run_dir / "colab_input"
    source_contract_path = input_dir / "spp_source_contract.json"
    source_contract_hash = ""
    if not source_contract_path.is_file():
        failures.append("missing {}".format(source_contract_path))
    else:
        try:
            source_contract = json.loads(source_contract_path.read_text())
            if source_contract.get("decision_effective_external_input") != SOURCE_INPUTS:
                failures.append("SPP source contract external input mismatch")
            if (
                source_contract.get("self_target_action_semantics")
                != "allowed_by_source_lookahead_and_replayed"
                or source_contract.get("queue_effect_canonicalization")
                != "per_target_min_fill_queue_effect"
            ):
                failures.append("SPP self-target/queue contract mismatch")
            source_contract_hash = sha256(source_contract_path)
        except (OSError, json.JSONDecodeError) as exc:
            failures.append("invalid SPP source contract: {}".format(exc))
    collection_manifest = {}
    collection_manifest_path = input_dir / "collection_manifest.json"
    if not collection_manifest_path.is_file():
        failures.append("missing {}".format(collection_manifest_path))
    else:
        try:
            collection_manifest = json.loads(collection_manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            failures.append("invalid collection manifest: {}".format(exc))
        expected_manifest = {
            "status": "PASS",
            "trace": TRACE,
            "experiment_revision": EXPERIMENT_REVISION,
            "event_logger_schema": EVENT_LOGGER_SCHEMA,
            "action_attachment_mode": ACTION_ATTACHMENT_MODE,
            "policy": POLICY,
            "neural_role": "standalone_direct_action_prefetcher",
            "model_input_is_causal_external_event_sequence_only": True,
            "cache_fill_feedback_used_as_raw_external_input": True,
            "cache_fill_private_state_used_as_model_input": False,
            "teacher_actions_are_model_inputs": False,
            "same_external_input_contract": True,
            "training_inference_input_encoder_identical": True,
            "decoder_training_mode": "free_running_autoregressive_same_as_inference",
            "decoder_previous_teacher_action_used_as_input": False,
            "normal_policy_outputs_used_as_model_inputs": False,
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
            "fill_lead_cutoff_used": False,
            "inference_policy_hardcodes_used": False,
            "gate_class_weighting_used": False,
            "gate_training_objective": (
                "empirical_prior_unweighted_categorical_nll"
            ),
            "gate_decoding_rule": "two_class_categorical_argmax",
            "normal_candidate_bank_is_fixed": False,
            "nn_can_generate_actions_not_emitted_by_teacher": True,
            "model_does_not_use_pc": True,
            "cache_hit_and_type_are_audit_only": True,
            "source_decision_effective_external_input": SOURCE_INPUTS,
            "teacher_source_page_lines": 64,
            "fill_classes": ["FILL_L2", "FILL_LLC"],
            "complete_neural_action_space": True,
            "self_target_actions_allowed": True,
            "teacher_action_canonicalization": (
                "per_target_min_fill_queue_effect"
            ),
            "spp_source_contract_sha256": source_contract_hash,
        }
        for key, expected in expected_manifest.items():
            if collection_manifest.get(key) != expected:
                failures.append(
                    "collection manifest {}={!r}; expected {!r}".format(
                        key, collection_manifest.get(key), expected
                    )
                )
        if set(collection_manifest.get("tracks", {})) != {POLICY}:
            failures.append("collection manifest contains a foreign policy track")
    input_info = {}
    for policy in POLICIES:
        input_info[policy] = {}
        for role in ("train", "guard", "eval"):
            input_info[policy][role] = {}
            paths = {
                "stream": input_dir / (
                    TRACE + "." + policy + "." + role + "_stream.csv.gz"
                ),
                "teacher_actions": input_dir / (
                    TRACE + "." + policy + "." + role + "_teacher_actions.csv.gz"
                ),
            }
            for kind, path in paths.items():
                if not path.is_file():
                    failures.append("missing input {}".format(path))
                    continue
                try:
                    input_info[policy][role][kind] = stream_hashes(path)
                except (OSError, gzip.BadGzipFile) as exc:
                    failures.append(
                        "cannot hash {}: {}".format(path, exc)
                    )

    metadata_by_tag = {}
    normal_hashes = {policy: {} for policy in POLICIES}
    pairs = defaultdict(list)
    for tag in model_tags:
        policy = POLICY
        required = (
            "offline_{}.replay.csv".format(policy),
            "offline_nn.replay.csv",
            "model.pt",
            "run_metadata.json",
            "training_history.csv",
        )
        for name in required:
            path = colab_root / tag / name
            if not path.is_file():
                failures.append("missing Colab output {}".format(path))
        metadata_path = colab_root / tag / "run_metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(
                "{} metadata parse failed: {}".format(tag, exc)
            )
            continue
        metadata_by_tag[tag] = metadata
        validate_metadata(
            metadata, tag, input_info, source_contract_hash, failures
        )
        behavior = metadata.get("heldout_behavior_metrics", {})
        for row in rows:
            if row.get("model_tag") == tag:
                for key, value in behavior.items():
                    row["behavior_" + key] = value
        pair_id = metadata.get("architecture_pair_id")
        if not pair_id:
            failures.append("{} lacks architecture_pair_id".format(tag))
        else:
            pairs[(policy, pair_id)].append(metadata)
        list_path = colab_root / tag / (
            "offline_{}.replay.csv".format(policy)
        )
        if list_path.is_file():
            try:
                normal_info = replay_list_info(list_path)
                normal_hashes[policy][tag] = normal_info["sha256"]
                if metadata.get("normal_list_sha256") != normal_info["sha256"]:
                    failures.append("{} normal-list SHA256 mismatch".format(tag))
                if metadata.get("offline_normal_entries") != normal_info["entries"]:
                    failures.append("{} normal-list entry count mismatch".format(tag))
                if metadata.get("offline_normal_fill_level_counts") != normal_info["fill_counts"]:
                    failures.append("{} normal-list fill counts mismatch".format(tag))
            except (OSError, RuntimeError, ValueError) as exc:
                failures.append("{} invalid normal replay list: {}".format(tag, exc))
        nn_path = colab_root / tag / "offline_nn.replay.csv"
        if nn_path.is_file():
            try:
                nn_info = replay_list_info(nn_path, allow_empty=True)
                if metadata.get("nn_list_sha256") != nn_info["sha256"]:
                    failures.append("{} NN-list SHA256 mismatch".format(tag))
                if metadata.get("offline_nn_entries") != nn_info["entries"]:
                    failures.append("{} NN-list entry count mismatch".format(tag))
                if metadata.get("offline_nn_fill_level_counts") != nn_info["fill_counts"]:
                    failures.append("{} NN-list fill counts mismatch".format(tag))
            except (OSError, RuntimeError, ValueError) as exc:
                failures.append("{} invalid NN replay list: {}".format(tag, exc))

    for policy, hashes in normal_hashes.items():
        if hashes and len(set(hashes.values())) != 1:
            failures.append(
                "offline {} list differs across architecture points".format(
                    policy
                )
            )
    for (policy, pair_id), members in pairs.items():
        families = {member.get("model_family") for member in members}
        if len(members) != 1 or families != {TRACK_MODEL_FAMILY}:
            failures.append(
                "{} group {} must contain one {} point".format(
                    policy, pair_id, TRACK_MODEL_FAMILY
                )
            )

    add_comparison_metrics(rows, failures)
    by_method = {row["method"]: row for row in rows}
    behavior_fields = sorted({
        key
        for row in rows
        for key in row
        if key.startswith("behavior_")
    })
    for row in rows:
        for field in behavior_fields:
            row.setdefault(field, "")
    fields = [
        "trace", "method", "model_tag", "comparison_policy",
        "matched_primary_comparison", "ipc", "speedup_vs_no_pref",
        "instructions", "cycles", "l2_loads", "l2_load_miss",
        "l2_load_miss_rate", "miss_reduction_vs_no_pref",
        "pf_requested", "pf_dropped", "pf_issued", "nodup_issued",
        "pf_filled", "pf_useful", "pf_useless", "pf_late",
        "pq_merged_duplicate_proxy", "accuracy", "selected_accuracy",
        "coverage_vs_no_pref_l2_miss", "timeliness", "late_per_issued",
        "drop_rate", "useless_per_issued", "pollution_risk_proxy",
        "balanced_parity_index", "parity_miss_rate",
        "parity_selected_accuracy", "parity_coverage", "parity_timeliness",
        "parity_bottleneck", "ipc_delta_vs_offline_normal",
        "ipc_pct_vs_offline_normal",
        "l2_miss_rate_delta_vs_offline_normal",
        "selected_accuracy_delta_vs_offline_normal",
        "coverage_delta_vs_offline_normal",
        "timeliness_delta_vs_offline_normal",
        "prefetch_request_ratio_vs_offline_normal",
        "prefetch_request_reduction_vs_offline_normal",
        "pollution_proxy_delta_vs_offline_normal",
        "demand_l2_loads", "demand_l2_hits", "demand_l2_misses",
        "prefetch_useful_demand_hits", "prefetch_late_demand_misses",
        "prefetch_request_events", "prefetch_accepted_events",
        "prefetch_duplicate_events", "prefetch_fill_l2_events",
        "prefetch_fill_llc_events", "cache_fill_feedback_events",
        "event_selected_accuracy_proxy",
        "event_coverage_vs_no_pref_l2_miss", "event_timeliness_proxy",
        "matched", "emitted", "callbacks", *behavior_fields,
        "log", "event_log",
    ]
    with (args.run_dir / "matched_comparison.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    insight_fields = [
        "comparison_policy", "method", "model_tag", "ipc",
        "ipc_delta_vs_offline_normal", "ipc_pct_vs_offline_normal",
        "l2_load_miss_rate", "l2_miss_rate_delta_vs_offline_normal",
        "selected_accuracy", "selected_accuracy_delta_vs_offline_normal",
        "coverage_vs_no_pref_l2_miss",
        "coverage_delta_vs_offline_normal", "timeliness",
        "timeliness_delta_vs_offline_normal", "pollution_risk_proxy",
        "pollution_proxy_delta_vs_offline_normal", "pf_requested",
        "prefetch_request_reduction_vs_offline_normal",
        "balanced_parity_index", "parity_bottleneck",
        *behavior_fields,
    ]
    with (args.run_dir / "insight_summary.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=insight_fields)
        writer.writeheader()
        writer.writerows(
            {key: row.get(key, "") for key in insight_fields}
            for row in rows if row["comparison_policy"]
        )

    warnings = []
    transport_fidelity = {}
    for policy in POLICIES:
        live = by_method.get("live_{}_reference".format(policy))
        offline = by_method.get("offline_" + policy)
        if live is None or offline is None:
            continue
        fidelity = {
            "offline_minus_live_ipc": offline["ipc"] - live["ipc"],
            "ipc_relative_error": abs(div(
                offline["ipc"] - live["ipc"], live["ipc"]
            )),
            "l2_miss_relative_error": abs(div(
                offline["l2_load_miss"] - live["l2_load_miss"],
                live["l2_load_miss"],
            )),
            "request_count_ratio": div(
                offline["pf_requested"], live["pf_requested"]
            ),
        }
        transport_fidelity[policy] = fidelity

    status = "FAIL" if failures else "PASS"
    payload = {
        "status": status,
        "fair_comparison_claim_allowed": not failures,
        "trace": TRACE,
        "model_family_track": TRACK_MODEL_FAMILY,
        "trace_selection": {
            "reason": (
                "SPP is evaluated in its own matched track because its "
                "signature/pattern state and fill actions differ from stride."
            ),
            "historical_ipc_context": {
                "no_pref": 0.36800,
                "spp": 0.42836,
            },
        },
        "primary_comparisons": {
            "spp_track": [
                "offline_spp",
                "offline_independent_delta_spp_{}_<capacity>".format(
                    TRACK_MODEL_FAMILY
                ),
            ],
        },
        "context_reference_only": [
            "no_pref", "live_spp_reference"
        ],
        "transport_fidelity": transport_fidelity,
        "warnings": warnings,
        "track_guardrail": (
            "Every neural point learns zero versus an unbounded positive request count, direct signed "
            "cache-line deltas, and fill level from the same chronological DEMAND(addr) and "
            "CACHE_FILL(evicted_addr) external callback stream. "
            "No fixed page-offset interface or captured SPP action is an inference input."
        ),
        "transport": (
            "Each normal SPP action is attached by explicit prior "
            "trigger_event_id. Normal and independently generated neural "
            "action lists use the same PC-line-occ replayer and preserve "
            "explicit FILL_L2/FILL_LLC actions. Repeated source calls to one "
            "target are canonicalized with ChampSim's PQ merge rule for the "
            "offline normal comparator and supervised training targets."
        ),
        "direct_action_contract": {
            "count_distribution": "learned zero/positive categorical mode plus learned positive log-count with unbounded support",
            "target_distribution": "autoregressive signed cache-line delta mixture",
            "fill_classes": ["FILL_L2", "FILL_LLC"],
            "decision": "categorical argmax, rounded-exp learned positive count, autoregressive mixture modes, fill argmax",
            "probability_threshold": None,
            "neural_degree_cap": None,
            "teacher_action_canonicalization": (
                "per_target_min_fill_queue_effect"
            ),
            "teacher_actions_are_model_inputs": False,
        },
        "spp_input_guardrail": {
            "audited_source_entry": (
                "SPP_dev2::invoke_prefetcher(..., addr, ...) and "
                "SPP_dev2::cache_fill(..., evicted_addr)"
            ),
            "decision_effective_external_input": SOURCE_INPUTS,
            "signature_fields_audit_or_transport_only": [
                "ip", "cache_hit", "type", "cache_fill.addr", "set",
                "way", "prefetch"
            ],
            "normal_spp_private_causal_state": [
                "signature_table", "pattern_table", "GHR", "confidence",
                "prefetch_filter_usefulness_feedback",
            ],
            "direct_nn_inputs": [
                "lossless uint64 callback address encoding",
                "one lossless DEMAND/FILL callback-kind bit",
            ],
            "not_nn_inputs": [
                "PC", "cycle", "hit/miss", "queue state", "SPP confidence",
                "SPP private tables/GHR/filter contents", "accepted/duplicate",
                "teacher actions", "future evaluation rows",
            ],
            "interpretation": (
                "The selected standalone architecture learns causal history "
                "from the complete source-effective external callback sequence. "
                "Raw evicted_addr feedback is shared because source SPP reads "
                "it, but normal SPP's derived FILTER/GHR state is not exposed."
            ),
        },
        "architecture_contract": (
            {
                "name": "causal residual temporal convolutional network",
                "temporal_convolution_layers": 2,
                "kernel_size_events": 17,
                "stride_events": 1,
                "dilations": [1, 17],
                "left_context_events": 288,
                "receptive_field_events": 289,
                "interpretation": (
                    "a contiguous causal sliding window over DEMAND/FILL; "
                    "output t sees no future input"
                ),
            }
            if TRACK_MODEL_FAMILY == "cnn"
            else {
                "name": "stateful LSTM",
                "history": "complete train then guard then evaluation chronology",
                "training": "chronological TBPTT with state carried and detached",
            }
        ),
        "metric_definitions": {
            "l2_load_miss_rate": (
                "Core_0_L2C_load_miss / Core_0_L2C_loads; lower is better"
            ),
            "selected_accuracy": (
                "Core_0_L2C_prefetch_useful / "
                "(Core_0_L2C_prefetch_issued - Core_0_L2C_pq_merged)"
            ),
            "coverage_vs_no_pref_l2_miss": (
                "Core_0_L2C_prefetch_useful / "
                "no-prefetch Core_0_L2C_load_miss"
            ),
            "timeliness": (
                "Core_0_L2C_prefetch_useful / "
                "(Core_0_L2C_prefetch_useful + Core_0_L2C_prefetch_late)"
            ),
            "pollution_risk_proxy": (
                "1 - selected_accuracy; proxy only, not harmful evictions"
            ),
            "balanced_parity_index": (
                "100 * (q_miss_rate * q_selected_accuracy * q_coverage * "
                "q_timeliness)^(1/4); each equally weighted ratio is "
                "capped at 1 against the model's own offline normal policy"
            ),
        },
        "balanced_parity_guardrail": (
            "BPI summarizes within-track normal-policy parity; IPC and "
            "speedup versus no-prefetch remain separate outcomes."
        ),
        "input_provenance": {
            "current_input_dir": str(input_dir),
            "collection_manifest": collection_manifest,
            "spp_source_contract_sha256": source_contract_hash,
            "policy_inputs": input_info,
        },
        "offline_normal_list_hashes_by_model_tag": normal_hashes,
        "runtime_encoder_sha256_by_model_tag": {
            tag: metadata_by_tag[tag].get("runtime_encoder_sha256")
            for tag in sorted(metadata_by_tag)
        },
        "architecture_pairs": {
            "{}:{}".format(policy, pair_id): [
                {
                    "model_tag": member.get("model_tag"),
                    "model_family": member.get("model_family"),
                    "model_size": member.get("model_size"),
                    "parameter_count": member.get("parameter_count"),
                }
                for member in sorted(
                    members, key=lambda item: item.get("model_family", "")
                )
            ]
            for (policy, pair_id), members in sorted(pairs.items())
        },
        "failures": failures,
        "rows": rows,
    }

    if not failures:
        no_pref = by_method["no_pref"]
        tracks = {}
        for policy in POLICIES:
            normal = by_method["offline_" + policy]
            points = []
            for tag in model_tags:
                if not tag.startswith("independent_delta_" + policy + "_"):
                    continue
                row = by_method["offline_" + tag]
                metadata = metadata_by_tag[tag]
                points.append({
                    "model_tag": tag,
                    "model_family": metadata["model_family"],
                    "model_size": metadata["model_size"],
                    "architecture_pair_id": metadata["architecture_pair_id"],
                    "parameter_count": metadata["parameter_count"],
                    "ipc": row["ipc"],
                    "ipc_delta_vs_offline_normal": (
                        row["ipc"] - normal["ipc"]
                    ),
                    "ipc_delta_vs_no_pref": row["ipc"] - no_pref["ipc"],
                    "beats_offline_normal": row["ipc"] > normal["ipc"],
                    "beats_no_pref": row["ipc"] > no_pref["ipc"],
                    "l2_load_miss_rate": row["l2_load_miss_rate"],
                    "selected_accuracy": row["selected_accuracy"],
                    "coverage_vs_no_pref_l2_miss": (
                        row["coverage_vs_no_pref_l2_miss"]
                    ),
                    "timeliness": row["timeliness"],
                    "pollution_risk_proxy": row["pollution_risk_proxy"],
                    "prefetch_request_reduction_vs_offline_normal": (
                        row["prefetch_request_reduction_vs_offline_normal"]
                    ),
                    "balanced_parity_index": (
                        row["balanced_parity_index"]
                    ),
                    "parity_bottleneck": row["parity_bottleneck"],
                })
            tracks[policy] = {
                "offline_normal_ipc": normal["ipc"],
                "models": points,
                "best_model_by_ipc": max(
                    points, key=lambda point: point["ipc"]
                ),
                "best_model_by_balanced_parity": max(
                    points,
                    key=lambda point: point["balanced_parity_index"],
                ),
                "any_model_beats_offline_normal": any(
                    point["beats_offline_normal"] for point in points
                ),
            }
        payload["no_pref_ipc"] = no_pref["ipc"]
        payload["tracks"] = tracks
        pair_rows = []
        for (policy, pair_id), members in sorted(pairs.items()):
            by_family = {
                member["model_family"]: member for member in members
            }
            if set(by_family) != {"lstm", "cnn"}:
                continue
            lstm_meta = by_family["lstm"]
            cnn_meta = by_family["cnn"]
            lstm_row = by_method["offline_" + lstm_meta["model_tag"]]
            cnn_row = by_method["offline_" + cnn_meta["model_tag"]]
            pair_rows.append({
                "comparison_policy": policy,
                "architecture_pair_id": pair_id,
                "lstm_tag": lstm_meta["model_tag"],
                "cnn_tag": cnn_meta["model_tag"],
                "lstm_parameters": lstm_meta["parameter_count"],
                "cnn_parameters": cnn_meta["parameter_count"],
                "cnn_minus_lstm_ipc": cnn_row["ipc"] - lstm_row["ipc"],
                "cnn_minus_lstm_l2_miss_rate": (
                    cnn_row["l2_load_miss_rate"]
                    - lstm_row["l2_load_miss_rate"]
                ),
                "cnn_minus_lstm_selected_accuracy": (
                    cnn_row["selected_accuracy"]
                    - lstm_row["selected_accuracy"]
                ),
                "cnn_minus_lstm_coverage": (
                    cnn_row["coverage_vs_no_pref_l2_miss"]
                    - lstm_row["coverage_vs_no_pref_l2_miss"]
                ),
                "cnn_minus_lstm_timeliness": (
                    cnn_row["timeliness"] - lstm_row["timeliness"]
                ),
                "cnn_minus_lstm_balanced_parity": (
                    cnn_row["balanced_parity_index"]
                    - lstm_row["balanced_parity_index"]
                ),
                "lstm_behavior_target_f1": lstm_meta["heldout_behavior_metrics"]["target_f1"],
                "cnn_behavior_target_f1": cnn_meta["heldout_behavior_metrics"]["target_f1"],
                "cnn_minus_lstm_behavior_target_f1": (
                    cnn_meta["heldout_behavior_metrics"]["target_f1"]
                    - lstm_meta["heldout_behavior_metrics"]["target_f1"]
                ),
                "lstm_count_exact_match_rate": lstm_meta["heldout_behavior_metrics"]["count_exact_match_rate"],
                "cnn_count_exact_match_rate": cnn_meta["heldout_behavior_metrics"]["count_exact_match_rate"],
                "lstm_fill_accuracy": lstm_meta["heldout_behavior_metrics"]["fill_accuracy_on_matched_targets"],
                "cnn_fill_accuracy": cnn_meta["heldout_behavior_metrics"]["fill_accuracy_on_matched_targets"],
                "ipc_winner": (
                    "cnn" if cnn_row["ipc"] > lstm_row["ipc"] else "lstm"
                ),
                "balanced_parity_winner": (
                    "cnn"
                    if cnn_row["balanced_parity_index"]
                    > lstm_row["balanced_parity_index"]
                    else "lstm"
                ),
                "behavior_imitation_winner": (
                    "cnn"
                    if cnn_meta["heldout_behavior_metrics"]["target_f1"]
                    > lstm_meta["heldout_behavior_metrics"]["target_f1"]
                    else "lstm"
                ),
            })
        if pair_rows:
            pair_fields = list(pair_rows[0].keys())
            with (args.run_dir / "architecture_pair_summary.csv").open(
                "w", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=pair_fields)
                writer.writeheader()
                writer.writerows(pair_rows)
        payload["architecture_pair_comparisons"] = pair_rows
        payload["cross_directory_interpretation_rule"] = {
            "cnn_wins": (
                "At paired reported capacities, the 289-event causal CNN "
                "improves IPC/miss rate without material BPI loss: immediate "
                "and medium-range address correlation is sufficient."
            ),
            "lstm_wins": (
                "At paired reported capacities, the stateful LSTM improves IPC and "
                "BPI: useful information extends beyond the TCN receptive field."
            ),
            "both_fail": (
                "Both direct students fail to discover enough useful direct "
                "actions: the restricted address representation or missing "
                "private SPP feedback is more important than architecture."
            ),
        }

    out_json = args.run_dir / "matched_comparison.json"
    out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print("[{}] {}".format(status, out_json))
    if failures:
        raise SystemExit(" | ".join(failures))


if __name__ == "__main__":
    main()
