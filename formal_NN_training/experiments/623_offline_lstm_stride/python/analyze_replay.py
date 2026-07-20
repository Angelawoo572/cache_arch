#!/usr/bin/env python3
"""Fail-closed analysis for the standalone 623 Stride/LSTM track."""
import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


TRACE = "623.xalancbmk_s-700B"
POLICY = "stride"
POLICIES = (POLICY,)
EXPERIMENT_REVISION = "stride_source_input_variable_delta_free_running_v9"
MODEL_REVISION = "compact_pc_keyed_crn_event_sampled_mixture_v15"
TRACK_MODEL_FAMILY = "lstm"
DEFAULT_MODEL_TAGS = (
    "independent_delta_stride_lstm_h8,independent_delta_stride_lstm_h16,"
    "independent_delta_stride_lstm_h32,independent_delta_stride_lstm_h64,"
    "independent_delta_stride_lstm_h128"
)
EXPECTED_POINTS = {
    ("lstm", 8): "p0",
    ("lstm", 16): "p1",
    ("lstm", 32): "p2",
    ("lstm", 64): "p3",
    ("lstm", 128): "p4",
}
EXPECTED_PARAMETERS = {8: 1923, 16: 5243, 32: 16107, 64: 54731, 128: 199563}
EVENT_LOGGER_SCHEMA = "623_causal_trigger_v5"
CANDIDATE_ATTACHMENT_MODE = "explicit_trigger_event_id"
KV = re.compile(r"^([A-Za-z0-9_]+)\s+([-+0-9.eE]+)\s*$")
FINISHED = re.compile(
    r"Finished CPU\s+0\s+instructions:\s+(\d+)\s+cycles:\s+(\d+)\s+"
    r"cumulative IPC:\s+([-+0-9.eE]+)"
)
REPLAYER = re.compile(
    r"emitted\s+(\d+)\s+candidates over\s+(\d+)\s+runtime ROI L2 LOAD "
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
    triggers = set()
    with Path(path).open(newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != ["pc", "line", "occ", "prefetch_addr"]:
            raise RuntimeError("invalid four-column stride replay header")
        for line_number, fields in enumerate(reader, 2):
            if len(fields) != 4:
                raise RuntimeError(
                    "invalid stride replay row {}".format(line_number)
                )
            try:
                pc = int(fields[0], 0)
                line = int(fields[1], 0)
                occurrence = int(fields[2], 10)
                address = int(fields[3], 0)
            except ValueError as exc:
                raise RuntimeError(
                    "invalid stride replay integer at row {}: {}".format(
                        line_number, exc
                    )
                )
            if (
                min(pc, line, occurrence, address) < 0
                or pc > (1 << 64) - 1
                or line > (1 << 58) - 1
                or address > (1 << 64) - 64
                or address % 64
            ):
                raise RuntimeError(
                    "out-of-range/unaligned stride replay row {}".format(
                        line_number
                    )
                )
            trigger = (pc, line, occurrence)
            triggers.add(trigger)
            count += 1
    if count <= 0 and not allow_empty:
        raise RuntimeError("empty stride replay list")
    return {
        "entries": count,
        "unique_triggers": len(triggers),
        "sha256": sha256(path),
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
        "one_minus_selected_accuracy": (
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
    }
    last_event_id = -1
    latest_demand_event_id = None
    latest_demand_identity = None
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "event", "event_id", "cpu", "cycle", "cache", "ip", "line",
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
                raise RuntimeError("stale/non-L2 623 event logger row")
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
                if fill_level != 2:
                    raise RuntimeError("stride PF did not preserve FILL_L2")
                result["prefetch_request_events"] += 1
                result["prefetch_accepted_events"] += int(row["accepted"])
                result["prefetch_duplicate_events"] += int(row["duplicate"])
                result["prefetch_fill_l2_events"] += 1
                continue
            if row["event"] != "DEMAND":
                continue
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
    normal = "offline_" + POLICY
    if method == normal or method.startswith("offline_independent_delta_" + POLICY + "_"):
        return POLICY
    return ""


def model_tag_for_method(method):
    if method.startswith("offline_independent_delta_" + POLICY + "_"):
        return method[len("offline_"):]
    return ""


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
        row["ipc_delta_vs_offline_normal"] = ""
        row["ipc_pct_vs_offline_normal"] = ""
        row["l2_miss_rate_delta_vs_offline_normal"] = ""
        row["selected_accuracy_delta_vs_offline_normal"] = ""
        row["coverage_delta_vs_offline_normal"] = ""
        row["timeliness_delta_vs_offline_normal"] = ""
        row["prefetch_request_ratio_vs_offline_normal"] = ""
        row["prefetch_request_reduction_vs_offline_normal"] = ""
        row["one_minus_selected_accuracy_delta_vs_offline_normal"] = ""

    for policy in POLICIES:
        baseline = by_method.get("offline_" + policy)
        if baseline is None:
            failures.append("offline {} baseline is missing".format(policy))
            continue
        for row in rows:
            if row["comparison_policy"] != policy:
                continue
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
            row["one_minus_selected_accuracy_delta_vs_offline_normal"] = (
                row["one_minus_selected_accuracy"]
                - baseline["one_minus_selected_accuracy"]
            )


def validate_metadata(metadata, tag, inputs, failures):
    policy = POLICY
    common = {
        "trace": TRACE,
        "model_tag": tag,
        "matched_normal_prefetcher": policy,
        "source_decision_effective_external_input": ["pc", "addr"],
        "same_external_input_contract": True,
        "training_inference_input_encoder_identical": True,
        "decoder_training_mode": "free_running_autoregressive_same_as_inference",
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_free_running_self_test": "PASS",
        "training_runtime_fields": ["pc", "addr"],
        "inference_runtime_fields": ["pc", "addr"],
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "normal_policy_request_rate_used_as_budget": False,
        "normal_policy_constants_used_by_neural_inference": False,
        "probability_threshold_used": False,
        "threshold_related_hardcodes_used": False,
        "neural_degree_cap": None,
        "fixed_page_offset_classes": None,
        "same_page_rule_used_by_neural_inference": False,
        "future_label_window_used": False,
        "handcrafted_semantic_features_used": False,
        "manual_loss_weights_used": False,
        "data_derived_gate_class_weights_used": False,
        "gate_class_weighting_used": False,
        "gate_training_objective": "unweighted_bernoulli_nll",
        "gate_decoding_rule": "event_keyed_bernoulli_inverse_cdf",
        "request_count_training_objective": (
            "unweighted_bernoulli_hurdle_plus_positive_poisson_excess_nll"
        ),
        "request_count_decoding_rule": (
            "event_keyed_bernoulli_plus_common_quantile_poisson_inverse_cdf"
        ),
        "request_count_residual_scope": "none_event_local",
        "training_regularization_used": False,
        "inference_policy_hardcodes_used": False,
        "learned_request_count": True,
        "nn_generates_own_target_addresses": True,
        "training_chunks_shuffled": False,
        "causal_no_future_self_test": "PASS",
        "event_keyed_crn_self_test": "PASS",
        "event_keyed_hurdle_count_self_test": "PASS",
        "canonicalized_mixture_sampling_self_test": "PASS",
        "decoder_probability_mass_carries_train_guard_history": False,
        "cross_event_probability_credit_used": False,
        "sampled_outputs_used_as_decoder_feedback": False,
        "stochastic_decoding_reproducible": True,
        "delta_mixture_decoding_rule": (
            "event_keyed_mean_sorted_categorical_inverse_cdf_then_component_mean"
        ),
        "delta_decoder_feedback_rule": (
            "complete_mixture_expectation_same_in_training_and_inference"
        ),
        "delta_mixture_components": 3,
        "cnn_architecture_self_test": "NOT_APPLICABLE",
        "event_logger_schema": EVENT_LOGGER_SCHEMA,
        "candidate_attachment_mode": CANDIDATE_ATTACHMENT_MODE,
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": MODEL_REVISION,
        "neural_role": "standalone_direct_action_prefetcher",
        "track_model_family": TRACK_MODEL_FAMILY,
        "runtime_feature_count": 122,
        "runtime_encoding": (
            "lossless uint64 PC plus lossless 58-bit cache-line number"
        ),
        "runtime_pc_bits": 64,
        "runtime_line_number_bits": 58,
        "runtime_constant_offset_bits_removed": 6,
        "common_random_numbers_across_capacities": True,
        "strict_common_random_numbers_across_capacities": True,
        "cross_event_rng_state_used": False,
        "decoder_sampling_roles": ["eval"],
        "decoder_train_sampling_performed": False,
        "decoder_guard_sampling_performed": False,
        "decoder_event_key_definition": "zero_based_eval_demand_idx",
        "decoder_event_key_uses_teacher_information": False,
        "decoder_action_rank_origin": 0,
        "decoder_key_includes_sampler_revision": True,
    }
    for key, expected in common.items():
        if metadata.get(key) != expected:
            failures.append(
                "{} metadata {}={!r}; expected {!r}".format(
                    tag, key, metadata.get(key), expected
                )
            )
    sampler = metadata.get("decoder_sampler")
    expected_key_fields = [
        "revision", "decoder_seed", "trace", "policy", "role",
        "event_key", "head", "action_rank",
    ]
    if (
        not isinstance(sampler, dict)
        or sampler.get("sampler_revision")
        != "sha256_event_keyed_inverse_cdf_crn_v1"
        or sampler.get("key_fields") != expected_key_fields
        or sampler.get("poisson_backend") != "scipy.stats.poisson.ppf"
        or sampler.get("cross_event_rng_state") is not False
        or metadata.get("decoder_key_fields") != expected_key_fields
    ):
        failures.append("{} keyed decoder sampler contract mismatch".format(tag))
    for key in (
        "decoder_event_key_stream_sha256",
        "decoder_sampler_source_sha256",
        "decoder_sampler_key_schedule_sha256",
        "decoder_sampling_schedule_sha256",
        "training_state_router_sha256",
        "inference_state_router_sha256",
    ):
        value = metadata.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            failures.append("{} invalid {}".format(tag, key))
    if metadata.get("training_state_router_sha256") != metadata.get(
        "inference_state_router_sha256"
    ):
        failures.append("{} train/inference state routers differ".format(tag))
    for key in (
        "peak_training_recurrent_state_bytes_float32",
        "peak_inference_recurrent_state_bytes_float32",
    ):
        if not isinstance(metadata.get(key), int) or metadata.get(key) <= 0:
            failures.append("{} invalid {}".format(tag, key))
    family = metadata.get("model_family")
    if family != TRACK_MODEL_FAMILY:
        failures.append(
            "{} model family {!r}; expected {!r}".format(
                tag, family, TRACK_MODEL_FAMILY
            )
        )
    point = EXPECTED_POINTS.get((family, metadata.get("model_size")))
    if point is None:
        failures.append("{} is not a pinned LSTM point".format(tag))
    else:
        if metadata.get("architecture_pair_id") != point:
            failures.append("{} architecture group mismatch".format(tag))
        expected_parameters = EXPECTED_PARAMETERS.get(metadata.get("model_size"))
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
        for kind in ("stream", "candidate"):
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
        if not tag.startswith("independent_delta_stride_lstm_"):
            raise SystemExit("invalid model tag {}".format(tag))

    methods = [
        "no_pref",
        "live_stride_reference",
        "offline_stride",
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
            "candidate_attachment_mode": CANDIDATE_ATTACHMENT_MODE,
            "policy": POLICY,
            "independent_matched_track": True,
            "neural_role": "standalone_direct_action_prefetcher",
            "source_decision_effective_external_input": ["pc", "addr"],
            "same_external_input_contract": True,
            "normal_policy_outputs_used_as_model_inputs": False,
            "normal_policy_candidates_used_as_model_inputs": False,
            "normal_policy_private_state_used_as_model_inputs": False,
            "normal_policy_outputs_used_as_training_targets": True,
            "normal_policy_request_rate_used_as_budget": False,
            "normal_policy_constants_used_by_neural_inference": False,
            "probability_threshold_used": False,
            "neural_degree_cap": None,
            "future_label_window_used": False,
            "inference_policy_hardcodes_used": False,
            "nn_generates_own_target_addresses": True,
            "model_input_excludes_action_outcomes": True,
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
                "candidate": input_dir / (
                    TRACE + "." + policy + "." + role + "_candidates.csv.gz"
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
    replay_lists = {}
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
        validate_metadata(metadata, tag, input_info, failures)
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
                replay_lists.setdefault("offline_" + policy, normal_info)
                normal_hashes[policy][tag] = normal_info["sha256"]
                if metadata.get("normal_list_sha256") != normal_info["sha256"]:
                    failures.append("{} normal-list SHA256 mismatch".format(tag))
                if metadata.get("offline_normal_entries") != normal_info["entries"]:
                    failures.append("{} normal-list entry count mismatch".format(tag))
            except (OSError, RuntimeError, ValueError) as exc:
                failures.append("{} invalid normal replay list: {}".format(tag, exc))
        nn_path = colab_root / tag / "offline_nn.replay.csv"
        if nn_path.is_file():
            try:
                nn_info = replay_list_info(nn_path, allow_empty=True)
                replay_lists["offline_" + tag] = nn_info
                if metadata.get("nn_list_sha256") != nn_info["sha256"]:
                    failures.append("{} NN-list SHA256 mismatch".format(tag))
                if metadata.get("offline_nn_entries") != nn_info["entries"]:
                    failures.append("{} NN-list entry count mismatch".format(tag))
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

    rows_by_method = {row["method"]: row for row in rows}
    for method in ["offline_" + POLICY] + [
        "offline_" + tag for tag in model_tags
    ]:
        row = rows_by_method.get(method)
        info = replay_lists.get(method)
        if row is None or info is None:
            failures.append("{} lacks replay accounting inputs".format(method))
            continue
        checks = {
            "list entries versus replayer emitted": (
                info["entries"], row["emitted"]
            ),
            "list entries versus simulator requested": (
                info["entries"], row["pf_requested"]
            ),
            "list entries versus PF events": (
                info["entries"], row["prefetch_request_events"]
            ),
            "unique list triggers versus matched triggers": (
                info["unique_triggers"], row["matched"]
            ),
            "replayer callbacks versus L2 loads": (
                row["callbacks"], row["l2_loads"]
            ),
        }
        for label, (left, right) in checks.items():
            if left != right:
                failures.append(
                    "{} {} {} != {}".format(method, label, left, right)
                )

    add_comparison_metrics(rows, failures)
    by_method = {row["method"]: row for row in rows}
    fields = [
        "trace", "method", "model_tag", "comparison_policy",
        "matched_primary_comparison", "ipc", "speedup_vs_no_pref",
        "instructions", "cycles", "l2_loads", "l2_load_miss",
        "l2_load_miss_rate", "miss_reduction_vs_no_pref",
        "pf_requested", "pf_dropped", "pf_issued", "nodup_issued",
        "pf_filled", "pf_useful", "pf_useless", "pf_late",
        "pq_merged_duplicate_proxy", "accuracy", "selected_accuracy",
        "coverage_vs_no_pref_l2_miss", "timeliness", "late_per_issued",
        "drop_rate", "useless_per_issued", "one_minus_selected_accuracy",
        "ipc_delta_vs_offline_normal",
        "ipc_pct_vs_offline_normal",
        "l2_miss_rate_delta_vs_offline_normal",
        "selected_accuracy_delta_vs_offline_normal",
        "coverage_delta_vs_offline_normal",
        "timeliness_delta_vs_offline_normal",
        "prefetch_request_ratio_vs_offline_normal",
        "prefetch_request_reduction_vs_offline_normal",
        "one_minus_selected_accuracy_delta_vs_offline_normal",
        "demand_l2_loads", "demand_l2_hits", "demand_l2_misses",
        "prefetch_useful_demand_hits", "prefetch_late_demand_misses",
        "prefetch_request_events", "prefetch_accepted_events",
        "prefetch_duplicate_events", "prefetch_fill_l2_events",
        "prefetch_fill_llc_events", "event_selected_accuracy_proxy",
        "event_coverage_vs_no_pref_l2_miss", "event_timeliness_proxy",
        "matched", "emitted", "callbacks", "log", "event_log",
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
        "timeliness_delta_vs_offline_normal", "one_minus_selected_accuracy",
        "one_minus_selected_accuracy_delta_vs_offline_normal", "pf_requested",
        "prefetch_request_reduction_vs_offline_normal",
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
                "On historical 623 data, stride is approximately neutral "
                "versus no-prefetch, making it a useful low-gain local-pattern "
                "track for standalone neural direct-action comparisons."
            ),
            "historical_ipc_context": {
                "no_pref": 0.35321,
                "stride": 0.35340,
            },
        },
        "primary_comparisons": {
            "stride_track": [
                "offline_stride",
                "offline_independent_delta_stride_{}_<capacity>".format(
                    TRACK_MODEL_FAMILY
                ),
            ],
        },
        "context_reference_only": [
            "no_pref", "live_stride_reference"
        ],
        "transport_fidelity": transport_fidelity,
        "replay_accounting": replay_lists,
        "warnings": warnings,
        "track_guardrail": (
            "Every neural point sees only the same PC/address stream as Stride. "
            "Captured Stride actions are supervised targets and comparator replay "
            "only; inference has no threshold, request budget, or degree cap."
        ),
        "transport": (
            "Captured Stride actions and learned neural actions are replayed "
            "through the same PC-line-occ ListReplayer."
        ),
        "direct_action_contract": (
            "The neural model learns an unweighted zero/positive hurdle, an "
            "unbounded conditional Poisson excess count, and an autoregressive "
            "mixture over direct signed cache-line deltas. Stateless SHA-256 "
            "event-keyed inverse-CDF sampling supplies common random numbers "
            "across capacities and uses no selected threshold, fixed page-offset "
            "table, same-page rule, or Stride degree cap."
        ),
        "model_input_guardrail": {
            "normal_stride_private_state": [
                "PC-indexed tracker table", "confidence", "last stride"
            ],
            "direct_nn_inputs": [
                "lossless uint64 PC and 58-bit cache-line-number encodings",
                "causal address/PC history represented by the model itself",
            ],
            "not_nn_inputs": [
                "normal Stride candidates", "cycle", "hit/miss", "queue state",
                "stride tracker state", "accepted/duplicate", "future evaluation rows at inference",
            ],
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
                    "a contiguous causal sliding window; output t sees no "
                    "future input"
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
            "one_minus_selected_accuracy": (
                "Arithmetic complement of selected_accuracy for audit only; "
                "it is not a cache-pollution measurement"
            ),
        },
        "aggregate_score_used": False,
        "direct_harmful_eviction_pollution_measured": False,
        "input_provenance": {
            "current_input_dir": str(input_dir),
            "collection_manifest": collection_manifest,
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
                    "one_minus_selected_accuracy": row[
                        "one_minus_selected_accuracy"
                    ],
                    "prefetch_request_reduction_vs_offline_normal": (
                        row["prefetch_request_reduction_vs_offline_normal"]
                    ),
                })
            tracks[policy] = {
                "offline_normal_ipc": normal["ipc"],
                "models": points,
                "best_model_by_ipc": max(
                    points, key=lambda point: point["ipc"]
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
                "ipc_winner": (
                    "cnn" if cnn_row["ipc"] > lstm_row["ipc"] else "lstm"
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
                "improves IPC and miss rate: immediate "
                "and medium-range address correlation is sufficient."
            ),
            "lstm_wins": (
                "At paired reported capacities, the stateful LSTM improves IPC: "
                "useful information extends beyond the TCN receptive field."
            ),
            "both_fail": (
                "Both standalone students fail to discover enough useful "
                "direct actions; the restricted input representation or "
                "training target is the bottleneck."
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
