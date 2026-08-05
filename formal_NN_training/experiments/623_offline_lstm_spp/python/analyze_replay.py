#!/usr/bin/env python3
"""Fail-closed analysis for the standalone 623 SPP-interface LSTM track."""
import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from model_contract import (
    EXPERIMENT_REVISION, EXTERNAL_INPUT_FIELDS, MODEL_REVISION as V20_MODEL_REVISION,
    POLICY, TRACE, describe_model_points, expected_parameter_count,
)

POLICIES = (POLICY,)
V15_MODEL_REVISION = "compact_crn_joint_delta_fill_mixture_v15"
V16A_MODEL_REVISION = "compact_crn_joint_delta_fill_guard_map_v16a"
V17_MODEL_REVISION = "compact_crn_factorized_delta_keyed_fill_v17"
V18_MODEL_REVISION = "compact_crn_hard_distinct_delta_keyed_fill_v18"
V19_MODEL_REVISION = "routed_page_lstm_rank_grammar_leb128_v19"
MODEL_REVISION = V20_MODEL_REVISION
TRACK_MODEL_FAMILY = "lstm"
V20_POINT_CONTRACT = describe_model_points()
# Retain enough immutable v19 metadata to diagnose archived runs after the
# active contract module moved to v20.
V19_POINT_CONTRACT = {
    "run_id": "623_offline_lstm_spp_routed_grammar_v19_seed7",
    "operation": "train-v19",
    "experiment_revision": EXPERIMENT_REVISION,
    "model_revision": V19_MODEL_REVISION,
    "decoder_revision": "keyed_stop_emit_zigzag_leb128_target_fill_v19",
    "runtime_feature_count": 59,
    "action_rollout_watchdog_ranks": 52,
    "points": [
        {"size": 8, "pair_id": "p0", "tag": "routed_grammar_spp_lstm_h8", "parameter_count": 2790},
        {"size": 16, "pair_id": "p1", "tag": "routed_grammar_spp_lstm_h16", "parameter_count": 7032},
    ],
}
DEFAULT_MODEL_TAGS = ",".join(
    point["tag"] for point in V20_POINT_CONTRACT["points"]
)
MODEL_TAG_PREFIXES = (
    "joint_delta_fill_spp_lstm_", "guard_joint_map_spp_lstm_",
    "factorized_delta_fill_spp_lstm_",
    "hard_distinct_delta_fill_spp_lstm_",
    "routed_grammar_spp_lstm_",
    "independent_vocab_spp_lstm_",
)
EXPECTED_POINTS = {
    ("lstm", 8): "p0",
    ("lstm", 16): "p1",
    ("lstm", 32): "p2",
    ("lstm", 64): "p3",
    ("lstm", 128): "p4",
}
EXPECTED_PARAMETERS_BY_REVISION = {
    V15_MODEL_REVISION: {
        8: 2682, 16: 6242, 32: 16050, 64: 46418, 128: 150162,
    },
    V16A_MODEL_REVISION: {
        8: 2682, 16: 6242, 32: 16050, 64: 46418, 128: 150162,
    },
    V17_MODEL_REVISION: {
        8: 2664, 16: 6208, 32: 15984, 64: 46288, 128: 149904,
    },
    V18_MODEL_REVISION: {
        8: 2664, 16: 6208, 32: 15984, 64: 46288, 128: 149904,
    },
}
EVENT_LOGGER_SCHEMA = "623_causal_trigger_fill_v6"
ACTION_ATTACHMENT_MODE = "explicit_trigger_event_id"
SOURCE_INPUTS = list(EXTERNAL_INPUT_FIELDS)
KV = re.compile(r"^([A-Za-z0-9_]+)\s+([-+0-9.eE]+)\s*$")
FINISHED = re.compile(
    r"Finished CPU\s+0\s+instructions:\s+(\d+)\s+cycles:\s+(\d+)\s+"
    r"cumulative IPC:\s+([-+0-9.eE]+)"
)
REPLAYER = re.compile(
    r"emitted\s+(\d+)\s+(?:candidates|actions) over\s+(\d+)\s+runtime ROI L2 LOAD "
    r"accesses \((\d+)\s+matched PC-line-occ triggers;\s+"
    r"(\d+)\s+loaded trigger keys"
)
REPLAYER_LOADED = re.compile(
    r"loaded\s+(\d+)\s+direct actions across\s+(\d+)\s+"
    r"PC-line-occ triggers"
)
PYTHON_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]


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
    trigger_entry_counts = defaultdict(int)
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
            if (
                min(pc, line, occurrence, address) < 0
                or pc > (1 << 64) - 1
                or line > (1 << 58) - 1
                or address > (1 << 64) - 64
                or address % 64
            ):
                raise RuntimeError(
                    "out-of-range/unaligned SPP replay row {}".format(
                        line_number
                    )
                )
            if fill_level not in (2, 4):
                raise RuntimeError(
                    "invalid SPP fill level at row {}".format(line_number)
                )
            fill_counts[
                "FILL_L2" if fill_level == 2 else "FILL_LLC"
            ] += 1
            trigger = (pc, line, occurrence)
            trigger_entry_counts[trigger] += 1
            count += 1
    if count <= 0 and not allow_empty:
        raise RuntimeError("empty SPP replay list")
    return {
        "entries": count,
        "unique_triggers": len(trigger_entry_counts),
        "trigger_entry_counts": dict(trigger_entry_counts),
        "sha256": sha256(path),
        "fill_counts": fill_counts,
    }


def parse_log(path):
    stats = {}
    text = path.read_text(errors="ignore")
    emitted = callbacks = matched = 0
    loaded_entries = loaded_triggers = dumped_loaded_triggers = 0
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
            dumped_loaded_triggers = int(match.group(4))
        match = REPLAYER_LOADED.search(raw)
        if match:
            loaded_entries = int(match.group(1))
            loaded_triggers = int(match.group(2))

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
        "loaded_entries": loaded_entries,
        "loaded_triggers": loaded_triggers,
        "dumped_loaded_triggers": dumped_loaded_triggers,
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
        "cache_fill_feedback_events": 0,
    }
    last_event_id = -1
    latest_demand_event_id = None
    latest_demand_identity = None
    runtime_occurrences = defaultdict(int)
    runtime_trigger_keys = set()
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
                raise RuntimeError("unknown 623 event kind {}".format(row["event"]))
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
            pair = (int(row["ip"]), int(row["line"]))
            occurrence = runtime_occurrences[pair]
            runtime_occurrences[pair] += 1
            runtime_trigger_keys.add((pair[0], pair[1], occurrence))
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
    result["_runtime_trigger_keys"] = runtime_trigger_keys
    return result


def policy_for_method(method):
    if method == "offline_" + POLICY or method.startswith(
        "offline_joint_delta_fill_spp_"
    ) or method.startswith("offline_guard_joint_map_spp_") or method.startswith(
        "offline_factorized_delta_fill_spp_"
    ) or method.startswith(
        "offline_hard_distinct_delta_fill_spp_"
    ) or method.startswith(
        "offline_routed_grammar_spp_"
    ) or method.startswith(
        "offline_independent_vocab_spp_"
    ):
        return POLICY
    return ""


def model_tag_for_method(method):
    if method.startswith(
        "offline_joint_delta_fill_spp_"
    ) or method.startswith("offline_guard_joint_map_spp_") or method.startswith(
        "offline_factorized_delta_fill_spp_"
    ) or method.startswith(
        "offline_hard_distinct_delta_fill_spp_"
    ) or method.startswith(
        "offline_routed_grammar_spp_"
    ) or method.startswith(
        "offline_independent_vocab_spp_"
    ):
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


def validate_v20_metadata(metadata, tag, inputs, source_contract_hash, failures):
    expected = {
        "run_id": V20_POINT_CONTRACT["run_id"], "trace": TRACE,
        "model_tag": tag, "matched_normal_prefetcher": POLICY,
        "model_family": "lstm", "track_model_family": "lstm",
        "operation": V20_POINT_CONTRACT["operation"],
        "experiment_revision": V20_POINT_CONTRACT["experiment_revision"],
        "model_revision": V20_POINT_CONTRACT["model_revision"],
        "decoder_revision": V20_POINT_CONTRACT["decoder_revision"],
        "neural_role": "standalone_direct_action_prefetcher",
        "source_decision_effective_external_input": SOURCE_INPUTS,
        "training_runtime_fields": SOURCE_INPUTS,
        "inference_runtime_fields": SOURCE_INPUTS,
        "runtime_feature_count": V20_POINT_CONTRACT["runtime_feature_count"],
        "same_external_input_contract": True,
        "training_inference_input_encoder_identical": True,
        "decoder_training_mode": "fully_supervised_independent_ranks_no_action_feedback",
        "decoder_previous_teacher_action_used_as_input": False,
        "teacher_action_values_used_as_decoder_feedback": False,
        "sampled_outputs_used_as_decoder_feedback": False,
        "model_does_not_use_pc": True, "pc_is_replay_transport_only": True,
        "model_input_is_causal_external_event_sequence_only": True,
        "cache_fill_feedback_used_as_raw_external_input": True,
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "normal_policy_request_rate_used_as_budget": False,
        "normal_policy_constants_used_by_neural_inference": False,
        "probability_threshold_used": False,
        "inference_policy_hardcodes_used": False, "neural_degree_cap": None,
        "fixed_page_offset_classes": None,
        "same_page_rule_used_by_neural_inference": False,
        "delta_other_escape": V20_POINT_CONTRACT["delta_other_escape"],
        "delta_other_decode_precision": V20_POINT_CONTRACT[
            "delta_other_decode_precision"
        ],
        "full_signed_line_delta_range_reachable": False,
        "every_signed_line_delta_exactly_representable": False,
        "exact_delta_representability_scope": "train_vocabulary_only",
        "request_count_sampling_performed": False,
        "fill_conditioned_on_actual_emitted_target": True,
        "fill_argmax_used": False, "global_chronological_lstm": True,
        "routed_demand_fill_recurrent_paths": False,
        "page_local_causal_state": False,
        "common_random_numbers_across_capacities": True,
        "strict_common_random_numbers_across_capacities": True,
        "decoder_train_sampling_performed": False,
        "decoder_guard_sampling_performed": True,
        "decoder_eval_sampling_performed": True,
        "guard_selected_checkpoint": True,
        "evaluation_used_for_selection": False, "evaluation_decode_count": 1,
        "output_materialization_watchdog_is_neural_degree_cap": False,
        "weights_retrained": True, "checkpoint_reused": False,
        "same_source_input_offline_claim_allowed": True,
        "closed_loop_live_claim_allowed": False,
        "keyed_sampling_self_test": "PASS",
        "integer_csv_exactness_self_test": "PASS",
        "cublas_workspace_config": ":4096:8",
        "torch_deterministic_algorithms_enabled": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "float32_matmul_precision": "highest",
        "determinism_fail_closed": True,
        "training_state_mode": "chronological_stateful_tbptt",
        "training_state_carried_across_chunks": True,
        "training_state_detached_between_chunks": True,
        "inference_history_mode": "fresh_state_then_complete_train_guard_eval_chronology",
        "collection_manifest_role": "historical_input_package_provenance_only",
        "collection_manifest_decoder_fields_are_current_contract": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            failures.append("{} metadata {}={!r}; expected {!r}".format(
                tag, key, metadata.get(key), value))
    for key, value in V20_POINT_CONTRACT["training_config"].items():
        if metadata.get(key) != value:
            failures.append("{} pinned {}={!r}; expected {!r}".format(
                tag, key, metadata.get(key), value))
    if metadata.get("training_config") != V20_POINT_CONTRACT["training_config"]:
        failures.append("{} pinned training_config mismatch".format(tag))
    if "A100" not in str(metadata.get("cuda_device_name", "")):
        failures.append("{} was not trained on the pinned A100 accelerator".format(tag))
    provenance_paths = {
        "trainer_source_sha256": PYTHON_DIR / "train_and_offline_infer.py",
        "model_contract_source_sha256": PYTHON_DIR / "model_contract.py",
        "threshold_free_policy_source_sha256": (
            REPO_ROOT / "formal_NN_training" / "common" / "threshold_free_policy.py"
        ),
        "decoder_sampler_source_sha256": (
            REPO_ROOT / "formal_NN_training" / "common" / "keyed_sampling.py"
        ),
    }
    for key, path in provenance_paths.items():
        if metadata.get(key) != sha256(path):
            failures.append("{} {} current-repo byte hash mismatch".format(tag, key))
    points = {(item["size"], item["pair_id"], item["tag"]): item
              for item in V20_POINT_CONTRACT["points"]}
    point = (metadata.get("model_size"), metadata.get("architecture_pair_id"), tag)
    vocab = metadata.get("exact_delta_vocabulary_size")
    if point not in points or not isinstance(vocab, int) or not 0 < vocab <= 255:
        failures.append("{} invalid v20 architecture/vocabulary point".format(tag))
    elif metadata.get("parameter_count") != expected_parameter_count(point[0], vocab):
        failures.append("{} invalid dataset-dependent parameter count".format(tag))
    if metadata.get("model_point_contract") != V20_POINT_CONTRACT:
        failures.append("{} model-point contract differs from trainer source".format(tag))
    hashes = {metadata.get("runtime_encoder_sha256"),
              metadata.get("training_runtime_encoder_sha256"),
              metadata.get("inference_runtime_encoder_sha256")}
    if len(hashes) != 1 or re.fullmatch(r"[0-9a-f]{64}", next(iter(hashes), "")) is None:
        failures.append("{} train/inference encoder hash mismatch".format(tag))
    for key in (
        "decoder_sampler_source_sha256", "decoder_sampler_key_schedule_sha256",
        "decoder_guard_event_key_stream_sha256", "decoder_eval_event_key_stream_sha256",
        "decoder_guard_sampling_schedule_sha256", "decoder_eval_sampling_schedule_sha256",
        "decision_router_source_sha256", "train_decision_router_sha256",
        "guard_decision_router_sha256", "eval_decision_router_sha256",
        "model_checkpoint_sha256", "training_history_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(metadata.get(key, ""))) is None:
            failures.append("{} invalid {}".format(tag, key))
    if metadata.get("source_contract_sha256") != source_contract_hash:
        failures.append("{} SPP source-contract SHA256 mismatch".format(tag))
    behavior = metadata.get("heldout_behavior_metrics")
    for key in ("count_exact_match_rate", "target_precision", "target_recall",
                "target_f1", "joint_action_f1", "predicted_l2_fraction",
                "teacher_l2_fraction", "trigger_f1"):
        value = behavior.get(key) if isinstance(behavior, dict) else None
        if not isinstance(value, (int, float)) or value < 0 or value > 1:
            failures.append("{} invalid behavior metric {}".format(tag, key))
    diagnostics = metadata.get("action_output_diagnostics")
    if (
        not isinstance(diagnostics, dict)
        or diagnostics.get("duplicate_outputs_are_preserved_for_replay") is not True
        or metadata.get("offline_nn_entries") != metadata.get("materialized_action_count")
        or metadata.get("raw_predicted_action_count") != metadata.get("materialized_action_count")
    ):
        failures.append("{} invalid v20 action output accounting".format(tag))
    for role in ("train", "guard", "eval"):
        for kind in ("stream", "teacher_actions"):
            key = role + "_" + kind + "_content_sha256"
            expected_hash = inputs.get(POLICY, {}).get(role, {}).get(kind, {}).get("content_sha256")
            if metadata.get(key) != expected_hash:
                failures.append("{} {} {} content SHA256 mismatch".format(tag, role, kind))


def validate_v19_metadata(metadata, tag, inputs, source_contract_hash, failures):
    expected = {
        "run_id": V19_POINT_CONTRACT["run_id"],
        "trace": TRACE,
        "model_tag": tag,
        "matched_normal_prefetcher": POLICY,
        "model_family": "lstm",
        "track_model_family": "lstm",
        "operation": V19_POINT_CONTRACT["operation"],
        "experiment_revision": V19_POINT_CONTRACT["experiment_revision"],
        "model_revision": V19_POINT_CONTRACT["model_revision"],
        "decoder_revision": V19_POINT_CONTRACT["decoder_revision"],
        "neural_role": "standalone_direct_action_prefetcher",
        "source_decision_effective_external_input": SOURCE_INPUTS,
        "training_runtime_fields": SOURCE_INPUTS,
        "inference_runtime_fields": SOURCE_INPUTS,
        "runtime_feature_count": V19_POINT_CONTRACT["runtime_feature_count"],
        "same_external_input_contract": True,
        "training_inference_input_encoder_identical": True,
        "decoder_training_mode": "sampled_rank_grammar_rollout_with_separate_teacher_prefix_output_nll",
        "decoder_previous_teacher_action_used_as_input": True,
        "decoder_previous_teacher_action_used_as_input_scope": "isolated_loss_only_teacher_prefix_likelihood_branch",
        "decoder_previous_teacher_action_used_as_main_rollout_input": False,
        "teacher_count_role": "labels_STOP_or_EMIT_only_at_ranks_reached_by_sampled_rollout",
        "teacher_count_used_as_decoder_feedback": False,
        "teacher_prefix_role": "loss_only_exact_autoregressive_target_likelihood_branch",
        "teacher_prefix_advances_loss_only_likelihood_byte_state": True,
        "teacher_prefix_used_as_main_rollout_recurrent_feedback": False,
        "teacher_target_conditions_loss_only_fill_factor": True,
        "teacher_action_values_used_as_main_rollout_recurrent_feedback": False,
        "model_does_not_use_pc": True,
        "pc_is_replay_transport_only": True,
        "model_input_is_causal_external_event_sequence_only": True,
        "cache_fill_feedback_used_as_raw_external_input": True,
        "cache_fill_private_state_used_as_model_input": False,
        "teacher_actions_are_model_inputs": False,
        "teacher_actions_are_model_inputs_scope": "external_or_runtime_inference_inputs_only",
        "teacher_actions_used_as_supervised_output_conditioning": True,
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "normal_policy_request_rate_used_as_budget": False,
        "probability_threshold_used": False,
        "threshold_related_hardcodes_used": False,
        "neural_degree_cap": None,
        "handcrafted_semantic_features_used": True,
        "causal_derived_features_use_external_input_only": True,
        "external_input_fields_unchanged": True,
        "manual_loss_weights_used": False,
        "gate_training_objective": None,
        "gate_decoding_rule": None,
        "request_count_training_objective": "rankwise_unweighted_stop_emit_categorical_nll",
        "request_count_decoding_rule": "first_keyed_learned_STOP_token_ends_action_sequence",
        "request_count_sampling_performed": True,
        "stop_emit_sampling_rule": "event_rank_keyed_categorical_inverse_cdf",
        "stop_emit_sampler_representability_check": "STOP_mass_strictly_above_open_uniform_half_bin",
        "action_rollout_fail_closed_watchdog_ranks": V19_POINT_CONTRACT["action_rollout_watchdog_ranks"],
        "action_rollout_watchdog_role": "error_without_replay_not_truncation_or_forced_STOP",
        "action_rollout_watchdog_is_neural_degree_cap": False,
        "delta_mixture_components": 0,
        "delta_training_objective": "exact_autoregressive_teacher_prefix_canonical_leb128_nll_with_sampled_history_duplicate_support",
        "delta_decoding_rule": "keyed_exact_signed_zigzag_canonical_leb128",
        "delta_zero_allowed": True,
        "self_target_actions_allowed": True,
        "delta_legality_constraints": ["distinct_target_within_callback"],
        "delta_legality_fallback": None,
        "duplicate_target_handling": "mask_categorical_probability_and_renormalize",
        "duplicate_prefix_feasibility_mask_used": True,
        "fill_training_objective": "unweighted_two_class_cross_entropy_conditioned_on_teacher_target_loss_only",
        "fill_conditioned_on_actual_emitted_target": True,
        "fill_argmax_used": False,
        "optimizer_gradient_normalization": "total_categorical_atom_count_per_accumulation_group",
        "routed_demand_fill_recurrent_paths": True,
        "page_local_causal_state": True,
        "common_random_numbers_across_capacities": True,
        "strict_common_random_numbers_across_capacities": True,
        "cross_event_rng_state_used": False,
        "decoder_sampling_roles": ["train", "eval"],
        "decoder_train_sampling_performed": True,
        "decoder_guard_sampling_performed": False,
        "decoder_count_sampling_performed": True,
        "sampled_outputs_used_as_decoder_feedback": True,
        "decoder_previous_teacher_action_used_as_input": True,
        "weights_retrained": True,
        "checkpoint_reused": False,
        "guard_selected_decoder": False,
        "same_source_input_offline_claim_allowed": True,
        "closed_loop_live_claim_allowed": False,
        "keyed_sampling_self_test": "PASS",
        "rank_stop_emit_grammar_self_test": "PASS",
        "exact_leb128_codec_self_test": "PASS",
        "duplicate_prefix_no_dead_end_self_test": "PASS",
        "teacher_prefix_state_isolation_self_test": "PASS",
        "stop_sampler_representability_self_test": "PASS",
        "always_emit_watchdog_self_test": "PASS",
        "integer_csv_exactness_self_test": "PASS",
        "target_conditioned_fill_self_test": "PASS",
        "routed_page_state_self_test": "PASS",
        "training_state_mode": "chronological_stateful_tbptt",
        "training_state_carried_across_chunks": True,
        "training_state_detached_between_chunks": True,
        "inference_history_mode": "fresh_state_then_complete_train_guard_eval_chronology",
        "collection_manifest_role": "historical_input_package_provenance_only",
        "collection_manifest_decoder_fields_are_current_contract": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            failures.append(
                "{} metadata {}={!r}; expected {!r}".format(
                    tag, key, metadata.get(key), value
                )
            )
    expected_points = {
        (item["size"], item["pair_id"], item["tag"]): item["parameter_count"]
        for item in V19_POINT_CONTRACT["points"]
    }
    point = (
        metadata.get("model_size"), metadata.get("architecture_pair_id"), tag,
    )
    if point not in expected_points or metadata.get("parameter_count") != expected_points.get(point):
        failures.append("{} invalid v19 architecture/parameter point".format(tag))
    if metadata.get("model_point_contract") != V19_POINT_CONTRACT:
        failures.append("{} model-point contract differs from trainer source".format(tag))
    hashes = {
        metadata.get("runtime_encoder_sha256"),
        metadata.get("training_runtime_encoder_sha256"),
        metadata.get("inference_runtime_encoder_sha256"),
    }
    if len(hashes) != 1 or re.fullmatch(r"[0-9a-f]{64}", next(iter(hashes), "")) is None:
        failures.append("{} train/inference encoder hash mismatch".format(tag))
    for key in (
        "decoder_sampler_source_sha256", "decoder_sampler_key_schedule_sha256",
        "decoder_eval_event_key_stream_sha256", "decoder_eval_sampling_schedule_sha256",
        "decoder_train_sampling_schedule_sha256", "decision_router_source_sha256",
        "train_decision_router_sha256", "guard_decision_router_sha256",
        "eval_decision_router_sha256", "model_checkpoint_sha256",
        "training_history_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(metadata.get(key, ""))) is None:
            failures.append("{} invalid {}".format(tag, key))
    if metadata.get("source_contract_sha256") != source_contract_hash:
        failures.append("{} SPP source-contract SHA256 mismatch".format(tag))
    sampler = metadata.get("decoder_sampler")
    if (
        not isinstance(sampler, dict)
        or sampler.get("sampler_revision") != "sha256_event_keyed_inverse_cdf_crn_v1"
        or sampler.get("key_fields") != list(metadata.get("decoder_key_fields") or [])
        or sampler.get("cross_event_rng_state") is not False
    ):
        failures.append("{} keyed decoder sampler contract mismatch".format(tag))
    behavior = metadata.get("heldout_behavior_metrics")
    for key in (
        "count_exact_match_rate", "target_precision", "target_recall", "target_f1",
        "joint_action_f1", "l2_joint_f1", "predicted_l2_fraction",
        "teacher_l2_fraction", "trigger_f1",
    ):
        value = behavior.get(key) if isinstance(behavior, dict) else None
        if not isinstance(value, (int, float)) or value < 0 or value > 1:
            failures.append("{} invalid behavior metric {}".format(tag, key))
    legality = metadata.get("action_legality_diagnostics")
    if (
        not isinstance(legality, dict)
        or legality.get("duplicate_target_actions") != 0
        or legality.get("self_target_actions_allowed") is not True
        or legality.get("delta_legality_fallback") is not None
        or metadata.get("offline_nn_entries")
           != metadata.get("materialized_distinct_action_count")
        or metadata.get("raw_predicted_action_count")
           != metadata.get("materialized_distinct_action_count")
    ):
        failures.append("{} invalid v19 action legality audit".format(tag))
    if (
        not isinstance(metadata.get("peak_persistent_recurrent_state_bytes"), int)
        or metadata.get("peak_persistent_recurrent_state_bytes") <= 0
        or not isinstance(metadata.get("dynamic_page_state_pages"), int)
        or metadata.get("dynamic_page_state_pages") <= 0
    ):
        failures.append("{} invalid dynamic recurrent-state accounting".format(tag))
    for role in ("train", "guard", "eval"):
        for kind in ("stream", "teacher_actions"):
            key = role + "_" + kind + "_content_sha256"
            expected_hash = inputs.get(POLICY, {}).get(role, {}).get(kind, {}).get("content_sha256")
            if metadata.get(key) != expected_hash:
                failures.append("{} {} {} content SHA256 mismatch".format(tag, role, kind))


def validate_metadata(metadata, tag, inputs, source_contract_hash, failures):
    policy = POLICY
    revision = metadata.get("model_revision")
    if revision == V20_MODEL_REVISION:
        validate_v20_metadata(metadata, tag, inputs, source_contract_hash, failures)
        return
    if revision == V19_MODEL_REVISION:
        validate_v19_metadata(metadata, tag, inputs, source_contract_hash, failures)
        return
    is_v16a = revision == V16A_MODEL_REVISION
    is_v17 = revision == V17_MODEL_REVISION
    is_v18 = revision == V18_MODEL_REVISION
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
        "threshold_related_hardcodes_used": False,
        "neural_degree_cap": None,
        "fixed_page_offset_classes": None,
        "same_page_rule_used_by_neural_inference": False,
        "future_label_window_used": False,
        "fill_lead_cutoff_used": False,
        "handcrafted_semantic_features_used": False,
        "manual_loss_weights_used": False,
        "gate_class_weighting_used": False,
        "gate_training_objective": "unweighted_bernoulli_nll",
        "gate_decoding_rule": "event_keyed_bernoulli_inverse_cdf",
        "gate_operating_point_learned_from_empirical_prior": False,
        "request_count_training_objective": (
            "unweighted_bernoulli_hurdle_plus_positive_poisson_excess_nll"
        ),
        "request_count_decoding_rule": (
            "event_keyed_bernoulli_plus_common_quantile_poisson_inverse_cdf"
        ),
        "request_count_residual_scope": "none_event_local",
        "fill_training_objective": (
            "joint_with_delta_component_unweighted_mixture_nll"
        ),
        "fill_decoding_rule": "single_joint_delta_fill_pair_sample",
        "fill_argmax_used": False,
        "fill_probability_feedback_used": True,
        "decoder_probability_mass_carries_train_guard_history": False,
        "cross_event_probability_credit_used": False,
        "sampled_outputs_used_as_decoder_feedback": False,
        "stochastic_decoding_reproducible": True,
        "training_regularization_used": False,
        "inference_policy_hardcodes_used": False,
        "learned_request_count": True,
        "causal_no_future_self_test": "PASS",
        "keyed_sampling_self_test": "PASS",
        "event_local_hurdle_count_self_test": "PASS",
        "joint_delta_fill_sampling_self_test": "PASS",
        "delta_mixture_decoding_rule": (
            "single_joint_component_fill_sample_then_component_mean"
        ),
        "delta_decoder_feedback_rule": (
            "complete_joint_distribution_expectation_same_in_training_and_inference"
        ),
        "decoder_mixture_components": 4,
        "cnn_architecture_self_test": "NOT_APPLICABLE",
        "event_logger_schema": EVENT_LOGGER_SCHEMA,
        "action_attachment_mode": ACTION_ATTACHMENT_MODE,
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": revision,
        "neural_role": "standalone_direct_action_prefetcher",
        "replay_preserves_explicit_fill_level": True,
        "source_decision_effective_external_input": SOURCE_INPUTS,
        "runtime_feature_count": 59,
        "runtime_encoding": (
            "lossless 58-bit cache-line number plus one DEMAND/FILL kind bit"
        ),
        "runtime_address_alignment_bits_removed": 6,
        "runtime_address_alignment_bits_were_constant_zero": True,
        "joint_delta_fill_dependency_modeled": True,
        "joint_delta_fill_class_count": 8,
        "joint_pair_classes": 8,
        "joint_delta_fill_training_objective": (
            "unweighted_joint_delta_component_fill_mixture_nll"
        ),
        "joint_delta_fill_decoding_rule": (
            "event_keyed_mean_sorted_joint_pair_inverse_cdf"
        ),
        "joint_component_canonicalization": (
            "ascending_delta_mean_then_fill_label_then_original_component"
        ),
        "same_source_input_offline_claim_allowed": True,
        "closed_loop_live_claim_allowed": False,
        "common_random_numbers_across_capacities": True,
        "strict_common_random_numbers_across_capacities": True,
        "cross_event_rng_state_used": False,
        "train_guard_decoder_rng_burn_used": False,
        "decoder_sampling_roles": ["eval"],
        "decoder_train_sampling_performed": False,
        "decoder_guard_sampling_performed": False,
        "decoder_event_key_definition": (
            "zero_based_role_decision_idx_plus_source_line"
        ),
        "decoder_event_key_uses_teacher_information": False,
        "decoder_action_rank_origin": 0,
        "decoder_key_includes_sampler_revision": True,
        "track_model_family": TRACK_MODEL_FAMILY,
    }
    if is_v16a:
        selected_mode = metadata.get("selected_decoder_mode")
        common.update({
            "operation": "redecode-v16a",
            "model_revision": V16A_MODEL_REVISION,
            "decoder_revision": (
                "guard_selected_deterministic_joint_map_v16a"
            ),
            "decoder_candidate_modes": [
                "joint_class_map", "component_peak_map"
            ],
            "selected_decoder_mode": selected_mode,
            "decoder_sampling_roles": ["guard", "eval"],
            "decoder_guard_sampling_performed": True,
            "fill_decoding_rule": "guard_selected_joint_pair_map",
            "joint_delta_fill_decoding_rule": selected_mode,
            "delta_mixture_decoding_rule": (
                "guard_selected_joint_component_then_component_mean"
            ),
            "weights_retrained": False,
            "checkpoint_reused": True,
            "decoder_only_change": True,
            "strict_checkpoint_validation_passed": True,
            "model_architecture_reused_unchanged": True,
            "parent_model_revision": V15_MODEL_REVISION,
            "weights_model_revision": V15_MODEL_REVISION,
            "parent_run_id": (
                "623_offline_lstm_spp_keyed_crn_joint_fill_v15_seed7"
            ),
            "parent_state_dict_strict_load": True,
            "parent_input_hash_validation": "PASS",
            "parent_encoder_hash_validation": "PASS",
            "parent_normal_replay_validation": "PASS",
            "guard_selection_uses_eval_labels": False,
            "guard_selection_uses_guard_labels_only": True,
            "guard_selected_decoder": True,
            "joint_map_used": True,
            "decoder_action_sampling_performed": False,
            "decoder_count_sampling_performed": True,
            "deterministic_joint_map_self_test": "PASS",
            "component_peak_map_exact_mixture_mode_claimed": False,
            "guard_selection_objective": [
                "maximize_joint_action_f1",
                "maximize_target_f1",
                "maximize_trigger_f1",
                "minimize_absolute_action_count_ratio_error",
                "maximize_l2_joint_f1",
                "minimize_absolute_l2_fraction_error",
                "canonical_mode_order",
            ],
        })
    elif is_v17:
        common.update({
            "operation": "train-v17",
            "decoder_revision": "factorized_delta_keyed_fill_v17",
            "decoder_candidate_modes": [],
            "selected_decoder_mode": (
                "deterministic_modal_delta_component_plus_keyed_fill_draw"
            ),
            "fill_training_objective": (
                "unweighted_two_class_cross_entropy"
            ),
            "fill_decoding_rule": (
                "event_keyed_categorical_inverse_cdf"
            ),
            "joint_delta_fill_sampling_self_test": "NOT_APPLICABLE",
            "factorized_delta_fill_sampling_self_test": "PASS",
            "delta_mixture_decoding_rule": (
                "deterministic_modal_component_then_component_mean"
            ),
            "delta_decoder_feedback_rule": (
                "factorized_distribution_expectation_same_in_training_and_inference"
            ),
            "joint_delta_fill_dependency_modeled": False,
            "joint_delta_fill_class_count": 0,
            "joint_pair_classes": 0,
            "joint_delta_fill_training_objective": None,
            "joint_delta_fill_decoding_rule": None,
            "joint_component_canonicalization": None,
            "delta_mixture_components": 4,
            "delta_training_objective": (
                "four_component_signed_log_delta_mixture_nll"
            ),
            "weights_retrained": True,
            "checkpoint_reused": False,
            "decoder_only_change": False,
            "guard_selected_decoder": False,
            "joint_map_used": False,
            "decoder_action_sampling_performed": True,
            "decoder_count_sampling_performed": True,
        })
    elif is_v18:
        common.pop("event_local_hurdle_count_self_test")
        common.update({
            "operation": "train-v18",
            "decoder_revision": "hard_distinct_delta_keyed_fill_v18",
            "decoder_candidate_modes": [],
            "selected_decoder_mode": (
                "scale_aware_hard_distinct_delta_plus_keyed_hard_fill"
            ),
            "decoder_training_mode": (
                "teacher_count_scheduled_loss_with_hard_self_action_feedback"
            ),
            "teacher_count_role": "schedules_loss_bearing_action_ranks_only",
            "teacher_count_used_as_decoder_feedback": False,
            "gate_decoding_rule": "deterministic_raw_logit_sign",
            "request_count_decoding_rule": (
                "deterministic_raw_hurdle_plus_rounded_conditional_excess_mean"
            ),
            "fill_training_objective": (
                "unweighted_two_class_cross_entropy"
            ),
            "fill_decoding_rule": "event_keyed_categorical_inverse_cdf",
            "fill_probability_feedback_used": False,
            "hard_fill_one_hot_feedback_used": True,
            "keyed_fill_uniform_dtype": "float64",
            "address_confidence_fill_heuristic_used": False,
            "joint_delta_fill_sampling_self_test": "NOT_APPLICABLE",
            "factorized_delta_fill_sampling_self_test": "PASS",
            "deterministic_hurdle_count_self_test": "PASS",
            "hard_distinct_action_feedback_self_test": "PASS",
            "delta_mixture_decoding_rule": (
                "component_peak_density_order_then_hard_quantized_legal_delta"
            ),
            "delta_decoder_feedback_rule": (
                "actual_hard_quantized_emitted_delta_with_straight_through_training"
            ),
            "fill_decoder_feedback_rule": (
                "actual_keyed_hard_fill_one_hot_with_straight_through_training"
            ),
            "straight_through_hard_action_feedback_used": True,
            "delta_component_order_score": (
                "log_mixture_mass_minus_log_scale"
            ),
            "delta_component_score_tie_break": (
                "ascending_component_index_stable"
            ),
            "delta_legality_constraints": [
                "nonzero_signed_delta", "distinct_target_within_callback",
            ],
            "delta_legality_fallback": (
                "nearest_signed_delta_only_if_all_component_means_are_illegal"
            ),
            "delta_legality_uses_teacher_or_private_state": False,
            "signed_delta_canonicalization": (
                "58_bit_modulo_with_positive_half_range_mapped_to_negative"
            ),
            "joint_delta_fill_dependency_modeled": False,
            "joint_delta_fill_class_count": 0,
            "joint_pair_classes": 0,
            "joint_delta_fill_training_objective": None,
            "joint_delta_fill_decoding_rule": None,
            "joint_component_canonicalization": None,
            "delta_mixture_components": 4,
            "delta_training_objective": (
                "four_component_signed_log_delta_mixture_nll"
            ),
            "weights_retrained": True,
            "checkpoint_reused": False,
            "decoder_only_change": False,
            "guard_selected_decoder": False,
            "joint_map_used": False,
            "decoder_sampling_roles": ["train", "eval"],
            "decoder_train_sampling_performed": True,
            "decoder_guard_sampling_performed": False,
            "decoder_action_sampling_performed": True,
            "decoder_count_sampling_performed": False,
            "sampled_outputs_used_as_decoder_feedback": True,
            "collection_manifest_role": (
                "historical_input_package_provenance_only"
            ),
            "collection_manifest_decoder_fields_are_current_contract": False,
        })
    if revision not in (
        V15_MODEL_REVISION, V16A_MODEL_REVISION, V17_MODEL_REVISION,
        V18_MODEL_REVISION,
    ):
        failures.append("{} unsupported model revision {!r}".format(
            tag, revision
        ))
    expected_prefix = (
        "guard_joint_map_spp_lstm_" if is_v16a else
        "hard_distinct_delta_fill_spp_lstm_" if is_v18 else
        "factorized_delta_fill_spp_lstm_" if is_v17 else
        "joint_delta_fill_spp_lstm_"
    )
    if not tag.startswith(expected_prefix):
        failures.append("{} tag/revision prefix mismatch".format(tag))
    if is_v16a and metadata.get("selected_decoder_mode") not in (
        "joint_class_map", "component_peak_map"
    ):
        failures.append("{} invalid selected v16A decoder mode".format(tag))
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
        or metadata.get("decoder_sampler_key_fields") != expected_key_fields
    ):
        failures.append("{} keyed decoder sampler contract mismatch".format(tag))
    hash_keys = [
        "decoder_sampler_source_sha256",
        "decoder_sampler_key_schedule_sha256",
        "decoder_eval_event_key_stream_sha256",
        "decoder_eval_sampling_schedule_sha256",
        "decision_router_source_sha256",
        "train_decision_router_sha256",
        "guard_decision_router_sha256",
        "eval_decision_router_sha256",
    ]
    if is_v16a:
        hash_keys.extend([
            "decoder_guard_sampling_schedule_sha256",
            "parent_checkpoint_sha256", "parent_run_metadata_sha256",
            "parent_training_history_sha256", "model_checkpoint_sha256",
            "training_history_sha256",
        ])
    elif is_v17 or is_v18:
        hash_keys.extend([
            "model_checkpoint_sha256", "training_history_sha256",
        ])
    if is_v18:
        hash_keys.append("decoder_train_sampling_schedule_sha256")
    for key in hash_keys:
        value = metadata.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            failures.append("{} invalid {}".format(tag, key))
    if (
        not isinstance(metadata.get("peak_persistent_recurrent_state_bytes"), int)
        or metadata.get("peak_persistent_recurrent_state_bytes") <= 0
    ):
        failures.append("{} invalid recurrent-state byte count".format(tag))
    if metadata.get("source_contract_sha256") != source_contract_hash:
        failures.append("{} SPP source-contract SHA256 mismatch".format(tag))
    behavior = metadata.get("heldout_behavior_metrics")
    required_behavior = (
        "count_exact_match_rate", "target_precision", "target_recall",
        "target_f1",
    )
    if is_v16a or is_v17 or is_v18:
        required_behavior += (
            "joint_action_f1", "l2_joint_f1", "predicted_l2_fraction",
            "teacher_l2_fraction", "trigger_f1",
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
        action_ratio = behavior.get("predicted_to_normal_action_ratio")
        if (is_v16a or is_v17 or is_v18) and (
            not isinstance(action_ratio, (int, float)) or action_ratio < 0
        ):
            failures.append("{} invalid action-count ratio".format(tag))
    if is_v18:
        legality = metadata.get("action_legality_diagnostics")
        if (
            not isinstance(legality, dict)
            or legality.get("self_target_actions") != 0
            or legality.get("duplicate_target_actions") != 0
            or metadata.get("raw_predicted_action_count")
            != legality.get("raw_predicted_action_count")
            or metadata.get("materialized_distinct_action_count")
            != legality.get("materialized_distinct_action_count")
            or metadata.get("offline_nn_entries")
            != metadata.get("materialized_distinct_action_count")
        ):
            failures.append("{} invalid hard-action legality audit".format(tag))
    if is_v16a:
        selection = metadata.get("guard_decoder_selection")
        if not isinstance(selection, dict) or set(selection) != {
            "joint_class_map", "component_peak_map"
        }:
            failures.append("{} invalid guard decoder audit".format(tag))
        else:
            valid_selection = True
            for mode, payload in selection.items():
                if (
                    not isinstance(payload, dict)
                    or not isinstance(payload.get("metrics"), dict)
                    or not isinstance(payload.get("selection_key"), list)
                    or len(payload.get("selection_key", [])) != 7
                ):
                    valid_selection = False
                    failures.append(
                        "{} invalid guard audit for {}".format(tag, mode)
                    )
            if valid_selection:
                chosen = max(
                    selection,
                    key=lambda mode: selection[mode]["selection_key"],
                )
                if chosen != metadata.get("selected_decoder_mode"):
                    failures.append(
                        "{} guard decoder selection was not maximal".format(tag)
                    )
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
        expected_parameters = EXPECTED_PARAMETERS_BY_REVISION.get(
            revision, {}
        ).get(metadata.get("model_size"))
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
        if not tag.startswith(MODEL_TAG_PREFIXES):
            raise SystemExit("invalid model tag {}".format(tag))
    revisions = {
        "v20" if tag.startswith("independent_vocab_spp_lstm_") else
        "v19" if tag.startswith("routed_grammar_spp_lstm_") else
        "v18" if tag.startswith("hard_distinct_delta_fill_spp_lstm_") else
        "v16a" if tag.startswith("guard_joint_map_spp_lstm_") else
        "v17" if tag.startswith("factorized_delta_fill_spp_lstm_") else
        "v15"
        for tag in model_tags
    }
    if len(revisions) != 1:
        raise SystemExit("do not mix SPP model revisions in one matched run")

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
    runtime_trigger_keys_by_method = {}

    for method in methods:
        log_path = logs / (TRACE + "." + method + ".log")
        event_path = events / (TRACE + "." + method + ".events.csv.gz")
        if not log_path.is_file():
            failures.append("missing log {}".format(log_path))
            continue
        if not event_path.is_file():
            failures.append("missing event log {}".format(event_path))
            continue
        if method == "offline_spp" or method.startswith(
            "offline_joint_delta_fill_spp_"
        ) or method.startswith("offline_guard_joint_map_spp_") or method.startswith(
            "offline_factorized_delta_fill_spp_"
        ) or method.startswith(
            "offline_hard_distinct_delta_fill_spp_"
        ) or method.startswith(
            "offline_routed_grammar_spp_"
        ) or method.startswith(
            "offline_independent_vocab_spp_"
        ):
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
            event_metrics = parse_events(event_path)
            runtime_trigger_keys_by_method[method] = event_metrics.pop(
                "_runtime_trigger_keys"
            )
            row.update(event_metrics)
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
        validate_metadata(
            metadata, tag, input_info, source_contract_hash, failures
        )
        if metadata.get("model_revision") in (
            V17_MODEL_REVISION, V18_MODEL_REVISION, V19_MODEL_REVISION,
            V20_MODEL_REVISION,
        ):
            for name, key in (
                ("model.pt", "model_checkpoint_sha256"),
                ("training_history.csv", "training_history_sha256"),
            ):
                artifact = colab_root / tag / name
                observed = sha256(artifact) if artifact.is_file() else None
                if metadata.get(key) != observed:
                    failures.append(
                        "{} {} byte hash mismatch".format(tag, name)
                    )
        if metadata.get("model_revision") == V16A_MODEL_REVISION:
            for name, key in (
                ("model.pt", "model_checkpoint_sha256"),
                ("training_history.csv", "training_history_sha256"),
            ):
                artifact = colab_root / tag / name
                observed = sha256(artifact) if artifact.is_file() else None
                if metadata.get(key) != observed:
                    failures.append("{} {} byte hash mismatch".format(tag, name))
            if metadata.get("model_checkpoint_sha256") != metadata.get(
                "parent_checkpoint_sha256"
            ):
                failures.append("{} did not reuse parent checkpoint bytes".format(tag))
            if metadata.get("training_history_sha256") != metadata.get(
                "parent_training_history_sha256"
            ):
                failures.append("{} did not reuse parent history bytes".format(tag))
            parent_tag = "joint_delta_fill_spp_lstm_h{}".format(
                metadata.get("model_size")
            )
            parent_dir = (
                args.run_dir.parent
                / metadata.get("parent_run_id", "")
                / "colab_output" / parent_tag
            )
            for name, key in (
                ("model.pt", "parent_checkpoint_sha256"),
                ("run_metadata.json", "parent_run_metadata_sha256"),
                ("training_history.csv", "parent_training_history_sha256"),
            ):
                artifact = parent_dir / name
                observed = sha256(artifact) if artifact.is_file() else None
                if metadata.get(key) != observed:
                    failures.append(
                        "{} canonical parent {} hash mismatch".format(
                            tag, name
                        )
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
                replay_lists.setdefault("offline_" + policy, normal_info)
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
                replay_lists["offline_" + tag] = nn_info
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

    rows_by_method = {row["method"]: row for row in rows}
    for method in ["offline_" + POLICY] + [
        "offline_" + tag for tag in model_tags
    ]:
        row = rows_by_method.get(method)
        info = replay_lists.get(method)
        if row is None or info is None:
            failures.append("{} lacks replay accounting inputs".format(method))
            continue
        runtime_keys = runtime_trigger_keys_by_method.get(method)
        if runtime_keys is None:
            failures.append("{} lacks runtime trigger keys".format(method))
            continue
        trigger_entry_counts = info["trigger_entry_counts"]
        reachable_keys = set(trigger_entry_counts).intersection(runtime_keys)
        reachable_entries = sum(
            trigger_entry_counts[key] for key in reachable_keys
        )
        reachable_triggers = len(reachable_keys)
        row["replay_list_entries"] = info["entries"]
        row["replay_list_triggers"] = info["unique_triggers"]
        row["runtime_reachable_list_entries"] = reachable_entries
        row["runtime_reachable_list_triggers"] = reachable_triggers
        row["runtime_unreached_list_entries"] = (
            info["entries"] - reachable_entries
        )
        row["runtime_unreached_list_triggers"] = (
            info["unique_triggers"] - reachable_triggers
        )
        # A replay list is keyed to the source-evaluation callback domain.
        # A separate intervention run can observe a different set of runtime
        # PC-line-occ keys.  Validate the loaded table against the complete
        # list and the emitted/requested counts against the exact intersection.
        checks = {
            "list entries versus replayer loaded entries": (
                info["entries"], row["loaded_entries"]
            ),
            "unique list triggers versus replayer loaded triggers": (
                info["unique_triggers"], row["loaded_triggers"]
            ),
            "replayer loaded triggers at startup versus final dump": (
                row["loaded_triggers"], row["dumped_loaded_triggers"]
            ),
            "runtime-reachable list entries versus replayer emitted": (
                reachable_entries, row["emitted"]
            ),
            "runtime-reachable list triggers versus replayer matched": (
                reachable_triggers, row["matched"]
            ),
            "replayer emitted versus simulator requested": (
                row["emitted"], row["pf_requested"]
            ),
            # Emitted and pf_requested count attempted actions for triggers
            # actually reached in this replay.  PF rows are logged only after
            # CACHE::prefetch_line passes its PQ-capacity gate, so they
            # correspond to pf_issued rather than all attempts.
            "simulator issued versus logged PF events": (
                row["pf_issued"], row["prefetch_request_events"]
            ),
            "simulator request conservation": (
                row["pf_requested"],
                row["pf_issued"] + row["pf_dropped"],
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
        "prefetch_fill_llc_events", "cache_fill_feedback_events",
        "event_selected_accuracy_proxy",
        "event_coverage_vs_no_pref_l2_miss", "event_timeliness_proxy",
        "replay_list_entries", "replay_list_triggers",
        "runtime_reachable_list_entries", "runtime_reachable_list_triggers",
        "runtime_unreached_list_entries", "runtime_unreached_list_triggers",
        "loaded_entries", "loaded_triggers", "dumped_loaded_triggers",
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
        "timeliness_delta_vs_offline_normal", "one_minus_selected_accuracy",
        "one_minus_selected_accuracy_delta_vs_offline_normal", "pf_requested",
        "prefetch_request_reduction_vs_offline_normal",
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

    # trigger_entry_counts is an internal exact-accounting index keyed by
    # (pc, line, occurrence) tuples.  Keep it in memory for the checks above,
    # but do not place tuple-keyed state in the JSON evidence payload.
    replay_accounting = {
        method: {
            "entries": info["entries"],
            "unique_triggers": info["unique_triggers"],
            "sha256": info["sha256"],
            "fill_counts": info["fill_counts"],
        }
        for method, info in sorted(replay_lists.items())
    }

    status = "FAIL" if failures else "PASS"
    payload = {
        "status": status,
        "fair_comparison_claim_allowed": not failures,
        "same_source_input_offline_claim_allowed": not failures,
        "closed_loop_live_claim_allowed": False,
        "trace": TRACE,
        "model_family_track": TRACK_MODEL_FAMILY,
        "model_revision": (
            V20_MODEL_REVISION if "v20" in revisions
            else V19_MODEL_REVISION if "v19" in revisions
            else V18_MODEL_REVISION if "v18" in revisions
            else V16A_MODEL_REVISION if "v16a" in revisions
            else V17_MODEL_REVISION if "v17" in revisions
            else V15_MODEL_REVISION
        ),
        "trace_selection": {
            "reason": (
                "SPP is evaluated in its own matched track because its "
                "signature/pattern state and fill actions differ from stride."
            ),
            "historical_ipc_context": {
                "no_pref": 0.35321,
                POLICY: 0.35391,
            },
        },
        "primary_comparisons": {
            "spp_track": [
                "offline_spp",
                (
                    "offline_independent_vocab_spp_{}_<capacity>"
                    if "v20" in revisions else
                    "offline_routed_grammar_spp_{}_<capacity>"
                    if "v19" in revisions else
                    "offline_hard_distinct_delta_fill_spp_{}_<capacity>"
                    if "v18" in revisions else
                    "offline_guard_joint_map_spp_{}_<capacity>"
                    if "v16a" in revisions else
                    "offline_factorized_delta_fill_spp_{}_<capacity>"
                    if "v17" in revisions else
                    "offline_joint_delta_fill_spp_{}_<capacity>"
                ).format(TRACK_MODEL_FAMILY),
            ],
        },
        "context_reference_only": [
            "no_pref", "live_spp_reference"
        ],
        "transport_fidelity": transport_fidelity,
        "replay_accounting": replay_accounting,
        "warnings": warnings,
        "track_guardrail": (
            "Every neural point learns independent rank-conditioned direct signed "
            "cache-line deltas and target-conditioned fill from the same chronological DEMAND(addr) and "
            "CACHE_FILL(evicted_addr) external callback stream. "
            "No page rule, normal-policy template, or captured SPP action is an inference input."
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
            "count_distribution": (
                "natural-prior gate MAP plus rounded positive log-count"
                if "v20" in revisions else
                "rank-wise learned STOP/EMIT categorical grammar"
                if "v19" in revisions else
                "deterministic raw hurdle plus rounded conditional excess mean"
                if "v18" in revisions else
                "unweighted Bernoulli hurdle plus conditional Poisson excess "
                "with non-negative unbounded support"
            ),
            "target_and_fill_distribution": (
                "TRAIN-derived exact signed-delta vocabulary with rounded "
                "bounded approximate signed-log OTHER; prior-corrected target-conditioned fill"
                if "v20" in revisions else
                "exact signed ZigZag/canonical-LEB128 increment followed by "
                "fill conditioned on the actual emitted target"
                if "v19" in revisions else
                "factorized four-component signed-delta mixture and "
                "two-class fill head" if (
                    "v17" in revisions or "v18" in revisions
                ) else
                "joint four-component signed-delta by two-fill-class mixture"
            ),
            "fill_classes": ["FILL_L2", "FILL_LLC"],
            "decision": (
                (
                    "deterministic gate/count/delta and fill-only stateless "
                    "event/rank-keyed categorical draw"
                    if "v20" in revisions else
                    "stateless event/rank-keyed learned STOP/EMIT, exact "
                    "LEB128 increment, and target-conditioned fill"
                    if "v19" in revisions else
                    "deterministic raw count, scale-aware hard quantized "
                    "distinct delta, and event-keyed hard fill draw"
                    if "v18" in revisions else
                    "stateless event-keyed Bernoulli/Poisson inverse-CDF count and "
                )
                + (
                    ""
                    if ("v18" in revisions or "v19" in revisions or "v20" in revisions) else
                    "guard-selected deterministic joint delta-component/fill MAP"
                    if "v16a" in revisions else
                    "deterministic modal delta-component mean plus one "
                    "event-keyed fill-class draw"
                    if "v17" in revisions else
                    "one mean-canonicalized joint delta-component/fill sample"
                )
            ),
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
                "lossless 58-bit callback cache-line-number encoding",
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
            "claim_boundary": (
                "Recorded fill callbacks came from the source SPP run; this is "
                "a matched-input open-loop offline comparison, not closed-loop "
                "live neural execution."
            ),
        },
        "architecture_contract": (
            {
                "name": "one bounded global chronological LSTM",
                "history": "complete train then guard then evaluation chronology",
                "training": "chronological TBPTT with state carried and detached",
                "causal_derived_features": [],
                "decoder": "independent rank code, TRAIN delta vocabulary plus OTHER, target-conditioned fill",
                "normal_policy_template": False,
            }
            if "v20" in revisions else
            {
                "name": "routed demand/fill plus page-local stateful LSTM",
                "history": "complete train then guard then evaluation chronology",
                "training": "chronological TBPTT with routed/page state carried and detached",
                "causal_derived_features": [
                    "raw line-derived page key", "log1p page reuse age",
                ],
                "decoder": "rank STOP/EMIT, exact ZigZag/LEB128 increment, then target-conditioned fill",
                "dynamic_state_reported_separately": True,
            }
            if "v19" in revisions else
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
            "collection_manifest_role": (
                "historical_input_package_provenance_only"
            ),
            "collection_manifest_decoder_fields_are_current_contract": False,
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
                if policy_for_method("offline_" + tag) != policy:
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
                "improves IPC and miss rate: immediate "
                "and medium-range address correlation is sufficient."
            ),
            "lstm_wins": (
                "At paired reported capacities, the stateful LSTM improves IPC: "
                "useful information extends beyond the TCN receptive field."
            ),
            "both_fail": (
                "Both direct students fail to discover enough useful direct "
                "actions: the restricted address representation or missing "
                "private SPP feedback is more important than architecture."
            ),
        }

    out_json = args.run_dir / "matched_comparison.json"
    tmp_json = out_json.with_name(out_json.name + ".tmp")
    tmp_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    tmp_json.replace(out_json)
    print("[{}] {}".format(status, out_json))
    if failures:
        raise SystemExit(" | ".join(failures))


if __name__ == "__main__":
    main()
