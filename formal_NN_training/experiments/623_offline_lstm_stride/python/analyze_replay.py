#!/usr/bin/env python3
"""Fail-closed analysis for the standalone 623 Stride/LSTM track."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from model_contract import (
    EXPERIMENT_REVISION, MODEL_REVISION as ACTIVE_MODEL_REVISION,
    MODEL_TAG_PREFIX, PARENT_INPUT_RUN_ID, POLICY, RUN_ID, TRACE,
    expected_parameter_count, model_points_description,
)

POLICIES = (POLICY,)
V15_MODEL_REVISION = "compact_pc_keyed_crn_event_sampled_mixture_v15"
V16_MODEL_REVISION = "compact_pc_keyed_balanced_deterministic_scalar_v16"
V17_MODEL_REVISION = "compact_pc_keyed_prior_corrected_hurdle_scalar_v17"
V18_MODEL_REVISION = "compact_pc_keyed_natural_hurdle_scalar_v18"
TRACK_MODEL_FAMILY = "lstm"
ACTIVE_CONTRACT = model_points_description()
ACTIVE_MODEL_TAGS = tuple(
    point["model_tag"] for point in ACTIVE_CONTRACT["points"]
)
DEFAULT_MODEL_TAGS = ",".join(ACTIVE_MODEL_TAGS)
SAFE_PATH_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
EXPECTED_POINTS = {
    ("lstm", 8): "p0",
    ("lstm", 16): "p1",
    ("lstm", 32): "p2",
    ("lstm", 64): "p3",
    ("lstm", 128): "p4",
}
EXPECTED_PARAMETERS_BY_REVISION = {
    V15_MODEL_REVISION: {
        8: 1923, 16: 5243, 32: 16107, 64: 54731, 128: 199563,
    },
    V16_MODEL_REVISION: {
        8: 1860, 16: 5124, 32: 15876, 64: 54276, 128: 198660,
    },
    V17_MODEL_REVISION: {
        8: 1860, 16: 5124, 32: 15876, 64: 54276, 128: 198660,
    },
    V18_MODEL_REVISION: {
        8: 1860, 16: 5124, 32: 15876, 64: 54276, 128: 198660,
    },
}
EVENT_LOGGER_SCHEMA = "623_causal_trigger_v5"
CANDIDATE_ATTACHMENT_MODE = "explicit_trigger_event_id"
KV = re.compile(r"^([A-Za-z0-9_]+)\s+([-+0-9.eE]+)\s*$")
FINISHED = re.compile(
    r"Finished CPU\s+0\s+instructions:\s+(\d+)\s+cycles:\s+(\d+)\s+"
    r"cumulative IPC:\s+([-+0-9.eE]+)"
)
REPLAYER = re.compile(
    r"emitted\s+(\d+)\s+candidates over\s+(\d+)\s+runtime ROI L2 LOAD "
    r"accesses \((\d+)\s+matched PC-line-occ triggers;\s+"
    r"(\d+)\s+loaded trigger keys"
)
REPLAYER_LOADED = re.compile(
    r"loaded\s+(\d+)\s+prefetch entries across\s+(\d+)\s+"
    r"PC-line-occ triggers"
)
REPLAYER_REJECTED = re.compile(
    r"rejected\s+(\d+)\s+malformed lines"
)


def div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def optional_div(numerator, denominator):
    """Return None when a rate has no defined denominator.

    Counts and coverage may correctly be zero; precision/timeliness-style
    rates with no selected/requested actions are N/A rather than numeric zero.
    CSV renders None as an empty cell and JSON renders it as null.
    """
    return float(numerator) / float(denominator) if denominator else None


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
    zero_target_entries = 0
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
            trigger_entry_counts[trigger] += 1
            zero_target_entries += int(address == 0)
            count += 1
    if count <= 0 and not allow_empty:
        raise RuntimeError("empty stride replay list")
    return {
        "entries": count,
        "unique_triggers": len(trigger_entry_counts),
        "trigger_entry_counts": dict(trigger_entry_counts),
        "zero_target_entries": zero_target_entries,
        "sha256": sha256(path),
    }


def parse_log(path):
    stats = {}
    text = path.read_text(errors="ignore")
    emitted = callbacks = matched = 0
    loaded_entries = loaded_triggers = dumped_loaded_triggers = 0
    rejected_entries = 0
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
            rejected = REPLAYER_REJECTED.search(raw)
            if rejected:
                rejected_entries = int(rejected.group(1))

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
        "rejected_entries": rejected_entries,
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
        "accuracy": optional_div(useful, issued),
        "selected_accuracy": optional_div(useful, nodup_issued),
        "timeliness": optional_div(useful, useful + late),
        "late_per_issued": optional_div(late, issued),
        "drop_rate": optional_div(dropped, requested),
        "useless_per_issued": optional_div(useless, issued),
        "one_minus_selected_accuracy": (
            max(0.0, 1.0 - float(useful) / nodup_issued)
            if nodup_issued else None
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
    runtime_occurrences = defaultdict(int)
    runtime_trigger_keys = set()
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
    result["event_selected_accuracy_proxy"] = optional_div(
        result["prefetch_useful_demand_hits"],
        result["prefetch_request_events"],
    )
    result["event_timeliness_proxy"] = optional_div(
        result["prefetch_useful_demand_hits"],
        result["prefetch_useful_demand_hits"]
        + result["prefetch_late_demand_misses"],
    )
    result["_runtime_trigger_keys"] = runtime_trigger_keys
    return result


def policy_for_method(method):
    normal = "offline_" + POLICY
    if (
        method == normal
        or method.startswith("offline_" + MODEL_TAG_PREFIX)
        or method.startswith("offline_rank_stop_emit_stride_lstm_")
        or method.startswith("offline_independent_delta_" + POLICY + "_")
        or method.startswith("offline_global_local_grammar_" + POLICY + "_")
    ):
        return POLICY
    return ""


def model_tag_for_method(method):
    if (
        method.startswith("offline_" + MODEL_TAG_PREFIX)
        or method.startswith("offline_rank_stop_emit_stride_lstm_")
        or method.startswith("offline_independent_delta_" + POLICY + "_")
        or method.startswith("offline_global_local_grammar_" + POLICY + "_")
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
                row["selected_accuracy"] - baseline["selected_accuracy"]
                if row["selected_accuracy"] is not None
                and baseline["selected_accuracy"] is not None else ""
            )
            row["coverage_delta_vs_offline_normal"] = (
                row["coverage_vs_no_pref_l2_miss"]
                - baseline["coverage_vs_no_pref_l2_miss"]
            )
            row["timeliness_delta_vs_offline_normal"] = (
                row["timeliness"] - baseline["timeliness"]
                if row["timeliness"] is not None
                and baseline["timeliness"] is not None else ""
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
                if row["one_minus_selected_accuracy"] is not None
                and baseline["one_minus_selected_accuracy"] is not None else ""
            )


def validate_active_metadata_v23_legacy(metadata, tag, inputs, failures):
    """Fail closed on the decoder-only v23 Stride ablation contract."""
    experiment = Path(__file__).resolve().parents[1]
    repository = Path(__file__).resolve().parents[4]
    source_hashes = {
        "trainer_source_sha256": sha256(
            experiment / "python" / "train_and_offline_infer.py"
        ),
        "redecoder_source_sha256": sha256(
            experiment / "python" / "redecode_prior_corrected.py"
        ),
        "model_contract_source_sha256": sha256(
            experiment / "python" / "model_contract.py"
        ),
        "threshold_free_policy_source_sha256": sha256(
            repository / "formal_NN_training" / "common"
            / "threshold_free_policy.py"
        ),
    }
    expected = {
        "run_id": RUN_ID,
        "operation": ACTIVE_CONTRACT["operation"],
        "trace": TRACE,
        "model_tag": tag,
        "matched_normal_prefetcher": POLICY,
        "neural_role": "standalone_direct_action_prefetcher",
        "model_family": "lstm",
        "track_model_family": TRACK_MODEL_FAMILY,
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": ACTIVE_MODEL_REVISION,
        "decoder_revision": ACTIVE_CONTRACT["decoder_revision"],
        "source_decision_effective_external_input": ["pc", "addr"],
        "same_external_input_contract": True,
        "training_inference_input_encoder_identical": True,
        "training_runtime_fields": ["pc", "addr"],
        "inference_runtime_fields": ["pc", "addr"],
        "runtime_feature_count": ACTIVE_CONTRACT["runtime_feature_count"],
        "raw_runtime_feature_count": ACTIVE_CONTRACT[
            "raw_runtime_feature_count"
        ],
        "causal_runtime_feature_count": 0,
        "runtime_encoding": "lossless_raw_pc64_plus_line58_only",
        "engineered_runtime_features": [],
        "raw_runtime_input_only": True,
        "derived_features_use_teacher_or_future": False,
        "teacher_actions_are_model_inputs": False,
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "normal_policy_request_rate_used_as_budget": False,
        "normal_policy_constants_used_by_neural_inference": False,
        "normal_policy_templates_used_by_neural_inference": False,
        "probability_threshold_used": False,
        "threshold_related_hardcodes_used": False,
        "inference_policy_hardcodes_used": False,
        "neural_degree_cap": None,
        "fixed_page_offset_classes": None,
        "same_page_rule_used_by_neural_inference": False,
        "future_label_window_used": False,
        "nn_generates_own_target_addresses": True,
        "all_deltas_relative_to_current_demand": True,
        "full_signed_line_delta_range_reachable": False,
        "every_signed_line_delta_exactly_representable": False,
        "exact_delta_representability_scope": "train_vocabulary_only",
        "decoder_training_mode": ACTIVE_CONTRACT[
            "decoder_training_mode"
        ],
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_previous_predicted_action_used_as_input": False,
        "decoder_previous_sampled_action_used_as_input": False,
        "all_teacher_ranks_supervised": True,
        "terminal_stop_supervised_for_every_teacher_sequence": False,
        "hurdle_training_objective": ACTIVE_CONTRACT[
            "hurdle_training_objective"
        ],
        "hurdle_classes": ["ZERO", "POSITIVE"],
        "hurdle_equal_aggregate_train_mass": True,
        "hurdle_bias_initialization": (
            "centered_log_effective_weighted_TRAIN_class_mass"
        ),
        "hurdle_decoding_rule": (
            "deterministic_prior_corrected_two_class_argmax"
        ),
        "hurdle_prior_correction_at_decode_used": True,
        "hurdle_prior_correction_rule": (
            "weighted_logits_minus_log_TRAIN_inverse_frequency_class_weight"
        ),
        "separate_global_gate_used": True,
        "separate_count_head_used": True,
        "log_count_used": True,
        "positive_count_training_objective": ACTIVE_CONTRACT[
            "positive_count_training_objective"
        ],
        "positive_log_count_bias_initialization": (
            "mean_log_positive_TRAIN_count"
        ),
        "positive_count_support": (
            "mathematically_unbounded_positive_integers"
        ),
        "positive_count_host_behavior": "fail_closed_no_clip_or_wrap",
        "positive_count_decoding_rule": "max_1_round_exp_log_count",
        "data_derived_hurdle_class_weights_used": True,
        "manual_loss_weights_used": False,
        "delta_training_objective": ACTIVE_CONTRACT[
            "delta_training_objective"
        ],
        "poisson_objective_used": False,
        "poisson_decoder_used": False,
        "gmm_objective_used": False,
        "gmm_decoder_used": False,
        "deterministic_decoding": True,
        "stochastic_decoding": False,
        "decoder_sampling_roles": [],
        "decoder_train_sampling_performed": False,
        "decoder_guard_sampling_performed": False,
        "decoder_eval_sampling_performed": False,
        "sampled_outputs_used_as_decoder_feedback": False,
        "decode_per_callback_resource_watchdog": ACTIVE_CONTRACT[
            "decode_per_callback_resource_watchdog"
        ],
        "decode_per_role_resource_watchdog": ACTIVE_CONTRACT[
            "decode_per_role_resource_watchdog"
        ],
        "decode_resource_watchdog_behavior": ACTIVE_CONTRACT[
            "decode_resource_watchdog_behavior"
        ],
        "decode_resource_watchdog_is_neural_degree_cap": False,
        "successful_run_hit_decode_resource_watchdog": False,
        "maximum_host_action_count": ACTIVE_CONTRACT[
            "maximum_host_action_count"
        ],
        "checkpoint_selection": ACTIVE_CONTRACT[
            "checkpoint_selection"
        ],
        "checkpoint_selection_roles": [
            "parent_v22_guard_selection"
        ],
        "checkpoint_selection_primary_role": "parent_v22_guard",
        "guard_selection_composite_or_mean_used": False,
        "guard_role": (
            "parent_v22_guard_selection_reused_no_v23_reselection"
        ),
        "evaluation_used_for_checkpoint_selection": False,
        "evaluation_decode_passes": 2,
        "parent_raw_reproduction_decode_passes": 1,
        "prior_corrected_evaluation_decode_passes": 1,
        "weights_retrained": False,
        "checkpoint_reused": True,
        "training_history_reused": True,
        "decoder_only_change": True,
        "parent_run_id": PARENT_RUN_ID,
        "parent_model_revision": PARENT_MODEL_REVISION,
        "parent_decoder_revision": PARENT_DECODER_REVISION,
        "parent_artifact_identity_required": True,
        "parent_raw_reproduction_matches": True,
        "parent_checkpoint_state_dict_reloaded": True,
        "training_chunks_shuffled": False,
        "training_state_mode": "exact_pc_keyed_stateful_tbptt",
        "training_state_carried_across_chunks": True,
        "training_state_detached_between_chunks": True,
        "training_state_reset": "only_at_epoch_start",
        "training_state_routing": (
            "one_lstm_state_per_exact_observed_PC"
        ),
        "inference_state_routing": (
            "one_lstm_state_per_exact_observed_PC"
        ),
        "hurdle_equal_mass_self_test": "PASS",
        "data_derived_stable_bias_initialization_self_test": "PASS",
        "finite_positive_count_mode_self_test": "PASS",
        "host_domain_count_rejection_self_test": "PASS",
        "separate_hurdle_and_count_heads_self_test": "PASS",
        "rank_no_action_feedback_self_test": "PASS",
        "training_config": ACTIVE_CONTRACT["training_config"],
        "training_config_pinned_by_run_id": True,
        "training_device": "cuda",
        "cublas_workspace_config": ACTIVE_CONTRACT[
            "determinism_contract"
        ]["cublas_workspace_config"],
        "torch_deterministic_algorithms_enabled": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "float32_matmul_precision": ACTIVE_CONTRACT[
            "determinism_contract"
        ]["float32_matmul_precision"],
    }
    expected.update(source_hashes)
    for key, value in expected.items():
        if metadata.get(key) != value:
            failures.append(
                "{} metadata {}={!r}; expected {!r}".format(
                    tag, key, metadata.get(key), value
                )
            )

    point_by_tag = {
        point["model_tag"]: point for point in ACTIVE_CONTRACT["points"]
    }
    active_point = point_by_tag.get(tag)
    expected_parent_tag = (
        active_point.get("parent_model_tag") if active_point else None
    )
    if (
        expected_parent_tag is None
        or metadata.get("parent_model_tag") != expected_parent_tag
    ):
        failures.append("{} invalid parent model-tag mapping".format(tag))
    for key in (
        "parent_run_metadata_sha256",
        "parent_model_checkpoint_sha256",
        "parent_training_history_sha256",
        "parent_normal_list_sha256",
        "parent_nn_list_sha256",
        "parent_raw_reproduction_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(metadata.get(key, ""))) is None:
            failures.append("{} invalid {}".format(tag, key))
    if metadata.get("parent_nn_list_sha256") != metadata.get(
        "parent_raw_reproduction_sha256"
    ):
        failures.append(
            "{} parent raw action-list reproduction hash differs".format(tag)
        )

    for key, value in ACTIVE_CONTRACT["training_config"].items():
        if metadata.get(key) != value:
            failures.append(
                "{} pinned training {}={!r}; expected {!r}".format(
                    tag, key, metadata.get(key), value
                )
            )
    accelerator = ACTIVE_CONTRACT["determinism_contract"][
        "required_accelerator_name_contains"
    ]
    if accelerator not in str(metadata.get("training_device_name", "")):
        failures.append(
            "{} training device {!r} lacks {!r}".format(
                tag, metadata.get("training_device_name"), accelerator
            )
        )
    if accelerator not in str(metadata.get("redecode_device_name", "")):
        failures.append(
            "{} re-decode device {!r} lacks {!r}".format(
                tag, metadata.get("redecode_device_name"), accelerator
            )
        )

    points = {
        point["model_size"]: point for point in ACTIVE_CONTRACT["points"]
    }
    point = points.get(metadata.get("model_size"))
    vocabulary = metadata.get("delta_vocabulary_exact")
    frequencies = metadata.get("delta_vocabulary_train_frequencies")
    size = metadata.get("delta_vocabulary_exact_size")
    classes = metadata.get("realized_delta_output_classes")
    if (
        point is None
        or metadata.get("architecture_pair_id")
        != (point or {}).get("architecture_pair_id")
        or tag != (point or {}).get("model_tag")
        or not isinstance(vocabulary, list)
        or not isinstance(frequencies, list)
        or not isinstance(size, int)
        or isinstance(size, bool)
        or not 0 < size <= ACTIVE_CONTRACT["delta_vocabulary_max_exact"]
        or classes != size + 1
        or len(vocabulary) != size
        or len(frequencies) != size
        or len(set(vocabulary)) != size
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in vocabulary + frequencies
        )
        or any(value <= 0 for value in frequencies)
        or metadata.get("delta_other_class") != size
    ):
        failures.append(
            "{} invalid dynamic TRAIN delta vocabulary/point".format(tag)
        )
    else:
        realized = expected_parameter_count(point["model_size"], classes)
        maximum = expected_parameter_count(point["model_size"])
        if any(
            metadata.get(key) != realized
            for key in (
                "parameter_count", "realized_parameter_count",
                "expected_parameter_count",
            )
        ):
            failures.append(
                "{} realized point/parameter mismatch".format(tag)
            )
        if metadata.get("maximum_parameter_count") != maximum:
            failures.append(
                "{} maximum point/parameter mismatch".format(tag)
            )
        if point.get("maximum_parameter_count") != maximum:
            failures.append(
                "{} contract point maximum mismatch".format(tag)
            )
    if metadata.get("model_point_contract") != ACTIVE_CONTRACT:
        failures.append(
            "{} model-point contract differs from trainer source".format(tag)
        )

    encoder_hashes = {
        metadata.get("runtime_encoder_sha256"),
        metadata.get("training_runtime_encoder_sha256"),
        metadata.get("inference_runtime_encoder_sha256"),
    }
    if (
        len(encoder_hashes) != 1
        or re.fullmatch(
            r"[0-9a-f]{64}", str(next(iter(encoder_hashes), ""))
        ) is None
    ):
        failures.append("{} train/inference encoder hash mismatch".format(tag))
    for key in (
        "training_state_router_sha256",
        "inference_state_router_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(metadata.get(key, ""))) is None:
            failures.append("{} invalid {}".format(tag, key))
    if metadata.get("training_state_router_sha256") != metadata.get(
        "inference_state_router_sha256"
    ):
        failures.append(
            "{} train/inference state router mismatch".format(tag)
        )

    if isinstance(vocabulary, list) and isinstance(frequencies, list):
        statistics = metadata.get("delta_vocabulary_statistics") or {}
        train_statistics = statistics.get("train") or {}
        other_count = train_statistics.get("other_escape_actions")
        teacher_actions = train_statistics.get("teacher_actions")
        priors = metadata.get("delta_class_empirical_prior")
        biases = metadata.get("delta_class_initial_bias")
        if (
            set(statistics) != {"train", "guard", "eval"}
            or not isinstance(other_count, int)
            or not isinstance(teacher_actions, int)
            or not isinstance(priors, list)
            or not isinstance(biases, list)
            or len(priors) != len(vocabulary) + 1
            or len(biases) != len(vocabulary) + 1
            or teacher_actions != sum(frequencies) + other_count
        ):
            failures.append(
                "{} missing dynamic delta prior evidence".format(tag)
            )
        else:
            counts = list(frequencies) + [other_count]
            denominator = float(teacher_actions + len(counts))
            expected_priors = [
                (value + 1.0) / denominator for value in counts
            ]
            if any(
                not math.isclose(
                    actual, expected, rel_tol=1e-6, abs_tol=1e-9
                )
                for actual, expected in zip(priors, expected_priors)
            ) or any(
                not math.isclose(
                    actual, math.log(expected),
                    rel_tol=1e-6, abs_tol=1e-7,
                )
                for actual, expected in zip(biases, expected_priors)
            ):
                failures.append(
                    "{} invalid dynamic delta prior/bias".format(tag)
                )
        coordinate_bias = metadata.get("delta_coordinate_initial_bias")
        if (
            not isinstance(coordinate_bias, (int, float))
            or isinstance(coordinate_bias, bool)
            or not math.isfinite(float(coordinate_bias))
            or metadata.get("delta_coordinate_bias_initialization")
            != "mean_TRAIN_signed_log_delta"
        ):
            failures.append(
                "{} invalid TRAIN delta-coordinate initialization".format(tag)
            )

    hurdle_stats = metadata.get("hurdle_training_statistics") or {}
    weights = metadata.get("hurdle_class_weights_ZERO_POSITIVE")
    zero_count = hurdle_stats.get("zero_labels")
    positive_count = hurdle_stats.get("positive_labels")
    if (
        not isinstance(zero_count, int)
        or not isinstance(positive_count, int)
        or zero_count <= 0
        or positive_count <= 0
        or not isinstance(weights, list)
        or len(weights) != 2
    ):
        failures.append("{} invalid ZERO/POSITIVE TRAIN weights".format(tag))
    else:
        total = float(zero_count + positive_count)
        expected_weights = [
            total / (2.0 * zero_count),
            total / (2.0 * positive_count),
        ]
        if any(
            not math.isclose(
                actual, expected, rel_tol=1e-7, abs_tol=1e-9
            )
            for actual, expected in zip(weights, expected_weights)
        ) or not math.isclose(
            zero_count * weights[0], positive_count * weights[1],
            rel_tol=1e-7, abs_tol=1e-6,
        ):
            failures.append(
                "{} hurdle weights do not equalize TRAIN mass".format(tag)
            )
        biases = metadata.get("hurdle_initial_bias_ZERO_POSITIVE")
        log_count_bias = metadata.get("positive_log_count_initial_bias")
        if (
            hurdle_stats.get("class_weights_ZERO_POSITIVE") != weights
            or not isinstance(biases, list) or len(biases) != 2
            or any(not math.isfinite(float(value)) for value in biases)
            or any(abs(float(value)) > 1e-8 for value in biases)
            or not isinstance(log_count_bias, (int, float))
            or not math.isfinite(float(log_count_bias))
            or not math.isclose(
                float(log_count_bias),
                float(hurdle_stats.get("positive_log_count_initial_bias")),
                rel_tol=1e-7, abs_tol=1e-9,
            )
        ):
            failures.append(
                "{} invalid TRAIN-derived hurdle/count initialization".format(tag)
            )

    selected_epoch = metadata.get("selected_guard_epoch")
    selection_key = metadata.get("selected_guard_key")
    if (
        not isinstance(selected_epoch, int)
        or isinstance(selected_epoch, bool)
        or not 1 <= selected_epoch <= metadata.get("epochs", 0)
        or not isinstance(selection_key, list)
        or len(selection_key) != 6
    ):
        failures.append(
            "{} invalid lexicographic guard selection".format(tag)
        )

    unique_pc = metadata.get("history_unique_pc_count")
    persistent = metadata.get("peak_persistent_recurrent_state_bytes")
    if (
        not isinstance(unique_pc, int)
        or unique_pc <= 0
        or persistent
        != unique_pc * 2 * metadata.get("model_size", 0) * 4
    ):
        failures.append(
            "{} invalid exact-PC recurrent-state accounting".format(tag)
        )

    for role in ("train", "guard", "eval"):
        for kind in ("stream", "candidate"):
            key = role + "_" + kind + "_content_sha256"
            expected_hash = (
                inputs.get(POLICY, {}).get(role, {}).get(kind, {})
                .get("content_sha256")
            )
            if metadata.get(key) != expected_hash:
                failures.append(
                    "{} {} {} content SHA256 mismatch".format(
                        tag, role, kind
                    )
                )
def validate_active_metadata(metadata, tag, inputs, failures):
    """Fail closed on the natural-cardinality v24 Stride contract."""
    experiment = Path(__file__).resolve().parents[1]
    repository = Path(__file__).resolve().parents[4]
    expected = {
        "run_id": RUN_ID,
        "parent_input_run_id": PARENT_INPUT_RUN_ID,
        "operation": ACTIVE_CONTRACT["operation"],
        "trace": TRACE,
        "model_tag": tag,
        "matched_normal_prefetcher": POLICY,
        "neural_role": "standalone_direct_action_prefetcher",
        "model_family": "lstm",
        "track_model_family": TRACK_MODEL_FAMILY,
        "experiment_revision": EXPERIMENT_REVISION,
        "model_revision": ACTIVE_MODEL_REVISION,
        "decoder_revision": ACTIVE_CONTRACT["decoder_revision"],
        "source_decision_effective_external_input": ["pc", "addr"],
        "same_external_input_contract": True,
        "training_inference_input_encoder_identical": True,
        "training_runtime_fields": ["pc", "addr"],
        "inference_runtime_fields": ["pc", "addr"],
        "runtime_feature_count": 122,
        "raw_runtime_feature_count": 122,
        "causal_runtime_feature_count": 0,
        "runtime_encoding": "lossless_raw_pc64_plus_line58_only",
        "engineered_runtime_features": [],
        "teacher_actions_are_model_inputs": False,
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "normal_policy_request_rate_used_as_budget": False,
        "normal_policy_templates_used_by_neural_inference": False,
        "probability_threshold_used": False,
        "threshold_related_hardcodes_used": False,
        "inference_policy_hardcodes_used": False,
        "neural_degree_cap": None,
        "fixed_page_offset_classes": None,
        "same_page_rule_used_by_neural_inference": False,
        "future_label_window_used": False,
        "decoder_training_mode": ACTIVE_CONTRACT["decoder_training_mode"],
        "decoding_rule": ACTIVE_CONTRACT["decoding_rule"],
        "decision_rule": ACTIVE_CONTRACT["decoding_rule"],
        "count_training_objective": ACTIVE_CONTRACT[
            "count_training_objective"
        ],
        "categorical_count_head_used": True,
        "count_head_used": True,
        "count_regression_used": False,
        "log_count_used": False,
        "hurdle_head_used": False,
        "stop_token_used": False,
        "separate_global_gate_used": False,
        "separate_count_head_used": False,
        "stop_padding_used": False,
        "loss_class_reweighting_used": False,
        "decode_prior_correction_used": False,
        "manual_loss_weights_used": False,
        "count_zero_is_implicit_hurdle": True,
        "count_support_is_dataset_derived": True,
        "count_support_is_normal_request_budget": False,
        "count_support_is_tuned_degree": False,
        "delta_training_objective": ACTIVE_CONTRACT[
            "delta_training_objective"
        ],
        "all_deltas_relative_to_current_demand": True,
        "stride_fill_level": "FILL_L2_only_no_learned_fill_head",
        "fill_level": "FILL_L2_only_no_fill_head",
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_previous_predicted_action_used_as_input": False,
        "decoder_previous_sampled_action_used_as_input": False,
        "action_loss_scope": "teacher_action_ranks_only",
        "blocked_validation_length_source": ACTIVE_CONTRACT[
            "blocked_validation_length_source"
        ],
        "original_guard_role": ACTIVE_CONTRACT["original_guard_role"],
        "blocked_validation_selected_checkpoint": True,
        "original_guard_used_for_checkpoint_selection": False,
        "original_guard_used_for_selection": False,
        "evaluation_used_for_checkpoint_selection": False,
        "evaluation_used_for_selection": False,
        "evaluation_policy_decode_count": 1,
        "diagnostic_eval_decode_count": 1,
        "oracle_diagnostics_replayed": False,
        "oracle_diagnostics_excluded_from_fair_claims": True,
        "weights_retrained": True,
        "checkpoint_reused": False,
        "decoder_only_change": False,
        "training_chunks_shuffled": False,
        "training_state_mode": "exact_pc_keyed_stateful_tbptt",
        "training_state_carried_across_chunks": True,
        "training_state_detached_between_chunks": True,
        "training_state_routing": "one_lstm_state_per_exact_observed_PC",
        "inference_state_routing": "one_lstm_state_per_exact_observed_PC",
        "training_config": ACTIVE_CONTRACT["training_config"],
        "cublas_workspace_config": ACTIVE_CONTRACT[
            "determinism_contract"
        ]["cublas_workspace_config"],
        "torch_deterministic_algorithms_enabled": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "float32_matmul_precision": "highest",
        "determinism_fail_closed": True,
        "stochastic_decoding": False,
        "input_archive_reused_byte_for_byte": True,
    }
    source_hashes = {
        "trainer_source_sha256": sha256(
            experiment / "python" / "train_and_offline_infer.py"
        ),
        "model_contract_source_sha256": sha256(
            experiment / "python" / "model_contract.py"
        ),
        "threshold_free_policy_source_sha256": sha256(
            repository / "formal_NN_training/common/threshold_free_policy.py"
        ),
    }
    expected.update(source_hashes)
    for key, value in expected.items():
        if metadata.get(key) != value:
            failures.append(
                "{} metadata {}={!r}; expected {!r}".format(
                    tag, key, metadata.get(key), value
                )
            )

    point_by_tag = {
        point["model_tag"]: point for point in ACTIVE_CONTRACT["points"]
    }
    point = point_by_tag.get(tag)
    count_support = metadata.get("count_support")
    vocabulary = metadata.get("exact_delta_vocabulary")
    if (
        point is None
        or metadata.get("model_size") != point["model_size"]
        or metadata.get("architecture_pair_id")
        != point["architecture_pair_id"]
        or not isinstance(count_support, list)
        or not count_support
        or count_support != list(range(len(count_support)))
        or not isinstance(vocabulary, list)
        or not 0 < len(vocabulary)
        <= ACTIVE_CONTRACT["delta_vocabulary_max_exact"]
        or len(set(vocabulary)) != len(vocabulary)
        or metadata.get("other_delta_class") != len(vocabulary)
    ):
        failures.append("{} invalid v24 realized support".format(tag))
    else:
        realized = expected_parameter_count(
            point["model_size"], len(count_support), len(vocabulary) + 1
        )
        for key in (
            "parameter_count", "realized_parameter_count",
            "expected_parameter_count",
        ):
            if metadata.get(key) != realized:
                failures.append(
                    "{} {} does not match realized v24 formula".format(
                        tag, key
                    )
                )

    if metadata.get("model_point_contract") != ACTIVE_CONTRACT:
        failures.append("{} embedded model contract differs".format(tag))
    selected_epoch = metadata.get("selected_epoch")
    selected_validation = metadata.get("selected_blocked_validation")
    if (
        not isinstance(selected_epoch, int)
        or isinstance(selected_epoch, bool)
        or not 1 <= selected_epoch <= metadata.get("epochs", 0)
        or not isinstance(selected_validation, dict)
        or not math.isfinite(float(selected_validation.get(
            "natural_action_list_nll_per_callback", float("nan")
        )))
    ):
        failures.append("{} invalid blocked-validation selection".format(tag))
    oracle = metadata.get("oracle_diagnostics")
    if (
        not isinstance(oracle, dict)
        or oracle.get("diagnosis_only") is not True
        or oracle.get("excluded_from_fair_replay_claims") is not True
        or (oracle.get("oracle_count_plus_nn_action") or {}).get("replayed")
        is not False
        or (oracle.get("nn_count_plus_oracle_action") or {}).get("replayed")
        is not False
    ):
        failures.append("{} oracle diagnosis leaked into replay".format(tag))

    encoder_hashes = {
        metadata.get("runtime_encoder_sha256"),
        metadata.get("training_runtime_encoder_sha256"),
        metadata.get("inference_runtime_encoder_sha256"),
    }
    if (
        len(encoder_hashes) != 1
        or re.fullmatch(
            r"[0-9a-f]{64}", str(next(iter(encoder_hashes), ""))
        ) is None
        or metadata.get("training_state_router_sha256")
        != metadata.get("inference_state_router_sha256")
    ):
        failures.append("{} encoder/state-router identity mismatch".format(tag))

    for role in ("train", "guard", "eval"):
        for kind in ("stream", "candidate"):
            key = role + "_" + kind + "_content_sha256"
            expected_hash = (
                inputs.get(POLICY, {}).get(role, {}).get(kind, {})
                .get("content_sha256")
            )
            if metadata.get(key) != expected_hash:
                failures.append(
                    "{} {} input content hash mismatch".format(tag, key)
                )


def validate_metadata(metadata, tag, inputs, failures):
    if metadata.get("model_revision") == ACTIVE_MODEL_REVISION:
        validate_active_metadata(metadata, tag, inputs, failures)
        return
    policy = POLICY
    common = {
        "trace": TRACE,
        "model_tag": tag,
        "matched_normal_prefetcher": policy,
        "source_decision_effective_external_input": ["pc", "addr"],
        "same_external_input_contract": True,
        "training_inference_input_encoder_identical": True,
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
        "request_count_residual_scope": "none_event_local",
        "training_regularization_used": False,
        "inference_policy_hardcodes_used": False,
        "learned_request_count": True,
        "nn_generates_own_target_addresses": True,
        "training_chunks_shuffled": False,
        "causal_no_future_self_test": "PASS",
        "decoder_probability_mass_carries_train_guard_history": False,
        "cross_event_probability_credit_used": False,
        "sampled_outputs_used_as_decoder_feedback": False,
        "cnn_architecture_self_test": "NOT_APPLICABLE",
        "event_logger_schema": EVENT_LOGGER_SCHEMA,
        "candidate_attachment_mode": CANDIDATE_ATTACHMENT_MODE,
        "experiment_revision": EXPERIMENT_REVISION,
        "neural_role": "standalone_direct_action_prefetcher",
        "track_model_family": TRACK_MODEL_FAMILY,
        "runtime_feature_count": 122,
        "runtime_encoding": (
            "lossless uint64 PC plus lossless 58-bit cache-line number"
        ),
        "runtime_pc_bits": 64,
        "runtime_line_number_bits": 58,
        "runtime_constant_offset_bits_removed": 6,
        "cross_event_rng_state_used": False,
        "decoder_train_sampling_performed": False,
        "decoder_guard_sampling_performed": False,
        "decoder_event_key_uses_teacher_information": False,
        "decoder_action_rank_origin": 0,
    }
    for key, expected in common.items():
        if metadata.get(key) != expected:
            failures.append(
                "{} metadata {}={!r}; expected {!r}".format(
                    tag, key, metadata.get(key), expected
                )
            )
    revision = metadata.get("model_revision")
    revision_common = {
        V15_MODEL_REVISION: {
            "decoder_training_mode": (
                "free_running_autoregressive_same_as_inference"
            ),
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
            "event_keyed_crn_self_test": "PASS",
            "event_keyed_hurdle_count_self_test": "PASS",
            "canonicalized_mixture_sampling_self_test": "PASS",
            "stochastic_decoding_reproducible": True,
            "delta_mixture_decoding_rule": (
                "event_keyed_mean_sorted_categorical_inverse_cdf_then_"
                "component_mean"
            ),
            "delta_decoder_feedback_rule": (
                "complete_mixture_expectation_same_in_training_and_inference"
            ),
            "delta_mixture_components": 3,
            "common_random_numbers_across_capacities": True,
            "strict_common_random_numbers_across_capacities": True,
            "decoder_sampling_roles": ["eval"],
            "decoder_event_key_definition": "zero_based_eval_demand_idx",
            "decoder_key_includes_sampler_revision": True,
        },
        V16_MODEL_REVISION: {
            "decoder_training_mode": (
                "free_running_autoregressive_same_as_inference"
            ),
            "data_derived_gate_class_weights_used": True,
            "gate_class_weighting_used": True,
            "gate_training_objective": (
                "data_derived_frequency_balanced_two_class_cross_entropy"
            ),
            "gate_decoding_rule": "deterministic_two_class_argmax",
            "request_count_training_objective": (
                "balanced_two_class_hurdle_plus_positive_log_count_smooth_l1"
            ),
            "request_count_decoding_rule": (
                "deterministic_gate_argmax_plus_rounded_exp_positive_log_count"
            ),
            "event_keyed_crn_self_test": "NOT_APPLICABLE",
            "event_keyed_hurdle_count_self_test": "NOT_APPLICABLE",
            "canonicalized_mixture_sampling_self_test": "NOT_APPLICABLE",
            "deterministic_decoding_reproducible": True,
            "stochastic_decoding_reproducible": False,
            "delta_mixture_decoding_rule": None,
            "delta_decoder_feedback_rule": (
                "emitted_scalar_coordinate_same_in_training_and_inference"
            ),
            "delta_mixture_components": 0,
            "common_random_numbers_across_capacities": False,
            "strict_common_random_numbers_across_capacities": False,
            "decoder_sampling_roles": [],
            "decoder_event_key_definition": None,
            "decoder_key_includes_sampler_revision": False,
            "deterministic_decoding": True,
            "stochastic_decoding": False,
            "guard_role": "causal_input_history_warmup_and_audit_only",
            "deterministic_count_and_balance_self_test": "PASS",
            "gate_class_weights_source": (
                "train_zero_positive_frequencies_equal_aggregate_loss_mass"
            ),
        },
        V17_MODEL_REVISION: {
            "decoder_training_mode": (
                "free_running_autoregressive_same_as_inference"
            ),
            "data_derived_gate_class_weights_used": True,
            "gate_class_weighting_used": True,
            "gate_training_objective": (
                "data_derived_frequency_balanced_two_class_cross_entropy"
            ),
            "gate_decoding_rule": (
                "prior_corrected_deterministic_two_class_argmax"
            ),
            "gate_prior_correction": (
                "subtract_log_training_class_weight_before_argmax"
            ),
            "gate_prior_correction_self_test": "PASS",
            "request_count_training_objective": (
                "balanced_two_class_hurdle_plus_positive_log_count_smooth_l1"
            ),
            "request_count_decoding_rule": (
                "prior_corrected_gate_argmax_plus_rounded_exp_"
                "positive_log_count"
            ),
            "event_keyed_crn_self_test": "NOT_APPLICABLE",
            "event_keyed_hurdle_count_self_test": "NOT_APPLICABLE",
            "canonicalized_mixture_sampling_self_test": "NOT_APPLICABLE",
            "deterministic_decoding_reproducible": True,
            "stochastic_decoding_reproducible": False,
            "delta_mixture_decoding_rule": None,
            "delta_decoder_feedback_rule": (
                "emitted_scalar_coordinate_same_in_training_and_inference"
            ),
            "delta_mixture_components": 0,
            "common_random_numbers_across_capacities": False,
            "strict_common_random_numbers_across_capacities": False,
            "decoder_sampling_roles": [],
            "decoder_event_key_definition": None,
            "decoder_key_includes_sampler_revision": False,
            "deterministic_decoding": True,
            "stochastic_decoding": False,
            "guard_role": "causal_input_history_warmup_and_audit_only",
            "deterministic_count_and_balance_self_test": "PASS",
            "gate_class_weights_source": (
                "train_zero_positive_frequencies_equal_aggregate_loss_mass"
            ),
        },
        V18_MODEL_REVISION: {
            "decoder_training_mode": (
                "teacher_count_scheduled_loss_with_free_running_"
                "self_action_feedback"
            ),
            "data_derived_gate_class_weights_used": False,
            "gate_class_weighting_used": False,
            "gate_training_objective": (
                "natural_frequency_unweighted_two_class_cross_entropy"
            ),
            "gate_decoding_rule": "raw_deterministic_two_class_argmax",
            "gate_prior_correction": None,
            "gate_prior_correction_self_test": "NOT_APPLICABLE",
            "gate_class_weights_source": None,
            "gate_class_weights": None,
            "gate_empirical_prior_source": (
                "train_zero_positive_frequencies"
            ),
            "gate_bias_initialization": (
                "log_train_empirical_zero_positive_prior"
            ),
            "gate_prior_bias_initialization_self_test": "PASS",
            "request_count_training_objective": (
                "natural_frequency_two_class_hurdle_plus_positive_"
                "log_count_smooth_l1"
            ),
            "request_count_decoding_rule": (
                "raw_gate_argmax_plus_rounded_exp_positive_log_count"
            ),
            "event_keyed_crn_self_test": "NOT_APPLICABLE",
            "event_keyed_hurdle_count_self_test": "NOT_APPLICABLE",
            "canonicalized_mixture_sampling_self_test": "NOT_APPLICABLE",
            "deterministic_decoding_reproducible": True,
            "stochastic_decoding_reproducible": False,
            "delta_mixture_decoding_rule": None,
            "delta_decoder_feedback_rule": (
                "emitted_scalar_coordinate_same_in_training_and_inference"
            ),
            "delta_mixture_components": 0,
            "common_random_numbers_across_capacities": False,
            "strict_common_random_numbers_across_capacities": False,
            "decoder_sampling_roles": [],
            "decoder_event_key_definition": None,
            "decoder_key_includes_sampler_revision": False,
            "deterministic_decoding": True,
            "stochastic_decoding": False,
            "guard_role": "causal_input_history_warmup_and_audit_only",
            "deterministic_count_and_balance_self_test": "NOT_APPLICABLE",
            "deterministic_count_and_natural_gate_self_test": "PASS",
        },
    }
    profile = revision_common.get(revision)
    if profile is None:
        failures.append("{} unsupported model revision {!r}".format(tag, revision))
    else:
        for key, expected in profile.items():
            if metadata.get(key) != expected:
                failures.append(
                    "{} metadata {}={!r}; expected {!r}".format(
                        tag, key, metadata.get(key), expected
                    )
                )

    if revision in (V16_MODEL_REVISION, V17_MODEL_REVISION):
        statistics = (
            metadata.get("request_count_training_label_statistics") or {}
        )
        decision_callbacks = statistics.get("decision_callbacks")
        positive_callbacks = statistics.get("positive_callbacks")
        zero_callbacks = statistics.get("zero_callbacks")
        weights = metadata.get("gate_class_weights")
        valid_statistics = (
            isinstance(decision_callbacks, int)
            and not isinstance(decision_callbacks, bool)
            and isinstance(positive_callbacks, int)
            and not isinstance(positive_callbacks, bool)
            and isinstance(zero_callbacks, int)
            and not isinstance(zero_callbacks, bool)
            and decision_callbacks > 0
            and positive_callbacks > 0
            and zero_callbacks > 0
            and positive_callbacks + zero_callbacks == decision_callbacks
        )
        valid_weights = isinstance(weights, list) and len(weights) == 2
        if not valid_statistics or not valid_weights:
            failures.append(
                "{} invalid data-derived gate class-weight evidence".format(tag)
            )
        else:
            expected_weights = [
                float(decision_callbacks) / (2.0 * zero_callbacks),
                float(decision_callbacks) / (2.0 * positive_callbacks),
            ]
            if any(
                not isinstance(actual, (int, float))
                or isinstance(actual, bool)
                or not math.isfinite(float(actual))
                or not math.isclose(
                    float(actual), expected,
                    rel_tol=1e-6, abs_tol=1e-7,
                )
                for actual, expected in zip(weights, expected_weights)
            ):
                failures.append(
                    "{} gate class weights {!r}; expected {!r}".format(
                        tag, weights, expected_weights
                    )
                )
            diagnostic_weights = (
                metadata.get("request_count_decoder_diagnostics") or {}
            ).get("gate_class_weights")
            if diagnostic_weights != weights:
                failures.append(
                    "{} gate class weights differ between metadata and "
                    "decoder diagnostics".format(tag)
                )

    if revision == V18_MODEL_REVISION:
        statistics = (
            metadata.get("request_count_training_label_statistics") or {}
        )
        decision_callbacks = statistics.get("decision_callbacks")
        positive_callbacks = statistics.get("positive_callbacks")
        zero_callbacks = statistics.get("zero_callbacks")
        prior = metadata.get("gate_empirical_prior")
        initial_bias = metadata.get("gate_initial_bias")
        valid_statistics = (
            isinstance(decision_callbacks, int)
            and not isinstance(decision_callbacks, bool)
            and isinstance(positive_callbacks, int)
            and not isinstance(positive_callbacks, bool)
            and isinstance(zero_callbacks, int)
            and not isinstance(zero_callbacks, bool)
            and decision_callbacks > 0
            and positive_callbacks > 0
            and zero_callbacks > 0
            and positive_callbacks + zero_callbacks == decision_callbacks
        )
        valid_vectors = (
            isinstance(prior, list) and len(prior) == 2
            and isinstance(initial_bias, list) and len(initial_bias) == 2
        )
        if not valid_statistics or not valid_vectors:
            failures.append(
                "{} invalid natural-frequency gate-prior evidence".format(tag)
            )
        else:
            expected_prior = [
                float(zero_callbacks) / float(decision_callbacks),
                float(positive_callbacks) / float(decision_callbacks),
            ]
            expected_bias = [math.log(value) for value in expected_prior]
            for name, actual_values, expected_values in (
                ("gate empirical prior", prior, expected_prior),
                ("gate initial bias", initial_bias, expected_bias),
            ):
                if any(
                    not isinstance(actual, (int, float))
                    or isinstance(actual, bool)
                    or not math.isfinite(float(actual))
                    or not math.isclose(
                        float(actual), expected,
                        rel_tol=1e-6, abs_tol=1e-7,
                    )
                    for actual, expected in zip(
                        actual_values, expected_values
                    )
                ):
                    failures.append(
                        "{} {} {!r}; expected {!r}".format(
                            tag, name, actual_values, expected_values
                        )
                    )
            diagnostics = (
                metadata.get("request_count_decoder_diagnostics") or {}
            )
            if diagnostics.get("gate_empirical_prior") != prior:
                failures.append(
                    "{} gate empirical prior differs between metadata and "
                    "decoder diagnostics".format(tag)
                )
            if diagnostics.get("gate_initial_bias") != initial_bias:
                failures.append(
                    "{} gate initial bias differs between metadata and "
                    "decoder diagnostics".format(tag)
                )

    sampler = metadata.get("decoder_sampler")
    expected_key_fields = [
        "revision", "decoder_seed", "trace", "policy", "role",
        "event_key", "head", "action_rank",
    ]
    if revision == V15_MODEL_REVISION:
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
        hash_keys = (
            "decoder_event_key_stream_sha256",
            "decoder_sampler_source_sha256",
            "decoder_sampler_key_schedule_sha256",
            "decoder_sampling_schedule_sha256",
            "training_state_router_sha256",
            "inference_state_router_sha256",
        )
    else:
        if sampler is not None or metadata.get("decoder_key_fields") != []:
            failures.append("{} unexpected stochastic decoder state".format(tag))
        hash_keys = (
            "training_state_router_sha256",
            "inference_state_router_sha256",
        )
    for key in hash_keys:
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
    if len(model_tags) != len(set(model_tags)):
        raise SystemExit("--model-tags contains duplicate tags")
    for tag in model_tags:
        if SAFE_PATH_TOKEN.match(tag) is None or not (
            tag.startswith(MODEL_TAG_PREFIX)
            or tag.startswith("independent_delta_stride_lstm_")
            or tag.startswith("global_local_grammar_stride_lstm_")
        ):
            raise SystemExit("invalid model tag {}".format(tag))
    if any(tag.startswith(MODEL_TAG_PREFIX) for tag in model_tags):
        if len(model_tags) != len(ACTIVE_MODEL_TAGS) or set(model_tags) != set(
            ACTIVE_MODEL_TAGS
        ):
            raise SystemExit(
                "active {} run requires the exact configured model-tag set: {}"
                .format(ACTIVE_MODEL_REVISION, ",".join(ACTIVE_MODEL_TAGS))
            )
    if not args.run_dir.is_dir():
        raise SystemExit("--run-dir is not an existing directory: {}".format(
            args.run_dir
        ))

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

    observed_model_revisions = {
        metadata.get("model_revision")
        for metadata in metadata_by_tag.values()
    }
    observed_model_revision = (
        next(iter(observed_model_revisions))
        if len(observed_model_revisions) == 1 else None
    )
    if len(observed_model_revisions) != 1:
        failures.append(
            "neural points do not share one model revision: {}".format(
                sorted(repr(value) for value in observed_model_revisions)
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
        row["replay_list_zero_target_entries"] = info[
            "zero_target_entries"
        ]
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
        if row["rejected_entries"]:
            failures.append(
                "{} replayer rejected {} list entries; replay list contains "
                "{} zero-address targets".format(
                    method, row["rejected_entries"],
                    row["replay_list_zero_target_entries"],
                )
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
        "replay_list_entries", "replay_list_triggers",
        "replay_list_zero_target_entries",
        "runtime_reachable_list_entries", "runtime_reachable_list_triggers",
        "runtime_unreached_list_entries", "runtime_unreached_list_triggers",
        "loaded_entries", "loaded_triggers", "dumped_loaded_triggers",
        "rejected_entries", "matched", "emitted", "callbacks", "log",
        "event_log",
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
    capacity_action_list_audit = {}
    for tag, metadata in sorted(metadata_by_tag.items()):
        decoder = metadata.get("decoder_eval_diagnostics") or {}
        behavior = metadata.get("heldout_behavior_metrics") or {}
        capacity_action_list_audit[tag] = {
            "model_size": metadata.get("model_size"),
            "nn_list_sha256": metadata.get("nn_list_sha256"),
            "raw_predicted_action_count": metadata.get(
                "offline_nn_entries"
            ),
            "decoded_count_distribution": decoder.get(
                "decoded_count_distribution"
            ),
            "count_confusion": behavior.get("count_confusion"),
            "count_mae": behavior.get("count_mae"),
            "mean_count_entropy": decoder.get("mean_count_entropy"),
            "mean_count_entropy_normalized": decoder.get(
                "mean_count_entropy_normalized"
            ),
            "mean_delta_entropy": decoder.get("mean_delta_entropy"),
            "mean_delta_entropy_normalized": decoder.get(
                "mean_delta_entropy_normalized"
            ),
            "oracle_diagnostics": metadata.get("oracle_diagnostics"),
        }
    action_hashes = {
        value.get("nn_list_sha256")
        for value in capacity_action_list_audit.values()
        if value.get("nn_list_sha256")
    }
    if len(capacity_action_list_audit) == 5 and len(action_hashes) == 1:
        warnings.append(
            "all five v24 Stride capacities exported the same hard action "
            "list; inspect count confusion and entropy before interpreting IPC"
        )
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
            "zero_target_entries": info["zero_target_entries"],
            "sha256": info["sha256"],
        }
        for method, info in sorted(replay_lists.items())
    }

    status = "FAIL" if failures else "PASS"
    direct_action_contracts = {
        V15_MODEL_REVISION: (
            "The neural model learns an unweighted zero/positive hurdle, an "
            "unbounded conditional Poisson excess count, and an autoregressive "
            "mixture over direct signed cache-line deltas. Stateless SHA-256 "
            "event-keyed inverse-CDF sampling supplies common random numbers "
            "across capacities and uses no selected threshold, fixed page-offset "
            "table, same-page rule, or Stride degree cap."
        ),
        V16_MODEL_REVISION: (
            "The neural model learns a training-frequency-balanced categorical "
            "zero/positive gate, a deterministic unbounded positive log-count, "
            "and autoregressive scalar signed-log cache-line deltas. Argmax and "
            "rounding decode the actions deterministically; emitted scalar "
            "coordinates are fed back in both training and inference. There is "
            "no selected threshold, candidate bank, fixed page-offset table, "
            "same-page rule, or Stride degree cap."
        ),
        V17_MODEL_REVISION: (
            "The neural model keeps the training-frequency-balanced gate, "
            "positive log-count, and scalar signed-log delta, but subtracts "
            "the log training class weights before gate argmax to restore "
            "the empirical prior. Decode remains deterministic and uses no "
            "selected threshold, request budget, candidate bank, fixed "
            "page-offset table, same-page rule, or Stride degree cap."
        ),
        V18_MODEL_REVISION: (
            "The neural model trains its zero/positive gate with unweighted "
            "natural-frequency cross-entropy, initializes the gate bias from "
            "the empirical training prior, and decodes the raw logits by "
            "deterministic argmax. Positive log-count and autoregressive "
            "scalar signed-log delta decoding are unchanged. There is no "
            "selected threshold, request budget, candidate bank, fixed "
            "page-offset table, same-page rule, or Stride degree cap."
        ),
        ACTIVE_MODEL_REVISION: (
            "The neural model uses one exact-PC keyed LSTM over lossless raw "
            "PC64/line58 bits with no engineered stride or reuse feature. "
            "One unweighted categorical head predicts natural callback "
            "cardinality, with zero as the implicit no-request case; only "
            "real teacher ranks supervise the TRAIN delta vocabulary/OTHER "
            "action head. A chronological suffix of TRAIN selects checkpoints "
            "by action-list NLL while the original GUARD is phase-shift audit "
            "only. There is no hurdle, count regression, STOP padding, class "
            "reweighting, prior correction, threshold, source template, page "
            "rule, request budget, neural degree cap, or action feedback."
        ),
    }
    payload = {
        "status": status,
        "fair_comparison_claim_allowed": not failures,
        "trace": TRACE,
        "model_family_track": TRACK_MODEL_FAMILY,
        "model_revision": observed_model_revision,
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
                "offline_<configured standalone Stride LSTM model tag>",
            ],
        },
        "context_reference_only": [
            "no_pref", "live_stride_reference"
        ],
        "transport_fidelity": transport_fidelity,
        "replay_accounting": replay_accounting,
        "warnings": warnings,
        "undefined_rate_semantics": (
            "JSON null / blank CSV means denominator zero; numeric 0 means "
            "the denominator existed and the numerator was zero"
        ),
        "capacity_action_list_audit": capacity_action_list_audit,
        "track_guardrail": (
            "Every neural point sees only the same PC/address stream as Stride. "
            "Captured Stride actions are supervised targets and comparator replay "
            "only; inference has no threshold, request budget, or degree cap."
        ),
        "transport": (
            "Captured Stride actions and learned neural actions are replayed "
            "through the same PC-line-occ ListReplayer."
        ),
        "direct_action_contract": direct_action_contracts.get(
            observed_model_revision,
            "Unsupported or mixed model revision; no direct-action claim.",
        ),
        "model_input_guardrail": {
            "normal_stride_private_state": [
                "PC-indexed tracker table", "confidence", "last stride"
            ],
            "direct_nn_inputs": [
                "lossless uint64 PC and 58-bit cache-line-number encodings",
                "exact-PC-routed recurrent history derived only from those raw callbacks",
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
                "name": "exact-PC keyed raw-input LSTM",
                "history": "one hidden/cell pair per exact observed PC",
                "training": "chronological keyed TBPTT with state carried and detached",
                "runtime_features": "raw PC64 plus aligned line58 only",
                "action_decoder": "natural categorical count followed by exactly K independent rank-conditioned TRAIN delta vocabulary/OTHER actions",
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
                if not (
                    tag.startswith(MODEL_TAG_PREFIX)
                    or tag.startswith("rank_stop_emit_stride_lstm_")
                    or tag.startswith("independent_delta_" + policy + "_")
                    or tag.startswith(
                        "global_local_grammar_" + policy + "_"
                    )
                ):
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
