#!/usr/bin/env python3
"""Fail-closed validation for the independent 623 stride track."""
import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from model_contract import (
    BLOCKED_VALIDATION_LENGTH_SOURCE, CHECKPOINT_SELECTION,
    DECODER_TRAINING_MODE,
    DELTA_OBJECTIVE, EXPERIMENT_REVISION, FULL_OBJECTIVE,
    FIT_DENOMINATOR, FIT_NUMERATOR, HURDLE_OBJECTIVE,
    ORIGINAL_GUARD_ROLE, POLICY,
    POSITIVE_COUNT_OBJECTIVE, TRACE, parse_exact_integer,
)

ROLES = ("train", "guard", "eval")
LOGGER_SCHEMA = "623_causal_trigger_v5"
ATTACHMENT_MODE = "explicit_trigger_event_id"
LINE_NUMBER_BITS = 58
LINE_MODULUS = 1 << LINE_NUMBER_BITS
LINE_MASK = LINE_MODULUS - 1


def signed_line_delta(target, base):
    value = (int(target) - int(base)) & LINE_MASK
    if value >= (1 << (LINE_NUMBER_BITS - 1)):
        value -= LINE_MODULUS
    return value


def as_int(value):
    return parse_exact_integer(value)


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


def identity_sha256(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(("{},{},{},{}\n".format(*row)).encode())
    return digest.hexdigest()


def read_stream(path):
    rows = []
    occurrences = defaultdict(int)
    last_cycle = -1
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "trace", "demand_idx", "cycle", "pc", "line", "pc_line_occ",
            "logger_schema",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError("{} missing stream columns {}".format(path, sorted(missing)))
        for index, row in enumerate(reader):
            cycle = as_int(row["cycle"])
            pc = as_int(row["pc"])
            line = as_int(row["line"])
            occ = as_int(row["pc_line_occ"])
            pair = (pc, line)
            expected_occ = occurrences[pair]
            occurrences[pair] += 1
            if row["trace"] != TRACE or as_int(row["demand_idx"]) != index:
                raise RuntimeError("{} identity/order failure at demand {}".format(path, index))
            if row["logger_schema"] != LOGGER_SCHEMA:
                raise RuntimeError("{} contains stale logger schema".format(path))
            if cycle < last_cycle:
                raise RuntimeError("{} cycle order regressed at demand {}".format(path, index))
            if occ != expected_occ:
                raise RuntimeError("{} occurrence mismatch at demand {}".format(path, index))
            rows.append((index, pc, line, occ))
            last_cycle = cycle
    if not rows:
        raise RuntimeError("empty demand stream {}".format(path))
    return rows


def read_candidates(path, policy, stream_rows):
    counts = defaultdict(int)
    fill_counts = defaultdict(int)
    total = 0
    delta_counts = Counter()
    self_targets = 0
    cross_4k_page_targets = 0
    last_pf_event_id = -1
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "trace", "policy", "demand_idx", "pc", "line", "pc_line_occ",
            "candidate_rank", "pf_line", "fill_level", "accepted", "duplicate",
            "trigger_event_id", "pf_event_id", "event_distance", "match_mode",
            "logger_schema",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError("{} missing candidate columns {}".format(path, sorted(missing)))
        for row in reader:
            demand_idx = as_int(row["demand_idx"])
            if demand_idx < 0 or demand_idx >= len(stream_rows):
                raise RuntimeError("{} demand_idx out of range".format(path))
            index, pc, line, occ = stream_rows[demand_idx]
            observed = (
                as_int(row["demand_idx"]),
                as_int(row["pc"]),
                as_int(row["line"]),
                as_int(row["pc_line_occ"]),
            )
            if observed != (index, pc, line, occ):
                raise RuntimeError("{} transport identity mismatch".format(path))
            if row["trace"] != TRACE or row["policy"] != policy:
                raise RuntimeError("{} trace/policy mismatch".format(path))
            if row["logger_schema"] != LOGGER_SCHEMA:
                raise RuntimeError("{} contains stale logger schema".format(path))
            if row["match_mode"] != ATTACHMENT_MODE:
                raise RuntimeError("{} contains non-explicit candidate attachment".format(path))

            counts[demand_idx] += 1
            if as_int(row["candidate_rank"]) != counts[demand_idx]:
                raise RuntimeError("{} candidate ranks are not contiguous".format(path))
            trigger_id = as_int(row["trigger_event_id"])
            pf_event_id = as_int(row["pf_event_id"])
            distance = as_int(row["event_distance"])
            if pf_event_id <= last_pf_event_id:
                raise RuntimeError("{} PF event IDs are not increasing".format(path))
            if trigger_id >= pf_event_id or distance != pf_event_id - trigger_id:
                raise RuntimeError("{} explicit trigger distance mismatch".format(path))
            if distance < 1:
                raise RuntimeError("{} PF event does not follow its trigger".format(path))
            accepted = as_int(row["accepted"])
            duplicate = as_int(row["duplicate"])
            if accepted not in (0, 1) or duplicate not in (0, 1) or (duplicate and not accepted):
                raise RuntimeError("{} invalid candidate outcome bits".format(path))
            fill_level = as_int(row["fill_level"])
            if fill_level != 2:
                raise RuntimeError("{} stride candidate is not FILL_L2".format(path))
            pf_line = as_int(row["pf_line"])
            if pf_line < 0 or pf_line >= LINE_MODULUS:
                raise RuntimeError("{} stride target exceeds uint64 line domain".format(path))
            delta_counts[signed_line_delta(pf_line, line)] += 1
            self_targets += int(pf_line == line)
            cross_4k_page_targets += int(pf_line // 64 != line // 64)
            fill_counts[fill_level] += 1
            last_pf_event_id = pf_event_id
            total += 1
    if total == 0:
        raise RuntimeError("empty candidate bank {}".format(path))
    ordered_deltas = sorted(
        delta_counts, key=lambda value: (-delta_counts[value], value)
    )
    profile = {
        "unique_signed_line_deltas": len(delta_counts),
        "minimum_signed_line_delta": min(delta_counts),
        "maximum_signed_line_delta": max(delta_counts),
        "self_target_actions": self_targets,
        "cross_4k_page_actions": cross_4k_page_targets,
        "top_signed_line_deltas": [
            {"delta": value, "count": delta_counts[value]}
            for value in ordered_deltas[:32]
        ],
    }
    return (
        total, max(counts.values()), dict(sorted(fill_counts.items())),
        profile,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--manifest-out", required=True, type=Path)
    args = parser.parse_args()

    manifest = {
        "status": "PASS",
        "trace": TRACE,
        "experiment_revision": EXPERIMENT_REVISION,
        "event_logger_schema": LOGGER_SCHEMA,
        "candidate_attachment_mode": ATTACHMENT_MODE,
        "policy": POLICY,
        "independent_matched_track": True,
        "neural_role": "standalone_direct_action_prefetcher",
        "source_decision_effective_external_input": ["pc", "addr"],
        "same_external_input_contract": True,
        "training_inference_input_encoder_identical": True,
        "decoder_training_mode": DECODER_TRAINING_MODE,
        "decoder_previous_teacher_action_used_as_input": False,
        "decoder_previous_predicted_action_used_as_input": False,
        "decoder_previous_sampled_action_used_as_input": False,
        "terminal_stop_supervised_for_every_teacher_sequence": False,
        "stop_padding_used": False,
        "dual_context_core_used": True,
        "global_chronological_lstm_used": True,
        "exact_pc_local_lstm_used": True,
        "learned_global_local_fusion_used": True,
        "separate_global_gate_used": True,
        "separate_count_head_used": True,
        "categorical_count_head_used": True,
        "positive_only_categorical_count_head_used": True,
        "count_regression_used": False,
        "log_count_used": False,
        "hurdle_head_used": True,
        "hurdle_training_objective": HURDLE_OBJECTIVE,
        "positive_count_training_objective": POSITIVE_COUNT_OBJECTIVE,
        "complete_training_objective": FULL_OBJECTIVE,
        "delta_training_objective": DELTA_OBJECTIVE,
        "loss_class_reweighting_used": False,
        "decode_prior_correction_used": False,
        "count_zero_is_implicit_hurdle": True,
        "positive_count_support_source_selection": "FIT_labels_only",
        "positive_count_support_source_final": "complete_TRAIN_labels_only",
        "count_support_is_dataset_derived": True,
        "count_support_is_normal_request_budget": False,
        "count_support_is_tuned_degree": False,
        "action_loss_scope": "all_58_bits_of_every_real_teacher_rank",
        "blocked_validation_length_source": BLOCKED_VALIDATION_LENGTH_SOURCE,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "fit_fraction": FIT_NUMERATOR / float(FIT_DENOMINATOR),
        "selection_protocol": {
            "fit": "first_80_percent_of_TRAIN",
            "validation": "last_20_percent_of_TRAIN",
            "selection_support": "FIT_only",
            "metric": "complete_validation_NLL_per_callback",
            "tie_break": "earlier_epoch",
        },
        "final_training_protocol": (
            "reset_seed_reinitialize_and_retrain_from_scratch_on_complete_"
            "TRAIN_for_selected_epoch_count"
        ),
        "original_guard_role": ORIGINAL_GUARD_ROLE,
        "runtime_encoding": "lossless_raw_pc64_plus_line58_only",
        "engineered_runtime_features": [],
        "training_runtime_fields": ["pc", "addr"],
        "inference_runtime_fields": ["pc", "addr"],
        "normal_policy_private_state": [
            "PC_indexed_stride_tracker_table", "last_stride", "confidence",
        ],
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "normal_policy_request_rate_used_as_budget": False,
        "normal_policy_constants_used_by_neural_inference": False,
        "delta_token_head_used": False,
        "delta_vocabulary_used": False,
        "delta_escape_head_used": False,
        "rank_delta_payload_head": "one_direct_58bit_modular_Bernoulli_head",
        "rank_delta_payload_bits": 58,
        "delta_decode_precision": "exact_all_58_modular_bits",
        "delta_bit_loss_scope": "all_58_bits_of_every_real_teacher_rank",
        "delta_bit_initialization": (
            "zero_weight_add_one_smoothed_partition_bit_marginal_logit_bias"
        ),
        "delta_bit_prior_source_selection": "all_real_FIT_teacher_actions",
        "delta_bit_prior_source_final": (
            "all_real_complete_TRAIN_teacher_actions"
        ),
        "full_modular_line_delta_range_reachable": True,
        "deterministic_target_uniqueness_constraint_used": True,
        "target_uniqueness_constraint_is_neural_action_feedback": False,
        "decoded_target_projection_or_mutation_used": False,
        "probability_threshold_used": False,
        "neural_degree_cap": None,
        "fixed_page_offset_classes": None,
        "same_page_rule_used_by_neural_inference": False,
        "normal_policy_templates_used_by_neural_inference": False,
        "future_label_window_used": False,
        "inference_policy_hardcodes_used": False,
        "threshold_related_hardcodes_used": False,
        "nn_generates_own_target_addresses": True,
        "weights_retrained": True,
        "checkpoint_reused": False,
        "training_history_reused": False,
        "decoder_only_change": False,
        "captured_candidate_files_role": (
            "normal replay, supervised labels, and audit only; never model input or budget"
        ),
        "model_input_excludes_action_outcomes": True,
        "tracks": {POLICY: {}},
    }
    for role in ROLES:
        stream_path = args.input_dir / "{}.{}.{}_stream.csv.gz".format(
            TRACE, POLICY, role
        )
        candidate_path = args.input_dir / "{}.{}.{}_candidates.csv.gz".format(
            TRACE, POLICY, role
        )
        if not stream_path.is_file() or not candidate_path.is_file():
            raise RuntimeError("missing normalized {} {} inputs".format(POLICY, role))
        stream_rows = read_stream(stream_path)
        (
            candidate_count, max_candidates, fill_counts, delta_profile,
        ) = read_candidates(
            candidate_path, POLICY, stream_rows
        )
        manifest["tracks"][POLICY][role] = {
            "demand_callbacks": len(stream_rows),
            "candidate_requests": candidate_count,
            "max_candidates_per_demand": max_candidates,
            "candidate_fill_level_counts": fill_counts,
            "teacher_delta_profile_for_output_design_audit": delta_profile,
            "demand_identity_sha256": identity_sha256(stream_rows),
            "stream_gzip_sha256": sha256(stream_path),
            "stream_content_sha256": gzip_content_sha256(stream_path),
            "candidate_gzip_sha256": sha256(candidate_path),
            "candidate_content_sha256": gzip_content_sha256(candidate_path),
        }

    manifest["maximum_complete_TRAIN_teacher_count"] = manifest["tracks"][
        POLICY
    ]["train"]["max_candidates_per_demand"]
    manifest["maximum_count_exposed_as_normal_request_budget"] = False

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")
    print("[PASS] {}".format(args.manifest_out))


if __name__ == "__main__":
    main()
