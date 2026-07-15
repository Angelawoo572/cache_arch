#!/usr/bin/env python3
"""Fail-closed validation for the strict-input direct 623 SPP track."""
import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path


TRACE = "623.xalancbmk_s-700B"
POLICY = "spp"
ROLES = ("train", "guard", "eval")
LOGGER_SCHEMA = "623_causal_trigger_fill_v6"
ATTACHMENT_MODE = "explicit_trigger_event_id"
EXPERIMENT_REVISION = "spp_threshold_free_fill_feedback_split_v9"
PAGE_LINES = 64
CANONICALIZATION_MODE = "per_target_min_fill_queue_effect"
SOURCE_INPUTS = [
    "callback_kind", "invoke_prefetcher.addr", "cache_fill.evicted_addr",
]


def as_int(value):
    text = str(value).strip()
    try:
        return int(text, 0)
    except ValueError:
        return int(float(text))


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
        digest.update((",".join(str(value) for value in row) + "\n").encode())
    return digest.hexdigest()


def read_stream(path):
    events = []
    demands = []
    occurrences = defaultdict(int)
    last_cycle = -1
    last_raw_event_id = -1
    kind_counts = {"DEMAND": 0, "FILL": 0}
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "trace", "event_idx", "raw_event_id", "cycle", "event_kind",
            "event_address", "event_line", "decision_idx", "pc",
            "cache_hit", "access_type", "pc_line_occ", "logger_schema",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                "{} missing stream columns {}".format(path, sorted(missing))
            )
        for index, row in enumerate(reader):
            raw_event_id = as_int(row["raw_event_id"])
            cycle = as_int(row["cycle"])
            kind = row["event_kind"]
            address = as_int(row["event_address"])
            line = as_int(row["event_line"])
            decision_idx = as_int(row["decision_idx"])
            pc = as_int(row["pc"])
            hit = as_int(row["cache_hit"])
            access_type = as_int(row["access_type"])
            occurrence = as_int(row["pc_line_occ"])
            if (
                row["trace"] != TRACE
                or as_int(row["event_idx"]) != index
                or row["logger_schema"] != LOGGER_SCHEMA
            ):
                raise RuntimeError(
                    "{} identity/order/schema failure at {}".format(path, index)
                )
            if raw_event_id <= last_raw_event_id:
                raise RuntimeError(
                    "{} raw event order regressed at {}".format(path, index)
                )
            if cycle < last_cycle:
                raise RuntimeError(
                    "{} cycle order regressed at {}".format(path, index)
                )
            if address != line << 6:
                raise RuntimeError(
                    "{} event address is not canonical line-aligned addr".format(path)
                )
            if access_type < 0 or kind not in kind_counts:
                raise RuntimeError("{} invalid callback kind/type".format(path))

            if kind == "DEMAND":
                expected_occurrence = occurrences[(pc, line)]
                occurrences[(pc, line)] += 1
                if (
                    decision_idx != len(demands)
                    or hit not in (0, 1)
                    or occurrence != expected_occurrence
                ):
                    raise RuntimeError(
                        "{} demand identity failure at {}".format(path, index)
                    )
                demands.append((decision_idx, pc, address, line, occurrence))
            else:
                if (
                    decision_idx != -1 or pc != 0 or hit != 0
                    or occurrence != -1
                ):
                    raise RuntimeError(
                        "{} cache-fill context row leaks transport state".format(path)
                    )
            events.append((index, raw_event_id, cycle, kind, address, decision_idx))
            kind_counts[kind] += 1
            last_cycle = cycle
            last_raw_event_id = raw_event_id
    if not events or not demands:
        raise RuntimeError("empty event/demand stream {}".format(path))
    if not kind_counts["FILL"]:
        raise RuntimeError("{} contains no SPP cache-fill feedback".format(path))
    return {
        "events": events,
        "demands": demands,
        "kind_counts": kind_counts,
    }


def read_teacher_actions(path, demand_rows):
    counts = defaultdict(int)
    seen = defaultdict(set)
    fill_counts = {"FILL_L2": 0, "FILL_LLC": 0}
    last_pf_event = -1
    total = 0
    raw_total = 0
    self_target_total = 0
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "trace", "policy", "demand_idx", "pc", "line", "pc_line_occ",
            "action_rank", "pf_line", "target_page_offset", "fill_level",
            "accepted", "duplicate", "trigger_event_id", "pf_event_id",
            "event_distance", "raw_action_count",
            "source_first_pf_event_id", "source_last_pf_event_id",
            "is_self_target", "canonicalization", "match_mode",
            "logger_schema",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                "{} missing action columns {}".format(path, sorted(missing))
            )
        for row in reader:
            demand_idx = as_int(row["demand_idx"])
            if demand_idx < 0 or demand_idx >= len(demand_rows):
                raise RuntimeError("{} action demand_idx out of range".format(path))
            index, pc, _, line, occurrence = demand_rows[demand_idx]
            identity = (
                demand_idx, as_int(row["pc"]), as_int(row["line"]),
                as_int(row["pc_line_occ"]),
            )
            if identity != (index, pc, line, occurrence):
                raise RuntimeError("{} action transport identity mismatch".format(path))
            if row["trace"] != TRACE or row["policy"] != POLICY:
                raise RuntimeError("{} trace/policy mismatch".format(path))
            if (
                row["logger_schema"] != LOGGER_SCHEMA
                or row["match_mode"] != ATTACHMENT_MODE
            ):
                raise RuntimeError("{} stale/noncausal action attachment".format(path))
            counts[demand_idx] += 1
            if as_int(row["action_rank"]) != counts[demand_idx]:
                raise RuntimeError("{} action ranks are not contiguous".format(path))
            if counts[demand_idx] > PAGE_LINES:
                raise RuntimeError(
                    "{} exceeds the complete page action space".format(path)
                )

            trigger = as_int(row["trigger_event_id"])
            pf_event = as_int(row["pf_event_id"])
            distance = as_int(row["event_distance"])
            if (
                pf_event <= last_pf_event or trigger >= pf_event
                or distance != pf_event - trigger or distance < 1
            ):
                raise RuntimeError("{} invalid explicit trigger ordering".format(path))
            pf_line = as_int(row["pf_line"])
            offset = as_int(row["target_page_offset"])
            fill = as_int(row["fill_level"])
            accepted = as_int(row["accepted"])
            duplicate = as_int(row["duplicate"])
            raw_action_count = as_int(row["raw_action_count"])
            source_first = as_int(row["source_first_pf_event_id"])
            source_last = as_int(row["source_last_pf_event_id"])
            is_self_target = as_int(row["is_self_target"])
            if offset != pf_line % PAGE_LINES or offset < 0 or offset >= PAGE_LINES:
                raise RuntimeError("{} target offset mismatch".format(path))
            if pf_line // PAGE_LINES != line // PAGE_LINES:
                raise RuntimeError("{} cross-page SPP action".format(path))
            if fill not in (2, 4):
                raise RuntimeError("{} invalid fill level".format(path))
            if accepted != 1 or duplicate not in (0, 1):
                raise RuntimeError("{} incomplete/invalid teacher action".format(path))
            if (
                raw_action_count < 1
                or source_first != pf_event
                or source_last < source_first
                or is_self_target != int(pf_line == line)
                or row["canonicalization"] != CANONICALIZATION_MODE
            ):
                raise RuntimeError(
                    "{} invalid queue-effect canonicalization".format(path)
                )
            if pf_line in seen[demand_idx]:
                raise RuntimeError(
                    "{} has two canonical actions for one target".format(path)
                )
            seen[demand_idx].add(pf_line)
            fill_counts["FILL_L2" if fill == 2 else "FILL_LLC"] += 1
            total += 1
            raw_total += raw_action_count
            self_target_total += is_self_target
            last_pf_event = pf_event
    if total == 0:
        raise RuntimeError("empty teacher action stream {}".format(path))
    return {
        "teacher_actions": total,
        "raw_source_prefetch_calls": raw_total,
        "collapsed_source_calls": raw_total - total,
        "self_target_actions": self_target_total,
        "self_target_action_rate": self_target_total / float(total),
        "max_actions_per_callback": max(counts.values()),
        "teacher_fill_level_counts": fill_counts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--manifest-out", required=True, type=Path)
    parser.add_argument("--source-contract", type=Path)
    args = parser.parse_args()

    manifest = {
        "status": "PASS",
        "trace": TRACE,
        "policy": POLICY,
        "experiment_revision": EXPERIMENT_REVISION,
        "event_logger_schema": LOGGER_SCHEMA,
        "action_attachment_mode": ATTACHMENT_MODE,
        "neural_role": "standalone_direct_action_prefetcher",
        "source_decision_effective_external_input": SOURCE_INPUTS,
        "model_input_is_causal_external_event_sequence_only": True,
        "cache_fill_feedback_used_as_raw_external_input": True,
        "cache_fill_private_state_used_as_model_input": False,
        "teacher_actions_are_model_inputs": False,
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
        "fill_lead_cutoff_used": False,
        "inference_policy_hardcodes_used": False,
        "threshold_related_hardcodes_used": False,
        "normal_candidate_bank_is_fixed": False,
        "nn_can_generate_actions_not_emitted_by_teacher": True,
        "model_does_not_use_pc": True,
        "cache_hit_and_type_are_audit_only": True,
        "action_count_classes": PAGE_LINES + 1,
        "target_offset_classes": PAGE_LINES,
        "fill_classes": ["FILL_L2", "FILL_LLC"],
        "complete_neural_action_space": True,
        "self_target_actions_allowed": True,
        "teacher_action_canonicalization": CANONICALIZATION_MODE,
        "tracks": {POLICY: {}},
    }
    if args.source_contract is not None:
        contract = json.loads(args.source_contract.read_text())
        if contract.get("decision_effective_external_input") != SOURCE_INPUTS:
            raise RuntimeError("SPP source contract has unexpected external input")
        manifest["spp_source_contract_sha256"] = sha256(args.source_contract)

    for role in ROLES:
        stream_path = args.input_dir / "{}.{}.{}_stream.csv.gz".format(
            TRACE, POLICY, role
        )
        action_path = args.input_dir / "{}.{}.{}_teacher_actions.csv.gz".format(
            TRACE, POLICY, role
        )
        if not stream_path.is_file() or not action_path.is_file():
            raise RuntimeError("missing normalized SPP {} inputs".format(role))
        stream = read_stream(stream_path)
        action_stats = read_teacher_actions(action_path, stream["demands"])
        manifest["tracks"][POLICY][role] = {
            "external_context_events": len(stream["events"]),
            "demand_callbacks": stream["kind_counts"]["DEMAND"],
            "cache_fill_callbacks": stream["kind_counts"]["FILL"],
            **action_stats,
            "external_event_identity_sha256": identity_sha256(stream["events"]),
            "demand_identity_sha256": identity_sha256(stream["demands"]),
            "stream_gzip_sha256": sha256(stream_path),
            "stream_content_sha256": gzip_content_sha256(stream_path),
            "teacher_actions_gzip_sha256": sha256(action_path),
            "teacher_actions_content_sha256": gzip_content_sha256(action_path),
        }

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")
    print("[PASS] {}".format(args.manifest_out))


if __name__ == "__main__":
    main()
