#!/usr/bin/env python3
"""Fail-closed validation for the direct 623 SPP I/O student track."""
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
LOGGER_SCHEMA = "623_causal_trigger_v5"
ATTACHMENT_MODE = "explicit_trigger_event_id"
EXPERIMENT_REVISION = "spp_direct_io_sliding_cnn_v4_independent_utility"
PAGE_LINES = 64
MAX_ACTIONS = 32
CANONICALIZATION_MODE = "per_target_min_fill_queue_effect"


def as_int(value):
    text = str(value).strip()
    return int(text, 16) if text.lower().startswith("0x") else int(float(text))


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
    rows = []
    occurrences = defaultdict(int)
    last_cycle = -1
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "trace", "demand_idx", "cycle", "pc", "address", "line",
            "cache_hit", "access_type", "pc_line_occ", "logger_schema",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError("{} missing stream columns {}".format(path, sorted(missing)))
        for index, row in enumerate(reader):
            cycle = as_int(row["cycle"])
            pc = as_int(row["pc"])
            address = as_int(row["address"])
            line = as_int(row["line"])
            hit = as_int(row["cache_hit"])
            access_type = as_int(row["access_type"])
            occurrence = as_int(row["pc_line_occ"])
            expected_occurrence = occurrences[(pc, line)]
            occurrences[(pc, line)] += 1
            if row["trace"] != TRACE or as_int(row["demand_idx"]) != index:
                raise RuntimeError("{} identity/order failure at {}".format(path, index))
            if row["logger_schema"] != LOGGER_SCHEMA:
                raise RuntimeError("{} contains stale logger schema".format(path))
            if cycle < last_cycle:
                raise RuntimeError("{} cycle order regressed at {}".format(path, index))
            if address != line << 6:
                raise RuntimeError("{} address is not canonical line-aligned addr".format(path))
            if hit not in (0, 1) or access_type < 0:
                raise RuntimeError("{} invalid callback audit fields".format(path))
            if occurrence != expected_occurrence:
                raise RuntimeError("{} occurrence mismatch at {}".format(path, index))
            rows.append((index, pc, address, line, occurrence))
            last_cycle = cycle
    if not rows:
        raise RuntimeError("empty demand stream {}".format(path))
    return rows


def read_teacher_actions(path, stream_rows):
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
            raise RuntimeError("{} missing action columns {}".format(path, sorted(missing)))
        for row in reader:
            demand_idx = as_int(row["demand_idx"])
            if demand_idx < 0 or demand_idx >= len(stream_rows):
                raise RuntimeError("{} action demand_idx out of range".format(path))
            index, pc, _, line, occurrence = stream_rows[demand_idx]
            identity = (
                demand_idx, as_int(row["pc"]), as_int(row["line"]),
                as_int(row["pc_line_occ"]),
            )
            if identity != (index, pc, line, occurrence):
                raise RuntimeError("{} action transport identity mismatch".format(path))
            if row["trace"] != TRACE or row["policy"] != POLICY:
                raise RuntimeError("{} trace/policy mismatch".format(path))
            if row["logger_schema"] != LOGGER_SCHEMA or row["match_mode"] != ATTACHMENT_MODE:
                raise RuntimeError("{} stale/noncausal action attachment".format(path))
            counts[demand_idx] += 1
            if as_int(row["action_rank"]) != counts[demand_idx]:
                raise RuntimeError("{} action ranks are not contiguous".format(path))
            if counts[demand_idx] > MAX_ACTIONS:
                raise RuntimeError("{} exceeds source action bound {}".format(path, MAX_ACTIONS))

            trigger = as_int(row["trigger_event_id"])
            pf_event = as_int(row["pf_event_id"])
            distance = as_int(row["event_distance"])
            if pf_event <= last_pf_event or trigger >= pf_event or distance != pf_event - trigger:
                raise RuntimeError("{} invalid explicit trigger ordering".format(path))
            if distance < 1 or distance > 256:
                raise RuntimeError("{} trigger distance outside bound".format(path))
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
                raise RuntimeError("{} has two canonical actions for one target".format(path))
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
        "source_decision_effective_external_input": ["addr"],
        "model_input_is_causal_address_sequence_only": True,
        "teacher_actions_are_model_inputs": False,
        "same_external_input_contract": True,
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_candidate_bank_is_fixed": False,
        "nn_can_generate_actions_not_emitted_by_teacher": True,
        "model_does_not_use_pc": True,
        "cache_hit_and_type_are_audit_only": True,
        "direct_action_classes": 128,
        "maximum_actions_per_callback": MAX_ACTIONS,
        "self_target_actions_allowed": True,
        "teacher_action_canonicalization": CANONICALIZATION_MODE,
        "tracks": {POLICY: {}},
    }
    if args.source_contract is not None:
        contract = json.loads(args.source_contract.read_text())
        if contract.get("decision_effective_external_input") != ["addr"]:
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
        stream_rows = read_stream(stream_path)
        action_stats = read_teacher_actions(
            action_path, stream_rows
        )
        manifest["tracks"][POLICY][role] = {
            "demand_callbacks": len(stream_rows),
            **action_stats,
            "demand_identity_sha256": identity_sha256(stream_rows),
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
