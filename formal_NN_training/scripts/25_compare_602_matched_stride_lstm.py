#!/usr/bin/env python3
"""Fail-closed summary for the live 602 stride/LSTM comparison.

Standard library only: no pandas and no dynamic import of another repository
script.  A fair-comparison claim is emitted only when the one-binary, fixed
window, model hash, live-execution, and runtime-input contracts all verify.
"""
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


TRACE = "602.gcc_s-734B"
KEY_VALUE = re.compile(r"^([A-Za-z0-9_]+)\s+([-+0-9.eE]+)\s*$")
FINISHED = re.compile(
    r"Finished CPU\s+0\s+instructions:\s+(\d+)\s+cycles:\s+(\d+)\s+cumulative IPC:\s+([-+0-9.eE]+)"
)
FAIL_PATTERNS = [
    "PREFETCHER_RUN_FAILED", "Segmentation fault", "Assertion", "runtime error", "ERROR:", "error:"
]
EXPECTED_INPUTS = [
    "current_l2_load_pc",
    "current_l2_load_cache_line_address",
    "causal_64_entry_lru_per_pc_previous_line",
    "causal_64_entry_lru_per_pc_previous_stride",
    "causal_callback_order_for_live_lstm_state",
]


def number(value, default=0.0):
    try:
        return float(value) if value not in (None, "") else float(default)
    except (TypeError, ValueError):
        return float(default)


def integer(value, default=0):
    try:
        return int(float(value)) if value not in (None, "") else int(default)
    except (TypeError, ValueError):
        return int(default)


def ratio(numerator, denominator):
    return float(numerator) / float(denominator) if float(denominator) else 0.0


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def existing_sha256(path):
    try:
        candidate = Path(path)
    except (TypeError, ValueError):
        return ""
    return sha256_file(candidate) if candidate.is_file() else ""


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def baseline_row(rows, method):
    matches = [
        row for row in rows
        if row.get("trace") == TRACE and row.get("prefetcher") == method
    ]
    if len(matches) != 1:
        raise ValueError("expected one {} row, got {}".format(method, len(matches)))
    row = dict(matches[0])
    row["method"] = method
    return row


def parse_lstm_log(path):
    stats, failure = {}, ""
    registry_marker = False
    input_marker = False
    with Path(path).open(errors="replace") as handle:
        for raw in handle:
            if "adding L2C_PREFETCHER: matched_stride_lstm (live inference)" in raw:
                registry_marker = True
            if "matched_stride_lstm_runtime_inputs pc,address,causal_pc_address_history" in raw:
                input_marker = True
            if not failure:
                for pattern in FAIL_PATTERNS:
                    if pattern in raw:
                        failure = raw[:200].strip()
                        break
            line = raw.strip()
            match = KEY_VALUE.match(line)
            if match:
                stats[match.group(1)] = number(match.group(2))
                continue
            match = FINISHED.search(line)
            if match:
                stats["finished_instructions"] = number(match.group(1))
                stats["finished_cycles"] = number(match.group(2))
                stats["finished_ipc"] = number(match.group(3))
    ipc = number(stats.get("Core_0_IPC")) or number(stats.get("finished_ipc"))
    instructions = integer(stats.get("Core_0_instructions")) or integer(stats.get("finished_instructions"))
    if not ipc or not instructions:
        failure = failure or "missing final IPC/instruction counters"
    return {
        "trace": TRACE,
        "method": "matched_lstm",
        "ipc": ipc,
        "instructions": instructions,
        "cycles": integer(stats.get("Core_0_cycles")) or integer(stats.get("finished_cycles")),
        "l2_loads": integer(stats.get("Core_0_L2C_loads")),
        "l2_load_miss": integer(stats.get("Core_0_L2C_load_miss")),
        "pf_requested": integer(stats.get("Core_0_L2C_prefetch_requested")),
        "pf_issued": integer(stats.get("Core_0_L2C_prefetch_issued")),
        "pf_useful": integer(stats.get("Core_0_L2C_prefetch_useful")),
        "pf_useless": integer(stats.get("Core_0_L2C_prefetch_useless")),
        "pf_late": integer(stats.get("Core_0_L2C_prefetch_late")),
        "matched_callbacks": integer(stats.get("matched_stride_lstm_callbacks")),
        "matched_candidates_scored": integer(stats.get("matched_stride_lstm_candidates_scored")),
        "matched_emitted": integer(stats.get("matched_stride_lstm_emitted")),
        "matched_tracker_evictions": integer(stats.get("matched_stride_lstm_tracker_evictions")),
        "registry_marker": int(registry_marker),
        "input_marker": int(input_marker),
        "run_failed": int(bool(failure)),
        "fail_reason": failure,
    }


def normalized_row(raw, role, matched, no_pref_ipc, stride_ipc, identity):
    ipc = number(raw.get("ipc"))
    issued = integer(raw.get("pf_issued"))
    useful = integer(raw.get("pf_useful"))
    late = integer(raw.get("pf_late"))
    return {
        "trace": TRACE,
        "method": raw.get("method", ""),
        "comparison_role": role,
        "matched_runtime_input": int(bool(matched)),
        "execution": "live_in_simulator",
        "ipc": ipc,
        "instructions": integer(raw.get("instructions")),
        "cycles": integer(raw.get("cycles")),
        "ipc_delta_vs_no_pref": ipc - no_pref_ipc,
        "ipc_delta_vs_stride": ipc - stride_ipc,
        "speedup_vs_no_pref": ratio(ipc, no_pref_ipc),
        "l2_loads": integer(raw.get("l2_loads")),
        "l2_load_miss": integer(raw.get("l2_load_miss")),
        "pf_requested": integer(raw.get("pf_requested")),
        "pf_issued": issued,
        "pf_useful": useful,
        "pf_useless": integer(raw.get("pf_useless")),
        "pf_late": late,
        "accuracy": ratio(useful, issued),
        "timeliness": ratio(useful, useful + late),
        "run_failed": integer(raw.get("run_failed")),
        "simulator_binary_sha256": identity["simulator_binary_sha256"],
        "warmup_instructions": identity["evaluation_window"]["warmup_instructions"],
        "simulation_instructions": identity["evaluation_window"]["simulation_instructions"],
    }


def write_csv(path, rows):
    fields = [
        "trace", "method", "comparison_role", "matched_runtime_input", "execution",
        "ipc", "instructions", "cycles", "ipc_delta_vs_no_pref", "ipc_delta_vs_stride", "speedup_vs_no_pref",
        "l2_loads", "l2_load_miss", "pf_requested", "pf_issued", "pf_useful",
        "pf_useless", "pf_late", "accuracy", "timeliness", "run_failed",
        "simulator_binary_sha256", "warmup_instructions", "simulation_instructions",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message, failures):
    if not condition:
        failures.append(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--lstm-log", required=True, type=Path)
    parser.add_argument("--training-metadata", required=True, type=Path)
    parser.add_argument("--run-identity", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.baseline_summary, args.lstm_log, args.training_metadata, args.run_identity):
        if not path.is_file():
            parser.error("missing {}".format(path))

    identity = json.loads(args.run_identity.read_text())
    training = json.loads(args.training_metadata.read_text())
    baseline_rows = read_csv(args.baseline_summary)
    no_pref = baseline_row(baseline_rows, "no_pref")
    stride = baseline_row(baseline_rows, "stride")
    lstm = parse_lstm_log(args.lstm_log)
    no_pref_log = parse_lstm_log(no_pref["log"])
    stride_log = parse_lstm_log(stride["log"])
    for row, parsed in ((no_pref, no_pref_log), (stride, stride_log)):
        row.update({
            "instructions": parsed["instructions"],
            "cycles": parsed["cycles"],
        })
    failures = []

    require(identity.get("schema") == "602_matched_stride_lstm_run_identity_v3", "unexpected identity schema", failures)
    require(identity.get("trace") == TRACE, "identity is not 602", failures)
    require(identity.get("champsim_head") == identity.get("expected_champsim_head"), "ChampSim HEAD is not pinned", failures)
    require(identity.get("libbf_head") == identity.get("expected_libbf_head"), "libbf HEAD is not pinned", failures)
    require(identity.get("same_binary_methods") == ["no_pref", "stride", "matched_stride_lstm"], "same-binary method set changed", failures)
    require(identity.get("primary_nn_execution") == "live_in_simulator_not_keyed_replay", "primary NN is not live", failures)
    require(identity.get("baseline_config", {}).get("stride_num_trackers") == 64, "stride trackers are not 64", failures)
    require(identity.get("baseline_config", {}).get("stride_pref_degree") == 2, "stride degree is not 2", failures)
    require(identity.get("training_window") == {"warmup_instructions": 0, "simulation_instructions": 20000000}, "training window changed", failures)
    require(identity.get("unseen_guard_before_measurement_instructions") == 5000000, "unseen pre-measurement guard changed", failures)
    require(identity.get("evaluation_window") == {"warmup_instructions": 25000000, "simulation_instructions": 25000000}, "evaluation window changed", failures)
    require(training.get("trace") == TRACE, "training metadata is not 602", failures)
    require(training.get("model_family") == "LSTM", "model family is not LSTM", failures)
    require(training.get("parameter_count") == 545, "parameter count is not 545", failures)
    require(training.get("candidate_slots") == 1, "LSTM has more than the single stride candidate", failures)
    require(training.get("runtime_inputs") == EXPECTED_INPUTS, "runtime input contract changed", failures)
    require(training.get("live_in_simulator_inference") is True, "training metadata does not require live inference", failures)
    require(training.get("keyed_offline_replay_used_for_primary_result") is False, "keyed replay was used as primary", failures)
    require(training.get("evaluation_data_used_for_training_or_policy") is False, "evaluation leaked into training/policy", failures)
    require(training.get("pandas_dependency") == "none", "training depends on pandas", failures)
    require(training.get("training_columns_consumed") == ["trace", "demand_idx", "pc", "line"], "training consumed fields beyond PC/address sequence identity", failures)
    require(training.get("train_stream_sha256") == identity.get("train_stream_sha256"), "training-stream hash mismatch", failures)
    require(training.get("runtime_model_sha256") == identity.get("runtime_model_sha256"), "runtime-model hash mismatch", failures)
    require(existing_sha256(identity.get("trace_file")) == identity.get("trace_sha256"), "trace file changed after evaluation", failures)
    require(existing_sha256(identity.get("simulator_binary")) == identity.get("simulator_binary_sha256"), "simulator binary changed after evaluation", failures)
    require(existing_sha256(identity.get("baseline_config_path")) == identity.get("baseline_config_sha256"), "stride config changed after evaluation", failures)
    require(existing_sha256(identity.get("train_stream")) == identity.get("train_stream_sha256"), "training stream changed after evaluation", failures)
    require(existing_sha256(identity.get("runtime_model")) == identity.get("runtime_model_sha256"), "runtime-model file changed after evaluation", failures)
    require(sha256_file(args.training_metadata) == identity.get("training_metadata_sha256"), "training metadata changed after evaluation", failures)
    require(number(training.get("export_math_max_abs_error"), 1.0) <= 1e-5, "PyTorch/export math parity failed", failures)
    require(integer(no_pref.get("run_failed")) == 0, "no-prefetch run failed", failures)
    require(integer(stride.get("run_failed")) == 0, "stride run failed", failures)
    require(no_pref_log.get("run_failed") == 0, "no-prefetch source log failed", failures)
    require(stride_log.get("run_failed") == 0, "stride source log failed", failures)
    require(lstm.get("run_failed") == 0, "LSTM run failed: {}".format(lstm.get("fail_reason", "")), failures)
    require(lstm.get("registry_marker") == 1, "live LSTM registry marker absent", failures)
    require(lstm.get("input_marker") == 1, "live LSTM input marker absent", failures)
    require(lstm.get("matched_callbacks", 0) > 0, "live LSTM processed no callbacks", failures)
    require(lstm.get("matched_candidates_scored", 0) > 0, "live LSTM scored no candidates", failures)

    no_pref_ipc = number(no_pref.get("ipc"))
    stride_ipc = number(stride.get("ipc"))
    lstm_ipc = number(lstm.get("ipc"))
    require(abs(no_pref_ipc - number(no_pref_log.get("ipc"))) <= 1e-8, "no-prefetch CSV/log IPC mismatch", failures)
    require(abs(stride_ipc - number(stride_log.get("ipc"))) <= 1e-8, "stride CSV/log IPC mismatch", failures)
    instruction_counts = {
        integer(no_pref_log.get("instructions")),
        integer(stride_log.get("instructions")),
        integer(lstm.get("instructions")),
    }
    require(0 not in instruction_counts and len(instruction_counts) == 1, "same-window instruction counts differ", failures)
    require(no_pref_ipc > 0 and stride_ipc > 0 and lstm_ipc > 0, "one or more IPC values are missing", failures)
    if failures:
        payload = {"status": "FAIL", "fair_comparison_claim_allowed": False, "failures": failures}
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        raise SystemExit("[verification FAIL] " + " | ".join(failures))

    rows = [
        normalized_row(no_pref, "same_binary_no_pref_control", False, no_pref_ipc, stride_ipc, identity),
        normalized_row(stride, "matched_input_traditional_baseline", True, no_pref_ipc, stride_ipc, identity),
        normalized_row(lstm, "matched_input_lstm", True, no_pref_ipc, stride_ipc, identity),
    ]
    write_csv(args.out_csv, rows)
    payload = {
        "status": "PASS",
        "trace": TRACE,
        "research_question": "Can a 545-parameter live LSTM improve held-out 602 IPC over stride when both receive only the same live PC/address callback stream?",
        "fair_comparison_claim_allowed": True,
        "matched_methods": ["stride", "matched_lstm"],
        "runtime_input_contract": training["input_contract"],
        "execution": "live_in_simulator_for_both_methods",
        "training_window": identity["training_window"],
        "evaluation_window": identity["evaluation_window"],
        "simulator_binary_sha256": identity["simulator_binary_sha256"],
        "stride_ipc": stride_ipc,
        "matched_lstm_ipc": lstm_ipc,
        "ipc_delta_lstm_minus_stride": lstm_ipc - stride_ipc,
        "lstm_beats_stride": bool(lstm_ipc > stride_ipc),
        "live_lstm_counters": {
            "callbacks": lstm["matched_callbacks"],
            "candidates_scored": lstm["matched_candidates_scored"],
            "emitted": lstm["matched_emitted"],
            "tracker_evictions": lstm["matched_tracker_evictions"],
        },
        "scope_note": "Only stride/LSTM is a matched-input comparison. The other normal-prefetcher IPCs remain general performance references.",
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("[verification PASS] stride IPC={:.6f} matched-LSTM IPC={:.6f} delta={:+.6f}".format(
        stride_ipc, lstm_ipc, lstm_ipc - stride_ipc
    ))
    print("[write] {}".format(args.out_csv))
    print("[write] {}".format(args.out_json))


if __name__ == "__main__":
    main()
