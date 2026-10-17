#!/usr/bin/env python3
"""Compare one completed frozen regression with its preserved frozen reference.

The original parser's 30 fields and all cache/service timelines must match.
The only permitted differences are retired training instrumentation (required
zero in the frozen reference and absent in the new run), newly initialized
zero counters, and the documented 56-to-24-byte event metadata simplification.
"""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REMOVED = set("""
available_positive_actions available_positive_decisions available_supervised_decisions
completed_updates max_tbptt_span peak_inflight_updates peak_padded_positions
peak_teacher_queue peak_training_queue prediction_weight_version_max publication_cycles
published_parameter_values published_versions sample_exposures teacher_calls
teacher_tag_comparison_bound training_admitted training_dropped training_forward_macs
training_modeled_macs training_modeled_nonlinears training_service_cycles model_version
training_queue update_inflight teacher_bytes_configured teacher_output_bytes_configured
teacher_output_bytes_peak retained_weight_bank_bytes_configured
retained_weight_bank_bytes_peak training_examples_bytes_configured publication_bytes_configured
""".split())
INITIALIZED = {"nn_started", "inference_scratch_traffic_bytes", "peak_decoded_addresses"}
LAYOUT = {"active_event_metadata_bytes", "input_queue_bytes_configured", "input_queue_bytes_peak"}
PROTOCOL = ("arm", "h", "hidden_size", "seed", "state_capacity", "macs", "protocol",
            "skip_records", "simulation_instructions", "warmup_instructions", "trace", "checkpoint")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_run(directory):
    meta = json.loads((directory / "run.json").read_text())
    require(meta.get("status") == "complete" and meta.get("exit_code") == 0,
            f"{directory}: run did not complete successfully")
    require(str(meta.get("arm", "")).startswith("frozen_"), f"{directory}: not a frozen arm")
    rows = [json.loads(line) for line in (directory / "snapshots.jsonl").read_text().splitlines() if line.strip()]
    final = [row for row in rows if row.get("event") == "snapshot" and row.get("final")]
    require(len(final) == 1, f"{directory}: expected one final snapshot")
    require(final[0]["instructions"] == meta["simulation_instructions"], f"{directory}: incomplete instruction scope")
    return meta, rows, final[0]


def trainer_key(key):
    return key in REMOVED or any(token in key.lower() for token in
        ("worker", "teacher", "trainer", "training", "optimizer", "gradient", "backward",
         "publication", "published", "label", "update", "tbptt"))


def check_no_trainer(value, location):
    if isinstance(value, dict):
        for key, child in value.items():
            require(not trainer_key(key), f"{location}: forbidden trainer field {key}")
            check_no_trainer(child, location + "." + key)
    elif isinstance(value, list):
        for child in value:
            check_no_trainer(child, location)


def compare(new_dir, reference_dir):
    new_meta, new_rows, final = read_run(new_dir)
    old_meta, old_rows, _ = read_run(reference_dir)
    for key in PROTOCOL:
        require(key in new_meta and new_meta[key] == old_meta.get(key), f"protocol mismatch: {key}")
    check_no_trainer(new_meta, "new run metadata")
    require(not list(new_dir.glob("worker*")), "new run contains worker artifacts")
    require(not any(token in str(new_meta.get("command", [])) for token in
                    ("online_worker", "--socket", "--training", "--optimizer")), "new command launches a trainer")
    spec = importlib.util.spec_from_file_location("original_602_parser", ROOT /
        "formal_NN_training/experiments/602_offline_lstm_stride/python/analyze_replay.py")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    counters = original.parse_log(new_dir / "run.log")
    reference = original.parse_log(reference_dir / "run.log")
    require(len(counters) == len(reference) == 30, "original parser field count changed")
    for key in counters:
        require(counters[key] == reference.get(key), f"original final counter mismatch: {key}")
    require(len(new_rows) == len(old_rows), "snapshot/cohort record count mismatch")
    compared = removed = initialized = layout = 0
    for index, (new, old) in enumerate(zip(new_rows, old_rows)):
        check_no_trainer(new, f"new record {index}")
        for key in old.keys() | new.keys():
            if key in REMOVED:
                require(key not in new and old[key] == 0, f"record {index}: nonzero removed field {key}")
                removed += 1
            elif key in LAYOUT:
                # Frozen-only event metadata keeps three uint64_t values. The old
                # structure also reserved teacher timing/action metadata. Queue
                # capacity and occupancy remain equal; only bytes per slot change.
                multiplier = 1 if key == "active_event_metadata_bytes" else (
                    16 if key == "input_queue_bytes_configured" else new["peak_input_queue"])
                require(new.get(key) == multiplier * 24 and old.get(key) == multiplier * 56,
                        f"record {index}: unexpected event/input layout {key}")
                layout += 1
            elif key not in old and (key == "stride_bytes_configured" or
                    key in INITIALIZED and new.get("instructions") == 0):
                require(new[key] == 0, f"record {index}: nonzero newly initialized {key}")
                initialized += 1
            else:
                require(key in new and key in old and new[key] == old[key],
                        f"record {index}: cache/service/cohort mismatch {key}: {old.get(key)} != {new.get(key)}")
                compared += 1
    basic = ("instructions", "cycles", "NL", "M", "P", "Ipf", "Q", "U", "L",
             "eligible_l2_callbacks", "nn_admitted", "nn_started", "nn_completed", "inference_dropped",
             "occupancy_lines", "fills_load", "fills_rfo", "fills_prefetch", "fills_writeback", "replacements")
    return {
        "status": "PASS",
        "scope": f"{final['instructions']}-instruction frozen regression; not a new final experiment",
        "protocol": {key: new_meta[key] for key in PROTOCOL},
        "new_run": str(new_dir), "reference_run": str(reference_dir),
        "host": {name: {key: meta[key] for key in ("simulator_wall_seconds", "simulator_peak_rss_kib")}
                 for name, meta in (("new", new_meta), ("reference", old_meta))},
        "checks": {"original_parser_fields_exact": len(counters), "timeline_records_exact": len(new_rows),
                   "snapshot_records": sum(row["event"] == "snapshot" for row in new_rows),
                   "cohort_records": sum(row["event"] == "cohort" for row in new_rows),
                   "cache_service_cohort_field_comparisons_exact": compared,
                   "removed_reference_fields_verified_zero": removed,
                   "newly_initialized_fields_verified_zero": initialized,
                   "event_input_layout_values_verified": layout,
                   "new_trainer_fields_absent": True, "new_worker_metadata_and_artifacts_absent": True},
        "layout_change": "Event metadata: reference 56 bytes, new 24 bytes; input queue remains 16 entries.",
        "counters": {key: final[key] for key in basic}, "ipc": final["instructions"] / final["cycles"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = compare(args.new_run.resolve(), args.reference_run.resolve())
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
