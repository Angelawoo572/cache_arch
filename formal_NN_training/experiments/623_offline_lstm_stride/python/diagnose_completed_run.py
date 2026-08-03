#!/usr/bin/env python3
"""Build a fail-closed diagnosis of a completed 623 Stride run.

This script reads existing analyzer and Colab metadata only.  It never changes
training, replay lists, or simulator outputs.  The output keeps request-count,
target-quality, and cache-lifecycle evidence separate so a PASS contract is not
mistaken for a successful neural policy.
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


POLICY = "stride"
TRACE = "623.xalancbmk_s-700B"
DEFAULT_RUN_ID = "623_offline_lstm_stride_keyed_crn_v15_seed7"
SOURCE_INPUTS = ["pc", "addr"]
NEURAL_METHOD_PREFIX = "offline_independent_delta_stride_lstm_"
EXPECTED_TAGS = {
    "independent_delta_stride_lstm_h{}".format(size)
    for size in (8, 16, 32, 64, 128)
}
EXPERIMENT = Path(__file__).resolve().parents[1]
REPOSITORY = Path(__file__).resolve().parents[4]
STRIDE_SOURCE = REPOSITORY / "external" / "ChampSim" / "prefetcher" / "stride.cc"


def div(numerator, denominator):
    return (
        float(numerator) / float(denominator)
        if denominator not in (None, 0) else None
    )


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def flatten(prefix, value, output):
    if isinstance(value, dict):
        for key in sorted(value):
            child = "{}.{}".format(prefix, key) if prefix else str(key)
            flatten(child, value[key], output)
    elif isinstance(value, (list, tuple)):
        output[prefix] = json.dumps(value, sort_keys=True)
    elif value is None:
        output[prefix] = ""
    else:
        output[prefix] = value


def input_contract_mismatches(metadata):
    expected = {
        "same_external_input_contract": True,
        "training_inference_input_encoder_identical": True,
        "training_runtime_fields": SOURCE_INPUTS,
        "inference_runtime_fields": SOURCE_INPUTS,
        "normal_policy_outputs_used_as_model_inputs": False,
        "normal_policy_candidates_used_as_model_inputs": False,
        "normal_policy_private_state_used_as_model_inputs": False,
        "normal_policy_outputs_used_as_training_targets": True,
        "future_label_window_used": False,
    }
    mismatches = []
    for key, value in expected.items():
        if metadata.get(key) != value:
            mismatches.append({
                "field": key,
                "actual": metadata.get(key),
                "expected": value,
            })
    encoder_hashes = {
        metadata.get("runtime_encoder_sha256"),
        metadata.get("training_runtime_encoder_sha256"),
        metadata.get("inference_runtime_encoder_sha256"),
    }
    if (
        len(encoder_hashes) != 1
        or not isinstance(next(iter(encoder_hashes)), str)
        or len(next(iter(encoder_hashes))) != 64
    ):
        mismatches.append({
            "field": "runtime_encoder_sha256",
            "actual": sorted(str(value) for value in encoder_hashes),
            "expected": "one identical 64-hex hash",
        })
    return mismatches


def analyzer_evidence_mismatches(matched, tag, metadata):
    """Bind current metadata to hashes stored by the completed analyzer."""
    mismatches = []
    method = "offline_" + tag
    accounting = matched.get("replay_accounting") or {}
    nn_info = accounting.get(method) or {}
    normal_info = accounting.get("offline_stride") or {}
    checks = (
        ("runtime_encoder_sha256", metadata.get("runtime_encoder_sha256"),
         (matched.get("runtime_encoder_sha256_by_model_tag") or {}).get(tag)),
        ("normal_list_sha256", metadata.get("normal_list_sha256"),
         (matched.get("offline_normal_list_hashes_by_model_tag") or {}).get(tag)),
        ("normal_list_sha256_vs_replay", metadata.get("normal_list_sha256"),
         normal_info.get("sha256")),
        ("offline_normal_entries", metadata.get("offline_normal_entries"),
         normal_info.get("entries")),
        ("nn_list_sha256", metadata.get("nn_list_sha256"),
         nn_info.get("sha256")),
        ("offline_nn_entries", metadata.get("offline_nn_entries"),
         nn_info.get("entries")),
    )
    for field, actual, expected in checks:
        if expected is None or actual != expected:
            mismatches.append({
                "field": field, "actual": actual, "expected": expected,
            })
    policy_inputs = (
        (matched.get("input_provenance") or {})
        .get("policy_inputs", {}).get(POLICY, {})
    )
    for role in ("train", "guard", "eval"):
        for kind in ("stream", "candidate"):
            key = "{}_{}_content_sha256".format(role, kind)
            expected = policy_inputs.get(role, {}).get(kind, {}).get(
                "content_sha256"
            )
            if expected is None or metadata.get(key) != expected:
                mismatches.append({
                    "field": key,
                    "actual": metadata.get(key),
                    "expected": expected,
                })
    return mismatches


def audit_stride_source(path):
    """Prove that the live Stride decision uses only PC and address.

    cache_hit and type are part of ChampSim's generic prefetcher signature, but
    the pinned Stride implementation must not read them in the function body.
    This complements the metadata/encoder checks with an audit of the teacher's
    actual source-visible input.
    """
    if not path.is_file():
        raise SystemExit("missing Stride source for input audit: {}".format(path))
    raw = path.read_bytes()
    source = raw.decode("utf-8")
    marker = "void StridePrefetcher::invoke_prefetcher"
    next_marker = "uint32_t StridePrefetcher::generate_prefetch"
    start = source.find(marker)
    end = source.find(next_marker, start + len(marker))
    if start < 0 or end < 0:
        raise SystemExit("cannot isolate Stride invoke_prefetcher in {}".format(path))
    function = source[start:end]
    brace = function.find("{")
    if brace < 0:
        raise SystemExit("missing Stride invoke_prefetcher body in {}".format(path))
    signature = function[:brace]
    body = function[brace + 1:]
    expected_signature_fields = ("pc", "address", "cache_hit", "type")
    missing = [
        name for name in expected_signature_fields
        if re.search(r"\b{}\b".format(name), signature) is None
    ]
    used = {
        name: len(re.findall(r"\b{}\b".format(name), body))
        for name in expected_signature_fields
    }
    if missing or used["pc"] == 0 or used["address"] == 0:
        raise SystemExit(
            "Stride source input audit failed: missing={} body_uses={}".format(
                missing, used
            )
        )
    if used["cache_hit"] != 0 or used["type"] != 0:
        raise SystemExit(
            "Stride source uses excluded generic inputs: {}".format(used)
        )
    return {
        "status": "PASS",
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "invoke_prefetcher_signature_fields": list(expected_signature_fields),
        "invoke_prefetcher_body_reference_counts": used,
        "effective_source_inputs": ["pc", "address"],
        "signature_only_inputs": ["cache_hit", "type"],
        "provenance_scope": (
            "current checkout only; the completed run did not record this "
            "source blob SHA"
        ),
    }


def model_record(row, metadata, normal, no_pref, matched):
    triggers = row.get("runtime_reachable_list_triggers")
    normal_triggers = normal.get("runtime_reachable_list_triggers")
    record = {
        "method": row.get("method"),
        "model_tag": metadata.get("model_tag"),
        "model_size": metadata.get("model_size"),
        "parameter_count": metadata.get("parameter_count"),
        "ipc": row.get("ipc"),
        "ipc_delta_vs_offline_normal": (
            row.get("ipc") - normal.get("ipc")
        ),
        "ipc_delta_vs_no_pref": row.get("ipc") - no_pref.get("ipc"),
        "l2_load_miss_rate": row.get("l2_load_miss_rate"),
        "l2_miss_rate_delta_vs_offline_normal": (
            row.get("l2_load_miss_rate")
            - normal.get("l2_load_miss_rate")
        ),
        "pf_requested": row.get("pf_requested"),
        "request_ratio_vs_offline_normal": div(
            row.get("pf_requested"), normal.get("pf_requested")
        ),
        "runtime_reachable_list_triggers": triggers,
        "trigger_ratio_vs_offline_normal": div(
            triggers, normal_triggers
        ),
        "actions_per_reached_trigger": div(
            row.get("pf_requested"), triggers
        ),
        "offline_normal_actions_per_reached_trigger": div(
            normal.get("pf_requested"), normal_triggers
        ),
        "pf_filled": row.get("pf_filled"),
        "pf_useful": row.get("pf_useful"),
        "pf_useless": row.get("pf_useless"),
        "pf_late": row.get("pf_late"),
        "pq_merged_duplicate_proxy": row.get(
            "pq_merged_duplicate_proxy"
        ),
        "selected_accuracy": row.get("selected_accuracy"),
        "coverage_vs_no_pref_l2_miss": row.get(
            "coverage_vs_no_pref_l2_miss"
        ),
        "timeliness": row.get("timeliness"),
        "runtime_unreached_list_entries": row.get(
            "runtime_unreached_list_entries"
        ),
        "runtime_unreached_list_triggers": row.get(
            "runtime_unreached_list_triggers"
        ),
    }
    diagnostic_keys = (
        "request_count_training_label_statistics",
        "request_count_decoder_diagnostics",
        "heldout_behavior_metrics",
        "train_action_summary",
        "guard_action_summary",
        "eval_action_summary",
    )
    for key in diagnostic_keys:
        if key in metadata:
            record[key] = metadata[key]
    history = metadata.get("train_history") or []
    if history:
        record["last_training_epoch"] = history[-1]
    mismatches = input_contract_mismatches(metadata)
    record["input_contract_verified"] = not mismatches
    record["input_contract_mismatches"] = mismatches
    evidence_mismatches = analyzer_evidence_mismatches(
        matched, metadata.get("model_tag"), metadata
    )
    record["analyzer_evidence_verified"] = not evidence_mismatches
    record["analyzer_evidence_mismatches"] = evidence_mismatches
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--stride-source", type=Path, default=STRIDE_SOURCE)
    args = parser.parse_args()
    run_dir = (
        args.run_dir
        if args.run_dir is not None
        else EXPERIMENT / "runs" / args.run_id
    )

    matched_path = run_dir / "matched_comparison.json"
    if not matched_path.is_file():
        raise SystemExit("missing {}".format(matched_path))
    matched = read_json(matched_path)
    if matched.get("status") != "PASS" or matched.get("failures"):
        raise SystemExit(
            "source matched comparison is not a root PASS with no failures"
        )
    if matched.get("trace") != TRACE:
        raise SystemExit("unexpected trace in {}".format(matched_path))

    rows = {
        row.get("method"): row for row in matched.get("rows", [])
    }
    no_pref = rows.get("no_pref")
    normal = rows.get("offline_" + POLICY)
    if no_pref is None or normal is None:
        raise SystemExit("missing no-prefetch or offline-normal row")

    expected_tags = {
        method[len("offline_"):]
        for method in rows
        if isinstance(method, str)
        and method.startswith(NEURAL_METHOD_PREFIX)
    }
    if not expected_tags:
        raise SystemExit("matched comparison contains no Stride neural rows")
    if expected_tags != EXPECTED_TAGS:
        raise SystemExit(
            "unexpected Stride v15 model set: observed={} expected={}".format(
                sorted(expected_tags), sorted(EXPECTED_TAGS)
            )
        )

    records = []
    selected_metadata = []
    metadata_root = run_dir / "colab_output"
    for metadata_path in sorted(metadata_root.glob("*/run_metadata.json")):
        metadata = read_json(metadata_path)
        tag = metadata.get("model_tag")
        if not tag or metadata.get("matched_normal_prefetcher") != POLICY:
            continue
        row = rows.get("offline_" + tag)
        if row is None:
            raise SystemExit("missing replay row for {}".format(tag))
        records.append(model_record(row, metadata, normal, no_pref, matched))
        selected_metadata.append(metadata)
    if not records:
        raise SystemExit("no Stride neural metadata found")
    observed_tags = {record["model_tag"] for record in records}
    if observed_tags != expected_tags:
        raise SystemExit(
            "Stride metadata/replay model set mismatch: observed={} expected={}".format(
                sorted(observed_tags), sorted(expected_tags)
            )
        )

    encoder_hashes = {
        metadata.get("runtime_encoder_sha256")
        for metadata in selected_metadata
    }
    contract_verified = (
        all(record["input_contract_verified"] for record in records)
        and all(record["analyzer_evidence_verified"] for record in records)
        and len(encoder_hashes) == 1
    )
    if not contract_verified:
        problems = [{
            "model_tag": record["model_tag"],
            "input_contract_mismatches": record["input_contract_mismatches"],
            "analyzer_evidence_mismatches": record[
                "analyzer_evidence_mismatches"
            ],
        } for record in records if (
            not record["input_contract_verified"]
            or not record["analyzer_evidence_verified"]
        )]
        raise SystemExit(
            "cross-model evidence verification failed: {}".format(
                json.dumps(problems, sort_keys=True)
            )
        )

    source_audit = audit_stride_source(args.stride_source)
    best = max(records, key=lambda item: item["ipc"])
    payload = {
        "status": "PASS",
        "source_matched_comparison": str(matched_path),
        "source_matched_comparison_status": matched.get("status"),
        "trace": TRACE,
        "policy": POLICY,
        "input_contract_verified": True,
        "current_metadata_bound_to_analyzer_evidence": True,
        "cross_capacity_runtime_encoder_identical": True,
        "runtime_encoder_sha256": next(iter(encoder_hashes)),
        "current_checkout_source_input_audit": source_audit,
        "completed_run_source_blob_provenance_verified": False,
        "same_external_input": SOURCE_INPUTS,
        "teacher_role": (
            "captured Stride actions are supervision and comparator replay "
            "only; they are not neural inference inputs"
        ),
        "no_pref_ipc": no_pref.get("ipc"),
        "offline_normal_ipc": normal.get("ipc"),
        "offline_normal_ipc_gain_vs_no_pref": (
            normal.get("ipc") - no_pref.get("ipc")
        ),
        "offline_normal_requests": normal.get("pf_requested"),
        "offline_normal_reached_triggers": normal.get(
            "runtime_reachable_list_triggers"
        ),
        "best_observed_neural_model_by_ipc": best["model_tag"],
        "best_observed_neural_ipc": best["ipc"],
        "best_neural_ipc_delta_vs_offline_normal": (
            best["ipc_delta_vs_offline_normal"]
        ),
        "any_neural_model_beats_offline_normal": any(
            record["ipc_delta_vs_offline_normal"] > 0
            for record in records
        ),
        "all_neural_models_lose_to_offline_normal": all(
            record["ipc_delta_vs_offline_normal"] <= 0
            for record in records
        ),
        "models": records,
        "interpretation_guardrails": [
            "PASS validates accounting and input fairness, not NN success.",
            "Request volume, target quality, miss rate, and IPC remain separate.",
            "This schema does not directly measure harmful victim eviction.",
            "The source audit covers the current checkout, not historical "
            "completed-run source provenance.",
            "One trace and one seed identify a best observed point, not an optimum.",
        ],
    }

    json_path = run_dir / "model_diagnosis.json"
    csv_path = run_dir / "model_diagnosis.csv"
    atomic_json(json_path, payload)

    flat_records = []
    for record in records:
        flat = {}
        flatten("", record, flat)
        flat_records.append(flat)
    preferred = [
        "method", "model_tag", "model_size", "parameter_count",
        "ipc", "ipc_delta_vs_offline_normal", "ipc_delta_vs_no_pref",
        "l2_load_miss_rate", "l2_miss_rate_delta_vs_offline_normal",
        "pf_requested", "request_ratio_vs_offline_normal",
        "runtime_reachable_list_triggers",
        "trigger_ratio_vs_offline_normal",
        "actions_per_reached_trigger",
        "offline_normal_actions_per_reached_trigger",
        "pf_filled", "pf_useful", "pf_useless", "pf_late",
        "selected_accuracy", "coverage_vs_no_pref_l2_miss", "timeliness",
        "input_contract_verified",
    ]
    remaining = sorted({
        key for record in flat_records for key in record
        if key not in preferred
    })
    fields = preferred + remaining
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({
            key: record.get(key, "") for key in fields
        } for record in flat_records)

    print("[PASS] {}".format(json_path))
    print("[PASS] {}".format(csv_path))
    print(
        "best_observed={} ipc={:.6f} delta_vs_offline={:.6f}".format(
            best["model_tag"], best["ipc"],
            best["ipc_delta_vs_offline_normal"],
        )
    )


if __name__ == "__main__":
    main()
