#!/usr/bin/env python3
"""Verify and summarize the isolated 602 offline streamer/LSTM capacity sweep."""
import argparse
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path


TRACE = "602.gcc_s-734B"
KV = re.compile(r"^([A-Za-z0-9_]+)\s+([-+0-9.eE]+)\s*$")
FINISHED = re.compile(
    r"Finished CPU\s+0\s+instructions:\s+(\d+)\s+cycles:\s+(\d+)\s+cumulative IPC:\s+([-+0-9.eE]+)"
)
REPLAYER = re.compile(
    r"emitted\s+(\d+)\s+candidates over\s+(\d+)\s+runtime ROI L2 LOAD accesses "
    r"\((\d+)\s+matched PC-line-occ triggers"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gzip_content_sha256(path):
    """Hash decompressed CSV bytes so gzip container metadata is irrelevant."""
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


def parse_log(path):
    result = {"ipc": 0.0, "instructions": 0, "cycles": 0, "emitted": 0, "callbacks": 0, "matched": 0}
    text = path.read_text(errors="ignore")
    for raw in text.splitlines():
        match = KV.match(raw.strip())
        if match:
            if match.group(1) == "Core_0_IPC":
                result["ipc"] = float(match.group(2))
            elif match.group(1) == "Core_0_instructions":
                result["instructions"] = int(float(match.group(2)))
            elif match.group(1) == "Core_0_cycles":
                result["cycles"] = int(float(match.group(2)))
        match = FINISHED.search(raw)
        if match:
            result["instructions"] = int(match.group(1))
            result["cycles"] = int(match.group(2))
            if not result["ipc"]:
                result["ipc"] = float(match.group(3))
        match = REPLAYER.search(raw)
        if match:
            result["emitted"] = int(match.group(1))
            result["callbacks"] = int(match.group(2))
            result["matched"] = int(match.group(3))
    return result


def parse_events(path):
    """Read post-warmup L2 demand/PF records emitted by patch_demand_logger.sh."""
    result = {
        "demand_l2_loads": 0,
        "demand_l2_hits": 0,
        "demand_l2_misses": 0,
        "prefetch_useful_demand_hits": 0,
        "prefetch_late_demand_misses": 0,
        "prefetch_request_events": 0,
    }
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"event", "hit", "was_prefetch", "late"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError("{} missing event columns {}".format(path, sorted(missing)))
        for row in reader:
            if row["event"] == "PF":
                result["prefetch_request_events"] += 1
                continue
            if row["event"] != "DEMAND":
                continue
            result["demand_l2_loads"] += 1
            hit = int(row["hit"])
            if hit:
                result["demand_l2_hits"] += 1
            else:
                result["demand_l2_misses"] += 1
            if int(row["was_prefetch"]):
                result["prefetch_useful_demand_hits"] += 1
            if int(row["late"]):
                result["prefetch_late_demand_misses"] += 1
    return result


def model_tag_to_hidden(tag):
    if not tag.startswith("h") or not tag[1:].isdigit():
        raise ValueError("model tag must be h<positive integer>: {}".format(tag))
    return int(tag[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--model-tags", default="h8,h16,h32,h64,h128")
    ap.add_argument("--base-model-tag", default="h8")
    ap.add_argument(
        "--source-input-dir",
        type=Path,
        default=None,
        help=(
            "Original colab_input directory for legacy Colab outputs that store only "
            "gzip-byte SHA256 values. Its decompressed streams must match this run."
        ),
    )
    args = ap.parse_args()
    model_tags = [x.strip() for x in args.model_tags.split(",") if x.strip()]
    if not model_tags or args.base_model_tag not in model_tags:
        raise SystemExit("base model tag must be one of --model-tags")
    methods = ["no_pref", "live_streamer_reference", "offline_streamer"]
    methods += ["offline_lstm_" + tag for tag in model_tags]
    logs = args.run_dir / "logs"
    events = args.run_dir / "events"
    colab_root = args.run_dir / "colab_output"
    rows = []
    failures = []

    for method in methods:
        log_path = logs / (TRACE + "." + method + ".log")
        event_path = events / (TRACE + "." + method + ".events.csv.gz")
        if not log_path.is_file():
            failures.append("missing log {}".format(log_path))
            continue
        if not event_path.is_file():
            failures.append("missing event log {}".format(event_path))
            continue
        row = {"trace": TRACE, "method": method, "log": str(log_path), "event_log": str(event_path)}
        row.update(parse_log(log_path))
        try:
            row.update(parse_events(event_path))
        except Exception as exc:
            failures.append("{} event parse failed: {}".format(method, exc))
        if row["ipc"] <= 0 or row["instructions"] <= 0:
            failures.append("{} lacks final simulator statistics".format(method))
        if method == "live_streamer_reference" and row["prefetch_request_events"] <= 0:
            failures.append("live_streamer_reference emitted no PF requests")
        # A threshold-free neural policy may legitimately learn request count
        # zero.  An empty NN action list is therefore a measured outcome, not a
        # replay failure.  The normal comparator must remain nonempty, while
        # every offline replay must still process the full callback stream.
        if method == "offline_streamer" and (row["matched"] <= 0 or row["emitted"] <= 0):
            failures.append("offline_streamer did not replay keyed entries")
        if method.startswith("offline_") and row["callbacks"] <= 0:
            failures.append("{} reported zero replay callbacks".format(method))
        row["matched_primary_comparison"] = int(method == "offline_streamer" or method.startswith("offline_lstm_"))
        row["model_tag"] = method[len("offline_lstm_"):] if method.startswith("offline_lstm_") else ""
        row["hidden_size"] = model_tag_to_hidden(row["model_tag"]) if row["model_tag"] else 0
        rows.append(row)

    by_method = {row["method"]: row for row in rows}
    if len(rows) != len(methods):
        failures.append("one or more methods are missing")
    if rows and len({row["instructions"] for row in rows}) != 1:
        failures.append("simulation instruction counts differ")
    current_input_dir = args.run_dir / "colab_input"
    current_streams = {
        "train": current_input_dir / (TRACE + ".train_stream.csv.gz"),
        "eval": current_input_dir / (TRACE + ".eval_stream.csv.gz"),
    }
    current_stream_info = {}
    for role, path in current_streams.items():
        if not path.is_file():
            failures.append("missing normalized {} stream {}".format(role, path))
            continue
        try:
            current_stream_info[role] = stream_hashes(path)
        except (OSError, gzip.BadGzipFile) as exc:
            failures.append("cannot hash current {} stream {}: {}".format(role, path, exc))

    source_stream_info = {}
    source_streams = {}
    if args.source_input_dir is not None:
        source_streams = {
            "train": args.source_input_dir / (TRACE + ".train_stream.csv.gz"),
            "eval": args.source_input_dir / (TRACE + ".eval_stream.csv.gz"),
        }
        for role, path in source_streams.items():
            if not path.is_file():
                failures.append("missing source {} stream {}".format(role, path))
                continue
            try:
                source_stream_info[role] = stream_hashes(path)
            except (OSError, gzip.BadGzipFile) as exc:
                failures.append("cannot hash source {} stream {}: {}".format(role, path, exc))

    input_provenance = {
        "current_input_dir": str(current_input_dir),
        "current_streams": current_stream_info,
        "per_model_tag": {},
    }
    if args.source_input_dir is not None:
        input_provenance["legacy_source_input_dir"] = str(args.source_input_dir)
        input_provenance["legacy_source_streams"] = source_stream_info

    metadata_by_tag = {}
    for tag in model_tags:
        for name in ("offline_streamer.replay.csv", "offline_lstm.replay.csv", "model.pt", "run_metadata.json"):
            path = colab_root / tag / name
            if not path.is_file():
                failures.append("missing Colab output {}".format(path))
        metadata_path = colab_root / tag / "run_metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            failures.append("{} metadata parse failed: {}".format(tag, exc))
            continue
        metadata_by_tag[tag] = metadata
        state_contract = {
            "training_state_mode": "chronological_stateful_tbptt",
            "training_chunks_shuffled": False,
            "training_state_carried_across_chunks": True,
            "training_state_detached_between_chunks": True,
            "experiment_revision": "source_input_variable_delta_free_running_v7",
            "neural_role": "standalone_direct_action_prefetcher",
            "same_external_input_contract": True,
            "training_inference_input_encoder_identical": True,
            "decoder_training_mode": "free_running_autoregressive_same_as_inference",
            "decoder_previous_teacher_action_used_as_input": False,
            "decoder_free_running_self_test": "PASS",
            "training_runtime_fields": ["cache_line_address"],
            "inference_runtime_fields": ["cache_line_address"],
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
            "handcrafted_semantic_features_used": False,
            "manual_loss_weights_used": False,
            "training_regularization_used": False,
            "inference_policy_hardcodes_used": False,
            "learned_request_count": True,
            "nn_generates_own_target_addresses": True,
        }
        for key, expected in state_contract.items():
            if metadata.get(key) != expected:
                failures.append(
                    "{} metadata {}={!r}; expected {!r}".format(
                        tag, key, metadata.get(key), expected
                    )
                )
        encoder_hashes = {metadata.get("runtime_encoder_sha256"), metadata.get("training_runtime_encoder_sha256"), metadata.get("inference_runtime_encoder_sha256")}
        encoder_hash = next(iter(encoder_hashes)) if len(encoder_hashes) == 1 else None
        if not isinstance(encoder_hash, str) or len(encoder_hash) != 64:
            failures.append("{} train/inference encoder hash mismatch".format(tag))
        if metadata.get("matched_normal_prefetcher") != "streamer":
            failures.append("{} metadata is not matched to streamer".format(tag))
        if set(current_stream_info) != {"train", "eval"}:
            continue

        canonical_keys = {
            "train": "train_stream_content_sha256",
            "eval": "eval_stream_content_sha256",
        }
        if all(metadata.get(key) for key in canonical_keys.values()):
            input_provenance["per_model_tag"][tag] = {
                "mode": "canonical_decompressed_content_sha256",
            }
            for role, key in canonical_keys.items():
                if metadata.get(key) != current_stream_info[role]["content_sha256"]:
                    failures.append(
                        "{} {}-stream content SHA256 does not match this run".format(tag, role)
                    )
            continue

        if args.source_input_dir is None:
            failures.append(
                "{} has legacy gzip-byte hashes only; rerun analyze with "
                "--source-input-dir pointing to the Colab input used for this output".format(tag)
            )
            continue
        if set(source_stream_info) != {"train", "eval"}:
            continue
        input_provenance["per_model_tag"][tag] = {
            "mode": "legacy_source_byte_hash_plus_decompressed_content_match",
        }
        for role in ("train", "eval"):
            metadata_key = role + "_stream_sha256"
            if metadata.get(metadata_key) != source_stream_info[role]["gzip_sha256"]:
                failures.append(
                    "{} {}-stream SHA256 does not match supplied source input".format(tag, role)
                )
            if current_stream_info[role]["content_sha256"] != source_stream_info[role]["content_sha256"]:
                failures.append(
                    "{} {}-stream decompressed content does not match this run".format(tag, role)
                )

    streamer_list_hashes = {}
    for tag in model_tags:
        path = colab_root / tag / "offline_streamer.replay.csv"
        if path.is_file():
            streamer_list_hashes[tag] = sha256(path)
    if streamer_list_hashes and len(set(streamer_list_hashes.values())) != 1:
        failures.append("offline streamer list differs across capacity points")

    if "no_pref" in by_method:
        no_pref_misses = by_method["no_pref"]["demand_l2_misses"]
        for row in rows:
            useful = row["prefetch_useful_demand_hits"]
            row["prefetch_coverage_vs_no_pref_misses"] = useful / float(no_pref_misses) if no_pref_misses else 0.0
            row["l2_miss_reduction_vs_no_pref"] = (
                (no_pref_misses - row["demand_l2_misses"]) / float(no_pref_misses) if no_pref_misses else 0.0
            )

    status = "FAIL" if failures else "PASS"
    fields = [
        "trace", "method", "model_tag", "hidden_size", "matched_primary_comparison",
        "ipc", "instructions", "cycles", "matched", "emitted", "callbacks",
        "demand_l2_loads", "demand_l2_hits", "demand_l2_misses",
        "prefetch_useful_demand_hits", "prefetch_late_demand_misses",
        "prefetch_request_events", "prefetch_coverage_vs_no_pref_misses",
        "l2_miss_reduction_vs_no_pref", "log", "event_log",
    ]
    out_csv = args.run_dir / "matched_comparison.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "status": status,
        "fair_comparison_claim_allowed": not failures,
        "trace": TRACE,
        "primary_comparison": ["offline_streamer", "offline_lstm_<capacity>"],
        "transport": "both policies see the same address evaluation stream and use the same PC-line-occ ListReplayer; the LSTM generates targets independently",
        "live_streamer_role": "validated general reference only; it is not the offline primary comparator",
        "matched_input_contract": (
            "effective Streamer input only: current cache-line address plus the LSTM's own causal state; "
            "PC is replay transport identity only"
        ),
        "neural_contract": "unbounded learned count plus autoregressive direct cache-line deltas; no Streamer candidates, page-offset table, degree, or tracker state enter inference",
        "metric_definitions": {
            "prefetch_useful_demand_hits": "post-warmup L2 load hits on a line marked as prefetched",
            "prefetch_late_demand_misses": "post-warmup L2 load misses merged into an in-flight prefetch MSHR",
            "prefetch_coverage_vs_no_pref_misses": "useful prefetch demand hits divided by no-prefetch L2 load misses",
            "l2_miss_reduction_vs_no_pref": "(no-prefetch L2 load misses - method L2 load misses) / no-prefetch L2 load misses",
        },
        "model_tags": model_tags,
        "required_recurrent_state_contract": {
            "training": "chronological stateful TBPTT; hidden/cell values cross chunks; graph detached at chunk boundaries",
            "inference": "continuous within the independent evaluation stream",
        },
        "offline_streamer_list_hashes_by_model_tag": streamer_list_hashes,
        "input_provenance": input_provenance,
        "failures": failures,
        "rows": rows,
    }
    if not failures:
        streamer = by_method["offline_streamer"]["ipc"]
        sweep = []
        first_better = None
        for tag in model_tags:
            row = by_method["offline_lstm_" + tag]
            metadata = metadata_by_tag[tag]
            point = {
                "model_tag": tag,
                "hidden_size": metadata["hidden_size"],
                "parameter_count": metadata["parameter_count"],
                "ipc": row["ipc"],
                "ipc_delta_lstm_minus_streamer": row["ipc"] - streamer,
                "lstm_beats_offline_streamer": row["ipc"] > streamer,
                "prefetch_coverage_vs_no_pref_misses": row["prefetch_coverage_vs_no_pref_misses"],
                "prefetch_useful_demand_hits": row["prefetch_useful_demand_hits"],
                "prefetch_late_demand_misses": row["prefetch_late_demand_misses"],
            }
            sweep.append(point)
            if first_better is None and point["lstm_beats_offline_streamer"]:
                first_better = point
        payload.update(
            {
                "offline_streamer_ipc": streamer,
                "capacity_sweep": sweep,
                "first_measured_capacity_beating_offline_streamer": first_better,
                "saturation_interpretation": (
                    "The first measured capacity above offline streamer is exploratory because "
                    "the evaluation stream is reused across capacity points; confirm it on a "
                    "fresh held-out window before selecting a final capacity."
                ),
                "offline_streamer_list_sha256": sha256(colab_root / args.base_model_tag / "offline_streamer.replay.csv"),
            }
        )
    out_json = args.run_dir / "matched_comparison.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("[{}] {}".format(status, out_json))
    if failures:
        raise SystemExit(" | ".join(failures))


if __name__ == "__main__":
    main()
