#!/usr/bin/env python3
"""Audit every training-size point against the original 602 offline protocol.

This validator is read-only with respect to experimental artifacts.  It emits
an explicit PASS/FAIL/INCOMPLETE report: unavailable identities are never
silently accepted, and raw results are never changed.
"""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "formal_NN_training/experiments/602_lstm_stride_live_inference"
CONTRACT_PATH = EXP / "config/offline_fairness_contract.json"
TRAINER = (
    ROOT / "formal_NN_training/experiments/602_offline_lstm_stride/python"
    / "train_and_offline_infer.py"
)
ANALYZER = (
    ROOT / "formal_NN_training/experiments/602_offline_lstm_stride/python"
    / "analyze_replay.py"
)
REPLAYER_SOURCE = ROOT / "external/ChampSim/prefetcher/list_replayer.cc"
FEATURE_SOURCE = ROOT / "formal_NN_training/common/threshold_free_policy.py"


def load(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gzip_content_sha256(path):
    digest = hashlib.sha256()
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def function_source(path, name):
    text = Path(path).read_text()
    lines = text.splitlines(True)
    start = None
    prefix = "def {}(".format(name)
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            start = index
            break
    if start is not None:
        stop = len(lines)
        for index in range(start + 1, len(lines)):
            line = lines[index]
            if line and not line[0].isspace() and line.strip():
                stop = index
                break
        return "".join(lines[start:stop]).rstrip("\n") + "\n"
    raise RuntimeError("function {} absent from {}".format(name, path))


def source_identities():
    encoder_payload = {
        "entrypoint_source": function_source(TRAINER, "runtime_features"),
        "primitive_source": function_source(FEATURE_SOURCE, "runtime_bits"),
        "fields": ["pc", "cache_line_address"],
        "address_bits": 64,
        "cache_line_bytes": 64,
    }
    encoder = hashlib.sha256(
        json.dumps(encoder_payload, sort_keys=True).encode()
    ).hexdigest()
    router_payload = (
        function_source(TRAINER, "_group_indices_by_pc")
        + function_source(TRAINER, "_chunk_batches")
    )
    router = hashlib.sha256(router_payload.encode()).hexdigest()
    return encoder, router


def value(metadata, run_metadata, manifest, *names):
    for source in (metadata, run_metadata, manifest):
        for name in names:
            if source.get(name) is not None:
                return source[name]
    return None


def check(row, name, actual, expected):
    if actual is None:
        row["checks"][name] = {"status": "INCOMPLETE", "actual": None,
                               "expected": expected}
        return
    status = "PASS" if actual == expected else "FAIL"
    row["checks"][name] = {"status": status, "actual": actual,
                           "expected": expected}


def identity(row, name, path, content=False):
    if not Path(path).is_file():
        row["identities"][name] = {"status": "INCOMPLETE", "path": str(path),
                                   "sha256": None}
        return None
    digest = gzip_content_sha256(path) if content else sha256(path)
    row["identities"][name] = {"status": "PASS", "path": str(path),
                               "sha256": digest}
    return digest


def overall(row):
    states = [item["status"] for item in row["checks"].values()]
    states += [item["status"] for item in row["identities"].values()]
    if "FAIL" in states:
        return "FAIL"
    if "INCOMPLETE" in states:
        return "INCOMPLETE"
    return "PASS"


def audit_point(path, contract, encoder_hash, router_hash):
    point = load(path)
    point_dir = path.parent
    offline = point_dir / "offline"
    run_metadata_path = offline / "run_metadata.json"
    run_metadata = load(run_metadata_path) if run_metadata_path.is_file() else {}
    export_path = point_dir / "export/model_metadata.json"
    export = load(export_path) if export_path.is_file() else {}
    manifest_path = (
        path.parents[4] / "training_prefixes" / point["budget_tag"]
        / "training_manifest.json"
    )
    manifest = load(manifest_path) if manifest_path.is_file() else {}
    row = {
        "hidden_size": point.get("hidden_size"),
        "budget_tag": point.get("budget_tag"),
        "instruction_budget": point.get("instruction_budget"),
        "seed": point.get("seed"),
        "checks": {},
        "identities": {},
    }
    expected = {
        "trace": contract["trace"],
        "instruction_budget": point.get("instruction_budget"),
        "training_warmup_instructions": 0,
        "training_simulation_instructions": point.get("instruction_budget"),
        "hidden_size": point.get("hidden_size"),
        "seed": contract["seed"],
        "model_revision": contract["model_revision"],
        "epochs": contract["epochs"],
        "chunk_length": contract["chunk_length"],
        "pc_batch_size": contract["pc_batch_size"],
        "optimizer": contract["optimizer"],
        "learning_rate": contract["learning_rate"],
        "teacher_policy": contract["teacher_policy"],
        "class_balancing_algorithm": contract["class_balancing_algorithm"],
        "offline_inference_state_semantics": contract["offline_inference_state_semantics"],
        "collection_semantics": contract["training_collection"],
    }
    actual = {
        "trace": value(point, run_metadata, manifest, "trace"),
        "instruction_budget": value(point, run_metadata, manifest, "instruction_budget"),
        "training_warmup_instructions": value(point, run_metadata, manifest, "training_warmup_instructions"),
        "training_simulation_instructions": value(point, run_metadata, manifest, "training_simulation_instructions"),
        "hidden_size": value(point, run_metadata, manifest, "hidden_size"),
        "seed": value(point, run_metadata, manifest, "seed"),
        "model_revision": value(point, run_metadata, manifest, "model_revision"),
        "epochs": value(point, run_metadata, manifest, "epochs"),
        "chunk_length": value(point, run_metadata, manifest, "chunk_length", "training_chunk_len"),
        "pc_batch_size": value(point, run_metadata, manifest, "pc_batch_size"),
        "optimizer": value(point, run_metadata, manifest, "optimizer"),
        "learning_rate": value(point, run_metadata, manifest, "learning_rate"),
        "teacher_policy": value(point, run_metadata, manifest, "teacher", "matched_normal_prefetcher"),
        "class_balancing_algorithm": value(point, run_metadata, manifest, "gate_imbalance_handling"),
        "offline_inference_state_semantics": value(point, run_metadata, manifest, "inference_state_mode"),
        "collection_semantics": value(point, run_metadata, manifest, "collection_semantics"),
    }
    for name in sorted(expected):
        check(row, name, actual[name], expected[name])
    check(row, "class_balance_uses_prefix_only",
          run_metadata.get("data_derived_class_balancing_used"), True)
    check(row, "teacher_is_labels_only",
          run_metadata.get("normal_policy_outputs_used_as_training_targets"), True)
    check(row, "teacher_not_runtime_input",
          run_metadata.get("normal_policy_outputs_used_as_model_inputs"), False)
    check(row, "external_inputs", run_metadata.get("effective_external_inputs"),
          contract["external_model_inputs"])
    check(row, "runtime_encoder_sha256",
          run_metadata.get("runtime_encoder_sha256") or export.get("runtime_encoder_sha256"),
          encoder_hash)
    check(row, "state_router_sha256",
          run_metadata.get("state_router_sha256") or export.get("state_router_sha256"),
          router_hash)
    check(row, "evaluation_warmup_instructions",
          point.get("evaluation_warmup_instructions"),
          contract["evaluation_warmup_instructions"])
    check(row, "evaluation_simulation_instructions",
          point.get("evaluation_simulation_instructions"),
          contract["evaluation_simulation_instructions"])
    check(row, "replay_list_format", point.get("replay_list_format"),
          contract["replay_list_format"])
    check(row, "keyed_replayer", point.get("keyed_replayer"),
          contract["keyed_replayer"])
    check(row, "champsim_configuration", point.get("champsim_configuration"),
          contract["champsim_configuration"])
    check(row, "no_pref_reference", point.get("no_pref_reference"),
          contract["no_pref_reference"])
    check(row, "offline_stride_reference", point.get("offline_stride_reference"),
          contract["offline_stride_reference"])

    train_stream = manifest.get("training_stream")
    if train_stream:
        candidate = Path(train_stream)
        if not candidate.is_file():
            candidate = manifest_path.parent / candidate.name
        identity(row, "training_stream_content", candidate, content=True)
    else:
        identity(row, "training_stream_content", Path("missing"), content=True)
    eval_stream = run_metadata.get("eval_stream") or point.get("evaluation_stream")
    if eval_stream:
        identity(row, "evaluation_stream_content", Path(eval_stream), content=True)
    else:
        default_eval = path.parents[4] / "evaluation/602.gcc_s-734B.eval_stream.csv.gz"
        identity(row, "evaluation_stream_content", default_eval, content=True)
    identity(row, "offline_lstm_replay_list", offline / "offline_lstm.replay.csv")
    identity(row, "offline_stride_teacher_list", offline / "offline_stride.replay.csv")
    identity(row, "offline_replay_log", offline / "replay.log")
    row["status"] = overall(row)
    return row


def write_csv(path, rows):
    fields = ("hidden_size", "budget_tag", "instruction_budget", "seed",
              "status", "failed_checks", "incomplete_checks")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            all_items = dict(row["checks"])
            all_items.update(row["identities"])
            writer.writerow({
                "hidden_size": row["hidden_size"],
                "budget_tag": row["budget_tag"],
                "instruction_budget": row["instruction_budget"],
                "seed": row["seed"],
                "status": row["status"],
                "failed_checks": ";".join(k for k, v in all_items.items() if v["status"] == "FAIL"),
                "incomplete_checks": ";".join(k for k, v in all_items.items() if v["status"] == "INCOMPLETE"),
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    contract = load(CONTRACT_PATH)
    encoder_hash, router_hash = source_identities()
    paths = sorted(args.run_dir.glob("points/h*/*/seed*/point_metadata.json"))
    rows = [audit_point(path, contract, encoder_hash, router_hash) for path in paths]
    shared = {
        "runtime_encoder_sha256": encoder_hash,
        "state_router_sha256": router_hash,
        "metric_parser_sha256": sha256(ANALYZER),
        "keyed_replayer_source_sha256": sha256(REPLAYER_SOURCE) if REPLAYER_SOURCE.is_file() else None,
        "fixed_epochs_note": contract["fixed_epochs_interpretation"],
    }
    for name, path in (
        ("no_pref_log_sha256", args.run_dir / "references/logs/no_pref.log"),
        ("offline_stride_log_sha256", args.run_dir / "references/logs/offline_stride.log"),
        ("keyed_replayer_binary_sha256",
         ROOT / "external/ChampSim/bin/champsim.602_offline_replay"),
    ):
        shared[name] = sha256(path) if path.is_file() else None
    evaluation_ids = sorted(set(
        row["identities"]["evaluation_stream_content"]["sha256"]
        for row in rows
        if row["identities"]["evaluation_stream_content"]["sha256"]
    ))
    shared["one_fixed_evaluation_stream"] = len(evaluation_ids) == 1
    teacher_ids = sorted(set(
        row["identities"]["offline_stride_teacher_list"]["sha256"]
        for row in rows
        if row["identities"]["offline_stride_teacher_list"]["sha256"]
    ))
    shared["one_fixed_offline_stride_teacher_list"] = len(teacher_ids) == 1
    statuses = {state: sum(row["status"] == state for row in rows)
                for state in ("PASS", "FAIL", "INCOMPLETE")}
    payload = {"schema_version": 1, "contract": contract, "shared": shared,
               "summary": statuses, "points": rows}
    output = args.output_dir or args.run_dir / "report"
    output.mkdir(parents=True, exist_ok=True)
    (output / "offline_fairness.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    write_csv(output / "offline_fairness.csv", rows)
    (output / "offline_fairness.tex").write_text(
        "Fairness audit: {} PASS, {} FAIL, {} INCOMPLETE points. "
        "Missing identities are not treated as matches.\\n".format(
            statuses["PASS"], statuses["FAIL"], statuses["INCOMPLETE"]
        )
    )
    print("[fairness] PASS={} FAIL={} INCOMPLETE={}".format(
        statuses["PASS"], statuses["FAIL"], statuses["INCOMPLETE"]))
    if statuses["FAIL"] or (args.require_complete and statuses["INCOMPLETE"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
