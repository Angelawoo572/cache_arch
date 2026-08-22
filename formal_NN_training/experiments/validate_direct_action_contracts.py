#!/usr/bin/env python3
"""Dependency-free static audit for the four matched-input 602 tracks."""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "formal_NN_training" / "experiments"
COMMON = ROOT / "formal_NN_training" / "common"

EXPECTED_TRACKS = (
    "602_offline_lstm_stride",
    "602_offline_lstm_streamer",
    "602_offline_lstm_ampm",
    "602_offline_lstm_spp",
)

EXPECTED_COMMON_FILES = (
    "__init__.py",
    "direct_action_lstm.py",
    "normal_policy_reference.py",
    "threshold_free_policy.py",
)

FORBIDDEN_RUNTIME_FIELD_PARTS = (
    "teacher",
    "target",
    "label",
    "candidate",
    "normal_action",
    "private_state",
    "future",
)


def fail(message):
    raise RuntimeError(message)


def read_json(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        fail("cannot read JSON contract {}: {}".format(path, exc))
    if not isinstance(value, dict):
        fail("{} must contain one JSON object".format(path))
    return value


def parse_python(path):
    source = path.read_text()
    return ast.parse(source, filename=str(path))


def imported_roots(tree):
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def validate_notebook(path):
    notebook = read_json(path)
    if notebook.get("nbformat") != 4:
        fail("{} is not a v4 notebook".format(path))
    for index, cell in enumerate(notebook.get("cells", ())):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", ()))
        if any(
            line.lstrip().startswith(("!", "%"))
            for line in source.splitlines()
        ):
            continue
        ast.parse(source, filename="{}:cell{}".format(path, index))


def contract_fields(contract):
    training = contract.get(
        "training_runtime_fields", contract.get("training_runtime_inputs")
    )
    inference = contract.get(
        "inference_runtime_fields", contract.get("inference_runtime_inputs")
    )
    if not isinstance(training, list) or not isinstance(inference, list):
        fail("stream contract must declare list-valued runtime fields")
    return training, inference


def validate_runtime_input_boundary(track_name, stream):
    training, inference = contract_fields(stream)
    if training != inference:
        fail("{} training/inference runtime fields differ".format(track_name))
    if not training:
        fail("{} has an empty external-input contract".format(track_name))

    for field in (str(value).lower() for value in training):
        for forbidden in FORBIDDEN_RUNTIME_FIELD_PARTS:
            if forbidden in field:
                fail(
                    "{} runtime field {!r} crosses the fairness boundary".format(
                        track_name, field
                    )
                )

    for alias in (
        "source_decision_effective_external_input",
        "decision_effective_external_input",
        "external_input_fields",
    ):
        if alias in stream and list(stream[alias]) != training:
            fail("{} {} differs from runtime fields".format(track_name, alias))

    if stream.get("training_inference_input_encoder_identical") is not True:
        fail("{} does not require one training/inference encoder".format(track_name))

    for key in (
        "normal_actions_are_model_inputs",
        "normal_candidates_are_model_inputs",
        "normal_private_state_is_model_input",
        "teacher_actions_are_model_inputs",
        "teacher_count_used_as_decoder_feedback",
        "normal_request_rate_is_neural_budget",
        "normal_policy_outputs_used_as_model_inputs",
        "normal_policy_candidates_used_as_model_inputs",
        "normal_policy_private_state_used_as_model_inputs",
        "normal_policy_request_rate_used_as_budget",
    ):
        if key in stream and stream[key] is not False:
            fail("{} must set {}=false".format(track_name, key))

    mode = str(stream.get("decoder_training_mode", "")).lower()
    if "free_running" not in mode:
        fail("{} decoder is not free-running".format(track_name))
    if stream.get("decoder_previous_teacher_action_used_as_input") is not False:
        fail("{} permits teacher-action feedback".format(track_name))


def validate_track(track):
    name = track.name
    stream_path = track / "data" / "stream_contract.json"
    if not stream_path.is_file():
        fail("{} has no data/stream_contract.json".format(name))
    validate_runtime_input_boundary(name, read_json(stream_path))

    notebooks = tuple((track / "colab").glob("*.ipynb"))
    if len(notebooks) != 1:
        fail("{} must have exactly one Colab notebook".format(name))
    validate_notebook(notebooks[0])

    required = (
        track / "python" / "train_and_offline_infer.py",
        track / "python" / "analyze_replay.py",
    )
    for path in required:
        if not path.is_file():
            fail("{} is missing {}".format(name, path.relative_to(track)))

    for path in track.rglob("*.py"):
        parse_python(path)

    analyzer = required[1].read_text()
    for token in (
        "training_runtime_encoder_sha256",
        "inference_runtime_encoder_sha256",
    ):
        if token not in analyzer:
            fail("{} analyzer omits {}".format(name, token))


def validate_portability():
    paths = []
    for filename in EXPECTED_COMMON_FILES:
        path = COMMON / filename
        if not path.is_file():
            fail("missing retained common helper {}".format(path))
        paths.append(path)
    for name in EXPECTED_TRACKS:
        paths.extend((EXPERIMENTS / name).rglob("*.py"))

    future_annotations = "from __future__ import " + "annotations"
    for path in paths:
        source = path.read_text()
        if future_annotations in source:
            fail("{} requires unsupported future annotations".format(path))
        tree = ast.parse(source, filename=str(path))
        if "pandas" in imported_roots(tree):
            fail("{} imports unavailable pandas".format(path))


def validate_spp_boundary():
    track = EXPERIMENTS / "602_offline_lstm_spp"
    for filename in ("normalize_events.py", "validate_collected_inputs.py"):
        path = track / "python" / filename
        if not path.is_file():
            fail("602 SPP is missing {}".format(path.relative_to(track)))
        if "target_page_offset" in path.read_text():
            fail("602 SPP exposes a teacher page-offset interface")


def main():
    validate_portability()
    for name in EXPECTED_TRACKS:
        track = EXPERIMENTS / name
        if not track.is_dir():
            fail("missing matched-input track {}".format(name))
        validate_track(track)
    validate_spp_boundary()

    print("[PASS] four 602 matched-input tracks satisfy the static contract")
    print("[PASS] runtime fields exclude teacher actions, candidates, and private state")
    print("[PASS] training and inference use identical causal input boundaries")
    print("[PASS] retained Python and Colab sources pass static portability checks")


if __name__ == "__main__":
    main()
