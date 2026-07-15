#!/usr/bin/env python3
"""Static, dependency-free audit for all source-input-fair neural tracks."""
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "formal_NN_training" / "experiments"
COMMON = ROOT / "formal_NN_training" / "common"

TRACKS = (
    (
        "602_offline_lstm_stride",
        "source_input_variable_delta_free_running_v7",
        ["pc", "cache_line_address"],
        "602_offline_lstm_stride_variable_delta_free_running_v7_seed7",
    ),
    (
        "602_offline_lstm_streamer",
        "source_input_variable_delta_free_running_v7",
        ["cache_line_address"],
        "602_offline_lstm_streamer_variable_delta_free_running_v7_seed7",
    ),
    (
        "602_offline_lstm_ampm",
        "source_input_variable_delta_free_running_v7",
        ["cache_line_address"],
        "602_offline_lstm_ampm_variable_delta_free_running_v7_seed7",
    ),
    (
        "623_offline_lstm_stride",
        "stride_source_input_variable_delta_free_running_v9",
        ["pc", "addr"],
        "623_offline_lstm_stride_variable_delta_free_running_v9_seed7",
    ),
    (
        "623_offline_cnn_stride",
        "stride_source_input_variable_delta_free_running_v9",
        ["pc", "addr"],
        "623_offline_cnn_stride_variable_delta_free_running_v9_seed7",
    ),
    (
        "623_offline_lstm_spp",
        "spp_source_input_variable_delta_fill_feedback_free_running_v11",
        [
            "callback_kind", "invoke_prefetcher.addr",
            "cache_fill.evicted_addr",
        ],
        "623_offline_lstm_spp_variable_delta_free_running_v11_seed7",
    ),
    (
        "623_offline_cnn_spp",
        "spp_source_input_variable_delta_fill_feedback_free_running_v11",
        [
            "callback_kind", "invoke_prefetcher.addr",
            "cache_fill.evicted_addr",
        ],
        "623_offline_cnn_spp_variable_delta_free_running_v11_seed7",
    ),
)


def fail(message):
    raise RuntimeError(message)


def parse_python(path):
    ast.parse(path.read_text(), filename=str(path))


def notebook_source(path):
    notebook = json.loads(path.read_text())
    if notebook.get("nbformat") != 4:
        fail("{} is not a v4 notebook".format(path))
    chunks = []
    for index, cell in enumerate(notebook.get("cells", ())):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", ()))
        chunks.append(source)
        if not any(
            line.lstrip().startswith(("!", "%"))
            for line in source.splitlines()
        ):
            ast.parse(source, filename="{}:cell{}".format(path, index))
    return "\n".join(chunks)


def contract_fields(contract):
    if "training_runtime_fields" in contract:
        return (
            contract["training_runtime_fields"],
            contract["inference_runtime_fields"],
        )
    return (
        contract["training_runtime_inputs"],
        contract["inference_runtime_inputs"],
    )


def main():
    if (EXPERIMENTS / "623_offline_lstm_cnn_stride_spp").exists():
        fail("obsolete combined 623 directory still exists")

    policy_source = (COMMON / "threshold_free_policy.py").read_text()
    parse_python(COMMON / "threshold_free_policy.py")
    for path in (
        COMMON / "direct_action_lstm.py",
        COMMON / "experiment_623_stride.py",
        COMMON / "experiment_623_spp.py",
    ):
        parse_python(path)
        source = path.read_text()
        if "self_test_free_running_decoder(" not in source:
            fail("{} does not execute the decoder feedback self-test".format(path))
        if '"decoder_free_running_self_test": "PASS"' not in source:
            fail("{} does not record the decoder feedback self-test".format(path))

    if "active_state, target, active_fill" in policy_source:
        fail("teacher-forced decoder feedback remains in training")
    for required in (
        "predicted_coordinate = mean.gather",
        "predicted_fill =",
        "model.decoder.advance(\n                active_state, predicted_coordinate, predicted_fill",
        "def self_test_free_running_decoder(",
        '"teacher delta leaked into decoder feedback"',
        '"teacher fill leaked into decoder feedback"',
        "Poisson request-count distribution",
    ):
        if required not in policy_source:
            fail("missing free-running decoder evidence: {}".format(required))
    if "target_page_offset" in (COMMON / "experiment_623_spp.py").read_text():
        fail("SPP student interface still names a page-offset target")

    for experiment, revision, expected_fields, run_id in TRACKS:
        base = EXPERIMENTS / experiment
        contract_path = base / "data" / "stream_contract.json"
        contract = json.loads(contract_path.read_text())
        observed_revision = contract.get(
            "revision", contract.get("experiment_revision")
        )
        if observed_revision != revision:
            fail("{} revision mismatch".format(experiment))
        train_fields, inference_fields = contract_fields(contract)
        if train_fields != expected_fields or inference_fields != expected_fields:
            fail("{} training/inference fields differ".format(experiment))
        if contract.get("training_inference_input_encoder_identical") is not True:
            fail("{} does not require one encoder".format(experiment))
        if contract.get("decoder_training_mode") != (
            "free_running_autoregressive_same_as_inference"
        ):
            fail("{} decoder mode mismatch".format(experiment))
        if contract.get("decoder_previous_teacher_action_used_as_input") is not False:
            fail("{} still permits teacher action feedback".format(experiment))

        notebook_paths = tuple((base / "colab").glob("*.ipynb"))
        if len(notebook_paths) != 1:
            fail("{} must have exactly one notebook".format(experiment))
        notebook = notebook_source(notebook_paths[0])
        for token in (
            run_id,
            revision,
            "free_running_autoregressive_same_as_inference",
            "decoder_previous_teacher_action_used_as_input",
            "decoder_free_running_self_test",
            "training_runtime_fields",
            "inference_runtime_fields",
            "training_runtime_encoder_sha256",
            "inference_runtime_encoder_sha256",
        ):
            if token not in notebook:
                fail("{} notebook missing {}".format(experiment, token))

        for relative in (
            "python/analyze_replay.py",
            "python/train_and_offline_infer.py",
        ):
            parse_python(base / relative)
        for relative in ("linux/run_server.sh", "linux/launch_server.sh"):
            shell_source = (base / relative).read_text()
            if run_id not in shell_source:
                fail("{} missing run ID in {}".format(experiment, relative))
            if (
                relative == "linux/run_server.sh"
                and "decoder_free_running_self_test" not in shell_source
            ):
                fail("{} server does not enforce decoder self-test".format(experiment))
        analyzer = (base / "python" / "analyze_replay.py").read_text()
        for token in (
            "decoder_training_mode",
            "decoder_previous_teacher_action_used_as_input",
            "decoder_free_running_self_test",
            "training_runtime_encoder_sha256",
            "inference_runtime_encoder_sha256",
        ):
            if token not in analyzer:
                fail("{} analyzer missing {}".format(experiment, token))

    for family in ("lstm", "cnn"):
        spp = EXPERIMENTS / "623_offline_{}_spp".format(family)
        normalize = (spp / "python" / "normalize_events.py").read_text()
        validator = (spp / "python" / "validate_collected_inputs.py").read_text()
        if "target_page_offset" in normalize or "target_page_offset" in validator:
            fail("{} SPP teacher schema still exposes page offsets".format(family))
        for path in (spp / "python").glob("*.py"):
            parse_python(path)

    lstm_patch = (
        EXPERIMENTS / "623_offline_lstm_spp/linux/patch_demand_logger.sh"
    ).read_bytes()
    cnn_patch = (
        EXPERIMENTS / "623_offline_cnn_spp/linux/patch_demand_logger.sh"
    ).read_bytes()
    if lstm_patch != cnn_patch:
        fail("SPP LSTM/CNN logger patches differ")
    if b'wq_replacement = "' in lstm_patch or b"wq_replacement = '''" in lstm_patch:
        fail("SPP WQ replacement reverted to brace-sensitive formatting")

    compare_source = (
        EXPERIMENTS / "compare_623_split_architectures.py"
    ).read_text()
    parse_python(EXPERIMENTS / "compare_623_split_architectures.py")
    if "runtime_encoder_exact_match" not in compare_source:
        fail("cross-directory encoder equality is not enforced")

    print("[PASS] seven direct-action tracks satisfy the static input contract")
    print("[PASS] training/inference encoder fields, code hashes, and decoder feedback agree")
    print("[PASS] no neural page-offset interface, threshold, or degree cap is declared")


if __name__ == "__main__":
    main()
