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
        "602_offline_lstm_spp",
        "spp_source_input_compact_empirical_prior_hurdle_delta_fill_free_running_v2",
        [
            "callback_kind", "invoke_prefetcher.addr",
            "cache_fill.evicted_addr",
        ],
        "602_offline_lstm_spp_empirical_prior_hurdle_free_running_v2_seed7",
    ),
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
        "623_offline_lstm_stride_keyed_crn_v15_seed7",
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
        "623_offline_lstm_spp_keyed_crn_joint_fill_v15_seed7",
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
    # Sacramento uses an older system Python and does not provide pandas.
    # Keep every server-side audit/analyzer dependency-free and fail closed if
    # either unsupported postponed annotations or pandas is introduced.
    future_annotations = "from __future__ import " + "annotations"
    for path in tuple(COMMON.rglob("*.py")) + tuple(EXPERIMENTS.rglob("*.py")):
        source = path.read_text()
        if future_annotations in source:
            fail("{} requires unsupported future annotations".format(path))
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""]
            else:
                continue
            if any(name == "pandas" or name.startswith("pandas.") for name in imported):
                fail("{} imports unavailable pandas".format(path))

    if (EXPERIMENTS / "623_offline_lstm_cnn_stride_spp").exists():
        fail("obsolete combined 623 directory still exists")

    policy_source = (COMMON / "threshold_free_policy.py").read_text()
    parse_python(COMMON / "threshold_free_policy.py")
    keyed_source = (COMMON / "keyed_sampling.py").read_text()
    parse_python(COMMON / "keyed_sampling.py")
    for token in (
        "sha256_event_keyed_inverse_cdf_crn_v1",
        "def keyed_uniform(",
        "def bernoulli_icdf(",
        "def poisson_icdf(",
        "poisson.ppf(",
        "def categorical_icdf(",
        "def canonical_component_order(",
        "def self_test_keyed_crn(",
    ):
        if token not in keyed_source:
            fail("keyed sampler missing {}".format(token))
    for forbidden in ("RandomState", "default_rng", ".binomial(", ".choice("):
        if forbidden in keyed_source:
            fail("keyed sampler retains mutable RNG API {}".format(forbidden))
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
        if experiment == "602_offline_lstm_spp":
            gate_contract = {
                "gate_class_weighting_used": False,
                "gate_training_objective": (
                    "empirical_prior_unweighted_categorical_nll"
                ),
                "gate_decoding_rule": "two_class_categorical_argmax",
            }
            for key, expected in gate_contract.items():
                if contract.get(key) != expected:
                    fail("602 SPP contract {} mismatch".format(key))
                if key not in notebook or str(expected) not in notebook:
                    fail("602 SPP notebook missing {}".format(key))
        if experiment == "623_offline_lstm_stride":
            stride_contract = {
                "model_revision": (
                    "compact_pc_keyed_crn_event_sampled_mixture_v15"
                ),
                "gate_decoding_rule": (
                    "event_keyed_bernoulli_inverse_cdf"
                ),
                "request_count_training_objective": (
                    "unweighted_bernoulli_hurdle_plus_positive_"
                    "poisson_excess_nll"
                ),
                "request_count_decoding_rule": (
                    "event_keyed_bernoulli_plus_common_quantile_"
                    "poisson_inverse_cdf"
                ),
                "request_count_residual_scope": "none_event_local",
                "cross_event_probability_credit_used": False,
                "sampled_outputs_used_as_decoder_feedback": False,
                "delta_mixture_decoding_rule": (
                    "event_keyed_mean_sorted_categorical_inverse_cdf_"
                    "then_component_mean"
                ),
                "delta_decoder_feedback_rule": (
                    "complete_mixture_expectation_same_in_training_"
                    "and_inference"
                ),
                "stochastic_decoding_reproducible": True,
                "common_random_numbers_across_capacities": True,
                "cross_event_rng_state_used": False,
                "decoder_probability_mass_carries_train_guard_history": False,
            }
            for key, expected in stride_contract.items():
                if contract.get(key) != expected:
                    fail("623 Stride contract {} mismatch".format(key))
                if key not in notebook or str(expected) not in notebook:
                    fail("623 Stride notebook missing {}".format(key))
            expected_decoder_key_fields = [
                "revision", "decoder_seed", "trace", "policy", "role",
                "event_key", "head", "action_rank",
            ]
            if contract.get("decoder_key_fields") != expected_decoder_key_fields:
                fail("623 Stride decoder key fields mismatch")
        if experiment == "623_offline_lstm_spp":
            spp_contract = {
                "model_revision": "compact_crn_joint_delta_fill_mixture_v15",
                "gate_class_weighting_used": False,
                "gate_training_objective": "unweighted_bernoulli_nll",
                "gate_decoding_rule": (
                    "event_keyed_bernoulli_inverse_cdf"
                ),
                "request_count_training_objective": (
                    "unweighted_bernoulli_hurdle_plus_positive_"
                    "poisson_excess_nll"
                ),
                "request_count_decoding_rule": (
                    "event_keyed_bernoulli_plus_common_quantile_"
                    "poisson_inverse_cdf"
                ),
                "request_count_residual_scope": "none_event_local",
                "joint_delta_fill_dependency_modeled": True,
                "joint_delta_fill_training_objective": (
                    "unweighted_joint_delta_component_fill_mixture_nll"
                ),
                "joint_delta_fill_decoding_rule": (
                    "event_keyed_mean_sorted_joint_pair_inverse_cdf"
                ),
                "cross_event_probability_credit_used": False,
                "sampled_outputs_used_as_decoder_feedback": False,
                "delta_mixture_decoding_rule": (
                    "single_joint_component_fill_sample_then_component_mean"
                ),
                "delta_decoder_feedback_rule": (
                    "complete_joint_distribution_expectation_same_in_"
                    "training_and_inference"
                ),
                "stochastic_decoding_reproducible": True,
                "common_random_numbers_across_capacities": True,
                "cross_event_rng_state_used": False,
                "same_source_input_offline_claim_allowed": True,
                "closed_loop_live_claim_allowed": False,
                "decoder_probability_mass_carries_train_guard_history": False,
            }
            for key, expected in spp_contract.items():
                if contract.get(key) != expected:
                    fail("623 SPP contract {} mismatch".format(key))
                if key not in notebook or str(expected) not in notebook:
                    fail("623 SPP notebook missing {}".format(key))
            expected_decoder_key_fields = [
                "revision", "decoder_seed", "trace", "policy", "role",
                "event_key", "head", "action_rank",
            ]
            if contract.get("decoder_key_fields") != expected_decoder_key_fields:
                fail("623 SPP decoder key fields mismatch")

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

    spp_tracks = (
        EXPERIMENTS / "602_offline_lstm_spp",
        EXPERIMENTS / "623_offline_lstm_spp",
        EXPERIMENTS / "623_offline_cnn_spp",
    )
    for spp in spp_tracks:
        family = "cnn" if "cnn" in spp.name else "lstm"
        normalize = (spp / "python" / "normalize_events.py").read_text()
        validator = (spp / "python" / "validate_collected_inputs.py").read_text()
        if "target_page_offset" in normalize or "target_page_offset" in validator:
            fail("{} SPP teacher schema still exposes page offsets".format(family))
        for token in (
            '"complete_neural_action_space": True',
            '"neural_degree_cap": None',
            '"fixed_page_offset_classes": None',
            '"nn_can_generate_actions_not_emitted_by_teacher": True',
        ):
            if token not in validator:
                fail("{} SPP manifest generator missing {}".format(family, token))
        spp_analyzer = (spp / "python" / "analyze_replay.py").read_text()
        if '"complete_neural_action_space": True' not in spp_analyzer:
            fail("{} SPP analyzer omits complete action-space check".format(family))
        if (
            "behavior_fields = sorted({" not in spp_analyzer
            or 'if key.startswith("behavior_")' not in spp_analyzer
        ):
            fail("{} SPP analyzer has a fixed behavior CSV schema".format(family))
        for path in (spp / "python").glob("*.py"):
            parse_python(path)

    spp_602_patch = (
        EXPERIMENTS / "602_offline_lstm_spp/linux/patch_demand_logger.sh"
    ).read_bytes()
    if (
        b"DEMAND_EVENT_LOG_SCHEMA_602_SPP_V1" not in spp_602_patch
        or b'"602_spp_causal_trigger_fill_v1"' not in spp_602_patch
        or b"DEMAND_EVENT_LOG_SCHEMA_(623_|602_SPP_)" not in spp_602_patch
    ):
        fail("602 SPP logger patch lacks schema or cross-track reset")
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

    spp_602_train = (
        EXPERIMENTS / "602_offline_lstm_spp/python/train_and_offline_infer.py"
    ).read_text()
    for token in (
        "self_test_model(args.model_size)",
        '"decoder_free_running_self_test": "PASS"',
        '"threshold_related_hardcodes_used": False',
        '"neural_degree_cap": None',
        '"fixed_page_offset_classes": None',
        "active_state, predicted_coordinate, predicted_fill",
        "expected_parameter_count",
        '"gate_class_weighting_used": False',
        '"gate_training_objective": (',
        '"empirical_prior_unweighted_categorical_nll"',
        '"gate_decoding_rule": "two_class_categorical_argmax"',
    ):
        if token not in spp_602_train:
            fail("602 SPP train script missing {}".format(token))
    spp_602_tree = ast.parse(
        spp_602_train,
        filename=str(
            EXPERIMENTS
            / "602_offline_lstm_spp/python/train_and_offline_infer.py"
        ),
    )
    for node in ast.walk(spp_602_tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and function.attr == "cross_entropy"
            and any(keyword.arg == "weight" for keyword in node.keywords)
        ):
            fail("602 SPP categorical loss still applies class weighting")
    for forbidden in (
        "_data_derived_gate_class_weights",
        "gate_class_weights",
        "gate_weights",
    ):
        if forbidden in spp_602_train:
            fail("602 SPP train script retains {}".format(forbidden))

    stride_623_train = (
        EXPERIMENTS
        / "623_offline_lstm_stride/python/train_and_offline_infer.py"
    ).read_text()
    for token in (
        "CompactPCKeyedSampledStrideLSTM",
        "event_keyed_hurdle_counts",
        "keyed_uniform",
        "canonical_component_order",
        "runtime_features",
        'RUNTIME_FEATURES = ADDRESS_BITS + LINE_NUMBER_BITS',
        '"compact_pc_keyed_crn_event_sampled_mixture_v15"',
        '"gate_training_objective": "unweighted_bernoulli_nll"',
        '"gate_decoding_rule": "event_keyed_bernoulli_inverse_cdf"',
        '"unweighted_bernoulli_hurdle_plus_positive_poisson_excess_nll"',
        '"strict_common_random_numbers_across_capacities": True',
        '"cross_event_rng_state_used": False',
        '"decoder_sampling_roles": ["eval"]',
        '"decoder_train_sampling_performed": False',
        '"decoder_guard_sampling_performed": False',
        '"runtime_feature_count": runtime["train"].shape[1]',
        '"probability_threshold_used": False',
        '"threshold_related_hardcodes_used": False',
        '"neural_degree_cap": None',
        '"fixed_page_offset_classes": None',
        '"normal_tracker_capacity_used_by_neural_inference": False',
        '"data_derived_gate_class_weights_used": False',
        '"event_keyed_crn_self_test": "PASS"',
        '"decoder_probability_mass_carries_train_guard_history": False',
        '"cross_event_probability_credit_used": False',
        '"sampled_outputs_used_as_decoder_feedback": False',
    ):
        if token not in stride_623_train:
            fail("623 Stride train script missing {}".format(token))
    for forbidden in (
        "_data_derived_gate_class_weights",
        "gate_class_weights",
        "two_class_categorical_argmax",
        ".argmax(",
        "_binary_probability_mass_choice",
        "_probability_mass_choice",
        "_mass_hurdle_counts",
        "RandomState",
        "default_rng",
        ".binomial(",
        ".poisson(",
        ".choice(",
    ):
        if forbidden in stride_623_train:
            fail("623 Stride train script retains {}".format(forbidden))

    spp_623_train = (
        EXPERIMENTS / "623_offline_lstm_spp/python/train_and_offline_infer.py"
    ).read_text()
    for token in (
        "CompactSPPLSTM",
        "predicted_fill_probabilities",
        "CompactSPPActionDecoder",
        "joint_action_head",
        "event_keyed_hurdle_counts",
        "keyed_uniform",
        "categorical_icdf",
        "canonical_joint_pair_order",
        'RUNTIME_FEATURES = LINE_ADDRESS_BITS + 1',
        '"compact_crn_joint_delta_fill_mixture_v15"',
        '"unweighted_bernoulli_hurdle_plus_positive_poisson_excess_nll"',
        '"joint_delta_fill_dependency_modeled": True',
        '"unweighted_joint_delta_component_fill_mixture_nll"',
        '"event_keyed_mean_sorted_joint_pair_inverse_cdf"',
        '"single_joint_delta_fill_pair_sample"',
        '"strict_common_random_numbers_across_capacities": True',
        '"cross_event_rng_state_used": False',
        '"decoder_sampling_roles": ["eval"]',
        '"same_source_input_offline_claim_allowed": True',
        '"closed_loop_live_claim_allowed": False',
        '"probability_threshold_used": False',
        '"threshold_related_hardcodes_used": False',
        '"neural_degree_cap": None',
        '"fixed_page_offset_classes": None',
        '"gate_class_weighting_used": False',
        '"gate_training_objective": "unweighted_bernoulli_nll"',
        '"gate_decoding_rule": "event_keyed_bernoulli_inverse_cdf"',
        '"stochastic_decoding_reproducible": True',
        '"decoder_probability_mass_carries_train_guard_history": False',
        '"cross_event_probability_credit_used": False',
        '"sampled_outputs_used_as_decoder_feedback": False',
    ):
        if token not in spp_623_train:
            fail("623 SPP train script missing {}".format(token))
    for forbidden in (
        "fill_logits.argmax(",
        "mix.argmax(",
        "_binary_probability_mass_choice",
        "_probability_mass_choice",
        "_mass_hurdle_counts",
        ".argmax(",
        "emit_head",
        "positive_log_count",
        "self.fill_head",
        "RandomState(",
        "default_rng(",
        ".binomial(",
        ".poisson(",
        ".choice(",
    ):
        if forbidden in spp_623_train:
            fail("623 SPP train script retains {}".format(forbidden))

    print("[PASS] eight direct-action tracks satisfy the static input contract")
    print("[PASS] training/inference encoder fields, code hashes, and decoder feedback agree")
    print("[PASS] no neural page-offset interface, threshold, or degree cap is declared")


if __name__ == "__main__":
    main()
