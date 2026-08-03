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
        "623_offline_lstm_stride_compact_hurdle_v16_seed7",
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
        "623_offline_lstm_spp_keyed_crn_joint_map_v16a_seed7",
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


# The active 623 LSTM implementation can advance without deleting the audit
# contract for the completed v15 negative checkpoint.  The selected profile is
# derived from data/stream_contract.json, so checking out the v15 sources still
# exercises the original sampled-decoder requirements while main exercises the
# controlled v16 revision.  Input revisions remain in TRACKS because neither
# v16 experiment changes or recollects the external callback stream.
LSTM_623_REVISION_PROFILES = {
    "623_offline_lstm_stride": {
        "compact_pc_keyed_crn_event_sampled_mixture_v15": {
            "run_id": "623_offline_lstm_stride_keyed_crn_v15_seed7",
            "historical_negative": True,
            "contract": {
                "model_revision": (
                    "compact_pc_keyed_crn_event_sampled_mixture_v15"
                ),
                "gate_decoding_rule": "event_keyed_bernoulli_inverse_cdf",
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
            },
            "decoder_key_fields": [
                "revision", "decoder_seed", "trace", "policy", "role",
                "event_key", "head", "action_rank",
            ],
            "train_required": (
                "CompactPCKeyedSampledStrideLSTM",
                "event_keyed_hurdle_counts",
                "keyed_uniform",
                "canonical_component_order",
                'RUNTIME_FEATURES = ADDRESS_BITS + LINE_NUMBER_BITS',
                '"compact_pc_keyed_crn_event_sampled_mixture_v15"',
                '"gate_training_objective": "unweighted_bernoulli_nll"',
                '"gate_decoding_rule": "event_keyed_bernoulli_inverse_cdf"',
                '"unweighted_bernoulli_hurdle_plus_positive_poisson_excess_nll"',
                '"strict_common_random_numbers_across_capacities": True',
                '"event_keyed_crn_self_test": "PASS"',
            ),
            "forbidden_identifiers": (
                "_data_derived_gate_class_weights", "gate_class_weights",
            ),
            "forbidden_source": (
                "two_class_categorical_argmax", ".argmax(",
                "_binary_probability_mass_choice",
                "_probability_mass_choice", "_mass_hurdle_counts",
                "RandomState", "default_rng", ".binomial(",
                ".poisson(", ".choice(",
            ),
        },
        "compact_pc_keyed_balanced_deterministic_scalar_v16": {
            "run_id": "623_offline_lstm_stride_compact_hurdle_v16_seed7",
            "historical_negative": False,
            "contract": {
                "model_revision": (
                    "compact_pc_keyed_balanced_deterministic_scalar_v16"
                ),
                "gate_training_objective": (
                    "data_derived_frequency_balanced_two_class_cross_entropy"
                ),
                "gate_class_weights_source": (
                    "train_zero_positive_frequencies_equal_aggregate_"
                    "loss_mass"
                ),
                "gate_decoding_rule": "deterministic_two_class_argmax",
                "request_count_training_objective": (
                    "balanced_two_class_hurdle_plus_positive_log_count_"
                    "smooth_l1"
                ),
                "request_count_decoding_rule": (
                    "deterministic_gate_argmax_plus_rounded_exp_positive_"
                    "log_count"
                ),
                "request_count_residual_scope": "none_event_local",
                "delta_mixture_decoding_rule": None,
                "delta_training_objective": (
                    "scalar_signed_log_delta_smooth_l1"
                ),
                "delta_decoding_rule": (
                    "deterministic_rounded_scalar_signed_log_delta"
                ),
                "delta_decoder_feedback_rule": (
                    "emitted_scalar_coordinate_same_in_training_and_inference"
                ),
                "deterministic_decoding": True,
                "deterministic_decoding_reproducible": True,
                "stochastic_decoding": False,
                "stochastic_decoding_reproducible": False,
                "common_random_numbers_across_capacities": False,
                "cross_event_rng_state_used": False,
                "decoder_probability_mass_carries_train_guard_history": False,
                "cross_event_probability_credit_used": False,
                "sampled_outputs_used_as_decoder_feedback": False,
                "delta_mixture_components": 0,
            },
            "decoder_key_fields": [],
            "train_required": (
                "CompactPCKeyedDeterministicStrideLSTM",
                "_data_derived_gate_class_weights",
                "_positive_counts_from_log_mean",
                "_compact_balanced_deterministic_loss",
                "self_test_deterministic_count_and_balance()",
                'RUNTIME_FEATURES = ADDRESS_BITS + LINE_NUMBER_BITS',
                '"compact_pc_keyed_balanced_deterministic_scalar_v16"',
                '"data_derived_gate_class_weights_used": True',
                '"gate_class_weighting_used": True',
                '"gate_class_weights_source": (',
                '"train_zero_positive_frequencies_equal_aggregate_loss_mass"',
                '"data_derived_frequency_balanced_two_class_cross_entropy"',
                '"gate_decoding_rule": "deterministic_two_class_argmax"',
                '"balanced_two_class_hurdle_plus_positive_log_count_smooth_l1"',
                '"scalar_signed_log_delta_smooth_l1"',
                '"deterministic_decoding_reproducible": True',
                '"stochastic_decoding_reproducible": False',
                '"deterministic_count_and_balance_self_test": "PASS"',
                '"delta_decoding_rule": "deterministic_rounded_scalar_signed_log_delta"',
                '"emitted_scalar_coordinate_same_in_training_and_inference"',
                '"guard_role": "causal_input_history_warmup_and_audit_only"',
                '"event_keyed_crn_self_test": "NOT_APPLICABLE"',
            ),
            "forbidden_identifiers": (
                "event_keyed_hurdle_counts", "keyed_uniform",
                "canonical_component_order",
            ),
            "forbidden_source": (
                "_binary_probability_mass_choice",
                "_probability_mass_choice", "_mass_hurdle_counts",
                "RandomState", "default_rng", ".binomial(",
                ".poisson(", ".choice(",
            ),
        },
    },
    "623_offline_lstm_spp": {
        "compact_crn_joint_delta_fill_mixture_v15": {
            "run_id": (
                "623_offline_lstm_spp_keyed_crn_joint_fill_v15_seed7"
            ),
            "historical_negative": True,
            "contract": {
                "model_revision": "compact_crn_joint_delta_fill_mixture_v15",
                "gate_class_weighting_used": False,
                "gate_training_objective": "unweighted_bernoulli_nll",
                "gate_decoding_rule": "event_keyed_bernoulli_inverse_cdf",
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
            },
            "decoder_key_fields": [
                "revision", "decoder_seed", "trace", "policy", "role",
                "event_key", "head", "action_rank",
            ],
            "train_required": (
                "CompactSPPLSTM", "predicted_fill_probabilities",
                "CompactSPPActionDecoder", "joint_action_head",
                "event_keyed_hurdle_counts", "keyed_uniform",
                "categorical_icdf", "canonical_joint_pair_order",
                'RUNTIME_FEATURES = LINE_ADDRESS_BITS + 1',
                '"compact_crn_joint_delta_fill_mixture_v15"',
                '"unweighted_joint_delta_component_fill_mixture_nll"',
                '"event_keyed_mean_sorted_joint_pair_inverse_cdf"',
                '"single_joint_delta_fill_pair_sample"',
                '"joint_delta_fill_dependency_modeled": True',
            ),
            "forbidden_identifiers": (),
            "forbidden_source": (
                "fill_logits.argmax(", "mix.argmax(",
                "_binary_probability_mass_choice",
                "_probability_mass_choice", "_mass_hurdle_counts",
                ".argmax(", "emit_head", "positive_log_count",
                "self.fill_head", "RandomState(", "default_rng(",
                ".binomial(", ".poisson(", ".choice(",
            ),
        },
        "compact_crn_joint_delta_fill_guard_map_v16a": {
            "run_id": (
                "623_offline_lstm_spp_keyed_crn_joint_map_v16a_seed7"
            ),
            "historical_negative": False,
            "contract": {
                "model_revision": (
                    "compact_crn_joint_delta_fill_guard_map_v16a"
                ),
                "parent_model_revision": (
                    "compact_crn_joint_delta_fill_mixture_v15"
                ),
                "weights_model_revision": (
                    "compact_crn_joint_delta_fill_mixture_v15"
                ),
                "gate_class_weighting_used": False,
                "gate_training_objective": "unweighted_bernoulli_nll",
                "gate_decoding_rule": "event_keyed_bernoulli_inverse_cdf",
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
                "fill_decoding_rule": "guard_selected_joint_pair_map",
                "delta_mixture_decoding_rule": (
                    "guard_selected_joint_component_then_component_mean"
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
                "cross_event_probability_credit_used": False,
                "sampled_outputs_used_as_decoder_feedback": False,
            },
            "contract_only": {
                "joint_delta_fill_decoding_rule": (
                    "guard_selected_joint_class_map_or_component_peak_map"
                ),
                "supported_model_revisions": [
                    "compact_crn_joint_delta_fill_mixture_v15",
                    "compact_crn_joint_delta_fill_guard_map_v16a",
                ],
            },
            "nested_contracts": (
                {
                    "field": "legacy_v15_decoder_contract",
                    "values": {
                        "model_revision": (
                            "compact_crn_joint_delta_fill_mixture_v15"
                        ),
                        "joint_delta_fill_decoding_rule": (
                            "event_keyed_mean_sorted_joint_pair_inverse_cdf"
                        ),
                        "old_run_preserved": True,
                    },
                },
                {
                    "field": "v16a_decoder_ablation",
                    "values": {
                        "run_id": (
                            "623_offline_lstm_spp_keyed_crn_joint_map_"
                            "v16a_seed7"
                        ),
                        "model_revision": (
                            "compact_crn_joint_delta_fill_guard_map_v16a"
                        ),
                        "parent_model_revision": (
                            "compact_crn_joint_delta_fill_mixture_v15"
                        ),
                        "weights_retrained": False,
                        "checkpoint_reused_with_strict_validation": True,
                        "candidate_modes": [
                            "joint_class_map", "component_peak_map",
                        ],
                        "selection_data": (
                            "guard labels only; evaluation labels are never "
                            "used for selection"
                        ),
                        "old_run_preserved": True,
                    },
                },
            ),
            "decoder_key_fields": [
                "revision", "decoder_seed", "trace", "policy", "role",
                "event_key", "head", "action_rank",
            ],
            "notebook_required": (
                "redecode-v16a",
                "623_offline_lstm_spp_keyed_crn_joint_fill_v15_seed7",
                "623_offline_lstm_spp_keyed_crn_joint_map_v16a_seed7",
                "guard_joint_map_spp_lstm_h",
                "--operation", "--parent-model", "--parent-metadata",
                "--parent-training-history",
                "decoder_revision", "decoder_candidate_modes",
                "selected_decoder_mode", "guard_decoder_selection",
                "parent_run_id",
                "parent_checkpoint_sha256", "parent_run_metadata_sha256",
                "parent_training_history_sha256",
                "strict_checkpoint_validation_passed",
                "deterministic_joint_map_self_test",
            ),
            "train_required": (
                "CompactSPPLSTM", "CompactSPPActionDecoder",
                "joint_action_head", "event_keyed_hurdle_counts",
                "keyed_uniform", "categorical_icdf",
                "canonical_joint_pair_order", "deterministic_joint_pair",
                "guard_selection_key", "complete_behavior_metrics",
                "validate_and_load_v15_parent",
                'RUNTIME_FEATURES = LINE_ADDRESS_BITS + 1',
                'V15_MODEL_REVISION = "compact_crn_joint_delta_fill_mixture_v15"',
                'V16A_MODEL_REVISION = "compact_crn_joint_delta_fill_guard_map_v16a"',
                'V16A_OPERATION = "redecode-v16a"',
                'V16A_DECODER_MODES = ("joint_class_map", "component_peak_map")',
                'V16A_DECODER_REVISION = "guard_selected_deterministic_joint_map_v16a"',
                'V15_PARENT_RUN_ID = "623_offline_lstm_spp_keyed_crn_joint_fill_v15_seed7"',
                'model.load_state_dict(payload["state_dict"], strict=True)',
                '"SPP guard metric matched an action across callback identities"',
                '"decoder_candidate_modes": (',
                '"selected_decoder_mode": selected_decoder_mode',
                '"guard_decoder_selection": (',
                '"parent_model_revision": V15_MODEL_REVISION',
                '"weights_model_revision": V15_MODEL_REVISION',
                '"parent_checkpoint_sha256": parent["checkpoint_sha256"]',
                '"parent_run_metadata_sha256": parent["metadata_sha256"]',
                '"parent_training_history_sha256": (',
                '"strict_checkpoint_validation_passed": is_v16a',
                '"weights_retrained": not is_v16a',
                '"checkpoint_reused": is_v16a',
                '"deterministic_joint_map_self_test": "PASS"',
                '"parent_run_id": args.parent_run_id',
            ),
            "forbidden_identifiers": (),
            "forbidden_source": (
                "fill_logits.argmax(", "mix.argmax(",
                "_binary_probability_mass_choice",
                "_probability_mass_choice", "_mass_hurdle_counts",
                ".argmax(", "emit_head", "positive_log_count",
                "self.fill_head", "RandomState(", "default_rng(",
                ".binomial(", ".poisson(", ".choice(",
            ),
        },
    },
}


def fail(message):
    raise RuntimeError(message)


def parse_python(path):
    ast.parse(path.read_text(), filename=str(path))


def python_identifiers(source, filename):
    """Return exact Python identifiers, excluding strings and comments.

    Substring checks are not safe here: for example, the required audit key
    ``data_derived_gate_class_weights_used`` contains the retired identifier
    ``gate_class_weights`` even though no weighted gate implementation exists.
    """
    tree = ast.parse(source, filename=str(filename))
    identifiers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg is not None:
            identifiers.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            identifiers.add(node.name)
        elif isinstance(node, ast.alias):
            identifiers.add(node.asname or node.name.rsplit(".", 1)[-1])
    return identifiers


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


def selected_623_profile(experiment, contract):
    profiles = LSTM_623_REVISION_PROFILES.get(experiment)
    if profiles is None:
        return None
    model_revision = contract.get("model_revision")
    profile = profiles.get(model_revision)
    if profile is None:
        fail(
            "{} has unaudited model revision {!r}; expected one of {}".format(
                experiment, model_revision, sorted(profiles)
            )
        )
    return profile


def validate_623_profile_contract(experiment, contract, notebook, profile):
    for key, expected in profile["contract"].items():
        if contract.get(key) != expected:
            fail("{} contract {} mismatch".format(experiment, key))
        if key not in notebook or str(expected) not in notebook:
            fail("{} notebook missing {}={!r}".format(
                experiment, key, expected
            ))
    for key, expected in profile.get("contract_only", {}).items():
        if contract.get(key) != expected:
            fail("{} parent contract {} mismatch".format(experiment, key))
    for nested in profile.get("nested_contracts", ()):
        observed = contract.get(nested["field"])
        if not isinstance(observed, dict):
            fail("{} missing {} contract".format(
                experiment, nested["field"]
            ))
        for key, expected in nested["values"].items():
            if observed.get(key) != expected:
                fail("{} {} contract {} mismatch".format(
                    experiment, nested["field"], key
                ))
    for token in profile.get("notebook_required", ()):
        if token not in notebook:
            fail("{} notebook missing {}".format(experiment, token))
    if contract.get("decoder_key_fields") != profile["decoder_key_fields"]:
        fail("{} decoder key fields mismatch".format(experiment))


def validate_623_profile_training(experiment, profile):
    path = EXPERIMENTS / experiment / "python/train_and_offline_infer.py"
    source = path.read_text()
    for token in (
        "runtime_features"
        if experiment == "623_offline_lstm_stride"
        else "runtime_array",
        '"runtime_feature_count"',
        '"probability_threshold_used": False',
        '"threshold_related_hardcodes_used": False',
        '"neural_degree_cap": None',
        '"fixed_page_offset_classes": None',
        '"decoder_probability_mass_carries_train_guard_history": False',
        '"cross_event_probability_credit_used": False',
        '"sampled_outputs_used_as_decoder_feedback": False',
    ) + tuple(profile["train_required"]):
        if token not in source:
            fail("{} train script missing {}".format(experiment, token))
    identifiers = python_identifiers(source, path)
    for forbidden in profile["forbidden_identifiers"]:
        if forbidden in identifiers:
            fail("{} train script retains {}".format(
                experiment, forbidden
            ))
    for forbidden in profile["forbidden_source"]:
        if forbidden in source:
            fail("{} train script retains {}".format(
                experiment, forbidden
            ))


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
    installer = COMMON / "install_colab_output.py"
    installer_source = installer.read_text()
    parse_python(installer)
    for token in (
        "def validate_members(",
        "member.issym()",
        'handle.extractall(str(output_dir), members=members)',
        "COMMON_REQUIRED_FILES",
    ):
        if token not in installer_source:
            fail("Colab output installer missing {}".format(token))
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
        profile = selected_623_profile(experiment, contract)
        if profile is not None:
            run_id = profile["run_id"]
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
        if profile is not None:
            validate_623_profile_contract(
                experiment, contract, notebook, profile
            )

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
            if (
                experiment in (
                    "623_offline_lstm_stride", "623_offline_lstm_spp"
                )
                and relative == "linux/run_server.sh"
                and "install_colab_output.py" not in shell_source
            ):
                fail("{} server does not auto-install Colab output".format(experiment))
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
    spp_602_identifiers = python_identifiers(
        spp_602_train,
        EXPERIMENTS / "602_offline_lstm_spp/python/train_and_offline_infer.py",
    )
    for forbidden in (
        "_data_derived_gate_class_weights",
        "gate_class_weights",
        "gate_weights",
    ):
        if forbidden in spp_602_identifiers:
            fail("602 SPP train script retains {}".format(forbidden))

    for experiment in (
        "623_offline_lstm_stride", "623_offline_lstm_spp",
    ):
        contract = json.loads(
            (EXPERIMENTS / experiment / "data/stream_contract.json")
            .read_text()
        )
        validate_623_profile_training(
            experiment, selected_623_profile(experiment, contract)
        )

    print("[PASS] eight direct-action tracks satisfy the static input contract")
    print("[PASS] training/inference encoder fields, code hashes, and decoder feedback agree")
    print("[PASS] no neural page-offset interface, threshold, or degree cap is declared")


if __name__ == "__main__":
    main()
