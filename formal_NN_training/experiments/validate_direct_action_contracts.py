#!/usr/bin/env python3
"""Dependency-free static audit for the matched-input direct-action tracks.

The active 623 architecture is intentionally not copied into this file.  Each
active track exposes a stable, torch-free ``python/model_contract.py`` and this
audit checks the semantic fairness/action-space contract returned by that
module.  Consequently a model revision or run ID can advance without editing a
second version-token registry here.
"""
import ast
import importlib.util
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
    "623_offline_lstm_stride",
    "623_offline_cnn_stride",
    "623_offline_lstm_spp",
    "623_offline_cnn_spp",
)
ACTIVE_MODEL_CONTRACT_TRACKS = (
    "623_offline_lstm_stride",
    "623_offline_lstm_spp",
)

FORBIDDEN_RUNTIME_FIELD_PARTS = (
    "teacher", "target", "label", "candidate", "normal_action",
    "private_state", "future",
)
LEGACY_V23_MODEL_SEMANTICS = {
    "delta_vocabulary_source": "train_labels_only",
    "delta_other_escape": "signed_log_continuous_bounded_approximation",
    "delta_other_decode_precision": (
        "rounded_float32_approximate_except_exact_vocabulary"
    ),
    "full_signed_line_delta_range_reachable": False,
    "every_signed_line_delta_exactly_representable": False,
    "exact_delta_representability_scope": "train_vocabulary_only",
    "fixed_page_offset_classes": None,
    "same_page_rule_used_by_neural_inference": False,
    "normal_policy_templates_used_by_neural_inference": False,
}


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


def notebook_source(path):
    notebook = read_json(path)
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
    training = contract.get(
        "training_runtime_fields", contract.get("training_runtime_inputs")
    )
    inference = contract.get(
        "inference_runtime_fields", contract.get("inference_runtime_inputs")
    )
    if not isinstance(training, list) or not isinstance(inference, list):
        fail("stream contract must declare list-valued runtime fields")
    return training, inference


def imported_roots(tree):
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def load_model_contract(track):
    """Load one stable stdlib-only model contract and return its description."""
    path = track / "python" / "model_contract.py"
    if not path.is_file():
        fail("{} has no stable python/model_contract.py".format(track.name))
    tree = parse_python(path)
    third_party = imported_roots(tree).intersection((
        "numpy", "pandas", "scipy", "sklearn", "torch",
    ))
    if third_party:
        fail("{} model contract is not dependency-free: {}".format(
            track.name, sorted(third_party)
        ))

    module_name = "_direct_action_contract_{}".format(track.name)
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        fail("cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    description = getattr(module, "MODEL_CONTRACT", None)
    if description is None:
        for name in (
            "describe_model_points", "model_points_description",
            "describe_contract",
        ):
            function = getattr(module, name, None)
            if callable(function):
                description = function()
                break
    if not isinstance(description, dict):
        fail("{} exposes no dictionary model description".format(path))
    try:
        json.dumps(description, sort_keys=True)
    except (TypeError, ValueError) as exc:
        fail("{} description is not JSON-serializable: {}".format(path, exc))
    return description


def same_value(track_name, stream, model, key, required=False):
    """Read a semantic field while rejecting two disagreeing sources."""
    in_stream = key in stream
    in_model = key in model
    if in_stream and in_model and stream[key] != model[key]:
        fail("{} disagrees on {} between stream/model contracts".format(
            track_name, key
        ))
    if in_model:
        return model[key]
    if in_stream:
        return stream[key]
    if required:
        fail("{} is missing semantic field {}".format(track_name, key))
    return None


def aliased_value(track_name, stream, model, keys, required=False):
    """Read one semantic value whose descriptive key has legacy aliases."""
    observed = []
    for key in keys:
        if key in model:
            observed.append(("model." + key, model[key]))
        if key in stream:
            observed.append(("stream." + key, stream[key]))
    if not observed:
        if required:
            fail("{} is missing one of {}".format(track_name, keys))
        return None
    value = observed[0][1]
    for location, other in observed[1:]:
        if other != value:
            fail("{} disagrees on {} at {}".format(
                track_name, keys, location
            ))
    return value


def validate_runtime_input_boundary(track_name, stream):
    training, inference = contract_fields(stream)
    if training != inference:
        fail("{} training/inference runtime fields differ".format(track_name))
    if not training:
        fail("{} has an empty external-input contract".format(track_name))
    lowered = [str(field).lower() for field in training]
    for field in lowered:
        for forbidden in FORBIDDEN_RUNTIME_FIELD_PARTS:
            if forbidden in field:
                fail("{} runtime field {!r} crosses the fairness boundary".format(
                    track_name, field
                ))

    for alias in (
        "source_decision_effective_external_input",
        "decision_effective_external_input",
        "external_input_fields",
    ):
        if alias in stream and list(stream[alias]) != training:
            fail("{} {} differs from runtime fields".format(track_name, alias))
    if stream.get("training_inference_input_encoder_identical") is not True:
        fail("{} does not require one training/inference encoder".format(
            track_name
        ))

    false_if_present = (
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
    )
    for key in false_if_present:
        if key in stream and stream[key] is not False:
            fail("{} must set {}=false".format(track_name, key))


def validate_active_model_contract_v23_legacy(track, stream, model):
    name = track.name
    model_fields = model.get(
        "external_input_fields", model.get("source_inputs")
    )
    training_fields, _ = contract_fields(stream)
    if list(model_fields or ()) != training_fields:
        fail("{} stable model/runtime input fields differ".format(name))

    required_false_aliases = (
        (
            "normal_actions_are_model_inputs",
            "teacher_actions_are_model_inputs",
            "normal_policy_outputs_used_as_model_inputs",
        ),
        (
            "normal_candidates_are_model_inputs",
            "normal_policy_candidates_used_as_model_inputs",
        ),
        (
            "normal_private_state_is_model_input",
            "normal_policy_private_state_used_as_model_inputs",
        ),
        (
            "normal_request_rate_is_neural_budget",
            "normal_policy_request_rate_used_as_budget",
        ),
    )
    for aliases in required_false_aliases:
        if aliased_value(name, stream, model, aliases, required=True) is not False:
            fail("{} crosses the fairness boundary at {}".format(name, aliases))

    stream_revision = stream.get(
        "experiment_revision", stream.get("revision")
    )
    if not stream_revision or model.get("experiment_revision") != stream_revision:
        fail("{} stable model/input revision mismatch".format(name))
    for key in ("run_id", "model_revision", "decoder_revision"):
        value = model.get(key)
        if not isinstance(value, str) or not value:
            fail("{} stable model contract lacks {}".format(name, key))
        if key in stream and stream[key] != value:
            fail("{} stream/model {} mismatch".format(name, key))

    for key, expected in LEGACY_V23_MODEL_SEMANTICS.items():
        if key not in model:
            fail("{} stable model contract lacks {}".format(name, key))
        if same_value(name, stream, model, key, required=True) != expected:
            fail("{} has invalid {}".format(name, key))
    for key in (
        "probability_threshold_used",
        "inference_policy_hardcodes_used",
        "neural_degree_cap",
        "neural_role",
        "decoder_previous_teacher_action_used_as_input",
        "decoder_previous_predicted_action_used_as_input",
        "decoder_previous_sampled_action_used_as_input",
    ):
        if key not in model:
            fail("{} stable model contract lacks {}".format(name, key))
    if "delta_vocabulary_max_exact" not in model:
        fail("{} stable model contract lacks delta_vocabulary_max_exact".format(
            name
        ))
    vocabulary_size = same_value(
        name, stream, model, "delta_vocabulary_max_exact", required=True
    )
    if isinstance(vocabulary_size, bool) or not isinstance(vocabulary_size, int):
        fail("{} delta_vocabulary_max_exact must be an integer".format(name))
    if vocabulary_size <= 0:
        fail("{} exact-delta vocabulary must be nonempty".format(name))

    if same_value(
        name, stream, model, "probability_threshold_used", required=True
    ) is not False:
        fail("{} enables a probability threshold".format(name))
    if same_value(
        name, stream, model, "inference_policy_hardcodes_used", required=True
    ) is not False:
        fail("{} enables an inference policy hardcode".format(name))
    if same_value(
        name, stream, model, "neural_degree_cap", required=True
    ) is not None:
        fail("{} inherits a fixed neural degree".format(name))
    if same_value(
        name, stream, model, "neural_role", required=True
    ) != "standalone_direct_action_prefetcher":
        fail("{} is not an independent direct-action learner".format(name))

    mode = same_value(
        name, stream, model, "decoder_training_mode", required=True
    )
    lowered_mode = str(mode).lower()
    if not isinstance(mode, str) or not any(
        token in lowered_mode for token in (
            "free_running", "no_action_feedback",
            "without_action_feedback",
            "without_teacher_or_predicted_action_feedback",
        )
    ):
        fail("{} decoder mode does not exclude action feedback".format(name))
    previous_teacher = same_value(
        name, stream, model,
        "decoder_previous_teacher_action_used_as_input", required=True,
    )
    if previous_teacher is not False:
        fail("{} feeds a previous teacher action to its decoder".format(name))

    previous_sampled = aliased_value(
        name, stream, model,
        (
            "decoder_previous_predicted_action_used_as_input",
            "decoder_previous_sampled_action_used_as_input",
        ),
        required=True,
    )
    if previous_sampled is not False:
        fail("{} feeds a previous decoded action back into inference".format(name))

    if name.endswith("_stride"):
        if not str(model.get("parameter_formula", "")).startswith(
            "5*H^2 + (201+K)*H + K+60"
        ):
            fail("{} direct-bit parameter formula changed".format(name))
        hurdle_objective = aliased_value(
            name, stream, model,
            (
                "hurdle_training_objective", "gate_training_objective",
                "zero_positive_training_objective",
                "request_hurdle_training_objective",
            ),
            required=True,
        )
        lowered_hurdle_objective = str(hurdle_objective).lower()
        if "inverse_frequency" not in lowered_hurdle_objective:
            fail("{} sparse hurdle loss is not TRAIN-balanced".format(name))
        hurdle_classes = model.get(
            "hurdle_classes", model.get("hurdle_class_order")
        )
        if hurdle_classes != ["ZERO", "POSITIVE"]:
            indices = model.get("hurdle_class_indices")
            if indices != {"ZERO": 0, "POSITIVE": 1}:
                fail("{} lacks ZERO/POSITIVE hurdle classes".format(name))
        positive_count_objective = aliased_value(
            name, stream, model,
            (
                "positive_count_training_objective",
                "count_training_objective", "positive_count_objective",
            ),
            required=True,
        )
        if not all(
            token in str(positive_count_objective).lower()
            for token in ("positive", "count")
        ):
            fail("{} lacks a positive-count objective".format(name))
        positive_count_decoding = aliased_value(
            name, stream, model,
            ("positive_count_decoding_rule", "positive_count_mode"),
            required=True,
        )
        lowered_count = str(positive_count_decoding).lower()
        if "log_count" not in lowered_count or "round_exp" not in lowered_count:
            fail("{} positive count is not learned log-count decode".format(name))
        terminal_stop = aliased_value(
            name, stream, model,
            (
                "terminal_stop_supervised",
                "terminal_stop_supervised_for_every_teacher_sequence",
            ),
            required=True,
        )
        if terminal_stop is not False:
            fail("{} decoder-only ablation unexpectedly uses STOP ranks".format(name))
        if same_value(
            name, stream, model,
            "hurdle_prior_correction_at_decode_used", required=True,
        ) is not True:
            fail("{} does not undo TRAIN hurdle weighting".format(name))
        correction = same_value(
            name, stream, model, "hurdle_prior_correction_rule", required=True
        )
        if "minus_log" not in str(correction).lower():
            fail("{} hurdle prior correction has the wrong direction".format(name))
        if model.get("weights_retrained") is not False:
            fail("{} v23 must reuse v22 weights".format(name))
        if model.get("checkpoint_reused") is not True:
            fail("{} v23 must reuse v22 checkpoints".format(name))
        if model.get("decoder_only_change") is not True:
            fail("{} v23 is not marked decoder-only".format(name))
    elif name.endswith("_spp"):
        if model.get("parameter_formula") != (
            "9*H^2 + (191+K)*H + K+118"
        ):
            fail("{} direct-bit parameter formula changed".format(name))
        objective = same_value(
            name, stream, model,
            "joint_action_training_objective", required=True,
        )
        if not all(
            token in str(objective).lower()
            for token in ("joint", "action", "cross_entropy")
        ):
            fail("{} lacks joint replay-action cross entropy".format(name))
        for key in (
            "separate_gate_head_used", "request_count_head_used",
            "separate_delta_head_used", "separate_fill_head_used",
        ):
            if same_value(name, stream, model, key, required=True) is not False:
                fail("{} retains factorized head {}".format(name, key))
        if same_value(
            name, stream, model,
            "terminal_stop_supervised_for_every_teacher_sequence",
            required=True,
        ) is not False:
            fail("{} adds an out-of-support terminal rank".format(name))
        if same_value(
            name, stream, model, "all_available_tail_stop_supervised",
            required=True,
        ) is not True:
            fail("{} lacks all-available-tail STOP supervision".format(name))
        if same_value(
            name, stream, model,
            "maximum_length_sequences_terminate_by_finite_support",
            required=True,
        ) is not True:
            fail("{} lacks finite-support termination".format(name))
        horizon = same_value(
            name, stream, model, "finite_output_horizon_source", required=True
        )
        if "train" not in str(horizon).lower() or "maximum" not in str(horizon).lower():
            fail("{} finite horizon is not TRAIN-derived".format(name))
        for key, expected in (
            ("finite_output_horizon_is_dataset_derived", True),
            ("finite_output_horizon_is_normal_request_budget", False),
            ("finite_output_horizon_is_tuned_degree", False),
            ("joint_action_prior_correction_at_decode_used", True),
        ):
            if same_value(name, stream, model, key, required=True) is not expected:
                fail("{} has invalid {}".format(name, key))
        correction = same_value(
            name, stream, model,
            "joint_action_prior_correction_rule", required=True,
        )
        if "minus_log" not in str(correction).lower():
            fail("{} joint prior correction has the wrong direction".format(name))
        if model.get("stochastic_decoding") is not False:
            fail("{} must use deterministic joint decoding".format(name))

    selection = model.get(
        "checkpoint_selection", model.get("guard_selection_rule")
    )
    if selection is None:
        fail("{} stable model contract lacks a guard selection rule".format(name))
    lowered_selection = str(selection).lower()
    if name.endswith("_stride"):
        if "parent_v22_guard" not in lowered_selection:
            fail("{} does not preserve parent v22 guard selection".format(name))
    elif "lexicographic" not in lowered_selection:
        fail("{} guard selection is not lexicographic".format(name))
    if any(token in lowered_selection for token in ("mean", "average", "composite")):
        fail("{} guard selection averages unlike metrics".format(name))

    forbidden_keys = (
        "source_action_templates",
        "normal_action_templates",
        "legal_source_templates",
    )
    for key in forbidden_keys:
        if key in model or key in stream:
            fail("{} retains forbidden output-template field {}".format(name, key))

    points = model.get("points")
    if not isinstance(points, list) or not points:
        fail("{} stable model contract has no architecture points".format(name))
    observed_sizes = []
    for point in points:
        if not isinstance(point, dict):
            fail("{} model point is not an object".format(name))
        size = point.get("model_size", point.get("size"))
        observed_sizes.append(size)
        count = point.get(
            "parameter_count",
            point.get("parameters", point.get("maximum_parameter_count")),
        )
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            fail("{} model point has invalid parameter count".format(name))
    if observed_sizes != [8, 16, 32, 64, 128]:
        fail("{} must sweep h8,h16,h32,h64,h128".format(name))

    parameter_contract = str(model.get("parameter_count_contract", "")).lower()
    if (
        model.get("parameter_count_is_dataset_dependent") is not True
        and not all(
            point.get("parameter_count_is_pre_run_maximum") is True
            for point in points
        )
        and not (
            "realized" in parameter_contract and "maximum" in parameter_contract
        )
    ):
        fail("{} does not distinguish realized dynamic-vocabulary size".format(name))

    if name.endswith("_stride"):
        if model.get("engineered_runtime_features") != []:
            fail("{} primary encoder retains engineered Stride features".format(name))
        if model.get("causal_runtime_feature_count") != 0:
            fail("{} primary encoder is not raw PC/address only".format(name))
    elif name.endswith("_spp"):
        if model.get("global_chronological_lstm") is not True:
            fail("{} is not a global chronological LSTM".format(name))

    # SPP may condition the supervised fill factor on the teacher target class
    # and rank.  That is output-loss factorization, not decoder feedback.  If
    # declared, it must remain explicitly loss-local and non-recurrent.
    fill_loss_conditioned = aliased_value(
        name, stream, model,
        (
            "teacher_target_conditions_loss_only_fill_factor",
            "teacher_target_used_for_loss_local_fill_conditioning",
        ),
    )
    if fill_loss_conditioned not in (None, False, True):
        fail("{} has an invalid fill-loss conditioning declaration".format(name))
    for key in (
        "teacher_action_values_used_as_main_rollout_recurrent_feedback",
        "teacher_prefix_used_as_main_rollout_recurrent_feedback",
        "teacher_count_used_as_decoder_feedback",
    ):
        if key in stream and stream[key] is not False:
            fail("{} uses teacher output in recurrent decoder state".format(name))

    train_path = track / "python" / "train_and_offline_infer.py"
    train_source = train_path.read_text()
    for token in (
        "SOURCE_ACTION_TEMPLATES",
        "legal_source_template",
        "normal_action_template",
    ):
        if token in train_source:
            fail("{} trainer retains hard-coded template token {}".format(
                name, token
            ))
    for required in (
        "delta_vocabulary_source",
        "delta_other_escape",
        "delta_other_decode_precision",
        "full_signed_line_delta_range_reachable",
        "every_signed_line_delta_exactly_representable",
        "exact_delta_representability_scope",
    ):
        if required not in train_source:
            fail("{} trainer omits {} provenance".format(name, required))


def validate_active_model_contract(track, stream, model):
    """Audit the active v25 contracts without copying implementation code."""
    name = track.name
    training_fields, inference_fields = contract_fields(stream)
    model_fields = model.get(
        "external_input_fields", model.get("source_inputs")
    )
    if list(model_fields or ()) != training_fields or training_fields != inference_fields:
        fail("{} stable model/runtime input fields differ".format(name))
    for key in ("run_id", "model_revision", "decoder_revision"):
        if not isinstance(model.get(key), str) or not model[key]:
            fail("{} stable model contract lacks {}".format(name, key))
        if key in stream and stream[key] != model[key]:
            fail("{} stream/model {} mismatch".format(name, key))
    stream_revision = stream.get("experiment_revision", stream.get("revision"))
    if model.get("experiment_revision") != stream_revision:
        fail("{} stable model/input revision mismatch".format(name))

    required_false = (
        "teacher_actions_are_model_inputs",
        "normal_policy_outputs_used_as_model_inputs",
        "normal_policy_candidates_used_as_model_inputs",
        "normal_policy_private_state_used_as_model_inputs",
        "normal_policy_request_rate_used_as_budget",
        "probability_threshold_used",
        "threshold_related_hardcodes_used",
        "inference_policy_hardcodes_used",
        "same_page_rule_used_by_neural_inference",
        "normal_policy_templates_used_by_neural_inference",
        "decoder_previous_teacher_action_used_as_input",
        "decoder_previous_predicted_action_used_as_input",
        "decoder_previous_sampled_action_used_as_input",
        "count_regression_used",
        "stop_token_used",
        "stop_padding_used",
        "loss_class_reweighting_used",
        "decode_prior_correction_used",
    )
    for key in required_false:
        if model.get(key) is not False:
            fail("{} must set {}=false".format(name, key))
    if model.get("neural_degree_cap") is not None:
        fail("{} inherits a fixed neural degree".format(name))
    if model.get("neural_role") != "standalone_direct_action_prefetcher":
        fail("{} is not an independent direct-action learner".format(name))
    if model.get("normal_policy_outputs_used_as_training_targets") is not True:
        fail("{} does not use source actions strictly as labels".format(name))
    if model.get("count_support_is_dataset_derived") is not True:
        fail("{} count support is not TRAIN-derived".format(name))
    if model.get("count_support_is_normal_request_budget") is not False:
        fail("{} count support copies the normal request rate".format(name))
    if model.get("count_support_is_tuned_degree") is not False:
        fail("{} count support is a tuned degree".format(name))
    if model.get("action_loss_scope") not in (
        "teacher_action_ranks_only", "real_teacher_action_ranks_only",
        "all_58_bits_of_every_real_teacher_rank",
    ):
        fail("{} creates loss outside real teacher ranks".format(name))
    mode = str(model.get("decoder_training_mode", "")).lower()
    if "without" not in mode or "feedback" not in mode:
        fail("{} decoder mode is not action-feedback-free".format(name))

    points = model.get("points")
    if not isinstance(points, list) or [
        point.get("size", point.get("model_size")) for point in points
    ] != [
        8, 16, 32, 64, 128
    ]:
        fail("{} must sweep h8,h16,h32,h64,h128".format(name))
    if not isinstance(model.get("parameter_formula"), (dict, str)) or not model.get(
        "parameter_formula"
    ):
        fail("{} lacks a realized parameter formula".format(name))
    if model.get("input_archive_reused_byte_for_byte") is not True:
        fail("{} does not require byte-identical input reuse".format(name))
    if model.get("operation") != "train-v25" or "v25" not in model.get(
        "run_id", ""
    ):
        fail("{} is not the active v25 training contract".format(name))

    if name.endswith("_stride"):
        if training_fields != ["pc", "addr"]:
            fail("{} changed the normal-Stride external input".format(name))
        if model.get("engineered_runtime_features") != []:
            fail("{} retains engineered Stride inputs".format(name))
        if model.get("causal_runtime_feature_count") != 0:
            fail("{} primary encoder is not raw PC/address only".format(name))
        for key in (
            "dual_context_core_used", "global_chronological_lstm_used",
            "exact_pc_local_lstm_used", "learned_global_local_fusion_used",
            "hurdle_head_used", "positive_only_categorical_count_head_used",
            "deterministic_target_uniqueness_constraint_used",
            "full_modular_line_delta_range_reachable",
        ):
            if model.get(key) is not True:
                fail("{} must set {}=true".format(name, key))
        if model.get("hurdle_classes") != ["ZERO", "POSITIVE"]:
            fail("{} lacks the natural ZERO/POSITIVE hurdle".format(name))
        if model.get("count_zero_is_implicit_hurdle") is not True:
            fail("{} must represent K=0 only through the hurdle".format(name))
        if model.get("hurdle_loss_class_weights") is not None:
            fail("{} reweights its sparse hurdle".format(name))
        if "unweighted" not in str(model.get(
            "hurdle_training_objective", ""
        )) or "unweighted" not in str(model.get(
            "positive_count_training_objective", ""
        )):
            fail("{} hurdle/count objectives are not natural unweighted CE".format(
                name
            ))
        if (
            model.get("delta_token_head_used") is not False
            or model.get("delta_vocabulary_used") is not False
            or model.get("delta_escape_head_used") is not False
            or model.get("rank_delta_payload_head")
            != "one_direct_58bit_modular_Bernoulli_head"
            or model.get("delta_decode_precision")
            != "exact_all_58_modular_bits"
            or model.get("action_loss_scope")
            != "all_58_bits_of_every_real_teacher_rank"
        ):
            fail("{} does not use direct all-rank 58-bit delta supervision".format(
                name
            ))
        routing = str(model.get("training_state_routing", "")).lower()
        if "global" not in routing or "local" not in routing or "exact_pc" not in routing:
            fail("{} lacks global plus exact-PC-local chronology".format(name))
        if model.get("fill_level") != "FILL_L2_only_no_fill_head":
            fail("{} introduces an unnecessary Stride fill head".format(name))
        if model.get("weights_retrained") is not True:
            fail("{} v25 must train from scratch".format(name))
        if model.get("checkpoint_reused") is not False:
            fail("{} v25 must not reuse a parent checkpoint".format(name))
        if model.get("original_guard_used_for_selection") is not False:
            fail("{} uses phase-shift GUARD for selection".format(name))
        if model.get("evaluation_used_for_selection") is not False:
            fail("{} leaks EVAL into selection".format(name))
        selection = model.get("selection_protocol")
        if selection != {
            "fit": "first_80_percent_of_TRAIN",
            "validation": "last_20_percent_of_TRAIN",
            "selection_support": "FIT_only",
            "metric": "complete_validation_NLL_per_callback",
            "tie_break": "earlier_epoch",
        }:
            fail("{} lacks the fixed chronological 80/20 selection".format(name))
        final_training = str(model.get("final_training_protocol", "")).lower()
        if not all(token in final_training for token in (
            "reset_seed", "reinitialize", "retrain_from_scratch",
            "complete_train", "selected_epoch",
        )):
            fail("{} does not refit complete TRAIN from scratch".format(name))
        if (
            model.get("positive_count_support_source_selection")
            != "FIT_labels_only"
            or model.get("positive_count_support_source_final")
            != "complete_TRAIN_labels_only"
            or model.get("delta_bit_prior_source_selection")
            != "all_real_FIT_teacher_actions"
            or model.get("delta_bit_prior_source_final")
            != "all_real_complete_TRAIN_teacher_actions"
        ):
            fail("{} selection/final count or bit priors use the wrong partition".format(
                name
            ))
        if model.get("decoded_target_projection_or_mutation_used") is not False:
            fail("{} mutates a decoded target to force uniqueness".format(name))
    elif name.endswith("_spp"):
        if training_fields != [
            "callback_kind", "invoke_prefetcher.addr",
            "cache_fill.evicted_addr",
        ]:
            fail("{} changed the source-SPP external input".format(name))
        if model.get("model_does_not_use_pc") is not True:
            fail("{} unexpectedly consumes PC".format(name))
        if model.get("count_head_used") is not True:
            fail("{} lacks a natural categorical K head".format(name))
        if model.get("count_zero_is_implicit_hurdle") is not True:
            fail("{} does not use K=0 as the no-request class".format(name))
        if "unweighted" not in str(model.get("count_training_objective", "")):
            fail("{} count objective is not natural unweighted CE".format(name))
        for key in (
            "joint_action_token_head_used", "action_vocabulary_used",
            "other_token_used",
        ):
            if model.get(key) is not False:
                fail("{} must set {}=false".format(name, key))
        if (
            model.get("core_type") != "global"
            or model.get("core_selection_used") is not False
            or model.get("event_routed_core_used") is not False
            or "one_global_chronological" not in str(model.get("global_core", ""))
        ):
            fail("{} is not one fixed global chronological LSTM".format(name))
        if (
            model.get("fill_head_used") is not True
            or model.get("fill_specific_delta_bit_heads_used") is not True
            or model.get("both_fill_bit_heads_require_train_supervision")
            is not True
            or "unweighted" not in str(model.get("fill_training_objective", ""))
            or "58_bit" not in str(model.get("delta_bit_training_objective", ""))
            or model.get("delta_payload_encoding")
            != "exact_58_bit_modular_line_delta"
            or model.get("delta_payload_float_or_clip_used") is not False
        ):
            fail("{} lacks direct fill plus teacher-fill-specific bit supervision".format(
                name
            ))
        if (
            model.get("target_uniqueness_feasibility_mask_used") is not True
            or model.get("target_uniqueness_ignores_fill_level") is not True
            or model.get("infeasible_unique_decode_behavior") != "fail_closed"
            or model.get(
                "rank_logits_conditionally_independent_of_previous_actions"
            ) is not True
            or model.get("kbest_payload_enumeration_exact") is not True
            or model.get("fill_and_payload_log_probability_combined") is not True
        ):
            fail("{} does not enforce deterministic unique targets".format(name))
        if model.get("training_and_guard_objective_identical") is not True:
            fail("{} selects on a different objective than it trains".format(name))
    else:
        fail("unexpected active track {}".format(name))

    train_source = (track / "python" / "train_and_offline_infer.py").read_text()
    for forbidden in (
        "SOURCE_ACTION_TEMPLATES", "legal_source_template",
        "normal_action_template",
    ):
        if forbidden in train_source:
            fail("{} trainer retains hard-coded template {}".format(name, forbidden))


def validate_track(track):
    name = track.name
    stream_path = track / "data" / "stream_contract.json"
    if not stream_path.is_file():
        fail("{} has no data/stream_contract.json".format(name))
    stream = read_json(stream_path)
    validate_runtime_input_boundary(name, stream)

    notebook_paths = tuple((track / "colab").glob("*.ipynb"))
    if len(notebook_paths) != 1:
        fail("{} must have exactly one Colab notebook".format(name))
    notebook = notebook_source(notebook_paths[0])

    for relative in (
        "python/train_and_offline_infer.py",
        "python/analyze_replay.py",
    ):
        path = track / relative
        if not path.is_file():
            fail("{} is missing {}".format(name, relative))
        parse_python(path)

    analyzer = (track / "python" / "analyze_replay.py").read_text()
    for token in (
        "training_runtime_encoder_sha256",
        "inference_runtime_encoder_sha256",
    ):
        if token not in analyzer:
            fail("{} analyzer omits {}".format(name, token))

    if name in ACTIVE_MODEL_CONTRACT_TRACKS:
        model = load_model_contract(track)
        validate_active_model_contract(track, stream, model)
        # Active notebooks deliberately derive run/model/decoder identifiers
        # from the stable contract.  Requiring duplicated literal values here
        # would make the notebook a second, drift-prone source of truth.
        if "--describe-model-points" not in notebook:
            fail("{} notebook does not query the stable model contract".format(
                name
            ))
        for token in (
            "files.upload()",
            "safe_extract_tar_gz",
            "validate_sha256sums",
            "validate_collected_inputs.py",
            "stderr=subprocess.STDOUT",
            "files.download(str(OUTPUT_ARCHIVE))",
            "[8,16,32,64,128]",
            "trainer.stdout_stderr.log",
        ):
            if token not in notebook:
                fail("{} notebook lacks active training token {}".format(
                    name, token
                ))
        if name.endswith("_spp"):
            for token in (
                "core-ablation", "--core-selection-file",
                "core_selection_uses_evaluation", "evaluation_files_loaded",
            ):
                if token in notebook:
                    fail("{} notebook retains stale core-selection token {}".format(
                        name, token
                    ))
        for relative in ("linux/run_server.sh", "linux/launch_server.sh"):
            source = (track / relative).read_text()
            if model["run_id"] not in source and "model_contract.py" not in source:
                fail("{} does not derive {} from the stable model contract".format(
                    relative, name
                ))
        run_server = (track / "linux" / "run_server.sh").read_text()
        for token in (
            '[[ "$MODEL_TAGS_CSV" == "$DEFAULT_MODEL_TAGS" ]]',
            "require_safe_path_token RUN_ID",
            "assert_active_model_metadata",
            "analyze() {\n  require_colab_outputs",
        ):
            if token not in run_server:
                fail("{} run server lacks fail-closed token {}".format(
                    name, token
                ))
        launcher = (track / "linux" / "launch_server.sh").read_text()
        if "RUN_ID must be one safe path token" not in launcher:
            fail("{} launcher does not reject unsafe RUN_ID values".format(name))
    else:
        mode = stream.get("decoder_training_mode", "")
        if mode and "free_running" not in str(mode):
            fail("{} legacy decoder is not free-running".format(name))
        if stream.get("decoder_previous_teacher_action_used_as_input") is not False:
            fail("{} legacy decoder permits teacher feedback".format(name))


def validate_portability():
    # Sacramento's system Python is intentionally treated as the portability
    # floor for server-side audit and replay scripts.
    future_annotations = "from __future__ import " + "annotations"
    paths = tuple(COMMON.rglob("*.py")) + tuple(EXPERIMENTS.rglob("*.py"))
    for path in paths:
        source = path.read_text()
        if future_annotations in source:
            fail("{} requires unsupported future annotations".format(path))
        tree = ast.parse(source, filename=str(path))
        if "pandas" in imported_roots(tree):
            fail("{} imports unavailable pandas".format(path))

    installer = COMMON / "install_colab_output.py"
    if installer.is_file():
        installer_source = installer.read_text()
        for token in (
            "def validate_members(",
            "member.issym()",
            'handle.extractall(str(output_dir), members=members)',
            "COMMON_REQUIRED_FILES",
        ):
            if token not in installer_source:
                fail("Colab output installer missing {}".format(token))

    transfer = COMMON / "split_colab_archive.py"
    transfer_source = transfer.read_text()
    for token in (
        "MAX_PART_MIB = 90", "def split_archive(", "def validate_parts(",
        "def reassemble_archive(", "def safe_extract_tar_gz(",
        "def validate_sha256sums(", "duplicate tar member",
    ):
        if token not in transfer_source:
            fail("multipart transfer helper missing {}".format(token))
    if "from __future__ import " + "annotations" in transfer_source:
        fail("multipart transfer helper is not Python-3.6 compatible")

    ignore = (ROOT / ".gitignore").read_text()
    for token in (
        "**/*.colab_input.tar.gz.part-*",
        "**/*.colab_output.tar.gz.part-*",
        "**/*.colab_input.tar.gz.parts.json",
        "**/*.colab_output.tar.gz.parts.json",
    ):
        if token not in ignore:
            fail(".gitignore does not exclude {}".format(token))


def validate_spp_boundaries():
    spp_names = (
        "602_offline_lstm_spp",
        "623_offline_lstm_spp",
        "623_offline_cnn_spp",
    )
    for name in spp_names:
        track = EXPERIMENTS / name
        for filename in ("normalize_events.py", "validate_collected_inputs.py"):
            path = track / "python" / filename
            source = path.read_text()
            if "target_page_offset" in source:
                fail("{} exposes a teacher page-offset interface".format(name))

    lstm_patch = (
        EXPERIMENTS
        / "623_offline_lstm_spp/linux/patch_demand_logger.sh"
    ).read_bytes()
    cnn_patch = (
        EXPERIMENTS
        / "623_offline_cnn_spp/linux/patch_demand_logger.sh"
    ).read_bytes()
    if lstm_patch != cnn_patch:
        fail("SPP LSTM/CNN source-input logger patches differ")


def main():
    validate_portability()
    if (EXPERIMENTS / "623_offline_lstm_cnn_stride_spp").exists():
        fail("obsolete combined 623 directory still exists")

    for name in EXPECTED_TRACKS:
        track = EXPERIMENTS / name
        if not track.is_dir():
            fail("missing direct-action track {}".format(name))
        validate_track(track)
    validate_spp_boundaries()

    print("[PASS] eight matched-input direct-action tracks satisfy the static contract")
    print("[PASS] active 623 contracts are loaded from stable model_contract.py files")
    print("[PASS] active 623 uses no threshold, normal template, page rule, or degree cap")
    print("[PASS] Stride v25 uses dual-context natural hurdle/count and direct 58-bit rank payloads")
    print("[PASS] SPP v25 uses one global LSTM, natural K/fill heads, and direct 58-bit rank payloads")
    print("[PASS] active Colab notebooks use one validated input/output archive")


if __name__ == "__main__":
    main()
