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
ACTIVE_MODEL_SEMANTICS = {
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


def validate_active_model_contract(track, stream, model):
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

    for key, expected in ACTIVE_MODEL_SEMANTICS.items():
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
    if not isinstance(mode, str) or not any(
        token in mode for token in (
            "free_running", "teacher_labels_only", "no_action_feedback",
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

    rank_objective = aliased_value(
        name, stream, model,
        ("rank_decision_training_objective", "stop_emit_training_objective"),
        required=True,
    )
    lowered_rank_objective = str(rank_objective).lower()
    if "stop" not in lowered_rank_objective or "emit" not in lowered_rank_objective:
        fail("{} does not train a direct STOP/EMIT rank decision".format(name))
    if "threshold" in lowered_rank_objective:
        fail("{} STOP/EMIT objective embeds a threshold".format(name))
    terminal_stop = aliased_value(
        name, stream, model,
        (
            "terminal_stop_supervised",
            "terminal_stop_supervised_for_every_teacher_sequence",
        ),
        required=True,
    )
    if terminal_stop is not True:
        fail("{} does not supervise a terminal STOP".format(name))

    for key in ("separate_global_gate_used", "separate_count_head_used", "log_count_used"):
        if key in model and same_value(name, stream, model, key) is not False:
            fail("{} retains the failed gate/count factorization at {}".format(
                name, key
            ))
    previous_sampled = aliased_value(
        name, stream, model,
        (
            "decoder_previous_predicted_action_used_as_input",
            "decoder_previous_sampled_action_used_as_input",
        ),
        required=True,
    )
    if previous_sampled is not False:
        fail("{} feeds a previous decoded action to the next rank".format(name))

    selection = model.get(
        "checkpoint_selection", model.get("guard_selection_rule")
    )
    if selection is None:
        fail("{} stable model contract lacks a guard selection rule".format(name))
    lowered_selection = str(selection).lower()
    if "lexicographic" not in lowered_selection:
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
        if "inverse_frequency" not in lowered_rank_objective:
            fail("{} sparse STOP/EMIT loss is not TRAIN-balanced".format(name))
        if model.get("engineered_runtime_features") != []:
            fail("{} primary encoder retains engineered Stride features".format(name))
        if model.get("causal_runtime_feature_count") != 0:
            fail("{} primary encoder is not raw PC/address only".format(name))
    elif name.endswith("_spp"):
        if not any(token in lowered_rank_objective for token in ("unweighted", "natural")):
            fail("{} dense STOP/EMIT loss does not preserve its natural prior".format(name))
        if model.get("stochastic_decoding") is not False:
            fail("{} must use deterministic action decoding".format(name))
        if "argmax" not in str(model.get("fill_decoding_rule", "")).lower():
            fail("{} fill decoding is not deterministic argmax".format(name))
        if model.get("fill_prior_correction_at_decode_used") is not True:
            fail("{} does not undo TRAIN fill reweighting at decode".format(name))

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
            "drive.mount(", "files.upload()", "split_colab_archive",
            ".parts.json", "MAX_PART_BYTES", "[8,16,32,64,128]",
        ):
            if token not in notebook:
                fail("{} notebook lacks multipart/Drive token {}".format(
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
            "active v21 replay requires the exact five configured MODEL_TAGS",
            "require_safe_path_token RUN_ID",
            "assert_model_metadata_v21",
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
    print("[PASS] active 623 uses rankwise STOP/EMIT with dynamic TRAIN vocabularies")
    print("[PASS] Colab transfer is SHA-verified and split into at-most-90-MiB parts")


if __name__ == "__main__":
    main()
