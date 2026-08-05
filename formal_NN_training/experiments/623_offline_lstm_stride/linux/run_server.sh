#!/usr/bin/env bash
# Independent 623 track: normal Stride versus direct LSTM only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/623_offline_lstm_stride"
TRAINER="$EXP/python/train_and_offline_infer.py"
MODEL_CONTRACT="$EXP/python/model_contract.py"
TRACE="$(python3 "$MODEL_CONTRACT" --field trace)"
POLICY="$(python3 "$MODEL_CONTRACT" --field policy)"
DEFAULT_RUN_ID="$(python3 "$MODEL_CONTRACT" --field run_id)"
RUN_ID="${RUN_ID:-$DEFAULT_RUN_ID}"
STAGE="${STAGE:-replay}"
FORCE="${FORCE:-0}"
JOBS="${JOBS:-8}"
BUILD="${BUILD:-1}"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
TRACE_FILE="${TRACE_FILE:-$ROOT/traces/$TRACE.champsimtrace.xz}"
RUN_DIR="${RUN_DIR:-$EXP/runs/$RUN_ID}"
LOG_DIR="$RUN_DIR/logs"
EVENT_DIR="$RUN_DIR/events"
STREAM_DIR="$RUN_DIR/colab_input"
COLAB_ROOT="$RUN_DIR/colab_output"
BIN="${BIN:-$CHAMP_DIR/bin/champsim.623_stride_lstm_replay}"
EXPECTED_LIBBF_HEAD="4c9efc1a4db7ed1ccf54cf0bd3a3641ce579206c"
BUILD_LOCK="${BUILD_LOCK:-$(git -C "$ROOT" rev-parse --absolute-git-dir)/champsim_build.lock}"

PATCH_LOGGER="$EXP/linux/patch_demand_logger.sh"
BUILD_REPLAYER="$EXP/linux/build_keyed_replayer.sh"
NORMALIZE="$EXP/python/normalize_events.py"
VALIDATE_INPUTS="$EXP/python/validate_collected_inputs.py"
ANALYZE="$EXP/python/analyze_replay.py"
INSTALL_COLAB_OUTPUT="$ROOT/formal_NN_training/common/install_colab_output.py"
COLLECTION_MANIFEST="$STREAM_DIR/collection_manifest.json"

DEFAULT_MODEL_TAGS="$(python3 "$MODEL_CONTRACT" --tags-csv)"
DEFAULT_BASE_TAG="$(python3 "$MODEL_CONTRACT" --base-tag)"
MODEL_TAGS_CSV="${MODEL_TAGS:-$DEFAULT_MODEL_TAGS}"
BASE_TAG="${BASE_TAG:-$DEFAULT_BASE_TAG}"

IFS=',' read -r -a MODEL_TAGS <<< "$MODEL_TAGS_CSV"
[[ "${#MODEL_TAGS[@]}" -gt 0 ]] || { echo "[error] MODEL_TAGS is empty" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$EVENT_DIR" "$STREAM_DIR" "$COLAB_ROOT"

require_repo_file() {
  [[ -f "$1" ]] || {
    echo "[error] missing required repository file $1" >&2
    exit 2
  }
}
for required_file in \
  "$PATCH_LOGGER" "$BUILD_REPLAYER" "$NORMALIZE" "$VALIDATE_INPUTS" \
  "$ANALYZE" "$TRAINER" "$MODEL_CONTRACT" "$INSTALL_COLAB_OUTPUT"; do
  require_repo_file "$required_file"
done

ensure_libbf() {
  if [[ -e "$CHAMP_DIR/libbf" && ! -d "$CHAMP_DIR/libbf/.git" ]]; then
    echo "[error] $CHAMP_DIR/libbf exists but is not the expected git checkout" >&2
    exit 2
  fi
  if [[ ! -d "$CHAMP_DIR/libbf/.git" ]]; then
    git clone https://github.com/mavam/libbf.git "$CHAMP_DIR/libbf"
    git -C "$CHAMP_DIR/libbf" checkout --detach "$EXPECTED_LIBBF_HEAD"
  fi
  local observed
  observed="$(git -C "$CHAMP_DIR/libbf" rev-parse HEAD)"
  [[ "$observed" == "$EXPECTED_LIBBF_HEAD" ]] || {
    echo "[error] libbf HEAD $observed != pinned $EXPECTED_LIBBF_HEAD" >&2
    exit 2
  }
  if [[ ! -f "$CHAMP_DIR/libbf/build/lib/libbf.a" ]]; then
    cmake -S "$CHAMP_DIR/libbf" -B "$CHAMP_DIR/libbf/build"
    cmake --build "$CHAMP_DIR/libbf/build" -j"$JOBS"
  fi
}

build() {
  command -v flock >/dev/null 2>&1 || {
    echo "[error] flock is required for safe ChampSim builds" >&2
    exit 2
  }
  mkdir -p "$(dirname "$BUILD_LOCK")"
  (
    echo "[build-lock] waiting for $BUILD_LOCK"
    flock -x 9
    echo "[build-lock] acquired by 623 stride run $RUN_ID"
    RUN_DIR="$RUN_DIR" RESET_PATCH="${RESET_PATCH:-0}" \
      CHAMP_DIR="$CHAMP_DIR" bash "$PATCH_LOGGER"
    ensure_libbf
    CHAMP_DIR="$CHAMP_DIR" OUT="$BIN" bash "$BUILD_REPLAYER"
    echo "[build-lock] 623 stride build complete"
  ) 9>"$BUILD_LOCK"
}

assert_live_policy() {
  local log="$1"
  grep -Eiq 'adding L2C_PREFETCHER:.*stride' "$log" || {
    echo "[error] live stride was not registered" >&2
    exit 3
  }
  grep -Fq 'stride_num_trackers 64' "$log" || {
    echo "[error] stride did not use 64 trackers" >&2
    exit 3
  }
  grep -Fq 'stride_pref_degree 2' "$log" || {
    echo "[error] stride did not use degree 2" >&2
    exit 3
  }
  grep -Eq '^stride_pref_generated [1-9][0-9]*$' "$log" || {
    echo "[error] stride generated zero candidates" >&2
    exit 3
  }
  grep -Eq '^Core_0_L2C_prefetch_dropped 0$' "$log" || {
    echo "[error] stride dropped requests; captured candidate bank is incomplete" >&2
    exit 3
  }
}

run_policy_events() {
  local role="$1" warmup="$2" simulation="$3"
  local raw="$EVENT_DIR/$TRACE.$POLICY.$role.events.csv"
  local gz="$raw.gz"
  local log="$LOG_DIR/$TRACE.$POLICY.$role.collect.log"
  if [[ "$FORCE" != 1 && -s "$gz" && -s "$log" ]] && gzip -t "$gz"; then
    echo "[skip] $POLICY $role"
    return
  fi
  rm -f "$raw" "$gz"
  DEMAND_EVENT_LOG="$raw" "$BIN" \
    --l2c_prefetcher_types=stride \
    --stride_num_trackers=64 --stride_pref_degree=2 \
    --warmup_instructions="$warmup" \
    --simulation_instructions="$simulation" \
    -traces "$TRACE_FILE" > "$log" 2>&1
  assert_live_policy "$log"
  [[ -s "$raw" ]] || { echo "[error] missing event output for $role" >&2; exit 3; }
  gzip -f "$raw"
  gzip -t "$gz"
}

assert_collection_count() {
  local role="$1"
  local log="$LOG_DIR/$TRACE.$POLICY.$role.collect.log"
  local stream="$STREAM_DIR/$TRACE.$POLICY.${role}_stream.csv.gz"
  python3 - "$log" "$stream" "$role" <<'PY'
import csv
import gzip
import re
import sys

log_path, stream_path, role = sys.argv[1:]
matches = re.findall(
    r"^Core_0_L2C_loads\s+(\d+)\s*$",
    open(log_path, errors="ignore").read(),
    flags=re.MULTILINE,
)
if not matches:
    raise SystemExit("missing Core_0_L2C_loads in {}".format(log_path))
expected = int(matches[-1])
with gzip.open(stream_path, "rt", newline="") as handle:
    observed = sum(1 for _ in csv.DictReader(handle))
if observed != expected:
    raise SystemExit(
        "stride {} completed demand callbacks {} != simulator L2 loads {}".format(
            role, observed, expected
        )
    )
print("[PASS] stride {} demand callbacks={}".format(role, observed))
PY
}

collect() {
  [[ -s "$TRACE_FILE" ]] || { echo "[error] missing trace $TRACE_FILE" >&2; exit 2; }
  build
  local role warmup simulation
  local input_files=()
  for role in train guard eval; do
    case "$role" in
      train) warmup=0; simulation=20000000 ;;
      guard) warmup=20000000; simulation=5000000 ;;
      eval) warmup=25000000; simulation=25000000 ;;
    esac
    run_policy_events "$role" "$warmup" "$simulation"
    python3 "$NORMALIZE" \
      --events "$EVENT_DIR/$TRACE.$POLICY.$role.events.csv.gz" \
      --policy "$POLICY" \
      --stream-out "$STREAM_DIR/$TRACE.$POLICY.${role}_stream.csv.gz" \
      --candidate-out "$STREAM_DIR/$TRACE.$POLICY.${role}_candidates.csv.gz"
    assert_collection_count "$role"
    input_files+=(
      "$TRACE.$POLICY.${role}_stream.csv.gz"
      "$TRACE.$POLICY.${role}_candidates.csv.gz"
    )
  done
  python3 "$VALIDATE_INPUTS" \
    --input-dir "$STREAM_DIR" --manifest-out "$COLLECTION_MANIFEST"
  input_files+=("collection_manifest.json")
  ( cd "$STREAM_DIR" && sha256sum "${input_files[@]}" > SHA256SUMS )
  tar -C "$STREAM_DIR" -czf "$RUN_DIR/$RUN_ID.colab_input.tar.gz" \
    "${input_files[@]}" SHA256SUMS
  echo "[ready for Colab] $RUN_DIR/$RUN_ID.colab_input.tar.gz"
}

validate_preserved_inputs() {
  local validated_manifest
  validated_manifest="$(mktemp "$RUN_DIR/.stride_collection_manifest.XXXXXX")"
  if ! python3 "$VALIDATE_INPUTS" \
    --input-dir "$STREAM_DIR" --manifest-out "$validated_manifest"; then
    rm -f "$validated_manifest"
    return 1
  fi
  # The archived collection_manifest.json intentionally records the historical
  # v9 collection semantics.  v20's validator adds output-design profiling, so
  # compare bytes through the archived SHA256SUMS and use the fresh manifest as
  # a semantic validation result; do not rewrite the reused input package.
  rm -f "$validated_manifest"
  ( cd "$STREAM_DIR" && sha256sum -c SHA256SUMS )
}

colab_dir() { printf '%s/%s' "$COLAB_ROOT" "$1"; }

# Active v20 validation.  The legacy validator above remains readable for old
# archived runs, but replay calls only this contract-focused path.
assert_model_metadata_v20() {
  python3 - "$1" "$POLICY" "$TRAINER" <<'PY'
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

metadata_path, policy, trainer = sys.argv[1:]
metadata = json.load(open(metadata_path))
contract = json.loads(subprocess.check_output(
    [sys.executable, trainer, "--describe-model-points"]
).decode("utf-8"))
trainer_path = Path(trainer).resolve()
contract_path = trainer_path.with_name("model_contract.py")
common_policy_path = (
    trainer_path.parents[4] / "formal_NN_training" / "common"
    / "threshold_free_policy.py"
)
source_hashes = {
    "trainer_source_sha256": hashlib.sha256(
        trainer_path.read_bytes()
    ).hexdigest(),
    "model_contract_source_sha256": hashlib.sha256(
        contract_path.read_bytes()
    ).hexdigest(),
    "threshold_free_policy_source_sha256": hashlib.sha256(
        common_policy_path.read_bytes()
    ).hexdigest(),
}
tag = metadata.get("model_tag")
expected = {
    "trace": contract["trace"],
    "matched_normal_prefetcher": policy,
    "neural_role": "standalone_direct_action_prefetcher",
    "source_decision_effective_external_input": ["pc", "addr"],
    "same_external_input_contract": True,
    "training_runtime_fields": ["pc", "addr"],
    "inference_runtime_fields": ["pc", "addr"],
    "training_inference_input_encoder_identical": True,
    "normal_policy_outputs_used_as_model_inputs": False,
    "normal_policy_candidates_used_as_model_inputs": False,
    "normal_policy_private_state_used_as_model_inputs": False,
    "normal_policy_outputs_used_as_training_targets": True,
    "normal_policy_request_rate_used_as_budget": False,
    "normal_policy_constants_used_by_neural_inference": False,
    "normal_policy_templates_used_by_neural_inference": False,
    "probability_threshold_used": False,
    "threshold_related_hardcodes_used": False,
    "inference_policy_hardcodes_used": False,
    "neural_degree_cap": None,
    "fixed_page_offset_classes": None,
    "same_page_rule_used_by_neural_inference": False,
    "future_label_window_used": False,
    "derived_features_use_teacher_or_future": False,
    "decoder_training_mode": contract["decoder_training_mode"],
    "decoder_previous_teacher_action_used_as_input": False,
    "decoder_previous_predicted_action_used_as_input": False,
    "all_teacher_ranks_supervised": True,
    "delta_vocabulary_source": "train_labels_only",
    "delta_vocabulary_max_exact": 255,
    "delta_class_bias_initialization": "log_add_one_smoothed_TRAIN_exact_plus_OTHER_frequency",
    "positive_log_count_bias_initialization": "TRAIN_positive_mean_log_count",
    "delta_other_escape": "signed_log_continuous_bounded_approximation",
    "delta_other_decode_precision": "rounded_float32_approximate_except_exact_vocabulary",
    "delta_coordinate_auxiliary_trained_on_all_teacher_actions": True,
    "delta_coordinate_used_for_decode_only_on_other": True,
    "full_signed_line_delta_range_reachable": False,
    "every_signed_line_delta_exactly_representable": False,
    "exact_delta_representability_scope": "train_vocabulary_only",
    "gate_training_objective": contract["gate_training_objective"],
    "positive_count_training_objective": contract[
        "positive_count_training_objective"
    ],
    "delta_training_objective": contract["delta_training_objective"],
    "deterministic_decoding": True,
    "stochastic_decoding": False,
    "decoder_sampling_roles": [],
    "decoder_train_sampling_performed": False,
    "decoder_guard_sampling_performed": False,
    "decoder_eval_sampling_performed": False,
    "sampled_outputs_used_as_decoder_feedback": False,
    "decode_per_callback_resource_watchdog": contract[
        "decode_per_callback_resource_watchdog"
    ],
    "decode_per_role_resource_watchdog": contract[
        "decode_per_role_resource_watchdog"
    ],
    "decode_resource_watchdog_behavior": contract[
        "decode_resource_watchdog_behavior"
    ],
    "decode_resource_watchdog_is_neural_degree_cap": False,
    "successful_run_hit_decode_resource_watchdog": False,
    "checkpoint_selection": contract["checkpoint_selection"],
    "checkpoint_selection_roles": ["guard"],
    "guard_role": "checkpoint_selection_only_no_threshold_calibration",
    "evaluation_used_for_checkpoint_selection": False,
    "evaluation_decode_passes": 1,
    "runtime_feature_count": contract["runtime_feature_count"],
    "raw_runtime_feature_count": contract["raw_runtime_feature_count"],
    "causal_runtime_feature_count": contract["causal_runtime_feature_count"],
    "training_state_mode": "exact_pc_keyed_stateful_tbptt",
    "training_state_carried_across_chunks": True,
    "training_state_detached_between_chunks": True,
    "experiment_revision": contract["experiment_revision"],
    "model_revision": contract["model_revision"],
    "decoder_revision": contract["decoder_revision"],
    "track_model_family": "lstm",
    "training_config": contract["training_config"],
    "training_config_pinned_by_run_id": True,
    "training_device": "cuda",
    "cublas_workspace_config": contract["determinism_contract"][
        "cublas_workspace_config"
    ],
    "torch_deterministic_algorithms_enabled": True,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "float32_matmul_precision": contract["determinism_contract"][
        "float32_matmul_precision"
    ],
}
expected.update(source_hashes)
bad = {
    key: (metadata.get(key), value)
    for key, value in expected.items() if metadata.get(key) != value
}
for key, value in contract["training_config"].items():
    if metadata.get(key) != value:
        bad["pinned_training_" + key] = (metadata.get(key), value)
if contract["determinism_contract"][
    "required_accelerator_name_contains"
] not in str(metadata.get("training_device_name")):
    bad["training_device_name"] = (
        metadata.get("training_device_name"),
        contract["determinism_contract"]["required_accelerator_name_contains"],
    )
points = {point["model_size"]: point for point in contract["points"]}
point = points.get(metadata.get("model_size"))
if metadata.get("model_family") != "lstm" or point is None:
    bad["model_point"] = (
        (metadata.get("model_family"), metadata.get("model_size")),
        sorted(points),
    )
else:
    for key in ("architecture_pair_id", "parameter_count", "model_tag"):
        expected_value = point[key]
        if metadata.get(key) != expected_value:
            bad[key] = (metadata.get(key), expected_value)

for key in (
    "runtime_encoder_sha256", "training_runtime_encoder_sha256",
    "inference_runtime_encoder_sha256", "training_state_router_sha256",
    "inference_state_router_sha256",
):
    if re.fullmatch(r"[0-9a-f]{64}", str(metadata.get(key))) is None:
        bad[key] = (metadata.get(key), "64 lowercase hex characters")
if len({
    metadata.get("runtime_encoder_sha256"),
    metadata.get("training_runtime_encoder_sha256"),
    metadata.get("inference_runtime_encoder_sha256"),
}) != 1:
    bad["runtime_encoder_hash_equality"] = ("different", "identical")
if metadata.get("training_state_router_sha256") != metadata.get(
    "inference_state_router_sha256"
):
    bad["state_router_hash_equality"] = ("different", "identical")

vocabulary = metadata.get("delta_vocabulary_exact")
frequencies = metadata.get("delta_vocabulary_train_frequencies")
size = metadata.get("delta_vocabulary_exact_size")
if (
    not isinstance(vocabulary, list) or not isinstance(frequencies, list)
    or not isinstance(size, int) or isinstance(size, bool)
    or not 0 < size <= 255 or len(vocabulary) != size
    or len(frequencies) != size or len(set(vocabulary)) != size
    or any(not isinstance(value, int) or isinstance(value, bool)
           for value in vocabulary + frequencies)
    or any(value <= 0 for value in frequencies)
):
    bad["delta_vocabulary"] = (
        {"size": size, "vocabulary": vocabulary, "frequencies": frequencies},
        "one unique TRAIN-frequency vocabulary of size 1..255",
    )
statistics = metadata.get("delta_vocabulary_statistics") or {}
if (
    set(statistics) != {"train", "guard", "eval"}
    or not isinstance(frequencies, list)
):
    bad["delta_vocabulary_statistics"] = (
        sorted(statistics), ["train", "guard", "eval"]
    )
else:
    train_vocabulary = statistics.get("train") or {}
    other_count = train_vocabulary.get("other_escape_actions")
    teacher_actions = train_vocabulary.get("teacher_actions")
    class_prior = metadata.get("delta_class_empirical_prior")
    class_bias = metadata.get("delta_class_initial_bias")
    if (
        not isinstance(other_count, int) or not isinstance(teacher_actions, int)
        or not isinstance(class_prior, list) or len(class_prior) != 256
        or not isinstance(class_bias, list) or len(class_bias) != 256
    ):
        bad["delta_class_prior"] = (
            (other_count, teacher_actions, class_prior, class_bias),
            "256 add-one-smoothed TRAIN class priors and log biases",
        )
    else:
        class_counts = [0] * 256
        for index, value in enumerate(frequencies):
            class_counts[index] = value
        class_counts[255] = other_count
        denominator = float(teacher_actions + 256)
        expected_prior = [
            (value + 1.0) / denominator for value in class_counts
        ]
        if any(
            not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9)
            for actual, expected in zip(class_prior, expected_prior)
        ) or any(
            not math.isclose(actual, math.log(expected), rel_tol=1e-6,
                             abs_tol=1e-7)
            for actual, expected in zip(class_bias, expected_prior)
        ):
            bad["delta_class_prior"] = (
                (class_prior, class_bias),
                "add-one-smoothed TRAIN class prior and its log",
            )

count_stats = metadata.get("request_count_training_label_statistics") or {}
if (
    not isinstance(count_stats.get("decision_callbacks"), int)
    or count_stats.get("decision_callbacks", 0) <= 0
    or count_stats.get("positive_callbacks", 0) <= 0
    or count_stats.get("zero_callbacks", 0) <= 0
    or count_stats.get("positive_callbacks", 0)
       + count_stats.get("zero_callbacks", 0)
       != count_stats.get("decision_callbacks")
):
    bad["request_count_training_label_statistics"] = (
        count_stats, "nonempty natural zero/positive TRAIN labels"
    )
else:
    distribution = count_stats.get("count_distribution") or {}
    expected_log_count_bias = sum(
        int(value) * math.log(int(key))
        for key, value in distribution.items() if int(key) > 0
    ) / float(count_stats["positive_callbacks"])
    actual_log_count_bias = metadata.get("positive_log_count_initial_bias")
    if (
        not isinstance(actual_log_count_bias, (int, float))
        or not math.isclose(
            actual_log_count_bias, expected_log_count_bias,
            rel_tol=1e-6, abs_tol=1e-7,
        )
    ):
        bad["positive_log_count_initial_bias"] = (
            actual_log_count_bias, expected_log_count_bias
        )
prior = metadata.get("gate_empirical_prior")
bias = metadata.get("gate_initial_bias")
if (
    not isinstance(prior, list) or not isinstance(bias, list)
    or len(prior) != 2 or len(bias) != 2
    or any(not isinstance(value, (int, float)) or value <= 0 for value in prior)
    or not math.isclose(sum(prior), 1.0, rel_tol=1e-7, abs_tol=1e-8)
    or any(not math.isclose(math.log(p), b, rel_tol=1e-6, abs_tol=1e-7)
           for p, b in zip(prior, bias))
):
    bad["gate_prior_bias"] = ((prior, bias), "natural prior and its log")
epoch = metadata.get("selected_guard_epoch")
if not isinstance(epoch, int) or isinstance(epoch, bool) or not (
    1 <= epoch <= metadata.get("epochs", 0)
):
    bad["selected_guard_epoch"] = (epoch, "within trained epochs")

def inspect_replay(path, allow_empty):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    count = 0
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != ["pc", "line", "occ", "prefetch_addr"]:
            raise SystemExit("invalid stride replay header in {}".format(path))
        for line_number, fields in enumerate(reader, 2):
            if len(fields) != 4:
                raise SystemExit("invalid stride replay row {}".format(line_number))
            pc, line, occ, address = (
                int(fields[0], 0), int(fields[1], 0),
                int(fields[2], 10), int(fields[3], 0),
            )
            if min(pc, line, occ, address) < 0 or address % 64:
                raise SystemExit("unaligned/negative replay row")
            count += 1
    if count <= 0 and not allow_empty:
        raise SystemExit("empty replay list {}".format(path))
    return count, digest

root = Path(metadata_path).parent
for name, count_key, hash_key, allow_empty in (
    ("offline_stride.replay.csv", "offline_normal_entries", "normal_list_sha256", False),
    ("offline_nn.replay.csv", "offline_nn_entries", "nn_list_sha256", True),
):
    path = root / name
    if not path.is_file():
        bad[name] = ("missing", "validated replay list")
        continue
    count, digest = inspect_replay(path, allow_empty)
    if metadata.get(count_key) != count:
        bad[count_key] = (metadata.get(count_key), count)
    if metadata.get(hash_key) != digest:
        bad[hash_key] = (metadata.get(hash_key), digest)
if bad:
    raise SystemExit("invalid 623 Stride v20 metadata: {}".format(bad))
PY
}

run_method() {
  local method="$1"
  local log="$LOG_DIR/$TRACE.$method.log"
  local raw="$EVENT_DIR/$TRACE.$method.events.csv"
  local gz="$raw.gz"
  if [[ "$FORCE" != 1 && -s "$log" && -s "$gz" ]] \
      && grep -q '^Core_0_IPC ' "$log" && gzip -t "$gz"; then
    echo "[skip] $method"
    return
  fi
  rm -f "$raw" "$gz"
  case "$method" in
    no_pref)
      DEMAND_EVENT_LOG="$raw" "$BIN" --l2c_prefetcher_types=none \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      ;;
    live_stride_reference)
      DEMAND_EVENT_LOG="$raw" "$BIN" --l2c_prefetcher_types=stride \
        --stride_num_trackers=64 --stride_pref_degree=2 \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      assert_live_policy "$log"
      ;;
    offline_stride)
      local list="$(colab_dir "$BASE_TAG")/offline_stride.replay.csv"
      [[ -s "$list" ]] || { echo "[error] missing $list" >&2; exit 2; }
      DEMAND_EVENT_LOG="$raw" PFETCH_LIST_PATH="$list" "$BIN" \
        --l2c_prefetcher_types=list_replayer \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      ;;
    offline_independent_rank_delta_stride_lstm_*)
      local tag="${method#offline_}"
      local list="$(colab_dir "$tag")/offline_nn.replay.csv"
      [[ -s "$list" ]] || { echo "[error] missing $list" >&2; exit 2; }
      DEMAND_EVENT_LOG="$raw" PFETCH_LIST_PATH="$list" "$BIN" \
        --l2c_prefetcher_types=list_replayer \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      ;;
    *) echo "[error] unknown method $method" >&2; exit 2 ;;
  esac
  grep -q '^Core_0_IPC ' "$log" || { echo "[error] missing final IPC for $method" >&2; exit 3; }
  [[ -s "$raw" ]] || { echo "[error] missing event output for $method" >&2; exit 3; }
  gzip -f "$raw"
  gzip -t "$gz"
}

require_colab_outputs() {
  local tag
  validate_preserved_inputs
  python3 "$INSTALL_COLAB_OUTPUT" \
    --archive "$RUN_DIR/$RUN_ID.colab_output.tar.gz" \
    --output-dir "$COLAB_ROOT" --model-tags "$MODEL_TAGS_CSV"
  for tag in "${MODEL_TAGS[@]}"; do
    for name in run_metadata.json offline_stride.replay.csv \
      offline_nn.replay.csv model.pt training_history.csv; do
      [[ -s "$(colab_dir "$tag")/$name" ]] || {
        echo "[error] missing Colab output $(colab_dir "$tag")/$name" >&2
        exit 2
      }
    done
    assert_model_metadata_v20 "$(colab_dir "$tag")/run_metadata.json"
  done
}

analyze() {
  validate_preserved_inputs
  python3 "$ANALYZE" --run-dir "$RUN_DIR" --model-tags "$MODEL_TAGS_CSV"
}

replay() {
  [[ -s "$TRACE_FILE" ]] || { echo "[error] missing trace $TRACE_FILE" >&2; exit 2; }
  if [[ "$BUILD" == 1 || ! -x "$BIN" ]]; then build; fi
  require_colab_outputs
  run_method no_pref
  run_method live_stride_reference
  run_method offline_stride
  local tag
  for tag in "${MODEL_TAGS[@]}"; do run_method "offline_$tag"; done
  analyze
}

case "$STAGE" in
  collect) collect ;;
  replay) replay ;;
  analyze) analyze ;;
  build) build ;;
  *) echo "[error] STAGE must be build, collect, replay, or analyze" >&2; exit 2 ;;
esac
