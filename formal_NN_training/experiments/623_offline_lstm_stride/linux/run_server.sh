#!/usr/bin/env bash
# Independent 623 track: normal Stride versus direct LSTM only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/623_offline_lstm_stride"
TRACE="623.xalancbmk_s-700B"
POLICY="stride"
RUN_ID="${RUN_ID:-623_offline_lstm_stride_compact_hurdle_v16_seed7}"
STAGE="${STAGE:-replay}"
FORCE="${FORCE:-0}"
JOBS="${JOBS:-8}"
BUILD="${BUILD:-1}"
MODEL_TAGS_CSV="${MODEL_TAGS:-independent_delta_stride_lstm_h8,independent_delta_stride_lstm_h16,independent_delta_stride_lstm_h32,independent_delta_stride_lstm_h64,independent_delta_stride_lstm_h128}"
BASE_TAG="${BASE_TAG:-independent_delta_stride_lstm_h8}"
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
  "$ANALYZE" "$INSTALL_COLAB_OUTPUT"; do
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

colab_dir() { printf '%s/%s' "$COLAB_ROOT" "$1"; }

assert_model_metadata() {
  python3 - "$1" "$POLICY" <<'PY'
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

metadata = json.load(open(sys.argv[1]))
policy = sys.argv[2]
tag = metadata.get("model_tag", "")
family = metadata.get("model_family")
common = {
    "trace": "623.xalancbmk_s-700B",
    "matched_normal_prefetcher": policy,
    "source_decision_effective_external_input": ["pc", "addr"],
    "same_external_input_contract": True,
    "training_inference_input_encoder_identical": True,
    "decoder_training_mode": "free_running_autoregressive_same_as_inference",
    "decoder_previous_teacher_action_used_as_input": False,
    "decoder_free_running_self_test": "PASS",
    "training_runtime_fields": ["pc", "addr"],
    "inference_runtime_fields": ["pc", "addr"],
    "normal_policy_outputs_used_as_model_inputs": False,
    "normal_policy_candidates_used_as_model_inputs": False,
    "normal_policy_private_state_used_as_model_inputs": False,
    "normal_policy_outputs_used_as_training_targets": True,
    "normal_policy_request_rate_used_as_budget": False,
    "normal_policy_constants_used_by_neural_inference": False,
    "probability_threshold_used": False,
    "threshold_related_hardcodes_used": False,
    "neural_degree_cap": None,
    "fixed_page_offset_classes": None,
    "same_page_rule_used_by_neural_inference": False,
    "future_label_window_used": False,
    "handcrafted_semantic_features_used": False,
    "manual_loss_weights_used": False,
    "data_derived_gate_class_weights_used": True,
    "gate_class_weighting_used": True,
    "gate_training_objective": "data_derived_frequency_balanced_two_class_cross_entropy",
    "gate_decoding_rule": "deterministic_two_class_argmax",
    "gate_class_weights_source": "train_zero_positive_frequencies_equal_aggregate_loss_mass",
    "request_count_training_objective": "balanced_two_class_hurdle_plus_positive_log_count_smooth_l1",
    "request_count_decoding_rule": "deterministic_gate_argmax_plus_rounded_exp_positive_log_count",
    "request_count_residual_scope": "none_event_local",
    "training_regularization_used": False,
    "inference_policy_hardcodes_used": False,
    "learned_request_count": True,
    "nn_generates_own_target_addresses": True,
    "training_chunks_shuffled": False,
    "causal_no_future_self_test": "PASS",
    "event_keyed_crn_self_test": "NOT_APPLICABLE",
    "event_keyed_hurdle_count_self_test": "NOT_APPLICABLE",
    "canonicalized_mixture_sampling_self_test": "NOT_APPLICABLE",
    "deterministic_count_and_balance_self_test": "PASS",
    "decoder_probability_mass_carries_train_guard_history": False,
    "cross_event_probability_credit_used": False,
    "sampled_outputs_used_as_decoder_feedback": False,
    "deterministic_decoding_reproducible": True,
    "stochastic_decoding_reproducible": False,
    "delta_mixture_decoding_rule": None,
    "delta_training_objective": "scalar_signed_log_delta_smooth_l1",
    "delta_decoding_rule": "deterministic_rounded_scalar_signed_log_delta",
    "delta_decoder_feedback_rule": "emitted_scalar_coordinate_same_in_training_and_inference",
    "delta_mixture_components": 0,
    "cnn_architecture_self_test": "NOT_APPLICABLE",
    "event_logger_schema": "623_causal_trigger_v5",
    "candidate_attachment_mode": "explicit_trigger_event_id",
    "experiment_revision": "stride_source_input_variable_delta_free_running_v9",
    "model_revision": "compact_pc_keyed_balanced_deterministic_scalar_v16",
    "neural_role": "standalone_direct_action_prefetcher",
    "track_model_family": "lstm",
    "runtime_feature_count": 122,
    "runtime_encoding": "lossless uint64 PC plus lossless 58-bit cache-line number",
    "deterministic_decoding": True,
    "stochastic_decoding": False,
    "common_random_numbers_across_capacities": False,
    "strict_common_random_numbers_across_capacities": False,
    "cross_event_rng_state_used": False,
    "decoder_sampling_roles": [],
    "decoder_train_sampling_performed": False,
    "decoder_guard_sampling_performed": False,
}
bad = {key: (metadata.get(key), expected) for key, expected in common.items()
       if metadata.get(key) != expected}
statistics = metadata.get("request_count_training_label_statistics") or {}
weights = metadata.get("gate_class_weights")
decision_callbacks = statistics.get("decision_callbacks")
positive_callbacks = statistics.get("positive_callbacks")
zero_callbacks = statistics.get("zero_callbacks")
if (
    not isinstance(decision_callbacks, int)
    or not isinstance(positive_callbacks, int)
    or not isinstance(zero_callbacks, int)
    or decision_callbacks <= 0
    or positive_callbacks <= 0
    or zero_callbacks <= 0
    or positive_callbacks + zero_callbacks != decision_callbacks
    or not isinstance(weights, list)
    or len(weights) != 2
):
    bad["gate_class_weights"] = (
        {"statistics": statistics, "weights": weights},
        "two inverse-frequency weights from a nonempty two-class train split",
    )
else:
    expected_weights = [
        float(decision_callbacks) / (2.0 * zero_callbacks),
        float(decision_callbacks) / (2.0 * positive_callbacks),
    ]
    if any(
        not isinstance(actual, (int, float))
        or isinstance(actual, bool)
        or not math.isfinite(float(actual))
        or not math.isclose(
            float(actual), expected, rel_tol=1e-6, abs_tol=1e-7
        )
        for actual, expected in zip(weights, expected_weights)
    ):
        bad["gate_class_weights"] = (weights, expected_weights)
for key in (
    "training_state_router_sha256", "inference_state_router_sha256",
):
    value = metadata.get(key)
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        bad[key] = (value, "64 lowercase hex characters")
if family != "lstm":
    bad["model_family"] = (family, "lstm")
if not tag.startswith("independent_delta_" + policy + "_lstm_"):
    bad["model_tag"] = (tag, "independent_delta_" + policy + "_lstm_<size>")
expected_points = {
    ("lstm", 8): "p0",
    ("lstm", 16): "p1",
    ("lstm", 32): "p2",
    ("lstm", 64): "p3",
    ("lstm", 128): "p4",
}
expected_parameters = {8: 1860, 16: 5124, 32: 15876, 64: 54276, 128: 198660}
point = expected_points.get((family, metadata.get("model_size")))
if point is None:
    bad["model_point"] = ((family, metadata.get("model_size")), "pinned point")
else:
    if metadata.get("architecture_pair_id") != point:
        bad["architecture_pair_id"] = (metadata.get("architecture_pair_id"), point)
expected_parameter_count = expected_parameters.get(metadata.get("model_size"))
if metadata.get("parameter_count") != expected_parameter_count:
    bad["parameter_count"] = (metadata.get("parameter_count"), expected_parameter_count)
encoder_hashes = {
    metadata.get("runtime_encoder_sha256"),
    metadata.get("training_runtime_encoder_sha256"),
    metadata.get("inference_runtime_encoder_sha256"),
}
encoder_hash = next(iter(encoder_hashes)) if len(encoder_hashes) == 1 else None
if not isinstance(encoder_hash, str) or len(encoder_hash) != 64:
    bad["runtime_encoder_sha256"] = (encoder_hashes, "one shared 64-hex digest")
expected = {
    "training_state_mode": "chronological_stateful_tbptt",
    "training_state_carried_across_chunks": True,
    "training_state_detached_between_chunks": True,
    "inference_history_mode": "fresh_state_then_complete_train_guard_eval_chronology",
    "cnn_temporal_layers": 0,
}
for key, value in expected.items():
    if metadata.get(key) != value:
        bad[key] = (metadata.get(key), value)

def inspect_replay(path, allow_empty):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    count = 0
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ["pc", "line", "occ", "prefetch_addr"]:
            raise SystemExit("invalid stride replay header in {}".format(path))
        for line_number, fields in enumerate(reader, 2):
            if len(fields) != 4:
                raise SystemExit("invalid stride replay row {} in {}".format(line_number, path))
            try:
                pc = int(fields[0], 0)
                line = int(fields[1], 0)
                occ = int(fields[2], 10)
                address = int(fields[3], 0)
            except ValueError as exc:
                raise SystemExit("invalid stride replay integer at {}: {}".format(line_number, exc))
            if min(pc, line, occ, address) < 0 or address % 64:
                raise SystemExit("unaligned/negative stride replay row {}".format(line_number))
            count += 1
    if count <= 0 and not allow_empty:
        raise SystemExit("empty stride replay list {}".format(path))
    return count, digest

root = Path(sys.argv[1]).parent
for name, count_key, hash_key, allow_empty in (
    ("offline_stride.replay.csv", "offline_normal_entries", "normal_list_sha256", False),
    ("offline_nn.replay.csv", "offline_nn_entries", "nn_list_sha256", True),
):
    path = root / name
    if not path.is_file():
        bad[name] = ("missing", "nonempty validated replay list")
        continue
    count, digest = inspect_replay(path, allow_empty)
    if metadata.get(count_key) != count:
        bad[count_key] = (metadata.get(count_key), count)
    if metadata.get(hash_key) != digest:
        bad[hash_key] = (metadata.get(hash_key), digest)
if bad:
    raise SystemExit("invalid 623 stride metadata: {}".format(bad))
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
    offline_independent_delta_stride_lstm_*)
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
  python3 "$VALIDATE_INPUTS" \
    --input-dir "$STREAM_DIR" --manifest-out "$COLLECTION_MANIFEST"
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
    assert_model_metadata "$(colab_dir "$tag")/run_metadata.json"
  done
}

analyze() {
  python3 "$VALIDATE_INPUTS" \
    --input-dir "$STREAM_DIR" --manifest-out "$COLLECTION_MANIFEST"
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
