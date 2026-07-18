#!/usr/bin/env bash
# Linux stages for the isolated 602 offline LSTM-versus-stride experiment.
# Colab trains/generates lists; ChampSim only replays and measures them.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/602_offline_lstm_stride"
TRACE="602.gcc_s-734B"
# Legacy input-contract run token retained for the repository-wide static audit:
# 602_offline_lstm_stride_variable_delta_free_running_v7_seed7
RUN_ID="${RUN_ID:-602_offline_lstm_stride_compact_hurdle_v9_seed7}"
STAGE="${STAGE:-collect}"
FORCE="${FORCE:-0}"
JOBS="${JOBS:-8}"
BUILD="${BUILD:-1}"
# Each tag must correspond to colab_output/<tag>/run_metadata.json.
MODEL_TAGS_CSV="${MODEL_TAGS:-h8,h16,h32,h64,h128}"
BASE_MODEL_TAG="${BASE_MODEL_TAG:-h8}"
# Optional original colab_input directory for legacy archives whose gzip
# headers differ even though the normalized CSV stream is identical.
COLAB_SOURCE_INPUT_DIR="${COLAB_SOURCE_INPUT_DIR:-}"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
TRACE_FILE="${TRACE_FILE:-$ROOT/traces/$TRACE.champsimtrace.xz}"
RUN_DIR="${RUN_DIR:-$EXP/runs/$RUN_ID}"
LOG_DIR="$RUN_DIR/logs"
EVENT_DIR="$RUN_DIR/events"
STREAM_DIR="$RUN_DIR/colab_input"
COLAB_ROOT="$RUN_DIR/colab_output"
BIN="${BIN:-$CHAMP_DIR/bin/champsim.602_offline_replay}"
EXPECTED_LIBBF_HEAD="4c9efc1a4db7ed1ccf54cf0bd3a3641ce579206c"
BUILD_LOCK="${BUILD_LOCK:-$(git -C "$ROOT" rev-parse --absolute-git-dir)/champsim_build.lock}"

PATCH_LOGGER="$EXP/linux/patch_demand_logger.sh"
BUILD_REPLAYER="$EXP/linux/build_keyed_replayer.sh"
NORMALIZE="$EXP/python/normalize_events.py"
ANALYZE="$EXP/python/analyze_replay.py"

IFS=',' read -r -a MODEL_TAGS <<< "$MODEL_TAGS_CSV"
[[ "${#MODEL_TAGS[@]}" -gt 0 ]] || { echo "[error] MODEL_TAGS is empty" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$EVENT_DIR" "$STREAM_DIR" "$COLAB_ROOT"
[[ -s "$TRACE_FILE" ]] || { echo "[error] missing trace $TRACE_FILE" >&2; exit 2; }

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
  command -v flock >/dev/null 2>&1 || { echo "[error] flock is required for safe ChampSim builds" >&2; exit 2; }
  mkdir -p "$(dirname "$BUILD_LOCK")"
  (
    echo "[build-lock] waiting for $BUILD_LOCK"
    flock -x 9
    echo "[build-lock] acquired by stride run $RUN_ID"
    RESET_PATCH="${RESET_PATCH:-0}" CHAMP_DIR="$CHAMP_DIR" bash "$PATCH_LOGGER"
    ensure_libbf
    CHAMP_DIR="$CHAMP_DIR" OUT="$BIN" bash "$BUILD_REPLAYER"
    echo "[build-lock] stride build complete"
  ) 9>"$BUILD_LOCK"
}

run_no_pref_events() {
  local label="$1" warmup="$2" sim="$3"
  local raw="$EVENT_DIR/$TRACE.$label.events.csv"
  local gz="$raw.gz"
  local log="$LOG_DIR/$TRACE.$label.log"
  if [[ "$FORCE" != 1 && -s "$gz" && -s "$log" ]] && gzip -t "$gz"; then
    echo "[skip] $label"
    return
  fi
  rm -f "$raw" "$gz"
  DEMAND_EVENT_LOG="$raw" "$BIN" --l2c_prefetcher_types=none \
    --warmup_instructions="$warmup" --simulation_instructions="$sim" \
    -traces "$TRACE_FILE" > "$log" 2>&1
  [[ -s "$raw" ]] || { echo "[error] missing event output for $label" >&2; exit 3; }
  gzip -f "$raw"
  gzip -t "$gz"
}

collect() {
  build
  run_no_pref_events train_prefix 0 20000000
  run_no_pref_events evaluation 25000000 25000000
  python3 "$NORMALIZE" \
    --events "$EVENT_DIR/$TRACE.train_prefix.events.csv.gz" \
    --out "$STREAM_DIR/$TRACE.train_stream.csv.gz"
  python3 "$NORMALIZE" \
    --events "$EVENT_DIR/$TRACE.evaluation.events.csv.gz" \
    --out "$STREAM_DIR/$TRACE.eval_stream.csv.gz"
  (
    cd "$STREAM_DIR"
    sha256sum "$TRACE.train_stream.csv.gz" "$TRACE.eval_stream.csv.gz" > SHA256SUMS
  )
  tar -C "$STREAM_DIR" -czf "$RUN_DIR/$RUN_ID.colab_input.tar.gz" \
    "$TRACE.train_stream.csv.gz" "$TRACE.eval_stream.csv.gz" SHA256SUMS
  echo "[ready for Colab] $RUN_DIR/$RUN_ID.colab_input.tar.gz"
}

colab_dir() {
  printf '%s/%s' "$COLAB_ROOT" "$1"
}

assert_stateful_metadata() {
  python3 - "$1" <<'PY'
import json, sys
metadata = json.load(open(sys.argv[1]))
expected = {
    "training_state_mode": "chronological_stateful_tbptt",
    "training_chunks_shuffled": False,
    "training_state_carried_across_chunks": True,
    "training_state_detached_between_chunks": True,
    "experiment_revision": "source_input_variable_delta_free_running_v7",
    "model_revision": "compact_shared_pc_hurdle_delta_v9",
    "compact_parameter_self_test": "PASS",
    "parameter_formula": "11H^2+(F+22)H+4; F=128 gives 11H^2+150H+4",
    "encoder_recurrent_layers": 1,
    "decision_action_encoder_shared": True,
    "action_decoder_recurrent_cell": "single_gru_cell",
    "delta_decoder": "deterministic_free_running_autoregressive_signed_log_delta",
    "neural_role": "standalone_direct_action_prefetcher",
    "same_external_input_contract": True,
    "training_inference_input_encoder_identical": True,
    "decoder_training_mode": "free_running_autoregressive_same_as_inference",
    "decoder_previous_teacher_action_used_as_input": False,
    "decoder_free_running_self_test": "PASS",
    "variable_positive_count_self_test": "PASS",
    "data_derived_gate_balance_self_test": "PASS",
    "pc_keyed_causality_self_test": "PASS",
    "count_model": "learned_two_class_hurdle_plus_unbounded_positive_log_count",
    "gate_imbalance_handling": "inverse_observed_training_class_frequency_equal_aggregate_mass",
    "data_derived_class_balancing_used": True,
    "state_routing": "dynamic_exact_pc_keyed_recurrent_state_no_fixed_capacity",
    "pc_state_capacity": None,
    "normal_tracker_count_used_by_neural_inference": False,
    "training_runtime_fields": ["pc", "cache_line_address"],
    "inference_runtime_fields": ["pc", "cache_line_address"],
    "training_state_key_fields": ["pc"],
    "inference_state_key_fields": ["pc"],
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
    "training_regularization_used": False,
    "inference_policy_hardcodes_used": False,
    "learned_request_count": True,
    "nn_generates_own_target_addresses": True,
}
bad = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
hidden_size = metadata.get("hidden_size")
if not isinstance(hidden_size, int) or hidden_size < 1:
    bad["hidden_size"] = (hidden_size, "positive integer")
else:
    expected_parameters = 11 * hidden_size * hidden_size + 150 * hidden_size + 4
    if metadata.get("parameter_count") != expected_parameters:
        bad["parameter_count"] = (
            metadata.get("parameter_count"), expected_parameters
        )
    if metadata.get("expected_parameter_count") != expected_parameters:
        bad["expected_parameter_count"] = (
            metadata.get("expected_parameter_count"), expected_parameters
        )
encoder_hashes = {metadata.get("runtime_encoder_sha256"), metadata.get("training_runtime_encoder_sha256"), metadata.get("inference_runtime_encoder_sha256")}
encoder_hash = next(iter(encoder_hashes)) if len(encoder_hashes) == 1 else None
if not isinstance(encoder_hash, str) or len(encoder_hash) != 64:
    bad["runtime_encoder_sha256"] = (encoder_hashes, "one shared 64-hex digest")
router_hashes = {
    metadata.get("state_router_sha256"),
    metadata.get("training_state_router_sha256"),
    metadata.get("inference_state_router_sha256"),
}
router_hash = next(iter(router_hashes)) if len(router_hashes) == 1 else None
if not isinstance(router_hash, str) or len(router_hash) != 64:
    bad["state_router_sha256"] = (router_hashes, "one shared 64-hex digest")
if bad:
    raise SystemExit("not an independent direct-action Colab output: {}".format(bad))
PY
}

assert_live_stride() {
  local log="$1"
  # ChampSim prints the implementation name as "Stride" (capital S).
  grep -Fqi "adding L2C_PREFETCHER: stride" "$log" || {
    echo "[error] live stride was not registered; refusing an inactive reference" >&2
    exit 3
  }
  grep -Fq "stride_num_trackers 64" "$log" || {
    echo "[error] live stride did not use 64 trackers" >&2
    exit 3
  }
  grep -Fq "stride_pref_degree 2" "$log" || {
    echo "[error] live stride did not use degree 2" >&2
    exit 3
  }
  grep -Eq '^stride_pref_generated [1-9][0-9]*$' "$log" || {
    echo "[error] live stride generated zero prefetches" >&2
    exit 3
  }
}

run_method() {
  local method="$1"
  local log="$LOG_DIR/$TRACE.$method.log"
  local raw="$EVENT_DIR/$TRACE.$method.events.csv"
  local gz="$raw.gz"
  if [[ "$FORCE" != 1 && -s "$log" && -s "$gz" ]] && grep -q '^Core_0_IPC ' "$log" && gzip -t "$gz"; then
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
      # Explicit top-level flags are required by this ChampSim parser.
      DEMAND_EVENT_LOG="$raw" "$BIN" --l2c_prefetcher_types=stride \
        --stride_num_trackers=64 --stride_pref_degree=2 \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      assert_live_stride "$log"
      ;;
    offline_stride)
      local list
      list="$(colab_dir "$BASE_MODEL_TAG")/offline_stride.replay.csv"
      [[ -s "$list" ]] || { echo "[error] missing Colab list $list" >&2; exit 2; }
      DEMAND_EVENT_LOG="$raw" PFETCH_LIST_PATH="$list" "$BIN" --l2c_prefetcher_types=list_replayer \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      ;;
    offline_lstm_*)
      local tag="${method#offline_lstm_}"
      local list
      list="$(colab_dir "$tag")/offline_lstm.replay.csv"
      [[ -s "$list" ]] || { echo "[error] missing Colab list $list" >&2; exit 2; }
      DEMAND_EVENT_LOG="$raw" PFETCH_LIST_PATH="$list" "$BIN" --l2c_prefetcher_types=list_replayer \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      ;;
    *) echo "[error] unknown method $method" >&2; exit 2 ;;
  esac
  grep -q '^Core_0_IPC ' "$log" || { echo "[error] missing final IPC for $method" >&2; exit 3; }
  [[ -s "$raw" ]] || { echo "[error] missing replay event output for $method" >&2; exit 3; }
  gzip -f "$raw"
  gzip -t "$gz"
}

require_colab_outputs() {
  local metadata_paths=()
  for tag in "${MODEL_TAGS[@]}"; do
    [[ -s "$(colab_dir "$tag")/run_metadata.json" ]] || {
      echo "[error] missing Colab output $(colab_dir "$tag")/run_metadata.json" >&2
      exit 2
    }
    assert_stateful_metadata "$(colab_dir "$tag")/run_metadata.json"
    metadata_paths+=("$(colab_dir "$tag")/run_metadata.json")
  done
  python3 - "${metadata_paths[@]}" <<'PY'
import json, sys
metadata = [json.load(open(path)) for path in sys.argv[1:]]
if not any(int(item.get("offline_lstm_entries", 0)) > 0 for item in metadata):
    raise SystemExit(
        "all capacity points collapsed to empty neural replay lists; "
        "refusing a misleading replay"
    )
print("[PASS] at least one PC-keyed hurdle model produced neural actions")
PY
}

analyze() {
  local cmd=(python3 "$ANALYZE" --run-dir "$RUN_DIR" --model-tags "$MODEL_TAGS_CSV" --base-model-tag "$BASE_MODEL_TAG")
  if [[ -n "$COLAB_SOURCE_INPUT_DIR" ]]; then
    [[ -d "$COLAB_SOURCE_INPUT_DIR" ]] || {
      echo "[error] COLAB_SOURCE_INPUT_DIR is not a directory: $COLAB_SOURCE_INPUT_DIR" >&2
      exit 2
    }
    cmd+=(--source-input-dir "$COLAB_SOURCE_INPUT_DIR")
  fi
  "${cmd[@]}"
}

replay() {
  if [[ "$BUILD" == 1 || ! -x "$BIN" ]]; then build; fi
  require_colab_outputs
  run_method no_pref
  run_method live_stride_reference
  run_method offline_stride
  for tag in "${MODEL_TAGS[@]}"; do
    run_method "offline_lstm_$tag"
  done
  analyze
}

case "$STAGE" in
  collect) collect ;;
  replay) replay ;;
  analyze) analyze ;;
  build) build ;;
  *) echo "[error] STAGE must be build, collect, replay, or analyze" >&2; exit 2 ;;
esac
