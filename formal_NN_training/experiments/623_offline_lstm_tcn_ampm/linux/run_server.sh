#!/usr/bin/env bash
# Linux stages for the 623 matched AMPM LSTM-versus-causal-TCN experiment.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/623_offline_lstm_tcn_ampm"
TRACE="623.xalancbmk_s-700B"
RUN_ID="${RUN_ID:-623_offline_lstm_tcn_ampm_seed7}"
STAGE="${STAGE:-collect}"
FORCE="${FORCE:-0}"
JOBS="${JOBS:-8}"
BUILD="${BUILD:-1}"
MODEL_TAGS_CSV="${MODEL_TAGS:-lstm_h8,lstm_h16,lstm_h32,tcn_c10,tcn_c16,tcn_c24}"
BASE_MODEL_TAG="${BASE_MODEL_TAG:-lstm_h8}"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
TRACE_FILE="${TRACE_FILE:-$ROOT/traces/$TRACE.champsimtrace.xz}"
RUN_DIR="${RUN_DIR:-$EXP/runs/$RUN_ID}"
LOG_DIR="$RUN_DIR/logs"
EVENT_DIR="$RUN_DIR/events"
STREAM_DIR="$RUN_DIR/colab_input"
COLAB_ROOT="$RUN_DIR/colab_output"
BIN="${BIN:-$CHAMP_DIR/bin/champsim.623_ampm_temporal_replay}"
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
  [[ "$observed" == "$EXPECTED_LIBBF_HEAD" ]] || { echo "[error] libbf HEAD $observed != pinned $EXPECTED_LIBBF_HEAD" >&2; exit 2; }
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
    echo "[build-lock] acquired by 623 temporal run $RUN_ID"
    RESET_PATCH="${RESET_PATCH:-0}" CHAMP_DIR="$CHAMP_DIR" bash "$PATCH_LOGGER"
    ensure_libbf
    CHAMP_DIR="$CHAMP_DIR" OUT="$BIN" bash "$BUILD_REPLAYER"
    echo "[build-lock] 623 temporal build complete"
  ) 9>"$BUILD_LOCK"
}

run_no_pref_events() {
  local label="$1" warmup="$2" simulation="$3"
  local raw="$EVENT_DIR/$TRACE.$label.events.csv"
  local gz="$raw.gz"
  local log="$LOG_DIR/$TRACE.$label.log"
  if [[ "$FORCE" != 1 && -s "$gz" && -s "$log" ]] && gzip -t "$gz"; then
    echo "[skip] $label"
    return
  fi
  rm -f "$raw" "$gz"
  DEMAND_EVENT_LOG="$raw" "$BIN" --l2c_prefetcher_types=none \
    --warmup_instructions="$warmup" --simulation_instructions="$simulation" \
    -traces "$TRACE_FILE" > "$log" 2>&1
  [[ -s "$raw" ]] || { echo "[error] missing event output for $label" >&2; exit 3; }
  gzip -f "$raw"
  gzip -t "$gz"
}

collect() {
  build
  run_no_pref_events train_prefix 0 20000000
  run_no_pref_events guard_prefix 20000000 5000000
  run_no_pref_events evaluation 25000000 25000000
  python3 "$NORMALIZE" --events "$EVENT_DIR/$TRACE.train_prefix.events.csv.gz" --out "$STREAM_DIR/$TRACE.train_stream.csv.gz"
  python3 "$NORMALIZE" --events "$EVENT_DIR/$TRACE.guard_prefix.events.csv.gz" --out "$STREAM_DIR/$TRACE.guard_stream.csv.gz"
  python3 "$NORMALIZE" --events "$EVENT_DIR/$TRACE.evaluation.events.csv.gz" --out "$STREAM_DIR/$TRACE.eval_stream.csv.gz"
  (
    cd "$STREAM_DIR"
    sha256sum "$TRACE.train_stream.csv.gz" "$TRACE.guard_stream.csv.gz" "$TRACE.eval_stream.csv.gz" > SHA256SUMS
  )
  tar -C "$STREAM_DIR" -czf "$RUN_DIR/$RUN_ID.colab_input.tar.gz" \
    "$TRACE.train_stream.csv.gz" "$TRACE.guard_stream.csv.gz" "$TRACE.eval_stream.csv.gz" SHA256SUMS
  echo "[ready for Colab] $RUN_DIR/$RUN_ID.colab_input.tar.gz"
}

colab_dir() { printf '%s/%s' "$COLAB_ROOT" "$1"; }

assert_model_metadata() {
  python3 - "$1" <<'PY'
import json, sys

metadata = json.load(open(sys.argv[1]))
common = {
    "trace": "623.xalancbmk_s-700B",
    "matched_normal_prefetcher": "ampm",
    "model_does_not_use_pc": True,
    "normal_candidate_bank_is_fixed": True,
    "nn_can_only_suppress_ampm_candidates": True,
    "training_chunks_shuffled": False,
    "causal_no_future_self_test": "PASS",
    "experiment_revision": "architecture_ablation_v1",
}
bad = {key: (metadata.get(key), expected) for key, expected in common.items() if metadata.get(key) != expected}
family = metadata.get("model_family")
if family == "lstm":
    expected = {
        "training_state_mode": "chronological_stateful_tbptt",
        "training_state_carried_across_chunks": True,
        "training_state_detached_between_chunks": True,
    }
elif family == "tcn":
    expected = {
        "training_state_mode": "finite_causal_left_context",
        "training_state_carried_across_chunks": False,
        "training_state_detached_between_chunks": False,
        "tcn_receptive_field_events": 127,
        "training_left_context_overlap": 126,
    }
else:
    expected = {"model_family": "lstm_or_tcn"}
for key, value in expected.items():
    if metadata.get(key) != value:
        bad[key] = (metadata.get(key), value)
if bad:
    raise SystemExit("invalid architecture-ablation metadata: {}".format(bad))
PY
}

assert_live_ampm() {
  local log="$1"
  grep -Fq "adding L2C_PREFETCHER: AMPM" "$log" || { echo "[error] live AMPM was not registered" >&2; exit 3; }
  grep -Fq "ampm_pb_size 64" "$log" || { echo "[error] live AMPM did not use 64 page-buffer entries" >&2; exit 3; }
  grep -Fq "ampm_pred_degree 4" "$log" || { echo "[error] live AMPM did not use prediction degree 4" >&2; exit 3; }
  grep -Fq "ampm_pref_degree 4" "$log" || { echo "[error] live AMPM did not use prefetch degree 4" >&2; exit 3; }
  grep -Fq "ampm_enable_pref_buffer 0" "$log" || { echo "[error] live AMPM prefetch buffer was not disabled" >&2; exit 3; }
  grep -Fq "ampm_max_delta 16" "$log" || { echo "[error] live AMPM did not use max delta 16" >&2; exit 3; }
  grep -Eq '^ampm\.pred\.total [1-9][0-9]*$' "$log" || { echo "[error] live AMPM generated zero candidates" >&2; exit 3; }
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
    live_spp_context)
      DEMAND_EVENT_LOG="$raw" "$BIN" --l2c_prefetcher_types=spp_dev2 \
        --spp_dev2_fill_threshold=90 --spp_dev2_pf_threshold=40 \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      grep -Eiq 'adding L2C_PREFETCHER:.*spp' "$log" || { echo "[error] live SPP context run was not registered" >&2; exit 3; }
      ;;
    live_ampm_reference)
      DEMAND_EVENT_LOG="$raw" "$BIN" --l2c_prefetcher_types=ampm \
        --ampm_pb_size=64 --ampm_pred_degree=4 --ampm_pref_degree=4 \
        --ampm_enable_pref_buffer=false --ampm_pref_buffer_size=256 --ampm_max_delta=16 \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      assert_live_ampm "$log"
      ;;
    offline_ampm)
      local list="$(colab_dir "$BASE_MODEL_TAG")/offline_ampm.replay.csv"
      [[ -s "$list" ]] || { echo "[error] missing Colab list $list" >&2; exit 2; }
      DEMAND_EVENT_LOG="$raw" PFETCH_LIST_PATH="$list" "$BIN" \
        --l2c_prefetcher_types=list_replayer \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      ;;
    offline_lstm_*|offline_tcn_*)
      local tag="${method#offline_}"
      local list="$(colab_dir "$tag")/offline_nn.replay.csv"
      [[ -s "$list" ]] || { echo "[error] missing Colab list $list" >&2; exit 2; }
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
  for tag in "${MODEL_TAGS[@]}"; do
    for name in run_metadata.json offline_ampm.replay.csv offline_nn.replay.csv model.pt; do
      [[ -s "$(colab_dir "$tag")/$name" ]] || { echo "[error] missing Colab output $(colab_dir "$tag")/$name" >&2; exit 2; }
    done
    assert_model_metadata "$(colab_dir "$tag")/run_metadata.json"
  done
}

analyze() {
  python3 "$ANALYZE" \
    --run-dir "$RUN_DIR" \
    --model-tags "$MODEL_TAGS_CSV" \
    --base-model-tag "$BASE_MODEL_TAG"
}

replay() {
  if [[ "$BUILD" == 1 || ! -x "$BIN" ]]; then build; fi
  require_colab_outputs
  run_method no_pref
  run_method live_spp_context
  run_method live_ampm_reference
  run_method offline_ampm
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
