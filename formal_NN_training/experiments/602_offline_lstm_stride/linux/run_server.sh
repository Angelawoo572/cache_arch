#!/usr/bin/env bash
# Linux stages for the isolated 602 offline LSTM-versus-stride experiment.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/602_offline_lstm_stride"
TRACE="602.gcc_s-734B"
RUN_ID="${RUN_ID:-602_offline_lstm_stride_seed7}"
STAGE="${STAGE:-collect}"
FORCE="${FORCE:-0}"
JOBS="${JOBS:-8}"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
TRACE_FILE="${TRACE_FILE:-$ROOT/traces/$TRACE.champsimtrace.xz}"
RUN_DIR="${RUN_DIR:-$EXP/runs/$RUN_ID}"
LOG_DIR="$RUN_DIR/logs"
EVENT_DIR="$RUN_DIR/events"
STREAM_DIR="$RUN_DIR/colab_input"
COLAB_OUT="$RUN_DIR/colab_output"
BIN="${BIN:-$CHAMP_DIR/bin/champsim.602_offline_replay}"
EXPECTED_LIBBF_HEAD="4c9efc1a4db7ed1ccf54cf0bd3a3641ce579206c"

PATCH_LOGGER="$EXP/linux/patch_demand_logger.sh"
BUILD_REPLAYER="$EXP/linux/build_keyed_replayer.sh"
NORMALIZE="$EXP/python/normalize_events.py"
ANALYZE="$EXP/python/analyze_replay.py"
STRIDE_CONFIG="$EXP/config/stride_64x_degree2.ini"

mkdir -p "$LOG_DIR" "$EVENT_DIR" "$STREAM_DIR" "$COLAB_OUT"
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
  RESET_PATCH="${RESET_PATCH:-0}" CHAMP_DIR="$CHAMP_DIR" bash "$PATCH_LOGGER"
  ensure_libbf
  CHAMP_DIR="$CHAMP_DIR" OUT="$BIN" bash "$BUILD_REPLAYER"
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

run_method() {
  local method="$1"
  local log="$LOG_DIR/$TRACE.$method.log"
  if [[ "$FORCE" != 1 && -s "$log" ]] && grep -q '^Core_0_IPC ' "$log"; then
    echo "[skip] $method"
    return
  fi
  case "$method" in
    no_pref)
      "$BIN" --l2c_prefetcher_types=none \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      ;;
    live_stride_reference)
      "$BIN" --config="$STRIDE_CONFIG" \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      ;;
    offline_stride|offline_lstm)
      local list="$COLAB_OUT/$method.replay.csv"
      [[ -s "$list" ]] || { echo "[error] missing Colab list $list" >&2; exit 2; }
      PFETCH_LIST_PATH="$list" "$BIN" --l2c_prefetcher_types=list_replayer \
        --warmup_instructions=25000000 --simulation_instructions=25000000 \
        -traces "$TRACE_FILE" > "$log" 2>&1
      ;;
  esac
  grep -q '^Core_0_IPC ' "$log" || { echo "[error] missing final IPC for $method" >&2; exit 3; }
}

replay() {
  [[ -x "$BIN" ]] || build
  [[ -s "$COLAB_OUT/run_metadata.json" ]] || {
    echo "[error] copy the complete Colab output into $COLAB_OUT" >&2
    exit 2
  }
  run_method no_pref
  run_method live_stride_reference
  run_method offline_stride
  run_method offline_lstm
  python3 "$ANALYZE" --run-dir "$RUN_DIR"
}

case "$STAGE" in
  collect) collect ;;
  replay) replay ;;
  analyze) python3 "$ANALYZE" --run-dir "$RUN_DIR" ;;
  build) build ;;
  *) echo "[error] STAGE must be build, collect, replay, or analyze" >&2; exit 2 ;;
esac
