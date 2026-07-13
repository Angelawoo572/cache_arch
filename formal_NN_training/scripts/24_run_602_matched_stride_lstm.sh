#!/usr/bin/env bash
# One-trace, one-baseline, live matched-input 602 experiment.
#
# STAGE=collect  : collect the first 20M no-pref L2 LOAD stream and build the
#                  PC/address training stream.
# STAGE=train    : train/calibrate one 545-parameter LSTM from that earlier
#                  window and export a C++ runtime model (NumPy + PyTorch).
# STAGE=evaluate : build one live-inference binary, then run no_pref, stride,
#                  and the LSTM on the same 25M-warmup/25M-simulation window.
#                  The last 5M warmup instructions are absent from training and
#                  form a guard before measured IPC begins.
# STAGE=analyze  : verify provenance/input contracts and write the comparison.
# STAGE=all      : run all four stages in order.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TRACE="602.gcc_s-734B"
EXPECTED_CHAMPSIM_HEAD="fd26fc51a44554976022e1ee13e73e7b06e2307e"
EXPECTED_LIBBF_HEAD="4c9efc1a4db7ed1ccf54cf0bd3a3641ce579206c"
TRAIN_WARMUP=0
TRAIN_SIM=20000000
EVAL_WARMUP=25000000
EVAL_SIM=25000000
STAGE="${STAGE:-all}"
RUN_ID="${RUN_ID:-602_matched_stride_lstm_seed7}"
SEED="${SEED:-7}"
FORCE="${FORCE:-0}"
BUILD="${BUILD:-1}"
RESET_PATCH="${RESET_PATCH:-0}"
JOBS="${JOBS:-8}"
TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"

CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
TRACE_DIR="${TRACE_DIR:-$ROOT/traces}"
TRACE_FILE="$TRACE_DIR/$TRACE.champsimtrace.xz"
OUT_ROOT="${OUT_ROOT:-$ROOT/formal_NN_training/results/602_matched_stride_lstm/$RUN_ID}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$ROOT/formal_NN_training/artifacts/602_matched_stride_lstm/$RUN_ID}"
LOG_DIR="$OUT_ROOT/logs"
EVENT_DIR="$OUT_ROOT/events"
DATA_DIR="$OUT_ROOT/training_data"
CONFIG_SNAPSHOT_DIR="$OUT_ROOT/config_snapshot"

STRIDE_CONFIG="$ROOT/formal_NN_training/configs/602_matched_stride.ini"
PATCH_LOGGER="$ROOT/formal_NN_training/scripts/02_patch_pythia_demand_logger.sh"
BUILD_COLLECTION_BIN="$ROOT/formal_NN_training/scripts/06_install_keyed_listreplayer.sh"
BUILD_LIVE_BIN="$ROOT/formal_NN_training/scripts/26_install_602_matched_stride_lstm.sh"
NORMALIZE_TRAIN_STREAM="$ROOT/formal_NN_training/scripts/05_build_standalone_oracle_dataset.py"
PARSE_BASELINE="$ROOT/formal_NN_training/scripts/01_parse_prefetch_behavior_audit.py"
TRAIN_LSTM="$ROOT/formal_NN_training/LSTM/train_602_matched_stride_lstm.py"
COMPARE="$ROOT/formal_NN_training/scripts/25_compare_602_matched_stride_lstm.py"
RUNTIME_SOURCE="$ROOT/formal_NN_training/LSTM/runtime/matched_stride_lstm.cc"
RUNTIME_HEADER="$ROOT/formal_NN_training/LSTM/runtime/matched_stride_lstm.h"
COLLECTION_BIN="${COLLECTION_BIN:-$CHAMP_DIR/bin/champsim.standalone_nn_replayer}"
EVAL_BIN="${EVAL_BIN:-$CHAMP_DIR/bin/champsim.602_matched_stride_lstm}"
RUNTIME_MODEL="$ARTIFACT_DIR/matched_stride_lstm.runtime.txt"

TRAIN_EVENTS="$EVENT_DIR/$TRACE.train_no_pref.events.csv.gz"
TRAIN_STREAM="$DATA_DIR/$TRACE.train_stream.csv.gz"
TRAIN_LOG="$LOG_DIR/$TRACE.train_no_pref.log"
NO_PREF_EVENTS="$EVENT_DIR/$TRACE.no_pref.events.csv.gz"
STRIDE_EVENTS="$EVENT_DIR/$TRACE.stride.events.csv.gz"
LSTM_EVENTS="$EVENT_DIR/$TRACE.matched_lstm.events.csv.gz"
NO_PREF_LOG="$LOG_DIR/$TRACE.no_pref.log"
STRIDE_LOG="$LOG_DIR/$TRACE.stride.log"
LSTM_LOG="$LOG_DIR/$TRACE.matched_lstm.log"
IDENTITY="$OUT_ROOT/run_identity.json"
BASELINE_SUMMARY="$OUT_ROOT/baseline_summary.csv"

case "$STAGE" in
  collect|train|evaluate|analyze|all) ;;
  *) echo "[error] STAGE must be collect, train, evaluate, analyze, or all" >&2; exit 2 ;;
esac
[[ "$FORCE" == 0 || "$FORCE" == 1 ]] || { echo "[error] FORCE must be 0 or 1" >&2; exit 2; }
[[ "$BUILD" == 0 || "$BUILD" == 1 ]] || { echo "[error] BUILD must be 0 or 1" >&2; exit 2; }
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "[error] JOBS must be positive" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$EVENT_DIR" "$DATA_DIR" "$CONFIG_SNAPSHOT_DIR" "$ARTIFACT_DIR"

require_file() {
  [[ -f "$1" ]] || { echo "[error] missing $1" >&2; exit 2; }
}

require_common_files() {
  for path in "$TRACE_FILE" "$STRIDE_CONFIG" "$PATCH_LOGGER" "$BUILD_COLLECTION_BIN" \
    "$BUILD_LIVE_BIN" "$NORMALIZE_TRAIN_STREAM" "$PARSE_BASELINE" "$TRAIN_LSTM" "$COMPARE" \
    "$RUNTIME_SOURCE" "$RUNTIME_HEADER"; do
    require_file "$path"
  done
  [[ -d "$CHAMP_DIR/.git" ]] || { echo "[error] not a ChampSim checkout: $CHAMP_DIR" >&2; exit 2; }
  local observed
  observed="$(git -C "$CHAMP_DIR" rev-parse HEAD)"
  if [[ "$observed" != "$EXPECTED_CHAMPSIM_HEAD" && "${ALLOW_CHAMPSIM_DRIFT:-0}" != 1 ]]; then
    echo "[error] ChampSim HEAD $observed != pinned $EXPECTED_CHAMPSIM_HEAD" >&2
    echo "        Inspect the diff, then explicitly set ALLOW_CHAMPSIM_DRIFT=1 if intentional." >&2
    exit 3
  fi
}

ensure_libbf() {
  if [[ -e "$CHAMP_DIR/libbf" && ! -d "$CHAMP_DIR/libbf/.git" ]]; then
    echo "[error] $CHAMP_DIR/libbf exists but is not a git checkout; resolve it manually" >&2
    exit 2
  fi
  if [[ ! -d "$CHAMP_DIR/libbf/.git" ]]; then
    git clone https://github.com/mavam/libbf.git "$CHAMP_DIR/libbf"
    git -C "$CHAMP_DIR/libbf" checkout --detach "$EXPECTED_LIBBF_HEAD"
  fi
  local observed
  observed="$(git -C "$CHAMP_DIR/libbf" rev-parse HEAD)"
  if [[ "$observed" != "$EXPECTED_LIBBF_HEAD" && "${ALLOW_LIBBF_DRIFT:-0}" != 1 ]]; then
    echo "[error] libbf HEAD $observed != pinned $EXPECTED_LIBBF_HEAD" >&2
    echo "        Preserve local work, then check out the pinned commit." >&2
    exit 3
  fi
  if [[ -f "$CHAMP_DIR/libbf/bf/all.hpp" && -f "$CHAMP_DIR/libbf/build/lib/libbf.a" ]]; then
    return 0
  fi
  command -v cmake >/dev/null || { echo "[error] cmake is required to build libbf" >&2; exit 2; }
  mkdir -p "$CHAMP_DIR/libbf/build"
  ( cd "$CHAMP_DIR/libbf/build" && cmake .. && make -j"$JOBS" )
  [[ -f "$CHAMP_DIR/libbf/bf/all.hpp" && -f "$CHAMP_DIR/libbf/build/lib/libbf.a" ]] || {
    echo "[error] libbf build did not produce required files" >&2
    exit 2
  }
}

patch_logger_and_dependencies() {
  RESET_PATCH="$RESET_PATCH" CHAMP_DIR="$CHAMP_DIR" bash "$PATCH_LOGGER"
  ensure_libbf
}

completed_event_run() {
  local events="$1" log="$2"
  [[ -s "$events" && -s "$log" ]] \
    && gzip -t "$events" >/dev/null 2>&1 \
    && grep -q '^Core_0_IPC ' "$log"
}

run_events() {
  local label="$1" binary="$2" mode="$3" events="$4" log="$5" warmup="$6" sim="$7"
  local raw="${events%.gz}"
  if [[ "$FORCE" != 1 ]] && completed_event_run "$events" "$log"; then
    echo "[skip] $label"
    return 0
  fi
  rm -f "$raw" "$events"
  echo "[run] $label warmup=$warmup simulation=$sim"
  case "$mode" in
    no_pref)
      DEMAND_EVENT_LOG="$raw" "$binary" \
        --l2c_prefetcher_types=none \
        --warmup_instructions="$warmup" --simulation_instructions="$sim" \
        -traces "$TRACE_FILE" > "$log" 2>&1
      ;;
    stride)
      DEMAND_EVENT_LOG="$raw" "$binary" \
        --config="$STRIDE_CONFIG" \
        --warmup_instructions="$warmup" --simulation_instructions="$sim" \
        -traces "$TRACE_FILE" > "$log" 2>&1
      ;;
    matched_lstm)
      MATCHED_LSTM_MODEL_PATH="$RUNTIME_MODEL" DEMAND_EVENT_LOG="$raw" "$binary" \
        --l2c_prefetcher_types=matched_stride_lstm \
        --warmup_instructions="$warmup" --simulation_instructions="$sim" \
        -traces "$TRACE_FILE" > "$log" 2>&1
      ;;
    *) echo "[error] unsupported method $mode" >&2; exit 2 ;;
  esac
  [[ -s "$raw" ]] || { echo "[error] logger wrote no events for $label" >&2; exit 4; }
  grep -q '^Core_0_IPC ' "$log" || { echo "[error] final IPC missing for $label" >&2; exit 4; }
  gzip -f "$raw"
  gzip -t "$events" >/dev/null 2>&1 || { echo "[error] corrupt event gzip for $label" >&2; exit 4; }
}

collect_stage() {
  local fingerprint_file="$OUT_ROOT/collection_fingerprint.sha256"
  local current_fingerprint previous_fingerprint=""
  local collection_force="$FORCE"
  require_common_files
  if [[ "$BUILD" == 1 ]]; then
    patch_logger_and_dependencies
    CHAMP_DIR="$CHAMP_DIR" bash "$BUILD_COLLECTION_BIN"
  fi
  [[ -x "$COLLECTION_BIN" ]] || { echo "[error] collection binary absent: $COLLECTION_BIN" >&2; exit 2; }
  current_fingerprint="$({
    sha256sum "$COLLECTION_BIN" "$TRACE_FILE"
    printf '%s\n' "warmup=$TRAIN_WARMUP" "simulation=$TRAIN_SIM"
  } | sha256sum | awk '{print $1}')"
  if [[ -f "$fingerprint_file" ]]; then
    previous_fingerprint="$(tr -d '[:space:]' < "$fingerprint_file")"
  fi
  if [[ "$current_fingerprint" != "$previous_fingerprint" ]]; then
    collection_force=1
    echo "[rerun] collection inputs changed; stale training data will not be reused"
  fi
  FORCE="$collection_force" run_events "training no-prefetch stream" "$COLLECTION_BIN" no_pref \
    "$TRAIN_EVENTS" "$TRAIN_LOG" "$TRAIN_WARMUP" "$TRAIN_SIM"
  if [[ "$collection_force" == 1 || ! -s "$TRAIN_STREAM" ]] || ! gzip -t "$TRAIN_STREAM" >/dev/null 2>&1; then
    python3 "$NORMALIZE_TRAIN_STREAM" \
      --events "$TRAIN_EVENTS" --trace "$TRACE" --out "$TRAIN_STREAM" \
      --meta-out "$TRAIN_STREAM.meta.json"
  else
    echo "[skip training stream]"
  fi
  printf '%s\n' "$current_fingerprint" > "$fingerprint_file"
  echo "[collect done] $TRAIN_STREAM"
}

train_stage() {
  require_file "$TRAIN_LSTM"
  require_file "$TRAIN_STREAM"
  python3 "$TRAIN_LSTM" \
    --train-stream "$TRAIN_STREAM" \
    --artifact-dir "$ARTIFACT_DIR" \
    --run-id "$RUN_ID" --seed "$SEED" --device "$TRAIN_DEVICE"
  echo "[train done] $RUNTIME_MODEL"
}

write_identity() {
  python3 - "$IDENTITY" "$ROOT" "$CHAMP_DIR" "$TRACE_FILE" "$EVAL_BIN" \
    "$STRIDE_CONFIG" "$PATCH_LOGGER" "$TRAIN_LSTM" "$RUNTIME_SOURCE" "$RUNTIME_HEADER" \
    "$TRAIN_STREAM" "$RUNTIME_MODEL" "$ARTIFACT_DIR/run_metadata.json" "$RUN_ID" <<'PY'
from __future__ import print_function
import hashlib
import json
import subprocess
import sys
from pathlib import Path

(out, root, champ, trace_file, binary, config, logger, trainer, runtime_source,
 runtime_header, train_stream, runtime_model, training_meta, run_id) = map(Path, sys.argv[1:])

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

def head(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"]).decode().strip()

payload = {
    "schema": "602_matched_stride_lstm_run_identity_v3",
    "trace": "602.gcc_s-734B",
    "run_id": str(run_id),
    "cache_arch_head": head(root),
    "champsim_head": head(champ),
    "expected_champsim_head": "fd26fc51a44554976022e1ee13e73e7b06e2307e",
    "libbf_head": head(champ / "libbf"),
    "expected_libbf_head": "4c9efc1a4db7ed1ccf54cf0bd3a3641ce579206c",
    "trace_file": str(trace_file),
    "trace_sha256": sha(trace_file),
    "simulator_binary": str(binary),
    "simulator_binary_sha256": sha(binary),
    "baseline_config_path": str(config),
    "baseline_config_sha256": sha(config),
    "baseline_config": {
        "l2c_prefetcher_types": "stride",
        "stride_num_trackers": 64,
        "stride_pref_degree": 2,
    },
    "logger_patch_sha256": sha(logger),
    "trainer_sha256": sha(trainer),
    "runtime_source_sha256": sha(runtime_source),
    "runtime_header_sha256": sha(runtime_header),
    "train_stream": str(train_stream),
    "train_stream_sha256": sha(train_stream),
    "runtime_model": str(runtime_model),
    "runtime_model_sha256": sha(runtime_model),
    "training_metadata_sha256": sha(training_meta),
    "training_window": {"warmup_instructions": 0, "simulation_instructions": 20000000},
    "unseen_guard_before_measurement_instructions": 5000000,
    "evaluation_window": {"warmup_instructions": 25000000, "simulation_instructions": 25000000},
    "matched_runtime_input": "live PC/address L2 callback stream plus causal state derived from PC/address only",
    "same_binary_methods": ["no_pref", "stride", "matched_stride_lstm"],
    "primary_nn_execution": "live_in_simulator_not_keyed_replay",
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("[write] {}".format(out))
PY
}

evaluate_stage() {
  local fingerprint_file="$OUT_ROOT/evaluation_fingerprint.sha256"
  local current_fingerprint previous_fingerprint=""
  local evaluation_force="$FORCE"
  require_common_files
  require_file "$RUNTIME_MODEL"
  require_file "$ARTIFACT_DIR/run_metadata.json"
  if [[ "$BUILD" == 1 ]]; then
    patch_logger_and_dependencies
    CHAMP_DIR="$CHAMP_DIR" OUT="$EVAL_BIN" bash "$BUILD_LIVE_BIN"
  fi
  [[ -x "$EVAL_BIN" ]] || { echo "[error] live evaluation binary absent: $EVAL_BIN" >&2; exit 2; }
  current_fingerprint="$({
    sha256sum "$EVAL_BIN" "$RUNTIME_MODEL" "$STRIDE_CONFIG" "$TRACE_FILE"
    printf '%s\n' "warmup=$EVAL_WARMUP" "simulation=$EVAL_SIM"
  } | sha256sum | awk '{print $1}')"
  if [[ -f "$fingerprint_file" ]]; then
    previous_fingerprint="$(tr -d '[:space:]' < "$fingerprint_file")"
  fi
  if [[ "$current_fingerprint" != "$previous_fingerprint" ]]; then
    evaluation_force=1
    echo "[rerun] evaluation inputs changed; stale event/log files will not be reused"
  fi
  cp -f "$STRIDE_CONFIG" "$CONFIG_SNAPSHOT_DIR/602_matched_stride.ini"
  FORCE="$evaluation_force" run_events "same-binary no-prefetch control" "$EVAL_BIN" no_pref \
    "$NO_PREF_EVENTS" "$NO_PREF_LOG" "$EVAL_WARMUP" "$EVAL_SIM"
  FORCE="$evaluation_force" run_events "same-binary stride baseline" "$EVAL_BIN" stride \
    "$STRIDE_EVENTS" "$STRIDE_LOG" "$EVAL_WARMUP" "$EVAL_SIM"
  FORCE="$evaluation_force" run_events "same-binary live matched LSTM" "$EVAL_BIN" matched_lstm \
    "$LSTM_EVENTS" "$LSTM_LOG" "$EVAL_WARMUP" "$EVAL_SIM"
  grep -q '^stride_num_trackers 64$' "$STRIDE_LOG" || { echo "[error] stride log does not confirm 64 trackers" >&2; exit 4; }
  grep -q '^stride_pref_degree 2$' "$STRIDE_LOG" || { echo "[error] stride log does not confirm degree 2" >&2; exit 4; }
  grep -q 'adding L2C_PREFETCHER: matched_stride_lstm (live inference)' "$LSTM_LOG" || { echo "[error] live LSTM registry marker absent" >&2; exit 4; }
  grep -q '^matched_stride_lstm_parameter_count 545$' "$LSTM_LOG" || { echo "[error] live LSTM parameter marker absent" >&2; exit 4; }
  grep -q '^matched_stride_lstm_runtime_inputs pc,address,causal_pc_address_history$' "$LSTM_LOG" || { echo "[error] live LSTM input marker absent" >&2; exit 4; }
  python3 "$PARSE_BASELINE" \
    --log-root "$LOG_DIR" --out "$BASELINE_SUMMARY" \
    --traces "$TRACE" --prefetchers "no_pref stride" --nodup
  write_identity
  printf '%s\n' "$current_fingerprint" > "$fingerprint_file"
  printf '%s\n' \
    "RUN_KIND=602_live_matched_stride_lstm" \
    "RUN_ID=$RUN_ID" "TRACE=$TRACE" \
    "TRAIN_WARMUP=$TRAIN_WARMUP" "TRAIN_SIM=$TRAIN_SIM" \
    "EVAL_WARMUP=$EVAL_WARMUP" "EVAL_SIM=$EVAL_SIM" \
    "EVAL_BIN=$EVAL_BIN" "RUNTIME_MODEL=$RUNTIME_MODEL" \
    "STRIDE_CONFIG=$STRIDE_CONFIG" \
    "CHAMPSIM_HEAD=$(git -C "$CHAMP_DIR" rev-parse HEAD)" \
    > "$OUT_ROOT/RUN_INFO.txt"
  echo "[evaluate done] $OUT_ROOT"
}

analyze_stage() {
  require_common_files
  python3 "$COMPARE" \
    --baseline-summary "$BASELINE_SUMMARY" \
    --lstm-log "$LSTM_LOG" \
    --training-metadata "$ARTIFACT_DIR/run_metadata.json" \
    --run-identity "$IDENTITY" \
    --out-csv "$OUT_ROOT/matched_comparison.csv" \
    --out-json "$OUT_ROOT/matched_comparison.json"
  echo "[analyze done] $OUT_ROOT/matched_comparison.json"
}

case "$STAGE" in
  collect) collect_stage ;;
  train) train_stage ;;
  evaluate) evaluate_stage ;;
  analyze) analyze_stage ;;
  all)
    collect_stage
    train_stage
    evaluate_stage
    analyze_stage
    ;;
esac
