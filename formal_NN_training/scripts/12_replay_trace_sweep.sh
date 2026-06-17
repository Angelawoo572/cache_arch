#!/usr/bin/env bash
# Prepare Colab action outputs and run no-prefetch / SPP / LSTM threshold sweeps.
#
# Usage after Colab output is copied to artifacts/packed/<tag>/:
#   TRACES="623.xalancbmk_s-700B 605.mcf_s-994B" \
#   THRESHOLDS="0p00001:0.00001 0p0001:0.0001 0p001:0.001" \
#   ALLOW_BYPASS_PREFETCH=1 MAX_JOBS=2 \
#     bash formal_NN_training/scripts/12_replay_trace_sweep.sh
#
# This script is safe to rerun. Existing baseline/SPP/LSTM logs are skipped unless
# FORCE_REPLAY=1 is set.

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

TRACES_STR="${TRACES:-602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B}"
WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"
MAX_JOBS="${MAX_JOBS:-1}"
FORCE_PREPARE="${FORCE_PREPARE:-1}"
FORCE_REPLAY="${FORCE_REPLAY:-0}"
ALLOW_BYPASS_PREFETCH="${ALLOW_BYPASS_PREFETCH:-1}"
POLICY="${POLICY:-threshold}"
BYPASS_THRESHOLD="${BYPASS_THRESHOLD:-1.00}"
THRESHOLDS="${THRESHOLDS:-0p00001:0.00001 0p00005:0.00005 0p0001:0.0001 0p0005:0.0005 0p001:0.001 0p005:0.005 0p01:0.01 0p02:0.02 0p05:0.05 0p10:0.10 0p20:0.20}"

TRACE_DIR="${TRACE_DIR:-$ROOT/traces}"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
NO_BIN="${NO_BIN:-$CHAMP_DIR/bin/champsim.baseline}"
SPP_BIN="${SPP_BIN:-$CHAMP_DIR/bin/champsim.l2_spp}"
if [ ! -x "$SPP_BIN" ] && [ -x "$CHAMP_DIR/bin/champsim.l2_spp_cand" ]; then
  SPP_BIN="$CHAMP_DIR/bin/champsim.l2_spp_cand"
fi
REPL_BIN="${REPL_BIN:-$CHAMP_DIR/bin/champsim.l2_replayer}"

OUT_ROOT="$ROOT/formal_NN_training/results/replay_compare"
LOG_DIR="$OUT_ROOT/logs"
PFETCH_DIR="$OUT_ROOT/prefetch_lists"
mkdir -p "$LOG_DIR" "$PFETCH_DIR"

require_exec () {
  local p="$1"
  if [ ! -x "$p" ]; then
    echo "[error] missing executable: $p"
    exit 1
  fi
}

require_exec "$NO_BIN"
require_exec "$SPP_BIN"
require_exec "$REPL_BIN"

short_tag () {
  local trace="$1"
  local s="${trace#*.}"
  s="${s%%.*}"
  echo "$s"
}

run_champsim_if_needed () {
  local log="$1"; shift
  if [ "$FORCE_REPLAY" != "1" ] && [ -s "$log" ]; then
    echo "[skip existing log] $log"
    return 0
  fi
  echo "[run] $log"
  "$@" > "$log" 2>&1
}

make_list () {
  local actions="$1"
  local pfetch="$2"
  local th="$3"
  local cmd=(python3 formal_NN_training/scripts/02_actions_to_prefetch_list.py
    --actions "$actions"
    --out "$pfetch"
    --policy "$POLICY"
    --prefetch-threshold "$th"
    --bypass-threshold "$BYPASS_THRESHOLD")
  if [ "$ALLOW_BYPASS_PREFETCH" = "1" ]; then
    cmd+=(--allow-bypass-prefetch)
  fi
  "${cmd[@]}"
}

run_one_trace () {
  local trace="$1"
  local tag="${trace%%.*}"
  local short
  short="$(short_tag "$trace")"
  local trfile="$TRACE_DIR/${trace}.champsimtrace.xz"
  local packed_dir="$ROOT/formal_NN_training/results/LSTM/draft/artifacts/packed/$tag"
  local actions="$ROOT/formal_NN_training/results/LSTM/draft/artifacts/by_trace/${trace}/full_lstm_cache_actions.csv"

  echo "============================================================"
  echo "[trace replay sweep] $trace"
  echo "warmup/sim : $WARMUP / $SIM"
  echo "tag        : $tag"
  echo "short      : $short"
  echo "allow bypass prefetch: $ALLOW_BYPASS_PREFETCH"
  echo "============================================================"

  if [ ! -s "$trfile" ]; then
    echo "[skip missing trace] $trfile"
    return 0
  fi

  if [ "$FORCE_PREPARE" = "1" ] || [ ! -s "$actions" ]; then
    if ls "$packed_dir"/full_lstm_cache_actions.csv.gz.part_* >/dev/null 2>&1; then
      python3 formal_NN_training/scripts/07_prepare_actions_for_replay.py \
        --trace "$trace" \
        --restore-packed \
        --copy-default
    else
      echo "[warn] no packed Colab actions under $packed_dir; using existing actions if present"
    fi
  fi

  if [ ! -s "$actions" ]; then
    echo "[skip no actions] $actions"
    return 0
  fi

  run_champsim_if_needed "$LOG_DIR/${trace}.no_prefetch.log" \
    "$NO_BIN" --warmup-instructions "$WARMUP" --simulation-instructions "$SIM" "$trfile"

  run_champsim_if_needed "$LOG_DIR/${trace}.spp.log" \
    "$SPP_BIN" --warmup-instructions "$WARMUP" --simulation-instructions "$SIM" "$trfile"

  for spec in $THRESHOLDS; do
    local th_tag="${spec%%:*}"
    local th="${spec#*:}"
    local suffix="th${th_tag}"
    if [ "$ALLOW_BYPASS_PREFETCH" = "1" ]; then
      suffix="${suffix}_allow_bypass"
    else
      suffix="${suffix}_bp${BYPASS_THRESHOLD}"
      suffix="${suffix//./p}"
    fi
    local model_tag="LSTM_${short}_L2_replayidx_hex_${suffix}"
    local pfetch="$PFETCH_DIR/prefetch_list_${model_tag}.txt"
    local log="$LOG_DIR/${trace}.${model_tag}.log"

    echo "============================================================"
    echo "[make list] trace=$trace th=$th model=$model_tag"
    echo "============================================================"
    make_list "$actions" "$pfetch" "$th"

    local lines
    lines=$(wc -l < "$pfetch" || echo 0)
    echo "[list lines] $lines"
    if [ "$lines" -eq 0 ]; then
      echo "[skip replay] empty list"
      continue
    fi

    export PFETCH_LIST_PATH="$pfetch"
    run_champsim_if_needed "$log" \
      "$REPL_BIN" --warmup-instructions "$WARMUP" --simulation-instructions "$SIM" "$trfile"
    unset PFETCH_LIST_PATH
  done

  python3 formal_NN_training/scripts/09_compare_spp_lstm_accuracy.py \
    --trace "$trace" \
    --include-lstm LSTM \
    --out "$OUT_ROOT/accuracy_compare_${trace}.csv"
}

running=0
for trace in $TRACES_STR; do
  run_one_trace "$trace" &
  running=$((running + 1))
  if [ "$running" -ge "$MAX_JOBS" ]; then
    wait -n
    running=$((running - 1))
  fi
done
wait

echo "[done] logs under $LOG_DIR"
