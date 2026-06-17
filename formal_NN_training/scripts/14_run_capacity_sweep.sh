#!/usr/bin/env bash
# Run L2 capacity sweep for traces that already have prepared LSTM actions.
#
# Expected binaries:
#   external/ChampSim/bin/champsim.baseline.L2_<CAP>
#   external/ChampSim/bin/champsim.spp.L2_<CAP>
#   external/ChampSim/bin/champsim.replayer.L2_<CAP>
#
# Usage:
#   TRACES="602.gcc_s-734B 619.lbm_s-4268B" CAPS="256K 512K 1M 2M" MAX_JOBS=2 \
#     bash formal_NN_training/scripts/14_run_capacity_sweep.sh

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

TRACES_STR="${TRACES:-602.gcc_s-734B 619.lbm_s-4268B}"
CAPS_STR="${CAPS:-256K 512K 1M 2M}"
WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"
MAX_JOBS="${MAX_JOBS:-1}"
FORCE_REPLAY="${FORCE_REPLAY:-0}"
PREFETCH_THRESHOLD="${PREFETCH_THRESHOLD:-0.20}"
BYPASS_THRESHOLD="${BYPASS_THRESHOLD:-1.00}"
ALLOW_BYPASS_PREFETCH="${ALLOW_BYPASS_PREFETCH:-0}"

CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
TRACE_DIR="${TRACE_DIR:-$ROOT/traces}"
OUT_ROOT="$ROOT/formal_NN_training/results/capacity_sweep"
LOG_DIR="$OUT_ROOT/logs"
PFETCH_DIR="$OUT_ROOT/prefetch_lists"
mkdir -p "$LOG_DIR" "$PFETCH_DIR"

short_tag () {
  local trace="$1"
  local s="${trace#*.}"
  s="${s%%.*}"
  echo "$s"
}

run_if_needed () {
  local log="$1"; shift
  if [ "$FORCE_REPLAY" != "1" ] && [ -s "$log" ]; then
    echo "[skip existing] $log"
    return 0
  fi
  echo "[run] $log"
  "$@" > "$log" 2>&1
}

make_pfetch () {
  local trace="$1"
  local actions="$ROOT/formal_NN_training/results/LSTM/draft/artifacts/by_trace/${trace}/full_lstm_cache_actions.csv"
  local pfetch="$PFETCH_DIR/prefetch_list_${trace}_th${PREFETCH_THRESHOLD}_bp${BYPASS_THRESHOLD}.txt"

  if [ ! -s "$actions" ]; then
    echo "[error] missing prepared actions: $actions"
    echo "Run 07_prepare_actions_for_replay.py or 12_replay_trace_sweep.sh first."
    exit 1
  fi

  local cmd=(python3 formal_NN_training/scripts/02_actions_to_prefetch_list.py
    --actions "$actions"
    --out "$pfetch"
    --policy threshold
    --prefetch-threshold "$PREFETCH_THRESHOLD"
    --bypass-threshold "$BYPASS_THRESHOLD")
  if [ "$ALLOW_BYPASS_PREFETCH" = "1" ]; then
    cmd+=(--allow-bypass-prefetch)
  fi
  "${cmd[@]}"

  if [ ! -s "$pfetch" ]; then
    echo "[error] empty prefetch list: $pfetch"
    exit 1
  fi
  echo "$pfetch"
}

run_one () {
  local trace="$1"
  local cap="$2"
  local trfile="$TRACE_DIR/${trace}.champsimtrace.xz"
  local pfetch="$PFETCH_DIR/prefetch_list_${trace}_th${PREFETCH_THRESHOLD}_bp${BYPASS_THRESHOLD}.txt"

  local no_bin="$CHAMP_DIR/bin/champsim.baseline.L2_${cap}"
  local spp_bin="$CHAMP_DIR/bin/champsim.spp.L2_${cap}"
  local repl_bin="$CHAMP_DIR/bin/champsim.replayer.L2_${cap}"

  for p in "$trfile" "$no_bin" "$spp_bin" "$repl_bin" "$pfetch"; do
    if [ ! -s "$p" ] && [ ! -x "$p" ]; then
      echo "[error] missing required file/binary: $p"
      exit 1
    fi
  done

  echo "============================================================"
  echo "[capacity replay] trace=$trace cap=$cap"
  echo "============================================================"

  run_if_needed "$LOG_DIR/${trace}.L2_${cap}.no_prefetch.log" \
    "$no_bin" --warmup-instructions "$WARMUP" --simulation-instructions "$SIM" "$trfile"

  run_if_needed "$LOG_DIR/${trace}.L2_${cap}.spp.log" \
    "$spp_bin" --warmup-instructions "$WARMUP" --simulation-instructions "$SIM" "$trfile"

  export PFETCH_LIST_PATH="$pfetch"
  run_if_needed "$LOG_DIR/${trace}.L2_${cap}.LSTM_th${PREFETCH_THRESHOLD}_bp${BYPASS_THRESHOLD}.log" \
    "$repl_bin" --warmup-instructions "$WARMUP" --simulation-instructions "$SIM" "$trfile"
  unset PFETCH_LIST_PATH
}

for trace in $TRACES_STR; do
  make_pfetch "$trace" >/dev/null
done

running=0
for trace in $TRACES_STR; do
  for cap in $CAPS_STR; do
    run_one "$trace" "$cap" &
    running=$((running + 1))
    if [ "$running" -ge "$MAX_JOBS" ]; then
      wait -n
      running=$((running - 1))
    fi
  done
done
wait

echo "[done] capacity logs under $LOG_DIR"
