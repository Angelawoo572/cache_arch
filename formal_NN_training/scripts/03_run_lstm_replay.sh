#!/usr/bin/env bash
# 03_run_lstm_replay.sh
#
# Single-trace helper for one LSTM replay configuration.
# For multi-threshold or multi-trace sweeps, prefer 12_replay_trace_sweep.sh.
#
# Usage:
#   TRACE=619.lbm_s-4268B \
#   POLICY=threshold PREFETCH_THRESHOLD=0.20 BYPASS_THRESHOLD=1.00 \
#   MODEL_TAG=LSTM_lbm_s-4268B_L2_replayidx_hex_th0.20_bp1.00 \
#     bash formal_NN_training/scripts/03_run_lstm_replay.sh

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

TRACE="${TRACE:-602.gcc_s-734B}"
SHORT_TAG="${TRACE#*.}"
SHORT_TAG="${SHORT_TAG%%.*}"
WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"
POLICY="${POLICY:-threshold}"              # action | threshold
PREFETCH_THRESHOLD="${PREFETCH_THRESHOLD:-0.20}"
BYPASS_THRESHOLD="${BYPASS_THRESHOLD:-1.00}"
ALLOW_BYPASS_PREFETCH="${ALLOW_BYPASS_PREFETCH:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MODEL_TAG="${MODEL_TAG:-LSTM_${SHORT_TAG}_L2_replayidx_hex_th${PREFETCH_THRESHOLD}_bp${BYPASS_THRESHOLD}}"

TRACE_DIR="${TRACE_DIR:-$ROOT/traces}"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
NO_BIN="${NO_BIN:-$CHAMP_DIR/bin/champsim.baseline}"
SPP_BIN="${SPP_BIN:-$CHAMP_DIR/bin/champsim.l2_spp}"
if [ ! -x "$SPP_BIN" ] && [ -x "$CHAMP_DIR/bin/champsim.l2_spp_cand" ]; then
  SPP_BIN="$CHAMP_DIR/bin/champsim.l2_spp_cand"
fi
REPL_BIN="${REPL_BIN:-$CHAMP_DIR/bin/champsim.l2_replayer}"
if [ ! -x "$REPL_BIN" ] && [ -x "$CHAMP_DIR/bin/champsim.replayer" ]; then
  REPL_BIN="$CHAMP_DIR/bin/champsim.replayer"
fi

TR_FILE="$TRACE_DIR/${TRACE}.champsimtrace.xz"
ACTIONS_DEFAULT="$ROOT/formal_NN_training/artifacts/by_trace/${TRACE}/full_lstm_cache_actions.csv"
ACTIONS_GLOBAL="$ROOT/formal_NN_training/artifacts/full_lstm_cache_actions.csv"
ACTIONS="${ACTIONS:-$ACTIONS_DEFAULT}"
if [ ! -f "$ACTIONS" ] && [ -f "$ACTIONS_GLOBAL" ]; then
  ACTIONS="$ACTIONS_GLOBAL"
fi

OUT_ROOT="$ROOT/formal_NN_training/results/replay_compare"
LOG_DIR="$OUT_ROOT/logs"
PFETCH_DIR="$OUT_ROOT/prefetch_lists"
SUMMARY="$OUT_ROOT/summary_${TRACE}.csv"
mkdir -p "$LOG_DIR" "$PFETCH_DIR"

PFETCH="${PFETCH:-$PFETCH_DIR/prefetch_list_${MODEL_TAG}.txt}"

require_file () {
  local p="$1"
  local label="$2"
  if [ ! -f "$p" ]; then
    echo "[error] missing $label: $p"
    exit 1
  fi
}

require_exec () {
  local p="$1"
  local label="$2"
  if [ ! -x "$p" ]; then
    echo "[error] missing executable $label: $p"
    exit 1
  fi
}

require_file "$TR_FILE" "trace"
require_file "$ACTIONS" "LSTM action CSV"
require_exec "$NO_BIN" "no-prefetch baseline binary"
require_exec "$SPP_BIN" "SPP binary"
require_exec "$REPL_BIN" "list-replayer binary"

run_with_heartbeat () {
  local log="$1"; shift
  if [ "$SKIP_EXISTING" = "1" ] && [ -s "$log" ]; then
    echo "  [skip existing] $log"
    return 0
  fi
  "$@" > "$log" 2>&1 &
  local pid=$!
  local seconds=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    seconds=$((seconds + 60))
    local last
    last=$(tail -1 "$log" 2>/dev/null | tr -d '\n' | head -c 120 || true)
    printf "  ...running elapsed=%dm%02ds last='%s'\n" $((seconds/60)) $((seconds%60)) "$last"
  done
  wait "$pid"
}

parse_ipc () {
  grep -E "CPU 0 cumulative IPC" "$1" 2>/dev/null | tail -1 | sed -E 's/.*CPU 0 cumulative IPC: ([0-9.]+).*/\1/' || true
}

parse_issued () {
  grep "issued.*prefetches" "$1" 2>/dev/null | tail -1 | grep -oE "issued [0-9]+ prefetches over [0-9]+" || true
}

extract_first_number () {
  echo "$1" | grep -oE "[0-9]+" | head -1 || true
}

extract_second_number () {
  echo "$1" | grep -oE "[0-9]+" | head -2 | tail -1 || true
}

speedup_vs () {
  local base="$1"
  local test="$2"
  if [[ "$base" =~ ^[0-9.]+$ ]] && [[ "$test" =~ ^[0-9.]+$ ]]; then
    python3 -c "print('{:.4f}'.format(float('$test')/float('$base')))"
  else
    echo "NA"
  fi
}

append_summary () {
  local method="$1"
  local ipc="$2"
  local speedup="$3"
  local issued="$4"
  local accesses="$5"
  local log="$6"
  local pfetch="$7"

  if [ ! -f "$SUMMARY" ]; then
    echo "trace,warmup,sim,method,ipc,speedup_vs_no_prefetch,prefetches_issued,accesses,log,pfetch" > "$SUMMARY"
  fi
  echo "$TRACE,$WARMUP,$SIM,$method,$ipc,$speedup,$issued,$accesses,$log,$pfetch" >> "$SUMMARY"
}

printf '%s\n' '============================================================'
printf 'FORMAL LSTM REPLAY COMPARE\n'
printf 'repo       : %s\n' "$ROOT"
printf 'trace      : %s\n' "$TRACE"
printf 'warmup/sim : %s / %s\n' "$WARMUP" "$SIM"
printf 'actions    : %s\n' "$ACTIONS"
printf 'policy     : %s\n' "$POLICY"
printf 'thresholds : prefetch=%s bypass=%s allow_bypass_prefetch=%s\n' "$PREFETCH_THRESHOLD" "$BYPASS_THRESHOLD" "$ALLOW_BYPASS_PREFETCH"
printf 'out root   : %s\n' "$OUT_ROOT"
printf '%s\n' '============================================================'

echo
printf '[1/4] Convert LSTM action CSV -> prefetch list\n'
cmd=(python3 formal_NN_training/scripts/02_actions_to_prefetch_list.py
  --actions "$ACTIONS"
  --out "$PFETCH"
  --policy "$POLICY"
  --prefetch-threshold "$PREFETCH_THRESHOLD"
  --bypass-threshold "$BYPASS_THRESHOLD")
if [ "$ALLOW_BYPASS_PREFETCH" = "1" ]; then
  cmd+=(--allow-bypass-prefetch)
fi
"${cmd[@]}"

if [ ! -s "$PFETCH" ]; then
  echo "[error] generated empty prefetch list: $PFETCH"
  exit 1
fi
PFETCH_LINES=$(wc -l < "$PFETCH")
echo "[check] prefetch list lines=$PFETCH_LINES path=$PFETCH"

echo
printf '[2/4] Run no-prefetch baseline\n'
NO_LOG="$LOG_DIR/${TRACE}.no_prefetch.log"
run_with_heartbeat "$NO_LOG" "$NO_BIN" --warmup-instructions "$WARMUP" --simulation-instructions "$SIM" "$TR_FILE"
NO_IPC=$(parse_ipc "$NO_LOG")
NO_IPC="${NO_IPC:-NA}"
echo "[parse] no-prefetch IPC=$NO_IPC"
append_summary "no_prefetch" "$NO_IPC" "1.0000" "NA" "NA" "$NO_LOG" "NA"

echo
printf '[3/4] Run SPP baseline\n'
SPP_LOG="$LOG_DIR/${TRACE}.spp.log"
run_with_heartbeat "$SPP_LOG" "$SPP_BIN" --warmup-instructions "$WARMUP" --simulation-instructions "$SIM" "$TR_FILE"
SPP_IPC=$(parse_ipc "$SPP_LOG")
SPP_IPC="${SPP_IPC:-NA}"
SPP_SPEEDUP=$(speedup_vs "$NO_IPC" "$SPP_IPC")
echo "[parse] SPP IPC=$SPP_IPC speedup=$SPP_SPEEDUP"
append_summary "spp" "$SPP_IPC" "$SPP_SPEEDUP" "NA" "NA" "$SPP_LOG" "NA"

echo
printf '[4/4] Run LSTM list replay\n'
LSTM_LOG="$LOG_DIR/${TRACE}.${MODEL_TAG}.log"
export PFETCH_LIST_PATH="$PFETCH"
run_with_heartbeat "$LSTM_LOG" "$REPL_BIN" --warmup-instructions "$WARMUP" --simulation-instructions "$SIM" "$TR_FILE"
LSTM_IPC=$(parse_ipc "$LSTM_LOG")
LSTM_IPC="${LSTM_IPC:-NA}"
LSTM_SPEEDUP=$(speedup_vs "$NO_IPC" "$LSTM_IPC")
ISSUED_LINE=$(parse_issued "$LSTM_LOG")
ISSUED_N=$(extract_first_number "$ISSUED_LINE")
ACCESS_N=$(extract_second_number "$ISSUED_LINE")
ISSUED_N="${ISSUED_N:-NA}"
ACCESS_N="${ACCESS_N:-NA}"
echo "[parse] LSTM IPC=$LSTM_IPC speedup=$LSTM_SPEEDUP"
echo "[parse] ${ISSUED_LINE:-issued line not found}"
append_summary "$MODEL_TAG" "$LSTM_IPC" "$LSTM_SPEEDUP" "$ISSUED_N" "$ACCESS_N" "$LSTM_LOG" "$PFETCH"

echo
printf '%s\n' '============================================================'
printf 'SUMMARY: %s\n' "$SUMMARY"
printf '%s\n' '============================================================'
column -t -s, "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
