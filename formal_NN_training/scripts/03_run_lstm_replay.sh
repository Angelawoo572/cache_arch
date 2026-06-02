#!/usr/bin/env bash
# 03_run_lstm_replay.sh
#
# Self-contained formal_NN_training replay/compare script.
# It does NOT call projects/legacy_gru_prefetch scripts.
#
# It runs three methods on the same trace/window:
#   1. no-prefetch baseline       -> champsim.baseline
#   2. SPP baseline               -> champsim.l2_spp or champsim.l2_spp_cand
#   3. LSTM action-list replay    -> champsim.replayer + PFETCH_LIST_PATH
#
# Usage from repo root:
#   TRACE=602.gcc_s-734B bash formal_NN_training/scripts/03_run_lstm_replay.sh
#
# Common overrides:
#   TRACE=602.gcc_s-734B WARMUP=25000000 SIM=25000000 \
#   POLICY=action PREFETCH_THRESHOLD=0.50 BYPASS_THRESHOLD=0.60 \
#   bash formal_NN_training/scripts/03_run_lstm_replay.sh
#
# Output:
#   formal_NN_training/results/replay_compare/logs/*.log
#   formal_NN_training/results/replay_compare/prefetch_lists/*.txt
#   formal_NN_training/results/replay_compare/summary_<trace>.csv

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

TRACE="${TRACE:-602.gcc_s-734B}"
SHORT_TAG="${TRACE#*.}"
WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"
POLICY="${POLICY:-action}"              # action | threshold
PREFETCH_THRESHOLD="${PREFETCH_THRESHOLD:-0.50}"
BYPASS_THRESHOLD="${BYPASS_THRESHOLD:-0.60}"
MODEL_TAG="${MODEL_TAG:-LSTM_${SHORT_TAG}_${POLICY}_th${PREFETCH_THRESHOLD}_bp${BYPASS_THRESHOLD}}"

TRACE_DIR="${TRACE_DIR:-$ROOT/traces}"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
NO_BIN="${NO_BIN:-$CHAMP_DIR/bin/champsim.baseline}"
SPP_BIN="${SPP_BIN:-$CHAMP_DIR/bin/champsim.l2_spp}"
if [ ! -x "$SPP_BIN" ] && [ -x "$CHAMP_DIR/bin/champsim.l2_spp_cand" ]; then
  SPP_BIN="$CHAMP_DIR/bin/champsim.l2_spp_cand"
fi
REPL_BIN="${REPL_BIN:-$CHAMP_DIR/bin/champsim.replayer}"

TR_FILE="$TRACE_DIR/${TRACE}.champsimtrace.xz"
ACTIONS_DEFAULT="$ROOT/formal_NN_training/artifacts/full_lstm_cache_actions.csv"
ACTIONS_FALLBACK="$ROOT/formal_NN_training/artifacts/val_lstm_cache_actions.csv"
ACTIONS="${ACTIONS:-$ACTIONS_DEFAULT}"
if [ ! -f "$ACTIONS" ] && [ -f "$ACTIONS_FALLBACK" ]; then
  ACTIONS="$ACTIONS_FALLBACK"
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
  grep -E "cumulative IPC" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1 || true
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

echo "============================================================"
echo "FORMAL LSTM REPLAY COMPARE"
echo "repo       : $ROOT"
echo "trace      : $TRACE"
echo "warmup/sim : $WARMUP / $SIM"
echo "actions    : $ACTIONS"
echo "policy     : $POLICY"
echo "thresholds : prefetch=$PREFETCH_THRESHOLD bypass=$BYPASS_THRESHOLD"
echo "out root   : $OUT_ROOT"
echo "============================================================"

echo
echo "[1/4] Convert LSTM action CSV -> prefetch list"
python3 formal_NN_training/scripts/02_actions_to_prefetch_list.py \
  --actions "$ACTIONS" \
  --out "$PFETCH" \
  --policy "$POLICY" \
  --prefetch-threshold "$PREFETCH_THRESHOLD" \
  --bypass-threshold "$BYPASS_THRESHOLD"

if [ ! -s "$PFETCH" ]; then
  echo "[error] generated empty prefetch list: $PFETCH"
  exit 1
fi
PFETCH_LINES=$(wc -l < "$PFETCH")
echo "[check] prefetch list lines=$PFETCH_LINES path=$PFETCH"

echo
echo "[2/4] Run no-prefetch baseline"
NO_LOG="$LOG_DIR/${TRACE}.no_prefetch.log"
run_with_heartbeat "$NO_LOG" "$NO_BIN" --warmup-instructions "$WARMUP" --simulation-instructions "$SIM" "$TR_FILE"
NO_IPC=$(parse_ipc "$NO_LOG")
NO_IPC="${NO_IPC:-NA}"
echo "[parse] no-prefetch IPC=$NO_IPC"
append_summary "no_prefetch" "$NO_IPC" "1.0000" "NA" "NA" "$NO_LOG" "NA"

echo
echo "[3/4] Run SPP baseline"
SPP_LOG="$LOG_DIR/${TRACE}.spp.log"
run_with_heartbeat "$SPP_LOG" "$SPP_BIN" --warmup-instructions "$WARMUP" --simulation-instructions "$SIM" "$TR_FILE"
SPP_IPC=$(parse_ipc "$SPP_LOG")
SPP_IPC="${SPP_IPC:-NA}"
SPP_SPEEDUP=$(speedup_vs "$NO_IPC" "$SPP_IPC")
SPP_FINAL=$(grep "SPP_FINAL" "$SPP_LOG" 2>/dev/null | tail -1 || true)
echo "[parse] SPP IPC=$SPP_IPC speedup=$SPP_SPEEDUP"
if [ -n "$SPP_FINAL" ]; then
  echo "[parse] $SPP_FINAL"
fi
append_summary "spp" "$SPP_IPC" "$SPP_SPEEDUP" "NA" "NA" "$SPP_LOG" "NA"

echo
echo "[4/4] Run LSTM list replay"
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
echo "============================================================"
echo "SUMMARY: $SUMMARY"
echo "============================================================"
column -t -s, "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
