#!/usr/bin/env bash
# run_nn_replay.sh -- v11 (real run)
#
# Defaults to 25M warmup + 25M sim (paper-grade short-mix). Set
# WARMUP=200000000 SIM=500000000 for full DPC-3 scale (~4.5h per run).
#
# This is the same logic as the v7 script -- only the WARMUP/SIM defaults
# changed.

set -uo pipefail

WORKDIR="$(pwd)"
CHAMP_DIR="$WORKDIR/external/ChampSim"
TRACE_DIR="$WORKDIR/traces"
OUT_DIR="$WORKDIR/results"
mkdir -p "$OUT_DIR"

TRACE="${TRACE:-605.mcf_s-994B}"
WARMUP="${WARMUP:-25000000}"        # REAL default
SIM="${SIM:-25000000}"              # REAL default
PFETCH="${PFETCH:-$WORKDIR/prefetch_list.txt}"
MODEL_TAG="${MODEL_TAG:-unknown}"

TR_FILE="$TRACE_DIR/${TRACE}.champsimtrace.xz"
BASE_BIN="$CHAMP_DIR/bin/champsim.baseline"
REPL_BIN="$CHAMP_DIR/bin/champsim.replayer"

[ -x "$BASE_BIN" ] || { echo "[error] $BASE_BIN missing"; exit 1; }
[ -x "$REPL_BIN" ] || { echo "[error] $REPL_BIN missing"; exit 1; }
[ -f "$TR_FILE"  ] || { echo "[error] $TR_FILE missing"; exit 1; }
[ -f "$PFETCH"   ] || { echo "[error] $PFETCH missing"; exit 1; }

run_with_heartbeat () {
  local logf=$1; shift
  "$@" > "$logf" 2>&1 &
  local pid=$!; local s=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60; s=$((s+60))                              # 1-min heartbeats for long runs
    local last=$(tail -1 "$logf" 2>/dev/null | tr -d '\n' | head -c 70)
    printf "  ...running (%dm %ds)  last: %s\n" $((s/60)) $((s%60)) "$last"
  done
  wait "$pid"
}

parse_ipc () {
  grep -E "cumulative IPC" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1
}

echo "============================================"
echo "[run] BASELINE  trace=$TRACE  warmup=$WARMUP sim=$SIM  (~20 min)"
echo "============================================"
BASE_LOG="$OUT_DIR/nn_demo.baseline.${TRACE}.log"
run_with_heartbeat "$BASE_LOG" \
  "$BASE_BIN" --warmup-instructions "$WARMUP" --simulation-instructions "$SIM" "$TR_FILE"
BASE_IPC=$(parse_ipc "$BASE_LOG")
echo "[parse] baseline IPC = '${BASE_IPC:-<empty>}'"

echo
echo "============================================"
echo "[run] NN REPLAY  trace=$TRACE  list=$PFETCH  model=$MODEL_TAG"
echo "============================================"
REPL_LOG="$OUT_DIR/nn_demo.replay.${TRACE}.${MODEL_TAG}.log"
export PFETCH_LIST_PATH="$PFETCH"
run_with_heartbeat "$REPL_LOG" \
  "$REPL_BIN" --warmup-instructions "$WARMUP" --simulation-instructions "$SIM" "$TR_FILE"
NN_IPC=$(parse_ipc "$REPL_LOG")
ISSUED=$(grep "issued.*prefetches" "$REPL_LOG" | tail -1 | grep -oE "issued [0-9]+ prefetches over [0-9]+" || echo "n/a")
echo "[parse] NN IPC = '${NN_IPC:-<empty>}'"
echo "[parse] $ISSUED"

BASE_IPC="${BASE_IPC:-NA}"; NN_IPC="${NN_IPC:-NA}"
SPEEDUP="NA"
if [[ "$BASE_IPC" =~ ^[0-9.]+$ ]] && [[ "$NN_IPC" =~ ^[0-9.]+$ ]]; then
  SPEEDUP=$(python3 -c "print(f'{float(\"$NN_IPC\")/float(\"$BASE_IPC\"):.4f}')")
fi

echo
echo "============================================"
echo "RESULTS  trace=$TRACE  model=$MODEL_TAG"
echo "  baseline IPC : $BASE_IPC"
echo "  NN IPC       : $NN_IPC"
echo "  speedup      : ${SPEEDUP}x"
echo "  $ISSUED"
echo "============================================"

SUMMARY="$OUT_DIR/nn_demo_summary.csv"
if [ ! -f "$SUMMARY" ] || ! grep -q "^trace,baseline_IPC" "$SUMMARY"; then
  echo "trace,baseline_IPC,nn_IPC,speedup,model,prefetches_issued,accesses,warmup,sim" > "$SUMMARY"
fi
ISSUED_N=$(echo "$ISSUED" | grep -oE "[0-9]+" | head -1)
ACCESS_N=$(echo "$ISSUED" | grep -oE "[0-9]+" | head -2 | tail -1)
echo "$TRACE,$BASE_IPC,$NN_IPC,$SPEEDUP,$MODEL_TAG,${ISSUED_N:-NA},${ACCESS_N:-NA},$WARMUP,$SIM" >> "$SUMMARY"
echo "[done] appended to $SUMMARY"