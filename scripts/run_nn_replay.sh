#!/usr/bin/env bash
# run_nn_replay.sh
# Step 3: take prefetch_list.txt produced by Colab and run it through
# ChampSim using the list_replayer prefetcher. Compare IPC against baseline.
#
# Inputs:
#   - prefetch_list.txt (default: $WORKDIR/prefetch_list.txt)
#   - TRACE env var (default: 605.mcf_s-994B)
#
# Output: results/nn_demo_summary.csv with baseline / NN IPC and speedup.

set -uo pipefail

WORKDIR="$(pwd)"
CHAMP_DIR="$WORKDIR/ChampSim"
TRACE_DIR="$WORKDIR/traces"
OUT_DIR="$WORKDIR/results"
mkdir -p "$OUT_DIR"

TRACE="${TRACE:-605.mcf_s-994B}"
WARMUP="${WARMUP:-1000000}"
SIM="${SIM:-5000000}"
PFETCH="${PFETCH:-$WORKDIR/prefetch_list.txt}"

TR_FILE="$TRACE_DIR/${TRACE}.champsimtrace.xz"
BASE_BIN="$CHAMP_DIR/bin/champsim.baseline"
REPL_BIN="$CHAMP_DIR/bin/champsim.replayer"

[ -x "$BASE_BIN" ] || { echo "[error] $BASE_BIN missing. Run install_and_build.sh."; exit 1; }
[ -x "$REPL_BIN" ] || { echo "[error] $REPL_BIN missing. Run install_and_build.sh."; exit 1; }
[ -f "$TR_FILE" ]  || { echo "[error] trace $TR_FILE not found."; exit 1; }
[ -f "$PFETCH" ]   || { echo "[error] prefetch list $PFETCH not found."; exit 1; }

run_with_heartbeat () {
  local logf=$1; shift
  "$@" > "$logf" 2>&1 &
  local pid=$!; local s=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30; s=$((s+30))
    echo "  ...running (${s}s)  last: $(tail -1 "$logf" 2>/dev/null | head -c 60)"
  done
  wait "$pid"
}

# ---- baseline ----
echo "============================================"
echo "[run] BASELINE (no prefetch)  trace=$TRACE"
echo "============================================"
BASE_LOG="$OUT_DIR/nn_demo.baseline.${TRACE}.log"
run_with_heartbeat "$BASE_LOG" \
  "$BASE_BIN" --warmup-instructions "$WARMUP" --simulation-instructions "$SIM" "$TR_FILE"
BASE_IPC=$(grep -E "cumulative IPC" "$BASE_LOG" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)

# ---- NN replay ----
echo
echo "============================================"
echo "[run] NN PREFETCH REPLAY  trace=$TRACE  list=$PFETCH"
echo "============================================"
REPL_LOG="$OUT_DIR/nn_demo.replay.${TRACE}.log"
export PFETCH_LIST_PATH="$PFETCH"
run_with_heartbeat "$REPL_LOG" \
  "$REPL_BIN" --warmup-instructions "$WARMUP" --simulation-instructions "$SIM" "$TR_FILE"
NN_IPC=$(grep -E "cumulative IPC" "$REPL_LOG" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
ISSUED=$(grep "issued.*prefetches" "$REPL_LOG" | tail -1)

# ---- summarize ----
BASE_IPC="${BASE_IPC:-NA}"; NN_IPC="${NN_IPC:-NA}"
SPEEDUP="NA"
if [ "$BASE_IPC" != "NA" ] && [ "$NN_IPC" != "NA" ]; then
  SPEEDUP=$(python3 -c "print(f'{float(\"$NN_IPC\")/float(\"$BASE_IPC\"):.3f}')")
fi

echo
echo "============================================"
echo "RESULTS  trace=$TRACE"
echo "  baseline IPC : $BASE_IPC"
echo "  NN-replay IPC: $NN_IPC"
echo "  speedup      : ${SPEEDUP}x"
echo "  replayer stats: $ISSUED"
echo "============================================"

SUMMARY="$OUT_DIR/nn_demo_summary.csv"
[ -f "$SUMMARY" ] || echo "trace,baseline_IPC,nn_IPC,speedup,model" > "$SUMMARY"
MODEL_TAG="${MODEL_TAG:-unknown}"
echo "$TRACE,$BASE_IPC,$NN_IPC,$SPEEDUP,$MODEL_TAG" >> "$SUMMARY"
echo "[done] $SUMMARY"
