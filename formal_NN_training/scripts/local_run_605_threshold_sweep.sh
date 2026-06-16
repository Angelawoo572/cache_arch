#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/qianruw/cache"
cd "$ROOT"

TRACE="605.mcf_s-994B"
SHORT="mcf_s-994B"
WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"

ACTIONS="$ROOT/formal_NN_training/artifacts/by_trace/${TRACE}/full_lstm_cache_actions.csv"
TRACE_FILE="$ROOT/traces/${TRACE}.champsimtrace.xz"

LOG_DIR="$ROOT/formal_NN_training/results/replay_compare/logs"
PFETCH_DIR="$ROOT/formal_NN_training/results/replay_compare/prefetch_lists"

NO_BIN="$ROOT/external/ChampSim/bin/champsim.baseline"
SPP_BIN="$ROOT/external/ChampSim/bin/champsim.l2_spp"
if [ ! -x "$SPP_BIN" ]; then
  SPP_BIN="$ROOT/external/ChampSim/bin/champsim.l2_spp_cand"
fi
REPL_BIN="$ROOT/external/ChampSim/bin/champsim.l2_replayer"

mkdir -p "$LOG_DIR" "$PFETCH_DIR"

echo "============================================================"
echo "[check files]"
echo "============================================================"
ls -lh "$ACTIONS"
ls -lh "$TRACE_FILE"
ls -lh "$NO_BIN"
ls -lh "$SPP_BIN"
ls -lh "$REPL_BIN"

echo "============================================================"
echo "[run no-prefetch once if missing]"
echo "============================================================"
if [ ! -s "$LOG_DIR/${TRACE}.no_prefetch.log" ]; then
  "$NO_BIN" \
    --warmup-instructions "$WARMUP" \
    --simulation-instructions "$SIM" \
    "$TRACE_FILE" \
    > "$LOG_DIR/${TRACE}.no_prefetch.log" 2>&1
else
  echo "[skip] existing $LOG_DIR/${TRACE}.no_prefetch.log"
fi

echo "============================================================"
echo "[run SPP once if missing]"
echo "============================================================"
if [ ! -s "$LOG_DIR/${TRACE}.spp.log" ]; then
  "$SPP_BIN" \
    --warmup-instructions "$WARMUP" \
    --simulation-instructions "$SIM" \
    "$TRACE_FILE" \
    > "$LOG_DIR/${TRACE}.spp.log" 2>&1
else
  echo "[skip] existing $LOG_DIR/${TRACE}.spp.log"
fi

echo "============================================================"
echo "[LSTM low-threshold sweep, bypass disabled by 1.01]"
echo "============================================================"

for SPEC in \
  0p00001:0.00001 \
  0p00005:0.00005 \
  0p0001:0.0001 \
  0p0002:0.0002 \
  0p0005:0.0005 \
  0p001:0.001
do
  TAG=${SPEC%%:*}
  TH=${SPEC#*:}

  MODEL_TAG="LSTM_${SHORT}_L2_replayidx_hex_th${TAG}_bp1p01"
  PFETCH="$PFETCH_DIR/prefetch_list_${MODEL_TAG}.txt"
  LOG="$LOG_DIR/${TRACE}.${MODEL_TAG}.log"

  echo "============================================================"
  echo "[make list] TH=$TH TAG=$TAG"
  echo "============================================================"

  python3 formal_NN_training/scripts/02_actions_to_prefetch_list.py \
    --actions "$ACTIONS" \
    --out "$PFETCH" \
    --policy threshold \
    --prefetch-threshold "$TH" \
    --bypass-threshold 1.01

  LINES=$(wc -l < "$PFETCH" || echo 0)
  echo "[list lines] $LINES"

  if [ "$LINES" -eq 0 ]; then
    echo "[skip replay] empty list for TH=$TH"
    continue
  fi

  echo "============================================================"
  echo "[replay] $MODEL_TAG"
  echo "============================================================"

  PFETCH_LIST_PATH="$PFETCH" \
  "$REPL_BIN" \
    --warmup-instructions "$WARMUP" \
    --simulation-instructions "$SIM" \
    "$TRACE_FILE" \
    > "$LOG" 2>&1
done
