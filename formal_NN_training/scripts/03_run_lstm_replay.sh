#!/usr/bin/env bash
# 03_run_lstm_replay.sh
#
# Convert the LSTM action table exported by the notebook into a list_replayer
# prefetch list, then run the existing ChampSim replay flow.
#
# Usage from repo root:
#   TRACE=602.gcc_s-734B bash formal_NN_training/scripts/03_run_lstm_replay.sh
#
# Common overrides:
#   ACTIONS=formal_NN_training/artifacts/full_lstm_cache_actions.csv \
#   PREFETCH_THRESHOLD=0.55 BYPASS_THRESHOLD=0.70 \
#   WARMUP=25000000 SIM=25000000 TRACE=602.gcc_s-734B \
#   bash formal_NN_training/scripts/03_run_lstm_replay.sh

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

TRACE="${TRACE:-602.gcc_s-734B}"
SHORT_TAG="${TRACE#*.}"
WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"
PREFETCH_THRESHOLD="${PREFETCH_THRESHOLD:-0.50}"
BYPASS_THRESHOLD="${BYPASS_THRESHOLD:-0.60}"
MODEL_TAG="${MODEL_TAG:-LSTM_CACHE_ACTION_${SHORT_TAG}_th${PREFETCH_THRESHOLD}_bp${BYPASS_THRESHOLD}}"

ACTIONS_DEFAULT="$ROOT/formal_NN_training/artifacts/full_lstm_cache_actions.csv"
ACTIONS_FALLBACK="$ROOT/formal_NN_training/artifacts/val_lstm_cache_actions.csv"
ACTIONS="${ACTIONS:-$ACTIONS_DEFAULT}"
if [ ! -f "$ACTIONS" ] && [ -f "$ACTIONS_FALLBACK" ]; then
  ACTIONS="$ACTIONS_FALLBACK"
fi

OUT_DIR="$ROOT/results/generated/prefetch_lists"
mkdir -p "$OUT_DIR"
PFETCH="${PFETCH:-$OUT_DIR/prefetch_list_${MODEL_TAG}.txt}"

if [ ! -f "$ACTIONS" ]; then
  echo "[error] action table missing: $ACTIONS"
  echo "        Run formal_NN_training/LSTM_cache_action_predictor.ipynb first."
  exit 1
fi

python3 formal_NN_training/scripts/02_actions_to_prefetch_list.py \
  --actions "$ACTIONS" \
  --out "$PFETCH" \
  --prefetch-threshold "$PREFETCH_THRESHOLD" \
  --bypass-threshold "$BYPASS_THRESHOLD"

if [ ! -s "$PFETCH" ]; then
  echo "[error] generated empty prefetch list: $PFETCH"
  echo "        Try lower PREFETCH_THRESHOLD or higher BYPASS_THRESHOLD."
  exit 1
fi

echo "============================================================"
echo "LSTM CACHE-ACTION REPLAY"
echo "trace      : $TRACE"
echo "actions    : $ACTIONS"
echo "pfetch     : $PFETCH"
echo "model tag  : $MODEL_TAG"
echo "warmup/sim : $WARMUP / $SIM"
echo "============================================================"

TRACE="$TRACE" \
PFETCH="$PFETCH" \
MODEL_TAG="$MODEL_TAG" \
WARMUP="$WARMUP" \
SIM="$SIM" \
bash projects/legacy_gru_prefetch/scripts/run_nn_replay.sh

echo
echo "[done] replay summary: results/nn_demo_summary.csv"
grep -E "^(trace,|.*${MODEL_TAG})" results/nn_demo_summary.csv || tail -5 results/nn_demo_summary.csv
