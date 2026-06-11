#!/usr/bin/env bash
# One-command post-Colab replay helper for one trace and one threshold.
#
# Assumption:
#   Colab output has already been copied to:
#     formal_NN_training/artifacts/packed/<UPLOAD_TAG>/full_lstm_cache_actions.csv.gz.part_*
#   and lstm_events_<TRACE>.csv already exists on the cluster.
#
# This script:
#   1. restores the latest packed Colab output,
#   2. prepares trace-specific actions with replay_access_idx,
#   3. runs no-prefetch / SPP / LSTM replay,
#   4. parses SPP-vs-LSTM replay metrics.

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

TRACE="${TRACE:-619.lbm_s-4268B}"
SHORT_TAG="${TRACE#*.}"
SHORT_TAG="${SHORT_TAG%%.*}"
WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"
POLICY="${POLICY:-threshold}"
PREFETCH_THRESHOLD="${PREFETCH_THRESHOLD:-0.20}"
BYPASS_THRESHOLD="${BYPASS_THRESHOLD:-1.00}"
REPL_BIN="${REPL_BIN:-$ROOT/external/ChampSim/bin/champsim.l2_replayer}"
MODEL_TAG="${MODEL_TAG:-LSTM_${SHORT_TAG}_L2_replayidx_hex_th${PREFETCH_THRESHOLD}_bp${BYPASS_THRESHOLD}}"

INCLUDE_LSTM="${INCLUDE_LSTM:-replayidx}"
EXCLUDE_LSTM="${EXCLUDE_LSTM:-aligned_hex}"
COMPARE_OUT="${COMPARE_OUT:-$ROOT/formal_NN_training/results/replay_compare/accuracy_compare_${TRACE}.csv}"

printf '%s\n' '============================================================'
printf 'POST-COLAB LSTM TRACE REPLAY\n'
printf 'trace      : %s\n' "$TRACE"
printf 'warmup/sim : %s / %s\n' "$WARMUP" "$SIM"
printf 'thresholds : prefetch=%s bypass=%s\n' "$PREFETCH_THRESHOLD" "$BYPASS_THRESHOLD"
printf 'model tag  : %s\n' "$MODEL_TAG"
printf '%s\n' '============================================================'

python3 formal_NN_training/scripts/07_prepare_actions_for_replay.py \
  --trace "$TRACE" \
  --restore-packed \
  --copy-default

TRACE="$TRACE" \
WARMUP="$WARMUP" \
SIM="$SIM" \
POLICY="$POLICY" \
PREFETCH_THRESHOLD="$PREFETCH_THRESHOLD" \
BYPASS_THRESHOLD="$BYPASS_THRESHOLD" \
REPL_BIN="$REPL_BIN" \
MODEL_TAG="$MODEL_TAG" \
  bash formal_NN_training/scripts/03_run_lstm_replay.sh

python3 formal_NN_training/scripts/09_compare_spp_lstm_accuracy.py \
  --trace "$TRACE" \
  --include-lstm "$INCLUDE_LSTM" \
  --exclude-lstm "$EXCLUDE_LSTM" \
  --out "$COMPARE_OUT"

echo "[done] compare CSV: $COMPARE_OUT"
