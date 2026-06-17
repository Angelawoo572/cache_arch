#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

TRACES="${TRACES:-619.lbm_s-4268B 605.mcf_s-994B 602.gcc_s-734B 620.omnetpp_s-874B}"
WARMUP="${WARMUP:-5000000}"
SIM="${SIM:-5000000}"
BUILD="${BUILD:-0}"
PATCH_SPP="${PATCH_SPP:-0}"
RESET_SPP="${RESET_SPP:-0}"

SUMMARY="$ROOT/formal_NN_training/results/trace_scout/trace_scout_summary.csv"
rm -f "$SUMMARY"
mkdir -p "$(dirname "$SUMMARY")"

echo "============================================================"
echo "TRACE SCOUT"
echo "traces     : $TRACES"
echo "warmup/sim : $WARMUP / $SIM"
echo "build      : $BUILD"
echo "patch      : $PATCH_SPP"
echo "summary    : $SUMMARY"
echo "============================================================"

for T in $TRACES; do
  TR_FILE="$ROOT/traces/${T}.champsimtrace.xz"
  if [ ! -f "$TR_FILE" ]; then
    echo "[skip] missing trace file: $TR_FILE"
    continue
  fi

  echo
  echo "============================================================"
  echo "[trace] $T"
  echo "============================================================"

  OUT="$ROOT/formal_NN_training/data/generated/lstm_events_${T}.csv"

  if [ ! -s "$OUT" ]; then
    TRACE="$T" \
    WARMUP="$WARMUP" \
    SIM="$SIM" \
    RESET_SPP="$RESET_SPP" \
    BUILD="$BUILD" \
    PATCH_SPP="$PATCH_SPP" \
    bash formal_NN_training/LSTM/scripts/01_run_spp_trace_dump.sh
  else
    echo "[reuse] existing $OUT"
  fi

  python3 formal_NN_training/LSTM/scripts/profile_lstm_events_no_pandas.py \
    --csv "$OUT" \
    --trace "$T" \
    --out "$SUMMARY" \
    --append
done

echo
echo "============================================================"
echo "SCOUT SUMMARY"
echo "============================================================"
column -t -s, "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
