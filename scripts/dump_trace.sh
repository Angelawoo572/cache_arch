#!/usr/bin/env bash
# dump_trace.sh
# Step 1: run ChampSim with the trace_dumper prefetcher to produce a CSV
# of (idx, addr, pc, hit) tuples that the Colab notebook will train on.
#
# Usage:  bash scripts/dump_trace.sh                    # default: mcf
#         TRACE=619.lbm_s-4268B bash scripts/dump_trace.sh

set -uo pipefail

WORKDIR="$(pwd)"
CHAMP_BIN="$WORKDIR/external/ChampSim/bin/champsim.dumper"
TRACE_DIR="$WORKDIR/traces"
OUT_DIR="$WORKDIR/results"
mkdir -p "$OUT_DIR"

TRACE="${TRACE:-605.mcf_s-994B}"
WARMUP="${WARMUP:-1000000}"
SIM="${SIM:-5000000}"

TR_FILE="$TRACE_DIR/${TRACE}.champsimtrace.xz"
DUMP_FILE="$OUT_DIR/access_trace.${TRACE}.csv"
RUN_LOG="$OUT_DIR/dumper.${TRACE}.log"

if [ ! -x "$CHAMP_BIN" ]; then
  echo "[error] $CHAMP_BIN not found. Run install_and_build.sh first."; exit 1
fi
if [ ! -f "$TR_FILE" ]; then
  echo "[error] $TR_FILE not found."; exit 1
fi

echo "[dump] trace: $TRACE"
echo "[dump] warmup=$WARMUP sim=$SIM"
echo "[dump] output csv: $DUMP_FILE"
echo "[dump] run log:    $RUN_LOG"

export TRACE_DUMP_PATH="$DUMP_FILE"

"$CHAMP_BIN" \
  --warmup-instructions "$WARMUP" \
  --simulation-instructions "$SIM" \
  "$TR_FILE" > "$RUN_LOG" 2>&1 &
pid=$!
seconds=0
while kill -0 "$pid" 2>/dev/null; do
  sleep 30
  seconds=$((seconds + 30))
  echo "  ...still running (${seconds}s)  csv size: $(stat -c%s "$DUMP_FILE" 2>/dev/null || echo 0) bytes"
done
wait "$pid"; rc=$?

echo "[dump] ChampSim exit=$rc"
echo "[dump] last 6 lines of run log:"
tail -6 "$RUN_LOG"

if [ -f "$DUMP_FILE" ]; then
  LINES=$(wc -l < "$DUMP_FILE")
  echo "[dump] csv lines: $LINES"
  echo "[dump] csv head:"
  head -5 "$DUMP_FILE"
  echo
  echo "[next] upload $DUMP_FILE to Colab and run neural_prefetcher_zoo.ipynb"
else
  echo "[error] CSV not produced. Check $RUN_LOG"
fi
