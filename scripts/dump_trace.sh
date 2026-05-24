#!/usr/bin/env bash
# dump_trace.sh -- v11 (real run, not smoke)
#
# Dumps a CSV of (idx, addr, pc, hit) using the trace_dumper module on a
# given trace. Two are needed for the cross-trace GRU sweep:
#   bash scripts/dump_trace.sh 605.mcf_s-994B           # train trace
#   bash scripts/dump_trace.sh 620.omnetpp_s-874B       # test  trace
#
# Defaults are the REAL evaluation regime (25M warmup + 25M sim, paper-grade
# short-mix configuration). This takes ~20 min per trace on the ECE box.
# For DPC-3 ranked submission scale (200M+500M, ~4.5h per run), set
# WARMUP=200000000 SIM=500000000.

set -uo pipefail

TRACE="${1:?usage: dump_trace.sh <trace_name_without_extension>}"
WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"

WORKDIR="$(pwd)"
CHAMP_BIN="$WORKDIR/external/ChampSim/bin/champsim.dumper"
TRACE_DIR="$WORKDIR/traces"
OUT_DIR="$WORKDIR/results"
mkdir -p "$OUT_DIR"

TR_FILE="$TRACE_DIR/${TRACE}.champsimtrace.xz"
DUMP_FILE="$OUT_DIR/access_trace.${TRACE}.csv"
RUN_LOG="$OUT_DIR/dumper.${TRACE}.log"

[ -x "$CHAMP_BIN" ] || { echo "[error] $CHAMP_BIN missing -- run install_and_build.sh"; exit 1; }
[ -f "$TR_FILE" ]   || { echo "[error] trace $TR_FILE missing"; exit 1; }

echo "============================================"
echo "[dump] trace : $TRACE"
echo "[dump] warmup: $WARMUP instructions"
echo "[dump] sim   : $SIM instructions"
echo "[dump] out   : $DUMP_FILE"
echo "[dump] log   : $RUN_LOG"
echo "[dump] expected wall clock: ~20 min for 25M+25M"
echo "============================================"

export TRACE_DUMP_PATH="$DUMP_FILE"
"$CHAMP_BIN" \
  --warmup-instructions "$WARMUP" \
  --simulation-instructions "$SIM" \
  "$TR_FILE" > "$RUN_LOG" 2>&1 &
pid=$!
secs=0
while kill -0 "$pid" 2>/dev/null; do
  sleep 30; secs=$((secs+30))
  size=$(stat -c%s "$DUMP_FILE" 2>/dev/null || echo 0)
  rows=$(wc -l < "$DUMP_FILE" 2>/dev/null || echo 0)
  echo "  ...still running (${secs}s)  csv: ${size} bytes, ${rows} rows"
done
wait "$pid"; rc=$?
echo "[dump] ChampSim exit code: $rc"

if [ -f "$DUMP_FILE" ]; then
  LINES=$(wc -l < "$DUMP_FILE")
  echo "[dump] final csv rows: $LINES"
  echo "[dump] head:"; head -3 "$DUMP_FILE"
  echo
  echo "[next] upload $DUMP_FILE to Colab for gru_sweep_cross_trace.ipynb"
else
  echo "[error] CSV not produced. See $RUN_LOG"; tail -20 "$RUN_LOG"
fi