#!/usr/bin/env bash
# smoke_one.sh
# Run ONE config on ONE trace with live heartbeat, just to confirm everything
# works and to give you a feel for how long each run takes BEFORE you commit
# to the full 60-minute upper_bound sweep.

set -uo pipefail

WORKDIR="$(pwd)"
CHAMP="$WORKDIR/external/ChampSim/bin/champsim"
TRACE="${TRACE:-619.lbm_s-4268B}"
TR_FILE="$WORKDIR/traces/${TRACE}.champsimtrace.xz"
WARMUP="${WARMUP:-1000000}"
SIM="${SIM:-5000000}"

if [ ! -x "$CHAMP" ]; then
  echo "[error] $CHAMP not found"; exit 1
fi
if [ ! -f "$TR_FILE" ]; then
  echo "[error] $TR_FILE not found"; exit 1
fi

mkdir -p "$WORKDIR/results/logs"
LOG="$WORKDIR/results/logs/smoke_one.${TRACE}.log"

echo "============================================"
echo "Running: champsim baseline LRU + no prefetch"
echo "Trace  : $TRACE"
echo "Warmup : $WARMUP instructions"
echo "Sim    : $SIM   instructions"
echo "Log    : $LOG"
echo "============================================"
echo "Starting now... will print heartbeat every 30 seconds."
echo "Expected total time for $TRACE at SIM=$SIM: ~1-3 minutes."
echo "If you see NO heartbeat for 5 minutes, something is wrong."
echo ""

T0=$(date +%s)
"$CHAMP" \
  --warmup-instructions "$WARMUP" \
  --simulation-instructions "$SIM" \
  "$TR_FILE" > "$LOG" 2>&1 &
pid=$!

seconds=0
while kill -0 "$pid" 2>/dev/null; do
  sleep 30
  seconds=$((seconds + 30))
  echo "[heartbeat] ${seconds}s elapsed, ChampSim PID $pid still running"
  echo "[heartbeat] last log line:"
  tail -1 "$LOG" 2>/dev/null | sed 's/^/    /'
done
wait "$pid"
RC=$?
TOTAL=$(( $(date +%s) - T0 ))

echo ""
echo "============================================"
echo "Process exited with code $RC after ${TOTAL}s"
echo "============================================"
echo ""
echo "Last 20 lines of $LOG:"
tail -20 "$LOG"
echo ""
IPC=$(grep -E "cumulative IPC" "$LOG" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
echo "Parsed IPC: ${IPC:-PARSE_FAILED}"
echo ""
echo "If IPC parsed correctly, your full run_baseline.sh and run_upper_bound.sh"
echo "will also work. Multiply the time above by 5 traces for baseline, or by 20"
echo "for upper_bound."
