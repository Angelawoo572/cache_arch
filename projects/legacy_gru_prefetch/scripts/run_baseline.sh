#!/usr/bin/env bash
# run_baseline.sh -- v3 with live progress
#
# Each ChampSim run on 5M sim instructions takes 1-3 minutes (slow workloads
# like lbm and mcf are at the slow end). The v2 script printed only the START
# message, then waited silently for the whole thing to finish. That made it
# LOOK frozen even though it was running. This v3 prints a heartbeat every
# 30 seconds while a run is in progress so you can see it's alive.
#
# Total wall-clock budget: 5 traces x ~2 min = ~10 minutes.

set -uo pipefail        # NOTE: removed -e so one slow trace doesn't kill the script

WARMUP="${WARMUP:-1000000}"
SIM="${SIM:-5000000}"

WORKDIR="$(pwd)"
CHAMP="$WORKDIR/external/ChampSim/bin/champsim"
TRACE_DIR="$WORKDIR/traces"
OUT_DIR="$WORKDIR/results"
LOG_DIR="$WORKDIR/results/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

if [ ! -x "$CHAMP" ]; then
  echo "[error] ChampSim binary not found at $CHAMP. Run setup_champsim.sh first."
  exit 1
fi

declare -a TRACES=(
  "619.lbm_s-4268B"
  "605.mcf_s-994B"
  "620.omnetpp_s-874B"
  "602.gcc_s-734B"
  "623.xalancbmk_s-700B"
)

CSV="$OUT_DIR/baseline.csv"
echo "trace,IPC,LLC_MPKI,L2_MPKI,L1D_MPKI" > "$CSV"

N=${#TRACES[@]}
i=0
T0=$(date +%s)

run_with_heartbeat () {
  local cmd_log="$1"
  local trace_name="$2"
  shift 2
  "$@" > "$cmd_log" 2>&1 &
  local pid=$!
  local seconds=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    seconds=$((seconds + 30))
    local last=$(tail -1 "$cmd_log" 2>/dev/null | head -c 200)
    printf "  ...still running %s (elapsed %ds inside ChampSim)  last: %s\n" \
           "$trace_name" "$seconds" "${last:0:80}"
  done
  wait "$pid"
  return $?
}

for t in "${TRACES[@]}"; do
  i=$((i+1))
  TR_FILE="$TRACE_DIR/${t}.champsimtrace.xz"
  if [ ! -f "$TR_FILE" ]; then
    echo "[skip $i/$N] $t -- trace file missing"
    continue
  fi

  LOG="$LOG_DIR/${t}.baseline.log"
  ELAPSED=$(( $(date +%s) - T0 ))
  echo "[run $i/$N] $t  (total elapsed: ${ELAPSED}s)  warmup=$WARMUP sim=$SIM"
  echo "         log: $LOG"

  if ! run_with_heartbeat "$LOG" "$t" \
        "$CHAMP" \
        --warmup-instructions "$WARMUP" \
        --simulation-instructions "$SIM" \
        "$TR_FILE"; then
    echo "  [FAIL] $t -- see $LOG"
    echo "$t,FAIL,FAIL,FAIL,FAIL" >> "$CSV"
    continue
  fi

  IPC=$(grep -E "cumulative IPC" "$LOG" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
  LLC_MPKI=$(grep -E "LLC.*MPKI" "$LOG" | head -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
  if [ -z "$LLC_MPKI" ]; then
    MISS=$(grep -E "cpu0->LLC TOTAL" "$LOG" | head -1 | grep -oE "MISS:[ ]+[0-9]+" | grep -oE "[0-9]+")
    INSTR=$(grep "cumulative IPC" "$LOG" | tail -1 | grep -oE "instructions: [0-9]+" | grep -oE "[0-9]+")
    if [ -n "$MISS" ] && [ -n "$INSTR" ] && [ "$INSTR" -gt 0 ]; then
      LLC_MPKI=$(python3 -c "print(f'{$MISS*1000/$INSTR:.3f}')")
    fi
  fi
  L2_MPKI=$(grep -E "L2C.*MPKI" "$LOG" | head -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
  L1D_MPKI=$(grep -E "L1D.*MPKI" "$LOG" | head -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)

  IPC="${IPC:-NA}"; LLC_MPKI="${LLC_MPKI:-NA}"; L2_MPKI="${L2_MPKI:-NA}"; L1D_MPKI="${L1D_MPKI:-NA}"

  echo "$t,$IPC,$LLC_MPKI,$L2_MPKI,$L1D_MPKI" >> "$CSV"
  echo "  [done $i/$N] IPC=$IPC  LLC_MPKI=$LLC_MPKI"
done

TOTAL=$(( $(date +%s) - T0 ))
echo
echo "[ALL DONE] total wall-clock: ${TOTAL}s"
echo "[csv] $CSV"
column -t -s, "$CSV" 2>/dev/null || cat "$CSV"