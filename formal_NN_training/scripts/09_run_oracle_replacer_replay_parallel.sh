#!/usr/bin/env bash
set -u

cd ~/cache

MAX_JOBS=${MAX_JOBS:-3}
WARMUP=${WARMUP:-25000000}
SIM=${SIM:-25000000}

# TODO: 如果你的 binary 名字不同，先 ls external/ChampSim/bin 看一下，然后改这里
BIN=${BIN:-external/ChampSim/bin/champsim.l2_replayer}

OUT_DIR=formal_NN_training/results/oracle_replacer_replay
LOG_DIR=${OUT_DIR}/logs
mkdir -p "$LOG_DIR"

TRACES=(
  "602.gcc_s-734B"
  "619.lbm_s-4268B"
  "605.mcf_s-994B"
  "620.omnetpp_s-874B"
  "623.xalancbmk_s-700B"
)

run_one () {
  TRACE="$1"

  TRACE_FILE="traces/${TRACE}.champsimtrace.xz"
  PLIST="formal_NN_training/artifacts/oracle_replacer/prefetch_list_${TRACE}_cl128_fair_dedup_lru2048.csv"
  LOG="${LOG_DIR}/${TRACE}.oracle_replacer.log"

  echo "============================================================"
  echo "[run] $TRACE"
  echo "[trace] $TRACE_FILE"
  echo "[plist] $PLIST"
  echo "[log] $LOG"
  echo "============================================================"

  if [[ ! -f "$TRACE_FILE" ]]; then
    echo "[ERROR] missing trace: $TRACE_FILE" | tee "$LOG"
    return 1
  fi

  if [[ ! -f "$PLIST" ]]; then
    echo "[ERROR] missing prefetch list: $PLIST" | tee "$LOG"
    return 1
  fi

  PFETCH_LIST_PATH="$PLIST" \
  "$BIN" \
    --warmup-instructions "$WARMUP" \
    --simulation-instructions "$SIM" \
    "$TRACE_FILE" \
    > "$LOG" 2>&1

  echo "[done] $TRACE"
}

running=0

for TRACE in "${TRACES[@]}"; do
  run_one "$TRACE" &
  running=$((running + 1))

  if [[ "$running" -ge "$MAX_JOBS" ]]; then
    wait -n
    running=$((running - 1))
  fi
done

wait
echo "[all done]"
