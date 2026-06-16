#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

CHAMP_DIR="$ROOT/external/ChampSim"
TRACE_DIR="$ROOT/traces"

WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"
MAX_JOBS="${MAX_JOBS:-1}"

OUT_ROOT="$ROOT/formal_NN_training/results/capacity_sweep"
LOG_DIR="$OUT_ROOT/logs"
PFETCH_DIR="$OUT_ROOT/prefetch_lists"
mkdir -p "$LOG_DIR" "$PFETCH_DIR"

TRACES=("619.lbm_s-4268B" "602.gcc_s-734B")
CAPS=("256K" "512K" "1M" "2M")

make_pfetch () {
  local trace="$1"

  echo "============================================================"
  echo "[prepare actions] $trace"
  echo "============================================================"

  python3 formal_NN_training/scripts/07_prepare_actions_for_replay.py \
    --trace "$trace" \
    --restore-packed \
    --copy-default

  local pfetch="$PFETCH_DIR/prefetch_list_${trace}_th0.20_bp1.00.txt"

  python3 formal_NN_training/scripts/02_actions_to_prefetch_list.py \
    --actions formal_NN_training/artifacts/full_lstm_cache_actions.csv \
    --out "$pfetch" \
    --policy threshold \
    --prefetch-threshold 0.20 \
    --bypass-threshold 1.00

  if [ ! -s "$pfetch" ]; then
    echo "[error] empty pfetch list: $pfetch"
    exit 1
  fi
}

run_one () {
  local trace="$1"
  local cap="$2"
  local trfile="$TRACE_DIR/${trace}.champsimtrace.xz"
  local pfetch="$PFETCH_DIR/prefetch_list_${trace}_th0.20_bp1.00.txt"

  local no_bin="$CHAMP_DIR/bin/champsim.baseline.L2_${cap}"
  local spp_bin="$CHAMP_DIR/bin/champsim.spp.L2_${cap}"
  local repl_bin="$CHAMP_DIR/bin/champsim.replayer.L2_${cap}"

  for b in "$no_bin" "$spp_bin" "$repl_bin"; do
    if [ ! -x "$b" ]; then
      echo "[error] missing binary: $b"
      exit 1
    fi
  done

  if [ ! -f "$trfile" ]; then
    echo "[error] missing trace: $trfile"
    exit 1
  fi

  echo "============================================================"
  echo "[capacity replay] trace=$trace cap=$cap"
  echo "============================================================"

  "$no_bin" \
    --warmup-instructions "$WARMUP" \
    --simulation-instructions "$SIM" \
    "$trfile" \
    > "$LOG_DIR/${trace}.L2_${cap}.no_prefetch.log" 2>&1

  "$spp_bin" \
    --warmup-instructions "$WARMUP" \
    --simulation-instructions "$SIM" \
    "$trfile" \
    > "$LOG_DIR/${trace}.L2_${cap}.spp.log" 2>&1

  PFETCH_LIST_PATH="$pfetch" \
  "$repl_bin" \
    --warmup-instructions "$WARMUP" \
    --simulation-instructions "$SIM" \
    "$trfile" \
    > "$LOG_DIR/${trace}.L2_${cap}.LSTM_th0.20_bp1.00.log" 2>&1
}

for trace in "${TRACES[@]}"; do
  make_pfetch "$trace"
done

running=0
for trace in "${TRACES[@]}"; do
  for cap in "${CAPS[@]}"; do
    run_one "$trace" "$cap" &
    running=$((running + 1))

    if [ "$running" -ge "$MAX_JOBS" ]; then
      wait -n
      running=$((running - 1))
    fi
  done
done

wait

echo "[done] logs under $LOG_DIR"
ls -lh "$LOG_DIR"/*.log
