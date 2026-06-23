#!/usr/bin/env bash
# Collect raw no-prefetch L2 demand streams for the standalone NN dataset.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
TRACES="${TRACES:-602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B}"
WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"
MAX_JOBS="${MAX_JOBS:-2}"
BUILD="${BUILD:-1}"
RESET_PATCH="${RESET_PATCH:-0}"
FORCE="${FORCE:-0}"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
TRACE_DIR="${TRACE_DIR:-$ROOT/traces}"
OUT_ROOT="${OUT_ROOT:-$ROOT/formal_NN_training/results/standalone_nn_data/demand_events}"
EVENT_DIR="$OUT_ROOT/events"
LOG_DIR="$OUT_ROOT/logs"
BIN="${BIN:-$CHAMP_DIR/bin/perceptron-no-multi-no-ship-1core}"
PATCH="$ROOT/formal_NN_training/scripts/02_patch_pythia_demand_logger.sh"

mkdir -p "$EVENT_DIR" "$LOG_DIR"
[[ -d "$CHAMP_DIR" ]] || { echo "[error] missing $CHAMP_DIR" >&2; exit 2; }

build_if_needed() {
  if [[ "$BUILD" != "1" && -x "$BIN" ]]; then
    echo "[build skip] using $BIN"
    return
  fi
  RESET_PATCH="$RESET_PATCH" CHAMP_DIR="$CHAMP_DIR" bash "$PATCH"
  if [[ ! -f "$CHAMP_DIR/libbf/build/lib/libbf.a" ]]; then
    echo "[error] missing libbf static library" >&2
    exit 2
  fi
  ( cd "$CHAMP_DIR" && bash ./build_champsim.sh no multi no 1 )
  [[ -x "$BIN" ]] || { echo "[error] expected binary missing: $BIN" >&2; exit 2; }
}

run_one() {
  local trace="$1"
  local trace_file="$TRACE_DIR/${trace}.champsimtrace.xz"
  local raw="$EVENT_DIR/${trace}.no_pref.events.csv"
  local out="$raw.gz"
  local log="$LOG_DIR/${trace}.no_pref.log"
  [[ -s "$trace_file" ]] || { echo "[error] missing trace: $trace_file" >&2; return 1; }
  if [[ "$FORCE" != "1" && -s "$out" && -s "$log" ]]; then
    echo "[skip] $trace"
    return 0
  fi
  echo "[run] $trace"
  DEMAND_EVENT_LOG="$raw" "$BIN" \
    --l2c_prefetcher_types=none \
    --warmup_instructions="$WARMUP" \
    --simulation_instructions="$SIM" \
    -traces "$trace_file" > "$log" 2>&1
  [[ -s "$raw" ]] || { echo "[error] logger wrote no event file: $raw" >&2; return 1; }
  gzip -f "$raw"
  grep -q '^Core_0_IPC ' "$log" || { echo "[error] missing final IPC: $log" >&2; return 1; }
}

build_if_needed
running=0
status=0
for trace in $TRACES; do
  run_one "$trace" &
  running=$((running+1))
  if (( running >= MAX_JOBS )); then
    wait -n || status=1
    running=$((running-1))
  fi
done
while (( running > 0 )); do
  wait -n || status=1
  running=$((running-1))
done
(( status == 0 )) || exit "$status"

cat > "$OUT_ROOT/RUN_INFO.txt" <<EOF
RUN_KIND=standalone_no_pref_demand_collection
TRACES=$TRACES
WARMUP=$WARMUP
SIM=$SIM
BIN=$BIN
EOF

echo "[done] event files: $EVENT_DIR"
