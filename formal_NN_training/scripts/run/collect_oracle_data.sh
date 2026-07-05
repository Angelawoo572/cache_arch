#!/usr/bin/env bash
# Collect raw no-prefetch L2C demand events for standalone model training.
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
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
PATCH="$ROOT/formal_NN_training/scripts/build/patch_demand_logger.sh"

mkdir -p "$EVENT_DIR" "$LOG_DIR"
[[ -d "$CHAMP_DIR" ]] || { echo "[error] missing $CHAMP_DIR" >&2; exit 2; }
[[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "[error] MAX_JOBS must be positive" >&2; exit 2; }

build_if_needed() {
  if [[ "$BUILD" != 1 && -x "$BIN" ]]; then echo "[build skip] using $BIN"; return; fi
  RESET_PATCH="$RESET_PATCH" CHAMP_DIR="$CHAMP_DIR" bash "$PATCH"
  [[ -f "$CHAMP_DIR/libbf/build/lib/libbf.a" ]] || { echo "[error] missing libbf static library" >&2; exit 2; }
  ( cd "$CHAMP_DIR" && bash ./build_champsim.sh no multi no 1 )
  [[ -x "$BIN" ]] || { echo "[error] expected binary missing: $BIN" >&2; exit 2; }
}

complete() { [[ -s "$1" ]] && gzip -t "$1" >/dev/null 2>&1 && grep -q '^Core_0_IPC ' "$2"; }

run_one() {
  local trace="$1" trace_file="$TRACE_DIR/${1}.champsimtrace.xz"
  local raw="$EVENT_DIR/${trace}.no_pref.events.csv" out="$EVENT_DIR/${trace}.no_pref.events.csv.gz" log="$LOG_DIR/${trace}.no_pref.log"
  [[ -s "$trace_file" ]] || { echo "[error] missing trace: $trace_file" >&2; return 1; }
  if [[ "$FORCE" != 1 ]] && complete "$out" "$log"; then echo "[skip] $trace"; return; fi
  rm -f "$raw" "$out"
  echo "[collect] $trace"
  DEMAND_EVENT_LOG="$raw" "$BIN" --l2c_prefetcher_types=none --warmup_instructions="$WARMUP" --simulation_instructions="$SIM" -traces "$trace_file" > "$log" 2>&1
  [[ -s "$raw" ]] && grep -q '^Core_0_IPC ' "$log" || { echo "[error] failed $trace" >&2; return 1; }
  gzip -f "$raw"
}

build_if_needed
running=0; status=0
for trace in $TRACES; do
  run_one "$trace" & running=$((running + 1))
  if (( running >= MAX_JOBS )); then wait -n || status=1; running=$((running - 1)); fi
done
while (( running > 0 )); do wait -n || status=1; running=$((running - 1)); done
(( status == 0 )) || exit "$status"
printf 'RUN_KIND=standalone_no_pref_demand_collection\nTRACES=%s\nWARMUP=%s\nSIM=%s\nBIN=%s\n' "$TRACES" "$WARMUP" "$SIM" "$BIN" > "$OUT_ROOT/RUN_INFO.txt"
echo "[done] event files: $EVENT_DIR"
