#!/usr/bin/env bash
# Run same-binary no-prefetch baselines for oracle-LSTM replay comparison.
#
# Use this with the exact binary produced by 11_install_oracle_l2_replayer.sh.
# That removes simulator/build/config differences from:
#     speedup = IPC(oracle-LSTM replay) / IPC(no-pref baseline)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MAX_JOBS=${MAX_JOBS:-2}
WARMUP=${WARMUP:-25000000}
SIM=${SIM:-25000000}
BIN=${BIN:-"$ROOT/external/ChampSim/bin/champsim.oracle_l2_replayer"}
OUT_DIR=${OUT_DIR:-formal_NN_training/results/oracle_replacer_no_pref_baseline_sig_v2}
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

[[ -x "$BIN" ]] || { echo "[error] binary is not executable: $BIN" >&2; exit 2; }

if [[ -n "${TRACES:-}" ]]; then
  read -r -a TRACE_LIST <<< "$TRACES"
else
  TRACE_LIST=(
    "602.gcc_s-734B"
    "619.lbm_s-4268B"
    "605.mcf_s-994B"
    "620.omnetpp_s-874B"
    "623.xalancbmk_s-700B"
  )
fi

run_one() {
  local trace="$1"
  local trace_file="traces/${trace}.champsimtrace.xz"
  local log="$LOG_DIR/${trace}.no_pref.log"

  [[ -f "$trace_file" ]] || { echo "[error] missing trace: $trace_file" >&2; return 1; }

  echo "[run no_pref] $trace"
  "$BIN" \
    --l2c_prefetcher_types=none \
    --warmup_instructions="$WARMUP" \
    --simulation_instructions="$SIM" \
    -traces "$trace_file" \
    > "$log" 2>&1

  grep -q '^Core_0_IPC ' "$log" || { echo "[error] $trace: missing ROI IPC in $log" >&2; return 1; }
  echo "[done no_pref] $trace"
}

running=0
status=0
for trace in "${TRACE_LIST[@]}"; do
  run_one "$trace" &
  ((running+=1))
  if (( running >= MAX_JOBS )); then
    if ! wait -n; then status=1; fi
    ((running-=1))
  fi
done
while (( running > 0 )); do
  if ! wait -n; then status=1; fi
  ((running-=1))
done

if (( status != 0 )); then
  echo "[failed] one or more no-pref baselines failed; see $LOG_DIR" >&2
  exit "$status"
fi

echo "[all done] same-binary no-pref baselines"
