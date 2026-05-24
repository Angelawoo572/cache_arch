#!/usr/bin/env bash
# run_gru_sweep.sh -- v11
# Loops over the 4 GRU variants from gru_sweep_cross_trace.ipynb and
# replays each through ChampSim on the TEST trace.
#
# Usage:  TRACE=620.omnetpp_s-874B bash scripts/run_gru_sweep.sh
# (TRACE must be the *test* trace, the same one the Colab notebook used as TEST_CSV.)

set -uo pipefail

WORKDIR="$(pwd)"
TRACE="${TRACE:-620.omnetpp_s-874B}"
WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"

VARIANTS=(V1 V2 V3 V4)

miss=0
for v in "${VARIANTS[@]}"; do
  pf="$WORKDIR/prefetch_list_GRU_${v}.txt"
  [ -f "$pf" ] || { echo "[error] missing $pf"; miss=$((miss+1)); }
done
[ "$miss" -gt 0 ] && { echo "[error] $miss prefetch lists missing -- run the Colab GRU notebook first"; exit 1; }

for v in "${VARIANTS[@]}"; do
  echo
  echo "########################################################"
  echo "# GRU variant $v  on  $TRACE  (warmup=$WARMUP, sim=$SIM)"
  echo "########################################################"
  TRACE="$TRACE" \
  PFETCH="$WORKDIR/prefetch_list_GRU_${v}.txt" \
  MODEL_TAG="GRU_${v}" \
  WARMUP="$WARMUP" \
  SIM="$SIM" \
  bash scripts/run_nn_replay.sh
done

SUMMARY="$WORKDIR/results/nn_demo_summary.csv"
echo
echo "============================================"
echo "[ALL DONE] cross-trace GRU sweep on $TRACE complete."
echo "[csv] $SUMMARY  (filtered to GRU_V*):"
grep -E "^(trace,|.*,GRU_V)" "$SUMMARY"
echo
echo "Compare each row against the previous GRU_V variant to attribute"
echo "the IPC change to the single feature added in that step."
echo "============================================"
