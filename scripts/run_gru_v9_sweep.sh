#!/usr/bin/env bash
# run_gru_v9_sweep.sh
# Runs the V9 replay on the standard demo traces.
#
# This script assumes gru_sweep_v9.ipynb has already generated prefetch lists in:
#   results/generated/prefetch_lists/prefetch_list_GRU_V9_<trace_tag>.txt
#
# Example files:
#   results/generated/prefetch_lists/prefetch_list_GRU_V9_mcf_s-994B.txt
#   results/generated/prefetch_lists/prefetch_list_GRU_V9_lbm_s-4268B.txt
#   results/generated/prefetch_lists/prefetch_list_GRU_V9_gcc_s-734B.txt
#
# Usage:
#   bash scripts/run_gru_v9_sweep.sh
#
# Optional:
#   TRACES="605.mcf_s-994B 619.lbm_s-4268B" bash scripts/run_gru_v9_sweep.sh
#   ORIGINAL_WARMUP=100000 ORIGINAL_SIM=100000 bash scripts/run_gru_v9_sweep.sh

set -uo pipefail

WORKDIR="$(pwd)"
PFETCH_DIR="${PFETCH_DIR:-$WORKDIR/results/generated/prefetch_lists}"
TRACES="${TRACES:-605.mcf_s-994B 619.lbm_s-4268B 602.gcc_s-734B}"

mkdir -p "$WORKDIR/results"

echo "============================================================"
echo "V9 GRU SWEEP"
echo "============================================================"
echo "workdir      : $WORKDIR"
echo "prefetch dir : $PFETCH_DIR"
echo "traces       : $TRACES"
echo "============================================================"

missing=0
for TRACE_NAME in $TRACES; do
  SHORT_TAG="${TRACE_NAME#*.}"
  PFETCH="$PFETCH_DIR/prefetch_list_GRU_V9_${SHORT_TAG}.txt"
  if [ ! -f "$PFETCH" ]; then
    echo "[missing] $TRACE_NAME needs $PFETCH"
    missing=1
  else
    echo "[found]   $TRACE_NAME -> $PFETCH ($(wc -l < "$PFETCH") lines)"
  fi
done

if [ "$missing" -ne 0 ]; then
  echo
  echo "[error] Some prefetch lists are missing."
  echo "        Run gru_sweep_v9.ipynb for the missing traces first, or set:"
  echo "        PFETCH_DIR=/path/to/prefetch_lists bash scripts/run_gru_v9_sweep.sh"
  exit 1
fi

echo
for TRACE_NAME in $TRACES; do
  SHORT_TAG="${TRACE_NAME#*.}"
  PFETCH="$PFETCH_DIR/prefetch_list_GRU_V9_${SHORT_TAG}.txt"

  echo
  echo "============================================================"
  echo "[sweep] TRACE=$TRACE_NAME"
  echo "[sweep] PFETCH=$PFETCH"
  echo "============================================================"

  TRACE="$TRACE_NAME" \
  PFETCH="$PFETCH" \
  bash scripts/run_gru_v9.sh

done

echo
echo "============================================================"
echo "[DONE] V9 sweep summary"
echo "============================================================"
if [ -f "$WORKDIR/results/nn_demo_summary.csv" ]; then
  grep -E "^(trace,|.*,GRU_V9)" "$WORKDIR/results/nn_demo_summary.csv" || cat "$WORKDIR/results/nn_demo_summary.csv"
else
  echo "[warn] no results/nn_demo_summary.csv found"
fi
