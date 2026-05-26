#!/usr/bin/env bash
# run_gru_v9_decode_sweep.sh
# Replay GRU V9 decode-policy sweep lists through ChampSim.
#
# This script assumes `scripts/gru_v9_export_decode_sweep.py` was run after
# the V9 notebook training/export stage, and the generated list files were copied
# to:
#   results/generated/prefetch_lists/
#
# Example list names:
#   prefetch_list_GRU_V9_gcc_s-734B_th030_deg1.txt
#   prefetch_list_GRU_V9_gcc_s-734B_th050_deg2.txt
#
# Usage:
#   TRACE=602.gcc_s-734B bash scripts/run_gru_v9_decode_sweep.sh
#
# Optional:
#   THRESHOLDS="030 050 070 090" DEGREES="1 2 4" TRACE=602.gcc_s-734B bash scripts/run_gru_v9_decode_sweep.sh

set -uo pipefail

WORKDIR="$(pwd)"
TRACE="${TRACE:-602.gcc_s-734B}"
SHORT_TAG="${TRACE#*.}"
PFETCH_DIR="${PFETCH_DIR:-$WORKDIR/results/generated/prefetch_lists}"
THRESHOLDS="${THRESHOLDS:-030 050 070 090}"
DEGREES="${DEGREES:-1 2 4}"

# Use the same full-index replay window as the original dump/replay coordinate system.
WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"

SUMMARY="$WORKDIR/results/gru_v9_decode_sweep_ipc.csv"
mkdir -p "$WORKDIR/results"

if [ ! -f "$SUMMARY" ]; then
  echo "trace,threshold,degree,baseline_IPC,nn_IPC,speedup,model,prefetches_issued,accesses,warmup,sim" > "$SUMMARY"
fi

echo "============================================================"
echo "GRU V9 DECODE SWEEP REPLAY"
echo "============================================================"
echo "trace       : $TRACE"
echo "short tag   : $SHORT_TAG"
echo "prefetch dir: $PFETCH_DIR"
echo "thresholds  : $THRESHOLDS"
echo "degrees     : $DEGREES"
echo "warmup/sim  : $WARMUP / $SIM"
echo "============================================================"

for TH in $THRESHOLDS; do
  for DEG in $DEGREES; do
    PFETCH="$PFETCH_DIR/prefetch_list_GRU_V9_${SHORT_TAG}_th${TH}_deg${DEG}.txt"
    MODEL_TAG="GRU_V9_${SHORT_TAG}_th${TH}_deg${DEG}"

    echo
    echo "============================================================"
    echo "[decode] TRACE=$TRACE TH=$TH DEG=$DEG"
    echo "[decode] PFETCH=$PFETCH"
    echo "============================================================"

    if [ ! -f "$PFETCH" ]; then
      echo "[missing] $PFETCH"
      echo "          Generate it from the V9 notebook runtime with:"
      echo "          %run -i scripts/gru_v9_export_decode_sweep.py"
      continue
    fi

    nlines=$(wc -l < "$PFETCH")
    first=$(head -1 "$PFETCH" 2>/dev/null || true)
    last=$(tail -1 "$PFETCH" 2>/dev/null || true)
    echo "[check] lines=$nlines first='$first' last='$last'"

    TRACE="$TRACE" \
    WARMUP="$WARMUP" \
    SIM="$SIM" \
    PFETCH="$PFETCH" \
    MODEL_TAG="$MODEL_TAG" \
    bash scripts/run_nn_replay.sh

    # Copy the last row from the generic summary into the decode-specific summary.
    LAST=$(tail -1 "$WORKDIR/results/nn_demo_summary.csv")
    # LAST fields: trace,baseline_IPC,nn_IPC,speedup,model,prefetches_issued,accesses,warmup,sim
    echo "$LAST" | awk -F, -v th="$TH" -v deg="$DEG" 'BEGIN{OFS=","} {print $1, th, deg, $2, $3, $4, $5, $6, $7, $8, $9}' >> "$SUMMARY"
  done
done

echo
echo "============================================================"
echo "[DONE] decode sweep summary: $SUMMARY"
echo "============================================================"
column -t -s, "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
