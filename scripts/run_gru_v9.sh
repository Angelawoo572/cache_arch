#!/usr/bin/env bash
# run_gru_v9.sh
# Replays the V9 in-distribution prefetch list through ChampSim.
#
# IMPORTANT INDEX-ALIGNMENT RULE
# ------------------------------
# The trace_dumper writes idx starting from 0 at the beginning of ChampSim's
# simulation phase. The list_replayer also starts its internal counter from 0 at
# the beginning of ChampSim's simulation phase.
#
# Therefore, a prefetch list exported from a CSV dumped with:
#     ORIGINAL_WARMUP=25M, ORIGINAL_SIM=25M
# must be replayed with the SAME warmup/sim by default:
#     WARMUP=25M, SIM=25M
#
# Do NOT default to "last 15%" replay by changing warmup to 46.25M and sim to
# 3.75M. That shifts the ChampSim simulation phase, so the replayer counter
# restarts at 0 while the prefetch-list idx values are still from the original
# 25M-sim CSV. The symptom is loaded entries but 0 or very few issued prefetches.
#
# If you want a clean last-15% replay, re-dump that last-15% window and export a
# prefetch list with LOCAL indices for that new CSV, or add an explicit/reliable
# idx-offset mechanism. The current safe default is full-index replay.
#
# Usage:
#   TRACE=605.mcf_s-994B bash scripts/run_gru_v9.sh
#
# Optional override:
#   PFETCH=/path/to/prefetch_list.txt TRACE=605.mcf_s-994B bash scripts/run_gru_v9.sh
#
# To do the full 3-trace sweep:
#   bash scripts/run_gru_v9_sweep.sh

set -uo pipefail

WORKDIR="$(pwd)"
TRACE="${TRACE:-605.mcf_s-994B}"

# Must match the window used by scripts/dump_trace.sh when the CSVs for
# gru_sweep_v9.ipynb were generated.
ORIGINAL_WARMUP="${ORIGINAL_WARMUP:-25000000}"
ORIGINAL_SIM="${ORIGINAL_SIM:-25000000}"

# Default: full-index replay, because existing V9 prefetch lists use the original
# dumper idx coordinate system. Keep an explicit opt-in for the old shifted mode
# only for debugging; it is not a valid default for existing lists.
V9_REPLAY_MODE="${V9_REPLAY_MODE:-full_index}"   # full_index | shifted_last15_debug
TRAIN_FRAC="${TRAIN_FRAC:-0.85}"

if [ "$V9_REPLAY_MODE" = "shifted_last15_debug" ]; then
  WARMUP=$(python3 -c "print(int($ORIGINAL_WARMUP + $ORIGINAL_SIM * $TRAIN_FRAC))")
  SIM=$(python3   -c "print(int($ORIGINAL_SIM * (1.0 - $TRAIN_FRAC)))")
  echo "[warn] V9_REPLAY_MODE=shifted_last15_debug"
  echo "[warn] Existing V9 lists use original full-window idx values; this mode can issue 0/partial prefetches unless the list was rebased."
else
  WARMUP="${WARMUP:-$ORIGINAL_WARMUP}"
  SIM="${SIM:-$ORIGINAL_SIM}"
fi

# Extract the trace short tag the notebook used (e.g. 605.mcf_s-994B -> mcf_s-994B)
SHORT_TAG="${TRACE#*.}"      # e.g. mcf_s-994B
MODEL_TAG="${MODEL_TAG:-GRU_V9_${SHORT_TAG}}"

# New notebook output location:
#   results/generated/prefetch_lists/prefetch_list_GRU_V9_<trace_tag>.txt
# Keep the old repo-root location as a fallback for older runs.
PFETCH_GENERATED="$WORKDIR/results/generated/prefetch_lists/prefetch_list_GRU_V9_${SHORT_TAG}.txt"
PFETCH_ROOT="$WORKDIR/prefetch_list_GRU_V9_${SHORT_TAG}.txt"

if [ -z "${PFETCH:-}" ]; then
  if [ -f "$PFETCH_GENERATED" ]; then
    PFETCH="$PFETCH_GENERATED"
  elif [ -f "$PFETCH_ROOT" ]; then
    PFETCH="$PFETCH_ROOT"
  else
    PFETCH="$PFETCH_GENERATED"
  fi
fi

if [ ! -f "$PFETCH" ]; then
  echo "[error] prefetch list missing for trace=$TRACE"
  echo "        tried generated path: $PFETCH_GENERATED"
  echo "        tried root path     : $PFETCH_ROOT"
  echo "        or set PFETCH=/full/path/to/prefetch_list.txt"
  exit 1
fi

nlines=$(wc -l < "$PFETCH")
first_line=$(head -1 "$PFETCH")
last_line=$(tail -1 "$PFETCH")
first_idx=$(echo "$first_line" | awk '{print $1}')
last_idx=$(echo "$last_line" | awk '{print $1}')
echo "[check] $PFETCH  lines=$nlines  first=$first_line  last=$last_line"

echo
echo "============================================================"
echo "  V9 GRU prefetcher replay"
echo "  trace          : $TRACE"
echo "  prefetch list  : $PFETCH"
echo "  replay mode    : $V9_REPLAY_MODE"
echo "  warmup / sim   : $WARMUP / $SIM instructions"
echo "  list idx range : ${first_idx:-?} .. ${last_idx:-?}"
echo "============================================================"

TRACE="$TRACE" \
PFETCH="$PFETCH" \
MODEL_TAG="$MODEL_TAG" \
WARMUP="$WARMUP" \
SIM="$SIM" \
bash scripts/run_nn_replay.sh

SUMMARY="$WORKDIR/results/nn_demo_summary.csv"
echo
echo "============================================================"
echo "[DONE]  filtered $SUMMARY:"
echo "============================================================"
grep -E "^(trace,|.*,GRU_V[0-9])" "$SUMMARY" || cat "$SUMMARY"

echo
echo "============================================================"
echo "Index-alignment note:"
echo "  - Existing V9 lists store idx from the original dumped CSV coordinate system."
echo "  - This script now defaults to full_index replay: same warmup/sim as the dumper."
echo "  - If issued prefetches is still near 0 while list entries were loaded, check that"
echo "    the prefetch list was generated from the same dump window as this replay."
echo "============================================================"