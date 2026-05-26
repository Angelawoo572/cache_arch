#!/usr/bin/env bash
# run_gru_v8.sh
# Replays the V8 prefetch list (delta-prediction GRU) through ChampSim on the
# TEST trace and compares against baseline + your previous V1..V4 numbers.
#
# Usage:
#   # default: run only the gated V8 prefetch list against omnetpp
#   TRACE=620.omnetpp_s-874B bash projects/legacy_gru_prefetch/scripts/run_gru_v8.sh
#
#   # also run the ungated ablation (V8_ungated.txt must exist):
#   ABLATE=1 TRACE=620.omnetpp_s-874B bash projects/legacy_gru_prefetch/scripts/run_gru_v8.sh

set -uo pipefail

WORKDIR="$(pwd)"
TRACE="${TRACE:-620.omnetpp_s-874B}"
WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"
ABLATE="${ABLATE:-0}"

PFETCH_MAIN="$WORKDIR/prefetch_list_GRU_V8.txt"
PFETCH_UNGATED="$WORKDIR/prefetch_list_GRU_V8_ungated.txt"

if [ ! -f "$PFETCH_MAIN" ]; then
  echo "[error] $PFETCH_MAIN missing -- run gru_sweep_v8.ipynb first"
  echo "        and download the prefetch list from Colab to this directory."
  exit 1
fi

# Quick sanity check on the prefetch list format
nlines=$(wc -l < "$PFETCH_MAIN")
first=$(head -1 "$PFETCH_MAIN")
echo "[check] $PFETCH_MAIN  lines=$nlines  first=$first"
if ! echo "$first" | grep -qE "^[0-9]+ 0x[0-9a-f]+$"; then
  echo "[error] prefetch list format looks wrong -- expected 'idx 0xhex' per line"
  exit 1
fi

echo
echo "==================================================================="
echo "  V8 GRU prefetcher replay on $TRACE"
echo "  delta-prediction, L1DM-filtered training, confidence-gated export"
echo "==================================================================="
echo

TRACE="$TRACE" \
PFETCH="$PFETCH_MAIN" \
MODEL_TAG="GRU_V8" \
WARMUP="$WARMUP" \
SIM="$SIM" \
bash projects/legacy_gru_prefetch/scripts/run_nn_replay.sh

if [ "$ABLATE" = "1" ]; then
  if [ ! -f "$PFETCH_UNGATED" ]; then
    echo "[warn] ABLATE=1 set but $PFETCH_UNGATED is missing -- skipping ablation"
  else
    echo
    echo "==================================================================="
    echo "  V8 ablation: SAME model, NO confidence gate"
    echo "  (compare IPC vs gated V8 to attribute IPC change to gating alone)"
    echo "==================================================================="
    echo
    TRACE="$TRACE" \
    PFETCH="$PFETCH_UNGATED" \
    MODEL_TAG="GRU_V8_ungated" \
    WARMUP="$WARMUP" \
    SIM="$SIM" \
    bash projects/legacy_gru_prefetch/scripts/run_nn_replay.sh
  fi
fi

SUMMARY="$WORKDIR/results/nn_demo_summary.csv"
echo
echo "==================================================================="
echo "[ALL DONE]  $SUMMARY  (filtered to V8 + the previous GRU runs):"
echo "==================================================================="
grep -E "^(trace,|.*,GRU_V[0-9]|.*,GRU_V8)" "$SUMMARY" || cat "$SUMMARY"
echo
echo "Interpretation:"
echo "  - Compare V8 IPC vs V4 IPC (same trace, same warmup/sim)."
echo "  - V8 SHOULD beat V4 because:"
echo "     * delta target is learnable on pointer-chase (V4's offset target was chance-level)"
echo "     * confidence gate reduces trigger rate -> less cache pollution"
echo "  - If V8 still loses to baseline, the next lever is per-PC bucketed models"
echo "    (Voyager-style hierarchical) -- see boss_report Idea B."
