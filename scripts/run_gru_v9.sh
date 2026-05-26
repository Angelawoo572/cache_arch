#!/usr/bin/env bash
# run_gru_v9.sh
# Replays the V9 in-distribution prefetch list through ChampSim.
#
# IMPORTANT change vs run_gru_v9_old:
#   V9 trains on first 70%, validates on next 15%, TESTS on last 15% of the trace.
#   The prefetch list it produces is keyed to dumper-indices in the LAST 15%.
#   So the ChampSim simulation window must match that last 15%.
#
#   For our existing 25M-warmup + 25M-sim runs, the simplest mapping is:
#       New warmup = 25M (original warmup) + 25M * 0.85 = 46.25M
#       New sim    = 25M * 0.15 = 3.75M
#   That keeps the "trace position" aligned with where the dumper recorded indices.
#
#   BUT: the existing dumper writes idx starting from 0 at the start of sim_phase
#   (it does NOT count warmup accesses). So we just need to skip 85% of sim and
#   replay the last 15%. Set:
#       WARMUP = original_warmup + (original_sim * 0.85)
#       SIM    = original_sim * 0.15
#
# Usage:
#   TRACE=605.mcf_s-994B bash scripts/run_gru_v9.sh
#
# To do the full 3-trace sweep:
#   bash scripts/run_gru_v9_sweep.sh

set -uo pipefail

WORKDIR="$(pwd)"
TRACE="${TRACE:-605.mcf_s-994B}"

# Default dumper window was 25M warmup + 25M sim. V9's "test" is the last 15% of
# the dumped CSV, so for the ChampSim replay we need to align the simulation
# window to that last 15%. See header for derivation.
ORIGINAL_WARMUP="${ORIGINAL_WARMUP:-25000000}"
ORIGINAL_SIM="${ORIGINAL_SIM:-25000000}"
TRAIN_FRAC="${TRAIN_FRAC:-0.85}"     # train+val end at 85% of dumped CSV

# Convert to ChampSim warmup/sim instruction counts.
# (Reuse ORIGINAL_SIM as a proxy for the dumper-window instruction count;
#  this is correct because the dumper writes one row per L1 demand miss but
#  ChampSim counts instructions, not misses. If we ran 25M+25M originally,
#  the position 85% into the dumped CSV corresponds to roughly 25M + 21.25M
#  retired instructions. For tighter alignment, re-dump with the new bounds.)
WARMUP=$(python3 -c "print(int($ORIGINAL_WARMUP + $ORIGINAL_SIM * $TRAIN_FRAC))")
SIM=$(python3   -c "print(int($ORIGINAL_SIM * (1.0 - $TRAIN_FRAC)))")

# Extract the trace short tag the notebook used (e.g. 605.mcf_s-994B -> mcf_s-994B)
SHORT_TAG="${TRACE#*.}"      # e.g. mcf_s-994B
PFETCH_DEFAULT="$WORKDIR/prefetch_list_GRU_V9_${SHORT_TAG}.txt"
PFETCH="${PFETCH:-$PFETCH_DEFAULT}"
MODEL_TAG="${MODEL_TAG:-GRU_V9_${SHORT_TAG}}"

if [ ! -f "$PFETCH" ]; then
  echo "[error] prefetch list $PFETCH missing -- run gru_sweep_v9.ipynb first"
  echo "        (look for prefetch_list_GRU_V9_<trace_tag>.txt in Colab output)"
  exit 1
fi

nlines=$(wc -l < "$PFETCH")
first=$(head -1 "$PFETCH")
echo "[check] $PFETCH  lines=$nlines  first=$first"

echo
echo "============================================================"
echo "  V9 GRU prefetcher replay"
echo "  trace          : $TRACE"
echo "  prefetch list  : $PFETCH"
echo "  warmup / sim   : $WARMUP / $SIM instructions"
echo "                   (corresponds to last $(python3 -c "print(int((1.0-$TRAIN_FRAC)*100))")% of original dumped window)"
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
echo "Interpretation:"
echo "  - V9 is in-distribution: train+test on the SAME workload."
echo "    This is the standard ML-prefetching evaluation (Voyager, Pythia, Hashemi)."
echo "  - V9 vs V8 on omnetpp would have been near-impossible because omnetpp's"
echo "    IPC is insensitive to prefetching (slide 8: 0.4% spread across configs)."
echo "  - Best traces to demo V9 on:"
echo "      gcc:  +139% IPC headroom from SPP (slide 8) - biggest room"
echo "      lbm:  +18% from SPP - streaming, NN should learn easily"
echo "      mcf:  in-distribution should beat V4's 0.7458 val_top1"
echo "============================================================"
