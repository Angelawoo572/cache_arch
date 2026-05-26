#!/usr/bin/env bash
# run_all_models.sh
# Loops over all 5 prefetch_list_<MODEL>.txt files and runs each through
# ChampSim+replayer. Produces a CSV with 5 rows of real IPC per trace.
#
# Expected layout:
#   $WORKDIR/prefetch_list_Perceptron.txt  (from Colab v3 notebook)
#   $WORKDIR/prefetch_list_MLP.txt
#   $WORKDIR/prefetch_list_CNN.txt
#   $WORKDIR/prefetch_list_LSTM.txt
#   $WORKDIR/prefetch_list_Transformer.txt
#
# Usage:  TRACE=605.mcf_s-994B bash projects/legacy_gru_prefetch/scripts/run_all_models.sh

set -uo pipefail

WORKDIR="$(pwd)"
TRACE="${TRACE:-605.mcf_s-994B}"

MODELS=(Perceptron MLP CNN LSTM Transformer)

# Sanity-check all 5 lists exist before we burn 30 minutes of sim time.
missing=0
for m in "${MODELS[@]}"; do
  pf="$WORKDIR/prefetch_list_${m}.txt"
  if [ ! -f "$pf" ]; then
    echo "[error] missing $pf"
    missing=$((missing+1))
  fi
done
if [ "$missing" -gt 0 ]; then
  echo "[error] $missing prefetch lists missing. Run the Colab v3 notebook first."
  exit 1
fi

# We only need to run the no-prefetch baseline ONCE per trace (the answer is
# independent of which model produced the prefetch list). The first call to
# run_nn_replay.sh below will compute it; we cache by reading the CSV.
SUMMARY="$WORKDIR/results/nn_demo_summary.csv"

for m in "${MODELS[@]}"; do
  echo
  echo "########################################################"
  echo "# Model = $m  on  $TRACE"
  echo "########################################################"
  TRACE="$TRACE" \
  PFETCH="$WORKDIR/prefetch_list_${m}.txt" \
  MODEL_TAG="$m" \
  bash projects/legacy_gru_prefetch/scripts/run_nn_replay.sh
done

echo
echo "============================================"
echo "[ALL DONE] Summary CSV:"
column -t -s, "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo "============================================"
