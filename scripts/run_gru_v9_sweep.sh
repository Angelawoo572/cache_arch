#!/usr/bin/env bash
# run_gru_v9_sweep.sh
# Full V9 protocol: 3 traces, dump each, train V9 in Colab, replay in ChampSim.
#
# This is a SHELL FRAGMENT showing the sequence. Run each step manually because
# the Colab training step requires uploading the CSV to Colab.
#
# Recommended order (cheapest to most expensive):
#   1. mcf  -- direct comparison vs V1..V8 (same trace as before)
#   2. lbm  -- streaming, NN should easily get high accuracy
#   3. gcc  -- biggest prefetch headroom

set -uo pipefail
WORKDIR="$(pwd)"

cat <<EOF
============================================================
V9 SWEEP PROTOCOL
============================================================

Step 1 (lab machine): dump trace CSVs (if not already done)
  bash scripts/dump_trace.sh 605.mcf_s-994B
  bash scripts/dump_trace.sh 619.lbm_s-4268B
  bash scripts/dump_trace.sh 602.gcc_s-734B

Step 2 (Colab): for EACH trace, upload its CSV and run gru_sweep_v9.ipynb
  - edit TRACE_CSV at the top of cell 4
  - Restart Runtime + Run All
  - download prefetch_list_GRU_V9_<tag>.txt to lab

Step 3 (lab machine): replay each one through ChampSim
  TRACE=605.mcf_s-994B    bash scripts/run_gru_v9.sh
  TRACE=619.lbm_s-4268B   bash scripts/run_gru_v9.sh
  TRACE=602.gcc_s-734B    bash scripts/run_gru_v9.sh

Step 4: examine results/nn_demo_summary.csv
  grep -E "GRU_V9" results/nn_demo_summary.csv

============================================================
What to look for
============================================================
- V9 on mcf (in-distribution): expect speedup >= 1.0x (V8 was 0.96x cross-binary)
- V9 on lbm: expect speedup >= 1.10x (lbm has +18% headroom from SPP)
- V9 on gcc: expect speedup >= 1.20x (gcc has +139% headroom from SPP)

If V9 beats V8 on mcf but not by much, the bottleneck is omnetpp/mcf workload class,
not the model. If V9 wins big on lbm and gcc, the model is fine and we just need
better-suited traces.
EOF
