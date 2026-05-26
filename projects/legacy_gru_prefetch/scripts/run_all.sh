#!/usr/bin/env bash
# run_all.sh
# Master script - runs everything in order. Stop at first failure.
# This is the script to run when you want to set up everything on Friday/Saturday.

set -e
cd "$(dirname "$0")/.."   # go to the working dir (parent of projects/legacy_gru_prefetch/scripts/)

echo "================ STEP 1/4 : setup ================"
bash projects/legacy_gru_prefetch/scripts/setup_champsim.sh

echo
echo "================ STEP 2/4 : baseline ================"
bash projects/legacy_gru_prefetch/scripts/run_baseline.sh

echo
echo "================ STEP 3/4 : upper-bound ================"
bash projects/legacy_gru_prefetch/scripts/run_upper_bound.sh

echo
echo "================ STEP 4/4 : slide 8 data ================"
python3 projects/legacy_gru_prefetch/scripts/make_slide8_data.py

echo
echo "================ ALL DONE ================"
echo
echo "Outputs:"
echo "  results/baseline.csv"
echo "  results/upper_bound.csv"
echo "  results/slide8_data.tex   <-- paste this into slide 8"
echo
echo "Next (for slide 13 MLP demo):"
echo "  1. Run neural_prefetcher_zoo.ipynb in Colab"
echo "  2. Download prefetch_list.txt to this directory"
echo "  3. bash projects/legacy_gru_prefetch/scripts/run_mlp_demo.sh"
