#!/bin/bash
# Reproduce every capture of the measurement study on Sacramento.
# Finished checkpoints and runs are skipped, so the script can be rerun.
# Usage: run_matrix.sh [parallel gem5 processes]     default 14
set -eu
here=$(cd "$(dirname "$0")" && pwd)
source "$here/env.sh"
cd "$GEM5_STUDY"
jobs=${1:-14}

micro="micro_stride micro_list micro_tree micro_csr micro_hash"
gap="gap_bfs_kron20 gap_bfs_urand20 gap_cc_kron20 gap_cc_urand20 gap_pr_kron20 gap_pr_urand20 gap_bc_kron20 gap_sssp_kron20"

# Validation programs: short windows, with and without the capture object.
python sim/np_runs.py checkpoint $micro -j 5
python sim/np_runs.py run --arms nopf,stride,fast --warmup 500000 --measure 2000000 $micro -j "$jobs"
python sim/np_runs.py run --arms nopf --no-capture --warmup 500000 --measure 2000000 $micro -j 5

# GAP kernels: 5M instructions of warm-up, 20M measured, two windows each.
python sim/np_runs.py checkpoint $gap -j 8
python sim/np_runs.py run --arms nopf,stride,fast --warmup 5000000 --measure 20000000 $gap -j "$jobs"
python sim/np_runs.py run --arms bop,ampm,imp --warmup 5000000 --measure 20000000 $gap -j "$jobs"
