#!/usr/bin/env bash
# One-minute v3.9 605 sidecar preflight.  It validates the executable and
# resolves all paths before launching the full compressed-trace scan.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/cache}"
TRACE="${TRACE:-$REPO_ROOT/traces/605.mcf_s-994B.champsimtrace.xz}"
ORACLE="${ORACLE:-$REPO_ROOT/formal_NN_training/results/standalone_nn_data/oracle/605.mcf_s-994B.oracle.csv.gz}"
BUILDER="$REPO_ROOT/formal_NN_training/scripts/16_build_trace_dependency_features.py"

[[ -f "$TRACE" ]] || { echo "[error] trace not found: $TRACE" >&2; exit 2; }
[[ -f "$ORACLE" ]] || { echo "[error] oracle not found: $ORACLE" >&2; exit 2; }
[[ -f "$BUILDER" ]] || { echo "[error] builder not found: $BUILDER" >&2; exit 2; }

python3 "$BUILDER" --help | grep -E -- '--align-mode|--max-anchor-search' >/dev/null

echo "[ok] python=$(python3 --version 2>&1)"
echo "[ok] builder=$BUILDER"
echo "[ok] trace=$TRACE"
echo "[ok] oracle=$ORACLE"
echo "[ok] builder supports page-offset alignment"
