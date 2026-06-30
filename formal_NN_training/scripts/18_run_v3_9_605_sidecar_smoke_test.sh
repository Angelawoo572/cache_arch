#!/usr/bin/env bash
# Fast, self-contained preflight for the v3.9 605 profile builder.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/cache}"
TRACE="${TRACE:-$REPO_ROOT/traces/605.mcf_s-994B.champsimtrace.xz}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/formal_NN_training/artifacts/v3_9_dependency_sidecars}"
BUILDER="$REPO_ROOT/formal_NN_training/scripts/16_build_trace_dependency_features.py"
TMP="$OUT_DIR/.605_dependency_profile_smoke.csv.gz"

[[ -f "$TRACE" ]] || { echo "[error] trace not found: $TRACE" >&2; exit 2; }
[[ -f "$BUILDER" ]] || { echo "[error] builder not found: $BUILDER" >&2; exit 2; }
mkdir -p "$OUT_DIR"

python3 "$BUILDER" --help | grep -E -- '--profile-records|--dry-run' >/dev/null
python3 "$BUILDER" \
  --trace "$TRACE" \
  --output "$TMP" \
  --warmup-records 0 \
  --profile-records 1024 \
  --progress-every 0 \
  --dry-run

rm -f "$TMP" "${TMP%.csv.gz}.json" "${TMP}.partial"
echo "[ok] python=$(python3 --version 2>&1)"
echo "[ok] builder parsed 1,024 input_instr records with no third-party Python packages"
