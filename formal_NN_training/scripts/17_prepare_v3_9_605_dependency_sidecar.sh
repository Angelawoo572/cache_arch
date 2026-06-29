#!/usr/bin/env bash
# Build and package the derived v3.9 dependency sidecar for 605.mcf_s-994B.
# This reads the existing original trace and no-prefetch oracle only. It does
# not change the trace, train a model, or run ChampSim.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/cache}"
TRACE="${TRACE:-$REPO_ROOT/traces/605.mcf_s-994B.champsimtrace.xz}"
ORACLE_DIR="${ORACLE_DIR:-$REPO_ROOT/formal_NN_training/results/standalone_nn_data/oracle}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/formal_NN_training/artifacts/v3_9_dependency_sidecars}"
WARMUP_RECORDS="${WARMUP_RECORDS:-25000000}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1000000}"

BUILDER="$REPO_ROOT/formal_NN_training/scripts/16_build_trace_dependency_features.py"
OUT="$OUT_DIR/605.mcf_s-994B.v3_9_dependency.npz"
META="${OUT%.npz}.json"
PACKAGE="$OUT_DIR/605.mcf_s-994B.v3_9_dependency_sidecar.tar.gz"

[[ -d "$REPO_ROOT/.git" ]] || { echo "[error] REPO_ROOT is not a cache repo: $REPO_ROOT" >&2; exit 2; }
[[ -f "$TRACE" ]] || { echo "[error] trace not found: $TRACE" >&2; exit 2; }
[[ -d "$ORACLE_DIR" ]] || { echo "[error] oracle directory not found: $ORACLE_DIR" >&2; exit 2; }
[[ -f "$BUILDER" ]] || { echo "[error] builder script not found: $BUILDER" >&2; exit 2; }

mkdir -p "$OUT_DIR"

echo "[build] trace=$TRACE"
echo "[build] oracle_dir=$ORACLE_DIR"
echo "[build] output=$OUT"
python3 "$BUILDER" \
  --trace "$TRACE" \
  --oracle-dir "$ORACLE_DIR" \
  --trace-stem "605.mcf_s-994B" \
  --output "$OUT" \
  --warmup-records "$WARMUP_RECORDS" \
  --progress-every "$PROGRESS_EVERY" \
  --min-alignment 1.0

[[ -s "$OUT" && -s "$META" ]] || { echo "[error] expected sidecar outputs were not produced" >&2; exit 3; }
tar -C "$OUT_DIR" -czf "$PACKAGE" "$(basename "$OUT")" "$(basename "$META")"

python3 - "$META" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1], encoding="utf-8"))
assert meta["alignment"] == 1.0, meta
print("[verified] aligned_events={:,} dependency_present_fraction={:.4f}".format(
    meta["aligned_events"], meta["dependency_present_fraction"]
))
PY

sha256sum "$OUT" "$META" "$PACKAGE"
echo "[done] package=$PACKAGE"
