#!/usr/bin/env bash
# Build and package the derived v3.9 dependency sidecar for 605.mcf_s-994B.
# Compatible with Sacramento's Python 3.6 system interpreter: no pandas, no
# numpy, and no Python annotations are required by the builder.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/cache}"
TRACE="${TRACE:-$REPO_ROOT/traces/605.mcf_s-994B.champsimtrace.xz}"
ORACLE="${ORACLE:-$REPO_ROOT/formal_NN_training/results/standalone_nn_data/oracle/605.mcf_s-994B.oracle.csv.gz}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/formal_NN_training/artifacts/v3_9_dependency_sidecars}"
WARMUP_RECORDS="${WARMUP_RECORDS:-25000000}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1000000}"
ALIGN_MODE="${ALIGN_MODE:-pc_page_offset}"
MAX_ANCHOR_SEARCH="${MAX_ANCHOR_SEARCH:-5000000}"

BUILDER="$REPO_ROOT/formal_NN_training/scripts/16_build_trace_dependency_features.py"
OUT="$OUT_DIR/605.mcf_s-994B.v3_9_dependency.csv.gz"
META="${OUT%.csv.gz}.json"
PACKAGE="$OUT_DIR/605.mcf_s-994B.v3_9_dependency_sidecar.tar.gz"

[[ -d "$REPO_ROOT/.git" ]] || { echo "[error] REPO_ROOT is not a cache repo: $REPO_ROOT" >&2; exit 2; }
[[ -f "$TRACE" ]] || { echo "[error] trace not found: $TRACE" >&2; exit 2; }
[[ -f "$ORACLE" ]] || { echo "[error] oracle not found: $ORACLE" >&2; exit 2; }
[[ -f "$BUILDER" ]] || { echo "[error] builder script not found: $BUILDER" >&2; exit 2; }

mkdir -p "$OUT_DIR"
rm -f "$OUT" "$META" "$PACKAGE" "$OUT.partial" "${META}.partial"

echo "[build] trace=$TRACE"
echo "[build] oracle=$ORACLE"
echo "[build] output=$OUT"
echo "[build] align_mode=$ALIGN_MODE max_anchor_search=$MAX_ANCHOR_SEARCH"
python3 "$BUILDER" \
  --trace "$TRACE" \
  --oracle "$ORACLE" \
  --output "$OUT" \
  --warmup-records "$WARMUP_RECORDS" \
  --progress-every "$PROGRESS_EVERY" \
  --min-alignment 1.0 \
  --align-mode "$ALIGN_MODE" \
  --max-anchor-search "$MAX_ANCHOR_SEARCH"

[[ -s "$OUT" && -s "$META" ]] || { echo "[error] expected sidecar outputs were not produced" >&2; exit 3; }
tar -C "$OUT_DIR" -czf "$PACKAGE" "$(basename "$OUT")" "$(basename "$META")"

python3 - "$META" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    meta = json.load(handle)
assert meta.get("alignment") == 1.0, meta
assert meta.get("alignment_mode") == "pc_page_offset", meta
print("[verified] aligned_events={:,} dependency_present_fraction={:.4f}".format(
    meta["aligned_events"], meta["dependency_present_fraction"]
))
PY

sha256sum "$OUT" "$META" "$PACKAGE"
echo "[done] package=$PACKAGE"
