#!/usr/bin/env bash
# Build and package the v3.9 605 PC-keyed static dependency profile.
#
# This wrapper creates its own output directory and log. Start it with:
#   nohup bash formal_NN_training/scripts/17_prepare_v3_9_605_dependency_sidecar.sh &
# Shell redirection is intentionally unnecessary; it was the source of the
# earlier "No such file or directory" failure before OUT_DIR existed.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/cache}"
TRACE="${TRACE:-$REPO_ROOT/traces/605.mcf_s-994B.champsimtrace.xz}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/formal_NN_training/artifacts/v3_9_dependency_sidecars}"
WARMUP_RECORDS="${WARMUP_RECORDS:-25000000}"
PROFILE_RECORDS="${PROFILE_RECORDS:-20000000}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1000000}"

BUILDER="$REPO_ROOT/formal_NN_training/scripts/16_build_trace_dependency_features.py"
OUT="$OUT_DIR/605.mcf_s-994B.v3_9_dependency_profile.csv.gz"
META="$OUT_DIR/605.mcf_s-994B.v3_9_dependency_profile.json"
PACKAGE="$OUT_DIR/605.mcf_s-994B.v3_9_dependency_sidecar.tar.gz"
LOG="$OUT_DIR/605_sidecar_build.out"

[[ -d "$REPO_ROOT/.git" ]] || { echo "[error] REPO_ROOT is not a cache repo: $REPO_ROOT" >&2; exit 2; }
[[ -f "$TRACE" ]] || { echo "[error] trace not found: $TRACE" >&2; exit 2; }
[[ -f "$BUILDER" ]] || { echo "[error] builder script not found: $BUILDER" >&2; exit 2; }

mkdir -p "$OUT_DIR"

# Capture the entire build inside the artifact directory. This makes the exact
# nohup command safe even on a fresh checkout.
exec >> "$LOG" 2>&1

echo "===== v3.9 605 dependency-profile build ====="
date
echo "[build] repo=$REPO_ROOT"
echo "[build] trace=$TRACE"
echo "[build] output=$OUT"
echo "[build] warmup_records=$WARMUP_RECORDS profile_records=$PROFILE_RECORDS"

rm -f "$OUT" "$META" "$PACKAGE" "${OUT}.partial" "${META}.partial"

python3 "$BUILDER" \
  --trace "$TRACE" \
  --output "$OUT" \
  --meta "$META" \
  --warmup-records "$WARMUP_RECORDS" \
  --profile-records "$PROFILE_RECORDS" \
  --progress-every "$PROGRESS_EVERY"

[[ -s "$OUT" && -s "$META" ]] || { echo "[error] expected profile outputs were not produced" >&2; exit 3; }

python3 - "$META" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    meta = json.load(handle)
assert meta.get("schema") == "v3_9_pc_static_dependency_profile", meta
assert meta.get("profile_scope") == "raw-trace training prefix only", meta
assert meta.get("uses_oracle_alignment") is False, meta
print("[verified] unique_pcs={:,} dependency_pcs={:,} profile_records={:,}".format(
    int(meta["unique_pcs"]),
    int(meta["pcs_with_dependency_observations"]),
    int(meta["profile_records"]),
))
PY

tar -C "$OUT_DIR" -czf "$PACKAGE" "$(basename "$OUT")" "$(basename "$META")"
sha256sum "$OUT" "$META" "$PACKAGE"
echo "[done] package=$PACKAGE"
date
