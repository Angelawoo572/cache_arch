#!/usr/bin/env bash
# Prepare one VALID capacity-specific standalone-NN training point.
#
# This driver does NOT replay a baseline-capacity frozen LSTM list under a changed
# cache. It builds one changed-capacity normal binary, collects its matching raw
# no-prefetch L2C demand stream, builds matching oracle files, and runs selected
# normal-prefetcher comparison baselines. The resulting oracle must be trained and
# exported in Colab before a matching keyed replay can be evaluated.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LEVEL="${LEVEL:?set LEVEL=L1D, L2C, or LLC}"
SCALE="${SCALE:?set SCALE=half, base, double, or quad}"
TRACES="${TRACES:-602.gcc_s-734B}"
PREFETCHERS="${PREFETCHERS:-no_pref sandbox}"
WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"
MAX_JOBS="${MAX_JOBS:-1}"
FORCE="${FORCE:-0}"
RESET_PATCH="${RESET_PATCH:-0}"

CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
CACHE_H="$CHAMP_DIR/inc/cache.h"
BUILD_VARIANT="$ROOT/formal_NN_training/scripts/13_build_cache_capacity_variant.sh"
COLLECT_EVENTS="$ROOT/formal_NN_training/scripts/03_collect_no_pref_demand_events.sh"
BUILD_ORACLE="$ROOT/formal_NN_training/scripts/05_build_standalone_oracle_dataset.py"
RUN_NORMAL="$ROOT/formal_NN_training/scripts/04_run_normal_prefetcher_sweep.sh"

[[ -f "$CACHE_H" ]] || { echo "[error] missing $CACHE_H" >&2; exit 2; }
[[ -f "$BUILD_VARIANT" && -f "$COLLECT_EVENTS" && -f "$BUILD_ORACLE" && -f "$RUN_NORMAL" ]] || {
  echo "[error] one or more prerequisite scripts are missing" >&2; exit 2; }

case "$LEVEL" in
  L1D|L2C|LLC) ;;
  *) echo "[error] LEVEL must be L1D, L2C, or LLC" >&2; exit 2 ;;
esac

macro_value() {
  local macro="$1"
  awk -v macro="$macro" '$1 == "#define" && $2 == macro { print $3; exit }' "$CACHE_H"
}

scale_sets() {
  local base="$1" scale="$2"
  case "$scale" in
    half)
      (( base >= 2 )) || { echo "[error] cannot halve SET count $base" >&2; return 1; }
      echo $((base / 2))
      ;;
    base) echo "$base" ;;
    double) echo $((base * 2)) ;;
    quad) echo $((base * 4)) ;;
    *) echo "[error] SCALE must be half, base, double, or quad" >&2; return 1 ;;
  esac
}

BASE_SETS="$(macro_value "${LEVEL}_SET")"
BASE_WAYS="$(macro_value "${LEVEL}_WAY")"
[[ "$BASE_SETS" =~ ^[0-9]+$ && "$BASE_WAYS" =~ ^[0-9]+$ ]] || {
  echo "[error] could not read ${LEVEL}_SET/${LEVEL}_WAY from $CACHE_H" >&2; exit 2; }

SETS="$(scale_sets "$BASE_SETS" "$SCALE")"
WAYS="$BASE_WAYS"
CAPACITY_BYTES=$((SETS * WAYS * 64))
CAP_TAG="${LEVEL,,}_${SETS}set_${WAYS}way_$((CAPACITY_BYTES / 1024))KiB"
OUT_ROOT="${OUT_ROOT:-$ROOT/formal_NN_training/results/capacity_training_points/${CAP_TAG}}"
BIN_DIR="$OUT_ROOT/binaries"
DEMAND_ROOT="$OUT_ROOT/no_pref_demand_events"
ORACLE_DIR="$OUT_ROOT/oracle"
BASELINE_ROOT="$OUT_ROOT/normal_baselines"
NORMAL_BIN="$BIN_DIR/champsim.${CAP_TAG}.normal"

mkdir -p "$BIN_DIR" "$ORACLE_DIR"

echo "[capacity point] level=$LEVEL scale=$SCALE sets=$SETS ways=$WAYS bytes=$CAPACITY_BYTES"
echo "[capacity point] output=$OUT_ROOT"

LEVEL="$LEVEL" \
SETS="$SETS" \
WAYS="$WAYS" \
FRONTEND=normal \
PATCH_LOGGER=1 \
RESET_PATCH="$RESET_PATCH" \
CHAMP_DIR="$CHAMP_DIR" \
OUT_DIR="$BIN_DIR" \
bash "$BUILD_VARIANT"

[[ -x "$NORMAL_BIN" ]] || { echo "[error] expected capacity binary missing: $NORMAL_BIN" >&2; exit 3; }

OUT_ROOT="$DEMAND_ROOT" \
BIN="$NORMAL_BIN" \
CHAMP_DIR="$CHAMP_DIR" \
TRACES="$TRACES" \
WARMUP="$WARMUP" \
SIM="$SIM" \
MAX_JOBS="$MAX_JOBS" \
BUILD=0 \
FORCE="$FORCE" \
bash "$COLLECT_EVENTS"

for trace in $TRACES; do
  events="$DEMAND_ROOT/events/${trace}.no_pref.events.csv.gz"
  oracle="$ORACLE_DIR/${trace}.oracle.csv.gz"
  meta="$ORACLE_DIR/${trace}.oracle.csv.gz.meta.json"
  [[ -s "$events" ]] || { echo "[error] missing no-pref events: $events" >&2; exit 3; }
  python3 "$BUILD_ORACLE" \
    --events "$events" \
    --trace "$trace" \
    --out "$oracle" \
    --meta-out "$meta"
done

OUT_ROOT="$BASELINE_ROOT" \
BIN="$NORMAL_BIN" \
CHAMP_DIR="$CHAMP_DIR" \
TRACES="$TRACES" \
PREFETCHERS="$PREFETCHERS" \
WARMUP="$WARMUP" \
SIM="$SIM" \
MAX_JOBS="$MAX_JOBS" \
BUILD=0 \
FORCE="$FORCE" \
bash "$RUN_NORMAL"

cat > "$OUT_ROOT/RUN_INFO.txt" <<EOF
RUN_KIND=capacity_specific_training_point_preparation
LEVEL=$LEVEL
SCALE=$SCALE
BASE_SETS=$BASE_SETS
BASE_WAYS=$BASE_WAYS
SETS=$SETS
WAYS=$WAYS
CAPACITY_BYTES=$CAPACITY_BYTES
CAP_TAG=$CAP_TAG
TRACES=$TRACES
PREFETCHERS=$PREFETCHERS
WARMUP=$WARMUP
SIM=$SIM
NORMAL_BINARY=$NORMAL_BIN
DEMAND_ROOT=$DEMAND_ROOT
ORACLE_DIR=$ORACLE_DIR
BASELINE_SUMMARY=$BASELINE_ROOT/summary.csv
NEXT_REQUIRED_STEP=Train_and_export_a_new_Colab_model_from_this_exact_oracle_before_keyed_replay.
EOF

echo "[ready] capacity-specific oracle directory: $ORACLE_DIR"
echo "[ready] capacity-specific normal baseline: $BASELINE_ROOT/summary.csv"
echo "[next] Train/export a new Colab artifact from this oracle. Do not replay a baseline-capacity artifact here."
