#!/usr/bin/env bash
# Run a frozen-policy capacity sensitivity control for L1D, L2C, and LLC.
#
# IMPORTANT: this script replays baseline-capacity frozen L2C lists under new
# cache capacities. It is a system-sensitivity control, not a capacity-trained
# NN result. A valid capacity-trained result requires: capacity-specific no-pref
# collection -> capacity-specific oracle -> Colab training/export -> replay.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LEVELS="${LEVELS:-L1D L2C LLC}"
SCALES="${SCALES:-half base double}"
TRACES="${TRACES:-602.gcc_s-734B 619.lbm_s-4268B 605.mcf_s-994B 620.omnetpp_s-874B 623.xalancbmk_s-700B}"
PREFETCHERS="${PREFETCHERS:-no_pref stride streamer ampm spp ipcp sms sandbox power7}"
# Semicolon-separated LABEL=ARTIFACT_DIR entries.
NN_VARIANTS="${NN_VARIANTS:-v3_1=formal_NN_training/artifacts/standalone_multihorizon_lstm_v3_1;v3_3=formal_NN_training/artifacts/standalone_multihorizon_lstm_v3_3_context_coverage}"
WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"
MAX_JOBS="${MAX_JOBS:-2}"
CHUNK_LEN="${CHUNK_LEN:-1024}"
DEDUP_CAPACITY="${DEDUP_CAPACITY:-256}"
EXPORT_SUFFIX="${EXPORT_SUFFIX:-pure_balanced_lru${DEDUP_CAPACITY}}"
FORCE="${FORCE:-0}"
PATCH_LOGGER="${PATCH_LOGGER:-1}"
RESET_PATCH="${RESET_PATCH:-0}"

CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
CACHE_H="$CHAMP_DIR/inc/cache.h"
OUT_ROOT="${OUT_ROOT:-$ROOT/formal_NN_training/results/cache_capacity_sweep/frozen_l2c_control_$(date +%Y%m%d_%H%M%S)}"
BUILD_CAPACITY="$ROOT/formal_NN_training/scripts/13_build_cache_capacity_variant.sh"
RUN_NORMAL="$ROOT/formal_NN_training/scripts/04_run_normal_prefetcher_sweep.sh"
RUN_REPLAY="$ROOT/formal_NN_training/scripts/08_run_standalone_lstm_replay.sh"

[[ -f "$CACHE_H" ]] || { echo "[error] missing $CACHE_H" >&2; exit 2; }
[[ -f "$BUILD_CAPACITY" ]] || { echo "[error] missing $BUILD_CAPACITY" >&2; exit 2; }
[[ -f "$RUN_NORMAL" && -f "$RUN_REPLAY" ]] || { echo "[error] missing normal/replay driver" >&2; exit 2; }
mkdir -p "$OUT_ROOT"

macro_value() {
  local macro="$1"
  awk -v macro="$macro" '$1 == "#define" && $2 == macro { print $3; exit }' "$CACHE_H"
}

scaled_sets() {
  local base="$1" scale="$2"
  case "$scale" in
    half) (( base >= 2 )) || { echo "[error] cannot halve SET count $base" >&2; return 1; }; echo $((base / 2)) ;;
    base) echo "$base" ;;
    double) echo $((base * 2)) ;;
    quad) echo $((base * 4)) ;;
    *) echo "[error] unsupported scale '$scale' (use half/base/double/quad)" >&2; return 1 ;;
  esac
}

variant_tag() {
  local level="$1" sets="$2" ways="$3"
  local bytes=$((sets * ways * 64))
  printf '%s_%sset_%sway_%sKiB' "${level,,}" "$sets" "$ways" "$((bytes / 1024))"
}

for level in $LEVELS; do
  case "$level" in L1D|L2C|LLC) ;; *) echo "[error] invalid LEVEL=$level" >&2; exit 2 ;; esac
  base_sets="$(macro_value "${level}_SET")"
  base_ways="$(macro_value "${level}_WAY")"
  [[ "$base_sets" =~ ^[0-9]+$ && "$base_ways" =~ ^[0-9]+$ ]] || {
    echo "[error] could not read ${level}_SET/${level}_WAY from $CACHE_H" >&2; exit 2; }

  for scale in $SCALES; do
    sets="$(scaled_sets "$base_sets" "$scale")"
    ways="$base_ways"
    tag="$(variant_tag "$level" "$sets" "$ways")"
    variant_root="$OUT_ROOT/$tag"
    bin_dir="$variant_root/binaries"
    normal_root="$variant_root/normal"
    mkdir -p "$bin_dir"

    echo "[capacity] level=$level scale=$scale sets=$sets ways=$ways tag=$tag"
    LEVEL="$level" SETS="$sets" WAYS="$ways" FRONTEND=normal \
      PATCH_LOGGER="$PATCH_LOGGER" RESET_PATCH="$RESET_PATCH" \
      CHAMP_DIR="$CHAMP_DIR" OUT_DIR="$bin_dir" \
      bash "$BUILD_CAPACITY"
    normal_bin="$bin_dir/champsim.${tag}.normal"
    [[ -x "$normal_bin" ]] || { echo "[error] missing normal capacity binary $normal_bin" >&2; exit 3; }

    OUT_ROOT="$normal_root" BIN="$normal_bin" TRACES="$TRACES" \
      PREFETCHERS="$PREFETCHERS" WARMUP="$WARMUP" SIM="$SIM" \
      MAX_JOBS="$MAX_JOBS" FORCE="$FORCE" BUILD=0 \
      bash "$RUN_NORMAL"
    normal_summary="$normal_root/summary.csv"
    [[ -s "$normal_summary" ]] || { echo "[error] no normal summary: $normal_summary" >&2; exit 3; }

    LEVEL="$level" SETS="$sets" WAYS="$ways" FRONTEND=replayer \
      PATCH_LOGGER="$PATCH_LOGGER" RESET_PATCH=0 \
      CHAMP_DIR="$CHAMP_DIR" OUT_DIR="$bin_dir" \
      bash "$BUILD_CAPACITY"
    replay_bin="$bin_dir/champsim.${tag}.replayer"
    [[ -x "$replay_bin" ]] || { echo "[error] missing replay capacity binary $replay_bin" >&2; exit 3; }

    IFS=';' read -r -a variants <<< "$NN_VARIANTS"
    for spec in "${variants[@]}"; do
      [[ -n "$spec" ]] || continue
      label="${spec%%=*}"
      art_dir="${spec#*=}"
      [[ "$label" != "$art_dir" ]] || { echo "[error] NN_VARIANTS requires LABEL=ARTIFACT_DIR" >&2; exit 2; }
      out_dir="$variant_root/standalone/$label"
      echo "[frozen L2C replay control] $tag $label"
      BIN="$replay_bin" ART_DIR="$art_dir" OUT_DIR="$out_dir" TRACES="$TRACES" \
        WARMUP="$WARMUP" SIM="$SIM" MAX_JOBS="$MAX_JOBS" \
        CHUNK_LEN="$CHUNK_LEN" DEDUP_CAPACITY="$DEDUP_CAPACITY" \
        EXPORT_SUFFIX="$EXPORT_SUFFIX" BASELINE_SUMMARY="$normal_summary" \
        FORCE="$FORCE" RUN_SAME_BINARY_NO_PREF=1 \
        bash "$RUN_REPLAY"
    done

    cat > "$variant_root/RUN_INFO.txt" <<EOF
RUN_KIND=frozen_l2c_capacity_sensitivity_control
LEVEL=$level
SCALE=$scale
SETS=$sets
WAYS=$ways
CAPACITY_BYTES=$((sets * ways * 64))
TRACES=$TRACES
PREFETCHERS=$PREFETCHERS
NN_VARIANTS=$NN_VARIANTS
WARMUP=$WARMUP
SIM=$SIM
CHUNK_LEN=$CHUNK_LEN
EXPORT_SUFFIX=$EXPORT_SUFFIX
NORMAL_BINARY=$normal_bin
REPLAYER_BINARY=$replay_bin
NORMAL_SUMMARY=$normal_summary
IMPORTANT=Frozen lists use the baseline-capacity raw oracle. This is not a capacity-trained NN experiment.
EOF
  done
done

echo "[done] $OUT_ROOT"
