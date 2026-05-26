#!/usr/bin/env bash
# Run real ChampSim with the current binary/config and parse cache hit/miss stats.
# Intended first use: SPP baseline before adding the RL post-prefetch filter.
#
# Usage from repo root:
#   TRACE=602.gcc_s-734B bash projects/post_prefetch_filter/scripts/02_run_spp_baseline_stats.sh
#
# Optional:
#   TRACES="602.gcc_s-734B 605.mcf_s-994B 619.lbm_s-4268B" \
#   WARMUP=25000000 SIM=25000000 \
#   bash projects/post_prefetch_filter/scripts/02_run_spp_baseline_stats.sh
#
# Notes:
# - This script does not reconfigure ChampSim. It runs external/ChampSim/bin/champsim.
# - Before using this as the SPP baseline, make sure the binary was configured with spp_dev
#   at the cache level you want, usually L2C.

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

CHAMP="${CHAMP:-$ROOT/external/ChampSim/bin/champsim}"
TRACE_DIR="${TRACE_DIR:-$ROOT/traces}"
OUT_DIR="$ROOT/projects/post_prefetch_filter/results/spp_baseline"
LOG_DIR="$OUT_DIR/logs"
SUMMARY="$OUT_DIR/spp_baseline_summary.csv"
PARSER="$ROOT/projects/post_prefetch_filter/scripts/parse_champsim_stats.py"

WARMUP="${WARMUP:-25000000}"
SIM="${SIM:-25000000}"

if [ -n "${TRACE:-}" ]; then
  TRACES="${TRACES:-$TRACE}"
else
  TRACES="${TRACES:-602.gcc_s-734B 605.mcf_s-994B 619.lbm_s-4268B}"
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"

if [ ! -x "$CHAMP" ]; then
  echo "[error] ChampSim binary not found or not executable: $CHAMP"
  echo "        Build first, e.g. cd external/ChampSim && python3 ./config.sh <config.json> && make -j"
  exit 1
fi

if [ ! -f "$PARSER" ]; then
  echo "[error] parser missing: $PARSER"
  exit 1
fi

echo "============================================================"
echo "SPP BASELINE REAL CHAMPSIM RUN"
echo "============================================================"
echo "repo       : $ROOT"
echo "champsim   : $CHAMP"
echo "trace dir  : $TRACE_DIR"
echo "traces     : $TRACES"
echo "warmup/sim : $WARMUP / $SIM"
echo "out dir    : $OUT_DIR"
echo "============================================================"

echo
if [ -f "$ROOT/external/ChampSim/_configuration.mk" ]; then
  echo "[config hint] modules in current ChampSim build:"
  grep -E "prefetcherD|Module Names|prefetcher/" "$ROOT/external/ChampSim/_configuration.mk" || true
  echo
fi

# Fresh summary header.
echo "trace,ipc,instructions,cycles,L1D_access,L1D_hit,L1D_miss,L1D_hit_rate,L1D_miss_rate,L1D_MPKI,L2C_access,L2C_hit,L2C_miss,L2C_hit_rate,L2C_miss_rate,L2C_MPKI,LLC_access,LLC_hit,LLC_miss,LLC_hit_rate,LLC_miss_rate,LLC_MPKI,prefetch_requested,prefetch_issued,prefetch_useful,prefetch_useless,prefetch_accuracy" > "$SUMMARY"

run_with_heartbeat () {
  local log="$1"
  local trace_name="$2"
  shift 2
  "$@" > "$log" 2>&1 &
  local pid=$!
  local seconds=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    seconds=$((seconds + 30))
    local last
    last=$(tail -1 "$log" 2>/dev/null | head -c 160 || true)
    printf "  ...running %-20s elapsed=%ds last='%s'\n" "$trace_name" "$seconds" "$last"
  done
  wait "$pid"
}

for t in $TRACES; do
  tr_file="$TRACE_DIR/${t}.champsimtrace.xz"
  if [ ! -f "$tr_file" ]; then
    echo "[skip] missing trace: $tr_file"
    continue
  fi

  log="$LOG_DIR/${t}.spp_baseline.log"
  echo
  echo "[run] $t"
  echo "      log: $log"

  if ! run_with_heartbeat "$log" "$t" \
       "$CHAMP" \
       --warmup-instructions "$WARMUP" \
       --simulation-instructions "$SIM" \
       "$tr_file"; then
    echo "[fail] ChampSim failed for $t; see $log"
    continue
  fi

  python3 "$PARSER" --trace "$t" --log "$log" --append-csv "$SUMMARY"
  tail -1 "$SUMMARY"
done

echo
echo "[done] summary: $SUMMARY"
column -t -s, "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
