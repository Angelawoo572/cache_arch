#!/usr/bin/env bash
# Restore local ChampSim files modified by the paused post_prefetch_filter experiment.
#
# This does not change the GitHub repo history. It only cleans the local
# external/ChampSim working tree so future projects do not accidentally run with
# the SPP candidate logger / SPP_FINAL patch still applied.
#
# Usage:
#   bash projects/post_prefetch_filter/scripts/00_restore_local_champsim.sh
#
# Optional cleanup of copied experiment binaries:
#   CLEAN_BIN=1 bash projects/post_prefetch_filter/scripts/00_restore_local_champsim.sh

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

CHAMP="$ROOT/external/ChampSim"
SPP_H="prefetcher/spp_dev/spp_dev.h"
SPP_CC="prefetcher/spp_dev/spp_dev.cc"

if [ ! -d "$CHAMP/.git" ]; then
  echo "[error] external/ChampSim is not a git checkout: $CHAMP"
  exit 1
fi

if [ ! -f "$CHAMP/$SPP_H" ] || [ ! -f "$CHAMP/$SPP_CC" ]; then
  echo "[error] cannot find spp_dev files under external/ChampSim"
  exit 1
fi

echo "[before] local modifications touching spp_dev:"
git -C "$CHAMP" status --short "$SPP_H" "$SPP_CC" || true

echo "[restore] checkout original spp_dev.h/spp_dev.cc from external/ChampSim HEAD"
git -C "$CHAMP" checkout -- "$SPP_H" "$SPP_CC"

echo "[after] local modifications touching spp_dev:"
git -C "$CHAMP" status --short "$SPP_H" "$SPP_CC" || true

if [ "${CLEAN_BIN:-0}" = "1" ]; then
  echo "[clean bin] removing copied post-prefetch experiment binaries if present"
  rm -f "$CHAMP/bin/champsim.l2_spp_cand" "$CHAMP/bin/champsim.l2_spp_stats" "$CHAMP/bin/champsim.l2_spp_filter" || true
fi

echo "[done] local ChampSim spp_dev restored."
echo "[note] If you later need a clean full ChampSim build, run make/config from external/ChampSim as usual."
