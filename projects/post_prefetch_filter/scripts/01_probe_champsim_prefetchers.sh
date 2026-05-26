#!/usr/bin/env bash
# Probe whether the local ChampSim checkout has spp_dev and other prefetchers.
#
# Usage from repo root:
#   bash projects/post_prefetch_filter/scripts/01_probe_champsim_prefetchers.sh
#
# Optional:
#   CHAMPSIM_DIR=/path/to/ChampSim bash projects/post_prefetch_filter/scripts/01_probe_champsim_prefetchers.sh

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

CANDIDATES=()
if [ -n "${CHAMPSIM_DIR:-}" ]; then
  CANDIDATES+=("$CHAMPSIM_DIR")
fi
CANDIDATES+=(
  "$ROOT/ChampSim"
  "$ROOT/ChampSim-ML"
  "$ROOT/external/ChampSim"
  "$ROOT/external/ChampSim-ML"
)

echo "============================================================"
echo "ChampSim prefetcher probe"
echo "repo root: $ROOT"
echo "============================================================"

FOUND_ROOT=""
for d in "${CANDIDATES[@]}"; do
  if [ -d "$d" ]; then
    FOUND_ROOT="$d"
    break
  fi
done

if [ -z "$FOUND_ROOT" ]; then
  echo "[error] No ChampSim checkout found. Tried:"
  printf '  - %s\n' "${CANDIDATES[@]}"
  echo
  echo "Set CHAMPSIM_DIR explicitly, for example:"
  echo "  CHAMPSIM_DIR=/scratch/qianruw/cache/ChampSim bash projects/post_prefetch_filter/scripts/01_probe_champsim_prefetchers.sh"
  exit 1
fi

echo "[found] ChampSim root: $FOUND_ROOT"
echo

echo "[prefetcher directories]"
if [ -d "$FOUND_ROOT/prefetcher" ]; then
  find "$FOUND_ROOT/prefetcher" -maxdepth 2 -type d | sort | sed "s#^$FOUND_ROOT/##"
else
  echo "  no prefetcher/ directory found"
fi

echo
echo "[search for spp_dev]"
if find "$FOUND_ROOT" -maxdepth 5 \( -iname '*spp*' -o -iname '*signature*' \) -print | sort | sed "s#^$FOUND_ROOT/##" | grep -q .; then
  find "$FOUND_ROOT" -maxdepth 5 \( -iname '*spp*' -o -iname '*signature*' \) -print | sort | sed "s#^$FOUND_ROOT/##"
else
  echo "  no spp/signature-path files found"
fi

echo
echo "[grep spp_dev references]"
if command -v rg >/dev/null 2>&1; then
  rg -n "spp_dev|SPP|signature path|signature_path" "$FOUND_ROOT" || true
else
  grep -RIn "spp_dev\|SPP\|signature path\|signature_path" "$FOUND_ROOT" 2>/dev/null || true
fi

echo
echo "[quick verdict]"
if [ -d "$FOUND_ROOT/prefetcher/spp_dev" ] || find "$FOUND_ROOT/prefetcher" -maxdepth 3 -iname '*spp*' 2>/dev/null | grep -q .; then
  echo "spp_dev / SPP-like prefetcher appears to exist. Use it as the first candidate generator."
else
  echo "No obvious spp_dev directory found. Use next_line/ip_stride first, or import spp_dev/Berti later."
fi
