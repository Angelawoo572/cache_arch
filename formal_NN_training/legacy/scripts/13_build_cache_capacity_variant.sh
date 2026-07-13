#!/usr/bin/env bash
# Build one reversible cache-capacity variant. The original cache.h is restored.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
LEVEL="${LEVEL:?set LEVEL=L1D, L2C, or LLC}"
SETS="${SETS:?set SETS to a positive power of two}"
WAYS="${WAYS:?set WAYS to a positive integer}"
FRONTEND="${FRONTEND:-normal}" # normal or replayer
PATCH_LOGGER="${PATCH_LOGGER:-0}"
RESET_PATCH="${RESET_PATCH:-0}"
OUT_DIR="${OUT_DIR:-$CHAMP_DIR/bin/capacity_variants}"
CACHE_H="$CHAMP_DIR/inc/cache.h"
PATCH="$ROOT/formal_NN_training/scripts/02_patch_pythia_demand_logger.sh"
REPLAYER_BUILD="$ROOT/formal_NN_training/scripts/06_install_keyed_listreplayer.sh"

mkdir -p "$OUT_DIR"
[[ -d "$CHAMP_DIR/.git" && -f "$CACHE_H" ]] || { echo "[error] missing ChampSim/cache.h" >&2; exit 2; }
case "$LEVEL" in
  L1D|L2C|LLC) ;;
  *) echo "[error] LEVEL must be L1D, L2C, or LLC" >&2; exit 2 ;;
esac
case "$FRONTEND" in
  normal|replayer) ;;
  *) echo "[error] FRONTEND must be normal or replayer" >&2; exit 2 ;;
esac
[[ "$SETS" =~ ^[0-9]+$ && "$WAYS" =~ ^[0-9]+$ && "$SETS" -gt 0 && "$WAYS" -gt 0 ]] || {
  echo "[error] invalid SETS/WAYS" >&2; exit 2; }
(( (SETS & (SETS - 1)) == 0 )) || { echo "[error] SETS must be a power of two" >&2; exit 2; }

if [[ "$PATCH_LOGGER" == 1 ]]; then
  RESET_PATCH="$RESET_PATCH" CHAMP_DIR="$CHAMP_DIR" bash "$PATCH"
fi

backup="$(mktemp "${TMPDIR:-/tmp}/cache.h.XXXXXX")"
cp "$CACHE_H" "$backup"
restore() { cp "$backup" "$CACHE_H"; rm -f "$backup"; }
trap restore EXIT

python3 - "$CACHE_H" "$LEVEL" "$SETS" "$WAYS" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
level, sets, ways = sys.argv[2:]
text = path.read_text()
for macro, value in ((f"{level}_SET", sets), (f"{level}_WAY", ways)):
    text, count = re.subn(
        rf"(?m)^#define\s+{re.escape(macro)}\s+.+$",
        f"#define {macro} {value}",
        text,
        count=1,
    )
    if count != 1:
        raise SystemExit(f"could not replace {macro}")
path.write_text(text)
PY

capacity_bytes=$((SETS * WAYS * 64))
tag="${LEVEL,,}_${SETS}set_${WAYS}way_$((capacity_bytes / 1024))KiB"
if [[ "$FRONTEND" == normal ]]; then
  ( cd "$CHAMP_DIR" && bash ./build_champsim.sh no multi no 1 )
  source_bin="$CHAMP_DIR/bin/perceptron-no-multi-no-ship-1core"
else
  CHAMP_DIR="$CHAMP_DIR" bash "$REPLAYER_BUILD"
  source_bin="$CHAMP_DIR/bin/champsim.standalone_nn_replayer"
fi
[[ -x "$source_bin" ]] || { echo "[error] no built binary: $source_bin" >&2; exit 3; }

out="$OUT_DIR/champsim.${tag}.${FRONTEND}"
cp -f "$source_bin" "$out"
{
  echo "built_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "champsim_head=$(git -C "$CHAMP_DIR" rev-parse HEAD)"
  echo "level=$LEVEL"
  echo "sets=$SETS"
  echo "ways=$WAYS"
  echo "line_bytes=64"
  echo "capacity_bytes=$capacity_bytes"
  echo "frontend=$FRONTEND"
  echo "source_binary=$source_bin"
} > "$out.build_info.txt"
echo "[ok] $out"
