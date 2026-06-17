#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

CHAMP_DIR="$ROOT/external/ChampSim"
CFG_DIR="$ROOT/formal_NN_training/_cfg/capacity_sweep"
mkdir -p "$CFG_DIR"

JOBS="${JOBS:-8}"

CAPS=("256K:512" "512K:1024" "1M:2048" "2M:4096")

build_one () {
  local tag="$1"
  local sets="$2"
  local kind="$3"
  local pf="$4"
  local outbin="$CHAMP_DIR/bin/champsim.${kind}.L2_${tag}"
  local cfg="$CFG_DIR/cfg_${kind}_L2_${tag}.json"

  if [ "$pf" = "none" ]; then
    cat > "$cfg" <<JSON
{
  "ooo_cpu": [
    {
      "L2C": {
        "sets": $sets,
        "ways": 8
      }
    }
  ],
  "LLC": { "replacement": "lru" }
}
JSON
  else
    cat > "$cfg" <<JSON
{
  "ooo_cpu": [
    {
      "L2C": {
        "sets": $sets,
        "ways": 8,
        "prefetcher": "$pf"
      }
    }
  ],
  "LLC": { "replacement": "lru" }
}
JSON
  fi

  echo "============================================================"
  echo "[build] kind=$kind tag=$tag sets=$sets pf=$pf"
  echo "cfg=$cfg"
  echo "out=$outbin"
  echo "============================================================"

  cd "$CHAMP_DIR"
  rm -f bin/champsim
  python3 ./config.sh "$cfg"
  make -j"$JOBS"
  cp bin/champsim "$outbin"
  cd "$ROOT"

  if [ ! -x "$outbin" ]; then
    echo "[error] missing $outbin"
    exit 1
  fi
}

for item in "${CAPS[@]}"; do
  tag="${item%%:*}"
  sets="${item#*:}"

  build_one "$tag" "$sets" "baseline" "none"
  build_one "$tag" "$sets" "spp" "spp_dev"
  build_one "$tag" "$sets" "replayer" "list_replayer"
done

echo "[done] capacity binaries:"
ls -lh "$CHAMP_DIR"/bin/champsim.*.L2_*
