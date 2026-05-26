#!/usr/bin/env bash
# rebuild_replayer_clean.sh
# Rebuild only bin/champsim.replayer after changing list_replayer.
# This avoids stale ChampSim generated config dependencies from old experiments.

set -uo pipefail

WORKDIR="$(pwd)"
CHAMP="$WORKDIR/external/ChampSim"

if [ ! -d "$CHAMP" ]; then
  echo "[error] $CHAMP not found"
  exit 1
fi

mkdir -p "$CHAMP/prefetcher/list_replayer"
cp "$WORKDIR/champsim_modules/list_replayer/list_replayer.h" "$CHAMP/prefetcher/list_replayer/list_replayer.h"
cp "$WORKDIR/champsim_modules/list_replayer/list_replayer.cc" "$CHAMP/prefetcher/list_replayer/list_replayer.cc"

echo "[copy] installed updated list_replayer module"

cd "$CHAMP" || exit 1
mkdir -p _cfg
cat > projects/legacy_gru_prefetch/_cfg/cfg_replayer.json <<'JSON'
{
  "ooo_cpu": [{ "L1D": { "prefetcher": "list_replayer" } }],
  "LLC":     { "replacement": "lru" }
}
JSON

# Move stale generated config away instead of deleting it.
if [ -d .csconfig ]; then
  backup=".csconfig.stale.$(date +%s)"
  echo "[clean] moving stale .csconfig -> $backup"
  mv .csconfig "$backup"
fi

rm -f bin/champsim

echo "[config] cfg_replayer.json"
python3 ./config.sh projects/legacy_gru_prefetch/_cfg/cfg_replayer.json > /tmp/config_replayer_clean.log 2>&1 || {
  echo "[error] config failed. Last 60 lines:"
  tail -60 /tmp/config_replayer_clean.log
  exit 2
}

echo "[build] replayer"
make -j8 > /tmp/build_replayer_clean.log 2>&1 || {
  echo "[error] make failed. Last 80 lines:"
  tail -80 /tmp/build_replayer_clean.log
  exit 3
}

if [ ! -x bin/champsim ]; then
  echo "[error] bin/champsim was not produced"
  exit 4
fi

cp bin/champsim bin/champsim.replayer
ls -l bin/champsim.replayer

echo "[done] rebuilt bin/champsim.replayer"
