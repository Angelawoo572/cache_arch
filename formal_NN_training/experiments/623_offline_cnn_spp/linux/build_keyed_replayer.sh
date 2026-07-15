#!/usr/bin/env bash
# Build SPP plus a PC-line-occ replayer for normal or direct-NN actions.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/formal_NN_training/experiments/623_offline_cnn_spp"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
GENERATED="$CHAMP_DIR/prefetcher/offline_623_direct_spp_replayer.l2c_pref"
TEMPLATE="$CHAMP_DIR/prefetcher/.offline_623_direct_spp_replayer.tmp"
HEADER="$CHAMP_DIR/inc/list_replayer_fill.h"
SOURCE="$CHAMP_DIR/prefetcher/list_replayer_fill.cc"
BUILT="$CHAMP_DIR/bin/perceptron-no-offline_623_direct_spp_replayer-no-ship-1core"
OUT="${OUT:-$CHAMP_DIR/bin/champsim.623_direct_spp_replay}"

cleanup() {
  rm -f "$GENERATED" "$TEMPLATE" "$HEADER" "$SOURCE"
}
trap cleanup EXIT

git -C "$CHAMP_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "[error] not a ChampSim checkout: $CHAMP_DIR" >&2
  exit 2
}
if [[ -e "$HEADER" || -e "$SOURCE" ]]; then
  if [[ -f "$HEADER" && -f "$SOURCE" ]] \
      && cmp -s "$EXP/cpp/list_replayer_fill.h" "$HEADER" \
      && cmp -s "$EXP/cpp/list_replayer_fill.cc" "$SOURCE"; then
    echo "[cleanup] removing identical files left by an interrupted 623 SPP build"
    rm -f "$HEADER" "$SOURCE"
  else
    echo "[error] temporary fill-replayer path has foreign content; refusing to overwrite it" >&2
    exit 2
  fi
fi
cp "$EXP/cpp/list_replayer_fill.h" "$HEADER"
cp "$EXP/cpp/list_replayer_fill.cc" "$SOURCE"
grep -Fq 'captured_fill_level' "$SOURCE" || {
  echo "[error] stale fill-preserving replayer source" >&2
  exit 3
}
grep -Fq 'struct Action' "$HEADER" || {
  echo "[error] stale fill-preserving replayer header" >&2
  exit 3
}

git -C "$CHAMP_DIR" show HEAD:prefetcher/multi.l2c_pref > "$TEMPLATE"
python3 - "$TEMPLATE" "$GENERATED" <<'PY'
from pathlib import Path
import sys

source, output = map(Path, sys.argv[1:])
text = source.read_text()
include = '#include "list_replayer_fill.h"\n'
if include not in text:
    anchor = '#include "pref_power7.h"\n'
    if anchor not in text:
        raise SystemExit("[error] include anchor absent")
    text = text.replace(anchor, anchor + include, 1)
if 'compare("list_replayer_fill")' not in text:
    anchor = '''\t\telse if(!knob::l2c_prefetcher_types[index].compare("next_line"))
\t\t{
\t\t\tcout << "adding L2C_PREFETCHER: next_line" << endl;
\t\t\tNextLinePrefetcher *pref_nl = new NextLinePrefetcher(knob::l2c_prefetcher_types[index]);
\t\t\tprefetchers.push_back(pref_nl);
\t\t}
'''
    insert = anchor + '''\t\telse if(!knob::l2c_prefetcher_types[index].compare("list_replayer_fill"))
\t\t{
\t\t\tcout << "adding L2C_PREFETCHER: list_replayer_fill (PC-line-occ plus captured fill level)" << endl;
\t\t\tListReplayerFill *pref_list = new ListReplayerFill(knob::l2c_prefetcher_types[index], this);
\t\t\tprefetchers.push_back(pref_list);
\t\t}
'''
    if anchor not in text:
        raise SystemExit("[error] registry anchor absent")
    text = text.replace(anchor, insert, 1)
output.write_text(text)
PY

( cd "$CHAMP_DIR" && ./build_champsim.sh no offline_623_direct_spp_replayer no 1 )
[[ -x "$BUILT" ]] || { echo "[error] expected binary missing: $BUILT" >&2; exit 4; }
cp -f "$BUILT" "$OUT"
echo "[ok] built fill-preserving direct-SPP replayer $OUT"
