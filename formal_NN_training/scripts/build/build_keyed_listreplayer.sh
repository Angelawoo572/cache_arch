#!/usr/bin/env bash
# Build the PC-line-occurrence ListReplayer without editing tracked Pythia.
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
GENERATED="$CHAMP_DIR/prefetcher/standalone_nn_replayer.l2c_pref"
TEMPLATE="$CHAMP_DIR/prefetcher/.standalone_nn_replayer.tmp"
HEADER="$CHAMP_DIR/inc/list_replayer.h"
SOURCE="$CHAMP_DIR/prefetcher/list_replayer.cc"
BUILT="$CHAMP_DIR/bin/perceptron-no-standalone_nn_replayer-no-ship-1core"
OUT="$CHAMP_DIR/bin/champsim.standalone_nn_replayer"
cleanup() { rm -f "$GENERATED" "$TEMPLATE"; }
trap cleanup EXIT

[[ -d "$CHAMP_DIR/.git" ]] || { echo "[error] not a ChampSim checkout: $CHAMP_DIR" >&2; exit 2; }
[[ -f "$HEADER" && -f "$SOURCE" ]] || { echo "[error] missing ListReplayer source/header" >&2; exit 2; }
for marker in 'PC-line-occ triggers' 'key=pc_line_occ' 'occurrences_'; do
  grep -Fq "$marker" "$SOURCE" || { echo "[error] stale ListReplayer source" >&2; exit 3; }
done
grep -Fq 'TriggerKey' "$HEADER" || { echo "[error] stale ListReplayer header" >&2; exit 3; }

git -C "$CHAMP_DIR" show HEAD:prefetcher/multi.l2c_pref > "$TEMPLATE"
python3 - "$TEMPLATE" "$GENERATED" <<'PY'
from pathlib import Path
import sys
src, out = map(Path, sys.argv[1:])
text = src.read_text()
if '#include "list_replayer.h"' not in text:
    anchor = '#include "pref_power7.h"\n'
    if anchor not in text:
        raise SystemExit('[error] include anchor absent')
    text = text.replace(anchor, anchor + '#include "list_replayer.h"\n', 1)
if 'compare("list_replayer")' not in text:
    anchor = '''\t\telse if(!knob::l2c_prefetcher_types[index].compare("next_line"))
\t\t{
\t\t\tcout << "adding L2C_PREFETCHER: next_line" << endl;
\t\t\tNextLinePrefetcher *pref_nl = new NextLinePrefetcher(knob::l2c_prefetcher_types[index]);
\t\t\tprefetchers.push_back(pref_nl);
\t\t}
'''
    insert = anchor + '''\t\telse if(!knob::l2c_prefetcher_types[index].compare("list_replayer"))
\t\t{
\t\t\tcout << "adding L2C_PREFETCHER: list_replayer (PC-line-occ trigger domain)" << endl;
\t\t\tListReplayer *pref_list = new ListReplayer(knob::l2c_prefetcher_types[index], this);
\t\t\tprefetchers.push_back(pref_list);
\t\t}
'''
    if anchor not in text:
        raise SystemExit('[error] next_line registry anchor absent')
    text = text.replace(anchor, insert, 1)
out.write_text(text)
PY

grep -Fq '#include "list_replayer.h"' "$GENERATED"
grep -Fq 'compare("list_replayer")' "$GENERATED"
( cd "$CHAMP_DIR" && ./build_champsim.sh no standalone_nn_replayer no 1 )
[[ -x "$BUILT" ]] || { echo "[error] expected binary missing: $BUILT" >&2; exit 4; }
cp -f "$BUILT" "$OUT"
{
  echo "built_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "champsim_head=$(git -C "$CHAMP_DIR" rev-parse HEAD)"
  echo "list_replayer_source_sha256=$(sha256sum "$SOURCE" | awk '{print $1}')"
  echo "list_replayer_header_sha256=$(sha256sum "$HEADER" | awk '{print $1}')"
  echo "replay_key=pc_line_occ"
} > "$OUT.build_info.txt"
echo "[ok] built $OUT"
