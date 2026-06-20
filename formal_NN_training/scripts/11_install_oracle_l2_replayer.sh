#!/usr/bin/env bash
# Install/build the ROI-aligned ListReplayer from Angelawoo572/ChampSim.
#
# It patches only the L2 registry (prefetcher/multi.l2c_pref) in the local Pythia
# checkout. The implementation itself is tracked in the ChampSim fork:
#   inc/list_replayer.h
#   prefetcher/list_replayer.cc
#
# The replayer counts only post-warmup L2 LOAD callbacks, exactly matching the
# oracle table's demand_idx domain. It also rejects the rich notebook CSV; feed
# it only a strict idx,0xprefetch_addr list produced by script 10 / notebook v2.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHAMP_DIR=${CHAMP_DIR:-"$ROOT/external/ChampSim"}
BUILD_JOBS=${BUILD_JOBS:-8}
MULTI="$CHAMP_DIR/prefetcher/multi.l2c_pref"
HEADER="$CHAMP_DIR/inc/list_replayer.h"
SOURCE="$CHAMP_DIR/prefetcher/list_replayer.cc"

[[ -d "$CHAMP_DIR/.git" ]] || { echo "[error] not a ChampSim git checkout: $CHAMP_DIR" >&2; exit 2; }
[[ -f "$MULTI" ]] || { echo "[error] missing Pythia L2 registry: $MULTI" >&2; exit 2; }
[[ -f "$HEADER" ]] || { echo "[error] missing $HEADER. Run: git -C $CHAMP_DIR pull" >&2; exit 2; }
[[ -f "$SOURCE" ]] || { echo "[error] missing $SOURCE. Run: git -C $CHAMP_DIR pull" >&2; exit 2; }

python3 - "$MULTI" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text()

if '#include "list_replayer.h"' not in s:
    anchor = '#include "pref_power7.h"\n'
    if anchor not in s:
        raise SystemExit('[error] cannot find include anchor in multi.l2c_pref')
    s = s.replace(anchor, anchor + '#include "list_replayer.h"\n', 1)

if 'compare("list_replayer")' not in s:
    anchor = '''\t\telse if(!knob::l2c_prefetcher_types[index].compare("next_line"))
\t\t{
\t\t\tcout << "adding L2C_PREFETCHER: next_line" << endl;
\t\t\tNextLinePrefetcher *pref_nl = new NextLinePrefetcher(knob::l2c_prefetcher_types[index]);
\t\t\tprefetchers.push_back(pref_nl);
\t\t}
'''
    insert = anchor + '''\t\telse if(!knob::l2c_prefetcher_types[index].compare("list_replayer"))
\t\t{
\t\t\tcout << "adding L2C_PREFETCHER: list_replayer (ROI L2 LOAD index domain)" << endl;
\t\t\tListReplayer *pref_list = new ListReplayer(knob::l2c_prefetcher_types[index], this);
\t\t\tprefetchers.push_back(pref_list);
\t\t}
'''
    if anchor not in s:
        raise SystemExit('[error] cannot find next_line registry anchor in multi.l2c_pref')
    s = s.replace(anchor, insert, 1)

p.write_text(s)
print('[patched]', p)
PY

# Pythia's Makefile compiles src/, branch/, replacement/, and prefetcher/.
# The selected L2 prefetcher is a runtime knob; runner script 09 supplies
# --l2c_prefetcher_types=list_replayer.
echo "[build] $CHAMP_DIR (jobs=$BUILD_JOBS)"
make -C "$CHAMP_DIR" clean
make -C "$CHAMP_DIR" -j"$BUILD_JOBS"

[[ -x "$CHAMP_DIR/bin/champsim" ]] || { echo "[error] build did not produce bin/champsim" >&2; exit 3; }
cp "$CHAMP_DIR/bin/champsim" "$CHAMP_DIR/bin/champsim.oracle_l2_replayer"
echo "[ok] built $CHAMP_DIR/bin/champsim.oracle_l2_replayer"
echo "[next] use BIN=$CHAMP_DIR/bin/champsim.oracle_l2_replayer with script 09"
