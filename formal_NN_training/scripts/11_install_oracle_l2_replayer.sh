#!/usr/bin/env bash
# Install/build the ROI-aligned ListReplayer from Angelawoo572/ChampSim.
#
# It patches the Pythia multi-L2 registry (prefetcher/multi.l2c_pref). The
# implementation itself is tracked in the ChampSim fork:
#   inc/list_replayer.h
#   prefetcher/list_replayer.cc
#
# CRITICAL: Pythia selects the compiled L2 frontend with build_champsim.sh.
# Running plain `make` compiles whichever l2c_prefetcher.cc happened to be
# left in the checkout (usually no.l2c_pref after a previous build), so runtime
# --l2c_prefetcher_types=list_replayer can never instantiate ListReplayer.
# We therefore build exactly: no / multi / no / 1 core.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHAMP_DIR=${CHAMP_DIR:-"$ROOT/external/ChampSim"}
BUILD_JOBS=${BUILD_JOBS:-8}
MULTI="$CHAMP_DIR/prefetcher/multi.l2c_pref"
HEADER="$CHAMP_DIR/inc/list_replayer.h"
SOURCE="$CHAMP_DIR/prefetcher/list_replayer.cc"
BUILD_SCRIPT="$CHAMP_DIR/build_champsim.sh"
BUILT="$CHAMP_DIR/bin/perceptron-no-multi-no-ship-1core"
OUT="$CHAMP_DIR/bin/champsim.oracle_l2_replayer"

[[ -d "$CHAMP_DIR/.git" ]] || { echo "[error] not a ChampSim git checkout: $CHAMP_DIR" >&2; exit 2; }
[[ -x "$BUILD_SCRIPT" ]] || { echo "[error] missing executable Pythia build script: $BUILD_SCRIPT" >&2; exit 2; }
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

echo "[build] compiling Pythia with L1D=no, L2=multi, LLC=no, cores=1"
(
  cd "$CHAMP_DIR"
  # build_champsim.sh itself runs make clean/make and creates the named binary.
  # BUILD_JOBS is informational only; this upstream script does not forward -j.
  ./build_champsim.sh no multi no 1
)

[[ -x "$BUILT" ]] || {
  echo "[error] expected Pythia multi binary not produced: $BUILT" >&2
  echo "[hint] inspect the build output above; build_champsim.sh must report L2C Prefetcher: multi" >&2
  exit 3
}
cp -f "$BUILT" "$OUT"

# Static, pre-run sanity: this confirms the multi frontend was compiled into the
# named binary. Runtime script 09 still checks the actual instantiation line.
if ! strings "$OUT" | grep -q 'adding L2C_PREFETCHER: list_replayer'; then
  echo "[error] $OUT does not contain list_replayer registry text; refusing invalid replay binary" >&2
  exit 4
fi

echo "[ok] built $OUT"
echo "[ok] active source frontend: $CHAMP_DIR/prefetcher/l2c_prefetcher.cc (copied from multi.l2c_pref during build)"
echo "[next] use BIN=$OUT with script 09"
