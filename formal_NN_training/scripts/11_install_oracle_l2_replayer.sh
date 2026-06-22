#!/usr/bin/env bash
# Build the ROI-aligned ListReplayer from Angelawoo572/ChampSim without
# leaving a local source patch in the Pythia checkout.
#
# Pythia's build_champsim.sh selects an L2 frontend by copying
# prefetcher/<name>.l2c_pref into prefetcher/l2c_prefetcher.cc before make.
# We therefore generate a temporary oracle_replayer.l2c_pref from the tracked
# HEAD version of multi.l2c_pref, add ListReplayer to that temporary frontend,
# build it, and delete the generated frontend on exit.
#
# Runtime instantiation is verified by Script 09. Do not use `strings` as a
# build gate: optimized binaries do not reliably preserve this exact diagnostic
# literal even when the generated frontend compiled correctly.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHAMP_DIR=${CHAMP_DIR:-"$ROOT/external/ChampSim"}
GENERATED="$CHAMP_DIR/prefetcher/oracle_replayer.l2c_pref"
TRACKED_TEMPLATE="$CHAMP_DIR/prefetcher/.oracle_replayer_multi_head.tmp"
HEADER="$CHAMP_DIR/inc/list_replayer.h"
SOURCE="$CHAMP_DIR/prefetcher/list_replayer.cc"
BUILD_SCRIPT="$CHAMP_DIR/build_champsim.sh"
BUILT="$CHAMP_DIR/bin/perceptron-no-oracle_replayer-no-ship-1core"
OUT="$CHAMP_DIR/bin/champsim.oracle_l2_replayer"

cleanup() {
  rm -f "$GENERATED" "$TRACKED_TEMPLATE"
}
trap cleanup EXIT

[[ -d "$CHAMP_DIR/.git" ]] || {
  echo "[error] not a ChampSim git checkout: $CHAMP_DIR" >&2
  exit 2
}
[[ -x "$BUILD_SCRIPT" ]] || {
  echo "[error] missing executable Pythia build script: $BUILD_SCRIPT" >&2
  exit 2
}
[[ -f "$HEADER" ]] || {
  echo "[error] missing $HEADER. Update external/ChampSim first." >&2
  exit 2
}
[[ -f "$SOURCE" ]] || {
  echo "[error] missing $SOURCE. Update external/ChampSim first." >&2
  exit 2
}

# Use the committed Pythia multi registry even when the checkout has an old
# local ListReplayer patch. This keeps the generated build deterministic and
# avoids carrying that old patch forward.
git -C "$CHAMP_DIR" show HEAD:prefetcher/multi.l2c_pref > "$TRACKED_TEMPLATE"

python3 - "$TRACKED_TEMPLATE" "$GENERATED" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
out = Path(sys.argv[2])
s = src.read_text()

if '#include "list_replayer.h"' not in s:
    anchor = '#include "pref_power7.h"\n'
    if anchor not in s:
        raise SystemExit('[error] cannot find include anchor in tracked multi.l2c_pref')
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
        raise SystemExit('[error] cannot find next_line registry anchor in tracked multi.l2c_pref')
    s = s.replace(anchor, insert, 1)

out.write_text(s)
print('[generated]', out)
PY

# Verify exactly the source that will be copied by build_champsim.sh.
grep -Fq '#include "list_replayer.h"' "$GENERATED" || {
  echo "[error] generated frontend lacks list_replayer header" >&2
  exit 3
}
grep -Fq 'compare("list_replayer")' "$GENERATED" || {
  echo "[error] generated frontend lacks list_replayer registry branch" >&2
  exit 3
}

echo "[build] compiling Pythia with L1D=no, L2=temporary oracle_replayer, LLC=no, cores=1"
(
  cd "$CHAMP_DIR"
  ./build_champsim.sh no oracle_replayer no 1
)

[[ -x "$BUILT" ]] || {
  echo "[error] expected Pythia binary not produced: $BUILT" >&2
  exit 4
}
cp -f "$BUILT" "$OUT"

# Keep a tiny provenance record next to the named replay binary.
{
  echo "built_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "champsim_head=$(git -C "$CHAMP_DIR" rev-parse HEAD)"
  echo "frontend=tracked HEAD:prefetcher/multi.l2c_pref + ListReplayer"
  echo "build_command=./build_champsim.sh no oracle_replayer no 1"
} > "$OUT.build_info.txt"

echo "[ok] built $OUT"
echo "[ok] frontend source was verified before compilation"
echo "[ok] runtime instantiation will be verified by Script 09"
echo "[next] run formal_NN_training/scripts/09_run_oracle_replacer_replay_parallel.sh"