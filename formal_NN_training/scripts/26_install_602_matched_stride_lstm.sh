#!/usr/bin/env bash
# Build one ChampSim binary that can run no_pref, stride, or the live matched
# 602 LSTM.  Runtime sources are copied into the ChampSim checkout only for the
# build and removed afterward; tracked submodule files are not overwritten.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
RUNTIME_DIR="$ROOT/formal_NN_training/LSTM/runtime"
SOURCE_IN="$RUNTIME_DIR/matched_stride_lstm.cc"
HEADER_IN="$RUNTIME_DIR/matched_stride_lstm.h"
SOURCE_OUT="$CHAMP_DIR/prefetcher/matched_stride_lstm.cc"
HEADER_OUT="$CHAMP_DIR/inc/matched_stride_lstm.h"
FRONTEND="$CHAMP_DIR/prefetcher/matched_stride_live.l2c_pref"
TEMPLATE="$CHAMP_DIR/prefetcher/.matched_stride_live.tmp"
BUILT="$CHAMP_DIR/bin/perceptron-no-matched_stride_live-no-ship-1core"
OUT="${OUT:-$CHAMP_DIR/bin/champsim.602_matched_stride_lstm}"

cleanup() {
  rm -f "$SOURCE_OUT" "$HEADER_OUT" "$FRONTEND" "$TEMPLATE"
}
trap cleanup EXIT

[[ -d "$CHAMP_DIR/.git" ]] || { echo "[error] not a ChampSim checkout: $CHAMP_DIR" >&2; exit 2; }
[[ -f "$SOURCE_IN" && -f "$HEADER_IN" ]] || { echo "[error] matched LSTM runtime sources are missing" >&2; exit 2; }
for path in "$SOURCE_OUT" "$HEADER_OUT" "$FRONTEND" "$TEMPLATE"; do
  [[ ! -e "$path" ]] || { echo "[error] refusing to overwrite existing $path" >&2; exit 2; }
done
[[ -f "$CHAMP_DIR/libbf/build/lib/libbf.a" ]] || { echo "[error] build libbf before installing the matched LSTM" >&2; exit 2; }

cp "$SOURCE_IN" "$SOURCE_OUT"
cp "$HEADER_IN" "$HEADER_OUT"
git -C "$CHAMP_DIR" show HEAD:prefetcher/multi.l2c_pref > "$TEMPLATE"

python3 - "$TEMPLATE" "$FRONTEND" <<'PY'
from __future__ import print_function
import sys
from pathlib import Path

source, output = map(Path, sys.argv[1:])
text = source.read_text()
include_anchor = '#include "pref_power7.h"\n'
if include_anchor not in text:
    raise SystemExit("[error] multi.l2c_pref include anchor is absent")
text = text.replace(
    include_anchor,
    include_anchor + '#include "matched_stride_lstm.h"\n',
    1,
)
stride_block = '''\t\telse if(!knob::l2c_prefetcher_types[index].compare("stride"))
\t\t{
\t\t\tcout << "adding L2C_PREFETCHER: Stride" << endl;
\t\t\tStridePrefetcher *pref_stride = new StridePrefetcher(knob::l2c_prefetcher_types[index]);
\t\t\tprefetchers.push_back(pref_stride);
\t\t}
'''
if stride_block not in text:
    raise SystemExit("[error] multi.l2c_pref stride registry anchor is absent")
matched_block = stride_block + '''\t\telse if(!knob::l2c_prefetcher_types[index].compare("matched_stride_lstm"))
\t\t{
\t\t\tcout << "adding L2C_PREFETCHER: matched_stride_lstm (live inference)" << endl;
\t\t\tMatchedStrideLSTM *pref_matched = new MatchedStrideLSTM(knob::l2c_prefetcher_types[index]);
\t\t\tprefetchers.push_back(pref_matched);
\t\t}
'''
text = text.replace(stride_block, matched_block, 1)
output.write_text(text)
PY

grep -Fq '#include "matched_stride_lstm.h"' "$FRONTEND" || { echo "[error] generated frontend lacks include" >&2; exit 3; }
grep -Fq 'compare("matched_stride_lstm")' "$FRONTEND" || { echo "[error] generated frontend lacks registry" >&2; exit 3; }

( cd "$CHAMP_DIR" && ./build_champsim.sh no matched_stride_live no 1 )
[[ -x "$BUILT" ]] || { echo "[error] expected binary missing: $BUILT" >&2; exit 4; }
cp -f "$BUILT" "$OUT"
{
  echo "built_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "champsim_head=$(git -C "$CHAMP_DIR" rev-parse HEAD)"
  echo "runtime_source_sha256=$(sha256sum "$SOURCE_IN" | awk '{print $1}')"
  echo "runtime_header_sha256=$(sha256sum "$HEADER_IN" | awk '{print $1}')"
  echo "frontend=tracked_HEAD_multi_plus_live_matched_stride_lstm"
  echo "supported_primary_methods=none,stride,matched_stride_lstm"
} > "$OUT.build_info.txt"
echo "[ok] built $OUT"
