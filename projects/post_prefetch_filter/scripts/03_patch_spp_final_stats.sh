#!/usr/bin/env bash
# Patch ChampSim's built-in spp_dev so it prints explicit final prefetch counters.
#
# Why:
#   Your current aggregate cache stats have IPC/miss-rate/hit-rate, but prefetch_issued
#   and prefetch_useful are NA because default spp_dev::prefetcher_final_stats() is empty.
#   This patch makes it print one machine-readable line:
#
#     SPP_FINAL pf_issued=<N> pf_useful=<N> pf_useless=<N> pf_accuracy=<float>
#
# Usage from repo root:
#   bash projects/post_prefetch_filter/scripts/03_patch_spp_final_stats.sh
#   cd external/ChampSim
#   python3 ./config.sh _cfg/cfg_l2_spp.json
#   make -j8
#   cp bin/champsim bin/champsim.l2_spp

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
CHAMP="${CHAMPSIM_DIR:-$ROOT/external/ChampSim}"
CC="$CHAMP/prefetcher/spp_dev/spp_dev.cc"

if [ ! -f "$CC" ]; then
  echo "[error] cannot find $CC"
  exit 1
fi

cp "$CC" "$CC.bak.$(date +%s)"

python3 - <<'PY' "$CC"
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
s = path.read_text()

if '#include <iostream>' not in s:
    s = s.replace('#include "spp_dev.h"\n', '#include "spp_dev.h"\n#include <iostream>\n')

new_func = '''void spp_dev::prefetcher_final_stats()
{
  uint64_t issued = GHR.pf_issued;
  uint64_t useful = GHR.pf_useful;
  uint64_t useless = (issued >= useful) ? (issued - useful) : 0;
  double accuracy = (issued > 0) ? static_cast<double>(useful) / static_cast<double>(issued) : 0.0;

  std::cout << "SPP_FINAL"
            << " pf_requested=" << issued
            << " pf_issued=" << issued
            << " pf_useful=" << useful
            << " pf_useless=" << useless
            << " pf_accuracy=" << accuracy
            << std::endl;
}
'''

pattern = r'void\s+spp_dev::prefetcher_final_stats\s*\(\s*\)\s*\{\s*\}'
if re.search(pattern, s, flags=re.S):
    s = re.sub(pattern, new_func, s, count=1, flags=re.S)
elif 'SPP_FINAL' in s:
    print('[ok] SPP_FINAL already present')
else:
    raise SystemExit('[error] could not find empty spp_dev::prefetcher_final_stats() to replace')

path.write_text(s)
print('[patched]', path)
PY

grep -n "SPP_FINAL\|prefetcher_final_stats\|#include <iostream>" "$CC"
