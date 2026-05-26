#!/usr/bin/env bash
# Patch local ChampSim spp_dev to dump per-candidate SPP events for RL-filter training.
#
# This creates a CSV log controlled by SPP_CAND_LOG:
#   SPP_CAND_LOG=projects/post_prefetch_filter/data/generated/spp_candidate_events.csv \
#     external/ChampSim/bin/champsim.l2_spp_cand ...
#
# CSV events:
#   CAND  = SPP candidate accepted by SPP's internal filter and sent to prefetch_line()
#   USE   = later demand access marked useful_prefetch by ChampSim
#
# First version logs MSHR exactly. PQ occupancy is written as 0 because this
# ChampSim version keeps CACHE::internal_PQ private to modules. We will add PQ
# later by exposing a public getter or logging from cache.cc.

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

SPP_H="$ROOT/external/ChampSim/prefetcher/spp_dev/spp_dev.h"
SPP_CC="$ROOT/external/ChampSim/prefetcher/spp_dev/spp_dev.cc"

if [ ! -f "$SPP_H" ] || [ ! -f "$SPP_CC" ]; then
  echo "[error] Cannot find spp_dev files:"
  echo "  $SPP_H"
  echo "  $SPP_CC"
  exit 1
fi

if grep -q "SPP_CAND_LOG" "$SPP_CC"; then
  echo "[ok] candidate logger patch already appears present"
  echo "     If make fails with private internal_PQ, run:"
  echo "       bash projects/post_prefetch_filter/scripts/04b_fix_spp_candidate_logger_compile.sh"
  grep -n "SPP_CAND_LOG\|event,cand_id" "$SPP_CC" || true
  exit 0
fi

cp "$SPP_H" "$SPP_H.bak.$(date +%Y%m%d_%H%M%S)"
cp "$SPP_CC" "$SPP_CC.bak.$(date +%Y%m%d_%H%M%S)"

python3 - "$SPP_H" "$SPP_CC" <<'PY'
import re
import sys
from pathlib import Path

h_path = Path(sys.argv[1])
cc_path = Path(sys.argv[2])

h = h_path.read_text()
cc = cc_path.read_text()

# ---------- header patch ----------
if '#include <fstream>' not in h:
    h = h.replace('#include <vector>\n', '#include <vector>\n#include <fstream>\n')

member_block = '''\n  // Post-prefetch-filter experiment logger.\n  // Opened only when SPP_CAND_LOG is set.\n  std::ofstream cand_log_;\n  uint64_t cand_id_ = 0;\n'''

if 'cand_log_' not in h:
    h = h.replace('  GLOBAL_REGISTER GHR;\n', '  GLOBAL_REGISTER GHR;\n' + member_block)

# ---------- cc includes ----------
if '#include <cstdlib>' not in cc:
    cc = cc.replace('#include <iostream>\n', '#include <iostream>\n#include <cstdlib>\n') if '#include <iostream>' in cc else cc.replace('#include "spp_dev.h"\n', '#include "spp_dev.h"\n#include <cstdlib>\n')
if '#include <iomanip>' not in cc:
    cc = cc.replace('#include <cstdlib>\n', '#include <cstdlib>\n#include <iomanip>\n')

# ---------- initialize patch ----------
init_pat = r'(void\s+spp_dev::prefetcher_initialize\s*\(\s*\)\s*\{)'
init_insert = '''\\1
  const char* cand_path = std::getenv("SPP_CAND_LOG");
  if (cand_path && cand_path[0] != 0) {
    cand_log_.open(cand_path);
    if (cand_log_.is_open()) {
      cand_log_ << "event,cand_id,addr,ip,pf_addr,delta,confidence,fill_l2,cache_hit,mshr_occupancy,mshr_size,pq_occupancy,pq_size,useful_prefetch,depth" << std::endl;
    }
  }
'''
cc, n = re.subn(init_pat, init_insert, cc, count=1, flags=re.S)
if n != 1:
    raise SystemExit('[error] failed to patch prefetcher_initialize()')

# ---------- cache_operate demand/use patch ----------
needle = '  FILTER.check(addr, spp_dev::L2C_DEMAND);\n'
replace = '''  FILTER.check(addr, spp_dev::L2C_DEMAND);
  if (cand_log_.is_open() && useful_prefetch) {
    cand_log_ << "USE"
              << ',' << 0
              << ',' << addr.template to<uint64_t>()
              << ',' << ip.template to<uint64_t>()
              << ',' << addr.template to<uint64_t>()
              << ',' << 0
              << ',' << 0
              << ',' << 0
              << ',' << static_cast<uint32_t>(cache_hit)
              << ',' << (intern_ ? intern_->get_mshr_occupancy() : 0)
              << ',' << (intern_ ? intern_->get_mshr_size() : 0)
              << ',' << 0
              << ',' << (intern_ ? intern_->PQ_SIZE : 0)
              << ',' << static_cast<uint32_t>(useful_prefetch)
              << ',' << 0
              << std::endl;
  }
'''
if needle not in cc:
    raise SystemExit('[error] did not find FILTER.check(addr, spp_dev::L2C_DEMAND);')
cc = cc.replace(needle, replace, 1)

# ---------- candidate issue patch ----------
old = '''          if (FILTER.check(pf_addr, ((confidence_q[i] >= FILL_THRESHOLD) ? spp_dev::SPP_L2C_PREFETCH : spp_dev::SPP_LLC_PREFETCH))) {
            prefetch_line(pf_addr, (confidence_q[i] >= FILL_THRESHOLD), 0); // Use addr (not base_addr) to obey the same physical page boundary
'''
new = '''          const bool fill_l2 = (confidence_q[i] >= FILL_THRESHOLD);
          if (FILTER.check(pf_addr, (fill_l2 ? spp_dev::SPP_L2C_PREFETCH : spp_dev::SPP_LLC_PREFETCH))) {
            if (cand_log_.is_open()) {
              const uint64_t my_cand_id = cand_id_++;
              cand_log_ << "CAND"
                        << ',' << my_cand_id
                        << ',' << addr.template to<uint64_t>()
                        << ',' << ip.template to<uint64_t>()
                        << ',' << pf_addr.template to<uint64_t>()
                        << ',' << delta_q[i]
                        << ',' << confidence_q[i]
                        << ',' << static_cast<uint32_t>(fill_l2)
                        << ',' << static_cast<uint32_t>(cache_hit)
                        << ',' << (intern_ ? intern_->get_mshr_occupancy() : 0)
                        << ',' << (intern_ ? intern_->get_mshr_size() : 0)
                        << ',' << 0
                        << ',' << (intern_ ? intern_->PQ_SIZE : 0)
                        << ',' << static_cast<uint32_t>(useful_prefetch)
                        << ',' << i
                        << std::endl;
            }
            prefetch_line(pf_addr, fill_l2, 0); // Use addr (not base_addr) to obey the same physical page boundary
'''
if old not in cc:
    raise SystemExit('[error] did not find SPP candidate FILTER.check/prefetch_line block; source may differ')
cc = cc.replace(old, new, 1)

# ---------- final stats close patch ----------
final_pat = r'(void\s+spp_dev::prefetcher_final_stats\s*\(\s*\)\s*\{)'
final_insert = '''\\1
  if (cand_log_.is_open()) {
    cand_log_.flush();
    cand_log_.close();
  }
'''
cc, n = re.subn(final_pat, final_insert, cc, count=1, flags=re.S)
if n != 1:
    raise SystemExit('[error] failed to patch prefetcher_final_stats()')

h_path.write_text(h)
cc_path.write_text(cc)
print('[patched]', h_path)
print('[patched]', cc_path)
PY

echo
printf '[check]\n'
grep -n "cand_log_\|SPP_CAND_LOG\|event,cand_id" "$SPP_H" "$SPP_CC" || true

echo
printf '[next] rebuild candidate-logging SPP binary:\n'
echo '  cd external/ChampSim'
echo '  python3 ./config.sh _cfg/cfg_l2_spp.json'
echo '  make -j8'
echo '  cp bin/champsim bin/champsim.l2_spp_cand'
