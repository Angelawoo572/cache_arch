#!/usr/bin/env bash
# Patch local ChampSim spp_dev to dump per-candidate SPP events for RL-filter training.
#
# This creates a CSV log controlled by SPP_CAND_LOG:
#   SPP_CAND_LOG=projects/post_prefetch_filter/data/generated/spp_candidate_events.csv \
#     external/ChampSim/bin/champsim.l2_spp_cand ...
#
# CSV events:
#   CAND = SPP candidate that passed SPP's FILTER.check() path.
#          The `issued` column records prefetch_line(...) return value.
#   USE  = later demand access marked useful_prefetch by ChampSim.
#
# MSHR is logged exactly. PQ occupancy is currently 0 because this ChampSim
# version keeps CACHE::internal_PQ private to modules.

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

cp "$SPP_H" "$SPP_H.bak.$(date +%Y%m%d_%H%M%S)"
cp "$SPP_CC" "$SPP_CC.bak.$(date +%Y%m%d_%H%M%S)"

python3 - "$SPP_H" "$SPP_CC" <<'PY'
import re
import sys
from pathlib import Path

h_path = Path(sys.argv[1])
cc_path = Path(sys.argv[2])

h = h_path.read_text(errors="ignore").replace("\x00", "")
cc = cc_path.read_text(errors="ignore").replace("\x00", "")

# ---------- header patch ----------
if '#include <fstream>' not in h:
    h = h.replace('#include <vector>\n', '#include <vector>\n#include <fstream>\n')

member_block = '''
  // Post-prefetch-filter experiment logger.
  // Opened only when SPP_CAND_LOG is set.
  std::ofstream cand_log_;
  uint64_t cand_id_ = 0;
'''
if 'cand_log_' not in h:
    h = h.replace('  GLOBAL_REGISTER GHR;\n', '  GLOBAL_REGISTER GHR;\n' + member_block)

# ---------- includes ----------
if '#include <cstdlib>' not in cc:
    if '#include <iostream>' in cc:
        cc = cc.replace('#include <iostream>\n', '#include <iostream>\n#include <cstdlib>\n')
    else:
        cc = cc.replace('#include "spp_dev.h"\n', '#include "spp_dev.h"\n#include <cstdlib>\n')
if '#include <iomanip>' not in cc:
    cc = cc.replace('#include <cstdlib>\n', '#include <cstdlib>\n#include <iomanip>\n')

# ---------- initialize patch ----------
header_new = '"event,cand_id,addr,ip,pf_addr,delta,confidence,fill_l2,issued,cache_hit,mshr_occupancy,mshr_size,pq_occupancy,pq_size,useful_prefetch,depth"'
header_old = '"event,cand_id,addr,ip,pf_addr,delta,confidence,fill_l2,cache_hit,mshr_occupancy,mshr_size,pq_occupancy,pq_size,useful_prefetch,depth"'
cc = cc.replace(header_old, header_new)

if 'SPP_CAND_LOG' not in cc:
    init_pat = r'(void\s+spp_dev::prefetcher_initialize\s*\(\s*\)\s*\{)'
    init_insert = r'''\1
  const char* cand_path = std::getenv("SPP_CAND_LOG");
  if (cand_path && cand_path[0] != 0) {
    cand_log_.open(cand_path);
    if (cand_log_.is_open()) {
      cand_log_ << "event,cand_id,addr,ip,pf_addr,delta,confidence,fill_l2,issued,cache_hit,mshr_occupancy,mshr_size,pq_occupancy,pq_size,useful_prefetch,depth" << std::endl;
    }
  }
'''
    cc, n = re.subn(init_pat, init_insert, cc, count=1, flags=re.S)
    if n != 1:
        raise SystemExit('[error] failed to patch prefetcher_initialize()')

# ---------- USE event patch ----------
# Add the issued=0 field to old USE blocks if needed.
use_pat = r'(cand_log_\s*<<\s*"USE".*?<<\s*\',\'\s*<<\s*0\s*\n\s*<<\s*\',\'\s*<<\s*0\s*\n\s*<<\s*\',\'\s*<<\s*0\s*\n)(\s*<<\s*\',\'\s*<<\s*static_cast<uint32_t>\(cache_hit\))'
if 'cand_log_ << "USE"' in cc and 'fill_l2,issued,cache_hit' in cc:
    cc = re.sub(use_pat, r'\1              << \',\' << 0\n\2', cc, count=1, flags=re.S)

if 'cand_log_ << "USE"' not in cc:
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

candidate_block = '''          const bool fill_l2 = (confidence_q[i] >= FILL_THRESHOLD);
          if (FILTER.check(pf_addr, (fill_l2 ? spp_dev::SPP_L2C_PREFETCH : spp_dev::SPP_LLC_PREFETCH))) {
            const auto mshr_occupancy_before = (intern_ ? intern_->get_mshr_occupancy() : 0);
            const auto mshr_size_snapshot = (intern_ ? intern_->get_mshr_size() : 0);
            const auto pq_size_snapshot = (intern_ ? intern_->PQ_SIZE : 0);
            const auto issued = prefetch_line(pf_addr, fill_l2, 0); // Use addr (not base_addr) to obey the same physical page boundary
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
                        << ',' << static_cast<uint32_t>(issued)
                        << ',' << static_cast<uint32_t>(cache_hit)
                        << ',' << mshr_occupancy_before
                        << ',' << mshr_size_snapshot
                        << ',' << 0
                        << ',' << pq_size_snapshot
                        << ',' << static_cast<uint32_t>(useful_prefetch)
                        << ',' << i
                        << std::endl;
            }
'''

# Case A: original unpatched candidate block.
old_original = '''          if (FILTER.check(pf_addr, ((confidence_q[i] >= FILL_THRESHOLD) ? spp_dev::SPP_L2C_PREFETCH : spp_dev::SPP_LLC_PREFETCH))) {
            prefetch_line(pf_addr, (confidence_q[i] >= FILL_THRESHOLD), 0); // Use addr (not base_addr) to obey the same physical page boundary
'''
if old_original in cc:
    cc = cc.replace(old_original, candidate_block, 1)
else:
    # Case B: already-patched logger block before prefetch_line. Replace from
    # const bool fill_l2 through the prefetch_line line, preserving code after it.
    pat_existing = re.compile(
        r'''          const bool fill_l2 = \(confidence_q\[i\] >= FILL_THRESHOLD\);\n
            \s*if \(FILTER\.check\(pf_addr, \(fill_l2 \? spp_dev::SPP_L2C_PREFETCH : spp_dev::SPP_LLC_PREFETCH\)\)\) \{\n
            .*?
            \s*prefetch_line\(pf_addr, fill_l2, 0\); // Use addr \(not base_addr\) to obey the same physical page boundary\n''',
        flags=re.S | re.X,
    )
    cc, n = pat_existing.subn(candidate_block, cc, count=1)
    if n != 1:
        # Case C: already patched to const auto issued. Keep it, but make sure it logs issued.
        if 'const auto issued = prefetch_line(pf_addr, fill_l2, 0)' not in cc and 'const bool issued = prefetch_line(pf_addr, fill_l2, 0)' not in cc:
            raise SystemExit('[error] did not find a recognizable SPP candidate prefetch_line block')

# ---------- final stats close patch ----------
if 'cand_log_.close();' not in cc:
    final_pat = r'(void\s+spp_dev::prefetcher_final_stats\s*\(\s*\)\s*\{)'
    final_insert = r'''\1
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
echo '[check header]'
grep -a -n "event,cand_id" "$SPP_CC" || true

echo
echo '[check candidate issue block]'
grep -a -n "const auto issued\|static_cast<uint32_t>(issued)\|prefetch_line(pf_addr" "$SPP_CC" || true

echo
echo '[next] rebuild candidate-logging SPP binary:'
echo '  cd external/ChampSim'
echo '  python3 ./config.sh _cfg/cfg_l2_spp.json'
echo '  make -j8'
echo '  cp bin/champsim bin/champsim.l2_spp_cand'
