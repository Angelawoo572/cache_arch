#!/usr/bin/env bash
# Patch local ChampSim spp_dev to dump per-candidate SPP events for RL-filter training.
#
# Recommended use when the local spp_dev.cc has been partially patched before:
#   RESET_SPP=1 bash projects/post_prefetch_filter/scripts/04_patch_spp_candidate_logger.sh
#
# This script is self-contained: it adds both
#   1. candidate event logging with an `issued` column from prefetch_line(...)
#   2. SPP_FINAL aggregate final stats
#
# CSV events:
#   CAND = candidate that passed SPP FILTER.check(); issued is prefetch_line return value
#   USE  = later demand access marked useful_prefetch by ChampSim

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

CHAMP="$ROOT/external/ChampSim"
SPP_H="$CHAMP/prefetcher/spp_dev/spp_dev.h"
SPP_CC="$CHAMP/prefetcher/spp_dev/spp_dev.cc"

if [ ! -f "$SPP_H" ] || [ ! -f "$SPP_CC" ]; then
  echo "[error] Cannot find spp_dev files:"
  echo "  $SPP_H"
  echo "  $SPP_CC"
  exit 1
fi

if [ "${RESET_SPP:-0}" = "1" ]; then
  echo "[reset] restoring spp_dev.h/spp_dev.cc from external/ChampSim git checkout"
  git -C "$CHAMP" checkout -- prefetcher/spp_dev/spp_dev.h prefetcher/spp_dev/spp_dev.cc
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


def replace_function(src: str, signature_regex: str, new_func: str) -> str:
    m = re.search(signature_regex, src, flags=re.S)
    if not m:
        raise SystemExit(f"[error] could not find function matching {signature_regex}")
    start = m.start()
    brace = src.find("{", m.end() - 1)
    if brace < 0:
        raise SystemExit("[error] could not find function opening brace")
    depth = 0
    end = None
    for i in range(brace, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise SystemExit("[error] could not find function closing brace")
    return src[:start] + new_func + src[end:]


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
    if '  GLOBAL_REGISTER GHR;\n' not in h:
        raise SystemExit('[error] could not find GLOBAL_REGISTER GHR in spp_dev.h')
    h = h.replace('  GLOBAL_REGISTER GHR;\n', '  GLOBAL_REGISTER GHR;\n' + member_block)

# ---------- includes ----------
if '#include <iostream>' not in cc:
    cc = cc.replace('#include "spp_dev.h"\n', '#include "spp_dev.h"\n#include <iostream>\n')
if '#include <cstdlib>' not in cc:
    cc = cc.replace('#include <iostream>\n', '#include <iostream>\n#include <cstdlib>\n')
if '#include <iomanip>' not in cc:
    cc = cc.replace('#include <cstdlib>\n', '#include <cstdlib>\n#include <iomanip>\n')

# ---------- initialize ----------
init_func = '''void spp_dev::prefetcher_initialize()
{
  const char* cand_path = std::getenv("SPP_CAND_LOG");
  if (cand_path && cand_path[0] != 0) {
    cand_log_.open(cand_path);
    if (cand_log_.is_open()) {
      cand_log_ << "event,cand_id,addr,ip,pf_addr,delta,confidence,fill_l2,issued,cache_hit,mshr_occupancy,mshr_size,pq_occupancy,pq_size,useful_prefetch,depth" << std::endl;
    }
  }
}
'''
cc = replace_function(cc, r'void\s+spp_dev::prefetcher_initialize\s*\(\s*\)', init_func)

# ---------- USE logging ----------
if 'cand_log_ << "USE"' not in cc:
    needle = '  FILTER.check(addr, spp_dev::L2C_DEMAND);\n'
    use_block = '''  FILTER.check(addr, spp_dev::L2C_DEMAND);
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
        raise SystemExit('[error] did not find demand FILTER.check line')
    cc = cc.replace(needle, use_block, 1)

# ---------- candidate issue block ----------
if 'static_cast<uint32_t>(issued)' not in cc:
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
    pat = re.compile(
        r'''          if\s*\(FILTER\.check\(pf_addr,\s*\(\(confidence_q\[i\]\s*>=\s*FILL_THRESHOLD\)\s*\?\s*spp_dev::SPP_L2C_PREFETCH\s*:\s*spp_dev::SPP_LLC_PREFETCH\)\)\)\s*\{\s*\n\s*prefetch_line\(pf_addr,\s*\(confidence_q\[i\]\s*>=\s*FILL_THRESHOLD\),\s*0\);\s*// Use addr \(not base_addr\) to obey the same physical page boundary\s*\n''',
        flags=re.S | re.X,
    )
    cc, n = pat.subn(candidate_block, cc, count=1)
    if n != 1:
        raise SystemExit('[error] could not patch clean SPP candidate issue block; run with RESET_SPP=1 or paste surrounding source')

# ---------- final stats with logger close + SPP_FINAL ----------
final_func = '''void spp_dev::prefetcher_final_stats()
{
  if (cand_log_.is_open()) {
    cand_log_.flush();
    cand_log_.close();
  }

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
cc = replace_function(cc, r'void\s+spp_dev::prefetcher_final_stats\s*\(\s*\)', final_func)

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
echo '[check final stats]'
grep -a -n "SPP_FINAL\|prefetcher_final_stats" "$SPP_CC" || true

echo
echo '[next] rebuild candidate-logging SPP binary:'
echo '  cd external/ChampSim'
echo '  python3 ./config.sh _cfg/cfg_l2_spp.json'
echo '  make -j8'
echo '  cp bin/champsim bin/champsim.l2_spp_cand'
