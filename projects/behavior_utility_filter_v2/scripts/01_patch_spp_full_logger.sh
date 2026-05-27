#!/usr/bin/env bash
# Patch local ChampSim prefetcher/spp_dev to emit a comprehensive SPP event log.
#
# This does not replace SPP with a learned policy. It only instruments SPP so we can
# train/debug a later candidate utility filter.
#
# Usage from cache_arch root:
#   RESET_SPP=1 bash projects/behavior_utility_filter_v2/scripts/01_patch_spp_full_logger.sh
#   cd external/ChampSim
#   python3 ./config.sh _cfg/cfg_l2_spp.json
#   make -j8
#   SPP_FULL_LOG=/scratch/qianruw/cache/results/generated/spp_full_602.gcc_s-734B.csv \
#     bin/champsim --warmup-instructions 25000000 --simulation-instructions 25000000 TRACE
#
# Output event types:
#   ACCESS    one demand-side prefetcher invocation after ST update
#   PT_UPDATE one pattern-table update opportunity
#   PT_READ   one pattern-table read/lookahead step
#   CAND      one candidate row, including suppressed, page-crossing, filter-rejected, and issued candidates
#   FILL      one cache fill/eviction callback row
#   SPP_FULL_FINAL printed to stdout at end

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

CHAMP="${CHAMP:-$ROOT/external/ChampSim}"
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
  git -C "$CHAMP" checkout -- prefetcher/spp_dev/spp_dev.h prefetcher/spp_dev/spp_dev.cc || true
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


def find_function_bounds(src: str, signature_regex: str):
    m = re.search(signature_regex, src, flags=re.S)
    if not m:
        raise SystemExit(f"[error] could not find function matching {signature_regex}")
    start = m.start()
    brace = src.find("{", m.end() - 1)
    if brace < 0:
        raise SystemExit("[error] could not find function opening brace")
    depth = 0
    for i in range(brace, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return start, brace, i + 1
    raise SystemExit("[error] could not find function closing brace")


def replace_function(src: str, signature_regex: str, new_func: str) -> str:
    start, _, end = find_function_bounds(src, signature_regex)
    return src[:start] + new_func + src[end:]


def insert_after_open_brace_once(src: str, signature_regex: str, marker: str, insert: str) -> str:
    start, brace, end = find_function_bounds(src, signature_regex)
    body = src[brace:end]
    if marker in body:
        return src
    return src[:brace + 1] + insert + src[brace + 1:]


def find_block_end(src: str, open_brace: int) -> int:
    depth = 0
    for i in range(open_brace, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise SystemExit("[error] could not find block end")


# ---------- header patch ----------
if '#include <fstream>' not in h:
    h = h.replace('#include <vector>\n', '#include <vector>\n#include <fstream>\n')

if 'spp_full_log_' not in h:
    insert = '''
  // Comprehensive SPP instrumentation logger.
  // Opened only when SPP_FULL_LOG is set.
  std::ofstream spp_full_log_;
  uint64_t spp_full_event_id_ = 0;
  uint64_t spp_full_cand_id_ = 0;
'''
    pos = h.rfind('\n};')
    if pos < 0:
        raise SystemExit('[error] could not find end of spp_dev struct in spp_dev.h')
    h = h[:pos] + insert + h[pos:]

# ---------- includes ----------
if '#include <iostream>' not in cc:
    cc = cc.replace('#include "spp_dev.h"\n', '#include "spp_dev.h"\n#include <iostream>\n')
if '#include <cstdlib>' not in cc:
    cc = cc.replace('#include <iostream>\n', '#include <iostream>\n#include <cstdlib>\n')
if '#include <iomanip>' not in cc:
    cc = cc.replace('#include <cstdlib>\n', '#include <cstdlib>\n#include <iomanip>\n')

# Fix older candidate logger header mismatch if that logger exists.
cc = cc.replace(
    'event,cand_id,addr,ip,pf_addr,delta,confidence,fill_l2,cache_hit,mshr_occupancy,mshr_size,pq_occupancy,pq_size,useful_prefetch,depth',
    'event,cand_id,addr,ip,pf_addr,delta,confidence,fill_l2,issued,cache_hit,mshr_occupancy,mshr_size,pq_occupancy,pq_size,useful_prefetch,depth'
)

# ---------- initialize: open full logger; preserve original SPP initialization ----------
full_header = (
    'event,event_id,cand_id,'
    'addr,addr_line,page,page_offset,ip,cache_hit,useful_prefetch,access_type,metadata_in,'
    'last_sig,curr_sig,observed_delta,global_accuracy,pf_issued_ctr,pf_useful_ctr,'
    'mshr_occ,mshr_size,rq_occ,rq_size,wq_occ,wq_size,pq_occ,pq_size,'
    'pt_lookahead_way,lookahead_conf,pf_q_head,pf_q_tail,depth,cand_index,'
    'base_addr,pf_addr,pf_line,pf_page,pf_page_offset,cand_delta,cand_conf,'
    'threshold_pass,fill_l2,same_page,filter_pass,issued,ghr_update,'
    'evicted_addr,set,way,prefetch,metadata_out'
)
init_insert = f'''
  const char* full_path = std::getenv("SPP_FULL_LOG");
  if (full_path && full_path[0] != 0) {{
    spp_full_log_.open(full_path);
    if (spp_full_log_.is_open()) {{
      spp_full_log_ << "{full_header}" << std::endl;
    }}
  }}
'''
cc = insert_after_open_brace_once(
    cc,
    r'void\s+spp_dev::prefetcher_initialize\s*\(\s*\)',
    'SPP_FULL_LOG',
    init_insert,
)

# ---------- cache_operate queue snapshot helpers ----------
helper_insert = r'''

  auto spp_qsum = [](const auto& xs) -> std::size_t {
    std::size_t total = 0;
    for (auto x : xs)
      total += static_cast<std::size_t>(x);
    return total;
  };
  auto spp_mshr_occ = [&]() -> std::size_t { return intern_ ? intern_->get_mshr_occupancy() : 0; };
  auto spp_mshr_size = [&]() -> std::size_t { return intern_ ? intern_->get_mshr_size() : 0; };
  auto spp_rq_occ = [&]() -> std::size_t { return intern_ ? spp_qsum(intern_->get_rq_occupancy()) : 0; };
  auto spp_rq_size = [&]() -> std::size_t { return intern_ ? spp_qsum(intern_->get_rq_size()) : 0; };
  auto spp_wq_occ = [&]() -> std::size_t { return intern_ ? spp_qsum(intern_->get_wq_occupancy()) : 0; };
  auto spp_wq_size = [&]() -> std::size_t { return intern_ ? spp_qsum(intern_->get_wq_size()) : 0; };
  auto spp_pq_occ = [&]() -> std::size_t { return intern_ ? spp_qsum(intern_->get_pq_occupancy()) : 0; };
  auto spp_pq_size = [&]() -> std::size_t { return intern_ ? spp_qsum(intern_->get_pq_size()) : 0; };
'''
if 'auto spp_qsum' not in cc:
    needle = '  GHR.global_accuracy = GHR.pf_issued ? ((100 * GHR.pf_useful) / GHR.pf_issued) : 0;\n'
    if needle not in cc:
        raise SystemExit('[error] could not find GHR.global_accuracy line')
    cc = cc.replace(needle, needle + helper_insert, 1)

# ---------- ACCESS log after ST update ----------
access_log = r'''

  if (spp_full_log_.is_open()) {
    spp_full_log_ << "ACCESS"
                  << ',' << spp_full_event_id_++
                  << ',' << 0
                  << ',' << addr.template to<uint64_t>()
                  << ',' << (addr.template to<uint64_t>() >> LOG2_BLOCK_SIZE)
                  << ',' << champsim::page_number{addr}.template to<uint64_t>()
                  << ',' << spp_dev::offset_type{addr}.template to<uint64_t>()
                  << ',' << ip.template to<uint64_t>()
                  << ',' << static_cast<uint32_t>(cache_hit)
                  << ',' << static_cast<uint32_t>(useful_prefetch)
                  << ',' << static_cast<int>(type)
                  << ',' << metadata_in
                  << ',' << last_sig
                  << ',' << curr_sig
                  << ',' << delta
                  << ',' << GHR.global_accuracy
                  << ',' << GHR.pf_issued
                  << ',' << GHR.pf_useful
                  << ',' << spp_mshr_occ()
                  << ',' << spp_mshr_size()
                  << ',' << spp_rq_occ()
                  << ',' << spp_rq_size()
                  << ',' << spp_wq_occ()
                  << ',' << spp_wq_size()
                  << ',' << spp_pq_occ()
                  << ',' << spp_pq_size()
                  << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ',' << depth << ',' << 0
                  << ',' << addr.template to<uint64_t>()
                  << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0
                  << ',' << 0 << ',' << 0
                  << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0
                  << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ',' << metadata_in
                  << std::endl;
  }
'''
if '<< "ACCESS"' not in cc:
    needle = '  ST.read_and_update_sig(addr, last_sig, curr_sig, delta);\n'
    if needle not in cc:
        raise SystemExit('[error] could not find ST.read_and_update_sig line')
    cc = cc.replace(needle, needle + access_log, 1)

# ---------- PT_UPDATE log before PT update ----------
pt_update_log = r'''
  if (spp_full_log_.is_open()) {
    spp_full_log_ << "PT_UPDATE"
                  << ',' << spp_full_event_id_++
                  << ',' << 0
                  << ',' << addr.template to<uint64_t>()
                  << ',' << (addr.template to<uint64_t>() >> LOG2_BLOCK_SIZE)
                  << ',' << champsim::page_number{addr}.template to<uint64_t>()
                  << ',' << spp_dev::offset_type{addr}.template to<uint64_t>()
                  << ',' << ip.template to<uint64_t>()
                  << ',' << static_cast<uint32_t>(cache_hit)
                  << ',' << static_cast<uint32_t>(useful_prefetch)
                  << ',' << static_cast<int>(type)
                  << ',' << metadata_in
                  << ',' << last_sig
                  << ',' << curr_sig
                  << ',' << delta
                  << ',' << GHR.global_accuracy
                  << ',' << GHR.pf_issued
                  << ',' << GHR.pf_useful
                  << ',' << spp_mshr_occ() << ',' << spp_mshr_size()
                  << ',' << spp_rq_occ() << ',' << spp_rq_size()
                  << ',' << spp_wq_occ() << ',' << spp_wq_size()
                  << ',' << spp_pq_occ() << ',' << spp_pq_size()
                  << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ',' << depth << ',' << 0
                  << ',' << addr.template to<uint64_t>()
                  << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0
                  << ',' << delta << ',' << 0
                  << ',' << static_cast<uint32_t>(last_sig != 0)
                  << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0
                  << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ',' << metadata_in
                  << std::endl;
  }
'''
if '<< "PT_UPDATE"' not in cc:
    needle = '  // Stage 2: Update delta patterns stored in PT\n'
    if needle not in cc:
        raise SystemExit('[error] could not find Stage 2 comment')
    cc = cc.replace(needle, pt_update_log + '\n' + needle, 1)

# ---------- PT_READ log after PT read_pattern ----------
pt_read_log = r'''

    if (spp_full_log_.is_open()) {
      spp_full_log_ << "PT_READ"
                    << ',' << spp_full_event_id_++
                    << ',' << 0
                    << ',' << addr.template to<uint64_t>()
                    << ',' << (addr.template to<uint64_t>() >> LOG2_BLOCK_SIZE)
                    << ',' << champsim::page_number{addr}.template to<uint64_t>()
                    << ',' << spp_dev::offset_type{addr}.template to<uint64_t>()
                    << ',' << ip.template to<uint64_t>()
                    << ',' << static_cast<uint32_t>(cache_hit)
                    << ',' << static_cast<uint32_t>(useful_prefetch)
                    << ',' << static_cast<int>(type)
                    << ',' << metadata_in
                    << ',' << last_sig
                    << ',' << curr_sig
                    << ',' << delta
                    << ',' << GHR.global_accuracy
                    << ',' << GHR.pf_issued
                    << ',' << GHR.pf_useful
                    << ',' << spp_mshr_occ() << ',' << spp_mshr_size()
                    << ',' << spp_rq_occ() << ',' << spp_rq_size()
                    << ',' << spp_wq_occ() << ',' << spp_wq_size()
                    << ',' << spp_pq_occ() << ',' << spp_pq_size()
                    << ',' << lookahead_way
                    << ',' << lookahead_conf
                    << ',' << pf_q_head
                    << ',' << pf_q_tail
                    << ',' << depth
                    << ',' << 0
                    << ',' << base_addr.template to<uint64_t>()
                    << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0
                    << ',' << 0 << ',' << 0
                    << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0
                    << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ',' << metadata_in
                    << std::endl;
    }
'''
if '<< "PT_READ"' not in cc:
    needle = '    PT.read_pattern(curr_sig, delta_q, confidence_q, lookahead_way, lookahead_conf, pf_q_tail, depth);\n'
    if needle not in cc:
        raise SystemExit('[error] could not find PT.read_pattern line')
    cc = cc.replace(needle, needle + pt_read_log, 1)

# ---------- Replace candidate loop to log all candidate outcomes ----------
if 'const bool threshold_pass = (confidence_q[i] >= PF_THRESHOLD);' not in cc:
    loop_pat = '    for (uint32_t i = pf_q_head; i < pf_q_tail; i++) {'
    loop_start = cc.find(loop_pat)
    if loop_start < 0:
        raise SystemExit('[error] could not find SPP candidate for-loop')
    open_brace = cc.find('{', loop_start)
    loop_end = find_block_end(cc, open_brace)
    candidate_loop = r'''    for (uint32_t i = pf_q_head; i < pf_q_tail; i++) {
      const bool threshold_pass = (confidence_q[i] >= PF_THRESHOLD);
      champsim::address pf_addr{champsim::block_number{base_addr} + delta_q[i]};
      const bool same_page = (champsim::page_number{pf_addr} == page);
      const bool fill_l2 = (confidence_q[i] >= FILL_THRESHOLD);
      bool filter_pass = false;
      bool issued = false;
      bool ghr_update = false;

      if (threshold_pass) {
        if (same_page) {
          filter_pass = FILTER.check(pf_addr, (fill_l2 ? spp_dev::SPP_L2C_PREFETCH : spp_dev::SPP_LLC_PREFETCH));
          if (filter_pass) {
            issued = prefetch_line(pf_addr, fill_l2, 0);

            if (fill_l2) {
              GHR.pf_issued++;
              if (GHR.pf_issued > GLOBAL_COUNTER_MAX) {
                GHR.pf_issued >>= 1;
                GHR.pf_useful >>= 1;
              }
            }
          }
        } else {
          if constexpr (GHR_ON) {
            GHR.update_entry(curr_sig, confidence_q[i], spp_dev::offset_type{pf_addr}, delta_q[i]);
            ghr_update = true;
          }
        }

        do_lookahead = 1;
        pf_q_head++;
      }

      if (spp_full_log_.is_open()) {
        const uint64_t my_cand_id = spp_full_cand_id_++;
        spp_full_log_ << "CAND"
                      << ',' << spp_full_event_id_++
                      << ',' << my_cand_id
                      << ',' << addr.template to<uint64_t>()
                      << ',' << (addr.template to<uint64_t>() >> LOG2_BLOCK_SIZE)
                      << ',' << champsim::page_number{addr}.template to<uint64_t>()
                      << ',' << spp_dev::offset_type{addr}.template to<uint64_t>()
                      << ',' << ip.template to<uint64_t>()
                      << ',' << static_cast<uint32_t>(cache_hit)
                      << ',' << static_cast<uint32_t>(useful_prefetch)
                      << ',' << static_cast<int>(type)
                      << ',' << metadata_in
                      << ',' << last_sig
                      << ',' << curr_sig
                      << ',' << delta
                      << ',' << GHR.global_accuracy
                      << ',' << GHR.pf_issued
                      << ',' << GHR.pf_useful
                      << ',' << spp_mshr_occ() << ',' << spp_mshr_size()
                      << ',' << spp_rq_occ() << ',' << spp_rq_size()
                      << ',' << spp_wq_occ() << ',' << spp_wq_size()
                      << ',' << spp_pq_occ() << ',' << spp_pq_size()
                      << ',' << lookahead_way
                      << ',' << lookahead_conf
                      << ',' << pf_q_head
                      << ',' << pf_q_tail
                      << ',' << depth
                      << ',' << i
                      << ',' << base_addr.template to<uint64_t>()
                      << ',' << pf_addr.template to<uint64_t>()
                      << ',' << (pf_addr.template to<uint64_t>() >> LOG2_BLOCK_SIZE)
                      << ',' << champsim::page_number{pf_addr}.template to<uint64_t>()
                      << ',' << spp_dev::offset_type{pf_addr}.template to<uint64_t>()
                      << ',' << delta_q[i]
                      << ',' << confidence_q[i]
                      << ',' << static_cast<uint32_t>(threshold_pass)
                      << ',' << static_cast<uint32_t>(fill_l2)
                      << ',' << static_cast<uint32_t>(same_page)
                      << ',' << static_cast<uint32_t>(filter_pass)
                      << ',' << static_cast<uint32_t>(issued)
                      << ',' << static_cast<uint32_t>(ghr_update)
                      << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ',' << metadata_in
                      << std::endl;
      }
    }
'''
    cc = cc[:loop_start] + candidate_loop + cc[loop_end:]

# ---------- FILL log in cache_fill ----------
fill_helper = r'''
  auto spp_qsum_fill = [](const auto& xs) -> std::size_t {
    std::size_t total = 0;
    for (auto x : xs)
      total += static_cast<std::size_t>(x);
    return total;
  };
  auto spp_mshr_occ_fill = [&]() -> std::size_t { return intern_ ? intern_->get_mshr_occupancy() : 0; };
  auto spp_mshr_size_fill = [&]() -> std::size_t { return intern_ ? intern_->get_mshr_size() : 0; };
  auto spp_rq_occ_fill = [&]() -> std::size_t { return intern_ ? spp_qsum_fill(intern_->get_rq_occupancy()) : 0; };
  auto spp_rq_size_fill = [&]() -> std::size_t { return intern_ ? spp_qsum_fill(intern_->get_rq_size()) : 0; };
  auto spp_wq_occ_fill = [&]() -> std::size_t { return intern_ ? spp_qsum_fill(intern_->get_wq_occupancy()) : 0; };
  auto spp_wq_size_fill = [&]() -> std::size_t { return intern_ ? spp_qsum_fill(intern_->get_wq_size()) : 0; };
  auto spp_pq_occ_fill = [&]() -> std::size_t { return intern_ ? spp_qsum_fill(intern_->get_pq_occupancy()) : 0; };
  auto spp_pq_size_fill = [&]() -> std::size_t { return intern_ ? spp_qsum_fill(intern_->get_pq_size()) : 0; };

  if (spp_full_log_.is_open()) {
    spp_full_log_ << "FILL"
                  << ',' << spp_full_event_id_++
                  << ',' << 0
                  << ',' << addr.template to<uint64_t>()
                  << ',' << (addr.template to<uint64_t>() >> LOG2_BLOCK_SIZE)
                  << ',' << champsim::page_number{addr}.template to<uint64_t>()
                  << ',' << spp_dev::offset_type{addr}.template to<uint64_t>()
                  << ',' << 0
                  << ',' << 0
                  << ',' << 0
                  << ',' << 0
                  << ',' << metadata_in
                  << ',' << 0 << ',' << 0 << ',' << 0
                  << ',' << GHR.global_accuracy
                  << ',' << GHR.pf_issued
                  << ',' << GHR.pf_useful
                  << ',' << spp_mshr_occ_fill() << ',' << spp_mshr_size_fill()
                  << ',' << spp_rq_occ_fill() << ',' << spp_rq_size_fill()
                  << ',' << spp_wq_occ_fill() << ',' << spp_wq_size_fill()
                  << ',' << spp_pq_occ_fill() << ',' << spp_pq_size_fill()
                  << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0
                  << ',' << 0
                  << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0
                  << ',' << 0 << ',' << 0
                  << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0
                  << ',' << evicted_addr.template to<uint64_t>()
                  << ',' << set
                  << ',' << way
                  << ',' << static_cast<uint32_t>(prefetch)
                  << ',' << metadata_in
                  << std::endl;
  }

'''
if '<< "FILL"' not in cc:
    cc = insert_after_open_brace_once(
        cc,
        r'uint32_t\s+spp_dev::prefetcher_cache_fill\s*\([^)]*\)',
        'spp_qsum_fill',
        '\n' + fill_helper,
    )

# ---------- final stats: close full logger and print summary ----------
final_func = r'''void spp_dev::prefetcher_final_stats()
{
  if (spp_full_log_.is_open()) {
    spp_full_log_.flush();
    spp_full_log_.close();
  }

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

  std::cout << "SPP_FULL_FINAL"
            << " events=" << spp_full_event_id_
            << " candidates=" << spp_full_cand_id_
            << " pf_issued_ctr=" << issued
            << " pf_useful_ctr=" << useful
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
echo '[check full logger members]'
grep -a -n "spp_full_log_\|spp_full_event_id_\|spp_full_cand_id_" "$SPP_H" "$SPP_CC" || true

echo
echo '[check full logger event writes]'
grep -a -n '"ACCESS"\|"PT_UPDATE"\|"PT_READ"\|"CAND"\|"FILL"\|SPP_FULL_FINAL' "$SPP_CC" || true

echo
echo '[next] rebuild SPP binary:'
echo '  cd external/ChampSim'
echo '  python3 ./config.sh _cfg/cfg_l2_spp.json'
echo '  make -j8'
echo
echo '[next] run with CSV output:'
echo '  SPP_FULL_LOG=/scratch/qianruw/cache/results/generated/spp_full_TRACE.csv bin/champsim --warmup-instructions 25000000 --simulation-instructions 25000000 TRACE'
