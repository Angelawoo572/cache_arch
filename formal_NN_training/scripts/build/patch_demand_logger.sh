#!/usr/bin/env bash
# Patch local Pythia to emit raw L2C demand/PF event streams.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
CACHE_CC="$CHAMP_DIR/src/cache.cc"
[[ -f "$CACHE_CC" ]] || { echo "[error] missing $CACHE_CC" >&2; exit 2; }

if [[ "${RESET_PATCH:-0}" == "1" ]]; then
  git -C "$CHAMP_DIR" checkout -- src/cache.cc
fi

python3 - "$CACHE_CC" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text(errors="ignore")
if "DEMAND_EVENT_LOG" in s and "demand_event_log_demand" in s:
    print("[skip] demand logger already present", p)
    raise SystemExit(0)
if "RESIDUAL_AUDIT_LOG" in s:
    raise SystemExit("[error] old logger patch detected; rerun with RESET_PATCH=1")
s = s.replace('#include <algorithm>\n#include "cache.h"', '#include <algorithm>\n#include <cstdlib>\n#include <fstream>\n#include "cache.h"')
helper = r'''
static std::ofstream demand_event_log;
static bool demand_event_log_ready = false;
static uint64_t demand_event_id = 0;
static void demand_event_open_once()
{
    if (demand_event_log_ready) return;
    demand_event_log_ready = true;
    const char* path = std::getenv("DEMAND_EVENT_LOG");
    if (path && path[0] != 0) {
        demand_event_log.open(path);
        if (demand_event_log.is_open())
            demand_event_log << "event,event_id,cpu,cycle,cache,op,type,ip,addr,line,hit,was_prefetch,late,accepted,duplicate,base_addr,pf_addr,pf_line,fill_level,pq_occ,pq_size,mshr_occ,mshr_size" << std::endl;
    }
}
static void demand_event_log_demand(const CACHE* cache, uint32_t cpu, uint64_t cycle, const char* op, uint32_t type,
                                    uint64_t ip, uint64_t full_addr, uint64_t line_addr, uint32_t hit,
                                    uint32_t was_prefetch, uint32_t late)
{
    demand_event_open_once();
    if (!demand_event_log.is_open()) return;
    demand_event_log << "DEMAND" << ',' << demand_event_id++ << ',' << cpu << ',' << cycle << ',' << cache->NAME
                     << ',' << op << ',' << type << ',' << ip << ',' << full_addr << ',' << line_addr
                     << ',' << hit << ',' << was_prefetch << ',' << late
                     << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0
                     << ',' << cache->PQ.occupancy << ',' << cache->PQ.SIZE
                     << ',' << cache->MSHR.occupancy << ',' << cache->MSHR.SIZE << std::endl;
}
static void demand_event_log_pf(const CACHE* cache, uint32_t cpu, uint64_t cycle, uint64_t ip,
                                uint64_t base_addr, uint64_t pf_addr, int fill_level,
                                uint32_t accepted, uint32_t duplicate)
{
    demand_event_open_once();
    if (!demand_event_log.is_open()) return;
    demand_event_log << "PF" << ',' << demand_event_id++ << ',' << cpu << ',' << cycle << ',' << cache->NAME
                     << ',' << "prefetch_line" << ',' << PREFETCH << ',' << ip << ',' << pf_addr
                     << ',' << (pf_addr >> LOG2_BLOCK_SIZE) << ',' << 0 << ',' << 0 << ',' << 0
                     << ',' << accepted << ',' << duplicate << ',' << base_addr << ',' << pf_addr
                     << ',' << (pf_addr >> LOG2_BLOCK_SIZE) << ',' << fill_level
                     << ',' << cache->PQ.occupancy << ',' << cache->PQ.SIZE
                     << ',' << cache->MSHR.occupancy << ',' << cache->MSHR.SIZE << std::endl;
}
'''
marker = 'uint64_t l2pf_access = 0;\n'
if marker not in s:
    raise SystemExit('[error] could not find l2pf_access marker')
s = s.replace(marker, marker + helper + '\n', 1)
hit_needle = '''                // update prefetch stats and reset prefetch bit
                if (block[set][way].prefetch)
                {
                    pf_useful++;'''
hit_insert = '''                // update prefetch stats and reset prefetch bit
                if (cache_type == IS_L2C && warmup_complete[read_cpu] && RQ.entry[index].type == LOAD) {
                    demand_event_log_demand(this, read_cpu, current_core_cycle[read_cpu], "read", RQ.entry[index].type,
                                             RQ.entry[index].ip, RQ.entry[index].full_addr, RQ.entry[index].address,
                                             1, block[set][way].prefetch ? 1 : 0, 0);
                }
                if (block[set][way].prefetch)
                {
                    pf_useful++;'''
if hit_needle not in s:
    raise SystemExit('[error] could not find read-hit marker')
s = s.replace(hit_needle, hit_insert, 1)
miss_needle = '''                int mshr_index = check_mshr(&RQ.entry[index]);

                if ((mshr_index == -1) && (MSHR.occupancy < MSHR_SIZE)) // this is a new miss'''
miss_insert = '''                int mshr_index = check_mshr(&RQ.entry[index]);

                if (cache_type == IS_L2C && warmup_complete[read_cpu] && RQ.entry[index].type == LOAD) {
                    uint32_t demand_late = (mshr_index != -1 && MSHR.entry[mshr_index].type == PREFETCH) ? 1 : 0;
                    demand_event_log_demand(this, read_cpu, current_core_cycle[read_cpu], "read", RQ.entry[index].type,
                                             RQ.entry[index].ip, RQ.entry[index].full_addr, RQ.entry[index].address,
                                             0, 0, demand_late);
                }

                if ((mshr_index == -1) && (MSHR.occupancy < MSHR_SIZE)) // this is a new miss'''
if miss_needle not in s:
    raise SystemExit('[error] could not find read-miss marker')
s = s.replace(miss_needle, miss_insert, 1)
pf_needle = '''        // give a dummy 0 as the IP of a prefetch
        add_pq(&pf_packet);
        pf_issued++;

        return 1;'''
pf_insert = '''        // give a dummy 0 as the IP of a prefetch
        int pq_result = add_pq(&pf_packet);
        pf_issued++;
        if (warmup_complete[cpu]) {
            demand_event_log_pf(this, cpu, current_core_cycle[cpu], ip, base_addr, pf_addr, pf_fill_level,
                                (pq_result == -2) ? 0 : 1, (pq_result >= 0) ? 1 : 0);
        }

        return 1;'''
if pf_needle not in s:
    raise SystemExit('[error] could not find prefetch marker')
s = s.replace(pf_needle, pf_insert, 1)
p.write_text(s)
print('[patched]', p)
PY

grep -n "DEMAND_EVENT_LOG\|demand_event_log_demand" "$CACHE_CC" || true
