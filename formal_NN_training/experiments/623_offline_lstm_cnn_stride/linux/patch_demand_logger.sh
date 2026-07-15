#!/usr/bin/env bash
# Patch local ChampSim/Pythia to emit a causally keyed L2 demand/PF log for
# the matched 623 stride experiment.  Instrumentation only: it does not alter
# the prefetch policy or create labels.
#
# The v5 schema records the exact demand event whose synchronous L2 callback
# issued each prefetch.  This avoids guessing from prefetch_line(base_addr).
# A logger from an older version of this experiment is backed up and replaced
# automatically.  Unknown cache.cc edits still fail closed.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
CACHE_CC="$CHAMP_DIR/src/cache.cc"

[[ -f "$CACHE_CC" ]] || { echo "[error] missing $CACHE_CC" >&2; exit 2; }

if [[ "${RESET_PATCH:-0}" == "1" ]]; then
  mkdir -p "${RUN_DIR:-/tmp}"
  git -C "$CHAMP_DIR" diff -- src/cache.cc \
    > "${RUN_DIR:-/tmp}/cache.cc.before_explicit_reset.patch" || true
  git -C "$CHAMP_DIR" checkout -- src/cache.cc
elif ! grep -Fq "DEMAND_EVENT_LOG_SCHEMA_623_V5" "$CACHE_CC" \
    && grep -Eq 'DEMAND_EVENT_LOG_SCHEMA_623_|demand_event_log_demand|RESIDUAL_AUDIT_LOG' "$CACHE_CC"; then
  backup="${RUN_DIR:-/tmp}/cache.cc.before_623_auto_reset.patch"
  mkdir -p "$(dirname "$backup")"
  git -C "$CHAMP_DIR" diff -- src/cache.cc > "$backup" || true
  echo "[auto-reset] known older experimental cache.cc logger detected"
  echo "[auto-reset] prior diff saved to $backup"
  git -C "$CHAMP_DIR" checkout -- src/cache.cc
fi

if ! grep -Fq "DEMAND_EVENT_LOG_SCHEMA_623_V5" "$CACHE_CC" \
    && ! git -C "$CHAMP_DIR" diff --quiet -- src/cache.cc; then
  echo "[error] cache.cc has unknown local edits; refusing to overwrite it" >&2
  echo "[hint] inspect: git -C \"$CHAMP_DIR\" diff -- src/cache.cc" >&2
  exit 2
fi

python3 - "$CACHE_CC" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text(errors="ignore")
revision_marker = "DEMAND_EVENT_LOG_SCHEMA_623_V5"
if revision_marker in s:
    print("[skip] 623 v5 causal demand logger already present", p)
    raise SystemExit(0)
if "DEMAND_EVENT_LOG" in s or "demand_event_log_demand" in s or "RESIDUAL_AUDIT_LOG" in s:
    raise SystemExit(
        "[error] stale/foreign cache.cc logger remained after safe reset"
    )

s = s.replace(
    '#include <algorithm>\n#include "cache.h"',
    '#include <algorithm>\n#include <cstdlib>\n#include <fstream>\n#include "cache.h"',
)
helper = r'''
// DEMAND_EVENT_LOG_SCHEMA_623_V5
// -----------------------------------------------------------------------------
// Causally keyed raw L2 demand/PF logger for the split 623 experiments.
// Enabled only when DEMAND_EVENT_LOG is set.  Single-core experiment only.
// -----------------------------------------------------------------------------
static std::ofstream demand_event_log;
static bool demand_event_log_ready = false;
static uint64_t demand_event_id = 0;
static const uint64_t demand_event_no_trigger = uint64_t(-1);
static const char demand_event_logger_schema[] = "623_causal_trigger_v5";
static uint64_t demand_event_active_trigger_id = demand_event_no_trigger;
static uint32_t demand_event_active_trigger_cpu = 0;
static uint64_t demand_event_active_trigger_ip = 0;
static uint64_t demand_event_active_trigger_line = 0;

static void demand_event_open_once()
{
    if (demand_event_log_ready)
        return;
    demand_event_log_ready = true;
    const char* path = std::getenv("DEMAND_EVENT_LOG");
    if (path && path[0] != 0) {
        demand_event_log.open(path);
        if (demand_event_log.is_open())
            demand_event_log << "event,event_id,cpu,cycle,cache,op,type,ip,addr,line,hit,was_prefetch,late,accepted,duplicate,base_addr,pf_addr,pf_line,fill_level,pq_occ,pq_size,mshr_occ,mshr_size,trigger_event_id,trigger_cpu,trigger_ip,trigger_line,logger_schema" << std::endl;
    }
}

static uint64_t demand_event_log_demand(const CACHE* cache, uint32_t cpu, uint64_t cycle, const char* op, uint32_t type,
                                        uint64_t ip, uint64_t full_addr, uint64_t line_addr, uint32_t hit,
                                        uint32_t was_prefetch, uint32_t late)
{
    demand_event_open_once();
    if (!demand_event_log.is_open())
        return demand_event_no_trigger;
    const uint64_t this_event_id = demand_event_id++;
    demand_event_log << "DEMAND"
                     << ',' << this_event_id << ',' << cpu << ',' << cycle << ',' << cache->NAME
                     << ',' << op << ',' << type << ',' << ip << ',' << full_addr << ',' << line_addr
                     << ',' << hit << ',' << was_prefetch << ',' << late
                     << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0
                     << ',' << cache->PQ.occupancy << ',' << cache->PQ.SIZE
                     << ',' << cache->MSHR.occupancy << ',' << cache->MSHR.SIZE
                     << ',' << this_event_id << ',' << cpu << ',' << ip << ',' << line_addr
                     << ',' << demand_event_logger_schema << std::endl;
    return this_event_id;
}

static void demand_event_begin_trigger(uint64_t event_id, uint32_t cpu, uint64_t ip, uint64_t line)
{
    demand_event_active_trigger_id = event_id;
    demand_event_active_trigger_cpu = cpu;
    demand_event_active_trigger_ip = ip;
    demand_event_active_trigger_line = line;
}

static void demand_event_end_trigger()
{
    demand_event_active_trigger_id = demand_event_no_trigger;
    demand_event_active_trigger_cpu = 0;
    demand_event_active_trigger_ip = 0;
    demand_event_active_trigger_line = 0;
}

static void demand_event_log_pf(const CACHE* cache, uint32_t cpu, uint64_t cycle, uint64_t ip,
                                uint64_t base_addr, uint64_t pf_addr, int fill_level,
                                uint32_t accepted, uint32_t duplicate)
{
    demand_event_open_once();
    if (!demand_event_log.is_open())
        return;
    demand_event_log << "PF"
                     << ',' << demand_event_id++ << ',' << cpu << ',' << cycle << ',' << cache->NAME
                     << ',' << "prefetch_line" << ',' << PREFETCH << ',' << ip << ',' << pf_addr
                     << ',' << (pf_addr >> LOG2_BLOCK_SIZE) << ',' << 0 << ',' << 0 << ',' << 0
                     << ',' << accepted << ',' << duplicate << ',' << base_addr << ',' << pf_addr
                     << ',' << (pf_addr >> LOG2_BLOCK_SIZE) << ',' << fill_level
                     << ',' << cache->PQ.occupancy << ',' << cache->PQ.SIZE
                     << ',' << cache->MSHR.occupancy << ',' << cache->MSHR.SIZE
                     << ',' << demand_event_active_trigger_id
                     << ',' << demand_event_active_trigger_cpu
                     << ',' << demand_event_active_trigger_ip
                     << ',' << demand_event_active_trigger_line
                     << ',' << demand_event_logger_schema << std::endl;
}
'''
marker = 'uint64_t l2pf_access = 0;\n'
if marker not in s:
    raise SystemExit('[error] could not find l2pf_access marker')
s = s.replace(marker, marker + helper + '\n', 1)

index_needle = '''            int index = RQ.head;

            // access cache'''
index_insert = '''            int index = RQ.head;
            uint64_t demand_logger_trigger_id = demand_event_no_trigger;

            // access cache'''
if index_needle not in s:
    raise SystemExit('[error] could not find L2 read index marker')
s = s.replace(index_needle, index_insert, 1)

hit_needle = '''                // update prefetcher on load instruction
                if (RQ.entry[index].type == LOAD)'''
hit_insert = '''                // Log the demand before invoking L2 prefetcher callbacks so every
                // synchronous PF row follows, and explicitly names, its trigger.
                if (cache_type == IS_L2C && warmup_complete[read_cpu] && RQ.entry[index].type == LOAD) {
                    demand_logger_trigger_id = demand_event_log_demand(
                        this, read_cpu, current_core_cycle[read_cpu], "read", RQ.entry[index].type,
                        RQ.entry[index].ip, RQ.entry[index].full_addr, RQ.entry[index].address,
                        1, block[set][way].prefetch ? 1 : 0, 0);
                }

                // update prefetcher on load instruction
                if (RQ.entry[index].type == LOAD)'''
if hit_needle not in s:
    raise SystemExit('[error] could not find read-hit prefetcher marker')
s = s.replace(hit_needle, hit_insert, 1)

hit_operate_needle = '''                    else if (cache_type == IS_L2C)
                        l2c_prefetcher_operate(block[set][way].address<<LOG2_BLOCK_SIZE, RQ.entry[index].ip, 1, RQ.entry[index].type, 0);'''
hit_operate_insert = '''                    else if (cache_type == IS_L2C)
                    {
                        demand_event_begin_trigger(demand_logger_trigger_id, read_cpu,
                                                   RQ.entry[index].ip, RQ.entry[index].address);
                        l2c_prefetcher_operate(block[set][way].address<<LOG2_BLOCK_SIZE, RQ.entry[index].ip, 1, RQ.entry[index].type, 0);
                        demand_event_end_trigger();
                    }'''
if hit_operate_needle not in s:
    raise SystemExit('[error] could not find read-hit L2 prefetcher call')
s = s.replace(hit_operate_needle, hit_operate_insert, 1)

late_needle = '''                int mshr_index = check_mshr(&RQ.entry[index]);

                if ((mshr_index == -1) && (MSHR.occupancy < MSHR_SIZE)) // this is a new miss'''
late_insert = '''                int mshr_index = check_mshr(&RQ.entry[index]);
                // Snapshot lateness before merge handling can replace the PREFETCH MSHR entry.
                uint32_t demand_logger_late =
                    (mshr_index != -1 && MSHR.entry[mshr_index].type == PREFETCH) ? 1 : 0;

                if ((mshr_index == -1) && (MSHR.occupancy < MSHR_SIZE)) // this is a new miss'''
if late_needle not in s:
    raise SystemExit('[error] could not find pre-merge lateness marker')
s = s.replace(late_needle, late_insert, 1)

miss_handled_needle = '''                if (miss_handled) 
                {
                    // update prefetcher on load instruction'''
miss_handled_insert = '''                if (miss_handled) 
                {
                    // Log only a completed demand callback.  A stalled RQ retry must
                    // not create a second model timestep for the same load.
                    if (cache_type == IS_L2C && warmup_complete[read_cpu] && RQ.entry[index].type == LOAD) {
                        demand_logger_trigger_id = demand_event_log_demand(
                            this, read_cpu, current_core_cycle[read_cpu], "read", RQ.entry[index].type,
                            RQ.entry[index].ip, RQ.entry[index].full_addr, RQ.entry[index].address,
                            0, 0, demand_logger_late);
                    }

                    // update prefetcher on load instruction'''
if miss_handled_needle not in s:
    raise SystemExit('[error] could not find handled read-miss marker')
s = s.replace(miss_handled_needle, miss_handled_insert, 1)

miss_operate_needle = '''                        if (cache_type == IS_L2C)
                        {
                            l2c_prefetcher_operate(RQ.entry[index].address<<LOG2_BLOCK_SIZE, RQ.entry[index].ip, 0, RQ.entry[index].type, 0);
                        }'''
miss_operate_insert = '''                        if (cache_type == IS_L2C)
                        {
                            demand_event_begin_trigger(demand_logger_trigger_id, read_cpu,
                                                       RQ.entry[index].ip, RQ.entry[index].address);
                            l2c_prefetcher_operate(RQ.entry[index].address<<LOG2_BLOCK_SIZE, RQ.entry[index].ip, 0, RQ.entry[index].type, 0);
                            demand_event_end_trigger();
                        }'''
if miss_operate_needle not in s:
    raise SystemExit('[error] could not find read-miss L2 prefetcher call')
s = s.replace(miss_operate_needle, miss_operate_insert, 1)

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

grep -n "DEMAND_EVENT_LOG_SCHEMA_623_V5\|demand_event_begin_trigger" "$CACHE_CC" || true
