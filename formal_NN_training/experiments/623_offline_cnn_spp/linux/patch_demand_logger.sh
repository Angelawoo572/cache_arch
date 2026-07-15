#!/usr/bin/env bash
# Patch local ChampSim/Pythia to emit a causally keyed L2 demand/PF log for
# the matched 623 SPP experiment.  Instrumentation only: it does not alter the
# prefetch policy or create labels.
#
# The v6 schema records the exact demand event whose synchronous L2 callback
# issued each prefetch, plus both L2 cache-fill call sites whose evicted_addr is
# an external SPP feedback input.  This avoids guessing from base_addr or from
# a future demand.
# A logger from an older version of this experiment is backed up and replaced
# automatically.  Unknown cache.cc edits still fail closed.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"
CHAMP_DIR="${CHAMP_DIR:-$ROOT/external/ChampSim}"
CACHE_CC="$CHAMP_DIR/src/cache.cc"

[[ -f "$CACHE_CC" ]] || { echo "[error] missing $CACHE_CC" >&2; exit 2; }
CACHE_CC_CLEAN_RESTORED=0

restore_clean_cache_cc() {
  local reason="$1"
  local save_dir="${RUN_DIR:-/tmp}"
  local commit=""
  local tmp=""

  mkdir -p "$save_dir"
  cp -f "$CACHE_CC" "$save_dir/cache.cc.before_623_reset.cc"
  git -C "$CHAMP_DIR" diff -- src/cache.cc \
    > "$save_dir/cache.cc.before_623_reset.patch" || true

  # HEAD itself can contain an older experiment's logger.  Restore the newest
  # historical cache.cc blob with no known instrumentation marker instead of
  # trusting `git checkout -- src/cache.cc`.
  tmp="$(mktemp "$save_dir/cache.cc.clean.XXXXXX")"
  while IFS= read -r commit; do
    [[ -n "$commit" ]] || continue
    git -C "$CHAMP_DIR" show "$commit:src/cache.cc" > "$tmp"
    if ! grep -Eq 'DEMAND_EVENT_LOG_SCHEMA_623_|demand_event_log_demand|RESIDUAL_AUDIT_LOG' "$tmp"; then
      mv -f "$tmp" "$CACHE_CC"
      CACHE_CC_CLEAN_RESTORED=1
      printf '%s\n' "$commit" > "$save_dir/cache.cc.clean_source_commit"
      echo "[clean-restore] $reason"
      echo "[clean-restore] restored src/cache.cc from $commit"
      echo "[clean-restore] original file: $save_dir/cache.cc.before_623_reset.cc"
      return 0
    fi
  done < <(git -C "$CHAMP_DIR" log --format=%H -- src/cache.cc)

  rm -f "$tmp"
  echo "[error] no logger-free src/cache.cc blob exists in ChampSim history" >&2
  echo "[hint] inspect: git -C \"$CHAMP_DIR\" log --oneline -- src/cache.cc" >&2
  return 2
}

if [[ "${RESET_PATCH:-0}" == "1" ]]; then
  restore_clean_cache_cc "RESET_PATCH=1 requested"
elif ! grep -Fq "DEMAND_EVENT_LOG_SCHEMA_623_V6" "$CACHE_CC" \
    && grep -Eq 'DEMAND_EVENT_LOG_SCHEMA_623_|demand_event_log_demand|RESIDUAL_AUDIT_LOG' "$CACHE_CC"; then
  echo "[auto-reset] known older experimental cache.cc logger detected"
  restore_clean_cache_cc "replacing stale/foreign experimental logger"
fi

if [[ "$CACHE_CC_CLEAN_RESTORED" != "1" ]] \
    && ! grep -Fq "DEMAND_EVENT_LOG_SCHEMA_623_V6" "$CACHE_CC" \
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
revision_marker = "DEMAND_EVENT_LOG_SCHEMA_623_V6"
if revision_marker in s:
    print("[skip] 623 v6 demand/fill causal logger already present", p)
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
// DEMAND_EVENT_LOG_SCHEMA_623_V6
// -----------------------------------------------------------------------------
// Causally keyed raw L2 demand/PF logger for the split 623 experiments.
// Enabled only when DEMAND_EVENT_LOG is set.  Single-core experiment only.
// -----------------------------------------------------------------------------
static std::ofstream demand_event_log;
static bool demand_event_log_ready = false;
static uint64_t demand_event_id = 0;
static const uint64_t demand_event_no_trigger = uint64_t(-1);
static const char demand_event_logger_schema[] = "623_causal_trigger_fill_v6";
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

static void demand_event_log_fill(const CACHE* cache, uint32_t cpu, uint64_t cycle,
                                  uint32_t type, uint64_t filled_addr,
                                  uint64_t evicted_addr, uint32_t was_prefetch)
{
    demand_event_open_once();
    if (!demand_event_log.is_open())
        return;
    demand_event_log << "FILL"
                     << ',' << demand_event_id++ << ',' << cpu << ',' << cycle << ',' << cache->NAME
                     << ',' << "cache_fill" << ',' << type << ',' << 0 << ',' << evicted_addr
                     << ',' << (evicted_addr >> LOG2_BLOCK_SIZE) << ',' << 0
                     << ',' << was_prefetch << ',' << 0 << ',' << 0 << ',' << 0
                     << ',' << filled_addr << ',' << 0 << ',' << 0 << ',' << 0
                     << ',' << cache->PQ.occupancy << ',' << cache->PQ.SIZE
                     << ',' << cache->MSHR.occupancy << ',' << cache->MSHR.SIZE
                     << ',' << demand_event_no_trigger << ',' << 0 << ',' << 0 << ',' << 0
                     << ',' << demand_event_logger_schema << std::endl;
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

# SPP_dev2::cache_fill consumes evicted_addr at both L2 fill call sites.  Log
# the external input immediately before each callback, preserving source order.
mshr_fill_needle = "MSHR.entry[mshr_index].pf_metadata = l2c_prefetcher_cache_fill("
if s.count(mshr_fill_needle) != 1:
    raise SystemExit('[error] could not uniquely find MSHR L2 cache-fill call')
s = s.replace(
    mshr_fill_needle,
    '''if (warmup_complete[fill_cpu]) {
                    demand_event_log_fill(
                        this, fill_cpu, current_core_cycle[fill_cpu],
                        MSHR.entry[mshr_index].type,
                        MSHR.entry[mshr_index].address << LOG2_BLOCK_SIZE,
                        block[set][way].address << LOG2_BLOCK_SIZE,
                        (MSHR.entry[mshr_index].type == PREFETCH) ? 1 : 0);
                }
                ''' + mshr_fill_needle,
    1,
)

wq_fill_needle = "WQ.entry[index].pf_metadata = l2c_prefetcher_cache_fill("
if s.count(wq_fill_needle) != 1:
    raise SystemExit('[error] could not uniquely find WQ L2 cache-fill call')
# The original branch is a braceless else-if.  Insert one compound statement
# by replacing the callback assignment through its terminating semicolon.
wq_start = s.index(wq_fill_needle)
wq_statement_end = s.index(';', wq_start) + 1
line_start = s.rfind('\n', 0, wq_start) + 1
indent = s[line_start:wq_start]
wq_statement = s[wq_start:wq_statement_end]
wq_replacement = (
    "{\n"
    + indent + "    if (warmup_complete[writeback_cpu]) {\n"
    + indent + "        demand_event_log_fill(\n"
    + indent + "            this, writeback_cpu, current_core_cycle[writeback_cpu],\n"
    + indent + "            WQ.entry[index].type,\n"
    + indent + "            WQ.entry[index].address << LOG2_BLOCK_SIZE,\n"
    + indent + "            block[set][way].address << LOG2_BLOCK_SIZE, 0);\n"
    + indent + "    }\n"
    + indent + "    " + wq_statement + "\n"
    + indent + "}"
)
s = s[:line_start] + indent + wq_replacement + s[wq_statement_end:]

p.write_text(s)
print('[patched]', p)
PY

python3 - "$CACHE_CC" <<'PY'
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text(errors="ignore")
expected = {
    "DEMAND_EVENT_LOG_SCHEMA_623_V6": 1,
    '"623_causal_trigger_fill_v6"': 1,
    "demand_event_log_demand(": 3,
    "demand_event_log_fill(": 3,
    "demand_event_begin_trigger(": 3,
    "demand_event_end_trigger(": 3,
    "demand_event_log_pf(": 2,
}
bad = {marker: (text.count(marker), count) for marker, count in expected.items()
       if text.count(marker) != count}
if bad:
    raise SystemExit("[error] installed 623 v6 logger structural audit failed: {}".format(bad))
print("[PASS] installed 623 v6 demand/fill logger structural audit")
PY
