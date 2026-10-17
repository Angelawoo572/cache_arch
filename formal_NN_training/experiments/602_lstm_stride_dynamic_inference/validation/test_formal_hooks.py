#!/usr/bin/env python3
"""Exercise the real measurement hook implementation, without a trace experiment."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

EXP=Path(__file__).resolve().parents[1]
STUBS={
'cache.h':r'''#pragma once
#include <cstdint>
const unsigned NUM_CPUS=1,DRAM_CHANNELS=1,LOG2_BLOCK_SIZE=6,LOG2_PAGE_SIZE=12;
const unsigned IS_L2C=2,LOAD=0,RFO=1,PREFETCH=2,WRITEBACK=3;
const int FILL_L2=2,FILL_LLC=4;
extern uint64_t current_core_cycle[1];extern uint8_t warmup_complete[1];
struct input_instr { uint64_t x[4]; };struct cloudsuite_instr { uint64_t x[8]; };
struct PACKET { unsigned type=0;int fill_level=FILL_L2;uint64_t dynamic602_measurement_id=0; };
struct CACHE {unsigned cpu=0,cache_type=IS_L2C,NUM_SET=1,NUM_WAY=1;
 struct Block{bool valid=false;} block[1][1];
 struct Queue{uint64_t MERGED=0,occupancy=0;}PQ,MSHR;
 uint64_t sim_access[1][4]={},sim_miss[1][4]={},ACCESS[4]={},MISS[4]={};
 uint64_t pf_requested=0,pf_issued=0,pf_useful=0,pf_late=0,pf_filled=0,pf_useless=0,pf_dropped=0;
};
''',
'ooo_cpu.h':r'''#pragma once
#include "cache.h"
struct OOO_CPU { CACHE L2C;uint64_t num_retired=0,begin_sim_instr=0,begin_sim_cycle=0;};extern OOO_CPU ooo_cpu[1];
''',
'uncore.h':r'''#pragma once
#include "cache.h"
struct UNCORE{struct DRAM_TYPE{struct Queue{uint64_t occupancy=0;}RQ[1],WQ[1];uint64_t dbus_cycle_available[1]={};}DRAM;};
'''
}
PROBE=r'''
#include "dynamic602_hooks.h"
#include "ooo_cpu.h"
#include "uncore.h"
#include <stdexcept>
uint64_t current_core_cycle[1]={0};uint8_t warmup_complete[1]={0};OOO_CPU ooo_cpu[1];UNCORE uncore;
std::string dynamic602_engine_json(unsigned){return "";}
int main(){
 CACHE& c=ooo_cpu[0].L2C;
 if(dynamic602_allow_request(&c,1,4096,4160,FILL_L2))throw std::runtime_error("warmup output escaped");
 current_core_cycle[0]=10;warmup_complete[0]=1;ooo_cpu[0].num_retired=100;ooo_cpu[0].begin_sim_instr=100;ooo_cpu[0].begin_sim_cycle=10;
 uncore.DRAM.RQ[0].occupancy=2;uncore.DRAM.WQ[0].occupancy=1;uncore.DRAM.dbus_cycle_available[0]=20;
 dynamic602_begin(0);
 ooo_cpu[0].num_retired=110;
 if(!dynamic602_allow_request(&c,1,4096,4160,FILL_L2))throw std::runtime_error("available credit rejected");
 dynamic602_cache_request_result(&c,-1);
 if(dynamic602_allow_request(&c,1,4096,4224,FILL_L2))throw std::runtime_error("request cap escaped");
 if(dynamic602_allow_request(&c,1,4096,8192,FILL_L2))throw std::runtime_error("page gate escaped");
 // Exactly one cache result follows each allowed call, as in actual prefetch_line.
 ooo_cpu[0].num_retired=120;
 if(!dynamic602_allow_request(&c,1,4096,4288,FILL_L2))throw std::runtime_error("second credit rejected");
 dynamic602_cache_request_result(&c,0);
 ooo_cpu[0].num_retired=130;
 if(!dynamic602_allow_request(&c,1,4096,4352,FILL_L2))throw std::runtime_error("third credit rejected");
 dynamic602_cache_request_result(&c,-2);
 dynamic602_dram_event(0,0,LOAD);
 current_core_cycle[0]=20;
 dynamic602_dram_bus_start(0,false,LOAD,30);
 dynamic602_dram_bus_start(0,false,PREFETCH,31);
 dynamic602_dram_bus_start(0,true,WRITEBACK,32);
 uncore.DRAM.RQ[0].occupancy=1;uncore.DRAM.WQ[0].occupancy=0;
 current_core_cycle[0]=29;dynamic602_observe_cycle(0,true);
 current_core_cycle[0]=30;dynamic602_observe_cycle(0,true);
 current_core_cycle[0]=32;dynamic602_observe_cycle(0,true);
}
'''

class FormalHooksTests(unittest.TestCase):
 def test_shared_budget_page_gate_and_exact_dram_bus_endpoint(self):
  with tempfile.TemporaryDirectory(prefix='formal602_hooks_',dir='/tmp') as temp:
   stage=Path(temp)
   for name,source in STUBS.items():(stage/name).write_text(source)
   (stage/'probe.cc').write_text(PROBE)
   command=[shutil.which('c++') or 'g++','-std=c++11','-O1','-I'+str(stage),'-I'+str(EXP/'measurement'),str(stage/'probe.cc'),str(EXP/'measurement/dynamic_observer.cc'),'-o',str(stage/'probe')]
   compiled=subprocess.run(command,text=True,capture_output=True)
   self.assertEqual(compiled.returncode,0,compiled.stdout+compiled.stderr)
   log=stage/'stats.jsonl'
   env=dict(os.environ,DYNAMIC602_STATS=str(log),DYNAMIC602_PAGE_POLICY='same_page',DYNAMIC602_FILTER_CURRENT='0',DYNAMIC602_REQUESTS_PER_MILLION_INSTRUCTIONS='100000')
   result=subprocess.run([str(stage/'probe')],env=env,text=True,capture_output=True)
   self.assertEqual(result.returncode,0,result.stdout+result.stderr)
   rows=[json.loads(line) for line in log.read_text().splitlines()]
   rows=[r for r in rows if r.get('event')=='snapshot']
   self.assertEqual(rows[0]['dram_initial_read_queue_pending'],2)
   self.assertEqual(rows[0]['dram_initial_write_queue_pending'],1)
   self.assertEqual(rows[0]['dram_initial_busy_channels'],1)
   before,first,final=rows[-3:]
   self.assertEqual(before['dram_total_bus_bytes'],0)
   self.assertEqual(before['dram_bus_transfers_in_progress'],3)
   self.assertEqual(first['dram_total_bus_bytes'],64)
   self.assertEqual(final['dram_read_bus_bytes'],128)
   self.assertEqual(final['dram_write_bus_bytes'],64)
   self.assertEqual(final['dram_total_bus_bytes'],192)
   self.assertEqual(final['dram_bus_transfers_in_progress'],0)
   self.assertEqual(final['dram_read_enqueued'],1)
   self.assertEqual(final['dram_read_queue_pending'],1)
   self.assertEqual(final['request_gate_generated'],5)
   self.assertEqual(final['request_gate_allowed'],3)
   self.assertEqual(final['request_gate_budget_suppressed'],1)
   self.assertEqual(final['request_gate_page_suppressed'],1)
   self.assertEqual(final['request_cache_enqueued'],1)
   self.assertEqual(final['request_cache_merged'],1)
   self.assertEqual(final['request_cache_rejected'],1)

if __name__=='__main__':unittest.main(verbosity=2)
