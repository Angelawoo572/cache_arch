#!/usr/bin/env python3
"""Compile generated source and exercise measurement units; no simulator ROI."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

EXP=Path(__file__).resolve().parents[1]
ROOT=EXP.parents[2]


def run(simulator,out):
    simulator=Path(simulator).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    cache_root=Path(os.environ['TMPDIR']).resolve()
    if not str(out).startswith(str(cache_root.parent.parent)):
        raise RuntimeError('Keep fixture outputs inside this cache study runtime')
    spec=importlib.util.spec_from_file_location('checked_four_install',EXP/'install.py')
    installer=importlib.util.module_from_spec(spec);spec.loader.exec_module(installer)
    observer=installer.observer_source();cache=installer.cache_handoff_source((simulator/'src/cache.cc').read_text())
    (out/'dynamic_observer.cc').write_text(observer);(out/'cache.cc').write_text(cache)
    # These anchors come from actual CACHE source, not a synthetic simulator.
    assert observer.count('void dynamic602_llc_shortcut(')==1
    assert cache.count('dynamic602_llc_shortcut(this, packet,')==3
    assert 'packet->dynamic602_measurement_id=o.enqueue' in observer
    assert '(s.llc_shortcuts.fully_tracked()?"true":"false")' in observer
    assert 'l2_cohort_llc_target_excluded' in observer and 'cohort_llc_only_excluded' not in observer
    assert 'MSHR.entry[mshr_index] = RQ.entry[index]' in cache
    assert 'dynamic602_late(this, &MSHR.entry[mshr_index], RQ.entry[index].type);' in cache
    assert 'PQ.entry[index] = *packet;' in cache
    # Four's native SPP preserves prefetch-only feedback; student consumes the
    # original training logger's all-raw-fill events, a documented semantic gap.
    adapter=(EXP/'runtime/four602_engine.cc').read_text()
    assert 'if(spp_&&prefetch)spp_->cache_fill' in adapter
    assert 'model_->Infer(evicted&~uint64_t(63),false)' in adapter
    assert adapter.index('count("fill_callbacks");callback(evicted,false);')<adapter.index('if(spp_&&prefetch)')
    cxx=shutil.which('c++')
    if not cxx:raise RuntimeError('C++ compiler missing')
    includes=['-I'+str(EXP/'runtime'),'-I'+str(ROOT/'formal_NN_training/common/live602_runtime'),
              '-I'+str(simulator/'inc'),'-I'+str(simulator/'libbf')]
    commands=[]
    def execute(command):
        commands.append(command)
        p=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        with (out/'compiler.log').open('a') as f:f.write(p.stdout)
        if p.returncode:raise RuntimeError('Compiler failed; see '+str(out/'compiler.log'))
        return p.stdout
    (out/'compiler.log').write_text('')
    for source in [out/'dynamic_observer.cc',out/'cache.cc',EXP/'runtime/four602_engine.cc']:
        execute([cxx,'-std=c++11','-fsyntax-only',*includes,str(source)])
    # Execute the exact generated JSON helper with a minimal measurement-only
    # state to catch valid-C++ but invalid-JSON escaping or scope labels.
    json_functions=observer[observer.index('std::string quoted('):observer.index('void phase_counts_json(')]
    llc_function=observer[observer.index('void llc_json('):observer.index('void thresholds(')]
    json_fixture=out/'test_generated_json.cc'
    json_fixture.write_text('''#include "four602_cohort.h"
#include <memory>
#include <sstream>
#include <iostream>
using dynamic602_measurement::Observer;
static const unsigned LOAD=0;
struct State {std::unique_ptr<Observer> llc;four602_measurement::LLCShortcuts llc_shortcuts;};
State states[1];
struct LLCCounters {uint64_t sim_access[1][4]={{}},sim_miss[1][4]={{}};};
struct Uncore {LLCCounters LLC;};Uncore uncore;
'''+json_functions+llc_function+'''
int main(){auto& s=states[0];s.llc.reset(new Observer(1));s.llc->next_id=four602_measurement::LLC_ID_FIRST;
 auto a=s.llc->enqueue(0);four602_measurement::llc_shortcut(*s.llc,s.llc_shortcuts,a,1,10);
 auto b=s.llc->enqueue(0);s.llc->fill(0,false,2,b,20);s.llc->demand(0,30,0);
 std::cout<<"{\\\"base\\\":true";llc_json(std::cout,0);std::cout<<"}"<<std::endl;
}
''')
    json_binary=out/'test_generated_json'
    execute([cxx,'-std=c++11','-O0',*includes,str(json_fixture),'-o',str(json_binary)])
    generated_json=json.loads(execute([str(json_binary)]))
    assert generated_json['llc_target_cohort']['enqueued']==2
    assert generated_json['llc_target_cohort']['redundant_inflight']==1
    assert generated_json['llc_target_cohort']['timely_load']==1
    assert generated_json['llc_target_write_queue_forwarded']==1
    (out/'generated_llc_summary.json').write_text(json.dumps(generated_json,indent=2)+'\n')
    unit=out/'test_cohort'
    execute([cxx,'-std=c++11','-O0',*includes,str(EXP/'validation/test_cohort.cc'),'-o',str(unit)])
    result=json.loads(execute([str(unit)]));assert result['status']=='PASS'
    result.update(generated_full_observer_syntax='PASS',generated_actual_cache_syntax='PASS',
                  adapter_syntax='PASS',generated_llc_json_runtime='PASS',actual_cache_source=str(simulator/'src/cache.cc'),
                  fixture_root=str(out),commands=commands,no_simulator_execution=True)
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='commands'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--simulator',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.simulator,a.out)
