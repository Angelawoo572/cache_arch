#!/usr/bin/env python3
"""Layer the new four-route protocol onto a fresh isolated, reproducible build.

Old source, binaries and result protocols remain unchanged. Every anchor is
checked before use. The LLC request cohort has its own observer and ID domain.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil

EXP=Path(__file__).resolve().parent
ROOT=EXP.parents[2]
OLD=EXP.parent/'602_lstm_stride_dynamic_inference'


def once(text,old,new,label):
    if text.count(old)!=1:raise RuntimeError('{}: expected1 anchor, found{}'.format(label,text.count(old)))
    return text.replace(old,new,1)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def observer_source():
    s=(OLD/'measurement/dynamic_observer.cc').read_text()
    s='#include "four602_engine.h"\n#include "four602_cohort.h"\n'+s
    s=once(s,'    std::unique_ptr<Observer> obs;','    std::unique_ptr<Observer> obs, llc;','LLC observer storage')
    s=once(s,'    std::unique_ptr<Observer> obs, llc;', '    std::unique_ptr<Observer> obs, llc;\n    four602_measurement::LLCShortcuts llc_shortcuts;', 'LLC handoff counters')
    marker='uint64_t phase(unsigned cpu) {'
    helper='''Observer* target_observer(CACHE* c) {
    if(!states[c->cpu].began)return nullptr;
    if(c->cache_type==IS_L2C)return states[c->cpu].obs.get();
    if(c->cache_type==IS_LLC)return states[c->cpu].llc.get();
    return nullptr;
}
bool llc_id(uint64_t id){return four602_measurement::is_llc_id(id);}
Observer* request_observer(CACHE* c,uint64_t id) {
    if(!states[c->cpu].began||!id)return nullptr;
    const auto target=four602_measurement::request_target(c->cache_type==IS_L2C,c->cache_type==IS_LLC,id);
    if(target==four602_measurement::LLC_TARGET)return states[c->cpu].llc.get();
    return target==four602_measurement::L2_TARGET?states[c->cpu].obs.get():nullptr;
}
'''
    s=once(s,marker,helper+marker,'observer selection')
    marker='void thresholds(unsigned cpu) {'
    helper='''void llc_json(std::ostream& out,unsigned cpu) {
    const Observer& o=*states[cpu].llc;dynamic602_measurement::Cohort total;
    for(const auto& item:o.cohorts) {
        const auto& c=item.second;
        total.enqueued+=c.enqueued;total.timely+=c.timely;total.late+=c.late;
        total.timely_load+=c.timely_load;total.late_load+=c.late_load;
        total.timely_rfo+=c.timely_rfo;total.late_rfo+=c.late_rfo;
        total.unused+=c.unused;total.redundant_cache+=c.redundant_cache;
        total.redundant_inflight+=c.redundant_inflight;total.pending+=c.pending;
        total.resident_unused+=c.resident_unused;
    }
    if(total.enqueued!=total.timely+total.late+total.unused+total.redundant_cache+
       total.redundant_inflight+total.pending+total.resident_unused ||
       total.pending+total.resident_unused!=o.pending.size())
        throw std::runtime_error("LLC target lifecycle conservation mismatch");
    out<<",\\"llc_target_cohort\\":{";cohort_json(out,total);out<<'}';
    out<<",\\"llc_target_classified_cohorts\\":[";bool first=true;
    for(const auto& item:o.classified){if(!first)out<<',';first=false;
        out<<"{\\"group\\":"<<quoted(item.first)<<",\\"cohort\\":{";
        cohort_json(out,item.second.cohort);out<<"}}";
    }out<<']';
    const auto& shortcuts=states[cpu].llc_shortcuts;
    out<<",\\"llc_target_write_queue_forwarded\\":"<<shortcuts.write_queue_forwarded
       <<",\\"llc_target_prefetch_queue_merged\\":"<<shortcuts.prefetch_queue_merged
       <<",\\"llc_target_prefetch_queue_rejected_untracked\\":"<<shortcuts.prefetch_queue_rejected
       <<",\\"llc_target_shortcut_missing_pending_id\\":"<<shortcuts.missing_pending_id
       <<",\\"llc_target_shortcut_semantics\\":\\"WQ_forward_and_PQ_merge_are_source_request_redundant_inflight; not_useful_LOAD_or_new_DRAM_transfer\\"";
    out<<",\\"LLC_LOAD_accesses\\":"<<uncore.LLC.sim_access[cpu][LOAD]
       <<",\\"LLC_LOAD_misses\\":"<<uncore.LLC.sim_miss[cpu][LOAD]
       <<",\\"llc_target_cohort_scope\\":\\"first_L2_PQ_insert_requested_LLC; followed_at_LLC_or_resolved_as_source_L2_redundancy_or_LLC_WQ_forward_or_PQ_merge; rejection_or_missing_ID_disables_complete_tracking; LOAD_consumers_include_all_LLC_LOAD_packets\\"";
}
'''
    # C++ quoted JSON strings are generated literally, not interpreted as code.
    s=once(s,marker,helper+marker,'LLC summary')
    marker='    s.obs->peak=s.obs->occupancy;'
    addition='''
    CACHE& l=uncore.LLC;
    s.llc.reset(new Observer(uint64_t(l.NUM_SET)*l.NUM_WAY));
    s.llc->next_id=four602_measurement::LLC_ID_FIRST;
    for(unsigned set=0;set<l.NUM_SET;++set)for(unsigned way=0;way<l.NUM_WAY;++way)
        if(l.block[set][way].valid)++s.llc->occupancy;
    s.llc->peak=s.llc->occupancy;'''
    s=once(s,marker,marker+addition,'LLC warm boundary')
    marker='        classified_json(s.out,o);'
    s=once(s,marker,marker+'\n        llc_json(s.out,cpu);\n        const std::string four=four602_engine_json(cpu);if(!four.empty())s.out<<\',\'<<four;','new summaries')
    s=once(s,'l2_target_fully_tracked_only','separate_L2_and_LLC_target_cohorts','cohort scope')
    s=once(s,'<< (s.scope_pq_inserted[1]?"false":"true")','<< (s.llc_shortcuts.fully_tracked()?"true":"false")','full tracking availability')
    s=once(s,'<< ",\\"llc_only_lifecycle_accuracy_available\\":false"','<< ",\\"llc_only_lifecycle_accuracy_available\\":" << (s.llc_shortcuts.fully_tracked()?"true":"false")','LLC accuracy availability')
    s=once(s,'cohort_llc_only_excluded','l2_cohort_llc_target_excluded','L2 excluded scope')
    start=s.index('void dynamic602_fill(');stop=s.index('\nvoid dynamic602_skip(',start)
    s=s[:start]+'''void dynamic602_fill(CACHE* c,unsigned set,unsigned way,const PACKET* packet) {
    Observer* o=target_observer(c);if(!o)return;
    uint64_t id=packet->dynamic602_measurement_id;
    if((c->cache_type==IS_LLC)!=llc_id(id))id=0;
    o->fill(uint64_t(set)*c->NUM_WAY+way,c->block[set][way].valid,packet->type,id,cycles(c->cpu));
    if(c->cache_type==IS_L2C)thresholds(c->cpu);
}
void dynamic602_invalidate(CACHE* c,unsigned set,unsigned way) {
    Observer* o=target_observer(c);if(o)o->invalidate(uint64_t(set)*c->NUM_WAY+way,cycles(c->cpu));
}
void dynamic602_demand_hit(CACHE* c,unsigned set,unsigned way,unsigned type) {
    Observer* o=target_observer(c);if(o)o->demand(uint64_t(set)*c->NUM_WAY+way,cycles(c->cpu),type);
}
void dynamic602_late(CACHE* c,const PACKET* packet,unsigned type) {
    uint64_t id=packet->dynamic602_measurement_id;
    if((c->cache_type==IS_LLC)!=llc_id(id))return;
    Observer* o=request_observer(c,id);if(o)o->resolve(id,1,cycles(c->cpu),type);
}
void dynamic602_enqueue(CACHE* c,PACKET* packet) {
    if(!active(c))return;
    State& s=states[c->cpu];bool l2=dynamic602_measurement::fully_tracked_l2(packet->fill_level,FILL_L2);
    ++s.request_pq_inserted;++s.scope_pq_inserted[l2?0:1];
    if(!s.current_classification.empty())++s.obs->classified.at(s.current_classification).pq_enqueued;
    Observer& o=l2?*s.obs:*s.llc;
    if(!l2&&!s.current_classification.empty())o.classified[s.current_classification].label=s.obs->classified.at(s.current_classification).label;
    if((l2&&o.next_id>=four602_measurement::LLC_ID_FIRST)||(!l2&&o.next_id<four602_measurement::LLC_ID_FIRST))
        throw std::runtime_error("request ID domain exhausted");
    packet->dynamic602_measurement_id=o.enqueue(instructions(c->cpu),s.current_classification);
}
void dynamic602_redundant(CACHE* c,const PACKET* packet,bool cache_hit) {
    if(c->cache_type!=IS_L2C&&c->cache_type!=IS_LLC)return;
    Observer* o=request_observer(c,packet->dynamic602_measurement_id);
    if(o)o->resolve(packet->dynamic602_measurement_id,cache_hit?3:4,cycles(c->cpu));
}
void dynamic602_llc_shortcut(CACHE* c,const PACKET* packet,unsigned reason) {
    if(c->cache_type!=IS_LLC||!states[c->cpu].began)return;
    auto& state=states[c->cpu];
    four602_measurement::llc_shortcut(*state.llc,state.llc_shortcuts,packet->dynamic602_measurement_id,reason,cycles(c->cpu));
}
''' + s[stop:]
    anchor='bool dynamic602_allow_request(CACHE* cache,uint64_t ip,uint64_t base,uint64_t address,int fill_level) {'
    s=once(s,anchor,anchor+'\n    four602_note_request(cache,ip,base,address,fill_level);','native direct output context')
    return s


def cache_handoff_source(source):
    """Instrument early LLC add_pq exits without changing queue scheduling."""
    start=source.index('int CACHE::add_pq(PACKET *packet)')
    stop=source.index('\nvoid CACHE::return_data(',start)
    body=source[start:stop]
    body=once(body,'        WQ.FORWARD++;',
        '        dynamic602_llc_shortcut(this, packet, 1);\n        WQ.FORWARD++;','LLC WQ-forward resolution')
    body=once(body,'        PQ.MERGED++;',
        '        dynamic602_llc_shortcut(this, packet, 2);\n        PQ.MERGED++;','LLC PQ-merge resolution')
    body=once(body,'        PQ.FULL++;',
        '        dynamic602_llc_shortcut(this, packet, 3);\n        PQ.FULL++;','LLC PQ-rejection accounting')
    declaration='class CACHE; class PACKET;\nvoid dynamic602_llc_shortcut(CACHE*, const PACKET*, unsigned);\n'
    return declaration+source[:start]+body+source[stop:]


def install(sim):
    sim=Path(sim).resolve()
    load('old_install',OLD/'patches/install.py').install(sim)
    (sim/'src/dynamic_observer.cc').write_text(observer_source())
    cache=sim/'src/cache.cc';cache.write_text(cache_handoff_source(cache.read_text()))
    helper=sim/'prefetcher/spp_dev2_helper.cc'
    text=helper.read_text()
    text='#include <stdexcept>\n'+text
    guard='if(pf_q_tail>=L2C_MSHR_SIZE)throw std::runtime_error("SPP prediction queue exceeded native capacity");'
    text=once(text,'                confidence_q[pf_q_tail] = pf_conf;',
        '                '+guard+'\n                confidence_q[pf_q_tail] = pf_conf;','SPP prediction queue positive write guard')
    text=once(text,'    } else confidence_q[pf_q_tail] = 0;',
        '    } else { '+guard+' confidence_q[pf_q_tail] = 0; }','SPP prediction queue zero write guard')
    helper.write_text(text)
    spp=sim/'prefetcher/spp_dev2.cc';text=spp.read_text()
    text=once(text,'(delta_q[i] << LOG2_BLOCK_SIZE)',
        '(static_cast<uint64_t>(delta_q[i]) * (uint64_t(1) << LOG2_BLOCK_SIZE))','SPP defined signed-delta address arithmetic')
    text=once(text,'(PT.delta[set][lookahead_way] << LOG2_BLOCK_SIZE)',
        '(static_cast<uint64_t>(PT.delta[set][lookahead_way]) * (uint64_t(1) << LOG2_BLOCK_SIZE))','SPP defined signed lookahead arithmetic')
    spp.write_text(text)
    for parent in [EXP/'runtime',ROOT/'formal_NN_training/common/live602_runtime']:
        for p in parent.glob('*'):
            if p.suffix not in ('.h','.cc') or p.name=='probe.cc':continue
            shutil.copy2(p,sim/('inc' if p.suffix=='.h' else 'prefetcher')/p.name)
    main=(sim/'src/main.cc').read_text()
    main='#include "four602_engine.h"\n'+main
    main=once(main,'        dynamic602_roi(&ooo_cpu[dynamic602_cpu].L2C);','        dynamic602_roi(&ooo_cpu[dynamic602_cpu].L2C);\n        four602_roi(&ooo_cpu[dynamic602_cpu].L2C);','four ROI boundary')
    (sim/'src/main.cc').write_text(main)
    registry=(sim/'prefetcher/dynamic602.l2c_pref').read_text()
    registry='#include "four602_engine.h"\n'+registry
    anchor='\t\telse if(!knob::l2c_prefetcher_types[index].compare("stride"))'
    extra='''\t\telse if(!knob::l2c_prefetcher_types[index].compare("four602"))
        {
            Four602* p=new Four602("four602",this);prefetchers.push_back(p);
        }
'''
    registry=once(registry,anchor,extra+anchor,'four registration')
    anchor='uint32_t CACHE::l2c_prefetcher_cache_fill(uint64_t addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_addr, uint32_t metadata_in)\n{'
    extra='''
    // Raw callbacks at both actual CACHE L2-fill sites, including demand fills.
    for(auto* p:prefetchers)if(p->get_type()=="four602")
        static_cast<Four602*>(p)->cache_fill(addr,set,way,prefetch,evicted_addr);
'''
    registry=once(registry,anchor,anchor+extra,'actual L2 fill dispatch')
    (sim/'prefetcher/four602.l2c_pref').write_text(registry)
    (sim/'.four602_protocol').write_text(json.loads((EXP/'config.json').read_text())['protocol']+'\n')
    print('Installed separate four-route MAC0 protocol',sim)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('simulator',type=Path);install(p.parse_args().simulator)
