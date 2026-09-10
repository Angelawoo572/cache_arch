#include "dynamic602_hooks.h"
#include "dynamic_observer_core.h"
#include "cache.h"
#include "ooo_cpu.h"
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
using dynamic602_measurement::Observer;
uint64_t env_number(const char* name) {
    const char* value=std::getenv(name);
    if(!value || !*value) return 0;
    char* end=nullptr; errno=0;
    unsigned long long number=std::strtoull(value,&end,10);
    if(errno || *end || *value=='-') throw std::runtime_error(std::string("invalid ")+name);
    return number;
}
uint64_t skip_count() { static uint64_t n=env_number("DYNAMIC602_SKIP_RECORDS"); return n; }
uint64_t max_count() { static uint64_t n=env_number("DYNAMIC602_MAX_RECORDS"); return n; }
uint64_t records[NUM_CPUS] = {};
struct State {
    std::unique_ptr<Observer> obs;
    std::ofstream out;
    bool began=false;
    uint64_t next_sample=1000, last_sample=uint64_t(-1), origin=0;
    bool thresholds[5]={false,false,false,false,false};
};
State states[NUM_CPUS];
uint64_t instructions(unsigned cpu) { return ooo_cpu[cpu].num_retired-ooo_cpu[cpu].begin_sim_instr; }
uint64_t cycles(unsigned cpu) { return current_core_cycle[cpu]-ooo_cpu[cpu].begin_sim_cycle; }
bool active(CACHE* c) { return c->cache_type==IS_L2C && states[c->cpu].began; }
void thresholds(unsigned cpu) {
    State& s=states[cpu];
    static const unsigned percents[5]={50,90,95,99,100};
    for(unsigned i=0;i<5;++i) if(!s.thresholds[i] &&
        s.obs->occupancy*100 >= s.obs->slots.size()*percents[i]) {
        s.thresholds[i]=true;
        if(s.out) s.out << "{\"event\":\"occupancy_threshold\",\"percent\":" << percents[i]
            << ",\"instructions\":" << instructions(cpu) << ",\"cycles\":" << cycles(cpu)
            << ",\"occupancy_lines\":" << s.obs->occupancy << "}\n";
    }
}
}

void dynamic602_begin(unsigned cpu) {
    State& s=states[cpu];
    if(s.began) throw std::runtime_error("dynamic602 measurement boundary called twice");
    CACHE& c=ooo_cpu[cpu].L2C;
    s.obs.reset(new Observer(uint64_t(c.NUM_SET)*c.NUM_WAY));
    for(unsigned set=0;set<c.NUM_SET;++set) for(unsigned way=0;way<c.NUM_WAY;++way)
        if(c.block[set][way].valid) ++s.obs->occupancy;
    s.obs->peak=s.obs->occupancy;
    s.origin=skip_count()+ooo_cpu[cpu].begin_sim_instr;
    const char* path=std::getenv("DYNAMIC602_STATS");
    if(path && *path) {
        if(NUM_CPUS!=1) throw std::runtime_error("dynamic602 observer requires a single core");
        s.out.open(path);
        if(!s.out) throw std::runtime_error("cannot open DYNAMIC602_STATS");
    }
    s.began=true;
    thresholds(cpu);
    dynamic602_observe_cycle(cpu);
}

void dynamic602_observe_cycle(unsigned cpu,bool final) {
    State& s=states[cpu];
    if(!s.began) return;
    const uint64_t ins=instructions(cpu);
    if(!final && s.last_sample!=uint64_t(-1) && ins<s.next_sample) return;
    CACHE& c=ooo_cpu[cpu].L2C;
    Observer& o=*s.obs;
    // A sampled scan validates event-based occupancy against actual simultaneous
    // valid ways; cumulative fills never stand in for occupancy.
    uint64_t valid=0;
    for(unsigned set=0;set<c.NUM_SET;++set) for(unsigned way=0;way<c.NUM_WAY;++way)
        valid += c.block[set][way].valid ? 1 : 0;
    if(valid!=o.occupancy) throw std::runtime_error("dynamic602 valid-line occupancy mismatch");
    if(s.out) {
        dynamic602_measurement::Cohort outcomes;
        for(const auto& item:o.cohorts) {
            const auto& c=item.second;
            outcomes.enqueued+=c.enqueued; outcomes.timely+=c.timely;
            outcomes.late+=c.late; outcomes.unused+=c.unused;
            outcomes.redundant_cache+=c.redundant_cache;
            outcomes.redundant_inflight+=c.redundant_inflight;
            outcomes.pending+=c.pending; outcomes.resident_unused+=c.resident_unused;
        }
        if(outcomes.enqueued != outcomes.timely+outcomes.late+outcomes.unused+
            outcomes.redundant_cache+outcomes.redundant_inflight+outcomes.pending+
            outcomes.resident_unused || outcomes.pending+outcomes.resident_unused!=o.pending.size())
            throw std::runtime_error("dynamic602 prefetch lifecycle conservation mismatch");
        s.out << "{\"event\":\"snapshot\",\"final\":" << (final?"true":"false")
            << ",\"instructions\":" << ins << ",\"cycles\":" << cycles(cpu)
            << ",\"trace_origin_records\":" << s.origin
            << ",\"trace_records_read\":" << records[cpu]
            << ",\"NL\":" << c.sim_access[cpu][LOAD]
            << ",\"M\":" << c.sim_miss[cpu][LOAD]
            << ",\"callback_loads\":" << c.ACCESS[LOAD]
            << ",\"callback_load_misses\":" << c.MISS[LOAD]
            << ",\"P\":" << c.pf_requested << ",\"Ipf\":" << c.pf_issued
            << ",\"Q\":" << c.PQ.MERGED << ",\"U\":" << c.pf_useful
            << ",\"L\":" << c.pf_late << ",\"pf_filled\":" << c.pf_filled
            << ",\"pf_useless\":" << c.pf_useless << ",\"pf_dropped\":" << c.pf_dropped
            << ",\"l2_sets\":" << c.NUM_SET << ",\"l2_ways\":" << c.NUM_WAY
            << ",\"line_bytes\":" << (uint64_t(1)<<LOG2_BLOCK_SIZE)
            << ",\"capacity_lines\":" << o.slots.size()
            << ",\"capacity_bytes\":" << (o.slots.size()<<LOG2_BLOCK_SIZE)
            << ",\"occupancy_lines\":" << o.occupancy << ",\"occupancy_peak\":" << o.peak
            << ",\"fills_load\":" << o.fills[LOAD] << ",\"fills_rfo\":" << o.fills[RFO]
            << ",\"fills_prefetch\":" << o.fills[PREFETCH]
            << ",\"fills_writeback\":" << o.fills[WRITEBACK]
            << ",\"replacements\":" << o.replacements << ",\"invalidations\":" << o.invalidations
            << ",\"line_lifetimes\":" << o.line_lifetimes
            << ",\"line_lifetime_cycles\":" << o.line_lifetime_cycles
            << ",\"useful_line_lifetimes\":" << o.useful_line_lifetimes
            << ",\"useful_line_lifetime_cycles\":" << o.useful_line_lifetime_cycles
            << ",\"prefetch_useful_lifetimes\":" << o.prefetch_useful_lifetimes
            << ",\"prefetch_fill_to_first_demand_cycles\":" << o.prefetch_fill_to_first_demand_cycles
            << ",\"pq_occupancy\":" << c.PQ.occupancy << ",\"mshr_occupancy\":" << c.MSHR.occupancy
            << ",\"cohort_enqueued\":" << outcomes.enqueued
            << ",\"cohort_timely\":" << outcomes.timely << ",\"cohort_late\":" << outcomes.late
            << ",\"cohort_unused\":" << outcomes.unused
            << ",\"cohort_redundant_cache\":" << outcomes.redundant_cache
            << ",\"cohort_redundant_inflight\":" << outcomes.redundant_inflight
            << ",\"cohort_pending\":" << outcomes.pending
            << ",\"cohort_resident_unused\":" << outcomes.resident_unused
            << ",\"measurement_unresolved_lifecycles\":" << o.pending.size()
            << ",\"measurement_unresolved_lifecycles_peak\":" << o.unresolved_peak;
        const std::string engine=dynamic602_engine_json(cpu);
        if(!engine.empty()) s.out << ',' << engine;
        s.out << "}\n";
        if(final) for(const auto& item:o.cohorts) {
            const auto& v=item.second;
            s.out << "{\"event\":\"cohort\",\"start_instructions\":" << item.first
                << ",\"enqueued\":" << v.enqueued << ",\"timely\":" << v.timely
                << ",\"late\":" << v.late << ",\"unused\":" << v.unused
                << ",\"redundant_cache\":" << v.redundant_cache
                << ",\"redundant_inflight\":" << v.redundant_inflight
                << ",\"pending\":" << v.pending << ",\"resident_unused\":" << v.resident_unused << "}\n";
        }
        s.out.flush();
    }
    s.last_sample=ins;
    s.next_sample=ins<1000?1000:ins<10000?10000:(ins/100000+1)*100000;
}

void dynamic602_fill(CACHE* c,unsigned set,unsigned way,const PACKET* packet) {
    if(!active(c)) return;
    states[c->cpu].obs->fill(uint64_t(set)*c->NUM_WAY+way,c->block[set][way].valid,
                            packet->type,packet->dynamic602_measurement_id,cycles(c->cpu));
    thresholds(c->cpu);
}
void dynamic602_invalidate(CACHE* c,unsigned set,unsigned way) {
    if(active(c)) states[c->cpu].obs->invalidate(uint64_t(set)*c->NUM_WAY+way,cycles(c->cpu));
}
void dynamic602_demand_hit(CACHE* c,unsigned set,unsigned way) {
    if(active(c)) states[c->cpu].obs->demand(uint64_t(set)*c->NUM_WAY+way,cycles(c->cpu));
}
void dynamic602_late(CACHE* c,const PACKET* packet) {
    if(active(c)) states[c->cpu].obs->resolve(packet->dynamic602_measurement_id,1,cycles(c->cpu));
}
void dynamic602_enqueue(CACHE* c,PACKET* packet) {
    if(active(c)) packet->dynamic602_measurement_id=states[c->cpu].obs->enqueue(instructions(c->cpu));
}
void dynamic602_redundant(CACHE* c,const PACKET* packet,bool cache_hit) {
    if(active(c)) states[c->cpu].obs->resolve(packet->dynamic602_measurement_id,cache_hit?3:4,cycles(c->cpu));
}

void dynamic602_skip(FILE* file,bool cloudsuite,unsigned cpu) {
    const size_t record_bytes=cloudsuite?sizeof(cloudsuite_instr):sizeof(input_instr);
    // Opaque records are discarded, never converted to instructions or predictor
    // observations. sizeof is taken from the exact reader's compiled structures.
    if(cloudsuite) dynamic602_measurement::skip_exact_records<cloudsuite_instr>(file,skip_count());
    else dynamic602_measurement::skip_exact_records<input_instr>(file,skip_count());
    std::cout << "dynamic602_trace_skip_records " << skip_count()
              << " reader_record_bytes " << record_bytes << " cpu " << cpu << std::endl;
}
bool dynamic602_read_allowed(unsigned cpu) { return !max_count() || records[cpu]<max_count(); }
void dynamic602_record_read(unsigned cpu) { ++records[cpu]; }
bool dynamic602_bounded_trace() { return max_count()!=0; }
