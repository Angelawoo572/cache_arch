#ifndef DYNAMIC602_OBSERVER_CORE_H
#define DYNAMIC602_OBSERVER_CORE_H
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <stdexcept>
#include <vector>

namespace dynamic602_measurement {
inline uint64_t phase_index(uint64_t instructions,uint64_t width,uint64_t total) {
    if(!width)throw std::runtime_error("measurement phase width must be positive");
    const uint64_t phase=instructions/width;
    return total?std::min<uint64_t>(phase,(total-1)/width):phase;
}
inline bool fully_tracked_l2(int requested_fill_level,int l2_fill_level) {
    return requested_fill_level<=l2_fill_level;
}
enum GateDecision { GATE_ALLOW, GATE_CURRENT, GATE_PAGE, GATE_BUDGET };
inline GateDecision gate_decision(bool filter_current,bool current,bool page_violation,
                                  uint64_t cap,uint64_t retired,uint64_t already_allowed) {
    if(filter_current&&current)return GATE_CURRENT;
    if(page_violation)return GATE_PAGE;
    if(cap>1000000000ULL)throw std::runtime_error("request cap exceeds supported domain");
    const uint64_t credits=(retired/1000000)*cap+((retired%1000000)*cap)/1000000;
    if(cap&&already_allowed>=credits)return GATE_BUDGET;
    return GATE_ALLOW;
}
template<class Record> void skip_exact_records(FILE* file,uint64_t remaining) {
    std::vector<Record> buffer(4096);
    while(remaining) {
        const size_t count=std::min<uint64_t>(remaining,buffer.size());
        if(std::fread(buffer.data(),sizeof(Record),count,file)!=count)
            throw std::runtime_error("trace ended inside requested skip region");
        remaining-=count;
    }
}
// All objects in this file are measurement-only storage, never predictor state.
struct Cohort {
    uint64_t enqueued=0, timely=0, late=0, unused=0;
    uint64_t timely_load=0,late_load=0,timely_rfo=0,late_rfo=0;
    uint64_t redundant_cache=0, redundant_inflight=0;
    uint64_t pending=0, resident_unused=0;
};
// Labels describe the originating emitted payload. Unknown native-prefetcher
// fields stay explicit rather than being assigned slot zero or teacher overlap.
struct RequestClass {
    std::string arm="unknown", fill_scope="l2_target";
    uint64_t phase=0;
    int64_t slot=-1;
    int current=0, duplicate=-1, teacher=-1;
    std::string key() const {
        auto label=[](int64_t n){return n<0?std::string("unknown"):std::to_string(n);};
        return "phase_"+std::to_string(phase)+"/slot_"+label(slot)+"/current_"+label(current)+
            "/duplicate_"+label(duplicate)+"/teacher_"+label(teacher);
    }
    std::string group() const { return arm+"/"+fill_scope+"/"+key(); }
};
struct Classified {
    RequestClass label;
    uint64_t attempted=0,allowed=0,current_suppressed=0,page_suppressed=0,budget_suppressed=0;
    uint64_t cache_enqueued=0,cache_merged=0,cache_rejected=0,pq_enqueued=0;
    std::map<uint64_t,uint64_t> attempted_phases,allowed_phases;
    Cohort cohort;
};
struct Pending { uint64_t cohort, fill_cycle; bool resident; std::string classification; };
struct Slot { uint64_t id=0, fill_cycle=0; bool observed=false, used=false; };
class Observer {
public:
    uint64_t occupancy=0, peak=0, replacements=0, invalidations=0;
    uint64_t line_lifetimes=0, line_lifetime_cycles=0;
    uint64_t useful_line_lifetimes=0, useful_line_lifetime_cycles=0;
    uint64_t prefetch_useful_lifetimes=0, prefetch_fill_to_first_demand_cycles=0;
    uint64_t unresolved_peak=0, next_id=1;
    std::array<uint64_t,4> fills{{0,0,0,0}};
    std::map<uint64_t,Cohort> cohorts;
    std::map<uint64_t,Pending> pending;
    std::map<std::string,Classified> classified;
    std::vector<Slot> slots;
    explicit Observer(uint64_t capacity=0): slots(capacity) {}
    std::string attempt(const RequestClass& label,uint64_t actual_phase) {
        const std::string key=label.group();
        Classified& c=classified[key];c.label=label;++c.attempted;++c.attempted_phases[actual_phase];return key;
    }
    uint64_t enqueue_tracked(uint64_t instructions,int requested_fill_level,int l2_fill_level,
                             const std::string& classification="") {
        return fully_tracked_l2(requested_fill_level,l2_fill_level)?enqueue(instructions,classification):0;
    }
    uint64_t enqueue(uint64_t instructions,const std::string& classification="") {
        uint64_t cohort=(instructions/100000)*100000;
        uint64_t id=next_id++;
        ++cohorts[cohort].enqueued;
        ++cohorts[cohort].pending;
        if(!classification.empty()) { ++classified.at(classification).cohort.enqueued;++classified.at(classification).cohort.pending; }
        pending.emplace(id,Pending{cohort,0,false,classification});
        unresolved_peak=std::max<uint64_t>(unresolved_peak,pending.size());
        return id;
    }
    void resolve(uint64_t id, int outcome, uint64_t cycle, unsigned type=0) {
        auto it=pending.find(id);
        if(!id || it==pending.end()) return;
        auto update=[&](Cohort& c) {
            if(it->second.resident) --c.resident_unused; else --c.pending;
            if(outcome==0) { ++c.timely;if(type==0)++c.timely_load;else if(type==1)++c.timely_rfo; }
            else if(outcome==1) { ++c.late;if(type==0)++c.late_load;else if(type==1)++c.late_rfo; }
            else if(outcome==2) ++c.unused;
            else if(outcome==3) ++c.redundant_cache;
            else if(outcome==4) ++c.redundant_inflight;
            else throw std::runtime_error("unknown measurement outcome");
        };
        update(cohorts[it->second.cohort]);
        if(!it->second.classification.empty())update(classified.at(it->second.classification).cohort);
        if(outcome==0) { ++prefetch_useful_lifetimes;prefetch_fill_to_first_demand_cycles += cycle-it->second.fill_cycle; }
        pending.erase(it);
    }
    void retire_slot(uint64_t slot, uint64_t cycle) {
        Slot& s=slots.at(slot);
        resolve(s.id,2,cycle);
        if(s.observed) {
            ++line_lifetimes; line_lifetime_cycles+=cycle-s.fill_cycle;
            if(s.used) { ++useful_line_lifetimes;
                useful_line_lifetime_cycles+=cycle-s.fill_cycle; }
        }
        s=Slot();
    }
    void fill(uint64_t slot, bool previously_valid, unsigned type,
              uint64_t id, uint64_t cycle) {
        if(previously_valid) { ++replacements; retire_slot(slot,cycle); }
        else { ++occupancy; peak=std::max(peak,occupancy); }
        if(type<4) ++fills[type];
        Slot& s=slots.at(slot);
        s.id=id; s.fill_cycle=cycle; s.observed=true;
        // A demand/RFO fill already serves its initiating demand.
        s.used=(type==0 || type==1);
        auto it=pending.find(id);
        if(id && it!=pending.end()) {
            if(!it->second.resident) {
                Cohort& c=cohorts[it->second.cohort];
                --c.pending; ++c.resident_unused;
                if(!it->second.classification.empty()) {
                    Cohort& g=classified.at(it->second.classification).cohort;
                    --g.pending;++g.resident_unused;
                }
            }
            it->second.resident=true; it->second.fill_cycle=cycle;
        }
    }
    void invalidate(uint64_t slot,uint64_t cycle) {
        if(!occupancy) throw std::runtime_error("occupancy underflow");
        --occupancy; ++invalidations; retire_slot(slot,cycle);
    }
    void demand(uint64_t slot,uint64_t cycle,unsigned type=0) {
        Slot& s=slots.at(slot); s.used=true; resolve(s.id,0,cycle,type); s.id=0;
    }
};
}
#endif
