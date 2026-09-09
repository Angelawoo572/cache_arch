#ifndef ONLINE602_OBSERVER_CORE_H
#define ONLINE602_OBSERVER_CORE_H
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <vector>

namespace online602_measurement {
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
    uint64_t redundant_cache=0, redundant_inflight=0;
    uint64_t pending=0, resident_unused=0;
};
struct Pending { uint64_t cohort, fill_cycle; bool resident; };
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
    std::vector<Slot> slots;
    explicit Observer(uint64_t capacity=0): slots(capacity) {}
    uint64_t enqueue(uint64_t instructions) {
        uint64_t cohort=(instructions/100000)*100000;
        uint64_t id=next_id++;
        ++cohorts[cohort].enqueued;
        ++cohorts[cohort].pending;
        pending.emplace(id,Pending{cohort,0,false});
        unresolved_peak=std::max<uint64_t>(unresolved_peak,pending.size());
        return id;
    }
    void resolve(uint64_t id, int outcome, uint64_t cycle) {
        auto it=pending.find(id);
        if(!id || it==pending.end()) return;
        Cohort& c=cohorts[it->second.cohort];
        if(it->second.resident) --c.resident_unused; else --c.pending;
        if(outcome==0) { ++c.timely; ++prefetch_useful_lifetimes;
            prefetch_fill_to_first_demand_cycles += cycle-it->second.fill_cycle; }
        else if(outcome==1) ++c.late;
        else if(outcome==2) ++c.unused;
        else if(outcome==3) ++c.redundant_cache;
        else if(outcome==4) ++c.redundant_inflight;
        else throw std::runtime_error("unknown measurement outcome");
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
            }
            it->second.resident=true; it->second.fill_cycle=cycle;
        }
    }
    void invalidate(uint64_t slot,uint64_t cycle) {
        if(!occupancy) throw std::runtime_error("occupancy underflow");
        --occupancy; ++invalidations; retire_slot(slot,cycle);
    }
    void demand(uint64_t slot,uint64_t cycle) {
        Slot& s=slots.at(slot); s.used=true; resolve(s.id,0,cycle); s.id=0;
    }
};
}
#endif
