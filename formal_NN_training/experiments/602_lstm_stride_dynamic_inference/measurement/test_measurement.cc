#include "dynamic_observer_core.h"
#include "instruction.h"
#include "champsim.h"
#include <cassert>
#include <iostream>

using namespace dynamic602_measurement;

template<class Record> void check_actual_trace_format() {
    FILE* trace=std::tmpfile(); assert(trace);
    for(uint64_t i=0;i<90;++i) { Record r={}; r.ip=1000+i;
        assert(std::fwrite(&r,sizeof(r),1,trace)==1); }
    std::rewind(trace);
    skip_exact_records<Record>(trace,25);
    for(uint64_t i=25;i<50;++i) { Record r={};
        assert(std::fread(&r,sizeof(r),1,trace)==1); assert(r.ip==1000+i); }
    // The next unread record is exactly the exclusive endpoint.
    Record r={}; assert(std::fread(&r,sizeof(r),1,trace)==1); assert(r.ip==1050);
    bool short_trace=false;
    try { skip_exact_records<Record>(trace,100); } catch(const std::runtime_error&) {short_trace=true;}
    assert(short_trace); std::fclose(trace);
}

int main() {
    // Simultaneous occupancy, demand/prefetch/writeback fill categories, valid
    // replacement, invalidation, and lifetime accounting.
    Observer o(4);
    o.fill(0,false,0,0,10); assert(o.occupancy==1);
    o.fill(1,false,2,0,20); assert(o.occupancy==2);
    o.fill(1,true,3,0,40); assert(o.occupancy==2 && o.replacements==1);
    o.invalidate(0,50); assert(o.occupancy==1 && o.invalidations==1);
    o.fill(0,false,1,0,60); assert(o.occupancy==2 && o.peak==2);
    assert(o.fills[0]==1 && o.fills[1]==1 && o.fills[2]==1 && o.fills[3]==1);
    assert(o.line_lifetimes==2 && o.line_lifetime_cycles==60);
    assert(o.useful_line_lifetimes==1 && o.useful_line_lifetime_cycles==40);

    // Each enqueued prefetch is resolved once or remains censored. Issue bucket
    // ownership does not change when use/completion crosses a sample boundary.
    Observer c(4);
    auto timely=c.enqueue(100); c.fill(0,false,2,timely,20); c.demand(0,60);
    auto late=c.enqueue(101); c.resolve(late,1,70); c.resolve(late,1,80);
    auto unused=c.enqueue(102); c.fill(1,false,2,unused,30); c.fill(1,true,0,0,90);
    c.enqueue(103); // pending in flight at end
    auto resident=c.enqueue(104); c.fill(2,false,2,resident,40);
    auto duplicate=c.enqueue(105); c.resolve(duplicate,3,100);
    auto merged=c.enqueue(106); c.resolve(merged,4,110);
    c.enqueue(100001); // independent second issue cohort
    const Cohort& first=c.cohorts.at(0);
    assert(first.enqueued==7 && first.timely==1 && first.late==1 && first.unused==1);
    assert(first.pending==1 && first.resident_unused==1);
    assert(first.redundant_cache==1 && first.redundant_inflight==1);
    assert(first.enqueued==first.timely+first.late+first.unused+first.pending+
           first.resident_unused+first.redundant_cache+first.redundant_inflight);
    assert(c.prefetch_useful_lifetimes==1 && c.prefetch_fill_to_first_demand_cycles==40);
    assert(c.cohorts.at(100000).enqueued==1 && c.cohorts.at(100000).pending==1);

    // Native SPP sends both L2 and LLC-only requests through the L2 PQ. Only
    // the former are followed by these L2 fill/use hooks. Do not manufacture
    // an indefinitely pending L2 identity for a request that bypasses L2 fill.
    Observer mixed(4);
    RequestClass l2;l2.arm="spp";l2.phase=0;l2.slot=-1;l2.teacher=-1;
    RequestClass llc=l2;llc.fill_scope="llc_only";
    auto l2_key=mixed.attempt(l2,0),llc_key=mixed.attempt(llc,0);
    auto l2_id=mixed.enqueue_tracked(100,FILL_L2,FILL_L2,l2_key);
    auto llc_id=mixed.enqueue_tracked(101,FILL_LLC,FILL_L2,llc_key);
    assert(l2_id!=0 && llc_id==0 && mixed.pending.size()==1);
    mixed.fill(0,false,2,l2_id,20);mixed.demand(0,30);
    assert(mixed.cohorts.at(0).enqueued==1 && mixed.cohorts.at(0).timely_load==1);
    assert(mixed.classified.at(llc_key).cohort.enqueued==0);
    assert(mixed.classified.at(l2_key).cohort.timely_load==1);

    // Classification is owned by the originating decoded payload even if the
    // gate, fill, and use occur in later phases. Duplicate payload elements
    // remain distinct requests, and late RFO is not mislabeled LOAD usefulness.
    RequestClass origin;origin.arm="nn";origin.phase=0;origin.slot=1;origin.current=0;origin.duplicate=1;origin.teacher=2;
    auto origin_key=mixed.attempt(origin,1);
    auto cross=mixed.enqueue_tracked(6000000,FILL_L2,FILL_L2,origin_key);
    mixed.fill(1,false,2,cross,50);mixed.demand(1,100,0);
    auto late_rfo=mixed.enqueue_tracked(6000001,FILL_L2,FILL_L2,origin_key);mixed.resolve(late_rfo,1,101,1);
    const auto& profile=mixed.classified.at(origin_key);
    assert(profile.label.phase==0 && profile.attempted_phases.at(1)==1);
    assert(profile.cohort.enqueued==2 && profile.cohort.timely_load==1 && profile.cohort.late_rfo==1 && profile.cohort.late_load==0);
    assert(origin.key()=="phase_0/slot_1/current_0/duplicate_1/teacher_2");
    assert(mixed.cohorts.at(6000000).enqueued==2 && mixed.pending.empty());

    // Current-line suppression is the first common gate, so it neither spends
    // a cap credit nor gets misreported as page/budget suppression.
    assert(gate_decision(true,true,true,1,0,0)==GATE_CURRENT);
    assert(gate_decision(true,false,false,1,1000000,0)==GATE_ALLOW);
    assert(gate_decision(true,false,false,1,1000000,1)==GATE_BUDGET);
    assert(gate_decision(false,true,false,0,0,0)==GATE_ALLOW);
    assert(gate_decision(false,false,true,0,0,0)==GATE_PAGE);
    assert(phase_index(4999999,5000000,25000000)==0);
    assert(phase_index(5000000,5000000,25000000)==1);
    assert(phase_index(25000003,5000000,25000000)==4);

    check_actual_trace_format<input_instr>();
    check_actual_trace_format<cloudsuite_instr>();
    std::cout << "PASS occupancy, lifetimes, L2-only complete cohorts, classified cross-phase outcomes, shared current gate, actual trace layouts\n";
}
