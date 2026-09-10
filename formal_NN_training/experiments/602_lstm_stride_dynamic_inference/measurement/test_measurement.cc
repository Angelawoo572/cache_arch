#include "dynamic_observer_core.h"
#include "instruction.h"
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

    check_actual_trace_format<input_instr>();
    check_actual_trace_format<cloudsuite_instr>();
    std::cout << "PASS occupancy, lifetimes, timely/late/unused/censored cohorts, actual trace layouts\n";
}
