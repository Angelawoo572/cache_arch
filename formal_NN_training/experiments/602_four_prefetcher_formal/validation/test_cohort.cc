#include "four602_cohort.h"
#include <cassert>
#include <iostream>
using namespace four602_measurement;
using dynamic602_measurement::Observer;
static void conserved(const Observer& o) {
    uint64_t unresolved=0;
    for(const auto& item:o.cohorts) {
        const auto& c=item.second;
        assert(c.enqueued==c.timely+c.late+c.unused+c.redundant_cache+c.redundant_inflight+c.pending+c.resident_unused);
        unresolved+=c.pending+c.resident_unused;
    }
    assert(unresolved==o.pending.size());
}
int main() {
    assert(request_target(true,false,1)==L2_TARGET);
    assert(request_target(false,true,1)==NO_TARGET);
    assert(request_target(true,false,LLC_ID_FIRST)==LLC_TARGET);
    assert(request_target(false,true,LLC_ID_FIRST)==LLC_TARGET);
    assert(request_target(false,false,LLC_ID_FIRST)==NO_TARGET);
    assert(request_target(true,false,0)==NO_TARGET);
    Observer l2(1),llc(1);llc.next_id=LLC_ID_FIRST;
    const auto l2id=l2.enqueue(0),timely=llc.enqueue(0);
    assert(l2id==1&&timely==LLC_ID_FIRST);
    llc.fill(0,false,2,timely,10);conserved(llc);
    assert(llc.pending.at(timely).resident);
    llc.demand(0,20,0);conserved(llc);
    assert(llc.cohorts.at(0).timely_load==1&&l2.pending.size()==1);
    auto late=llc.enqueue(0);llc.resolve(late,1,25,0);conserved(llc);
    assert(llc.cohorts.at(0).late_load==1);
    auto unused=llc.enqueue(0);llc.fill(0,true,2,unused,30);llc.invalidate(0,35);conserved(llc);
    assert(llc.cohorts.at(0).unused==1);
    auto redundant=llc.enqueue(0);llc.resolve(redundant,3,36);conserved(llc);
    LLCShortcuts counters;
    auto forwarded=llc.enqueue(0);llc_shortcut(llc,counters,forwarded,1,40);conserved(llc);
    auto merged=llc.enqueue(0);llc_shortcut(llc,counters,merged,2,41);conserved(llc);
    assert(counters.write_queue_forwarded==1&&counters.prefetch_queue_merged==1&&counters.fully_tracked());
    assert(llc.cohorts.at(0).redundant_inflight==2);
    assert(llc.cohorts.at(0).timely_load==1&&llc.cohorts.at(0).late_load==1);
    llc_shortcut(llc,counters,l2id,1,42);
    assert(counters.write_queue_forwarded==1&&l2.pending.count(l2id));
    auto rejected=llc.enqueue(0);llc_shortcut(llc,counters,rejected,3,43);conserved(llc);
    assert(counters.prefetch_queue_rejected==1&&!counters.fully_tracked()&&llc.pending.count(rejected));
    llc_shortcut(llc,counters,LLC_ID_FIRST+999,1,44);conserved(llc);
    assert(counters.missing_pending_id==1&&!counters.fully_tracked());
    llc.resolve(rejected,2,45);conserved(llc);
    assert(llc.pending.empty()&&l2.pending.size()==1);
    std::cout<<"{\"status\":\"PASS\",\"routing_cases\":6,\"lifecycle_cases\":8,\"queue_shortcut_cases\":5}"<<std::endl;
}
