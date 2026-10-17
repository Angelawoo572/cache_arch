#ifndef FOUR602_COHORT_H
#define FOUR602_COHORT_H
#include "dynamic_observer_core.h"
#include <cstdint>
#include <stdexcept>

// Measurement-only routing and LLC handoff accounting; never predictor inputs.
namespace four602_measurement {
static const uint64_t LLC_ID_FIRST=uint64_t(1)<<63;
inline bool is_llc_id(uint64_t id) { return (id&LLC_ID_FIRST)!=0; }
enum Target { NO_TARGET, L2_TARGET, LLC_TARGET };
inline Target request_target(bool at_l2,bool at_llc,uint64_t id) {
    if(!id || (!at_l2&&!at_llc))return NO_TARGET;
    if(is_llc_id(id))return LLC_TARGET; // Can resolve source-L2 redundancy, too.
    return at_l2?L2_TARGET:NO_TARGET;
}
struct LLCShortcuts {
    uint64_t write_queue_forwarded=0,prefetch_queue_merged=0;
    uint64_t prefetch_queue_rejected=0,missing_pending_id=0;
    bool fully_tracked() const {return !prefetch_queue_rejected&&!missing_pending_id;}
};
inline void llc_shortcut(dynamic602_measurement::Observer& observer,LLCShortcuts& counters,
                         uint64_t id,unsigned reason,uint64_t cycle) {
    if(!is_llc_id(id))return;
    if(reason<1||reason>3)throw std::runtime_error("invalid LLC handoff outcome");
    if(reason==3) {++counters.prefetch_queue_rejected;return;}
    if(observer.pending.find(id)==observer.pending.end()) {++counters.missing_pending_id;return;}
    if(reason==1)++counters.write_queue_forwarded;
    else ++counters.prefetch_queue_merged;
    // A source L2 request satisfied by an LLC WQ forward or merged into an
    // existing LLC PQ entry creates no new LLC resident-prefetch lifecycle.
    // Keep these distinct counters and classify their source request as
    // redundant-inflight; neither is a useful LOAD or a DRAM bus transfer.
    observer.resolve(id,4,cycle);
}
}
#endif
