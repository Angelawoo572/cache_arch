#include "dynamic602_hooks.h"
#include "dynamic_observer_core.h"
#include "cache.h"
#include "ooo_cpu.h"
#include "uncore.h"
extern UNCORE uncore;
#include <cerrno>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {
using dynamic602_measurement::Observer;
using dynamic602_measurement::RequestClass;
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
    uint64_t dram[7]={},dram_types[7][4]={};
    uint64_t dram_completed[2]={};
    uint64_t initial_dram_reads=0,initial_dram_writes=0,initial_dram_busy_channels=0;
    std::multimap<uint64_t,std::pair<bool,unsigned>> dram_bus_pending;
    uint64_t request_generated=0,request_allowed=0,request_budget_suppressed=0,request_page_suppressed=0;
    uint64_t request_cache_enqueued=0,request_cache_merged=0,request_cache_rejected=0;
    uint64_t request_current_suppressed=0,request_pq_inserted=0;
    // Scope index 0 is fully followed L2-target; 1 is LLC-only, not followed at LLC.
    uint64_t scope_generated[2]={},scope_allowed[2]={},scope_pq_inserted[2]={};
    uint64_t scope_merged[2]={},scope_rejected[2]={};
    std::map<std::tuple<uint64_t,uint64_t,uint64_t>,std::deque<RequestClass>> labels;
    std::string current_classification;
    unsigned current_scope=0;
    uint64_t next_sample=1000, last_sample=uint64_t(-1), origin=0;
    bool thresholds[5]={false,false,false,false,false};
};
State states[NUM_CPUS];
uint64_t instructions(unsigned cpu) { return ooo_cpu[cpu].num_retired-ooo_cpu[cpu].begin_sim_instr; }
uint64_t cycles(unsigned cpu) { return current_core_cycle[cpu]-ooo_cpu[cpu].begin_sim_cycle; }
bool active(CACHE* c) { return c->cache_type==IS_L2C && states[c->cpu].began; }
uint64_t phase(unsigned cpu) {
    static const uint64_t width=env_number("DYNAMIC602_PHASE_INSTRUCTIONS")?env_number("DYNAMIC602_PHASE_INSTRUCTIONS"):5000000;
    static const uint64_t total=env_number("DYNAMIC602_ROI_INSTRUCTIONS");
    return dynamic602_measurement::phase_index(instructions(cpu),width,total);
}
uint64_t clamp_phase(uint64_t value) {
    static const uint64_t width=env_number("DYNAMIC602_PHASE_INSTRUCTIONS")?env_number("DYNAMIC602_PHASE_INSTRUCTIONS"):5000000;
    static const uint64_t total=env_number("DYNAMIC602_ROI_INSTRUCTIONS");
    return total?std::min<uint64_t>(value,(total-1)/width):value;
}
std::string arm() {
    const char* value=std::getenv("DYNAMIC602_ARM");return value&&*value?value:"unknown";
}
std::string quoted(const std::string& value) {
    std::ostringstream out;out<<'"';
    for(unsigned char c:value) { if(c=='"'||c=='\\')out<<'\\'; if(c<32)throw std::runtime_error("control character in measurement label");out<<c; }
    out<<'"';return out.str();
}
void cohort_json(std::ostream& out,const dynamic602_measurement::Cohort& c) {
    out<<"\"enqueued\":"<<c.enqueued<<",\"timely\":"<<c.timely<<",\"late\":"<<c.late
       <<",\"timely_load\":"<<c.timely_load<<",\"late_load\":"<<c.late_load
       <<",\"timely_rfo\":"<<c.timely_rfo<<",\"late_rfo\":"<<c.late_rfo
       <<",\"useful\":"<<(c.timely+c.late)<<",\"useful_load\":"<<(c.timely_load+c.late_load)<<",\"unused\":"<<c.unused
       <<",\"redundant_cache\":"<<c.redundant_cache<<",\"redundant_inflight\":"<<c.redundant_inflight
       <<",\"pending\":"<<c.pending<<",\"resident_unused\":"<<c.resident_unused;
}
void phase_counts_json(std::ostream& out,const std::map<uint64_t,uint64_t>& counts) {
    bool first=true;out<<'{';for(const auto& kv:counts){if(!first)out<<',';first=false;out<<'"'<<kv.first<<"\":"<<kv.second;}out<<'}';
}
void classified_json(std::ostream& out,const Observer& o) {
    out<<",\"mechanism_classified_requests\":[";bool first=true;
    for(const auto& item:o.classified) {
        if(!first)out<<',';first=false;const auto& c=item.second;const auto& l=c.label;
        const auto& v=c.cohort;
        if(c.attempted!=c.allowed+c.current_suppressed+c.page_suppressed+c.budget_suppressed ||
           c.allowed!=c.cache_enqueued+c.cache_merged+c.cache_rejected ||
           v.enqueued!=v.timely+v.late+v.unused+v.redundant_cache+v.redundant_inflight+v.pending+v.resident_unused ||
           (l.fill_scope=="l2_target"&&c.pq_enqueued!=v.enqueued))
            throw std::runtime_error("classified request/lifecycle conservation mismatch");
        out<<"{\"arm\":"<<quoted(l.arm)<<",\"fill_scope\":"<<quoted(l.fill_scope)<<",\"key\":"<<quoted(l.key())
           <<",\"origin_phase\":"<<l.phase<<",\"slot\":"<<l.slot<<",\"current_line\":"<<l.current
           <<",\"list_duplicate\":"<<l.duplicate<<",\"teacher_category\":"<<l.teacher
           <<",\"cache_attempted\":"<<c.attempted<<",\"allowed\":"<<c.allowed
           <<",\"current_suppressed\":"<<c.current_suppressed<<",\"page_suppressed\":"<<c.page_suppressed
           <<",\"budget_suppressed\":"<<c.budget_suppressed<<",\"cache_accepted_nonmerged\":"<<c.cache_enqueued
           <<",\"cache_merged\":"<<c.cache_merged<<",\"cache_rejected\":"<<c.cache_rejected<<",\"pq_enqueued\":"<<c.pq_enqueued
           <<",\"actual_attempt_phase_counts\":";phase_counts_json(out,c.attempted_phases);
        out<<",\"actual_allowed_phase_counts\":";phase_counts_json(out,c.allowed_phases);
        out<<",\"l2_lifecycle_fully_tracked\":"<<(l.fill_scope=="l2_target"?"true":"false")<<",\"l2_cohort\":";
        if(l.fill_scope=="l2_target") {out<<'{';cohort_json(out,c.cohort);out<<'}';}else out<<"null";
        out<<",\"l2_load_accuracy_enqueued\":";
        if(l.fill_scope=="l2_target"&&v.enqueued)out<<double(v.timely_load+v.late_load)/v.enqueued;else out<<"null";
        out<<",\"l2_load_timeliness\":";
        if(l.fill_scope=="l2_target"&&(v.timely_load+v.late_load))out<<double(v.timely_load)/(v.timely_load+v.late_load);else out<<"null";
        out<<'}';
    }
    out<<']';
}
uint64_t dram_queue_occupancy(bool write) {
    uint64_t total=0;for(unsigned channel=0;channel<DRAM_CHANNELS;++channel)total+=write?uncore.DRAM.WQ[channel].occupancy:uncore.DRAM.RQ[channel].occupancy;return total;
}
void finish_dram_bus(unsigned cpu) {
    State& s=states[cpu];
    while(!s.dram_bus_pending.empty() && s.dram_bus_pending.begin()->first<=current_core_cycle[cpu]) {
        ++s.dram_completed[s.dram_bus_pending.begin()->second.first ? 1 : 0];
        s.dram_bus_pending.erase(s.dram_bus_pending.begin());
    }
}
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
        s.out<<std::setprecision(17);
    }
    s.initial_dram_reads=dram_queue_occupancy(false);s.initial_dram_writes=dram_queue_occupancy(true);
    for(unsigned channel=0;channel<DRAM_CHANNELS;++channel)s.initial_dram_busy_channels+=(uncore.DRAM.dbus_cycle_available[channel]>current_core_cycle[cpu]);
    s.began=true;
    thresholds(cpu);
    dynamic602_observe_cycle(cpu);
}

void dynamic602_observe_cycle(unsigned cpu,bool final) {
    State& s=states[cpu];
    if(!s.began) return;
    finish_dram_bus(cpu);
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
            outcomes.timely_load+=c.timely_load;outcomes.late_load+=c.late_load;
            outcomes.timely_rfo+=c.timely_rfo;outcomes.late_rfo+=c.late_rfo;
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
            << ",\"cohort_timely_load\":" << outcomes.timely_load << ",\"cohort_late_load\":" << outcomes.late_load
            << ",\"cohort_timely_rfo\":" << outcomes.timely_rfo << ",\"cohort_late_rfo\":" << outcomes.late_rfo
            << ",\"cohort_unused\":" << outcomes.unused
            << ",\"cohort_redundant_cache\":" << outcomes.redundant_cache
            << ",\"cohort_redundant_inflight\":" << outcomes.redundant_inflight
            << ",\"cohort_pending\":" << outcomes.pending
            << ",\"cohort_resident_unused\":" << outcomes.resident_unused
            << ",\"measurement_unresolved_lifecycles\":" << o.pending.size()
            << ",\"measurement_unresolved_lifecycles_peak\":" << o.unresolved_peak;
        static const char* dram_names[]={"read_enqueued","write_enqueued","read_merged","write_merged","write_forwarded_read","read_bus_transfers","write_bus_transfers"};
        for(unsigned k=0;k<7;++k) {
            s.out << ",\"dram_" << dram_names[k] << "\":" << s.dram[k];
            for(unsigned t=0;t<4;++t) s.out << ",\"dram_" << dram_names[k] << "_type" << t << "\":" << s.dram_types[k][t];
        }
        s.out << ",\"dram_read_bus_bytes\":" << (s.dram_completed[0]<<LOG2_BLOCK_SIZE)
              << ",\"dram_write_bus_bytes\":" << (s.dram_completed[1]<<LOG2_BLOCK_SIZE)
              << ",\"dram_total_bus_bytes\":" << ((s.dram_completed[0]+s.dram_completed[1])<<LOG2_BLOCK_SIZE)
              << ",\"dram_read_bus_completed_transactions\":" << s.dram_completed[0]
              << ",\"dram_write_bus_completed_transactions\":" << s.dram_completed[1]
              << ",\"dram_read_bus_started_bytes\":" << (s.dram[5]<<LOG2_BLOCK_SIZE)
              << ",\"dram_write_bus_started_bytes\":" << (s.dram[6]<<LOG2_BLOCK_SIZE)
              << ",\"dram_bus_transfers_in_progress\":" << s.dram_bus_pending.size()
              << ",\"dram_read_queue_pending\":" << dram_queue_occupancy(false)
              << ",\"dram_write_queue_pending\":" << dram_queue_occupancy(true)
              << ",\"dram_initial_read_queue_pending\":" << s.initial_dram_reads
              << ",\"dram_initial_write_queue_pending\":" << s.initial_dram_writes
              << ",\"dram_initial_busy_channels\":" << s.initial_dram_busy_channels
              << ",\"request_gate_generated\":" << s.request_generated
              << ",\"request_gate_allowed\":" << s.request_allowed
              << ",\"request_gate_budget_suppressed\":" << s.request_budget_suppressed
              << ",\"request_gate_page_suppressed\":" << s.request_page_suppressed
              << ",\"request_gate_current_suppressed\":" << s.request_current_suppressed
              << ",\"request_cache_enqueued\":" << s.request_cache_enqueued
              << ",\"request_cache_merged\":" << s.request_cache_merged
              << ",\"request_cache_rejected\":" << s.request_cache_rejected
              << ",\"request_cache_pq_inserted\":" << s.request_pq_inserted
              << ",\"request_l2_target_generated\":" << s.scope_generated[0]
              << ",\"request_llc_only_generated\":" << s.scope_generated[1]
              << ",\"request_l2_target_allowed\":" << s.scope_allowed[0]
              << ",\"request_llc_only_allowed\":" << s.scope_allowed[1]
              << ",\"request_l2_target_cache_merged\":" << s.scope_merged[0]
              << ",\"request_llc_only_cache_merged\":" << s.scope_merged[1]
              << ",\"request_l2_target_cache_rejected\":" << s.scope_rejected[0]
              << ",\"request_llc_only_cache_rejected\":" << s.scope_rejected[1]
              << ",\"cohort_llc_only_excluded\":" << s.scope_pq_inserted[1]
              << ",\"measurement_cohort_scope\":\"l2_target_fully_tracked_only\""
              << ",\"cohort_all_enqueued_requests_fully_tracked\":" << (s.scope_pq_inserted[1]?"false":"true")
              << ",\"llc_only_lifecycle_accuracy_available\":false"
              << ",\"l2_cohort_coverage_scope\":\"partial_L2_target_subset_when_LLC_only_requests_exist\""
              << ",\"mechanism_phase_instructions\":" << (env_number("DYNAMIC602_PHASE_INSTRUCTIONS")?env_number("DYNAMIC602_PHASE_INSTRUCTIONS"):5000000)
              << ",\"mechanism_profile_scope\":\"cache_call_attempts_after_engine_output_filters; generated_profile_is_engine_separate; origin_phase_is_payload_generation; allowed_phases_are_actual_gate_time; lifecycle_follows_origin\""
              << ",\"measurement_unconsumed_context_labels\":";
        uint64_t labels=0;for(const auto& v:s.labels)labels+=v.second.size();s.out<<labels;
        classified_json(s.out,o);
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
void dynamic602_demand_hit(CACHE* c,unsigned set,unsigned way,unsigned type) {
    if(active(c)) states[c->cpu].obs->demand(uint64_t(set)*c->NUM_WAY+way,cycles(c->cpu),type);
}
void dynamic602_late(CACHE* c,const PACKET* packet,unsigned type) {
    if(active(c)) states[c->cpu].obs->resolve(packet->dynamic602_measurement_id,1,cycles(c->cpu),type);
}
void dynamic602_enqueue(CACHE* c,PACKET* packet) {
    if(!active(c))return;
    State& s=states[c->cpu];
    const bool tracked=dynamic602_measurement::fully_tracked_l2(packet->fill_level,FILL_L2);
    ++s.request_pq_inserted;++s.scope_pq_inserted[tracked?0:1];
    if(!s.current_classification.empty())++s.obs->classified.at(s.current_classification).pq_enqueued;
    // Native SPP retains its confidence-based FILL_LLC request. Such a packet
    // bypasses the L2 MSHR/fill path, so assigning an L2 lifecycle ID would turn
    // unobserved LLC completion/use into a falsely censored L2 request.
    packet->dynamic602_measurement_id=s.obs->enqueue_tracked(instructions(c->cpu),packet->fill_level,FILL_L2,s.current_classification);
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

// Retired program progress for bounded measurement-only recurrent-state examples.
uint64_t dynamic602_retired_instructions(unsigned cpu) { return instructions(cpu); }

// Arrival to the actual DRAM controller queue and actual scheduled data-bus
// transfer are distinct. Reads forwarded by a pending write cause no read bus
// transfer. Counts are sampled by ROI cycle and never derived from prefetches.
void dynamic602_dram_event(unsigned cpu,unsigned event,unsigned type) {
    if(cpu>=NUM_CPUS || event>=7 || !states[cpu].began)return;
    ++states[cpu].dram[event];if(type<4)++states[cpu].dram_types[event][type];
}
uint64_t dynamic602_total_retired_instructions(unsigned cpu) {return ooo_cpu[cpu].num_retired;}
void dynamic602_label_request(CACHE* cache,uint64_t ip,uint64_t base,uint64_t address,
    uint64_t origin_phase,unsigned slot,bool current,bool list_duplicate,unsigned teacher_category) {
    if(!active(cache))return;
    RequestClass label;label.arm=arm();label.phase=clamp_phase(origin_phase);label.slot=slot;
    label.current=current;label.duplicate=list_duplicate;label.teacher=teacher_category;
    states[cache->cpu].labels[std::make_tuple(ip,base,address)].push_back(std::move(label));
}
bool dynamic602_allow_request(CACHE* cache,uint64_t ip,uint64_t base,uint64_t address,int fill_level) {
    if(!warmup_complete[cache->cpu])return false;
    if(!active(cache))return true;
    State& s=states[cache->cpu];++s.request_generated;
    s.current_classification.clear();
    RequestClass label;label.arm=arm();label.phase=phase(cache->cpu);label.current=(base>>LOG2_BLOCK_SIZE)==(address>>LOG2_BLOCK_SIZE);
    auto it=s.labels.find(std::make_tuple(ip,base,address));
    if(it!=s.labels.end()) {
        label=it->second.front();it->second.pop_front();if(it->second.empty())s.labels.erase(it);
        if(label.current!=int((base>>LOG2_BLOCK_SIZE)==(address>>LOG2_BLOCK_SIZE)))throw std::runtime_error("request current-line context mismatch");
    }
    s.current_scope=dynamic602_measurement::fully_tracked_l2(fill_level,FILL_L2)?0:1;
    label.fill_scope=s.current_scope?"llc_only":"l2_target";
    ++s.scope_generated[s.current_scope];
    const std::string group=s.obs->attempt(label,phase(cache->cpu));
    auto& classified=s.obs->classified.at(group);
    // This shared output-only intervention precedes page and budget accounting.
    // Suppressing a current-line call cannot spend request credits.
    const char* page=std::getenv("DYNAMIC602_PAGE_POLICY");
    static const uint64_t cap=env_number("DYNAMIC602_REQUESTS_PER_MILLION_INSTRUCTIONS");
    const bool wrong_page=page && std::string(page)=="same_page" && (base>>LOG2_PAGE_SIZE)!=(address>>LOG2_PAGE_SIZE);
    const auto decision=dynamic602_measurement::gate_decision(env_number("DYNAMIC602_FILTER_CURRENT"),label.current,wrong_page,cap,instructions(cache->cpu),s.request_allowed);
    if(decision==dynamic602_measurement::GATE_CURRENT){++s.request_current_suppressed;++classified.current_suppressed;return false;}
    if(decision==dynamic602_measurement::GATE_PAGE){++s.request_page_suppressed;++classified.page_suppressed;return false;}
    if(decision==dynamic602_measurement::GATE_BUDGET){++s.request_budget_suppressed;++classified.budget_suppressed;return false;}
    ++s.request_allowed;++s.scope_allowed[s.current_scope];++classified.allowed;++classified.allowed_phases[phase(cache->cpu)];s.current_classification=group;return true;
}
void dynamic602_cache_request_result(CACHE* cache,int result) {
    if(!active(cache))return;
    State& s=states[cache->cpu];
    if(result==-2)++s.request_cache_rejected;
    else if(result>=0)++s.request_cache_merged;
    else ++s.request_cache_enqueued;
    if(result==-2)++s.scope_rejected[s.current_scope];else if(result>=0)++s.scope_merged[s.current_scope];
    if(!s.current_classification.empty()) {
        auto& c=s.obs->classified.at(s.current_classification);
        if(result==-2)++c.cache_rejected;else if(result>=0)++c.cache_merged;else ++c.cache_enqueued;
        s.current_classification.clear();
    }
}

// A bus reservation consumes actual modeled DRAM bandwidth. Preserve its real
// completion cycle so ROI-end bytes exclude transfers still in progress.
void dynamic602_dram_bus_start(unsigned cpu,bool write,unsigned type,uint64_t finish_cycle) {
    if(cpu>=NUM_CPUS || !states[cpu].began)return;
    dynamic602_dram_event(cpu,write?6:5,type);
    finish_dram_bus(cpu);
    states[cpu].dram_bus_pending.emplace(finish_cycle,std::make_pair(write,type));
}
