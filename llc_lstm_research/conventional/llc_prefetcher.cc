#include "cache.h"
#include "correlation_llc.h"
#include "learned_llc.h"
#include "ooo_cpu.h"
#include "policies.h"

#include <array>
#include <cerrno>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace {
using llc_conventional::Policy;
using llc_conventional::PolicyKind;

struct Counters {
    uint64_t ingress = 0;
    uint64_t hit = 0;
    uint64_t miss = 0;
    uint64_t forwarded = 0;
    uint64_t merged = 0;
    uint64_t full = 0;
};

struct Prefetch {
    uint32_t id;
    uint64_t fill_cycle;
};

std::array<Counters, NUM_CPUS> cores;
std::array<std::unique_ptr<Policy>, NUM_CPUS> predictors;
std::unordered_map<uint64_t, Prefetch> outstanding;
std::ofstream events;

struct Trigger {
    uint64_t event_id = 0;
    uint64_t instruction = 0;
    uint32_t operand = 0;
    uint64_t cycle = 0;
};

struct PendingPrediction {
    llc_learned::Observation observation;
    Trigger trigger;
};

struct PrefetchTiming {
    Trigger trigger;
    Trigger demand;
    uint32_t cpu;
    uint32_t demand_cpu = 0;
    uint64_t pc;
    uint64_t line;
    llc_learned::Candidate candidate;
    int32_t score;
    uint64_t retired;
    uint64_t epoch;
    uint64_t start;
    uint64_t ready;
    uint64_t accepted;
    uint64_t dispatch = 0;
    uint64_t fill = 0;
    const char* outcome = "unresolved";
    uint64_t outcome_cycle = 0;
};

struct DemandService {
    Trigger trigger;
    uint32_t cpu;
    uint64_t pc;
    uint64_t line;
    uint32_t mshr_occupancy;
    uint64_t dispatch;
    uint64_t fill = 0;
};

std::ofstream prefetch_events;
std::ofstream demand_service;
std::ofstream pace_epochs;
std::map<uint32_t, PrefetchTiming> prefetch_timing;
std::map<uint64_t, DemandService> demand_timing;
std::array<uint64_t, NUM_CPUS> sampled_demands{};
uint64_t prefetch_sample = 0;
uint64_t demand_service_sample = 0;

struct IssueBudget {
    uint64_t issued = 0;
    // A new epoch starts with an empty count and no retained credits.
    std::vector<std::pair<uint64_t, uint64_t>> epochs;
};

std::array<IssueBudget, NUM_CPUS> budgets;
uint64_t pace_instructions = 0;
uint64_t pace_requests = 0;

struct LearnedCore {
    llc_learned::History history;
    llc_learned::LlcCorrelationHistory correlation_history;
    llc_learned::State state;
    std::deque<PendingPrediction> waiting;
    PendingPrediction running{};
    std::array<int32_t, llc_learned::max_candidates> scores{};
    uint64_t start = 0;
    uint64_t ready = 0;
    uint64_t next_start = 0;
};

std::array<LearnedCore, NUM_CPUS> learned;
std::unique_ptr<llc_learned::Model> neural_model;
std::unique_ptr<llc_learned::PagePc> page_pc_model;
std::unique_ptr<llc_learned::LlcCorrelationTable> correlation_model;
uint64_t prediction_latency = 0;
uint64_t prediction_interval = 0;
uint64_t issue_limit = 0;
int32_t prediction_threshold = -32768;
uint64_t prediction_started = 0;
uint64_t prediction_completed = 0;
uint64_t prediction_overflow = 0;
uint64_t prediction_abstained = 0;
uint64_t prediction_filtered = 0;
uint64_t prediction_pace_blocked = 0;
uint64_t cross_page_issued = 0;

PolicyKind policy = PolicyKind::none;
std::string policy_name = "nopf";
uint32_t degree = 1;
uint64_t next_event = 0;
uint64_t next_lookup = 0;
uint64_t candidates = 0;
uint64_t issued = 0;
uint64_t fills = 0;
uint64_t timely = 0;
uint64_t slack_sum = 0;
uint64_t slack_min = UINT64_MAX;
uint64_t slack_max = 0;
uint64_t late = 0;
uint64_t unused = 0;
uint64_t redundant = 0;
uint64_t dram_demand_read = 0;
uint64_t dram_prefetch_read = 0;
uint64_t dram_write = 0;

void fail(const std::string &message)
{
    std::cerr << "LLC prefetch experiment: " << message << '\n';
    std::exit(EXIT_FAILURE);
}

uint64_t numeric_env(const char *name, uint64_t fallback)
{
    const char *value = std::getenv(name);
    if (!value)
        return fallback;
    char *end = nullptr;
    errno = 0;
    const auto number = std::strtoull(value, &end, 10);
    if (value[0] < '0' || value[0] > '9' || *end || errno)
        fail(std::string("invalid integer in ") + name);
    return number;
}

int32_t threshold_env()
{
    const char* value = std::getenv("LLC_PF_THRESHOLD");
    if (!value)
        return -32768;
    char* end = nullptr;
    errno = 0;
    const auto threshold = std::strtol(value, &end, 10);
    if (value == end || *end || errno || threshold < -32768 || threshold > 32768)
        fail("LLC_PF_THRESHOLD must be an integer from -32768 to 32768");
    return static_cast<int32_t>(threshold);
}

bool learned_active()
{
    return neural_model || page_pc_model;
}

bool roi_active()
{
    return all_warmup_complete > NUM_CPUS && all_simulation_complete < NUM_CPUS;
}

bool core_in_roi(uint32_t core)
{
    return roi_active() && !simulation_complete[core];
}

bool demand_load(const PACKET &packet)
{
    return packet.type == LOAD && !packet.instruction;
}

Trigger trigger_of(const PACKET& packet)
{
    Trigger trigger;
    trigger.event_id = packet.llc_pf_event_id;
    trigger.instruction = packet.instr_id;
    trigger.operand = packet.data_index;
    trigger.cycle = packet.llc_pf_ingress_cycle;
    return trigger;
}

uint64_t retired_in_roi(uint32_t core)
{
    return ooo_cpu[core].num_retired - ooo_cpu[core].begin_sim_instr;
}

uint64_t issue_epoch(uint32_t core)
{
    return retired_in_roi(core) / pace_instructions;
}

bool pace_allows(uint32_t core)
{
    const auto& epochs = budgets[core].epochs;
    return !pace_instructions || epochs.empty() || epochs.back().first != issue_epoch(core)
        || epochs.back().second < pace_requests;
}

bool below_issue_limit(uint32_t core)
{
    return !issue_limit || budgets[core].issued < issue_limit;
}

void record_issue(uint32_t core)
{
    auto& budget = budgets[core];
    ++budget.issued;
    if (!pace_instructions)
        return;
    const auto epoch = issue_epoch(core);
    if (budget.epochs.empty() || budget.epochs.back().first != epoch)
        budget.epochs.emplace_back(epoch, 0);
    ++budget.epochs.back().second;
}

void record_outcome(uint32_t id, const char* outcome, uint64_t cycle, const PACKET* demand = nullptr)
{
    if (!prefetch_sample)
        return;
    const auto sample = prefetch_timing.find(id);
    if (sample == prefetch_timing.end())
        return;
    auto& timing = sample->second;
    timing.outcome = outcome;
    timing.outcome_cycle = cycle;
    if (demand) {
        timing.demand = trigger_of(*demand);
        timing.demand_cpu = demand->cpu;
    }
}

void open_log(std::ofstream& output, const char* path)
{
    output.open(path);
    if (!output)
        fail(std::string("cannot open ") + path);
}

void close_log(std::ofstream& output)
{
    if (!output.is_open())
        return;
    output.close();
    if (!output)
        fail("writing experiment CSV failed");
}

void optional_number(std::ofstream& output, uint64_t value)
{
    if (value)
        output << value;
}

void write_timing_logs()
{
    if (prefetch_sample) {
        for (const auto& entry : prefetch_timing) {
            const auto& timing = entry.second;
            const auto& trigger = timing.trigger;
            const auto& candidate = timing.candidate;
            prefetch_events << entry.first << ',' << timing.cpu << ',' << trigger.event_id << ','
                            << trigger.instruction << ',' << trigger.operand << ',' << timing.pc << ','
                            << timing.line << ',' << candidate.line << ',' << candidate.source_pc << ','
                            << candidate.source_offset << ',' << candidate.age << ',' << candidate.delta << ','
                            << timing.score << ',' << timing.retired << ',';
            if (pace_instructions)
                prefetch_events << timing.epoch;
            prefetch_events << ',' << trigger.cycle << ',' << timing.start << ',' << timing.ready << ','
                            << timing.accepted << ',';
            optional_number(prefetch_events, timing.dispatch);
            prefetch_events << ',';
            optional_number(prefetch_events, timing.fill);
            prefetch_events << ',' << timing.outcome << ',';
            optional_number(prefetch_events, timing.outcome_cycle);
            prefetch_events << ',';
            if (timing.demand.event_id)
                prefetch_events << timing.demand_cpu << ',' << timing.demand.event_id << ','
                                << timing.demand.instruction << ',' << timing.demand.operand << ',' << timing.demand.cycle;
            else
                prefetch_events << ",,,,";
            prefetch_events << '\n';
        }
    }
    if (demand_service_sample) {
        for (const auto& entry : demand_timing) {
            const auto& timing = entry.second;
            const auto& trigger = timing.trigger;
            demand_service << entry.first << ',' << timing.cpu << ',' << trigger.instruction << ',' << trigger.operand
                           << ',' << timing.pc << ',' << timing.line << ',' << timing.mshr_occupancy << ','
                           << trigger.cycle << ',' << timing.dispatch << ',';
            optional_number(demand_service, timing.fill);
            demand_service << '\n';
        }
    }
    if (pace_epochs.is_open()) {
        for (uint32_t core = 0; core < NUM_CPUS; ++core)
            for (const auto& epoch : budgets[core].epochs)
                pace_epochs << core << ',' << epoch.first << ',' << epoch.second << '\n';
    }
    close_log(prefetch_events);
    close_log(demand_service);
    close_log(pace_epochs);
}

void emit_core(uint32_t core, const char *metric, uint64_t count)
{
    std::cout << "LLC_PF_core_" << core << '_' << metric << ' ' << count << '\n';
}

void emit(const char *metric, uint64_t count)
{
    std::cout << "LLC_PF_" << metric << ' ' << count << '\n';
}

void discard_spp_feedback(uint64_t address)
{
    if (policy == PolicyKind::spp) {
        for (auto &predictor : predictors)
            predictor->demand_outcome(address, false);
    }
}

bool finish_prefetch(const PACKET &packet, uint64_t &counter, const PACKET* demand = nullptr)
{
    auto tracked = outstanding.find(packet.address);
    if (tracked == outstanding.end() || tracked->second.id != packet.pf_metadata)
        return false;
    ++counter;
    record_outcome(tracked->second.id, demand ? "late" : "redundant",
                   current_core_cycle[demand ? demand->cpu : packet.cpu], demand);
    discard_spp_feedback(packet.address);
    outstanding.erase(tracked);
    return true;
}
} // namespace

uint64_t llc_pf_trace_skip()
{
    static const uint64_t skip = numeric_env("LLC_PF_SKIP", 0);
    return skip;
}

void llc_pf_dram_transfer(uint8_t type, bool write)
{
    if (!roi_active())
        return;
    if (write)
        dram_write += BLOCK_SIZE;
    else if (type == PREFETCH)
        dram_prefetch_read += BLOCK_SIZE;
    else
        dram_demand_read += BLOCK_SIZE;
}

void CACHE::llc_prefetcher_initialize()
{
    if (const char *name = std::getenv("LLC_PF_POLICY"))
        policy_name = name;

    const std::map<std::string, PolicyKind> policies = {
        {"nopf", PolicyKind::none},
        {"next_line", PolicyKind::next_line},
        {"stride", PolicyKind::stride},
        {"streamer", PolicyKind::streamer},
        {"ampm", PolicyKind::ampm},
        {"spp", PolicyKind::spp},
    };
    auto selected = policies.find(policy_name);
    if (selected != policies.end()) {
        policy = selected->second;
    } else {
        const char* model = std::getenv("LLC_PF_MODEL");
        try {
            if (policy_name == "page_pc")
                page_pc_model.reset(new llc_learned::PagePc(model));
            else if (policy_name == "lstm" || policy_name == "event")
                neural_model.reset(new llc_learned::Model(model, policy_name));
            else
                fail("unknown policy " + policy_name);
        } catch (const std::runtime_error& error) {
            fail(error.what());
        }
        prediction_latency = numeric_env("LLC_PF_PREDICTION_LATENCY", 16);
        prediction_interval = numeric_env("LLC_PF_PREDICTION_INTERVAL", 16);
        if (!prediction_latency || !prediction_interval || prediction_latency > 1000000 || prediction_interval > 1000000)
            fail("prediction latency and interval must be between 1 and 1000000 cycles");
        prediction_threshold = threshold_env();
    }
    if (const char* table = std::getenv("LLC_PF_CORRELATION")) {
        if (!learned_active())
            fail("a correlation table requires a learned scorer");
        try {
            correlation_model.reset(new llc_learned::LlcCorrelationTable(table));
        } catch (const std::runtime_error& error) {
            fail(error.what());
        }
    }
    if (neural_model && neural_model->nonlinear_width() && !correlation_model)
        fail("a nonlinear scorer requires a correlation candidate table");
    pace_instructions = numeric_env("LLC_PF_PACE_INSTRUCTIONS", 0);
    pace_requests = numeric_env("LLC_PF_PACE_REQUESTS", 0);
    if (bool(pace_instructions) != bool(pace_requests))
        fail("pacing requires both a retired instruction window and a request allowance");
    if (learned_active() || pace_instructions)
        issue_limit = numeric_env("LLC_PF_ISSUE_LIMIT", 0);
    const auto requested_degree = numeric_env("LLC_PF_DEGREE", 1);
    if (requested_degree < 1 || requested_degree > 64)
        fail("LLC_PF_DEGREE must be between 1 and 64");
    if (learned_active() && requested_degree != 1)
        fail("learned comparisons require a prefetch degree of one");
    degree = requested_degree;

    if (policy != PolicyKind::none) {
        for (auto &predictor : predictors)
            predictor.reset(new Policy(policy));
    }

    if (const char *path = std::getenv("LLC_PF_EVENTS")) {
        open_log(events, path);
        events << "event_id,ingress_cycle,lookup_order,tag_cycle,cpu,pc,physical_line,mshr_occupancy,hit,"
               << "instr_id,operand,virtual_line,prefetch_use\n";
    }
    if (const char* path = std::getenv("LLC_PF_PREFETCH_EVENTS")) {
        prefetch_sample = numeric_env("LLC_PF_PREFETCH_SAMPLE", 1);
        if (!learned_active() || !prefetch_sample)
            fail("prefetch event tracing requires a learned scorer and positive sample period");
        open_log(prefetch_events, path);
        prefetch_events << "prefetch_id,cpu,trigger_event_id,trigger_instr_id,trigger_operand,trigger_pc,"
                        << "trigger_line,target_line,source_pc,source_offset,source_age,delta,score,roi_retired,"
                        << "pace_epoch,ingress_cycle,predict_start_cycle,ready_cycle,accepted_cycle,dispatch_cycle,"
                        << "fill_cycle,outcome,outcome_cycle,demand_cpu,demand_event_id,demand_instr_id,"
                        << "demand_operand,demand_ingress_cycle\n";
    }
    if (const char* path = std::getenv("LLC_PF_DEMAND_SERVICE")) {
        demand_service_sample = numeric_env("LLC_PF_DEMAND_SERVICE_SAMPLE", 1);
        if (policy_name != "nopf" || !demand_service_sample)
            fail("demand service tracing requires NoPF and positive sample period");
        open_log(demand_service, path);
        demand_service << "event_id,cpu,instr_id,operand,pc,physical_line,mshr_occupancy,"
                       << "ingress_cycle,dispatch_cycle,fill_cycle\n";
    }
    if (pace_instructions) {
        const auto path = std::getenv("LLC_PF_PACE_EVENTS");
        if (path) {
            open_log(pace_epochs, path);
            pace_epochs << "cpu,epoch,issued\n";
        }
    }
}

void CACHE::llc_pf_ingress(PACKET &packet)
{
    if (!demand_load(packet))
        return;

    const auto core = packet.cpu;
    const bool measured = core_in_roi(core);
    if (measured) {
        ++cores[core].ingress;
        packet.llc_pf_event_id = ++next_event;
        if (events.is_open() || prefetch_sample || demand_service_sample) {
            packet.llc_pf_ingress_cycle = current_core_cycle[core];
            packet.llc_pf_mshr_occupancy = MSHR.occupancy;
        }
    }
    if (learned_active()) {
        if (measured) {
            auto& lane = learned[core];
            const auto observation = correlation_model
                ? lane.correlation_history.observe(packet.address, packet.ip, current_core_cycle[core], *correlation_model)
                : lane.history.observe(packet.address, packet.ip);
            if (lane.waiting.size() < 32)
                lane.waiting.push_back({observation, trigger_of(packet)});
            else
                ++prediction_overflow;
        }
        return;
    }
    if (!predictors[core])
        return;

    // Observe during warmup, with issuance gated to the measured interval.
    const auto predictions = predictors[core]->access(packet.address, packet.ip);
    if (!measured)
        return;

    uint32_t accepted = 0;
    const auto lines_per_page = PAGE_SIZE / BLOCK_SIZE;
    for (const auto target : predictions) {
        if (accepted == degree || PQ.occupancy == PQ.SIZE || !below_issue_limit(core) || !pace_allows(core))
            break;
        ++candidates;
        if (target / lines_per_page != packet.address / lines_per_page)
            continue;
        if (llc_pf_try_issue(packet.address, target, packet.ip, core)) {
            predictors[core]->issued(target);
            ++accepted;
        }
    }
}

bool CACHE::llc_pf_eligible(uint64_t base, uint64_t target, uint32_t core)
{
    if (!target || target == base || PQ.occupancy == PQ.SIZE || outstanding.count(target))
        return false;

    PACKET candidate;
    candidate.address = target;
    candidate.full_addr = target << LOG2_BLOCK_SIZE;
    candidate.cpu = core;
    if (RQ.check_queue(&candidate) != -1 || WQ.check_queue(&candidate) != -1
        || PQ.check_queue(&candidate) != -1 || check_mshr(&candidate) != -1)
        return false;
    return get_way(target, get_set(target)) == NUM_WAY;
}

bool CACHE::llc_pf_try_issue(uint64_t base, uint64_t target, uint64_t pc, uint32_t core)
{
    if (!llc_pf_eligible(base, target, core))
        return false;

    if (issued == UINT32_MAX)
        fail("prefetch ID space exhausted");
    const uint32_t id = issued + 1;
    const auto before = PQ.occupancy;
    const auto previous_cpu = cpu;
    cpu = core;
    prefetch_line(pc, base << LOG2_BLOCK_SIZE, target << LOG2_BLOCK_SIZE, FILL_LLC, id);
    cpu = previous_cpu;
    if (PQ.occupancy == before)
        return false;
    outstanding.emplace(target, Prefetch{id, 0});
    ++issued;
    record_issue(core);
    cross_page_issued += target / llc_learned::page_lines != base / llc_learned::page_lines;
    return true;
}

void CACHE::llc_pf_tick()
{
    if (!learned_active())
        return;
    for (uint32_t core = 0; core < NUM_CPUS; ++core) {
        if (!core_in_roi(core))
            continue;
        auto& lane = learned[core];
        const uint64_t now = current_core_cycle[core];
        if (lane.ready && lane.ready <= now) {
            ++prediction_completed;
            const auto ready = lane.ready;
            lane.ready = 0;
            const auto& observation = lane.running.observation;
            const bool below_limit = below_issue_limit(core);
            if (below_limit && !pace_allows(core)) {
                ++prediction_pace_blocked;
            } else if (below_limit) {
                size_t best = observation.count;
                int32_t best_score = prediction_threshold - 1;
                bool passes_threshold = false;
                for (size_t i = 0; i < observation.count; ++i) {
                    passes_threshold |= lane.scores[i] >= prediction_threshold;
                    if (lane.scores[i] <= best_score)
                        continue;
                    ++candidates;
                    if (llc_pf_eligible(observation.line, observation.candidates[i].line, core)) {
                        best = i;
                        best_score = lane.scores[i];
                    }
                }
                bool accepted = false;
                if (best < observation.count)
                    accepted = llc_pf_try_issue(observation.line, observation.candidates[best].line, observation.pc, core);
                if (accepted && prefetch_sample && (issued - 1) % prefetch_sample == 0) {
                    PrefetchTiming sample{};
                    sample.trigger = lane.running.trigger;
                    sample.cpu = core;
                    sample.pc = observation.pc;
                    sample.line = observation.line;
                    sample.candidate = observation.candidates[best];
                    sample.score = best_score;
                    sample.retired = retired_in_roi(core);
                    sample.epoch = pace_instructions ? issue_epoch(core) : 0;
                    sample.start = lane.start;
                    sample.ready = ready;
                    sample.accepted = now;
                    prefetch_timing.emplace(issued, sample);
                }
                prediction_abstained += !passes_threshold;
                prediction_filtered += passes_threshold && !accepted;
            }
        }
        if (lane.ready || lane.waiting.empty() || lane.next_start > now)
            continue;
        lane.running = lane.waiting.front();
        lane.waiting.pop_front();
        const auto& observation = lane.running.observation;
        if (page_pc_model)
            lane.scores = page_pc_model->score(observation);
        else if (correlation_model)
            lane.scores = neural_model->score_compact(observation, lane.state);
        else
            lane.scores = neural_model->score(observation, lane.state);
        lane.start = now;
        lane.ready = now + prediction_latency;
        lane.next_start = now + prediction_interval;
        ++prediction_started;
    }
}

void CACHE::llc_pf_dispatch(const PACKET &packet)
{
    if (prefetch_sample && packet.type == PREFETCH) {
        const auto sample = prefetch_timing.find(packet.pf_metadata);
        if (sample != prefetch_timing.end() && sample->second.candidate.line == packet.address)
            sample->second.dispatch = current_core_cycle[packet.cpu];
    }
    if (demand_service_sample && demand_load(packet) && packet.llc_pf_event_id && core_in_roi(packet.cpu)
        && sampled_demands[packet.cpu]++ % demand_service_sample == 0) {
        DemandService sample{};
        sample.trigger = trigger_of(packet);
        sample.cpu = packet.cpu;
        sample.pc = packet.ip;
        sample.line = packet.address;
        sample.mshr_occupancy = packet.llc_pf_mshr_occupancy;
        sample.dispatch = current_core_cycle[packet.cpu];
        demand_timing.emplace(packet.llc_pf_event_id, sample);
    }
}

void CACHE::llc_pf_rq_disposition(const PACKET &packet, LlcRqDisposition disposition)
{
    if (!demand_load(packet) || !core_in_roi(packet.cpu))
        return;

    auto &counters = cores[packet.cpu];
    switch (disposition) {
    case LlcRqDisposition::forwarded:
        ++counters.forwarded;
        break;
    case LlcRqDisposition::merged:
        ++counters.merged;
        break;
    case LlcRqDisposition::full:
        ++counters.full;
        break;
    }
}

void CACHE::llc_pf_invalidate(uint64_t address)
{
    auto tracked = outstanding.find(address);
    if (tracked != outstanding.end() && tracked->second.fill_cycle) {
        ++unused;
        record_outcome(tracked->second.id, "unused", current_core_cycle[0]);
        discard_spp_feedback(address);
        outstanding.erase(tracked);
    }
}

void CACHE::llc_pf_demand(const PACKET &packet, bool hit)
{
    bool timely_use = false;
    if (hit) {
        auto tracked = outstanding.find(packet.address);
        if (tracked != outstanding.end() && tracked->second.fill_cycle) {
            if (demand_load(packet) && packet.llc_pf_event_id && core_in_roi(packet.cpu)) {
                ++timely;
                timely_use = true;
                record_outcome(tracked->second.id, "timely", current_core_cycle[packet.cpu], &packet);
                const auto slack = current_core_cycle[packet.cpu] - tracked->second.fill_cycle;
                slack_sum += slack;
                slack_min = std::min(slack_min, slack);
                slack_max = std::max(slack_max, slack);
            } else {
                ++redundant;
                record_outcome(tracked->second.id, "redundant", current_core_cycle[packet.cpu]);
                discard_spp_feedback(packet.address);
            }
            outstanding.erase(tracked);
        }
    }
    if (!demand_load(packet))
        return;
    if (policy == PolicyKind::spp) {
        for (auto &predictor : predictors)
            predictor->demand_outcome(packet.address, hit);
    }
    if (!packet.llc_pf_event_id || !core_in_roi(packet.cpu))
        return;
    if (hit)
        ++cores[packet.cpu].hit;
    else
        ++cores[packet.cpu].miss;
    if (events.is_open()) {
        events << packet.llc_pf_event_id << ',' << packet.llc_pf_ingress_cycle << ','
               << ++next_lookup << ',' << current_core_cycle[packet.cpu] << ','
               << packet.cpu << ',' << packet.ip << ','
               << packet.address << ',' << packet.llc_pf_mshr_occupancy << ',' << hit << ','
               << packet.instr_id << ',' << packet.data_index << ',' << packet.llc_pf_virtual_line << ','
               << (timely_use ? "timely" : packet.llc_pf_late_use ? "late" : "none") << '\n';
    }
}

void CACHE::llc_pf_late(const PACKET &prefetch, PACKET &demand)
{
    if (demand_load(demand) && demand.llc_pf_event_id && core_in_roi(demand.cpu))
        demand.llc_pf_late_use = finish_prefetch(prefetch, late, &demand);
    else
        finish_prefetch(prefetch, redundant);
}

void CACHE::llc_pf_redundant(const PACKET &packet)
{
    if (packet.type == PREFETCH)
        finish_prefetch(packet, redundant);
}

void CACHE::llc_pf_fill(const PACKET &packet, const BLOCK &victim)
{
    if (victim.valid) {
        auto tracked = outstanding.find(victim.address);
        if (tracked != outstanding.end() && tracked->second.fill_cycle) {
            ++unused;
            record_outcome(tracked->second.id, "unused", current_core_cycle[packet.cpu]);
            outstanding.erase(tracked);
        }
    }
    if (demand_service_sample && packet.llc_pf_event_id) {
        const auto sample = demand_timing.find(packet.llc_pf_event_id);
        if (sample != demand_timing.end() && sample->second.line == packet.address)
            sample->second.fill = current_core_cycle[packet.cpu];
    }
    if (prefetch_sample && packet.pf_metadata) {
        const auto sample = prefetch_timing.find(packet.pf_metadata);
        if (sample != prefetch_timing.end() && sample->second.candidate.line == packet.address
            && sample->second.dispatch && !sample->second.fill)
            sample->second.fill = current_core_cycle[packet.cpu];
    }
    const bool is_prefetch = packet.type == PREFETCH;
    if (is_prefetch) {
        auto tracked = outstanding.find(packet.address);
        if (tracked != outstanding.end() && tracked->second.id == packet.pf_metadata) {
            tracked->second.fill_cycle = current_core_cycle[packet.cpu];
            ++fills;
        }
    }
    if (policy == PolicyKind::spp) {
        for (auto &predictor : predictors)
            predictor->fill(packet.address, is_prefetch, victim.address, victim.valid);
    }
}

void CACHE::llc_prefetcher_print_config()
{
    std::cout << "LLC_PF_policy " << policy_name << '\n';
    emit("cores", NUM_CPUS);
    emit("degree", degree);
    emit("skip", llc_pf_trace_skip());
    if (learned_active()) {
        emit("prediction_latency", prediction_latency);
        emit("prediction_interval", prediction_interval);
        emit("issue_limit", issue_limit);
        std::cout << "LLC_PF_threshold " << prediction_threshold << '\n';
    }
    if (correlation_model)
        emit("correlation", 1);
    if (neural_model && neural_model->nonlinear_width())
        emit("nonlinear_head_width", neural_model->nonlinear_width());
    if (pace_instructions) {
        emit("pace_instructions", pace_instructions);
        emit("pace_requests", pace_requests);
        if (!learned_active())
            emit("issue_limit", issue_limit);
    }
    if (prefetch_sample)
        emit("prefetch_sample", prefetch_sample);
    if (demand_service_sample)
        emit("demand_service_sample", demand_service_sample);
}

void CACHE::llc_prefetcher_final_stats()
{
    for (uint32_t core = 0; core < NUM_CPUS; ++core) {
        emit_core(core, "ingress", cores[core].ingress);
        emit_core(core, "hit", cores[core].hit);
        emit_core(core, "miss", cores[core].miss);
        emit_core(core, "rq_forwarded", cores[core].forwarded);
        emit_core(core, "rq_merged", cores[core].merged);
        emit_core(core, "rq_full", cores[core].full);
        if (learned_active() || pace_instructions)
            emit_core(core, "issued", budgets[core].issued);
    }
    emit("prefetch_candidates", candidates);
    emit("prefetch_issued", issued);
    emit("prefetch_fills", fills);
    emit("prefetch_timely", timely);
    emit("prefetch_slack_cycles_sum", slack_sum);
    emit("prefetch_slack_cycles_min", timely ? slack_min : 0);
    emit("prefetch_slack_cycles_max", slack_max);
    emit("prefetch_late", late);
    emit("prefetch_unused", unused);
    emit("prefetch_redundant", redundant);
    emit("prefetch_unresolved", outstanding.size());
    emit("dram_demand_read_bytes", dram_demand_read);
    emit("dram_prefetch_read_bytes", dram_prefetch_read);
    emit("dram_write_bytes", dram_write);
    if (learned_active()) {
        emit("prediction_started", prediction_started);
        emit("prediction_completed", prediction_completed);
        emit("prediction_overflow", prediction_overflow);
        emit("prediction_abstained", prediction_abstained);
        emit("prediction_filtered", prediction_filtered);
        emit("cross_page_issued", cross_page_issued);
        if (pace_instructions)
            emit("prediction_pace_blocked", prediction_pace_blocked);
    }

    close_log(events);
    write_timing_logs();
}

// Keep Pythia's post-tag prefetch interface inactive.
uint32_t CACHE::llc_prefetcher_operate(uint64_t, uint64_t, uint8_t, uint8_t, uint32_t metadata)
{
    return metadata;
}

uint32_t CACHE::llc_prefetcher_cache_fill(uint64_t, uint32_t, uint32_t, uint8_t, uint64_t, uint32_t metadata)
{
    return metadata;
}

uint32_t CACHE::llc_prefetcher_prefetch_hit(uint64_t, uint64_t, uint32_t metadata)
{
    return metadata;
}

void CACHE::llc_prefetcher_broadcast_bw(uint8_t) {}
void CACHE::llc_prefetcher_broadcast_ipc(uint8_t) {}
void CACHE::llc_prefetcher_broadcast_acc(uint32_t) {}
