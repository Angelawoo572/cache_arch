#include "cache.h"
#include "learned_llc.h"
#include "policies.h"

#include <array>
#include <cerrno>
#include <deque>
#include <fstream>
#include <memory>
#include <unordered_map>

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

struct LearnedCore {
    llc_learned::History history;
    llc_learned::State state;
    std::deque<llc_learned::Observation> waiting;
    llc_learned::Observation running{};
    std::array<int32_t, llc_learned::max_candidates> scores{};
    uint64_t ready = 0;
    uint64_t next_start = 0;
    uint64_t issued = 0;
};

std::array<LearnedCore, NUM_CPUS> learned;
std::unique_ptr<llc_learned::Model> neural_model;
std::unique_ptr<llc_learned::PagePc> page_pc_model;
uint64_t prediction_latency = 0;
uint64_t prediction_interval = 0;
uint64_t issue_limit = 0;
int32_t prediction_threshold = -32768;
uint64_t prediction_started = 0;
uint64_t prediction_completed = 0;
uint64_t prediction_overflow = 0;
uint64_t prediction_abstained = 0;
uint64_t prediction_filtered = 0;
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

bool finish_prefetch(const PACKET &packet, uint64_t &counter)
{
    auto tracked = outstanding.find(packet.address);
    if (tracked == outstanding.end() || tracked->second.id != packet.pf_metadata)
        return false;
    ++counter;
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
        issue_limit = numeric_env("LLC_PF_ISSUE_LIMIT", 0);
        prediction_threshold = threshold_env();
    }
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
        events.open(path);
        if (!events)
            fail("cannot open demand event CSV");
        events << "event_id,ingress_cycle,lookup_order,tag_cycle,cpu,pc,physical_line,mshr_occupancy,hit,"
               << "instr_id,operand,virtual_line,prefetch_use\n";
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
        if (events.is_open()) {
            packet.llc_pf_ingress_cycle = current_core_cycle[core];
            packet.llc_pf_mshr_occupancy = MSHR.occupancy;
        }
    }
    if (learned_active()) {
        if (measured) {
            auto& lane = learned[core];
            auto observation = lane.history.observe(packet.address, packet.ip);
            if (lane.waiting.size() < 32)
                lane.waiting.push_back(observation);
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
        if (accepted == degree || PQ.occupancy == PQ.SIZE)
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
            lane.ready = 0;
            const auto& observation = lane.running;
            if (!issue_limit || lane.issued < issue_limit) {
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
                lane.issued += accepted;
                prediction_abstained += !passes_threshold;
                prediction_filtered += passes_threshold && !accepted;
            }
        }
        if (lane.ready || lane.waiting.empty() || lane.next_start > now)
            continue;
        lane.running = lane.waiting.front();
        lane.waiting.pop_front();
        lane.scores = neural_model ? neural_model->score(lane.running, lane.state) : page_pc_model->score(lane.running);
        lane.ready = now + prediction_latency;
        lane.next_start = now + prediction_interval;
        ++prediction_started;
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
                const auto slack = current_core_cycle[packet.cpu] - tracked->second.fill_cycle;
                slack_sum += slack;
                slack_min = std::min(slack_min, slack);
                slack_max = std::max(slack_max, slack);
            } else {
                ++redundant;
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
        demand.llc_pf_late_use = finish_prefetch(prefetch, late);
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
            outstanding.erase(tracked);
        }
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
        if (learned_active())
            emit_core(core, "issued", learned[core].issued);
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
    }

    if (events.is_open()) {
        events.close();
        if (!events)
            fail("writing demand event CSV failed");
    }
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
