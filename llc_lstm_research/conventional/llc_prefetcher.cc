#include "cache.h"
#include "policies.h"

#include <array>
#include <cerrno>
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

void finish_prefetch(const PACKET &packet, uint64_t &counter)
{
    auto tracked = outstanding.find(packet.address);
    if (tracked != outstanding.end() && tracked->second.id == packet.pf_metadata) {
        ++counter;
        discard_spp_feedback(packet.address);
        outstanding.erase(tracked);
    }
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
    if (selected == policies.end())
        fail("unknown policy " + policy_name);
    policy = selected->second;
    const auto requested_degree = numeric_env("LLC_PF_DEGREE", 1);
    if (requested_degree < 1 || requested_degree > 64)
        fail("LLC_PF_DEGREE must be between 1 and 64");
    degree = requested_degree;

    if (policy != PolicyKind::none) {
        for (auto &predictor : predictors)
            predictor.reset(new Policy(policy));
    }

    if (const char *path = std::getenv("LLC_PF_EVENTS")) {
        events.open(path);
        if (!events)
            fail("cannot open demand event CSV");
        events << "event_id,ingress_cycle,lookup_order,tag_cycle,cpu,pc,physical_line,mshr_occupancy,hit\n";
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
        if (!target || target == packet.address || target / lines_per_page != packet.address / lines_per_page)
            continue;
        if (outstanding.count(target))
            continue;

        PACKET candidate;
        candidate.address = target;
        candidate.full_addr = target << LOG2_BLOCK_SIZE;
        candidate.cpu = core;
        if (RQ.check_queue(&candidate) != -1 || WQ.check_queue(&candidate) != -1
            || PQ.check_queue(&candidate) != -1 || check_mshr(&candidate) != -1)
            continue;

        const auto set = get_set(target);
        bool resident = false;
        for (uint32_t way = 0; way < NUM_WAY; ++way) {
            if (block[set][way].valid && block[set][way].tag == target) {
                resident = true;
                break;
            }
        }
        if (resident)
            continue;

        if (issued == UINT32_MAX)
            fail("prefetch ID space exhausted");
        const uint32_t id = issued + 1;
        const auto before = PQ.occupancy;
        const auto previous_cpu = cpu;
        cpu = core;
        prefetch_line(packet.ip, packet.address << LOG2_BLOCK_SIZE,
                      candidate.full_addr, FILL_LLC, id);
        cpu = previous_cpu;
        if (PQ.occupancy == before)
            continue;
        outstanding.emplace(target, Prefetch{id, 0});
        predictors[core]->issued(target);
        ++issued;
        ++accepted;
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
    if (hit) {
        auto tracked = outstanding.find(packet.address);
        if (tracked != outstanding.end() && tracked->second.fill_cycle) {
            if (demand_load(packet) && packet.llc_pf_event_id && core_in_roi(packet.cpu)) {
                ++timely;
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
               << packet.address << ',' << packet.llc_pf_mshr_occupancy << ',' << hit << '\n';
    }
}

void CACHE::llc_pf_late(const PACKET &prefetch, const PACKET &demand)
{
    if (demand_load(demand) && demand.llc_pf_event_id && core_in_roi(demand.cpu))
        finish_prefetch(prefetch, late);
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
