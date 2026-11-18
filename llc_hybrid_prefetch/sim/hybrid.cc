#include "ooo_cpu.h"
#include "uncore.h"
#include "hybrid.h"
#include "stride.h"
#include <zlib.h>
#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdarg>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
using U = unsigned long long;
unsigned active_cores;
uint64_t warm_count, roi_count, skip[NUM_CPUS] = {};
unsigned page_slots[NUM_CPUS];
bool stride_mode, compact_log;
std::string directory;
struct Log {
    bool enabled = true;
    FILE *file = nullptr;
    gzFile zipped = nullptr;
    void open(const std::string &name, const char *header, bool compress) {
        const char *mode = std::getenv("HYBRID_DETAIL");
        const bool summary = name == "cores" || name == "stats";
        enabled =
            summary || !mode || !std::strcmp(mode, "1") ||
            (!std::strcmp(mode, "victims") && (name == "victims" || name == "fills" || name == "lookups"));
        if (!enabled)
            return;
        auto path = directory + "/" + name + ".csv";
        if (compress) {
            zipped = gzopen((path + ".gz").c_str(), "wb1");
            if (!zipped)
                throw std::runtime_error("cannot create " + path);
            gzbuffer(zipped, 128 * 1024);
        } else {
            file = std::fopen(path.c_str(), "w");
            if (!file)
                throw std::runtime_error("cannot create " + path);
            std::setvbuf(file, nullptr, _IOFBF, 128 * 1024);
        }
        write("%s\n", header);
    }
    void write(const char *fmt, ...) {
        if (!enabled)
            return;
        char buf[640];
        va_list ap;
        va_start(ap, fmt);
        int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        if (n < 0 || static_cast<size_t>(n) >= sizeof(buf))
            throw std::runtime_error("CSV row overflow");
        if (zipped ? gzwrite(zipped, buf, n) != n : std::fwrite(buf, 1, n, file) != static_cast<size_t>(n))
            throw std::runtime_error("CSV write failure");
    }
    void close() {
        if ((zipped && gzclose(zipped) != Z_OK) || (file && std::fclose(file)))
            throw std::runtime_error("CSV close failed");
        zipped = nullptr;
        file = nullptr;
    }
};
Log loads, candidates, heads, cores, totals, page_log, victims, fills, lookups;
bool pollution_probe;
std::unique_ptr<StridePrefetcher> strides[NUM_CPUS];
std::unordered_map<uint64_t, uint64_t> maps[NUM_CPUS];
std::array<uint64_t, NUM_INSTR_SOURCES> head_counts[NUM_CPUS]{};
std::map<std::tuple<unsigned, uint64_t, unsigned>, int> plan;
uint64_t next_id, plan_hits, issued, demand_bytes, prefetch_bytes, write_bytes;
uint64_t cycle_count, mshr_sum, drq_sum, dwq_sum;
unsigned warm_rq, warm_mshr, warm_wq, warm_dram_rq, warm_dram_wq, warm_private_wq;
uint64_t head_union[NUM_CPUS] = {}, ambiguous_head[NUM_CPUS] = {};
const char *stride_reason;
int64_t stride_delta;
struct Candidate {
    uint64_t id, base, target, instr, pc;
    unsigned core, operand;
    bool filled = false, claimed = false, late = false;
};
std::unordered_map<uint64_t, Candidate> pending;
bool measured() {
    return all_warmup_complete > NUM_CPUS;
}
uint64_t ordinal(unsigned core, uint64_t id) {
    return skip[core] + id + 1;
}
bool roi_id(uint64_t id) {
    return id >= warm_count && id - warm_count < roi_count;
}
bool demand(const PACKET &p) {
    return p.type == LOAD && !p.instruction && !p.tlb_access;
}
void candidate(const Candidate &c, const char *stage, const char *why = "-", const PACKET *d = nullptr) {
    candidates.write("%llu,%u,%llu,%u,%llu,%llu,%llu,%llu,%s,%s,%d,%lld,%d\n", U(c.id), c.core, U(c.instr),
                     c.operand, U(c.pc), U(c.base), U(c.target), U(current_core_cycle[c.core]), stage, why,
                     d ? int(d->cpu) : -1, d ? (long long)ordinal(d->cpu, d->instr_id) : -1LL,
                     d ? int(d->data_index) : -1);
}
void register_address(unsigned c, uint64_t addr) {
    if (!addr)
        return;
    auto &map = maps[c];
    uint64_t vpn = addr >> LOG2_PAGE_SIZE;
    if (map.find(vpn) != map.end())
        return;
    // Assign 64-page chunks round-robin among address spaces; assignment uses trace order, not memory
    // arrival.
    uint64_t i = map.size();
    auto ppn = ((i / 64) * NUM_CPUS + page_slots[c]) * 64 + i % 64 + 1;
    if (ppn >= DRAM_PAGES)
        throw std::runtime_error("address space exceeds physical-memory placement");
    map.emplace(vpn, ppn);
}
const char *cache_name(const CACHE &c) {
    if (c.cache_type == IS_L1D)
        return "L1";
    if (c.cache_type == IS_L2C)
        return "L2";
    if (c.cache_type == IS_LLC)
        return "LLC";
    return nullptr;
}
void cache_line(CACHE &c, const PACKET &p, const char *stage, int hit) {
    const auto level = cache_name(c);
    if (!level || !measured() || !roi_id(p.instr_id) || !demand(p))
        return;
    if (compact_log && c.cache_type == IS_L1D)
        return;
    loads.write("%u,%llu,%u,%llu,%llu,%llu,%s,%s,%llu,%d,%u,%u,%u,%u,%u\n", p.cpu,
                U(ordinal(p.cpu, p.instr_id)), p.data_index, U(p.ip), U(p.hybrid_virtual_line), U(p.address),
                level, stage, U(current_core_cycle[p.cpu]), hit, c.MSHR.occupancy, c.RQ.occupancy,
                c.PQ.occupancy, uncore.DRAM.RQ[0].occupancy, uncore.DRAM.WQ[0].occupancy);
}
void stat(const std::string &key, uint64_t value) {
    totals.write("%s,%llu\n", key.c_str(), U(value));
}
} // namespace
void hybrid_init(unsigned active, uint64_t w, uint64_t r) {
    if (!active || active > NUM_CPUS || !r)
        throw std::runtime_error("invalid core count or ROI length");
    active_cores = active;
    warm_count = w;
    roi_count = r;
    const char *dest = std::getenv("HYBRID_OUT");
    if (!dest)
        throw std::runtime_error("set HYBRID_OUT to a new run directory");
    directory = dest;
    auto name = std::getenv("LLC_PF_POLICY");
    if (!name || (std::strcmp(name, "stride") && std::strcmp(name, "nopf")))
        throw std::runtime_error("LLC_PF_POLICY must be stride or nopf");
    stride_mode = std::strcmp(name, "stride") == 0;
    const char *detail = std::getenv("HYBRID_DETAIL");
    if (detail && std::strcmp(detail, "0") && std::strcmp(detail, "1") && std::strcmp(detail, "victims"))
        throw std::runtime_error("HYBRID_DETAIL must be 0, 1 or victims");
    const bool compressed =
        std::getenv("HYBRID_COMPRESS") && !std::strcmp(std::getenv("HYBRID_COMPRESS"), "1");
    compact_log = std::getenv("HYBRID_COMPACT") && !std::strcmp(std::getenv("HYBRID_COMPACT"), "1");
    loads.open(
        "loads",
        "cpu,instr_id,operand,pc,virtual_line,physical_line,level,stage,cycle,hit,mshr,rq,pq,dram_rq,dram_wq",
        compressed);
    candidates.open(
        "candidates",
        "id,cpu,instr_id,operand,pc,base,target,cycle,stage,reason,demand_cpu,demand_instr_id,demand_operand",
        compressed);
    heads.open("head", "cpu,instr_id,operand,cycles", compressed);
    cores.open("cores", "cpu,warmup,instructions,cycles", false);
    totals.open("stats", "key,value", false);
    if (auto env = std::getenv("HYBRID_SKIPS")) {
        std::istringstream in(env);
        std::string value;
        for (unsigned c = 0; c < active; ++c) {
            if (!std::getline(in, value, ','))
                throw std::runtime_error("HYBRID_SKIPS needs one entry per active core");
            skip[c] = std::stoull(value);
        }
        if (std::getline(in, value, ','))
            throw std::runtime_error("too many HYBRID_SKIPS");
    } else if (auto env = std::getenv("LLC_PF_SKIP"))
        std::fill(skip, skip + active, std::stoull(env));
    page_log.open("pages", "cpu,page_slot,virtual_page,physical_page", compressed);
    victims.open("victims", "cpu,prefetch_id,victim_line,target_line,cycle", compressed);
    pollution_probe = detail && !std::strcmp(detail, "victims");
    if (pollution_probe) {
        fills.open("fills", "cycle,cpu,line,victim_valid,victim_line,kind,prefetch_id", compressed);
        lookups.open("lookups", "cycle,cpu,instr_id,operand,pc,virtual_line,line,hit", compressed);
    }
    for (unsigned c = 0; c < active; ++c)
        page_slots[c] = c;
    if (auto env = std::getenv("HYBRID_PAGE_SLOTS")) {
        std::istringstream in(env);
        std::string value;
        std::array<bool, NUM_CPUS> used{};
        for (unsigned c = 0; c < active; ++c) {
            if (!std::getline(in, value, ','))
                throw std::runtime_error("one page slot per active core required");
            unsigned slot = std::stoul(value);
            if (slot >= NUM_CPUS || used[slot])
                throw std::runtime_error("invalid or repeated page slot");
            page_slots[c] = slot;
            used[slot] = true;
        }
        if (std::getline(in, value, ','))
            throw std::runtime_error("too many page slots");
    }
    if (auto path = std::getenv("HYBRID_PLAN")) {
        if (!stride_mode)
            throw std::runtime_error("counterfactual requires Stride baseline");
        std::ifstream in(path);
        std::string line;
        if (!std::getline(in, line) || line.find("cpu,instr_id,operand,delta") != 0)
            throw std::runtime_error("invalid intervention CSV header");
        while (std::getline(in, line)) {
            if (line.empty())
                continue;
            std::replace(line.begin(), line.end(), ',', ' ');
            unsigned c, operand;
            uint64_t id;
            int delta;
            std::istringstream row(line);
            if (!(row >> c >> id >> operand >> delta) || c >= active || operand >= NUM_INSTR_SOURCES ||
                !delta || delta < -63 || delta > 63 ||
                !plan.emplace(std::make_tuple(c, id, operand), delta).second)
                throw std::runtime_error("invalid same-page intervention row");
        }
    }
    for (unsigned c = 0; c < active; ++c)
        if (stride_mode)
            strides[c].reset(new StridePrefetcher("stride"));
}
unsigned hybrid_active() {
    return active_cores;
}
uint64_t hybrid_skip(unsigned c) {
    return skip[c];
}
bool hybrid_may_fetch(const O3_CPU &core) {
    return core.cpu < active_cores &&
           core.instr_unique_id < (measured() ? warm_count + roi_count : warm_count);
}
FILE *hybrid_open(const char *raw, unsigned c) {
    std::string path(raw);
    const bool xz = path.size() > 3 && path.substr(path.size() - 3) == ".xz";
    const bool gz = path.size() > 3 && path.substr(path.size() - 3) == ".gz";
    FILE *f = nullptr;
    if (xz || gz) {
        std::string q = "'";
        for (char ch : path)
            q += ch == '\'' ? "'\\''" : std::string(1, ch);
        q += "'";
        std::string cmd = (xz ? "xz -dc -- " : "gzip -dc -- ") + q;
        f = popen(cmd.c_str(), "r");
    } else
        f = std::fopen(raw, "rb");
    if (!f)
        throw std::runtime_error("cannot open trace " + path);
    // Read the skipped prefix only into the address-placement map. No cache, predictor,
    // translation timing, branch or OOO state is touched.
    if (skip[c]) {
        std::vector<input_instr> records(1024);
        for (uint64_t remaining = skip[c]; remaining;) {
            size_t n = std::min<uint64_t>(remaining, records.size());
            if (std::fread(records.data(), sizeof(input_instr), n, f) != n)
                throw std::runtime_error("trace too short");
            for (size_t j = 0; j < n; ++j) {
                const auto &rec = records[j];
                register_address(c, rec.ip);
                for (auto addr : rec.destination_memory)
                    register_address(c, addr);
                for (auto addr : rec.source_memory)
                    register_address(c, addr);
            }
            remaining -= n;
        }
    }
    return f;
}
void hybrid_observe_instruction(unsigned c, const ooo_model_instr &i) {
    register_address(c, i.ip);
    for (auto addr : i.destination_memory)
        register_address(c, addr);
    for (unsigned op = 0; op < NUM_INSTR_SOURCES; ++op) {
        auto addr = i.source_memory[op];
        register_address(c, addr);
        if (addr && measured() && roi_id(i.instr_id))
            loads.write("%u,%llu,%u,%llu,%llu,0,CORE,trace,%llu,-1,0,0,0,0,0\n", c, U(ordinal(c, i.instr_id)),
                        op, U(i.ip), U(addr >> LOG2_BLOCK_SIZE), U(current_core_cycle[c]));
    }
}
uint64_t hybrid_ppn(unsigned core, uint64_t va) {
    return maps[core].at(va >> LOG2_PAGE_SIZE);
}
void hybrid_head(const O3_CPU &c) {
    if (!measured() || c.cpu >= active_cores || !c.ROB.occupancy)
        return;
    const auto &i = c.ROB.entry[c.ROB.head];
    if (!i.ip || i.executed == COMPLETED || !roi_id(i.instr_id))
        return;
    unsigned num = 0;
    for (unsigned op = 0; op < NUM_INSTR_SOURCES; ++op) {
        auto ix = i.lq_index[op];
        if (ix >= c.LQ.SIZE)
            continue;
        const auto &l = c.LQ.entry[ix];
        if (l.instr_id == i.instr_id && l.data_index == op && l.fetched == INFLIGHT) {
            ++head_counts[c.cpu][op];
            ++num;
        }
    }
    if (num)
        ++head_union[c.cpu];
    if (num > 1)
        ++ambiguous_head[c.cpu];
}
void hybrid_retire(unsigned c, uint64_t id) {
    for (unsigned op = 0; op < NUM_INSTR_SOURCES; ++op)
        if (head_counts[c][op]) {
            heads.write("%u,%llu,%u,%llu\n", c, U(ordinal(c, id)), op, U(head_counts[c][op]));
            head_counts[c][op] = 0;
        }
}
void hybrid_cache(CACHE &c, const PACKET &p, const char *stage, int hit) {
    cache_line(c, p, stage, hit);
    if (pollution_probe && c.cache_type == IS_LLC && hit >= 0 && measured() && demand(p) &&
        roi_id(p.instr_id))
        lookups.write("%llu,%u,%llu,%u,%llu,%llu,%llu,%d\n", U(current_core_cycle[p.cpu]), p.cpu,
                      U(ordinal(p.cpu, p.instr_id)), p.data_index, U(p.ip), U(p.hybrid_virtual_line),
                      U(p.address), hit);
    if (c.cache_type == IS_LLC && hit == 1 && demand(p)) {
        auto it = pending.find(p.address);
        if (it != pending.end() && it->second.filled && !it->second.claimed) {
            candidate(it->second, "timely", "-", &p);
            pending.erase(it);
        }
    }
}
void hybrid_stride_state(const char *reason, int64_t delta) {
    stride_reason = reason;
    stride_delta = delta;
}
void hybrid_ingress(CACHE &c, const PACKET &p) {
    if (!demand(p))
        return;
    cache_line(c, p, "ingress", -1);
    if (!stride_mode)
        return;
    std::vector<uint64_t> native;
    stride_reason = "cold";
    stride_delta = 0;
    strides[p.cpu]->invoke_prefetcher(p.ip, p.address << LOG2_BLOCK_SIZE, 0, LOAD, native);
    if (!measured() || !roi_id(p.instr_id))
        return;
    Candidate item{};
    item.id = ++next_id;
    item.base = p.address;
    item.instr = ordinal(p.cpu, p.instr_id);
    item.pc = p.ip;
    item.core = p.cpu;
    item.operand = p.data_index;
    auto diagnostic = plan.find(std::make_tuple(p.cpu, item.instr, item.operand));
    bool intervention = diagnostic != plan.end();
    if (intervention) {
        ++plan_hits;
        stride_delta = diagnostic->second;
    } else if (std::strcmp(stride_reason, "match")) {
        candidate(item, "silent", stride_reason);
        return;
    }
    item.target = static_cast<uint64_t>(static_cast<int64_t>(item.base) + stride_delta);
    candidate(item, "proposed", intervention ? "intervention" : "stride");
    PACKET probe;
    probe.cpu = p.cpu;
    probe.address = item.target;
    probe.full_addr = item.target << LOG2_BLOCK_SIZE;
    const char *reject = nullptr;
    if (!item.target || item.base == item.target)
        reject = "self";
    else if (item.base / 64 != item.target / 64)
        reject = "page";
    else if (!intervention && std::find(native.begin(), native.end(), probe.full_addr) == native.end())
        throw std::runtime_error("native Stride returned unexpected target");
    else if (c.get_way(item.target, c.get_set(item.target)) != c.NUM_WAY)
        reject = "resident";
    else if (c.check_mshr(&probe) != -1)
        reject = "mshr";
    else if (c.RQ.check_queue(&probe) != -1)
        reject = "rq";
    else if (c.WQ.check_queue(&probe) != -1)
        reject = "wq";
    else if (c.PQ.check_queue(&probe) != -1)
        reject = "pq";
    else if (pending.count(item.target))
        reject = "outstanding";
    else if (c.PQ.occupancy == c.PQ.SIZE)
        reject = "pq_full";
    if (reject) {
        candidate(item, "rejected", reject);
        return;
    }
    auto old = c.cpu;
    c.cpu = p.cpu;
    auto was = c.PQ.occupancy;
    c.prefetch_line(p.ip, p.full_addr, probe.full_addr, FILL_LLC, item.id);
    c.cpu = old;
    if (c.PQ.occupancy == was) {
        candidate(item, "rejected", "enqueue");
        return;
    }
    ++issued;
    candidate(item, "accepted", intervention ? "intervention" : "stride");
    pending.emplace(item.target, item);
}
void hybrid_terminal(const PACKET &req, const char *stage, const PACKET *d) {
    auto it = pending.find(req.address);
    if (it == pending.end() || it->second.id != req.pf_metadata || it->second.claimed)
        return;
    if (d && !demand(*d)) {
        candidate(it->second, "redundant", "other_consumer");
        pending.erase(it);
        return;
    }
    candidate(it->second, stage, "-", d);
    if (d) {
        it->second.late = true;
        it->second.claimed = true;
    } else
        pending.erase(it);
}
void hybrid_dispatch(const PACKET &p) {
    if (demand(p))
        cache_line(uncore.LLC, p, "dispatch", -1);
    if (p.type != PREFETCH)
        return;
    auto it = pending.find(p.address);
    if (it != pending.end() && it->second.id == p.pf_metadata)
        candidate(it->second, "dispatch");
}
void hybrid_fill(CACHE &c, const PACKET &p, const BLOCK &victim) {
    if (pollution_probe && measured())
        fills.write("%llu,%u,%llu,%d,%llu,%s,%u\n", U(current_core_cycle[p.cpu]), p.cpu, U(p.address),
                    int(victim.valid), U(victim.address),
                    p.type == PREFETCH    ? "prefetch"
                    : p.type == WRITEBACK ? "writeback"
                    : p.type == RFO       ? "rfo"
                    : p.instruction       ? "instruction"
                                          : "load",
                    p.type == PREFETCH ? p.pf_metadata : 0);
    if (victim.valid && p.type == PREFETCH && measured())
        victims.write("%u,%u,%llu,%llu,%llu\n", p.cpu, p.pf_metadata, U(victim.address), U(p.address),
                      U(current_core_cycle[p.cpu]));
    if (victim.valid) {
        auto it = pending.find(victim.address);
        if (it != pending.end() && it->second.filled && !it->second.claimed) {
            candidate(it->second, "evicted", "unused");
            pending.erase(it);
        }
    }
    if (demand(p))
        cache_line(c, p, "fill", -1);
    auto it = pending.find(p.address);
    if (it == pending.end())
        return;
    if (it->second.late) {
        candidate(it->second, "fill", "after_merge");
        pending.erase(it);
        return;
    }
    if (p.type == PREFETCH && it->second.id == p.pf_metadata) {
        it->second.filled = true;
        candidate(it->second, "fill");
    }
}
void hybrid_transfer(unsigned type, bool write) {
    if (!measured())
        return;
    if (write)
        write_bytes += BLOCK_SIZE;
    else if (type == PREFETCH)
        prefetch_bytes += BLOCK_SIZE;
    else
        demand_bytes += BLOCK_SIZE;
}
void hybrid_bank_cycle() {
    if (!measured() || all_simulation_complete == NUM_CPUS)
        return;
    if (!cycle_count) {
        warm_rq = uncore.LLC.RQ.occupancy;
        warm_mshr = uncore.LLC.MSHR.occupancy;
        warm_wq = uncore.LLC.WQ.occupancy;
        warm_dram_rq = uncore.DRAM.RQ[0].occupancy;
        warm_dram_wq = uncore.DRAM.WQ[0].occupancy;
        for (unsigned c = 0; c < active_cores; ++c)
            warm_private_wq += ooo_cpu[c].L1D.WQ.occupancy + ooo_cpu[c].L2C.WQ.occupancy;
    }
    ++cycle_count;
    mshr_sum += uncore.LLC.MSHR.occupancy;
    drq_sum += uncore.DRAM.RQ[0].occupancy;
    dwq_sum += uncore.DRAM.WQ[0].occupancy;
}
void hybrid_finish() {
    for (const auto &p : pending)
        if (!p.second.claimed)
            candidate(p.second, "unresolved", p.second.filled ? "resident" : "inflight");
    for (unsigned c = 0; c < active_cores; ++c) {
        auto &cpu = ooo_cpu[c];
        if (cpu.begin_sim_instr != warm_count || cpu.finish_sim_instr != roi_count)
            throw std::runtime_error("inexact ROI boundary");
        cores.write("%u,%llu,%llu,%llu\n", c, U(cpu.begin_sim_instr), U(cpu.finish_sim_instr),
                    U(cpu.finish_sim_cycle));
        stat("head_unique_cycles_core_" + std::to_string(c), head_union[c]);
        stat("page_slot_core_" + std::to_string(c), page_slots[c]);
        stat("pages_core_" + std::to_string(c), maps[c].size());
        for (const auto &kv : maps[c])
            page_log.write("%u,%u,%llu,%llu\n", c, page_slots[c], U(kv.first), U(kv.second));
        stat("multi_operand_head_cycles_core_" + std::to_string(c), ambiguous_head[c]);
    }
    stat("dram_demand_bytes", demand_bytes);
    stat("dram_prefetch_bytes", prefetch_bytes);
    stat("dram_write_bytes", write_bytes);
    stat("prefetch_accepted", issued);
    stat("plan_anchors", plan.size());
    stat("plan_matched", plan_hits);
    stat("pressure_cycles", cycle_count);
    stat("llc_mshr_cycle_sum", mshr_sum);
    stat("dram_rq_cycle_sum", drq_sum);
    stat("dram_wq_cycle_sum", dwq_sum);
    stat("roi_start_llc_rq", warm_rq);
    stat("roi_start_llc_wq", warm_wq);
    stat("roi_start_llc_mshr", warm_mshr);
    stat("roi_start_dram_rq", warm_dram_rq);
    stat("roi_start_dram_wq", warm_dram_wq);
    stat("roi_start_private_wq", warm_private_wq);
    loads.close();
    candidates.close();
    heads.close();
    cores.close();
    totals.close();
    page_log.close();
    victims.close();
    if (pollution_probe) {
        fills.close();
        lookups.close();
    }
}
