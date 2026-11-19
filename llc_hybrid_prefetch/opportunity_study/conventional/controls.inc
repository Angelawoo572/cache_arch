// Included in a separate build of the A–C simulator. Prediction uses only already-admitted events.
#include "spp.inc"

struct Transition {
    std::array<uint64_t, 3> key{};
    int64_t value = 0;
    uint8_t confidence = 0;
    bool valid = false;
};

struct TransitionTable {
    std::array<Transition, 256> entries{};
    static size_t index(const std::array<uint64_t, 3> &k) {
        uint64_t z = 0;
        for (auto word : k) {
            z = z * 131 + word;
            z ^= z >> 19;
            z ^= z >> 9;
        }
        return z & 255;
    }
    void observe(const std::array<uint64_t, 3> &key, int64_t value) {
        auto &e = entries[index(key)];
        if (!e.valid || e.key != key) {
            e.key = key;
            e.value = value;
            e.valid = true;
            e.confidence = 1;
        } else if (e.value == value) {
            if (e.confidence < 3)
                ++e.confidence;
        } else if (e.confidence > 1)
            --e.confidence;
        else {
            e.value = value;
            e.confidence = 1;
        }
    }
    const Transition *find(const std::array<uint64_t, 3> &key) const {
        const auto &e = entries[index(key)];
        return e.valid && e.key == key && e.confidence >= 2 ? &e : nullptr;
    }
};

struct PCState {
    uint64_t pc = 0, last = 0;
    int64_t delta = 0, learned_step = 0, learned_jump = 0;
    unsigned run = 0, learned_length = 0;
};

struct StudyLane {
    std::array<PCState, 64> pcs{};
    TransitionTable multi, path, temporal;
    uint64_t prev_pc = 0, prev_line = 0;
    std::array<uint64_t, 3> prev_path{};
    uint64_t issues = 0, epoch = UINT64_MAX, epoch_issues = 0, cap = UINT64_MAX;
    std::unique_ptr<SppPolicy> spp;
};

struct Proposal {
    uint64_t line;
    const char *source;
    bool cross;
    bool agree;
};
struct Queued {
    Candidate item;
    uint64_t ready;
    bool cross;
    bool agree;
    const char *source;
};

std::array<StudyLane, NUM_CPUS> lanes;
std::deque<Queued> decision_queue;
std::string variant = "stride";
uint64_t epoch_limit = 0, latency = 0;
uint64_t next_port = 0;
bool study_on = false, track_choices = false;
Log choices;
unsigned rate_epochs = 512;
bool fast_native = false;
const char *env_or(const char *key, const char *def) {
    auto s = std::getenv(key);
    return s ? s : def;
}

void study_init() {
    variant = env_or("STUDY_POLICY", "stride");
    static const std::set<std::string> names = {
        "stride",   "next", "look2", "look4",   "multi",    "boundary", "xpc",        "temporal_sp",
        "temporal", "spp",  "union", "confirm", "priority", "pressure", "confidence", "promote"};
    if (!names.count(variant))
        throw std::runtime_error("unknown STUDY_POLICY: " + variant);
    epoch_limit = std::stoull(env_or("STUDY_EPOCH_LIMIT", "0"));
    fast_native = std::string(env_or("STUDY_FAST_NATIVE", "0")) == "1";
    if (fast_native && variant != "union")
        throw std::runtime_error("fast native test requires unconditional union");
    latency = std::stoull(env_or("STUDY_LATENCY", "0"));
    if (auto caps = std::getenv("STUDY_CAPS")) {
        std::istringstream in(caps);
        std::string s;
        for (unsigned c = 0; c < active_cores; ++c) {
            if (!std::getline(in, s, ','))
                throw std::runtime_error("one total issue cap required per active core");
            lanes[c].cap = std::stoull(s);
        }
    }
    track_choices = std::string(env_or("STUDY_CHOICES", "0")) == "1";
    if (track_choices)
        choices.open(
            "choices",
            "cpu,instr_id,operand,pc,line,cycle,stride,xpc,temporal,chosen,source,agree,mshr,dram_rq", true);
    if (variant == "spp")
        for (unsigned c = 0; c < active_cores; ++c)
            lanes[c].spp.reset(new SppPolicy());
    study_on = true;
}

std::array<uint64_t, 3> token(uint64_t a, uint64_t b = 0, uint64_t c = 0) { return {{a, b, c}}; }
bool pressured(CACHE &c) { return c.MSHR.occupancy >= 16 || uncore.DRAM.RQ[0].occupancy >= 16; }
uint64_t add_delta(uint64_t base, int64_t d) {
    return d && int64_t(base) + d > 0 ? uint64_t(int64_t(base) + d) : 0;
}

// Only a ready entry may pass other ready work. MSHR and resident confirmations keep the original
// transaction.
void confirm_queued(CACHE &c, uint64_t line) {
    if (!line)
        return;
    auto live = pending.find(line);
    if (live == pending.end() || !live->second.study_stride_source)
        return;
    ++diagnostics["confirm_existing_stride"];
    PACKET p;
    p.address = line;
    p.cpu = live->second.core;
    p.full_addr = line << LOG2_BLOCK_SIZE;
    int ix = c.PQ.check_queue(&p);
    if (ix < 0) {
        if (c.check_mshr(&p) != -1)
            ++diagnostics["confirm_already_mshr"];
        else if (c.get_way(line, c.get_set(line)) != c.NUM_WAY)
            ++diagnostics["confirm_already_resident"];
        return;
    }
    auto &q = c.PQ;
    if (unsigned(ix) == q.head) {
        ++diagnostics["confirm_already_head"];
        return;
    }
    if (q.entry[ix].event_cycle > current_core_cycle[p.cpu]) {
        ++diagnostics["confirm_not_ready"];
        return;
    }
    ++diagnostics["promoted_ready_entry"];
    PACKET saved = q.entry[ix];
    for (unsigned j = ix; j != q.head;) {
        unsigned prior = (j + q.SIZE - 1) % q.SIZE;
        q.entry[j] = q.entry[prior];
        j = prior;
    }
    q.entry[q.head] = saved;
}

Proposal study_predict(CACHE &c, const PACKET &p, bool log) {
    auto &l = lanes[p.cpu];
    auto &state = l.pcs[((p.ip >> 2) ^ (p.ip >> 9)) & 63];
    if (state.pc != p.ip) {
        state = {};
        state.pc = p.ip;
    }
    const auto line = p.address;
    const uint64_t stride = !std::strcmp(stride_reason, "match") ? add_delta(line, stride_delta) : 0;
    int64_t d = state.last ? int64_t(line) - int64_t(state.last) : 0;
    auto path_key = token(l.prev_pc, p.ip, line % 64);
    if (l.prev_line) {
        l.temporal.observe(token(l.prev_line), line);
        l.path.observe(l.prev_path, int64_t(line) - int64_t(l.prev_line));
    }
    if (state.last && d) {
        if (state.run)
            l.multi.observe(token(p.ip, uint64_t(state.delta)), d);
        if (d == state.delta) {
            state.run = std::min(state.run + 1, 255u);
        } else {
            if (state.run >= 3) {
                state.learned_length = state.run;
                state.learned_step = state.delta;
                state.learned_jump = d;
            }
            state.delta = d;
            state.run = 1;
        }
    }
    auto mt = l.multi.find(token(p.ip, uint64_t(d)));
    auto xp = l.path.find(path_key);
    auto tp = l.temporal.find(token(line));
    const auto mult = mt ? add_delta(line, mt->value) : 0;
    const auto cross = xp ? add_delta(line, xp->value) : 0;
    const uint64_t temporal = tp ? tp->value : 0;
    uint64_t boundary = 0;
    if (state.learned_length && state.delta == state.learned_step && state.run <= state.learned_length &&
        state.learned_length - state.run <= 2) {
        boundary =
            add_delta(line, (state.learned_length - state.run) * state.learned_step + state.learned_jump);
    }
    bool same_xpc = cross && cross / 64 == line / 64;
    bool agree = stride && (stride == cross || stride == temporal);
    Proposal result{stride, "stride", false, agree};
    if (variant == "next")
        result = {line + 1, "next", false, false};
    else if (variant == "look2" || variant == "look4")
        result = {stride ? add_delta(line, (variant == "look2" ? 2 : 4) * stride_delta) : 0, "lookahead",
                  false, false};
    else if (variant == "multi" && mult)
        result = {mult, "previous_delta", false, false};
    else if (variant == "boundary" && boundary)
        result = {boundary, "repeat_length", false, false};
    else if (variant == "xpc")
        result = {cross, "pc_context", false, false};
    else if (variant == "temporal" || variant == "temporal_sp")
        result = {temporal, "seen_successor", variant == "temporal", false};
    else if (variant == "spp") {
        auto a = l.spp->access(line);
        result = {a.empty() ? 0 : a.front(), "spp", false, false};
    } else if (variant == "union" || variant == "priority" || variant == "confirm" || variant == "promote") {
        if (log && variant == "promote") {
            confirm_queued(c, cross);
            if (temporal != cross)
                confirm_queued(c, temporal);
        }
        if (!stride)
            result = same_xpc ? Proposal{cross, "pc_context", false, false}
                              : Proposal{temporal, "seen_successor", true, false};
        if ((variant == "confirm" || (variant == "priority" && pressured(c))) && !agree)
            result = {0, "uncorroborated", false, false};
    } else if (variant == "pressure" && pressured(c))
        result = {0, "pressure_gate", false, false};
    else if (variant == "confidence" && state.run < 4)
        result = {0, "confidence_gate", false, false};
    if (log) {
        ++diagnostics["anchors"];
        if (pressured(c))
            ++diagnostics["pressure_anchors"];
        if (stride)
            ++diagnostics["stride_raw"];
        if (cross)
            ++diagnostics["xpc_raw"];
        if (temporal)
            ++diagnostics["temporal_raw"];
        if (agree)
            ++diagnostics["agreed_anchors"];
        if (stride && (cross || temporal) && !agree)
            ++diagnostics["disagreed_anchors"];
        if (result.line)
            ++diagnostics["chosen_" + std::string(result.source)];
        else
            ++diagnostics["silent_" + std::string(result.source)];
        if (track_choices)
            choices.write("%u,%llu,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%s,%u,%u,%u\n", p.cpu,
                          U(ordinal(p.cpu, p.instr_id)), p.data_index, U(p.ip), U(line),
                          U(current_core_cycle[p.cpu]), U(stride), U(cross), U(temporal), U(result.line),
                          result.source, agree, c.MSHR.occupancy, uncore.DRAM.RQ[0].occupancy);
    }
    if (d || !state.last)
        state.last = line;
    l.prev_pc = p.ip;
    l.prev_line = line;
    l.prev_path = path_key;
    return result;
}

void study_issue(CACHE &c, Queued q) {
    auto &item = q.item;
    auto &lane = lanes[item.core];
    uint64_t epoch = (ooo_cpu[item.core].num_retired - warm_count) / rate_epochs;
    if (epoch != lane.epoch) {
        lane.epoch = epoch;
        lane.epoch_issues = 0;
    }
    PACKET probe;
    probe.cpu = item.core;
    probe.address = item.target;
    probe.full_addr = item.target << LOG2_BLOCK_SIZE;
    const char *reject = nullptr;
    if (!item.target || item.base == item.target)
        reject = "self";
    else if (!q.cross && item.base / 64 != item.target / 64)
        reject = "page";
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
    else if (lane.issues >= lane.cap || (epoch_limit && lane.epoch_issues >= epoch_limit))
        reject = "budget";
    candidate(item, "ready", q.source);
    if (reject) {
        ++diagnostics["rejected_" + std::string(reject)];
        candidate(item, "rejected", reject);
        return;
    }
    auto old = c.cpu;
    c.cpu = item.core;
    auto prior = c.PQ.occupancy;
    c.prefetch_line(item.pc, item.base << LOG2_BLOCK_SIZE, probe.full_addr, FILL_LLC, item.id);
    c.cpu = old;
    if (prior == c.PQ.occupancy) {
        ++diagnostics["rejected_enqueue"];
        candidate(item, "rejected", "enqueue");
        return;
    }
    ++lane.issues;
    ++lane.epoch_issues;
    ++issued;
    if (lane.spp)
        lane.spp->issued(item.target);
    if (q.agree)
        ++diagnostics["accepted_agreement"];
    if (item.target / 64 != item.base / 64)
        ++diagnostics["accepted_crosspage"];
    candidate(item, "accepted", q.source);
    pending.emplace(item.target, item);
}

void study_submit(CACHE &c, const PACKET &p, Proposal q) {
    Candidate item{};
    item.id = ++next_id;
    item.base = p.address;
    item.instr = ordinal(p.cpu, p.instr_id);
    item.pc = p.ip;
    item.core = p.cpu;
    item.operand = p.data_index;
    auto action = plan.find(std::make_tuple(p.cpu, item.instr, item.operand));
    if (action != plan.end()) {
        ++plan_hits;
        q = {add_delta(item.base, action->second), "intervention", false, false};
    }
    if (!q.line) {
        candidate(item, "silent",
                  strcmp(stride_reason, "match") && variant == "stride" ? stride_reason : q.source);
        return;
    }
    item.target = q.line;
    item.study_stride_source = !std::strcmp(q.source, "stride");
    const auto &pcstate = lanes[p.cpu].pcs[((p.ip >> 2) ^ (p.ip >> 9)) & 63];
    item.study_group = (q.agree ? 4u : 0u) + (pressured(c) ? 2u : 0u) + (pcstate.run >= 4 ? 1u : 0u);
    candidate(item, "proposed", q.source);
    auto now = current_core_cycle[p.cpu];
    if (!latency || (fast_native && item.study_stride_source)) {
        study_issue(c, {item, now, q.cross, q.agree, q.source});
        return;
    }
    if (decision_queue.size() >= 16) {
        ++diagnostics["rejected_engine_full"];
        candidate(item, "rejected", "engine_full");
        return;
    }
    uint64_t start = std::max(now, next_port);
    next_port = start + 1;
    decision_queue.push_back({item, start + latency, q.cross, q.agree, q.source});
}

void study_tick() {
    if (!study_on || !measured())
        return;
    uint64_t now = current_core_cycle[0];
    if (!decision_queue.empty() && decision_queue.front().ready <= now) {
        auto q = decision_queue.front();
        decision_queue.pop_front();
        if (simulation_complete[q.item.core]) {
            ++diagnostics["rejected_finished_core"];
            candidate(q.item, "rejected", "finished_core");
        } else
            study_issue(uncore.LLC, q);
    }
}

void directed_duplicates() {
    static const bool enabled = std::string(env_or("STUDY_DUAL_ISSUER_TEST", "0")) == "1";
    static unsigned phase = 0;
    if (!enabled || !measured() || active_cores != 2 || phase == 4)
        return;
    auto &cache = uncore.LLC;
    const uint64_t target = uint64_t(DRAM_PAGES - 1) * (PAGE_SIZE / BLOCK_SIZE) + 19;
    PACKET probe;
    probe.address = target;
    probe.cpu = 1;
    probe.full_addr = target << LOG2_BLOCK_SIZE;
    auto submit = [&](unsigned cpu, const char *key) {
        uint64_t accepted = issued;
        Candidate item{};
        item.id = ++next_id;
        item.base = target - 1;
        item.target = target;
        item.instr = 1;
        item.pc = 4;
        item.core = cpu;
        auto before = diagnostics["rejected_" + std::string(key)];
        study_issue(cache, {item, current_core_cycle[cpu], false, true, "directed_fixture"});
        bool ok = (!std::strcmp(key, "accepted"))
                      ? issued == accepted + 1
                      : (issued == accepted && diagnostics["rejected_" + std::string(key)] == before + 1);
        if (!ok)
            throw std::runtime_error("unexpected duplicate-filter stage " + std::string(key));
        diagnostics["fixture_" + std::string(key) + "_cycle"] = current_core_cycle[cpu];
    };
    if (phase == 0) {
        submit(0, "accepted");
        submit(1, "pq");
        phase = 2;
    } else if (phase == 2 && cache.check_mshr(&probe) != -1) {
        submit(1, "mshr");
        phase = 3;
    } else if (phase == 3 && cache.get_way(target, cache.get_set(target)) != cache.NUM_WAY) {
        submit(1, "resident");
        phase = 4;
    }
}
