// Passive features. The predictor never reads these arrays or sampled-interval membership.
constexpr unsigned phase_count = 4;
constexpr unsigned ring_length = 4096; // much greater than maximum out-of-order instruction distance
uint64_t sample_width = 250000;
struct RegionCounts {
    uint64_t l2_hit = 0, l2_miss = 0, llc_hit = 0, llc_miss = 0;
    uint64_t ingress = 0, timely = 0, late = 0, loads = 0, end_cycle = 0;
};
RegionCounts regions[NUM_CPUS][phase_count];
struct OutcomeStream {
    gzFile file = nullptr;
    uint16_t ring[ring_length] = {};
    unsigned char output[32768];
    unsigned used = 0;
    uint64_t written = 0;
};
OutcomeStream outcomes[NUM_CPUS];
unsigned phase_of(uint64_t id) {
    return static_cast<unsigned>(((id - warm_count) * phase_count) / roi_count);
}
uint64_t phase_start(unsigned phase) { return (uint64_t(phase) * roi_count + phase_count - 1) / phase_count; }
bool sampled(uint64_t id) {
    if (id < warm_count || id - warm_count >= roi_count)
        return false;
    const uint64_t pos = id - warm_count;
    return pos - phase_start(phase_of(id)) < sample_width;
}
void record_outcome(const PACKET &p, unsigned mask) {
    auto &s = outcomes[p.cpu];
    const auto offset = p.instr_id - warm_count;
    if (offset < s.written || offset - s.written >= ring_length || p.data_index >= NUM_INSTR_SOURCES)
        throw std::runtime_error("LLC outcome outside active retirement window");
    auto &word = s.ring[offset % ring_length];
    const auto shift = 4 * p.data_index;
    if ((word >> shift) & mask)
        throw std::runtime_error("multiple identical LLC tag/first-use outcomes for dynamic operand");
    word |= uint16_t(mask << shift);
}
void flush_outcomes(OutcomeStream &s) {
    if (s.used && gzwrite(s.file, s.output, s.used) != static_cast<int>(s.used))
        throw std::runtime_error("cannot write LLC outcomes");
    s.used = 0;
}
void retire_outcomes(unsigned c, uint64_t id) {
    auto &s = outcomes[c];
    const auto pos = id - warm_count;
    if (pos != s.written)
        throw std::runtime_error("unexpected retirement order");
    auto &word = s.ring[pos % ring_length];
    s.output[s.used++] = static_cast<unsigned char>(word);
    s.output[s.used++] = static_cast<unsigned char>(word >> 8);
    word = 0;
    ++s.written;
    if (s.used == sizeof(s.output))
        flush_outcomes(s);
    unsigned phase = phase_of(id);
    if (s.written == phase_start(phase + 1))
        regions[c][phase].end_cycle = current_core_cycle[c] - ooo_cpu[c].begin_sim_cycle;
}
void start_observations() {
    if (const char *width = std::getenv("FULL_SAMPLE_WIDTH"))
        sample_width = std::stoull(width);
    for (unsigned c = 0; c < active_cores; ++c) {
        const auto path = directory + "/outcomes_core" + std::to_string(c) + ".u16le.gz";
        auto &s = outcomes[c];
        s.file = gzopen(path.c_str(), "wb1");
        if (!s.file || gzbuffer(s.file, 131072))
            throw std::runtime_error("cannot create LLC outcome stream");
    }
}
void finish_observations() {
    Log phase_log;
    phase_log.open("phases",
                   "cpu,quarter,first_ordinal,last_ordinal,cycles,trace_loads,l2_tag_hits,l2_tag_misses,llc_"
                   "ingress,llc_tag_hits,llc_tag_misses,timely,late",
                   false);
    for (unsigned c = 0; c < active_cores; ++c) {
        auto &s = outcomes[c];
        flush_outcomes(s);
        if (s.written != roi_count || gzclose(s.file) != Z_OK)
            throw std::runtime_error("incomplete LLC outcomes");
        uint64_t prev = 0;
        for (unsigned q = 0; q < phase_count; ++q) {
            const auto &r = regions[c][q];
            phase_log.write("%u,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n", c, q,
                            U(full_source_origin + skip[c] + warm_count + phase_start(q) + 1),
                            U(full_source_origin + skip[c] + warm_count + phase_start(q + 1)),
                            U(r.end_cycle - prev), U(r.loads), U(r.l2_hit), U(r.l2_miss), U(r.ingress),
                            U(r.llc_hit), U(r.llc_miss), U(r.timely), U(r.late));
            prev = r.end_cycle;
        }
    }
    phase_log.close();
}
