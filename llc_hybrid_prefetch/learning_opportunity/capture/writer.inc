// One independent gzip member per fixed-size block. Only observers write these files.
class CaptureStream {
    static constexpr unsigned block_rows = 32768;
    std::vector<uint64_t> words;
    std::vector<unsigned char> compressed;
    FILE *out = nullptr;
    std::ofstream index;
    unsigned width = 0;
    uint64_t count = 0;

    void flush() {
        if (words.empty())
            return;
        const auto rows = words.size() / width;
        const auto bytes = words.size() * sizeof(uint64_t);
        z_stream z{};
        if (deflateInit2(&z, 1, Z_DEFLATED, 31, 8, Z_DEFAULT_STRATEGY) != Z_OK)
            throw std::runtime_error("cannot initialize capture compression");
        compressed.resize(deflateBound(&z, bytes));
        z.next_in = reinterpret_cast<Bytef *>(words.data());
        z.avail_in = bytes;
        z.next_out = compressed.data();
        z.avail_out = compressed.size();
        const int rc = deflate(&z, Z_FINISH);
        const auto written = z.total_out;
        if (deflateEnd(&z) != Z_OK || rc != Z_STREAM_END)
            throw std::runtime_error("cannot finish capture member");
        const auto offset = ftello(out);
        if (offset < 0 || fwrite(compressed.data(), 1, written, out) != written)
            throw std::runtime_error("cannot write capture member");
        index << count << ',' << rows << ',' << offset << ',' << written << '\n';
        count += rows;
        words.clear();
    }

public:
    void open(const char *name, unsigned columns) {
        uint16_t endian = 1;
        if (*reinterpret_cast<unsigned char *>(&endian) != 1 || active_cores != 1)
            throw std::runtime_error("capture supports little-endian single-active-core runs");
        const auto base = directory + "/" + name;
        width = columns;
        words.reserve(block_rows * width);
        out = fopen((base + ".u64le.gz").c_str(), "wb");
        index.open(base + ".index.csv");
        if (!out || !index)
            throw std::runtime_error("cannot open capture stream");
        index << "first_row,rows,byte_offset,compressed_bytes\n";
    }
    void row(std::initializer_list<uint64_t> fields) {
        if (fields.size() != width)
            throw std::runtime_error("invalid capture row width");
        words.insert(words.end(), fields.begin(), fields.end());
        if (words.size() == block_rows * width)
            flush();
    }
    void close() {
        flush();
        if (!count) { // even an empty logical stream must be valid gzip
            const unsigned char empty[] = {31,139,8,0,0,0,0,0,0,3,3,0,0,0,0,0,0,0,0,0};
            if (fwrite(empty, 1, sizeof(empty), out) != sizeof(empty))
                throw std::runtime_error("cannot write empty capture stream");
        }
        if (fclose(out)) throw std::runtime_error("cannot close capture");
        index.close();
        if (!index) throw std::runtime_error("cannot close capture index");
    }
};

CaptureStream arrivals, ingress_state, cache_events, core_events, offer_events, wait_events, eviction_events;
uint64_t capture_order = 0;
std::map<std::string,uint64_t> vocabulary;

uint64_t symbol(const char *value) {
    auto it = vocabulary.find(value);
    if (it != vocabulary.end()) return it->second;
    const auto id = vocabulary.size() + 1;
    vocabulary.emplace(value, id);
    return id;
}

uint64_t operand_key(unsigned cpu, uint64_t instr, unsigned op) {
    return (ordinal(cpu, instr) << 2) | op;
}

void start_capture() {
    arrivals.open("arrival", 4);
    ingress_state.open("ingress_state", 2);
    cache_events.open("cache", 5);
    core_events.open("source_load", 2);
    offer_events.open("offer", 7);
    wait_events.open("head_wait", 3);
    eviction_events.open("prefetch_victim", 4);
}

void capture_cache(CACHE &c, const PACKET &p, const char *stage, int hit) {
    const auto key = operand_key(p.cpu, p.instr_id, p.data_index);
    const auto now = current_core_cycle[p.cpu];
    if (c.cache_type == IS_LLC && !std::strcmp(stage, "ingress")) {
        const auto pressure = uint64_t(c.MSHR.occupancy) | (uint64_t(c.RQ.occupancy) << 8) |
            (uint64_t(c.PQ.occupancy) << 16) | (uint64_t(uncore.DRAM.RQ[0].occupancy) << 24) |
            (uint64_t(uncore.DRAM.WQ[0].occupancy) << 32) | (uint64_t(c.WQ.occupancy) << 40);
        arrivals.row({key, p.ip, p.address, now});
        ingress_state.row({capture_order++, pressure});
        return;
    }
    unsigned level = c.cache_type == IS_L1D ? 1 : c.cache_type == IS_L2C ? 2 : 3;
    // The static tag IDs in the file dictionary also distinguish pre-lookup queue events.
    auto code = symbol(stage) << 16 | (unsigned(hit + 1) << 8) | level;
    cache_events.row({capture_order++, key, now, p.address, code});
}

void capture_offer(const Candidate &c, const char *stage, const char *reason, const PACKET *d) {
    const bool terminal_load = d && demand(*d);
    const auto consumer = terminal_load ? operand_key(d->cpu, d->instr_id, d->data_index) : 0;
    offer_events.row({capture_order++, (c.instr << 2) | c.operand, c.id, c.target,
                      current_core_cycle[c.core], (symbol(stage) << 32) | symbol(reason), consumer});
}

void finish_capture() {
    arrivals.close();
    ingress_state.close();
    cache_events.close();
    core_events.close();
    offer_events.close();
    wait_events.close();
    eviction_events.close();
    std::ofstream out(directory + "/symbols.csv");
    out << "id,value\n";
    for (const auto &item : vocabulary) out << item.second << ',' << item.first << '\n';
    out.close();
    if (!out) throw std::runtime_error("cannot close capture dictionary");
}
