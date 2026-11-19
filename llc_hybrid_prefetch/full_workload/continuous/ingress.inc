// Export only admitted demand observations. The simulator never reads this stream.
gzFile arrival_file = nullptr;
unsigned char arrival_buffer[131072];
unsigned arrival_bytes = 0;
uint64_t arrival_count = 0;

void arrival_flush() {
    if (arrival_bytes &&
        gzwrite(arrival_file, arrival_buffer, arrival_bytes) != static_cast<int>(arrival_bytes))
        throw std::runtime_error("arrival stream write failure");
    arrival_bytes = 0;
}
void arrival_start() {
    if (active_cores != 1)
        throw std::runtime_error("registered continuous audit uses one active core");
    const auto path = directory + "/arrival.u64le.gz";
    arrival_file = gzopen(path.c_str(), "wb1");
    if (!arrival_file || gzbuffer(arrival_file, 131072))
        throw std::runtime_error("cannot open arrival stream");
}
void arrival_record(const PACKET &p) {
    const std::array<uint64_t, 4> fields = {
        {(ordinal(p.cpu, p.instr_id) << 2) | p.data_index, p.ip, p.address, current_core_cycle[p.cpu]}};
    for (auto word : fields)
        for (unsigned byte = 0; byte < 8; ++byte)
            arrival_buffer[arrival_bytes++] = static_cast<unsigned char>(word >> (byte * 8));
    if (arrival_bytes == sizeof(arrival_buffer))
        arrival_flush();
    ++arrival_count;
}
void arrival_finish() {
    arrival_flush();
    if (gzclose(arrival_file) != Z_OK)
        throw std::runtime_error("arrival stream close failure");
    uint64_t expected = 0;
    for (const auto &r : regions[0])
        expected += r.ingress;
    if (arrival_count != expected)
        throw std::runtime_error("arrival stream omits admitted events");
}
