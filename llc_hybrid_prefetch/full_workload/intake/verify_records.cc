// Check every classic ChampSim input record and optionally materialize a record range.
// Expected byte offsets are shared with workload_expansion/intake/check_reader.cc.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstddef>
#include <string>
#include <iostream>
#include <limits>
#include <memory>
#include <unordered_set>

struct Record {
    uint64_t pc;
    uint8_t branch, taken, dst_registers[2], src_registers[4];
    uint64_t stores[2], loads[4];
};

static_assert(sizeof(Record) == 64 && offsetof(Record, branch) == 8 &&
              offsetof(Record, taken) == 9 && offsetof(Record, dst_registers) == 10 &&
              offsetof(Record, src_registers) == 12 && offsetof(Record, stores) == 16 &&
              offsetof(Record, loads) == 32);

int main(int argc, char **argv) {
    const uint16_t endian_probe = 1;
    if ((argc != 4 && argc != 5) || *reinterpret_cast<const uint8_t*>(&endian_probe) != 1) return 2;
    const auto begin = std::stoull(argv[2]);
    const auto limit = std::stoull(argv[3]);
    if (!limit || begin > std::numeric_limits<uint64_t>::max() - limit) return 2;
    auto out = std::fopen(argv[1], "wb");
    if (!out) return 3;
    auto seed = argc == 5 ? std::fopen(argv[4], "wb") : nullptr;
    if (argc == 5 && !seed) return 3;
    std::unordered_set<uint64_t> pages;
    std::array<uint64_t, 4096> recent{};
    auto note_page = [&](uint64_t addr) {
        if (!addr) return true;
        const auto vpn = addr >> 12;
        auto &hint = recent[vpn % recent.size()];
        if (hint == vpn + 1) return true;
        hint = vpn + 1;
        return !pages.insert(vpn).second || std::fwrite(&vpn, sizeof(vpn), 1, seed) == 1;
    };
    constexpr size_t capacity = 32768;
    auto records = std::make_unique<Record[]>(capacity);
    uint64_t count = 0, emitted = 0, branches = 0, taken = 0, load_instructions = 0;
    std::array<uint64_t, 4> loads{}, source_registers{};
    std::array<uint64_t, 2> stores{}, dest_registers{};
    while (auto bytes = std::fread(records.get(), 1, capacity * sizeof(Record), stdin)) {
        if (bytes % sizeof(Record)) {
            std::cerr << "Partial final record after " << count << " instructions\n";
            return 4;
        }
        const uint64_t n = bytes / sizeof(Record);
        for (size_t i = 0; i < n; ++i) {
            const auto &r = records[i];
            if (!r.pc || r.branch > 1 || r.taken > r.branch) {
                std::cerr << "Invalid PC/branch flags at instruction " << (count + i) << "\n";
                return 5;
            }
            if (seed && count + i < begin) {
                if (!note_page(r.pc)) return 6;
                for (auto addr : r.stores) if (!note_page(addr)) return 6;
                for (auto addr : r.loads) if (!note_page(addr)) return 6;
            }
            branches += r.branch;
            taken += r.taken;
            bool has_load = false;
            for (size_t slot = 0; slot < loads.size(); ++slot) {
                loads[slot] += r.loads[slot] != 0;
                has_load |= r.loads[slot] != 0;
                source_registers[slot] += r.src_registers[slot] != 0;
            }
            for (size_t slot = 0; slot < stores.size(); ++slot) {
                stores[slot] += r.stores[slot] != 0;
                dest_registers[slot] += r.dst_registers[slot] != 0;
            }
            load_instructions += has_load;
        }
        const uint64_t low = std::max<uint64_t>(count, begin);
        const uint64_t high = std::min<uint64_t>(count + n, begin + limit);
        if (low < high) {
            const size_t chunk = high - low;
            if (std::fwrite(records.get() + (low - count), sizeof(Record), chunk, out) != chunk) return 6;
            emitted += chunk;
        }
        count += n;
    }
    if (seed && std::fclose(seed)) return 6;
    if (std::ferror(stdin) || std::fclose(out) || !count || !emitted) return 7;
    std::cout << count << "," << emitted << "," << branches << "," << taken << "," << load_instructions;
    for (auto x : loads) std::cout << "," << x;
    for (auto x : stores) std::cout << "," << x;
    for (auto x : source_registers) std::cout << "," << x;
    for (auto x : dest_registers) std::cout << "," << x;
    std::cout << "\n";
}
