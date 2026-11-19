// Describe original LLC tag misses in retirement order throughout the measured interval.
// All policy arms read the same trace record and operand. No label drives speculation.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <zlib.h>

using U = uint64_t;
struct Record {
    U pc;
    uint8_t metadata[8];
    U stores[2], loads[4];
};
static_assert(sizeof(Record) == 64 && offsetof(Record, loads) == 32, "classic ChampSim layout");

struct Stream {
    U visits = 0, direct = 0, last_line = 0, last_pc = 0, index = 0, streak = 0;
};
struct Bin {
    U count = 0, switched_pc = 0, crossed_page = 0, interleaved = 0;
};
struct View {
    std::unordered_map<U, Stream> groups;
    Bin counts[3][5]{};
    void visit(U key, U pc, U line, U index, int result) {
        auto &g = groups[key];
        const int previous = !g.visits ? 0 : g.streak ? 1 : 2;
        auto &bin = counts[previous][result];
        ++bin.count;
        if (g.visits) {
            bin.switched_pc += g.last_pc != pc;
            bin.crossed_page += (g.last_line >> 6) != (line >> 6);
            bin.interleaved += index > g.index + 1;
        }
        ++g.visits;
        g.direct += result == 0;
        g.streak = result == 0 ? g.streak + 1 : 0;
        g.last_line = line;
        g.last_pc = pc;
        g.index = index;
    }
};

std::unordered_map<U,U> read_pages(const std::string &folder) {
    auto file = gzopen((folder + "/pages.csv.gz").c_str(), "rb");
    if (!file) throw std::runtime_error("missing page map");
    std::unordered_map<U,U> pages;
    char row[160];
    if (!gzgets(file, row, sizeof(row))) throw std::runtime_error("empty page map");
    while (gzgets(file, row, sizeof(row))) {
        unsigned core, slot;
        unsigned long long vpn, ppn;
        if (std::sscanf(row, "%u,%u,%llu,%llu", &core,&slot,&vpn,&ppn) != 4 || core || slot ||
            !pages.emplace(vpn,ppn).second) throw std::runtime_error("invalid solo mapping");
    }
    if (gzclose(file) != Z_OK) throw std::runtime_error("page map read failed");
    return pages;
}

int category(unsigned f) {
    if (f & 8) return 0;
    if (f & 4) return 1;
    if (f & 2) return 2;
    if (f & 1) return 3;
    return 4;
}

int main(int argc, char **argv) {
    if (argc != 9) return 2;
    const U offset = std::stoull(argv[2]), length = std::stoull(argv[3]);
    std::ifstream trace(argv[1], std::ios::binary);
    trace.seekg(offset * sizeof(Record));
    constexpr unsigned chunk = 32768;
    auto records = std::make_unique<Record[]>(chunk);
    std::array<std::vector<uint16_t>,4> words;
    std::array<gzFile,4> files;
    const std::array<const char*,3> policies = {"stride","multi","spp"};
    const auto pages = read_pages(argv[4]);
    for (unsigned a=0; a<4; ++a) {
        if (a && pages != read_pages(argv[4+a])) throw std::runtime_error("different physical page map");
        files[a] = gzopen((std::string(argv[4+a]) + "/outcomes_core0.u16le.gz").c_str(), "rb");
        if (!files[a] || gzbuffer(files[a], 131072)) throw std::runtime_error("missing outcomes");
        words[a].resize(chunk);
    }
    const uint16_t little = 1;
    if (*reinterpret_cast<const char*>(&little) != 1) throw std::runtime_error("little endian reader required");
    View views[3][3];
    const std::array<const char*,3> names = {"all","pc","physical_256KiB_region"};
    U index = 0;
    for (U n=0; n<length; n+=chunk) {
        unsigned k = std::min<U>(chunk, length - n);
        if (!trace.read(reinterpret_cast<char*>(records.get()), k * sizeof(Record)))
            throw std::runtime_error("short trace");
        for (unsigned a=0; a<4; ++a)
            if (gzread(files[a], words[a].data(), k * 2) != int(k * 2)) throw std::runtime_error("short outcomes");
        for (unsigned i=0; i<k; ++i) {
            if (!words[0][i]) continue;
            const auto &r = records[i];
            for (unsigned j=0; j<4; ++j) {
                unsigned shift = 4*j;
                if (!((words[0][i] >> shift) & 2)) continue;
                if (!r.loads[j]) throw std::runtime_error("miss attached to empty operand");
                U line = pages.at(r.loads[j] >> 12) * 64 + ((r.loads[j] >> 6) & 63);
                ++index;
                for (unsigned a=0; a<3; ++a) {
                    auto f = (words[a+1][i] >> shift) & 15;
                    int result = category(f);
                    if ((f & 8) && (!(f & 1) || (f & 4))) throw std::runtime_error("bad direct first-use flags");
                    views[a][0].visit(0,r.pc,line,index,result);
                    views[a][1].visit(r.pc,r.pc,line,index,result);
                    views[a][2].visit(line >> 12,r.pc,line,index,result);
                }
            }
        }
    }
    for (auto file : files) if (gzgetc(file) != -1 || !gzeof(file) || gzclose(file) != Z_OK)
        throw std::runtime_error("unexpected outcome trailer");
    const std::array<const char*,3> history = {"first","after_timely","after_nondirect"};
    const std::array<const char*,5> outcomes = {"timely_first_use","late_first_use","other_tag_miss","ordinary_tag_hit","no_llc_tag"};
    for (unsigned a=0; a<3; ++a) for (unsigned v=0; v<3; ++v) {
        auto &view = views[a][v];
        U total = 0;
        for (unsigned h=0; h<3; ++h) for (unsigned o=0; o<5; ++o) {
            auto &b = view.counts[h][o];
            total += b.count;
            std::cout << "event," << argv[8] << ',' << policies[a] << ',' << names[v] << ','
                      << history[h] << ',' << outcomes[o] << ',' << b.count << ',' << b.switched_pc
                      << ',' << b.crossed_page << ',' << b.interleaved << '\n';
        }
        if (total != index) throw std::runtime_error("episode partition incomplete");
        for (unsigned min : {1,32}) {
            U groups=0, visits=0, never=0, missed=0;
            for (const auto &kv : view.groups) {
                const auto &g = kv.second;
                if (g.visits < min) continue;
                ++groups;
                visits += g.visits;
                if (!g.direct) { ++never; missed += g.visits; }
            }
            std::cout << "group," << argv[8] << ',' << policies[a] << ',' << names[v] << ','
                      << min << ',' << groups << ',' << never << ',' << visits << ',' << missed << '\n';
        }
    }
}
