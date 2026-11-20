// Source log: four little-endian 64-bit words (operand key, PC, physical line, cycle).
#include <zlib.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct Event { uint64_t key, pc, line, cycle; };
static_assert(sizeof(Event) == 32);
constexpr int HISTORY = 64;
constexpr std::array<int, 3> FUTURES{1, 8, 32};
constexpr int64_t ABSENT = -1000000000;

uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct Reader {
    gzFile gz;
    std::array<Event, 8192> buffer;
    size_t pos = 0, size = 0;
    explicit Reader(const char *path): gz(gzopen(path, "rb")) {
        if (!gz) throw std::runtime_error("Cannot open arrivals");
        gzbuffer(gz, 1 << 20);
    }
    bool next(Event &e) {
        if (pos == size) {
            int n = gzread(gz, buffer.data(), sizeof(buffer));
            if (n < 0 || n % sizeof(Event)) throw std::runtime_error("Invalid arrival block");
            pos = 0;
            size = n / sizeof(Event);
            if (!size) return false;
        }
        e = buffer[pos++];
        return true;
    }
    ~Reader() { gzclose(gz); }
};

void cell(int64_t x) { std::cout << '\t' << x; }
int64_t delta(uint64_t a, uint64_t b) { return int64_t(a) - int64_t(b); }
void remember(std::deque<Event> &q, const Event &e, size_t capacity) {
    q.push_front(e);
    if (q.size() > capacity) q.pop_back();
}
void token(const Event &now, const Event *old) {
    if (!old) {
        for(int i=0; i<7; ++i) cell(ABSENT);
        return;
    }
    cell(delta(old->line, now.line));
    cell(old->line % 64);
    cell(old->pc == now.pc);
    cell(old->line / 64 == now.line / 64);
    cell(old->line / 4096 == now.line / 4096);
    cell(delta(now.cycle, old->cycle));
    cell(old->pc);
}
void feature_names(const std::string &name) {
    for (const auto field : {"delta", "offset", "samepc", "samepage", "sameregion", "dt", "pc"})
        std::cout << '\t' << name << "_" << field;
}

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    Reader input(argv[1]);
    const uint64_t count = std::stoull(argv[2]), block = std::stoull(argv[3]);
    if (!block || count < 100) throw std::runtime_error("Invalid sample plan");
    std::deque<Event> future, global;
    std::unordered_map<uint64_t,std::deque<Event>> pcs, regions;
    std::unordered_set<uint64_t> lines, pages, visited_regions;
    Event event;
    for(int i=0; i<=FUTURES.back(); ++i) {
        if (!input.next(event)) throw std::runtime_error("Short input");
        future.push_back(event);
    }
    std::cout << "index\tkey\tpc\tline\tcycle";
    for (int lag=1; lag<=HISTORY; ++lag) feature_names("g"+std::to_string(lag));
    for (int lag : {1,2,4,8}) {
        feature_names("p"+std::to_string(lag));
        feature_names("r"+std::to_string(lag));
    }
    for(int r=0;r<8;++r) std::cout << "\tbank" << r;
    for(int h : FUTURES)
        for (const auto name : {"key","pc","line","cycle","seen","seenpage","seenregion"})
            std::cout << "\ty" << h << "_" << name;
    std::cout << '\n';
    uint64_t i = 0, previous_cycle = 0;
    while(!future.empty()) {
        const auto now = future.front();
        if (now.cycle < previous_cycle) throw std::runtime_error("Arrival cycle reversed");
        previous_cycle = now.cycle;
        auto &pc = pcs[now.pc], &region = regions[now.line / 4096];
        lines.insert(now.line);
        pages.insert(now.line / 64);
        visited_regions.insert(now.line / 4096);
        if(future.size() > FUTURES.back() && i % block == mix(i / block) % block) {
            std::cout << i;
            cell(now.key); cell(now.pc); cell(now.line); cell(now.cycle);
            for (int lag=1; lag<=HISTORY; ++lag) token(now, global.size() >= size_t(lag) ? &global[lag-1] : nullptr);
            for (int lag : {1,2,4,8}) {
                token(now, pc.size() >= size_t(lag) ? &pc[lag-1] : nullptr);
                token(now, region.size() >= size_t(lag) ? &region[lag-1] : nullptr);
            }
            std::vector<uint64_t> bank{now.line / 64};
            for (const Event &old : global) {
                if (bank.size() == 8) break;
                uint64_t page = old.line / 64;
                if (std::find(bank.begin(),bank.end(),page)==bank.end()) bank.push_back(page);
            }
            for(int r=0;r<8;++r) cell(size_t(r)<bank.size()?int64_t(bank[r]):-1);
            for (int h : FUTURES) {
                const auto &target=future[h];
                cell(target.key); cell(target.pc); cell(target.line); cell(target.cycle);
                cell(lines.count(target.line)); cell(pages.count(target.line/64));
                cell(visited_regions.count(target.line/4096));
            }
            std::cout << '\n';
        }
        remember(global, now, 64); remember(pc, now, 8); remember(region, now, 8);
        future.pop_front();
        if(input.next(event)) future.push_back(event);
        ++i;
    }
    if(i!=count) throw std::runtime_error("Complete ingress count differs from phase counters");
}
