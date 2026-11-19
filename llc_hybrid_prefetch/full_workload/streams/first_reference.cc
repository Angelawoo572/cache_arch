// First data demand in the original native trace for each queried virtual line.
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using U = uint64_t;
struct Record { U pc, metadata, stores[2], loads[4]; };
static_assert(sizeof(Record)==64 && offsetof(Record, loads)==32, "native input layout");
int main(int argc,char **argv) {
    if (argc!=3) return 2;
    std::ifstream f(argv[1],std::ios::binary);
    const U last=std::stoull(argv[2]);
    std::unordered_map<U,U> first;
    U query;
    while (std::cin>>query) first[query]=0;
    std::vector<Record> block(32768);
    for (U base=0; base<last; base+=block.size()) {
        U count=std::min<U>(block.size(),last-base);
        if (!f.read(reinterpret_cast<char*>(block.data()),count*sizeof(Record))) return 3;
        for (U i=0; i<count; ++i) {
            auto inspect = [&](U addr) {
                if (!addr) return;
                auto entry=first.find(addr>>6);
                if (entry!=first.end() && !entry->second) entry->second=base+i+1;
            };
            for (U addr:block[i].loads) inspect(addr);
            for (U addr:block[i].stores) inspect(addr);
        }
    }
    for (const auto &item:first) std::cout << item.first << ' ' << item.second << '\n';
}
