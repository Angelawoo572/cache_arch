// Count exact LLC outcomes for identically retired dynamic instruction and operand IDs.
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <zlib.h>

constexpr unsigned capacity = 131072;
using Chunk = std::vector<unsigned char>;

void read_exact(gzFile f, Chunk &chunk, unsigned n) {
    unsigned taken = 0;
    while (taken < n) {
        int got = gzread(f, chunk.data() + taken, n - taken);
        if (got <= 0)
            throw std::runtime_error("incomplete outcome file");
        taken += got;
    }
}

void count_pair(const std::vector<Chunk> &chunks, unsigned bytes, std::vector<uint64_t> &counts) {
    for (unsigned j = 0; j < bytes; ++j) {
        const unsigned x = chunks[0][j], y = chunks[1][j];
        if (!(x | y))
            continue;
        ++counts[((x & 15) << 4) | (y & 15)];
        ++counts[(x & 240) | (y >> 4)];
    }
}

void count_joint(const std::vector<Chunk> &chunks, unsigned bytes, std::vector<uint64_t> &counts) {
    for (unsigned j = 0; j < bytes; ++j) {
        const auto base = chunks[0][j];
        if (!base)
            continue;
        for (unsigned shift : {0u, 4u}) {
            if (((base >> shift) & 15) != 2)
                continue;
            unsigned key = 0;
            for (unsigned arm = 1; arm < chunks.size(); ++arm)
                key = (key << 4) | ((chunks[arm][j] >> shift) & 15);
            ++counts[key];
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 7)
        return 2;
    const bool joint = argc == 7;
    const auto roi = std::stoull(argv[argc - 1]);
    std::vector<gzFile> files;
    for (int i = 1; i < argc - 1; ++i) {
        gzFile f = gzopen(argv[i], "rb");
        if (!f)
            throw std::runtime_error("cannot open outcome stream");
        gzbuffer(f, 262144);
        files.push_back(f);
    }
    std::vector<Chunk> chunks(files.size(), Chunk(capacity));
    if (joint)
        std::cout << "quarter,stride,multi,spp,temporal,operands\n";
    else
        std::cout << "quarter,nopf_status,arm_status,operands\n";
    for (unsigned phase = 0; phase < 4; ++phase) {
        std::vector<uint64_t> counts(joint ? 65536 : 256);
        auto start = (phase * roi + 3) / 4, end = ((phase + 1) * roi + 3) / 4;
        uint64_t bytes = (end - start) * 2;
        while (bytes) {
            unsigned n = std::min<uint64_t>(bytes, capacity);
            for (unsigned i = 0; i < files.size(); ++i)
                read_exact(files[i], chunks[i], n);
            if (joint)
                count_joint(chunks, n, counts);
            else
                count_pair(chunks, n, counts);
            bytes -= n;
        }
        // Pair comparisons exclude empty/empty slots (which include instructions with no load).
        // Joint comparisons condition on an observed baseline miss; all-zero arm outcomes matter here.
        for (unsigned cell = joint ? 0u : 1u; cell < counts.size(); ++cell) {
            if (!counts[cell])
                continue;
            std::cout << phase;
            const int high = joint ? 12 : 4;
            for (int shift = high; shift >= 0; shift -= 4)
                std::cout << ',' << ((cell >> shift) & 15);
            std::cout << ',' << counts[cell] << '\n';
        }
    }
    for (auto f : files) {
        if (gzgetc(f) != -1 || !gzeof(f) || gzclose(f) != Z_OK)
            return 4;
    }
    return 0;
}
