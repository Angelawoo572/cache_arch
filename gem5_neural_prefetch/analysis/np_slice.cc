/*
 * np_slice: dependence edges between loads, from the architectural streams.
 *
 * Reads S1 (uop), S2 (load) and S3 (store) of one capture and writes, for
 * every committed load, the earlier loads whose values flow into its address
 * through register dataflow and through store-to-load pairs.
 *
 *   np_slice CAPTURE_DIR OUT_FILE
 *
 * Output: raw little-endian u64 rows, one per load, in S2 order:
 *   ld, count, then four pairs (producer_ld, info)
 * info = depth | (through_memory << 16). depth counts the micro-ops between
 * the producer's value and the consumer's address register. Unused pairs are
 * all ones. Shift and base are fitted later from the recorded values.
 */

#include <zlib.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{

constexpr int maxProducers = 4;
constexpr unsigned maxDepth = 24;
constexpr uint64_t none = ~0ULL;

struct Entry
{
    uint64_t ld;
    uint16_t depth;
    uint16_t throughMemory;
};

struct Prov
{
    int n = 0;
    Entry e[maxProducers];

    void
    add(const Entry &entry)
    {
        if (entry.depth > maxDepth)
            return;
        for (int i = 0; i < n; ++i) {
            if (e[i].ld == entry.ld) {
                if (entry.depth < e[i].depth)
                    e[i] = entry;
                return;
            }
        }
        // Keep the nearest producers: smaller depth first, then the most
        // recent load.
        int pos = n;
        for (int i = 0; i < n; ++i) {
            if (entry.depth < e[i].depth ||
                (entry.depth == e[i].depth && entry.ld > e[i].ld)) {
                pos = i;
                break;
            }
        }
        if (pos >= maxProducers)
            return;
        const int last = n < maxProducers ? n : maxProducers - 1;
        for (int i = last; i > pos; --i)
            e[i] = e[i - 1];
        e[pos] = entry;
        if (n < maxProducers)
            ++n;
    }

    void
    merge(const Prov &other, unsigned extraDepth, bool viaMemory)
    {
        for (int i = 0; i < other.n; ++i) {
            Entry entry = other.e[i];
            entry.depth += extraDepth;
            if (viaMemory)
                entry.throughMemory = 1;
            add(entry);
        }
    }
};

class Reader
{
  public:
    Reader(const std::string &path, unsigned _width) : width(_width)
    {
        file = gzopen(path.c_str(), "rb");
        if (!file) {
            std::fprintf(stderr, "cannot open %s\n", path.c_str());
            std::exit(2);
        }
        gzbuffer(file, 1 << 20);
        row.resize(width);
        advance();
    }

    ~Reader() { gzclose(file); }

    bool valid() const { return ok; }
    const uint64_t &operator[](unsigned i) const { return row[i]; }

    void
    advance()
    {
        const int want = width * sizeof(uint64_t);
        ok = gzread(file, row.data(), want) == want;
    }

  private:
    gzFile file;
    unsigned width;
    std::vector<uint64_t> row;
    bool ok = false;
};

inline uint64_t
maskToSize(uint64_t value, unsigned size)
{
    return size >= 8 ? value : value & ((1ULL << (8 * size)) - 1);
}

} // anonymous namespace

int
main(int argc, char **argv)
{
    if (argc != 3) {
        std::fprintf(stderr, "usage: np_slice CAPTURE_DIR OUT_FILE\n");
        return 2;
    }
    const std::string dir = argv[1];
    Reader uop(dir + "/uop.u64le.gz", 5);
    Reader load(dir + "/load.u64le.gz", 5);
    Reader store(dir + "/store.u64le.gz", 5);

    FILE *out = std::fopen(argv[2], "wb");
    if (!out) {
        std::fprintf(stderr, "cannot create %s\n", argv[2]);
        return 2;
    }

    std::vector<Prov> regProv(1 << 16);
    std::vector<uint64_t> regValue(1 << 16, 0);
    std::vector<uint8_t> regKnown(1 << 16, 0);
    std::unordered_map<uint64_t, Prov> memProv;
    memProv.reserve(1 << 20);

    uint64_t uopIndex = 0, loadIndex = 0, storesMatched = 0, storesSeen = 0;
    uint64_t loadsWithProducer = 0;

    for (; uop.valid(); uop.advance(), ++uopIndex) {
        const uint64_t meta = uop[1];
        const bool isLoad = (meta >> 16) & 1;
        const bool isStore = (meta >> 17) & 1;
        const bool hasValue = (meta >> 23) & 1;

        uint16_t src[4], dst[2];
        for (int i = 0; i < 4; ++i)
            src[i] = (uop[2] >> (16 * i)) & 0xffff;
        for (int i = 0; i < 2; ++i)
            dst[i] = (uop[3] >> (16 * i)) & 0xffff;

        if (isLoad) {
            if (!load.valid() || load[0] != uopIndex) {
                std::fprintf(stderr, "load stream out of step at uop %llu\n",
                             (unsigned long long)uopIndex);
                return 3;
            }
            // Producers of this load's address.
            Prov address;
            for (int i = 0; i < 4; ++i)
                if (src[i] != 0xffff)
                    address.merge(regProv[src[i]], 0, false);

            uint64_t row[2 + 2 * maxProducers];
            row[0] = loadIndex;
            row[1] = address.n;
            for (int i = 0; i < maxProducers; ++i) {
                if (i < address.n) {
                    row[2 + 2 * i] = address.e[i].ld;
                    row[3 + 2 * i] = address.e[i].depth |
                        (uint64_t(address.e[i].throughMemory) << 16);
                } else {
                    row[2 + 2 * i] = row[3 + 2 * i] = none;
                }
            }
            std::fwrite(row, sizeof(uint64_t), 2 + 2 * maxProducers, out);
            loadsWithProducer += address.n > 0;

            // The loaded value starts a new chain. If the bytes were put
            // there by a store in this window, its producers carry over.
            Prov value;
            value.add(Entry{loadIndex, 0, 0});
            const unsigned size = load[3] & 0xff;
            if (size == 8 || size == 4) {
                auto it = memProv.find(load[2]);
                if (it != memProv.end())
                    value.merge(it->second, 1, true);
            }
            for (int i = 0; i < 2; ++i)
                if (dst[i] != 0xffff)
                    regProv[dst[i]] = value;

            load.advance();
            ++loadIndex;
        } else if (isStore) {
            ++storesSeen;
            if (store.valid() && store[0] == uopIndex) {
                const unsigned size = store[3] & 0xff;
                if (size == 8 || size == 4) {
                    // The data register is the source whose last value is
                    // the stored value.
                    const uint64_t stored = maskToSize(store[4], size);
                    int data = -1;
                    for (int i = 0; i < 4; ++i) {
                        if (src[i] == 0xffff || !regKnown[src[i]])
                            continue;
                        if (maskToSize(regValue[src[i]], size) == stored) {
                            data = i;
                            // Prefer a register that has producers.
                            if (regProv[src[i]].n > 0)
                                break;
                        }
                    }
                    if (data >= 0 && regProv[src[data]].n > 0) {
                        memProv[store[2]] = regProv[src[data]];
                        ++storesMatched;
                    } else {
                        memProv.erase(store[2]);
                    }
                }
                store.advance();
            }
        } else {
            Prov result;
            for (int i = 0; i < 4; ++i)
                if (src[i] != 0xffff)
                    result.merge(regProv[src[i]], 1, false);
            for (int i = 0; i < 2; ++i)
                if (dst[i] != 0xffff)
                    regProv[dst[i]] = result;
        }

        if (hasValue && dst[0] != 0xffff) {
            regValue[dst[0]] = uop[4];
            regKnown[dst[0]] = 1;
        }
    }
    std::fclose(out);

    std::fprintf(stderr,
                 "uops %llu loads %llu loads_with_producer %llu "
                 "stores %llu stores_with_traced_data %llu\n",
                 (unsigned long long)uopIndex, (unsigned long long)loadIndex,
                 (unsigned long long)loadsWithProducer,
                 (unsigned long long)storesSeen,
                 (unsigned long long)storesMatched);
    return 0;
}
