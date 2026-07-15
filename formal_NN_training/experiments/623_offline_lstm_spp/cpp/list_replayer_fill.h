#ifndef LIST_REPLAYER_FILL_H
#define LIST_REPLAYER_FILL_H

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "prefetcher.h"

class CACHE;

// Replays either a captured normal-SPP action or a direct-NN action without
// changing its requested cache destination.
// PFETCH_LIST_PATH format:
//   pc,line,occ,prefetch_addr,fill_level
// `fill_level` is an explicit ChampSim action (FILL_L2=2 or FILL_LLC=4).
class ListReplayerFill : public Prefetcher
{
public:
    ListReplayerFill(std::string type, CACHE* cache);
    ~ListReplayerFill();

    void invoke_prefetcher(uint64_t pc, uint64_t address, uint8_t cache_hit,
                           uint8_t type, std::vector<uint64_t>& pref_addr);
    void dump_stats();
    void print_config();

private:
    struct TriggerKey {
        uint64_t pc = 0;
        uint64_t line = 0;
        uint64_t occ = 0;
        bool operator==(const TriggerKey& other) const
        {
            return pc == other.pc && line == other.line && occ == other.occ;
        }
    };

    struct TriggerKeyHash {
        size_t operator()(const TriggerKey& key) const
        {
            const size_t h1 = std::hash<uint64_t>()(key.pc);
            const size_t h2 = std::hash<uint64_t>()(key.line);
            const size_t h3 = std::hash<uint64_t>()(key.occ);
            return h1 ^ (h2 << 1) ^ (h3 << 7);
        }
    };

    struct PairKey {
        uint64_t pc = 0;
        uint64_t line = 0;
        bool operator==(const PairKey& other) const
        {
            return pc == other.pc && line == other.line;
        }
    };

    struct PairKeyHash {
        size_t operator()(const PairKey& key) const
        {
            const size_t h1 = std::hash<uint64_t>()(key.pc);
            const size_t h2 = std::hash<uint64_t>()(key.line);
            return h1 ^ (h2 << 1);
        }
    };

    struct Action {
        uint64_t address = 0;
        int fill_level = 0;
    };

    void load_table();

    CACHE* cache_;
    std::unordered_map<TriggerKey, std::vector<Action>, TriggerKeyHash> table_;
    std::unordered_map<PairKey, uint64_t, PairKeyHash> occurrences_;
    std::string path_;
    uint64_t callbacks_ = 0;
    uint64_t loaded_ = 0;
    uint64_t matched_ = 0;
    uint64_t emitted_ = 0;
    uint64_t emitted_l2_ = 0;
    uint64_t emitted_llc_ = 0;
    bool stats_dumped_ = false;
};

#endif
