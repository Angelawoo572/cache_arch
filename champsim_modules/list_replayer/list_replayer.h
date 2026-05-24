#ifndef LIST_REPLAYER_H
#define LIST_REPLAYER_H

#include "address.h"
#include "cache.h"
#include "modules.h"
#include <cstdio>
#include <cstdint>
#include <unordered_map>
#include <vector>

// list_replayer: reads a prefetch list (env var PFETCH_LIST_PATH) in the
// format produced by neural_prefetcher_zoo.ipynb:
//      idx  prefetch_addr_hex
// On every L1D access we increment a global access counter and, if the
// counter matches an entry in the list, issue prefetch_line on its address.
// This lets us replay any offline-NN's prefetch decisions inside ChampSim
// and measure real IPC.
class list_replayer : public champsim::modules::prefetcher
{
public:
  using champsim::modules::prefetcher::prefetcher;

  void prefetcher_initialize();
  void prefetcher_final_stats();
  uint32_t prefetcher_cache_operate(champsim::address addr,
                                    champsim::address ip,
                                    bool cache_hit,
                                    bool useful_prefetch,
                                    access_type type,
                                    uint32_t metadata_in);

private:
  // Map: access index -> prefetch target address
  std::unordered_map<uint64_t, uint64_t> table_;
  uint64_t counter_ = 0;
  uint64_t issued_ = 0;
};

#endif
