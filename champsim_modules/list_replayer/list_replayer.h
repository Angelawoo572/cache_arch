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
// format produced by the neural prefetcher notebooks/scripts:
//      idx  prefetch_addr_hex
// On every L1D demand access we increment a global access counter and, if the
// counter matches one or more entries in the list, issue prefetch_line for each
// target address.
//
// Important: V9 decode sweeps can emit multiple targets for the SAME idx
// (prefetch degree > 1). Therefore the table maps idx -> vector<address>, not a
// single address. Older versions used one address per idx and silently
// overwrote earlier candidates, which made degree-2/degree-4 sweeps invalid.
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
  // Map: access index -> one or more prefetch target addresses.
  std::unordered_map<uint64_t, std::vector<uint64_t>> table_;
  uint64_t counter_ = 0;
  uint64_t loaded_ = 0;
  uint64_t matched_accesses_ = 0;
  uint64_t attempted_ = 0;
  uint64_t issued_ = 0;
};

#endif