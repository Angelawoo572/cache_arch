#include "list_replayer.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cctype>

void list_replayer::prefetcher_initialize()
{
  const char* env = std::getenv("PFETCH_LIST_PATH");
  const char* path = (env && std::strlen(env) > 0) ? env : "/tmp/prefetch_list.txt";

  FILE* fh = std::fopen(path, "r");
  if (!fh) {
    std::fprintf(stderr, "[list_replayer] WARNING: could not open %s; "
                         "no prefetches will be issued.\n", path);
    return;
  }

  // Parse "idx hex" lines (idx in decimal, address in 0xhex form).
  // Skip header lines that don't start with a digit.
  //
  // V9 decode sweeps may contain multiple rows with the same idx when the
  // offline policy wants degree > 1. Keep ALL of them instead of overwriting.
  char line[256];
  loaded_ = 0;
  while (std::fgets(line, sizeof(line), fh)) {
    if (!std::isdigit((unsigned char)line[0])) continue;
    uint64_t idx; unsigned long long addr;
    if (std::sscanf(line, "%lu %llx", &idx, &addr) == 2 ||
        std::sscanf(line, "%lu 0x%llx", &idx, &addr) == 2 ||
        std::sscanf(line, "%lu,0x%llx", &idx, &addr) == 2 ||
        std::sscanf(line, "%lu,%llx", &idx, &addr) == 2) {
      table_[idx].push_back((uint64_t)addr);
      ++loaded_;
    }
  }
  std::fclose(fh);
  std::fprintf(stderr, "[list_replayer] loaded %lu prefetch entries across %lu access indices from %s\n",
               loaded_, (uint64_t)table_.size(), path);
  counter_ = 0;
  matched_accesses_ = 0;
  attempted_ = 0;
  issued_ = 0;
}

void list_replayer::prefetcher_final_stats()
{
  std::fprintf(stderr,
               "[list_replayer] issued %lu prefetches over %lu accesses "
               "(%lu attempted, %lu matched access indices)\n",
               issued_, counter_, attempted_, matched_accesses_);
}

uint32_t list_replayer::prefetcher_cache_operate(champsim::address addr,
                                                 champsim::address /*ip*/,
                                                 bool /*cache_hit*/,
                                                 bool /*useful_prefetch*/,
                                                 access_type type,
                                                 uint32_t metadata_in)
{
  if (type == access_type::LOAD || type == access_type::RFO) {
    auto it = table_.find(counter_);
    if (it != table_.end()) {
      ++matched_accesses_;
      for (auto raw_tgt : it->second) {
        champsim::address tgt{raw_tgt};
        ++attempted_;
        bool ok = prefetch_line(tgt, true /*fill_this_level*/, metadata_in);
        if (ok) ++issued_;
      }
    }
    ++counter_;
  }
  return metadata_in;
}