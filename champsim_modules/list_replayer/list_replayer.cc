#include "list_replayer.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>

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
  char line[256];
  uint64_t loaded = 0;
  while (std::fgets(line, sizeof(line), fh)) {
    if (!std::isdigit((unsigned char)line[0])) continue;
    uint64_t idx; unsigned long long addr;
    if (std::sscanf(line, "%lu %llx", &idx, &addr) == 2 ||
        std::sscanf(line, "%lu 0x%llx", &idx, &addr) == 2 ||
        std::sscanf(line, "%lu,0x%llx", &idx, &addr) == 2 ||
        std::sscanf(line, "%lu,%llx", &idx, &addr) == 2) {
      table_[idx] = (uint64_t)addr;
      ++loaded;
    }
  }
  std::fclose(fh);
  std::fprintf(stderr, "[list_replayer] loaded %lu prefetch entries from %s\n",
               loaded, path);
  counter_ = 0;
  issued_ = 0;
}

void list_replayer::prefetcher_final_stats()
{
  std::fprintf(stderr, "[list_replayer] issued %lu prefetches over %lu accesses\n",
               issued_, counter_);
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
      // Build a champsim::address from the raw uint64_t. Use the same address
      // class as `addr` to stay in the correct address space (virtual/phys).
      champsim::address tgt{it->second};
      // Fill into the L1D (this prefetcher will be installed at L1D); request
      // demand-prefetch priority.
      bool ok = prefetch_line(tgt, true /*fill_this_level*/, metadata_in);
      if (ok) ++issued_;
    }
    ++counter_;
  }
  return metadata_in;
}
