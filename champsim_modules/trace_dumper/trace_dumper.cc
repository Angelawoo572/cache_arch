#include "trace_dumper.h"
#include <cstdlib>
#include <cstring>

void trace_dumper::prefetcher_initialize()
{
  const char* env = std::getenv("TRACE_DUMP_PATH");
  const char* path = (env && std::strlen(env) > 0) ? env : "/tmp/access_trace.csv";
  fh_ = std::fopen(path, "w");
  if (fh_) {
    std::fprintf(fh_, "idx,addr_hex,pc_hex,hit\n");
    std::fflush(fh_);
  }
  counter_ = 0;
}

void trace_dumper::prefetcher_final_stats()
{
  if (fh_) {
    std::fflush(fh_);
    std::fclose(fh_);
    fh_ = nullptr;
  }
}

uint32_t trace_dumper::prefetcher_cache_operate(champsim::address addr,
                                                champsim::address ip,
                                                bool cache_hit,
                                                bool /*useful_prefetch*/,
                                                access_type type,
                                                uint32_t metadata_in)
{
  // Only log demand loads (skip prefetch self-traffic, writebacks, etc.)
  if (type == access_type::LOAD || type == access_type::RFO) {
    if (fh_) {
      // Get raw uint64. Modern ChampSim provides .to<uint64_t>(); if the API
      // ever changes, this is the only line that needs updating.
      uint64_t a = addr.template to<uint64_t>();
      uint64_t p = ip.template to<uint64_t>();
      std::fprintf(fh_, "%lu,0x%lx,0x%lx,%d\n",
                   counter_++, a, p, cache_hit ? 1 : 0);
    }
  }
  // No prefetches issued.
  return metadata_in;
}
