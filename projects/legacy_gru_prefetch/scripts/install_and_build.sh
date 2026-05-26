#!/usr/bin/env bash
# install_and_build.sh
# Build three project-specific ChampSim binaries under external/ChampSim:
#   bin/champsim.baseline
#   bin/champsim.dumper
#   bin/champsim.replayer

set -uo pipefail

WORKDIR="$(pwd)"
CHAMP="$WORKDIR/external/ChampSim"

if [ ! -d "$CHAMP" ]; then
  echo "[error] $CHAMP not found. Run setup_champsim.sh first."
  exit 1
fi

# ---- 1. Write trace_dumper module ----
mkdir -p "$CHAMP/prefetcher/trace_dumper"
cat > "$CHAMP/prefetcher/trace_dumper/trace_dumper.h" <<'EOF'
#ifndef TRACE_DUMPER_H
#define TRACE_DUMPER_H

#include "address.h"
#include "cache.h"
#include "modules.h"
#include <cstdio>
#include <cstdint>

class trace_dumper : public champsim::modules::prefetcher
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
  std::FILE* fh_ = nullptr;
  uint64_t   counter_ = 0;
};

#endif
EOF

cat > "$CHAMP/prefetcher/trace_dumper/trace_dumper.cc" <<'EOF'
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
  if (type == access_type::LOAD || type == access_type::RFO) {
    if (fh_) {
      uint64_t a = addr.template to<uint64_t>();
      uint64_t p = ip.template to<uint64_t>();
      std::fprintf(fh_, "%lu,0x%lx,0x%lx,%d\n",
                   counter_++, a, p, cache_hit ? 1 : 0);
    }
  }
  return metadata_in;
}
EOF

# ---- 2. Write list_replayer module ----
mkdir -p "$CHAMP/prefetcher/list_replayer"
cat > "$CHAMP/prefetcher/list_replayer/list_replayer.h" <<'EOF'
#ifndef LIST_REPLAYER_H
#define LIST_REPLAYER_H

#include "address.h"
#include "cache.h"
#include "modules.h"
#include <cstdio>
#include <cstdint>
#include <unordered_map>

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
  std::unordered_map<uint64_t, uint64_t> table_;
  uint64_t counter_ = 0;
  uint64_t issued_ = 0;
};

#endif
EOF

cat > "$CHAMP/prefetcher/list_replayer/list_replayer.cc" <<'EOF'
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
    std::fprintf(stderr, "[list_replayer] WARNING: could not open %s; no prefetches will be issued.\n", path);
    return;
  }

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
  std::fprintf(stderr, "[list_replayer] loaded %lu prefetch entries from %s\n", loaded, path);
  counter_ = 0;
  issued_ = 0;
}

void list_replayer::prefetcher_final_stats()
{
  std::fprintf(stderr, "[list_replayer] issued %lu prefetches over %lu accesses\n", issued_, counter_);
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
      champsim::address tgt{it->second};
      bool ok = prefetch_line(tgt, true, metadata_in);
      if (ok) ++issued_;
    }
    ++counter_;
  }
  return metadata_in;
}
EOF

echo "[install] C++ modules written to:"
echo "  $CHAMP/prefetcher/trace_dumper/"
echo "  $CHAMP/prefetcher/list_replayer/"

# ---- 3. Write 3 JSON configs ----
mkdir -p "$CHAMP/_cfg"

cat > "$CHAMP/_cfg/cfg_baseline.json" <<'JSON'
{ "LLC": { "replacement": "lru" } }
JSON

cat > "$CHAMP/_cfg/cfg_dumper.json" <<'JSON'
{
  "ooo_cpu": [{ "L1D": { "prefetcher": "trace_dumper" } }],
  "LLC":     { "replacement": "lru" }
}
JSON

cat > "$CHAMP/_cfg/cfg_replayer.json" <<'JSON'
{
  "ooo_cpu": [{ "L1D": { "prefetcher": "list_replayer" } }],
  "LLC":     { "replacement": "lru" }
}
JSON

build () {
  local tag=$1
  local cfg=$2
  echo
  echo "[build] tag=$tag cfg=$cfg"
  cd "$CHAMP" || exit 1
  rm -f bin/champsim

  python3 ./config.sh "$cfg" > "/tmp/config_${tag}.log" 2>&1 || {
    echo "[error] config.sh failed for $tag. Last 60 lines:"
    tail -60 "/tmp/config_${tag}.log"
    return 1
  }

  if ! make -j8 > "/tmp/build_${tag}.log" 2>&1; then
    echo "[error] make failed for $tag. Last 80 lines:"
    tail -80 "/tmp/build_${tag}.log"
    echo "[hint] full log: /tmp/build_${tag}.log"
    return 1
  fi

  if [ ! -x bin/champsim ]; then
    echo "[error] bin/champsim was not produced for $tag. Last 80 build lines:"
    tail -80 "/tmp/build_${tag}.log"
    return 1
  fi

  cp bin/champsim "bin/champsim.${tag}"
  echo "[build] OK -> bin/champsim.${tag}"
  return 0
}

OK_COUNT=0
build "baseline" "projects/legacy_gru_prefetch/_cfg/cfg_baseline.json" && OK_COUNT=$((OK_COUNT+1))
build "dumper"   "projects/legacy_gru_prefetch/_cfg/cfg_dumper.json"   && OK_COUNT=$((OK_COUNT+1))
build "replayer" "projects/legacy_gru_prefetch/_cfg/cfg_replayer.json" && OK_COUNT=$((OK_COUNT+1))

echo
echo "============================================"
if [ "$OK_COUNT" -eq 3 ]; then
  echo "[install] ALL DONE. Three binaries ready:"
  ls -l "$CHAMP/bin/champsim.baseline" "$CHAMP/bin/champsim.dumper" "$CHAMP/bin/champsim.replayer"
else
  echo "[install] PARTIAL: only $OK_COUNT/3 binaries built."
  echo "Build logs at /tmp/build_{baseline,dumper,replayer}.log"
  exit 6
fi
echo "============================================"
