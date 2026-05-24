#!/usr/bin/env bash
# install_bypass.sh -- lru_bypass for your ChampSim tree
# Matches your official replacement/lru/lru.cc API:
#   lru(CACHE* cache) : lru(cache, cache->NUM_SET, cache->NUM_WAY)
#   find_victim(... long set ...)
#   replacement_cache_fill(...)
#   update_replacement_state(... uint8_t hit)
#
# Key fix:
#   use local cycle++ just like official LRU, not current_cycle.

set -uo pipefail

WORKDIR="$(pwd)"
CHAMP="$WORKDIR/external/ChampSim"
[ -d "$CHAMP" ] || { echo "[error] $CHAMP not found"; exit 1; }

mkdir -p "$CHAMP/replacement/lru_bypass"

cat > "$CHAMP/replacement/lru_bypass/lru_bypass.h" <<'EOF'
#ifndef LRU_BYPASS_H
#define LRU_BYPASS_H

#include "address.h"
#include "cache.h"
#include "modules.h"

#include <cstdint>
#include <unordered_set>
#include <vector>

class lru_bypass : public champsim::modules::replacement
{
public:
  lru_bypass(CACHE* cache);
  lru_bypass(CACHE* cache, long sets, long ways);

  long find_victim(uint32_t triggering_cpu,
                   uint64_t instr_id,
                   long set,
                   const champsim::cache_block* current_set,
                   champsim::address ip,
                   champsim::address full_addr,
                   access_type type);

  void replacement_cache_fill(uint32_t triggering_cpu,
                              long set,
                              long way,
                              champsim::address full_addr,
                              champsim::address ip,
                              champsim::address victim_addr,
                              access_type type);

  void update_replacement_state(uint32_t triggering_cpu,
                                long set,
                                long way,
                                champsim::address full_addr,
                                champsim::address ip,
                                champsim::address victim_addr,
                                access_type type,
                                uint8_t hit);

  void replacement_final_stats();

private:
  long NUM_WAY;
  std::vector<uint64_t> last_used_cycles;
  uint64_t cycle = 0;

  std::unordered_set<uint64_t> bypass_pcs_;

  uint64_t bypassed_ = 0;
  uint64_t candidates_ = 0;

  void touch(long set, long way);
};

#endif
EOF

cat > "$CHAMP/replacement/lru_bypass/lru_bypass.cc" <<'EOF'
#include "lru_bypass.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

lru_bypass::lru_bypass(CACHE* cache)
    : lru_bypass(cache, cache->NUM_SET, cache->NUM_WAY)
{
}

lru_bypass::lru_bypass(CACHE* cache, long sets, long ways)
    : replacement(cache),
      NUM_WAY(ways),
      last_used_cycles(static_cast<std::size_t>(sets * ways), 0)
{
  const char* env = std::getenv("BYPASS_PC_LIST");

  if (env && std::strlen(env) > 0) {
    std::ifstream fh(env);

    if (fh) {
      std::string ln;

      while (std::getline(fh, ln)) {
        size_t i = 0;
        while (i < ln.size() && std::isspace(static_cast<unsigned char>(ln[i]))) {
          ++i;
        }

        if (i >= ln.size()) continue;
        if (ln[i] == '#') continue;

        unsigned long long pc = 0;

        // %llx accepts both "40245c" and "0x40245c".
        if (std::sscanf(ln.c_str() + i, "%llx", &pc) == 1) {
          bypass_pcs_.insert(static_cast<uint64_t>(pc));
        }
      }

      std::fprintf(stderr,
                   "[lru_bypass] loaded %zu bypass PCs from %s\n",
                   bypass_pcs_.size(),
                   env);
    } else {
      std::fprintf(stderr,
                   "[lru_bypass] WARNING: cannot open BYPASS_PC_LIST=%s; behaves as plain LRU.\n",
                   env);
    }
  } else {
    std::fprintf(stderr,
                 "[lru_bypass] no BYPASS_PC_LIST; behaves as plain LRU.\n");
  }
}

long lru_bypass::find_victim(uint32_t /*triggering_cpu*/,
                             uint64_t /*instr_id*/,
                             long set,
                             const champsim::cache_block* /*current_set*/,
                             champsim::address ip,
                             champsim::address /*full_addr*/,
                             access_type /*type*/)
{
  ++candidates_;

  if (!bypass_pcs_.empty()) {
    uint64_t pc = ip.template to<uint64_t>();

    if (bypass_pcs_.count(pc) > 0) {
      ++bypassed_;
      return NUM_WAY;
    }
  }

  auto begin = std::next(std::begin(last_used_cycles), set * NUM_WAY);
  auto end = std::next(begin, NUM_WAY);

  auto victim = std::min_element(begin, end);

  assert(begin <= victim);
  assert(victim < end);

  return std::distance(begin, victim);
}

void lru_bypass::touch(long set, long way)
{
  if (way < 0 || way >= NUM_WAY) return;

  last_used_cycles.at(static_cast<std::size_t>(set * NUM_WAY + way)) = cycle++;
}

void lru_bypass::replacement_cache_fill(uint32_t /*triggering_cpu*/,
                                        long set,
                                        long way,
                                        champsim::address /*full_addr*/,
                                        champsim::address /*ip*/,
                                        champsim::address /*victim_addr*/,
                                        access_type /*type*/)
{
  touch(set, way);
}

void lru_bypass::update_replacement_state(uint32_t /*triggering_cpu*/,
                                          long set,
                                          long way,
                                          champsim::address /*full_addr*/,
                                          champsim::address /*ip*/,
                                          champsim::address /*victim_addr*/,
                                          access_type type,
                                          uint8_t hit)
{
  if (hit && access_type{type} != access_type::WRITE) {
    touch(set, way);
  }
}

void lru_bypass::replacement_final_stats()
{
  std::fprintf(stderr,
               "[lru_bypass] bypassed %lu of %lu candidate fills (%.2f%%); bypass-PC-list size %zu\n",
               bypassed_,
               candidates_,
               candidates_ > 0 ? 100.0 * static_cast<double>(bypassed_) / static_cast<double>(candidates_) : 0.0,
               bypass_pcs_.size());
}
EOF

echo "[install] wrote replacement/lru_bypass/"

mkdir -p "$CHAMP/_cfg"

cat > "$CHAMP/_cfg/cfg_bypass_lru.json" <<'JSON'
{ "LLC": { "replacement": "lru_bypass" } }
JSON

build () {
  local tag=$1
  local cfg=$2

  echo
  echo "[build] tag=$tag cfg=$cfg"

  cd "$CHAMP" || exit 1
  rm -f bin/champsim

  python3 ./config.sh "$cfg" > /tmp/config_${tag}.log 2>&1 || {
    echo "[error] config.sh failed. See /tmp/config_${tag}.log"
    return 1
  }

  if ! make -j8 > /tmp/build_${tag}.log 2>&1; then
    echo "[error] make failed. First 12 error lines:"
    grep -n "error:" /tmp/build_${tag}.log | head -12
    echo "[hint] full log: /tmp/build_${tag}.log"
    return 1
  fi

  if [ ! -x bin/champsim ]; then
    echo "[error] bin/champsim not produced"
    return 1
  fi

  cp bin/champsim "bin/champsim.${tag}"
  echo "[build] OK -> bin/champsim.${tag}"
}

OK=0
build "bypass_lru" "_cfg/cfg_bypass_lru.json" && OK=$((OK+1))

echo
echo "============================================"
if [ "$OK" -eq 1 ]; then
  echo "[install] DONE."
  ls -l "$CHAMP/bin/champsim.bypass_lru"
else
  echo "[install] FAILED. See /tmp/build_bypass_lru.log"
fi
echo "============================================"