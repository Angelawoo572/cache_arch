#pragma once
#include <cstdint>
#include <cstdio>

class CACHE;
class PACKET;
class BLOCK;
class O3_CPU;
class ooo_model_instr;

void hybrid_init(unsigned active, uint64_t warmup, uint64_t instructions);
unsigned hybrid_active();
uint64_t hybrid_skip(unsigned cpu);
bool hybrid_may_fetch(const O3_CPU &core);
FILE *hybrid_open(const char *path, unsigned core);
void hybrid_observe_instruction(unsigned core, const ooo_model_instr &instruction);
uint64_t hybrid_ppn(unsigned core, uint64_t virtual_address);
void hybrid_head(const O3_CPU &core);
void hybrid_retire(unsigned core, uint64_t id);
void hybrid_cache(CACHE &cache, const PACKET &packet, const char *stage, int hit = -1);
void hybrid_ingress(CACHE &cache, const PACKET &packet);
void hybrid_stride_state(const char *reason, int64_t delta);
void hybrid_fill(CACHE &cache, const PACKET &packet, const BLOCK &victim);
void hybrid_terminal(const PACKET &request, const char *stage, const PACKET *demand = nullptr);
void hybrid_dispatch(const PACKET &request);
void hybrid_transfer(unsigned type, bool write);
void hybrid_bank_cycle();
void hybrid_finish();
