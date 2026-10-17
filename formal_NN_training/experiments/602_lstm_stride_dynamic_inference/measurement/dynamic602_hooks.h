#ifndef DYNAMIC602_HOOKS_H
#define DYNAMIC602_HOOKS_H
#include <cstdint>
#include <cstdio>
#include <string>
class CACHE;
class PACKET;
// Engine hooks run from the outer simulator cycle loop, including core stalls.
void dynamic602_service(CACHE* cache);
void dynamic602_roi(CACHE* cache);
bool dynamic602_allow_request(CACHE*, uint64_t ip, uint64_t base, uint64_t address, int fill_level);
// Diagnostic labels are copied before vector output / the direct cache call.
// FIFO matching consumes each actual attempt once, before shared filters.
void dynamic602_label_request(CACHE*, uint64_t ip, uint64_t base, uint64_t address,
    uint64_t phase, unsigned slot, bool current, bool list_duplicate, unsigned teacher_category);
void dynamic602_cache_request_result(CACHE*, int result);
void dynamic602_dram_event(unsigned cpu, unsigned event, unsigned type);
void dynamic602_dram_bus_start(unsigned cpu, bool write, unsigned type, uint64_t finish_cycle);
uint64_t dynamic602_total_retired_instructions(unsigned cpu);
// Return comma-separated JSON members, without enclosing braces or leading comma.
std::string dynamic602_engine_json(unsigned cpu);
// Measurement-only hooks do not feed neural features or affect scheduling.
void dynamic602_begin(unsigned cpu);
void dynamic602_observe_cycle(unsigned cpu, bool final = false);
void dynamic602_fill(CACHE*, unsigned set, unsigned way, const PACKET*);
void dynamic602_invalidate(CACHE*, unsigned set, unsigned way);
void dynamic602_demand_hit(CACHE*, unsigned set, unsigned way, unsigned type);
void dynamic602_late(CACHE*, const PACKET*, unsigned type);
void dynamic602_enqueue(CACHE*, PACKET*);
void dynamic602_redundant(CACHE*, const PACKET*, bool cache_hit);
void dynamic602_skip(FILE* file, bool cloudsuite, unsigned cpu);
bool dynamic602_read_allowed(unsigned cpu);
void dynamic602_record_read(unsigned cpu);
bool dynamic602_bounded_trace();
#endif
