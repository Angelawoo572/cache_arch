#ifndef DYNAMIC602_HOOKS_H
#define DYNAMIC602_HOOKS_H
#include <cstdint>
#include <cstdio>
#include <string>
class CACHE;
class PACKET;
// Engine hooks run from the outer simulator cycle loop, including core stalls.
void dynamic602_service(CACHE* cache);
// Return comma-separated JSON members, without enclosing braces or leading comma.
std::string dynamic602_engine_json(unsigned cpu);
// Measurement-only hooks do not feed neural features or affect scheduling.
void dynamic602_begin(unsigned cpu);
void dynamic602_observe_cycle(unsigned cpu, bool final = false);
void dynamic602_fill(CACHE*, unsigned set, unsigned way, const PACKET*);
void dynamic602_invalidate(CACHE*, unsigned set, unsigned way);
void dynamic602_demand_hit(CACHE*, unsigned set, unsigned way);
void dynamic602_late(CACHE*, const PACKET*);
void dynamic602_enqueue(CACHE*, PACKET*);
void dynamic602_redundant(CACHE*, const PACKET*, bool cache_hit);
void dynamic602_skip(FILE* file, bool cloudsuite, unsigned cpu);
bool dynamic602_read_allowed(unsigned cpu);
void dynamic602_record_read(unsigned cpu);
bool dynamic602_bounded_trace();
#endif
