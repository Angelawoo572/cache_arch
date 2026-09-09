#ifndef ONLINE602_HOOKS_H
#define ONLINE602_HOOKS_H
#include <cstdint>
#include <cstdio>
#include <string>
class CACHE;
class PACKET;
// Engine hooks run from the outer simulator cycle loop, including core stalls.
void online602_service(CACHE* cache);
// Return comma-separated JSON members, without enclosing braces or leading comma.
std::string online602_engine_json(unsigned cpu);
// Measurement has no connection to neural features, labels, or scheduling.
void online602_begin(unsigned cpu);
void online602_observe_cycle(unsigned cpu, bool final = false);
void online602_fill(CACHE*, unsigned set, unsigned way, const PACKET*);
void online602_invalidate(CACHE*, unsigned set, unsigned way);
void online602_demand_hit(CACHE*, unsigned set, unsigned way);
void online602_late(CACHE*, const PACKET*);
void online602_enqueue(CACHE*, PACKET*);
void online602_redundant(CACHE*, const PACKET*, bool cache_hit);
void online602_skip(FILE* file, bool cloudsuite, unsigned cpu);
bool online602_read_allowed(unsigned cpu);
void online602_record_read(unsigned cpu);
bool online602_bounded_trace();
#endif
