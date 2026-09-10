#include <cstdint>
#include <cstring>
#include "stride.h"
#include "champsim.h"
#include "memory_class.h"
#include <cassert>
#include <iostream>

namespace knob { uint32_t stride_num_trackers=64, stride_pref_degree=2; }

int main() {
    StridePrefetcher s("stride");
    std::vector<uint64_t> result;
    s.invoke_prefetcher(7,60*64,0,LOAD,result); assert(result.empty());
    s.invoke_prefetcher(7,61*64,0,LOAD,result); assert(result.empty());
    s.invoke_prefetcher(7,62*64,0,LOAD,result);
    assert(result.size()==2 && result[0]==62*64 && result[1]==63*64);
    result.clear(); s.invoke_prefetcher(7,63*64,0,LOAD,result);
    assert(result.size()==1 && result[0]==63*64); // page-boundary truncation
    result.clear(); s.invoke_prefetcher(7,63*64,0,LOAD,result);
    assert(result.empty()); // zero stride remains silent without state update
    result.clear(); s.invoke_prefetcher(7,64*64,0,LOAD,result);
    assert(result.size()==2 && result[0]==64*64 && result[1]==65*64);
    result.clear();
    s.invoke_prefetcher(17,200*64,0,LOAD,result);
    s.invoke_prefetcher(17,199*64,0,LOAD,result);
    s.invoke_prefetcher(17,198*64,0,LOAD,result);
    assert(result.size()==2 && result[0]==198*64 && result[1]==197*64);
    assert(s.dynamic602_tracker_count()==2);
    std::cout << "PASS preserved conventional current-line-first Stride, sign, zero, and page edge\n";
}
