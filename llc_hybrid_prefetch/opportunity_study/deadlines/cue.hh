#pragma once
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace earlier_cue {
using U = uint64_t;
inline unsigned token(U old, U line) {
    if (old / 64 != line / 64) return 33;
    const int delta = int(line % 64) - int(old % 64);
    return unsigned(std::max(-16, std::min(16, delta)) + 16);
}
inline unsigned set_of(U k) { return unsigned(k ^ (k >> 11) ^ (k >> 23) ^ (k >> 33)) & 31; }
struct Slot { U pc=0, line=0; int step=0; unsigned repeats=0; bool valid=false; };
struct Entry { U key=0; int delta=0; bool valid=false; };
struct Engine {
    Slot trackers[16];
    Entry table[32][4];
    bool enabled=false, seen=false, credit=true;
    unsigned mode=0, recent=65535, marker=65535;
    U index=0, marked_at=0, last=0, watermark=0, gate=0, last_spent=0;
    std::ofstream decisions;
    Engine() {
        const char* file=std::getenv("CUE_TABLE");
        if (!file) return;
        enabled=true;
        std::ifstream f(file); std::string row;
        if (!(f >> mode >> gate) || mode>2) throw std::runtime_error("invalid cue table header");
        U key; int delta; unsigned ways[32]={};
        while(f >> key >> delta) {
            auto s=set_of(key);
            if(ways[s]>=4 || !delta || std::abs(delta)>32) throw std::runtime_error("invalid cue entry");
            Entry& e=table[s][ways[s]++]; e.key=key; e.delta=delta; e.valid=true;
        }
        decisions.open(std::string(std::getenv("HYBRID_OUT"))+"/cue.csv");
        decisions<<"instr,operand,cycle,physical_base,delta,key,cue_age,action\n";
    }
    int step(U pc, U line, U instr, unsigned operand, U cycle, int native_delta, bool native_match) {
        if (!enabled) return 0;
        watermark=std::max(watermark,instr);
        const unsigned dx=seen ? token(last,line) : 34;
        const unsigned cue=(marker!=65535 && index-marked_at<=64) ? marker : 65535;
        const U base=((pc & 65535)<<9) | (((line & 63)>>3)<<6) | dx;
        const U key=(base<<16) | (mode==0 ? 65535 : mode==1 ? recent : cue);
        const auto age=cue==65535 ? 0 : index-marked_at;
        const unsigned signature=((pc & 255)<<6)|dx;
        auto& slot=trackers[(pc>>2)&15];
        if(slot.valid && slot.pc==pc) {
            const int local=token(slot.line,line);
            if(slot.repeats>=2 && local!=slot.step) { marker=signature; marked_at=index; }
            slot.repeats=(local==16 || local>=33) ? 0 : (local==slot.step ? std::min(2u,slot.repeats+1) : 1);
            slot.step=local;
        } else { slot.pc=pc; slot.valid=true; slot.step=34; slot.repeats=0; }
        slot.line=line; recent=signature; last=line; seen=true; ++index;
        if (watermark<gate) return 0;
        if (!credit && index-last_spent>=32) credit=true;
        for(const auto& e:table[set_of(key)]) if(e.valid && e.key==key) {
            const int off=int(line&63)+e.delta;
            const bool agree=native_match && native_delta==e.delta;
            const bool take=credit && !agree && off>=0 && off<64;
            const char* action=agree ? "agree" : off<0 || off>=64 ? "page" : take ? "override" : "budget";
            decisions<<instr<<','<<operand<<','<<cycle<<','<<line<<','<<e.delta<<','<<key<<','<<age<<','<<action<<'\n';
            if(take){ credit=false;last_spent=index;return e.delta; }
        }
        return 0;
    }
};
}
