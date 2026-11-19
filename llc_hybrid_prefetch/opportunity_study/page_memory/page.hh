#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace page_memory {
using U=uint64_t;
using Scores=std::array<double,64>;
struct Histogram { unsigned support=0; std::array<unsigned,64> yes{}; };
struct Visit { U page=UINT64_MAX,index=0,cycle=0; unsigned offset=64,pc=4096; };
inline U mix(U a,U b) { return (a^b)*UINT64_C(11400714819323198485); }
inline bool granted(U index) {
    U x=index/8+447;
    x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);
    return index%8==(x^(x>>31))%8;
}
class Engine {
    U index=0,first=0,end=0;
    std::array<Visit,64> window{};
    std::array<Visit,16> bounded{};
    std::array<std::unordered_map<U,Histogram>,3> tables;
    Scores prior{};
    std::string mode;
    std::ofstream log;
    bool small=false;
    static bool valid(const Visit& v,U pg,U i,U t) {
        return v.page==pg && i-v.index<=64 && t-v.cycle<=4096;
    }
    Scores score(unsigned m,U key,const Scores& fallback) const {
        auto hit=tables[m].find(key);
        if(hit==tables[m].end()) return fallback;
        const Histogram& h=hit->second;
        Scores out{};
        for(unsigned d=0;d<64;++d) out[d]=(h.yes[d]+8*fallback[d])/(h.support+8);
        return out;
    }
    static unsigned pick(const Scores& scores,unsigned self) {
        unsigned best=(self==0?1:0);
        for(unsigned d=0;d<64;++d) if(d!=self && scores[d]>scores[best]) best=d;
        return best;
    }
public:
    Engine() {
        const char* file=std::getenv("PAGE_TABLE");
        const char* which=std::getenv("PAGE_MODE");
        if(!file || !which) throw std::runtime_error("PAGE_TABLE and PAGE_MODE required");
        mode=which;
        small=mode=="offset16" || mode=="link16";
        std::ifstream in(file);
        if(!(in>>first>>end)) throw std::runtime_error("invalid page table header");
        for(auto& x:prior) if(!(in>>x)) throw std::runtime_error("missing prior");
        unsigned stage;U key;
        while(in>>stage>>key) {
            if(stage>2) throw std::runtime_error("unknown table");
            auto& h=tables[stage][key];
            if(!(in>>h.support)) throw std::runtime_error("missing support");
            for(auto& v:h.yes) if(!(in>>v)) throw std::runtime_error("missing counts");
        }
        log.open(std::string(std::getenv("HYBRID_OUT"))+"/decisions.csv");
        log<<"index,instr,operand,pc,base,cycle,current,page_offset_only,page_link,prior_age,prior_cycles,chosen,action\n";
    }
    int step(U ip,U line,U instr,unsigned operand,U cycle,bool native_matches,int64_t native_delta) {
        const U i=index++, pg=line/64;
        const unsigned off=line%64,pc=ip%4096;
        Visit previous;
        if(small) {
            const auto& v=bounded[pg%bounded.size()];
            if(valid(v,pg,i,cycle)) previous=v;
        } else {
            for(U d=1;d<=std::min<U>(i,64);++d) {
                const auto& v=window[(i-d)%window.size()];
                if(valid(v,pg,i,cycle)) { previous=v;break; }
            }
        }
        Visit now; now.page=pg; now.index=i; now.cycle=cycle; now.offset=off; now.pc=pc;
        window[i%window.size()]=now;
        bounded[pg%bounded.size()]=now;
        if(i<first || i>=end || !granted(i)) return 0;
        const bool known=previous.page!=UINT64_MAX;
        U key=mix(pc,off),ok=mix(key,previous.offset),pk=mix(ok,previous.pc);
        const auto base=score(0,key,prior);
        const unsigned selected=pick(base,off),by_offset=pick(score(1,ok,base),off),by_link=pick(score(2,pk,base),off);
        int target=-1;
        if(mode=="current") target=selected;
        else if(mode=="offset" || mode=="offset16") target=by_offset;
        else if(mode=="link" || mode=="link16") target=by_link;
        else if(mode=="next") target=off+1;
        else if(mode=="repeat" && known) target=2*static_cast<int>(off)-static_cast<int>(previous.offset);
        else if(mode!="shadow" && mode!="repeat") throw std::runtime_error("unknown page replay");
        bool legal=target>=0 && target<64 && target!=static_cast<int>(off);
        const int delta=legal?target-static_cast<int>(off):0;
        bool agreement=legal && native_matches && native_delta==delta;
        log<<i<<','<<instr<<','<<operand<<','<<ip<<','<<line<<','<<cycle<<','<<64*pg+selected<<','
           <<64*pg+by_offset<<','<<64*pg+by_link<<','<<(known?static_cast<int64_t>(i-previous.index):-1)<<','
           <<(known?static_cast<int64_t>(cycle-previous.cycle):-1)<<','<<(legal?static_cast<int64_t>(64*pg+target):-1)
           <<','<<(agreement?"agree":legal?"replace":"none")<<'\n';
        return legal&&!agreement?delta:0;
    }
};
}
