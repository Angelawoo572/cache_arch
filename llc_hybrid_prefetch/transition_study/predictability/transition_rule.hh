#pragma once
#include <array>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace boundary_forecast {
using U = std::uint64_t;
using I = std::int64_t;
using Score = std::array<double,76>;
struct Row { U count; std::array<unsigned,76> positives; };
struct Visit { U page=UINT64_MAX, offset=0, index=0, cycle=0; };
inline U group(U a,U b) { return (a^b)*UINT64_C(11400714819323198485); }
inline U sample(U i) {
    U x=i/8+447;
    x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);
    return i%8==(x^(x>>31))%8;
}
inline U delta_class(I d) {
    const U mag=d<0?-d:d;
    const int rank=mag<=4?mag:(mag<=16?5:(mag<64?6:7));
    return 7+(d<0?-rank:rank);
}
class Engine {
    U index_=0,first_,end_,mode_=0,previous_global_pc_=4096;
    std::array<std::unordered_map<U,Row>,2> tables_;
    std::array<Visit,66> recent_{};
    std::unordered_map<U,U> last_pc_;
    std::unordered_set<U> known_;
    Score prior_{};
    void refine(unsigned stage,U key,Score& score) const {
        const auto it=tables_[stage].find(key);
        if (it==tables_[stage].end()) return;
        const Row& row=it->second;
        for (unsigned a=0;a<score.size();++a)
            score[a]=(row.positives[a]+8*score[a])/(row.count+8);
    }
public:
    explicit Engine(const char *path) {
        std::ifstream f(path);
        std::string head;
        if (!std::getline(f,head)) throw std::runtime_error("missing forecast table header");
        std::istringstream in(head);
        if (!(in>>first_>>end_)) throw std::runtime_error("invalid forecast table");
        if (!(in>>mode_))mode_=0;
        if (mode_>3)throw std::runtime_error("unsupported cue");
        for (double& a:prior_) if (!(f>>a)) throw std::runtime_error("truncated prior");
        unsigned stage; U key;
        while (f>>stage>>key) {
            if(stage>=2)throw std::runtime_error("invalid stage");
            Row& row=tables_[stage][key];
            if(!(f>>row.count))throw std::runtime_error("missing support");
            for (auto& c:row.positives)if(!(f>>c))throw std::runtime_error("truncated count");
        }
    }
    // Invoke on every measured per-core demand ingress, also while its grant is closed.
    // Returns -1 when abstaining; a positive return is a candidate, never an issue bypass.
    I step(U pc,U line,U cycle) {
        const U ix=index_++,page=line/64,off=line%64;
        const auto it=last_pc_.find(pc);
        const I local=it==last_pc_.end()?0:I(line)-I(it->second);
        last_pc_[pc]=line;known_.insert(page);
        U prior_offset=64,older_index=(ix?ix-1:0);
        for (unsigned age=1;age<=64 && age<=ix;++age) {
            const Visit& v=recent_[(ix-age)%recent_.size()];
            if (cycle-v.cycle>4096)break;
            if (v.page==page) {prior_offset=v.offset;older_index=(v.index?v.index-1:0);break;}
        }
        const U older_offset=(older_index==ix)?off:recent_[older_index%recent_.size()].offset;
        const U previous_pc=previous_global_pc_;
        previous_global_pc_=pc%4096;
        recent_[ix%recent_.size()]={page,off,ix,cycle};
        if (ix<first_ || ix>=end_ || !sample(ix)) return -1;
        Score scores=prior_;
        refine(0,group(group(pc%4096,off/8),delta_class(local)),scores);
        U exact=group(pc%4096,off), key=exact;
        if(mode_==0)key=group(exact,prior_offset);
        if(mode_==2)key=group(exact,older_offset);
        if(mode_==3)key=group(exact,previous_pc);
        refine(1,key,scores);
        static constexpr I wide[]={-512,-256,-128,-64,-32,-16,16,32,64,128,256,512};
        double best=0;I choice=-1;
        for (unsigned a=0;a<scores.size();++a) {
            const I target=a<64?I(page*64+a):I(line)+wide[a-64];
            if(target<0 || U(target)==line || (a>=64 && (U(target)/64==page || !known_.count(U(target)/64))))continue;
            if(scores[a]>best) {best=scores[a];choice=target;}
        }
        return choice;
    }
};
}
