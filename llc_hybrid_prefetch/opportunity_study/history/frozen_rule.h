#pragma once
// Reference diagnostic. Each Engine belongs to one core and starts cold at ROI.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace wide_probe {
class Engine {
    using U = uint64_t;
    using I = int64_t;
    static constexpr unsigned width = 59;
    using Rank = std::array<unsigned, width>;
    struct Entry { bool valid=false; U first=0, second=0; I value=0; };
    using Table = std::array<Entry,256>;
    std::string mode;
    Rank global{};
    std::unordered_map<U, Rank> ranks;
    std::unordered_set<U> pages;
    std::unordered_map<U, I> last_pc_line;
    std::unordered_map<U, unsigned> last_page_offset;
    std::vector<U> recent_pages;
    Table by_addr{}, by_pc{}, by_chain{}, chain_delta{};
    std::array<U,3> past_pc{}, past_line{}; // 0 = last ingress
    U n=0, last_key=0;
    unsigned action=width;
    bool last_grant=false;

    static U mix(U a,U b) {return (a^b)*UINT64_C(11400714819323198485);}
    static U delta_class(I delta) {
        U x=delta<0 ? U(-delta) : U(delta);
        return (x==0 ? 0 : x==1 ? 1 : x<=4 ? 2 : x<64 ? 3 : 4)+(delta<0 ? 5:0);
    }
    static bool grant(U i) {
        U x=i/8+447;
        x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
        x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);
        return (i%8)==((x^(x>>31))%8);
    }
    static U slot(U a,U b) {return (a*1000003)^b;}
    static void put(Table& tab,U hash,U a,U b,I value) {
        auto& e=tab[hash%256];e.valid=true;e.first=a;e.second=b;e.value=value;
    }
    static bool get(const Table& tab,U hash,U a,U b,I& out) {
        const auto& e=tab[hash%256];
        if(!e.valid || e.first!=a || e.second!=b)return false;
        out=e.value;return true;
    }
    void read_table(const char* path) {
        if(!path)throw std::runtime_error("WIDE_TABLE required for recent");
        std::ifstream in(path);
        if(!in)throw std::runtime_error("cannot open WIDE_TABLE");
        std::string kind; U key; unsigned size,global_count=0;
        while(in>>kind>>key>>size) {
            Rank rank{};std::array<bool,width> used{};
            for(auto& r:rank) {
                if(!(in>>r) || r>=width || used[r])throw std::runtime_error("invalid frozen candidate ranking");
                used[r]=true;
            }
            if(kind=="global"){global=rank;++global_count;}
            else if(kind=="key" && size>=8)ranks.emplace(key,rank);
            else throw std::runtime_error("invalid frozen context");
        }
        if(global_count!=1)throw std::runtime_error("missing frozen global prior");
    }
public:
    Engine():mode(std::getenv("WIDE_MODE")?std::getenv("WIDE_MODE"):"off") {
        if(mode=="recent")read_table(std::getenv("WIDE_TABLE"));
        else if(mode!="plus128" && mode!="off")throw std::runtime_error("invalid WIDE_MODE");
    }
    bool granted() const {return last_grant;}
    unsigned chosen_action() const {return action;}
    U context_key() const {return last_key;}
    size_t seen_pages() const {return pages.size();}
    size_t seen_pcs() const {return last_pc_line.size();}
    size_t trained_contexts() const {return ranks.size();}

    // Call once per measured LLC demand ingress. Return zero for no candidate.
    U step(U pc,U line) {
        last_grant=grant(n);action=width;
        if(mode=="off"){++n;return 0;}
        const U page=line/64, offset=line%64;
        pages.insert(page);
        if(mode=="plus128") {
            ++n;
            if(last_grant && pages.count((line+128)/64)){action=34;return line+128;}
            return 0;
        }
        if(n) {
            U pg=past_line[0]/64;
            last_page_offset[pg]=past_line[0]%64;
            recent_pages.erase(std::remove(recent_pages.begin(),recent_pages.end(),pg),recent_pages.end());
            recent_pages.insert(recent_pages.begin(),pg);
            if(recent_pages.size()>5)recent_pages.resize(5);
        }
        if(n>=2) {
            U olderpc=past_pc[1], olderline=past_line[1], predecessor=n>=3?past_pc[2]:0;
            U chain=slot(predecessor,olderpc);
            // Tuples are stored separately to preserve keys wider than 64 bits.
            put(by_addr,olderline,0,olderline,past_line[0]);
            put(by_pc,olderpc,0,olderpc,past_line[0]);
            put(by_chain,chain,predecessor,olderpc,past_line[0]);
            put(chain_delta,chain,predecessor,olderpc,I(past_line[0])-I(olderline));
        }
        last_key=mix(mix(pc,offset/8),delta_class(n?I(line)-I(past_line[0]):0));
        U target=0;
        if(last_grant) {
            const I same[]={-32,-16,-8,-4,-3,-2,-1,1,2,3,4,8,16,32};
            const I wide[]={-1024,-512,-256,-128,-96,-64,-32,-16,-8,-4,-2,-1,1,2,4,8,16,32,64,96,128,256,512,1024};
            std::array<I,width> cand;cand.fill(-1);
            for(unsigned j=0;j<14;++j){I x=I(line)+same[j];if(x>=0 && U(x)/64==page)cand[j]=x;}
            for(unsigned j=0;j<24;++j){I x=I(line)+wide[j];if(x>=0 && U(x)/64!=page && pages.count(U(x)/64))cand[14+j]=x;}
            unsigned j=38;
            for(U prior:recent_pages) {
                if(prior==page)continue;
                I old=last_page_offset.at(prior);
                for(I o:{I(offset),old,old-1,old+1}) {
                    if(o>=0 && o<64)cand[j]=I(prior*64)+o;
                    ++j;
                }
                if(j==54)break;
            }
            U predecessor=n?past_pc[0]:0, chain=slot(predecessor,pc);
            I x;
            if(get(by_addr,line,0,line,x))cand[54]=x;
            if(get(by_pc,pc,0,pc,x))cand[55]=x;
            if(get(by_chain,chain,predecessor,pc,x))cand[56]=x;
            auto it=last_pc_line.find(pc);
            if(it!=last_pc_line.end())cand[57]=2*I(line)-it->second;
            if(get(chain_delta,chain,predecessor,pc,x))cand[58]=I(line)+x;
            auto ranked=ranks.find(last_key);
            const auto& rank=ranked==ranks.end()?global:ranked->second;
            for(auto k:rank){
                I addr=cand[k];
                if(addr>0 && U(addr)!=line && pages.count(U(addr)/64)) {
                    target=U(addr);action=k;break;
                }
            }
        }
        last_pc_line[pc]=I(line);
        for(unsigned j=2;j>0;--j){past_pc[j]=past_pc[j-1];past_line[j]=past_line[j-1];}
        past_pc[0]=pc;past_line[0]=line;++n;return target;
    }
};
}
