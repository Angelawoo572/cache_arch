// Included after the conventional Proposal and native Stride result are available.
struct FrozenState {
    struct Local { uint64_t pc = 0, line = 0; int delta = 65; };
    std::array<Local,64> slots{};
    std::array<uint64_t,3> previous{};
    std::array<uint64_t,64> pages{};
    uint64_t count = 0;
};
std::array<FrozenState, NUM_CPUS> frozen_state{};
std::map<uint64_t,int> frozen_rules;
std::string frozen_mode;
inline uint64_t key_word(uint64_t key, uint64_t word) {
    return (key ^ word) * UINT64_C(1099511628211);
}
void frozen_init() {
    const char* mode = std::getenv("DISCOVERY_RULE");
    if (!mode) return;
    frozen_mode = mode;
    if (frozen_mode == "sparse1" || frozen_mode == "sparse4" || frozen_mode == "sparse3" || frozen_mode == "shadow") return;
    if (frozen_mode != "local" && frozen_mode != "ordered" && frozen_mode != "shape")
        throw std::runtime_error("unknown discovery mode");
    std::ifstream file(std::getenv("DISCOVERY_FIT"));
    if (!file) throw std::runtime_error("missing frozen rules");
    uint64_t key = 0;
    int delta = 0;
    while (file >> key >> delta) {
        if (!delta || delta < -512 || delta > 512 || !frozen_rules.emplace(key, delta).second)
            throw std::runtime_error("invalid frozen row");
    }
    if (!file.eof() || frozen_rules.size()>256) throw std::runtime_error("invalid frozen table");
}
Proposal frozen_apply(const PACKET& p, Proposal native) {
    auto &s = frozen_state[p.cpu];
    const uint64_t pc=p.ip, line=p.address, page=line/64, ix=s.count++;
    auto &slot=s.slots[((pc>>2)^(pc>>9)) & 63];
    const int64_t displacement=int64_t(line)-int64_t(slot.line);
    const int delta=slot.pc==pc ? int(std::max<int64_t>(-64,std::min<int64_t>(64,displacement))) : 65;
    const int previous_delta=slot.pc==pc ? slot.delta : 65;
    slot.pc=pc; slot.line=line; slot.delta=delta;
    auto chain=s.previous;
    s.previous={{chain[1],chain[2],pc}};
    s.pages[ix%64]=page;
    if (ix%8!=3 || frozen_mode=="shadow") return native;
    uint64_t key=UINT64_C(1469598103934665603);
    if (frozen_mode!="shape") key=key_word(key,pc);
    key=key_word(key,line%64);
    key=key_word(key,uint64_t(delta+64));
    key=key_word(key,uint64_t(previous_delta+64));
    if (frozen_mode=="ordered")
        for (auto x : chain) key=key_word(key,x);
    if (frozen_mode=="shape") {
        std::array<uint64_t,4> seq={{chain[0],chain[1],chain[2],pc}};
        for (size_t i=0;i<seq.size();++i) {
            size_t j=0;
            while(j<i && seq[j]!=seq[i]) ++j;
            key=key_word(key,j);
        }
    }
    int action=frozen_mode=="sparse4" ? 4 : (frozen_mode=="sparse3" ? 3 : 1);
    if (frozen_mode!="sparse1" && frozen_mode!="sparse4" && frozen_mode!="sparse3") {
        const auto found=frozen_rules.find(key);
        if(found==frozen_rules.end()) return native;
        action=found->second;
    }
    const int64_t target=int64_t(line)+action;
    const size_t valid=std::min<uint64_t>(ix+1,s.pages.size());
    if(target<=0 || ((frozen_mode=="sparse1" || frozen_mode=="sparse4" || frozen_mode=="sparse3") && uint64_t(target)/64!=page) ||
       std::find(s.pages.begin(),s.pages.begin()+valid,uint64_t(target)/64)==s.pages.begin()+valid)
        return native;
    if(uint64_t(target)==native.line) return native;
    return {uint64_t(target),"frozen",true,false};
}
