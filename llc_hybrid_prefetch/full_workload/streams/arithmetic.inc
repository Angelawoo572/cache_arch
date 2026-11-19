// One optional arithmetic candidate on the same sparse grant as frozen-rule replay.
struct ArithmeticSlot {
    uint64_t pc = 0, line = 0;
    int64_t delta = 0;
};
struct ArithmeticState {
    std::array<ArithmeticSlot,64> table{};
    std::array<uint64_t,8> recent{};
    uint64_t count = 0;
};
std::array<ArithmeticState,NUM_CPUS> arithmetic_state{};
std::string arithmetic_mode;

unsigned arithmetic_slot(uint64_t pc) {
    return unsigned(((pc >> 2) ^ (pc >> 8)) & 63);
}
Proposal arithmetic_offer(const PACKET &p, Proposal native) {
    auto &s = arithmetic_state[p.cpu];
    auto &own = s.table[arithmetic_slot(p.ip)];
    const int64_t increment = own.pc == p.ip ? int64_t(p.address) - int64_t(own.line) : 0;
    own.pc = p.ip;
    own.line = p.address;
    own.delta = increment;
    const auto ordinal = s.count++;
    uint64_t other_pc = 0;
    for (unsigned age=1; age<=std::min<uint64_t>(ordinal,8); ++age) {
        const auto pc = s.recent[(ordinal-age) % 8];
        if (pc != p.ip) { other_pc = pc; break; }
    }
    s.recent[ordinal % 8] = p.ip;
    if ((ordinal % 8) != 3) return native;
    ++diagnostics["arithmetic_grants"];
    int64_t base = p.address, delta = 0;
    if (arithmetic_mode == "local") delta = increment;
    else if (arithmetic_mode == "plus1") delta = 1;
    else if (arithmetic_mode == "plus3") delta = 3;
    else {
        const auto &other = s.table[arithmetic_slot(other_pc)];
        if (!other_pc || other.pc != other_pc) return native;
        delta = other.delta;
        if (arithmetic_mode == "other_continuation") base = other.line;
    }
    const int64_t target = base + delta;
    if (!delta || target <= 0 || target == int64_t(p.address) || (target >> 6) != (base >> 6)) return native;
    if (uint64_t(target) == native.line) {
        ++diagnostics["arithmetic_agrees"];
        return native;
    }
    ++diagnostics[native.line ? "arithmetic_displaces" : "arithmetic_extra"];
    return {uint64_t(target),"arithmetic",true,false};
}
