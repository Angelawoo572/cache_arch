// A PC tag partitions the ingress streams. The predictor has no target identity or retrospective label
// input.
struct TurnState {
    uint64_t pc = 0, line = 0, time = 0;
    int64_t stride = 0, old_step = 0, old_jump = 0;
    unsigned run = 0, old_length = 0, repeats = 0;
    bool valid = false;
};

struct Forecast {
    int64_t delta;
    const char *source;
};

class DistanceRule {
    std::array<std::array<TurnState, 64>, NUM_CPUS> table{};

  public:
    enum Type { native, look2, paced, turn };
    Type type = native;

    Forecast observe(unsigned core, uint64_t pc, uint64_t line, uint64_t cycle, bool match,
                     int64_t native_delta, unsigned mshr, unsigned dram_rq) {
        auto &s = table[core][((pc >> 2) ^ (pc >> 9)) & 63];
        if (!s.valid || s.pc != pc) {
            s = {};
            s.pc = pc;
        }
        bool prior = s.valid;
        uint64_t gap = prior ? cycle - s.time : 0;
        int64_t d = prior ? int64_t(line) - int64_t(s.line) : 0;
        if (d && d == s.stride) {
            s.run = std::min(255u, s.run + 1);
        } else {
            if (d && s.run >= 3) {
                if (s.old_step == s.stride && s.old_length == s.run && s.old_jump == d) {
                    s.repeats = std::min(3u, s.repeats + 1);
                } else {
                    s.old_step = s.stride;
                    s.old_length = s.run;
                    s.old_jump = d;
                    s.repeats = 1;
                }
            }
            s.run = d ? 1 : 0;
            s.stride = d;
        }
        s.valid = true;
        s.line = line;
        s.time = cycle;
        auto legal = [&](int64_t delta) {
            auto a = int64_t(line) + delta;
            return delta && a > 0 && uint64_t(a) / 64 == line / 64;
        };
        if (type == turn && s.repeats >= 2 && s.stride == s.old_step && s.run <= s.old_length &&
            s.old_length - s.run <= 2) {
            int64_t turn_delta = (s.old_length - s.run) * s.old_step + s.old_jump;
            if (legal(turn_delta))
                return {turn_delta, "turn"};
        }
        if (!match)
            return {0, "silent"};
        if (type == look2)
            return {2 * native_delta, "look2"};
        if (type != native && s.run >= 4 && s.stride == native_delta && gap <= 128 && mshr < 16 &&
            dram_rq < 16 && legal(2 * native_delta))
            return {2 * native_delta, "paced"};
        return {native_delta, "stride"};
    }
};
