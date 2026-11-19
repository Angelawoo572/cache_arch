#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
constexpr unsigned NUM_CPUS = 2;
#include "rules.hh"

struct Input {
    unsigned pc, line, cycle, pressure;
};
using Output = std::pair<long long, std::string>;
std::vector<Output> evaluate(DistanceRule::Type mode, const std::vector<Input> &stream) {
    DistanceRule rule;
    rule.type = mode;
    std::vector<Output> out;
    for (auto e : stream) {
        auto f = rule.observe(0, e.pc, e.line, e.cycle, true, 1, e.pressure, 1);
        out.emplace_back(f.delta, f.source);
    }
    return out;
}
int main() {
    std::vector<Input> a;
    for (unsigned i = 0; i < 26; ++i)
        a.push_back({4, 128 + (i % 8 < 6 ? i % 8 : (i % 8) + 3), i * 12u, 0});
    auto prefix = a.size();
    auto b = a;
    for (unsigned i = 1; i < 160; ++i) {
        a.push_back({4, 128 + i, 400 + i * 10, 0});
        b.push_back({i % 8, 7519 + (i * 1781) % 17119, 700 + i * 23, 33});
    }
    for (auto mode : {DistanceRule::native, DistanceRule::look2, DistanceRule::paced, DistanceRule::turn}) {
        auto x = evaluate(mode, a), y = evaluate(mode, b);
        assert(std::equal(x.begin(), x.begin() + prefix, y.begin()));
    }
    auto x = evaluate(DistanceRule::paced, {{4, 128, 0, 0},
                                            {4, 129, 10, 0},
                                            {4, 130, 20, 0},
                                            {4, 131, 30, 0},
                                            {4, 132, 40, 0},
                                            {4, 133, 50, 16}});
    assert(x[4].second == "paced" && x[5].second == "stride");
}
