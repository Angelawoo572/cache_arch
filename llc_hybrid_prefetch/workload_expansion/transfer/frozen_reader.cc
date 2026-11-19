#include <cstdint>
#include <iostream>
#include "transition_rule.hh"

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    boundary_forecast::Engine exact(argv[1]), page(argv[2]), prior(argv[3]);
    std::uint64_t pc, line, cycle, index = 0;
    while (std::cin >> pc >> line >> cycle) {
        const auto a = exact.step(pc, line, cycle);
        const auto b = page.step(pc, line, cycle);
        const auto c = prior.step(pc, line, cycle);
        if (boundary_forecast::sample(index))
            std::cout << index << ' ' << a << ' ' << b << ' ' << c << '\n';
        ++index;
    }
}
