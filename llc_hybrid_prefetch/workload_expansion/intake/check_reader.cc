#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include "instruction.h"

// Use the existing local simulator header for a second, native read of the input.
static_assert(sizeof(input_instr) == 64, "input instruction width");
static_assert(offsetof(input_instr, ip) == 0 && offsetof(input_instr, is_branch) == 8 &&
              offsetof(input_instr, branch_taken) == 9, "instruction and branch offsets");
static_assert(offsetof(input_instr, destination_registers) == 10 && offsetof(input_instr, source_registers) == 12,
              "register offsets");
static_assert(offsetof(input_instr, destination_memory) == 16 && offsetof(input_instr, source_memory) == 32,
              "memory operand offsets");

struct Count {
    unsigned instructions = 0;
    unsigned branch = 0;
    unsigned taken = 0;
    unsigned loads = 0;
    unsigned stores = 0;
    unsigned load_instructions = 0;
    unsigned load_with_prior_writer = 0;
    unsigned load_with_prior_load_writer = 0;
    unsigned readers_of_prior_load = 0;
};

int main(int argc, char **argv) {
    if (argc != 5) return 2;
    const int start = std::stoi(argv[2]), end = std::stoi(argv[3]), expected = std::stoi(argv[4]);
    if (start < 0 || end <= start || expected < end) return 2;
    std::ifstream file(argv[1], std::ios::binary);
    if (!file) return 3;
    std::array<int, 256> last_writer;
    last_writer.fill(-1000);
    std::array<bool, 256> from_load{};
    Count c{};
    input_instr item;
    int position = 0;
    while (file.read(reinterpret_cast<char *>(&item), sizeof(item))) {
        unsigned loads = 0, stores = 0;
        for (const auto a : item.source_memory) loads += a != 0;
        for (const auto a : item.destination_memory) stores += a != 0;
        bool potential_input = false, potential_load_input = false;
        for (const auto reg : item.source_registers) {
            if (reg == 0 || position - last_writer[reg] > 256) continue;
            potential_input = true;
            potential_load_input |= from_load[reg];
        }
        for (const auto reg : item.destination_registers) {
            if (!reg) continue;
            last_writer[reg] = position;
            from_load[reg] = loads != 0;
        }
        if (position >= start && position < end) {
            c.instructions++;
            c.branch += item.is_branch;
            c.taken += item.branch_taken;
            c.loads += loads;
            c.stores += stores;
            c.load_instructions += loads != 0;
            c.load_with_prior_writer += loads && potential_input;
            c.load_with_prior_load_writer += loads && potential_load_input;
            c.readers_of_prior_load += potential_load_input;
        }
        ++position;
    }
    if (!file.eof() || file.gcount() != 0 || position != expected) return 4;
    std::cout << c.instructions << "," << c.branch << "," << c.taken << ","
                  << c.loads << "," << c.stores << "," << c.load_instructions << ","
                  << c.load_with_prior_writer << "," << c.load_with_prior_load_writer << ","
                  << c.readers_of_prior_load << "\n";
}
