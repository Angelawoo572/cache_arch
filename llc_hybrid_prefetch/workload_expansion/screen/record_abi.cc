// Check the actual Pythia input type compiled for the simulator used by this screen.
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include "instruction.h"

static_assert(sizeof(input_instr) == 64, "trace record width");
static_assert(offsetof(input_instr, ip) == 0, "PC offset");
static_assert(offsetof(input_instr, is_branch) == 8, "branch offset");
static_assert(offsetof(input_instr, branch_taken) == 9, "taken offset");
static_assert(offsetof(input_instr, destination_registers) == 10, "destination registers");
static_assert(offsetof(input_instr, source_registers) == 12, "source registers");
static_assert(offsetof(input_instr, destination_memory) == 16, "destination memory addresses");
static_assert(offsetof(input_instr, source_memory) == 32, "source memory addresses");

int main() {
    std::uint16_t pattern = 1;
    if (*reinterpret_cast<const std::uint8_t*>(&pattern) != 1) return 1;
    std::cout << "64-byte little-endian instruction record and all seven offsets verified\n";
}
