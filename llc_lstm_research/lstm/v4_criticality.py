#!/usr/bin/env python3
"""Add offline retirement-head load observations to an isolated simulator copy."""

import argparse
from pathlib import Path


OBSERVATIONS = '''
namespace {
struct HeadLoadObservation {
    uint64_t wait_cycles[NUM_INSTR_SOURCES] = {};
};

HeadLoadObservation head_load_observations[NUM_CPUS];

std::ofstream open_head_load_log()
{
    const char* path = std::getenv("LLC_PF_ROB_STALLS");
    if (!path || !*path)
        return {};
    std::ofstream stream(path);
    if (!stream)
        throw std::runtime_error("Cannot open LLC_PF_ROB_STALLS");
    stream << "cpu,instr_id,operand,head_wait_cycles\\n";
    return stream;
}

std::ofstream& head_load_log()
{
    static auto stream = open_head_load_log();
    return stream;
}

bool observe_head_load(uint32_t cpu)
{
    return head_load_log().is_open() && all_warmup_complete > NUM_CPUS
           && !simulation_complete[cpu];
}
}

void record_head_load_wait(const O3_CPU& core)
{
    if (!observe_head_load(core.cpu))
        return;
    const auto& instruction = core.ROB.entry[core.ROB.head];
    if (!instruction.ip || instruction.executed == COMPLETED)
        return;
    auto& observation = head_load_observations[core.cpu];
    for (uint32_t operand = 0; operand < NUM_INSTR_SOURCES; ++operand) {
        const uint32_t index = instruction.lq_index[operand];
        if (index >= core.LQ.SIZE)
            continue;
        const auto& load = core.LQ.entry[index];
        if (load.instr_id == instruction.instr_id && load.data_index == operand
            && load.fetched == INFLIGHT)
            ++observation.wait_cycles[operand];
    }
}
'''

RETIRE = '''
        if (observe_head_load(cpu)) {
            auto& observation = head_load_observations[cpu];
            for (uint32_t operand = 0; operand < NUM_INSTR_SOURCES; ++operand) {
                const uint64_t wait = observation.wait_cycles[operand];
                if (wait)
                    head_load_log() << cpu << ',' << ROB.entry[ROB.head].instr_id << ','
                                    << operand << ',' << wait << '\\n';
            }
            observation = {};
        }
'''


def replace_once(source, anchor, replacement):
    if source.count(anchor) != 1:
        raise ValueError(f'Expected one simulator source anchor {anchor!r}')
    return source.replace(anchor, replacement, 1)


def patch(simulator):
    path = simulator / 'src' / 'ooo_cpu.cc'
    source = path.read_text()
    main_path = simulator / 'src' / 'main.cc'
    main = main_path.read_text()
    if 'record_head_load_wait(const O3_CPU& core)' in source:
        return
    include = '#include "set.h"\n'
    source = replace_once(source, include,
                          include + '#include <cstdlib>\n#include <fstream>\n#include <stdexcept>\n')
    start = 'void O3_CPU::retire_rob()\n{'
    source = replace_once(source, start, OBSERVATIONS + '\n' + start)
    release = '        // release ROB entry\n'
    source = replace_once(source, release, RETIRE + '\n' + release)
    declare = 'std::default_random_engine generator;\n'
    main = replace_once(main, declare, 'void record_head_load_wait(const O3_CPU& core);\n\n' + declare)
    retire = '                // retire\n'
    main = replace_once(main, retire, retire + '                record_head_load_wait(ooo_cpu[i]);\n')
    path.write_text(source)
    main_path.write_text(main)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('isolated_simulator', type=Path)
    patch(parser.parse_args().isolated_simulator)
