#!/usr/bin/env python3
"""Install narrowly anchored hooks into an isolated copy of active ChampSim.

This operates only on the caller's explicitly supplied staging directory. It
never resets/cleans Git, touches a parent checkout, or stops a process.
"""
from pathlib import Path
import argparse
import shutil

EXP = Path(__file__).resolve().parents[1]


def replace_once(text, old, new, name):
    if text.count(old) != 1:
        raise RuntimeError(f"{name}: expected exactly one source anchor, found {text.count(old)}")
    return text.replace(old, new, 1)


def patch_sim(sim):
    marker = sim / ".dynamic602_hooks_installed"
    if marker.exists():
        return
    texts = {p: (sim / p).read_text() for p in (
        "src/main.cc", "src/cache.cc", "src/ooo_cpu.cc", "inc/block.h",
        "inc/stride.h", "prefetcher/stride.cc")}
    main = texts["src/main.cc"]
    main = '#include "dynamic602_hooks.h"\n' + main
    main = replace_once(main, "            count_traces++;", "            dynamic602_skip(ooo_cpu[count_traces].trace_file, knob::knob_cloudsuite, count_traces);\n\n            count_traces++;", "reader skip")
    main = replace_once(main, "    uncore.LLC.LATENCY = LLC_LATENCY;", "    uncore.LLC.LATENCY = LLC_LATENCY;\n    for (unsigned dynamic602_cpu=0; dynamic602_cpu<NUM_CPUS; ++dynamic602_cpu)\n        dynamic602_begin(dynamic602_cpu);", "measurement start")
    main = replace_once(main, "    uint8_t run_simulation = 1;", """    // Cold suffix mode starts with real latencies and all startup costs included.
    // No instruction, cache operation, or predictor callback has executed yet.
    if (knob::warmup_instructions == 0) {
        all_warmup_complete = NUM_CPUS + 1;
        for (unsigned dynamic602_cpu=0; dynamic602_cpu<NUM_CPUS; ++dynamic602_cpu)
            warmup_complete[dynamic602_cpu] = 1;
        finish_warmup();
    }
    uint8_t run_simulation = 1;""", "cold start")
    main = replace_once(main, "            current_core_cycle[i]++;", "            current_core_cycle[i]++;\n            dynamic602_service(&ooo_cpu[i].L2C);", "unconditional cycle hook")
    main = replace_once(main, "                ooo_cpu[i].finish_sim_cycle = current_core_cycle[i] - ooo_cpu[i].begin_sim_cycle;", "                ooo_cpu[i].finish_sim_cycle = current_core_cycle[i] - ooo_cpu[i].begin_sim_cycle;\n                dynamic602_observe_cycle(i, true);", "final measurement")
    texts["src/main.cc"] = main

    cpu = '#include "dynamic602_hooks.h"\n#include <stdexcept>\n' + texts["src/ooo_cpu.cc"]
    cpu = replace_once(cpu, "    while (continue_reading) {", "    while (continue_reading) {\n        if (!dynamic602_read_allowed(this->cpu)) return;", "bounded input reader")
    for target in ("current_cloudsuite_instr", "current_instr"):
        anchor = f"            if (!fread(&{target}, instr_size, 1, trace_file)) {{"
        cpu = replace_once(cpu, anchor, anchor + '\n                if (dynamic602_bounded_trace()) throw std::runtime_error("trace ended before selected record budget; repetition forbidden");', "bounded EOF")
    anchor = "            } else { // successfully read the trace\n"
    if cpu.count(anchor) != 2:
        raise RuntimeError("expected both actual trace readers")
    cpu = cpu.replace(anchor, anchor + "                dynamic602_record_read(this->cpu);\n")
    cpu = replace_once(cpu, "        num_retired++;", "        num_retired++;\n        dynamic602_observe_cycle(cpu);", "exact retirement sampling")
    texts["src/ooo_cpu.cc"] = cpu

    cache = '#include "dynamic602_hooks.h"\n' + texts["src/cache.cc"]
    cache = replace_once(cache, "void CACHE::fill_cache(uint32_t set, uint32_t way, PACKET *packet)\n{", "void CACHE::fill_cache(uint32_t set, uint32_t way, PACKET *packet)\n{\n    dynamic602_fill(this, set, way, packet);", "valid line fill")
    cache = replace_once(cache, "            block[set][way].valid = 0;", "            dynamic602_invalidate(this, set, way);\n            block[set][way].valid = 0;", "valid line invalidation")
    cache = replace_once(cache, "                // update prefetch stats and reset prefetch bit", "                dynamic602_demand_hit(this, set, way);\n                // update prefetch stats and reset prefetch bit", "first demand")
    cache = replace_once(cache, "                            // RBERA: add late prefetch stats here", "                            dynamic602_late(this, &MSHR.entry[mshr_index]);\n                            // RBERA: add late prefetch stats here", "late demand")
    cache = replace_once(cache, "            if (way >= 0) // prefetch hit\n            {", "            if (way >= 0) // prefetch hit\n            {\n                dynamic602_redundant(this, &PQ.entry[index], true);", "prefetch resident duplicate")
    cache = replace_once(cache, "                        MSHR_MERGED[PQ.entry[index].type]++;", "                        dynamic602_redundant(this, &PQ.entry[index], false);\n                        MSHR_MERGED[PQ.entry[index].type]++;", "prefetch inflight duplicate")
    cache = replace_once(cache, "    PQ.entry[index] = *packet;", "    dynamic602_enqueue(this, packet);\n    PQ.entry[index] = *packet;", "prefetch enqueue identity")
    texts["src/cache.cc"] = cache

    block = texts["inc/block.h"]
    block = replace_once(block, "    uint32_t pf_metadata;", "    uint32_t pf_metadata;\n    uint64_t dynamic602_measurement_id; // measurement-only identity; not a predictor feature", "packet identity")
    block = replace_once(block, "    PACKET() {", "    PACKET() {\n        dynamic602_measurement_id = 0;", "packet identity initialization")
    texts["inc/block.h"] = block
    stride = texts["prefetcher/stride.cc"]
    stride = replace_once(stride, "StridePrefetcher::StridePrefetcher(string type) : Prefetcher(type)\n{\n\n}", "StridePrefetcher::StridePrefetcher(string type) : Prefetcher(type)\n{\n   init_stats(); // Initialize reporting counters; prediction algorithm is unchanged.\n}", "Stride counter initialization")
    texts["prefetcher/stride.cc"] = stride
    texts["inc/stride.h"] = replace_once(texts["inc/stride.h"], "   ~StridePrefetcher();", "   ~StridePrefetcher();\n   size_t dynamic602_tracker_count() const { return trackers.size(); }", "Stride storage observation")
    # Validate every anchor before writing any simulator source.
    for path, text in texts.items():
        (sim / path).write_text(text)
    marker.write_text("dynamic602 isolated simulator hooks v1\n")


def install(sim):
    sim = Path(sim).resolve()
    if not (sim / "src/cache.cc").is_file():
        raise RuntimeError(f"not a ChampSim source directory: {sim}")
    if sim.name not in {"simulator", "sacramento_sim", "sim_test"} and not (sim / ".dynamic602_isolated").exists():
        raise RuntimeError("create .dynamic602_isolated in an intentional isolated simulator copy first")
    patch_sim(sim)
    for name in ("dynamic602_hooks.h", "dynamic_observer_core.h"):
        shutil.copy2(EXP / "measurement" / name, sim / "inc" / name)
    shutil.copy2(EXP / "measurement/dynamic_observer.cc", sim / "src/dynamic_observer.cc")
    for name in ("dynamic602_engine.h", "dynamic602_engine.cc"):
        source = EXP / "runtime" / name
        if source.exists():
            shutil.copy2(source, sim / ("inc" if name.endswith(".h") else "prefetcher") / name)
    registry = (sim / "prefetcher/multi.l2c_pref").read_text()
    registry = replace_once(registry, '#include "stride.h"', '#include "stride.h"\n#include "dynamic602_engine.h"', "dynamic registry include")
    registry = replace_once(registry, '\t\telse if(!knob::l2c_prefetcher_types[index].compare("stride"))', '''\t\telse if(!knob::l2c_prefetcher_types[index].compare("dynamic602"))
\t\t{
\t\t\tDynamic602 *pref_dynamic602 = new Dynamic602(knob::l2c_prefetcher_types[index], this);
\t\t\tprefetchers.push_back(pref_dynamic602);
\t\t}
\t\telse if(!knob::l2c_prefetcher_types[index].compare("stride"))''', "dynamic registry")
    # Reuse the original keyed replay class for measured summer references.
    # Only the registry changes: its warmup gate and PC-line-occ behavior stay intact.
    if '#include "list_replayer.h"' not in registry:
        registry = replace_once(registry, '#include "stride.h"', '#include "stride.h"\n#include "list_replayer.h"', "original replay include")
    if 'compare("list_replayer")' not in registry:
        registry = replace_once(registry, '\t\telse if(!knob::l2c_prefetcher_types[index].compare("stride"))', '\t\telse if(!knob::l2c_prefetcher_types[index].compare("list_replayer"))\n\t\t{\n\t\t\tListReplayer *pref_list = new ListReplayer(knob::l2c_prefetcher_types[index], this);\n\t\t\tprefetchers.push_back(pref_list);\n\t\t}\n\t\telse if(!knob::l2c_prefetcher_types[index].compare("stride"))', "original replay registry")
    (sim / "prefetcher/dynamic602.l2c_pref").write_text(registry)
    print(f"Installed dynamic602 hooks and registry in {sim}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simulator", type=Path)
    install(parser.parse_args().simulator)
