#!/usr/bin/env python3
"""Build the isolated A–C tracer from released Pythia v1.3 and libbf v1.0.0."""

import argparse
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess

SIM = Path(__file__).resolve().parent
ROOT = SIM.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cores", type=int, choices=(1, 2, 4, 8), default=2)
parser.add_argument("--output", type=Path, default=ROOT / "raw/build-2")
parser.add_argument(
    "--source",
    type=Path,
    help="optional local Git clone of Pythia v1.3 with libbf checkout",
)
parser.add_argument("--jobs", type=int, default=8)
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(exist_ok=True, parents=True)
source = out / "pythia"


def command(*argv, cwd=out):
    subprocess.run([str(arg) for arg in argv], cwd=cwd, check=True)


def obtain(path, repo, tag, cache):
    if not (path / ".git").is_dir():
        origin = str(cache) if cache and (cache / ".git").is_dir() else repo
        command("git", "clone", "--quiet", "--branch", tag, origin, path)


def original(path, name):
    return subprocess.check_output(["git", "-C", path, "show", "HEAD:" + name]).decode()


def put(path, data):
    if not path.exists() or path.read_text() != data:
        path.write_text(data)


def change(text, old, new):
    if text.count(old) != 1:
        raise ValueError("unexpected source anchor: " + old[:100])
    return text.replace(old, new)


def inject(text, anchor, addition):
    return change(text, anchor, addition + anchor)


fallback = (
    ROOT.parent / "llc_lstm_research/conventional/runs/build-v5-pressure-local/pythia"
)
cache = args.source or fallback
obtain(source, "https://github.com/CMU-SAFARI/Pythia.git", "v1.3", cache)
lib = source / "libbf"
obtain(lib, "https://github.com/mavam/libbf.git", "v1.0.0", cache / "libbf")
put(
    lib / "CMakeLists.txt",
    original(lib, "CMakeLists.txt").replace(
        "add_library(libbf SHARED", "add_library(libbf STATIC"
    ),
)
put(lib / "src/hash.cpp", "#include <stdexcept>\n" + original(lib, "src/hash.cpp"))
for dest, orig in [
    ("branch/branch_predictor.cc", "branch/perceptron.bpred"),
    ("prefetcher/l1d_prefetcher.cc", "prefetcher/no.l1d_pref"),
    ("prefetcher/l2c_prefetcher.cc", "prefetcher/no.l2c_pref"),
    ("prefetcher/llc_prefetcher.cc", "prefetcher/no.llc_pref"),
]:
    put(source / dest, original(source, orig))
put(
    source / "replacement/llc_replacement.cc",
    original(source, "replacement/ship.llc_repl").replace("    srand(time(NULL));", ""),
)
put(
    source / "inc/champsim.h",
    change(
        original(source, "inc/champsim.h"),
        "#define NUM_CPUS 1",
        f"#define NUM_CPUS {args.cores}",
    ),
)
put(
    source / "inc/block.h",
    inject(
        original(source, "inc/block.h"),
        "    uint32_t pf_metadata;",
        "    uint64_t hybrid_virtual_line = 0;\n",
    ),
)
put(source / "inc/hybrid.h", (SIM / "hybrid.h").read_text())
put(source / "src/hybrid.cc", (SIM / "hybrid.cc").read_text())
pref = '#include "hybrid.h"\n'
main = pref + original(source, "src/main.cc")
# Preserve the simulator's first-touch/TLB timing; lay out the same per-program pages across arms.
start = main.index("            uint8_t fragmented = 0;")
end = main.index("            // insert translation", start)
main = main[:start] + "            random_ppage = hybrid_ppn(cpu, va);\n" + main[end:]
start = main.index("            if (fragmented) {")
end = main.index("\n        }", start)
main = main[:start] + main[end:]
main = inject(
    main,
    "            // TODO: elaborate page replacement algorithm",
    '            throw std::runtime_error("physical page capacity exceeded");\n',
)
start = main.index('            printf("trace_%d %s')
end = main.index("            count_traces++;", start)
main = (
    main[:start]
    + """            if (count_traces >= static_cast<int>(hybrid_active()))
                throw std::runtime_error("too many traces");
            ooo_cpu[count_traces].trace_file = hybrid_open(argv[i], count_traces);
            cout << "trace_" << count_traces << " " << argv[i] << endl;
"""
    + main[end:]
)
main = inject(
    main,
    "    uint32_t seed_number = 0;",
    """    const char* active = std::getenv("HYBRID_ACTIVE_CORES");
    hybrid_init(active ? std::stoul(active) : NUM_CPUS, knob::warmup_instructions, knob::simulation_instructions);
""",
)
main = change(main, "uint32_t seed_number = 0;", "uint32_t seed_number = 447;")
main = main.replace(
    "if (count_traces != NUM_CPUS)",
    "if (count_traces != static_cast<int>(hybrid_active()))",
)
main = change(
    main,
    "    generator.seed(champsim_seed);",
    """    for (unsigned i = hybrid_active(); i < NUM_CPUS; ++i) {
        warmup_complete[i] = simulation_complete[i] = 1;
        ++all_warmup_complete;
        ++all_simulation_complete;
    }
    generator.seed(champsim_seed);""",
)
main = change(
    main,
    "if (stall_cycle[i] <= current_core_cycle[i])",
    "if (i < static_cast<int>(hybrid_active()) && stall_cycle[i] <= current_core_cycle[i])",
)
main = change(
    main,
    "ooo_cpu[i].num_retired > knob::warmup_instructions",
    "ooo_cpu[i].num_retired >= knob::warmup_instructions",
)
start = main.index("            if (all_warmup_complete == NUM_CPUS)")
end = main.index("\n            }", start) + len("\n            }")
main = main[:start] + main[end:]
main = inject(
    main,
    "        uncore.LLC.operate();",
    """        if (all_warmup_complete == NUM_CPUS) {
            ++all_warmup_complete;
            finish_warmup();
        }
        hybrid_bank_cycle();
""",
)
main = inject(
    main,
    "            // core might be stalled",
    "            hybrid_head(ooo_cpu[i]);\n",
)
main = inject(
    main, "    uncore.LLC.llc_prefetcher_final_stats();\n\n", "    hybrid_finish();\n"
)
put(source / "src/main.cc", main)
ooo = pref + original(source, "src/ooo_cpu.cc")
ooo = change(
    ooo,
    "while (continue_reading) {",
    "while (continue_reading && hybrid_may_fetch(*this)) {",
)
ooo = re.sub(
    r"                // reached end of file for this trace\n.*?\n                }\n",
    '                throw std::runtime_error("trace exhausted before fixed instruction boundary");\n',
    ooo,
    flags=re.S,
)
ooo = inject(
    ooo,
    "    ROB.entry[index] = *arch_instr;",
    "    hybrid_observe_instruction(cpu, *arch_instr);\n",
)
ooo = inject(
    ooo,
    "    data_packet.full_addr = LQ.entry[lq_index].physical_address;",
    "    data_packet.hybrid_virtual_line = LQ.entry[lq_index].virtual_address >> LOG2_BLOCK_SIZE;\n",
)
ooo = inject(
    ooo,
    "        // release ROB entry",
    "        hybrid_retire(cpu, ROB.entry[ROB.head].instr_id);\n",
)
put(source / "src/ooo_cpu.cc", ooo)
cache_cc = pref + original(source, "src/cache.cc")
for marker, commandline in [
    (
        "                // update prefetch stats and reset prefetch bit",
        'hybrid_cache(*this, RQ.entry[index], "lookup", 1);',
    ),
    (
        "                if (miss_handled) \n                {\n                    // update prefetcher on load instruction",
        "",
    ),
    (
        "                            // RBERA: add late prefetch stats here",
        'if (cache_type == IS_LLC) hybrid_terminal(MSHR.entry[mshr_index], "late", &RQ.entry[index]);',
    ),
    (
        "                                lower_level->add_rq(&RQ.entry[index]);",
        "hybrid_dispatch(RQ.entry[index]);",
    ),
    (
        "                                lower_level->add_rq(&PQ.entry[index]); // add it to the DRAM RQ",
        "hybrid_dispatch(PQ.entry[index]);",
    ),
    (
        "                    llc_prefetcher_prefetch_hit(block[set][way].address",
        'hybrid_terminal(PQ.entry[index], "redundant");',
    ),
    (
        "                        // no need to update request except fill_level",
        'if (cache_type == IS_LLC) hybrid_terminal(PQ.entry[index], "redundant");',
    ),
    (
        "    if (block[set][way].prefetch && (block[set][way].used == 0))",
        "if (cache_type == IS_LLC) hybrid_fill(*this, *packet, block[set][way]);",
    ),
    (
        "        WQ.FORWARD++;\n        RQ.ACCESS++;",
        'hybrid_cache(*this, *packet, "forwarded");',
    ),
    (
        "        RQ.MERGED++;\n        RQ.ACCESS++;",
        'hybrid_cache(*this, *packet, "merged");',
    ),
    (
        "        RQ.FULL++;\n\n        return -2;",
        'hybrid_cache(*this, *packet, "full");',
    ),
    (
        "    RQ.TO_CACHE++;\n    RQ.ACCESS++;",
        "if (cache_type == IS_LLC) hybrid_ingress(*this, RQ.entry[index]);",
    ),
]:
    if commandline:
        cache_cc = inject(
            cache_cc,
            marker,
            " " * (len(marker) - len(marker.lstrip())) + commandline + "\n",
        )
    else:
        cache_cc = change(
            cache_cc,
            marker,
            marker.replace(
                "// update prefetcher",
                'hybrid_cache(*this, RQ.entry[index], "lookup", 0);\n                    // update prefetcher',
            ),
        )
# Discriminate a merge into an existing MSHR from primary misses.
cache_cc = (
    inject(
        cache_cc,
        "                    // mark merged consumer",
        '                    hybrid_cache(*this, RQ.entry[index], "mshr_merge");\n',
    )
    if cache_cc.count("                    // mark merged consumer") == 1
    else cache_cc
)
put(source / "src/cache.cc", cache_cc)
stride = pref + original(source, "prefetcher/stride.cc")
stride = change(
    stride, "Prefetcher(type)\n{\n\n}", "Prefetcher(type)\n{\n   init_stats();\n}"
)
stride = inject(
    stride, "      stats.tracker.insert++;", '      hybrid_stride_state("cold", 0);\n'
)
stride = inject(
    stride,
    "      stats.stride.zero++;",
    '      hybrid_stride_state("zero_delta", 0);\n',
)
stride = inject(
    stride,
    "   if(stride == tracker->last_stride)",
    '   hybrid_stride_state(stride == tracker->last_stride ? "match" : "delta_changed", stride);\n',
)
put(source / "prefetcher/stride.cc", stride)
dram = pref + original(source, "src/dram_controller.cc")
dram = inject(
    dram,
    "            // remove the oldest entry\n            queue->remove_queue",
    "            hybrid_transfer(queue->entry[request_index].type, queue->is_WQ);\n",
)
put(source / "src/dram_controller.cc", dram)
make = original(source, "Makefile").replace(
    "$(objects) $(LDFlags) -o $@", "$(objects) $(LDFlags) -lz -o $@"
)
put(source / "Makefile", make + "\n-include $(objects:.o=.d)\n")
cc = (
    os.getenv("CXX")
    or (shutil.which("g++-15") if platform.system() == "Darwin" else None)
    or "g++"
)
command(
    "cmake",
    "-DCMAKE_CXX_COMPILER=" + cc,
    "-S",
    lib,
    "-B",
    lib / "build",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
)
command("cmake", "--build", lib / "build", "--parallel", args.jobs)
command("make", "buildrepo", cwd=source)
command("make", "-j" + str(args.jobs), "CXX=" + cc, cwd=source)
print(source / "bin/champsim")
