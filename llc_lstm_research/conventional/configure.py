"""Adapt the Pythia v1.3 simulator source downloaded by build.sh."""
import argparse
from pathlib import Path
import re
import subprocess


def write_changed(path, content):
    if path.read_text() != content:
        path.write_text(content)


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Expected exactly one occurrence of {!r}".format(old))
    return text.replace(old, new, 1)


def add_before(text, marker, snippet):
    return replace_once(text, marker, snippet + marker)


def patch_cache(text):

    # Count demand arrivals that never receive their own LLC tag lookup.
    for marker, outcome in (
        ("        WQ.FORWARD++;\n        RQ.ACCESS++;", "forwarded"),
        ("        RQ.MERGED++;\n        RQ.ACCESS++;", "merged"),
        ("        RQ.FULL++;\n\n        return -2;", "full"),
    ):
        text = add_before(
            text, marker,
            "        if (cache_type == IS_LLC)\n"
            "            llc_pf_rq_disposition(*packet, LlcRqDisposition::{});\n\n".format(outcome),
        )
    text = add_before(
        text,
        "    RQ.TO_CACHE++;\n    RQ.ACCESS++;",
        "    if (cache_type == IS_LLC)\n"
        "        llc_pf_ingress(RQ.entry[index]);\n\n",
    )
    text = add_before(
        text,
        "    if (PQ.occupancy && (reads_available_this_cycle > 0))",
        "    if (cache_type == IS_LLC)\n"
        "        llc_pf_tick();\n\n",
    )
    text = add_before(
        text,
        "                // update prefetch stats and reset prefetch bit",
        "                if (cache_type == IS_LLC)\n"
        "                    llc_pf_demand(RQ.entry[index], true);\n\n",
    )
    text = add_before(
        text,
        "                            // RBERA: add late prefetch stats here",
        "                            if (cache_type == IS_LLC)\n"
        "                                llc_pf_late(MSHR.entry[mshr_index], RQ.entry[index]);\n",
    )
    text = add_before(
        text,
        "                    // update prefetcher on load instruction",
        "                    if (cache_type == IS_LLC)\n"
        "                        llc_pf_demand(RQ.entry[index], false);\n\n",
    )
    text = add_before(
        text,
        "    if (block[set][way].prefetch && (block[set][way].used == 0))",
        "    if (cache_type == IS_LLC)\n"
        "        llc_pf_fill(*packet, block[set][way]);\n\n",
    )
    text = add_before(
        text,
        "                    llc_prefetcher_prefetch_hit(block[set][way].address",
        "                    llc_pf_redundant(PQ.entry[index]);\n",
    )
    text = add_before(
        text,
        "                        // no need to update request except fill_level",
        "                        if (cache_type == IS_LLC)\n"
        "                            llc_pf_redundant(PQ.entry[index]);\n",
    )

    text = add_before(
        text,
        "            block[set][way].valid = 0;\n\n            match_way = way;",
        "            if (cache_type == IS_LLC)\n"
        "                llc_pf_invalidate(inval_addr);\n",
    )
    return add_before(
        text,
        "            // update replacement policy\n"
        "            if (cache_type == IS_LLC)\n"
        "            {\n"
        "                llc_update_replacement_state(fill_cpu, set, way, MSHR.entry[mshr_index].full_addr, MSHR.entry[mshr_index].ip, 0,",
        "            llc_pf_redundant(MSHR.entry[mshr_index]);\n",
    )


def patch_main(text):
    text = add_before(
        text,
        '            printf("trace_%d %s\\n", count_traces, argv[i]);',
        '            if (count_traces >= NUM_CPUS) {\n'
        '                cerr << "Too many traces for the configured core count" << endl;\n'
        '                return EXIT_FAILURE;\n'
        '            }\n',
    )
    return add_before(
        text,
        "            ooo_cpu[count_traces].trace_file = popen",
        "            const auto skip = llc_pf_trace_skip();\n"
        "            if (skip) {\n"
        "                const auto width = knob::knob_cloudsuite ? sizeof(cloudsuite_instr) : sizeof(input_instr);\n"
        "                if (skip > (UINT64_MAX - 1) / width) {\n"
        '                    cerr << "Trace skip is too large" << endl;\n'
        "                    return EXIT_FAILURE;\n"
        "                }\n"
        "                auto &command = ooo_cpu[count_traces].gunzip_command;\n"
        "                const auto length = strlen(command);\n"
        '                const auto tail_length = snprintf(command + length, sizeof(command) - length,\n'
        '                    " | tail -c +%llu", static_cast<unsigned long long>(skip * width + 1));\n'
        "                if (tail_length < 0 || static_cast<size_t>(tail_length) >= sizeof(command) - length) {\n"
        '                    cerr << "Trace command is too long" << endl;\n'
        "                    return EXIT_FAILURE;\n"
        "                }\n"
        "            }\n",
    )


def patch_eof(text):
    text, count = re.subn(
        r"                // reached end of file for this trace\n.*?\n                }\n",
        '                cerr << "Trace exhausted before the simulation completed for core " << cpu\n'
        '                     << ": " << trace_string << endl;\n'
        "                exit(EXIT_FAILURE);\n",
        text,
        flags=re.DOTALL,
    )
    if count != 2:
        raise ValueError("Expected both Pythia trace formats to have an EOF handler")
    return text


def patch_ooo(text):
    return add_before(
        patch_eof(text),
        "    data_packet.full_addr = LQ.entry[lq_index].physical_address;",
        "    data_packet.llc_pf_virtual_line = LQ.entry[lq_index].virtual_address >> LOG2_BLOCK_SIZE;\n",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("cores", type=int, choices=(1, 2, 4, 8))
    args = parser.parse_args()
    source = args.source
    here = Path(__file__).resolve().parent

    def put(path, content):
        target = source / path
        if target.exists():
            write_changed(target, content)
        else:
            target.write_text(content)

    # Always transform pristine tagged inputs. This supports repeated builds
    # and catches mismatches between the hooks and the upstream release.
    def patch(path, change, repo=source):
        pristine = subprocess.check_output(
            ["git", "-C", str(repo), "show", "HEAD:" + path],
            universal_newlines=True,
        )
        put(str(repo.relative_to(source) / path), change(pristine))

    def copy(source_name, target_name):
        put(target_name, (source / source_name).read_text())

    patch("inc/champsim.h", lambda content: add_before(
        replace_once(content, "#define NUM_CPUS 1", "#define NUM_CPUS {}".format(args.cores)),
        "void print_stats();",
        "uint64_t llc_pf_trace_skip();\n"
        "void llc_pf_dram_transfer(uint8_t type, bool write);\n",
    ))
    patch("inc/cache.h", lambda content: add_before(
        add_before(
            content, "class CACHE : public MEMORY {",
            "enum class LlcRqDisposition { forwarded, merged, full };\n\n",
        ),
        "    uint32_t cpu;",
        "    void llc_pf_ingress(PACKET &packet);\n"
        "    void llc_pf_tick();\n"
        "    bool llc_pf_eligible(uint64_t base, uint64_t target, uint32_t core);\n"
        "    bool llc_pf_try_issue(uint64_t base, uint64_t target, uint64_t pc, uint32_t core);\n"
        "    void llc_pf_rq_disposition(const PACKET &packet, LlcRqDisposition disposition);\n"
        "    void llc_pf_invalidate(uint64_t address);\n"
        "    void llc_pf_demand(const PACKET &packet, bool hit);\n"
        "    void llc_pf_late(const PACKET &prefetch, PACKET &demand);\n"
        "    void llc_pf_redundant(const PACKET &packet);\n"
        "    void llc_pf_fill(const PACKET &packet, const BLOCK &victim);\n\n",
    ))
    patch("inc/block.h", lambda content: add_before(
        content, "    uint32_t pf_metadata;",
        "    uint64_t llc_pf_event_id = 0;\n"
        "    uint64_t llc_pf_ingress_cycle = 0;\n"
        "    uint64_t llc_pf_virtual_line = 0;\n"
        "    uint32_t llc_pf_mshr_occupancy = 0;\n"
        "    bool llc_pf_late_use = false;\n",
    ))
    patch("src/cache.cc", patch_cache)
    patch("src/main.cc", patch_main)
    patch("src/ooo_cpu.cc", patch_ooo)

    patch("prefetcher/stride.cc", lambda content: replace_once(
        content,
        "StridePrefetcher::StridePrefetcher(string type) : Prefetcher(type)\n{\n\n}",
        "StridePrefetcher::StridePrefetcher(string type) : Prefetcher(type)\n{\n   init_stats();\n}",
    ))
    patch("src/dram_controller.cc", lambda content: add_before(
        content,
        "            // remove the oldest entry\n            queue->remove_queue",
        "            llc_pf_dram_transfer(queue->entry[request_index].type, queue->is_WQ);\n\n",
    ))

    copy("branch/perceptron.bpred", "branch/branch_predictor.cc")
    copy("prefetcher/no.l1d_pref", "prefetcher/l1d_prefetcher.cc")
    copy("prefetcher/no.l2c_pref", "prefetcher/l2c_prefetcher.cc")
    replacement = (source / "replacement/ship.llc_repl").read_text()
    put("replacement/llc_replacement.cc", replace_once(replacement, "    srand(time(NULL));\n", ""))
    put("prefetcher/llc_prefetcher.cc", (here / "llc_prefetcher.cc").read_text())
    put("inc/policies.h", (here / "policies.h").read_text())
    put("inc/learned_llc.h", (here / "learned_llc.h").read_text())
    # Upstream's Makefile names dependency files but never includes them.
    patch("Makefile", lambda content: content + "\n-include $(objects:.o=.d)\n")

    libbf = source / "libbf"
    patch("CMakeLists.txt", lambda content: replace_once(
        content, "add_library(libbf SHARED", "add_library(libbf STATIC",
    ), repo=libbf)
    patch("src/hash.cpp", lambda content: "#include <stdexcept>\n" + content, repo=libbf)


if __name__ == "__main__":
    main()
