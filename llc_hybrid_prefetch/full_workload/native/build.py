"""Add passive full-ROI counters and sampled event logs to the study's validated conventional simulator."""

import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
SOURCE = BASE / "opportunity_study/raw/conventional/build/pythia"
BUILD = HERE.parent / "raw/native/build/pythia"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Unexpected upstream instrumentation anchor: " + old)
    return source.replace(old, new)


def build(jobs):
    if not (SOURCE / "bin/champsim").exists():
        raise RuntimeError(
            "First build the existing opportunity_study/conventional/build.py simulator."
        )
    if not BUILD.exists():
        shutil.copytree(SOURCE, BUILD, ignore=shutil.ignore_patterns(".git"))
    shutil.copyfile(HERE / "observations.inc", BUILD / "src/observations.inc")
    s = (SOURCE / "src/hybrid.cc").read_text()
    patches = [
        (
            "unsigned active_cores;",
            "unsigned active_cores;\nuint64_t full_source_origin;",
        ),
        (
            "return skip[core] + id + 1;",
            "return full_source_origin + skip[core] + id + 1;",
        ),
        (
            "void candidate(const Candidate",
            '#include "observations.inc"\nvoid candidate(const Candidate',
        ),
        (
            "    candidates.write(",
            """    if (d && demand(*d) && roi_id(d->instr_id)) {
        auto &r = regions[d->cpu][phase_of(d->instr_id)];
        if (!std::strcmp(stage, "timely")) {
            ++r.timely;
            record_outcome(*d, 8);
        } else if (!std::strcmp(stage, "late")) {
            ++r.late;
            record_outcome(*d, 4);
        }
    }
    if (!sampled(c.instr - full_source_origin - skip[c.core] - 1) && !(d && sampled(d->instr_id)))
        return;
    candidates.write(""",
        ),
        (
            "    if (compact_log && c.cache_type == IS_L1D)",
            """    auto &r = regions[p.cpu][phase_of(p.instr_id)];
    if (c.cache_type == IS_L2C && hit >= 0)
        ++(hit ? r.l2_hit : r.l2_miss);
    if (c.cache_type == IS_LLC) {
        if (hit >= 0) {
            ++(hit ? r.llc_hit : r.llc_miss);
            record_outcome(p, hit ? 1 : 2);
        } else if (!std::strcmp(stage, "ingress"))
            ++r.ingress;
    }
    if (!sampled(p.instr_id))
        return;
    if (compact_log && c.cache_type == IS_L1D)""",
        ),
        (
            "if (addr && measured() && roi_id(i.instr_id))",
            """if (addr && measured() && roi_id(i.instr_id))
            ++regions[c][phase_of(i.instr_id)].loads;
        if (addr && measured() && sampled(i.instr_id))""",
        ),
        (
            "void hybrid_retire(unsigned c, uint64_t id) {",
            """void hybrid_retire(unsigned c, uint64_t id) {
    if (measured() && roi_id(id))
        retire_outcomes(c, id);""",
        ),
        ("            heads.write(", "            if (sampled(id)) heads.write("),
        (
            "    study_init();",
            """    if (const char *origin = std::getenv("FULL_SOURCE_ORIGIN"))
        full_source_origin = std::stoull(origin);
    if (full_source_origin) {
        const char *path = std::getenv("FULL_PREFIX_PAGES");
        if (!path || active != 1)
            throw std::runtime_error("far view requires a single-core ordered prefix VPN seed");
        std::ifstream seed(path, std::ios::binary);
        if (!seed)
            throw std::runtime_error("cannot open source-prefix mapping seed");
        unsigned char bytes[8];
        while (seed.read(reinterpret_cast<char *>(bytes), sizeof(bytes))) {
            uint64_t page = 0;
            for (unsigned k = 0; k < 8; ++k)
                page |= uint64_t(bytes[k]) << (8 * k);
            size_t before = maps[0].size();
            register_address(0, (page << LOG2_PAGE_SIZE) | 1);
            if (maps[0].size() != before + 1)
                throw std::runtime_error("duplicate source-prefix VPN");
        }
        if (seed.gcount() || !seed.eof())
            throw std::runtime_error("incomplete source-prefix mapping seed");
    }
    start_observations();
    study_init();""",
        ),
        ("    loads.close();", "    finish_observations();\n    loads.close();"),
        (
            "    if (victim.valid && p.type == PREFETCH && measured())",
            """    const auto source = pending.find(p.address);
    if (victim.valid && p.type == PREFETCH && measured() && source != pending.end() &&
        sampled(source->second.instr - full_source_origin - skip[source->second.core] - 1))""",
        ),
    ]
    for old, new in patches:
        s = replace_once(s, old, new)
    (BUILD / "src/hybrid.cc").write_text(s)
    cc = (
        os.getenv("CXX")
        or (shutil.which("g++-15") if platform.system() == "Darwin" else None)
        or "g++"
    )
    subprocess.run(["make", "-j" + str(jobs), "CXX=" + cc], cwd=BUILD, check=True)
    print(BUILD / "bin/champsim")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    build(parser.parse_args().jobs)
