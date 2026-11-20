"""Compile passive continuous observers against the retained, validated native simulator."""

import os
from pathlib import Path
import platform
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
FULL = STUDY.parent / "full_workload"
SOURCE = FULL / "raw/native/build/pythia"
RAW = STUDY / "raw/capture"
BUILD = RAW / "build/pythia"


def build(directory=BUILD, permute=False):
    if not directory.exists():
        shutil.copytree(SOURCE, directory, ignore=shutil.ignore_patterns(".git"))
    code = (SOURCE / "src/hybrid.cc").read_text()
    anchors = (
        (
            '#include "observations.inc"',
            '#include "observations.inc"\n#include "writer.inc"',
        ),
        (
            "    start_observations();",
            "    start_observations();\n    start_capture();",
        ),
        (
            "    finish_observations();",
            "    finish_capture();\n    finish_observations();",
        ),
        (
            "    if (stage && measured()) {",
            "    if (measured()) capture_offer(c, stage, why, d);\n    if (stage && measured()) {",
        ),
        (
            "    if (c.cache_type == IS_L2C && hit >= 0)",
            "    capture_cache(c, p, stage, hit);\n    if (c.cache_type == IS_L2C && hit >= 0)",
        ),
        (
            "            ++regions[c][phase_of(i.instr_id)].loads;",
            """        {
            ++regions[c][phase_of(i.instr_id)].loads;
            core_events.row({operand_key(c, i.instr_id, op), current_core_cycle[c]});
        }""",
        ),
        (
            "            if (sampled(id)) heads.write",
            """            wait_events.row({operand_key(c, id, op), current_core_cycle[c], head_counts[c][op]});
            if (sampled(id)) heads.write""",
        ),
        (
            "    const auto source = pending.find(p.address);",
            """    const auto source = pending.find(p.address);
    if (victim.valid && p.type == PREFETCH && measured() && source != pending.end())
        eviction_events.row({p.pf_metadata, victim.address, p.address, current_core_cycle[p.cpu]});""",
        ),
    )
    for old, new in anchors:
        if code.count(old) != 1:
            raise ValueError(old)
        code = code.replace(old, new)
    if permute:
        # Match the previous physical_replay experiment's six-bit within-chunk reversal.
        old = "    auto ppn = ((i / 64) * NUM_CPUS + page_slots[c]) * 64 + i % 64 + 1;"
        new = """    uint64_t offset = 0;
    for (unsigned bit = 0; bit < 6; ++bit)
        offset = (offset << 1) | ((i >> bit) & 1u);
    auto ppn = ((i / 64) * NUM_CPUS + page_slots[c]) * 64 + offset + 1;"""
        if code.count(old) != 1:
            raise ValueError("Expected native page allocator")
        code = code.replace(old, new)
    (directory / "src/hybrid.cc").write_text(code)
    shutil.copyfile(HERE / "writer.inc", directory / "src/writer.inc")
    compiler = (
        os.getenv("CXX")
        or (shutil.which("g++-15") if platform.system() == "Darwin" else None)
        or "g++"
    )
    subprocess.run(["make", "-j3", "CXX=" + compiler], cwd=directory, check=True)


if __name__ == "__main__":
    build()
