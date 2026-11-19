"""Build an independent native hook without writing into the existing simulator."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
BUILD = HERE.parent / "raw" / "cue_build"
SRC = BUILD / "study" / "sim"
SRC.mkdir(parents=True, exist_ok=True)
for path in ("build.py", "hybrid.h"):
    shutil.copy2(PROJECT / "sim" / path, SRC / path)
code = (PROJECT / "sim/hybrid.cc").read_text()
marker = "    if (intervention) {\n        ++plan_hits;"
assert code.count(marker) == 1
code = (
    (HERE / "cue.hh").read_text().replace("#pragma once", "")
    + "\n"
    + code.replace(
        marker,
        """    static earlier_cue::Engine cue;
    const int delta = p.cpu == 0 ? cue.step(p.ip,p.address,item.instr,item.operand,current_core_cycle[p.cpu],
        stride_delta,!std::strcmp(stride_reason,"match")) : 0;
    if (delta) { intervention=true; stride_delta=delta; }
    if (diagnostic!=plan.end()) {
        ++plan_hits;""",
    ).replace(
        '} else if (std::strcmp(stride_reason, "match")) {',
        '} else if (!intervention && std::strcmp(stride_reason, "match")) {',
    )
)
(SRC / "hybrid.cc").write_text(code)
subprocess.run(
    [
        sys.executable,
        str(SRC / "build.py"),
        "--cores",
        "2",
        "--source",
        str(PROJECT / "raw/build-2/pythia"),
        "--jobs",
        os.environ.get("CUE_JOBS", "3"),
        "--output",
        str(BUILD / "native"),
    ],
    check=True,
)
