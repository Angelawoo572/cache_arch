"""Apply the conventional comparison only inside its own isolated simulator checkout."""

import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
RAW = BASE / "opportunity_study/raw/conventional"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build", type=Path, default=RAW / "build")
parser.add_argument("--jobs", type=int, default=6)
p = parser.parse_args()
dest = p.build.resolve()
if not dest.is_relative_to(RAW.resolve()):
    raise ValueError("select a directory under the new study raw tree")
sim = dest / "pythia"
if not (sim / "bin/champsim").is_file():
    subprocess.run(
        [
            "python3",
            str(BASE / "sim/build.py"),
            "--source",
            str(BASE / "raw/build-2/pythia"),
            "--output",
            str(dest),
            "--jobs",
            str(p.jobs),
        ],
        check=True,
    )


def inject(s, x, y):
    if s.count(x) != 1:
        raise ValueError(x)
    return s.replace(x, y)


s = (BASE / "sim/hybrid.cc").read_text()
s = inject(
    s,
    '#include "stride.h"',
    '#include "stride.h"\n#include "spp_dev2_helper.h"\n#include <set>\n#include <deque>',
)
s = inject(s, "void hybrid_init(", 'namespace {\n#include "controls.inc"\n}\nvoid hybrid_init(')
s = inject(
    s,
    "struct Candidate {",
    "struct Candidate {\n    unsigned study_group=0;bool study_stride_source=false;",
)

s = inject(
    s,
    '    if (auto path = std::getenv("HYBRID_PLAN"))',
    '    study_init();\n    if (auto path = std::getenv("HYBRID_PLAN"))',
)
beg = s.index("    if (!measured() || !roi_id(p.instr_id))", s.index("void hybrid_ingress("))
end = s.index("\n}\nvoid hybrid_terminal", beg)
s = (
    s[:beg]
    + """    const bool observe=measured() && roi_id(p.instr_id);
    auto choice=study_predict(c,p,observe);
    if(observe)study_submit(c,p,choice);"""
    + s[end:]
)
s = inject(
    s,
    "    cache_line(c, p, stage, hit);",
    "    if(c.cache_type==IS_LLC && hit>=0 && demand(p) && lanes[p.cpu].spp)lanes[p.cpu].spp->demand_outcome(p.address,hit);\n    cache_line(c, p, stage, hit);",
)
s = inject(
    s,
    "void hybrid_bank_cycle() {",
    "void hybrid_bank_cycle() {\n    directed_duplicates();\n    study_tick();",
)
s = inject(
    s,
    "void hybrid_fill(CACHE &c, const PACKET &p, const BLOCK &victim) {",
    "void hybrid_fill(CACHE &c, const PACKET &p, const BLOCK &victim) {\n    for(unsigned i=0;i<active_cores;++i)if(lanes[i].spp)lanes[i].spp->fill(p.address,p.type==PREFETCH,victim.address,victim.valid);",
)
s = inject(
    s,
    "    loads.close();",
    """    for(const auto& e:diagnostics)stat("study_"+e.first,e.second);
    for(unsigned cpu=0;cpu<active_cores;++cpu)stat("study_accepted_core_"+std::to_string(cpu),lanes[cpu].issues);
    stat("study_engine_pending_end",decision_queue.size());
    if(track_choices)choices.close();
    loads.close();""",
)
s = inject(
    s,
    "    candidates.write(",
    """    if (stage && measured()) {
        const std::string name(stage);
        if (name=="timely" || name=="late" || name=="evicted" || name=="unresolved" || name=="redundant")
            ++diagnostics["pf_"+name];
        if (name=="accepted" || name=="timely" || name=="late" || name=="evicted" || name=="redundant" || name=="unresolved")
            ++diagnostics["cohort_"+std::to_string(c.study_group)+"_"+name];
    }
    candidates.write(""",
)
# Declare the counting map before the base lifecycle hook sees it.
s = inject(
    s, "std::string directory;", "std::string directory;\nstd::map<std::string,uint64_t> diagnostics;"
)
for name in ("controls.inc", "spp.inc"):
    data = (HERE / name).read_text()
    (sim / "src" / name).write_text(data)
(sim / "src/hybrid.cc").write_text(s)
cc = os.getenv("CXX") or (shutil.which("g++-15") if platform.system() == "Darwin" else None) or "g++"
subprocess.run(["make", "-j" + str(p.jobs), "CXX=" + cc], cwd=sim, check=True)
print(sim / "bin/champsim")
