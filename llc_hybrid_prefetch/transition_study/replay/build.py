"""Build a separate shared-LLC simulator for causal distance/rule tests."""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
BUILD = HERE.parent / "raw/replay/build"
SRC = BUILD / "pythia"


def replace(s, a, b):
    if s.count(a) != 1:
        raise ValueError("Unexpected upstream hook " + a)
    return s.replace(a, b)


if __name__ == "__main__":
    if not (SRC / "bin/champsim").is_file():
        subprocess.run(
            [
                sys.executable,
                str(BASE / "sim/build.py"),
                "--source",
                str(BASE / "raw/build-2/pythia"),
                "--output",
                str(BUILD),
                "--jobs",
                "6",
            ],
            check=True,
        )
    s = (BASE / "sim/hybrid.cc").read_text()
    s = replace(
        s,
        "Log loads, candidates,",
        '#include "rules.hh"\nDistanceRule distance_rule;\nLog arrival_history;\nLog loads, candidates,',
    )
    s = replace(
        s,
        "    loads.open(",
        """    const char* mode=std::getenv("TRANSITION_RULE");
    if(!mode || !std::strcmp(mode,"stride"))distance_rule.type=DistanceRule::native;
    else if(!std::strcmp(mode,"look2"))distance_rule.type=DistanceRule::look2;
    else if(!std::strcmp(mode,"paced"))distance_rule.type=DistanceRule::paced;
    else if(!std::strcmp(mode,"turn"))distance_rule.type=DistanceRule::turn;
    else throw std::runtime_error("unknown transition rule");
    arrival_history.open("ingress_history","core,pc,line,cycle,native_match,native_delta,mshr,dram_rq,delta,source,measured,instr_id,operand",compressed);
    loads.open(""",
    )
    term = "    if (!measured() || !roi_id(p.instr_id))"
    ix = s.index(term, s.index("void hybrid_ingress("))
    s = (
        s[:ix]
        + """    const bool native_match=!std::strcmp(stride_reason,"match");
    auto decision=distance_rule.observe(p.cpu,p.ip,p.address,current_core_cycle[p.cpu],native_match,stride_delta,c.MSHR.occupancy,uncore.DRAM.RQ[0].occupancy);
    arrival_history.write("%u,%llu,%llu,%llu,%u,%lld,%u,%u,%lld,%s,%u,%llu,%u\\n",p.cpu,U(p.ip),U(p.address),U(current_core_cycle[p.cpu]),unsigned(native_match),(long long)stride_delta,c.MSHR.occupancy,uncore.DRAM.RQ[0].occupancy,(long long)decision.delta,decision.source,unsigned(measured() && roi_id(p.instr_id)),U(ordinal(p.cpu,p.instr_id)),p.data_index);
"""
        + s[ix:]
    )
    start = s.index("    auto diagnostic = plan.find(", ix)
    end = s.index("    PACKET probe;", start)
    s = (
        s[:start]
        + """    if(!decision.delta){candidate(item,"silent",stride_reason);return;}
    const bool changed=std::strcmp(decision.source,"stride");
    item.target=uint64_t(int64_t(item.base)+decision.delta);
    candidate(item,"proposed",decision.source);
"""
        + s[end:]
    )
    s = replace(s, "else if (!intervention && std::find", "else if (!changed && std::find")
    s = replace(
        s,
        'candidate(item, "accepted", intervention ? "intervention" : "stride");',
        'candidate(item, "accepted", decision.source);',
    )
    s = replace(s, "    loads.close();", "    arrival_history.close();\n    loads.close();")
    start = s.index("    pollution_probe =")
    stop = s.index(";", start)
    s = (
        s[:start]
        + '    pollution_probe = (detail && !std::strcmp(detail, "victims")) || std::getenv("TRANSITION_VICTIMS");'
        + s[stop + 1 :]
    )
    (SRC / "src/hybrid.cc").write_text(s)
    shutil.copy(HERE / "rules.hh", SRC / "src/rules.hh")
    cc = (
        os.environ.get("CXX")
        or (shutil.which("g++-15") if platform.system() == "Darwin" else None)
        or "g++"
    )
    subprocess.run(["make", "-j6", "CXX=" + cc], cwd=SRC, check=True)
