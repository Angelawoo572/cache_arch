"""Build the isolated CactuBSSN physical-address replay on the baseline machine."""

from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parent
BUILD = STUDY / "raw/physical_build"
SRC = BUILD / "src/sim"
SRC.mkdir(parents=True, exist_ok=True)
for file in ("hybrid.h", "build.py"):
    shutil.copy2(ROOT / "sim" / file, SRC / file)
code = (ROOT / "sim/hybrid.cc").read_text()


def change(old, new):
    global code
    if code.count(old) != 1:
        raise ValueError("ambiguous insertion " + old)
    code = code.replace(old, new)


change(
    "    auto ppn = ((i / 64)",
    """    uint64_t offset=i%64;
    if (std::getenv("WIDE_PERMUTE")) {
        offset=0;
        for(unsigned bit=0;bit<6;++bit) offset=(offset<<1)|((i>>bit)&1);
    }
    auto ppn = ((i / 64)""",
)
change(" * 64 + i % 64 + 1;", " * 64 + offset + 1;")
change(
    "    if (intervention) {\n        ++plan_hits;",
    """    static wide_probe::Engine wide;
    static bool initialized = false;
    static std::ofstream choices;
    static std::array<uint64_t,16> recent{};
    static std::set<uint64_t> observed;
    static const bool bounded=std::getenv("WIDE_BOUND")!=nullptr;
    const bool active = p.cpu==0 && std::getenv("WIDE_MODE")!=nullptr;
    bool wide_choice=false;
    if(active) {
        if(!initialized) {
            choices.open(std::string(std::getenv("HYBRID_OUT"))+"/wide.csv");
            choices<<"instr,operand,pc,base,cycle,target,native,action\\n";
            recent.fill(UINT64_MAX);
            initialized=true;
        }
        const uint64_t pg=p.address/64;
        observed.insert(pg);
        auto pos=std::find(recent.begin(),recent.end(),pg);
        if(pos==recent.end()) pos=recent.end()-1;
        std::move_backward(recent.begin(),pos,pos+1);
        recent[0]=pg;
        const uint64_t target=wide.step(p.ip,p.address);
        if(target && target!=p.address) {
            const auto destination=target/64;
            if(!observed.count(destination)) throw std::runtime_error("unseen page requested");
            bool legal=!bounded || std::find(recent.begin(),recent.end(),destination)!=recent.end();
            bool match=!std::strcmp(stride_reason,"match");
            bool same=match && destination==pg && static_cast<int64_t>(target)-static_cast<int64_t>(p.address)==stride_delta;
            choices<<item.instr<<','<<item.operand<<','<<item.pc<<','<<item.base<<','
                   <<current_core_cycle[p.cpu]<<','<<target<<','<<(match ? static_cast<int64_t>(p.address)+stride_delta : 0)
                   <<','<<(!legal?"page_history":same?"agree":"override")<<'\\n';
            if(legal && !same) {
                wide_choice=true;intervention=true;
                stride_delta=static_cast<int64_t>(target)-static_cast<int64_t>(p.address);
            }
        }
    }
    if (diagnostic!=plan.end()) {
        ++plan_hits;""",
)
change(
    '} else if (std::strcmp(stride_reason, "match")) {',
    '} else if (!intervention && std::strcmp(stride_reason, "match")) {',
)
change(
    "else if (item.base / 64 != item.target / 64)",
    "else if (!wide_choice && item.base / 64 != item.target / 64)",
)
header = (STUDY / "history/frozen_rule.h").read_text().replace("#pragma once", "")
(SRC / "hybrid.cc").write_text(
    "#include <array>\n#include <set>\n" + header + "\n" + code
)
subprocess.run(
    [
        sys.executable,
        str(SRC / "build.py"),
        "--source",
        str(ROOT / "raw/build-2/pythia"),
        "--jobs",
        "4",
        "--cores",
        "2",
        "--output",
        str(BUILD / "native"),
    ],
    check=True,
)
