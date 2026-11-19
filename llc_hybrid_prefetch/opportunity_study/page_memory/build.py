"""Build a separate page-memory replay against the existing machine."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BUILD=HERE.parent/'raw/page_memory/build'
SRC=BUILD/'src/sim'
SRC.mkdir(parents=True,exist_ok=True)
for file in ('hybrid.h','build.py'):
    shutil.copy2(ROOT/'sim'/file,SRC/file)
code=(ROOT/'sim/hybrid.cc').read_text()
point='    if (intervention) {\n        ++plan_hits;'
assert code.count(point)==1
code=code.replace(point,'''    static page_memory::Engine page;
    int extra=page.step(p.ip,p.address,item.instr,item.operand,current_core_cycle[p.cpu],
                        !std::strcmp(stride_reason,"match"),stride_delta);
    if(extra) { intervention=true; stride_delta=extra; }
    if (diagnostic!=plan.end()) {
        ++plan_hits;''')
point='} else if (std::strcmp(stride_reason, "match")) {'
assert code.count(point)==1
code=code.replace(point,'} else if (!intervention && std::strcmp(stride_reason, "match")) {')
(SRC/'hybrid.cc').write_text((HERE/'page.hh').read_text().replace('#pragma once','')+'\n'+code)
subprocess.run([sys.executable,str(SRC/'build.py'),'--source',str(ROOT/'raw/build-2/pythia'),
                '--jobs',os.environ.get('PAGE_JOBS','4'),'--cores','2','--output',str(BUILD/'native')],check=True)
