"""Attach the independently frozen offset forecast to the already validated simulator and run two fixed cases."""

import json
import os
import platform
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

from build import replace
from run import BASE, HERE, RAW, RUNS, csvs, original, stats

AUX = {"gap_bfs3_late": "bfs3-late", "row_period2_paced": "row2"}
MODES = ("shadow", "matched_next", "frozen")


def build():
    # Copy the finished independent study checkout to leave the 48 primary arms reproducible.
    dest = RAW / "aux_build/pythia"
    if not (dest / "bin/champsim").exists():
        dest.parent.mkdir(exist_ok=True)
        shutil.copytree(RAW / "build/pythia", dest, symlinks=True)
    base = (RAW / "build/pythia/src/hybrid.cc").read_text()
    s = replace(base, '#include "hybrid.h"', '#include "hybrid.h"\n#include "transition_rule.hh"')
    s = replace(
        s,
        "DistanceRule distance_rule;",
        """DistanceRule distance_rule;
    std::unique_ptr<boundary_forecast::Engine> supplement;
    std::string supp_mode;
    Log aux_candidates;""",
    )
    s = replace(
        s,
        "    loads.open(",
        """    supplement.reset(new boundary_forecast::Engine(std::getenv("TRANSITION_TABLE")));
    supp_mode=std::getenv("TRANSITION_AUX");
    if(supp_mode!="shadow" && supp_mode!="matched_next" && supp_mode!="frozen")throw std::runtime_error("bad supplement mode");
    aux_candidates.open("aux_grants","instr_id,operand,pc,base,cycle,offered,valid,applied,selected",compressed);
    loads.open(""",
    )
    key = "    arrival_history.write("
    s = replace(
        s,
        key,
        """    if(measured() && roi_id(p.instr_id)){
        auto offered=supplement->step(p.ip,p.address,current_core_cycle[p.cpu]); 
        if(offered>=0){
            bool valid=offered>0 && uint64_t(offered)/64==p.address/64 && uint64_t(offered)!=p.address;
            int64_t choice=-1;
            if(valid && supp_mode=="frozen")choice=offered;
            if(valid && supp_mode=="matched_next" && (p.address+1)/64==p.address/64)choice=p.address+1;
            if(choice>=0 && (!native_match || int64_t(p.address)+stride_delta!=choice))decision={choice-int64_t(p.address),"auxiliary"};
            aux_candidates.write("%llu,%u,%llu,%llu,%llu,%lld,%u,%lld,%lld\\n",U(ordinal(p.cpu,p.instr_id)),p.data_index,U(p.ip),U(p.address),U(current_core_cycle[p.cpu]),(long long)offered,unsigned(valid),(long long)choice,(long long)(decision.delta?int64_t(p.address)+decision.delta:-1));
        }
    }
"""
        + key,
    )
    s = replace(
        s, "    arrival_history.close();", "    aux_candidates.close();\n    arrival_history.close();"
    )
    (dest / "src/hybrid.cc").write_text(s)
    shutil.copy(HERE.parent / "predictability/transition_rule.hh", dest / "inc/transition_rule.hh")
    makefile = dest / "Makefile"
    contents = makefile.read_text()
    if "-std=c++11" in contents:
        makefile.write_text(replace(contents, "-std=c++11", "-std=c++14"))
    compiler = (
        os.environ.get("CXX")
        or (shutil.which("g++-15") if platform.system() == "Darwin" else None)
        or "g++"
    )
    subprocess.run(["make", "-j6", "CXX=" + compiler], cwd=dest, check=True)
    return dest / "bin/champsim"


def run(job, binary):
    case, mode = job
    src = original(case, "stride")
    spec = json.loads((src / "run.json").read_text())
    folder = RUNS / (case + "__" + mode)
    folder.mkdir(parents=True, exist_ok=True)
    meta = dict(case=case, mode=mode, table=AUX[case], warmup=spec["warmup"], roi=spec["roi"])
    if (folder / "complete.json").is_file():
        assert json.loads((folder / "complete.json").read_text()) == meta
        return
    trace = BASE / "opportunity_study/raw/traces" / (spec["traces"][0] + ".champsimtrace")
    args = [
        str(binary),
        f"--warmup_instructions={spec['warmup']}",
        f"--simulation_instructions={spec['roi']}",
        "-traces",
        str(trace),
    ]
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("HYBRID_", "TRANSITION_", "LLC_PF_", "STUDY_"))
    }
    env.update(
        HYBRID_DETAIL="1",
        HYBRID_COMPRESS="1",
        LLC_PF_POLICY="stride",
        TRANSITION_RULE="stride",
        HYBRID_OUT=str(folder),
        HYBRID_ACTIVE_CORES="1",
        HYBRID_PAGE_SLOTS="0",
        HYBRID_SKIPS=str(spec["skip"]),
        TRANSITION_AUX=mode,
        TRANSITION_TABLE=str(HERE.parent / "raw/predictability" / (AUX[case] + ".txt")),
    )
    with (folder / "stdout.log").open("w") as f:
        subprocess.run(args, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    if mode == "shadow":
        assert stats(folder) == stats(src) and csvs(folder / "cores.csv") == csvs(src / "cores.csv")
    (folder / "complete.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    binary = build()
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda j: run(j, binary), [(c, m) for c in AUX for m in MODES]))
    from score import save, summarize

    result = [summarize((c, m)) for c in AUX for m in MODES]
    for j, title in enumerate(("cohorts", "examples", "unique_targets", "performance")):
        save(HERE / ("supplement_" + title + ".csv"), [row for item in result for row in item[j]])
