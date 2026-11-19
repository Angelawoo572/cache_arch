"""Replay the R1-trained LBM cue alternatives at common grants on LBM R2."""

import argparse
import csv
import gzip
import json
import os
import platform
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

from build import replace
from run import BASE, HERE, RAW, RUNS, csvs, original, stats
from score import save, summarize

CASE = "lbm-r2"
REPS = ("page", "exact", "unrelated_older", "last_global_pc")
MODES = ("shadow", "plus1", "plus5", *REPS)


def build():
    dest = RAW / "lbm_build/pythia"
    if not (dest / "bin/champsim").exists():
        dest.parent.mkdir(exist_ok=True)
        shutil.copytree(RAW / "build/pythia", dest, symlinks=True)
    s = (RAW / "build/pythia/src/hybrid.cc").read_text()
    s = replace(s, '#include "hybrid.h"', '#include "hybrid.h"\n#include "transition_rule.hh"')
    s = replace(
        s,
        "DistanceRule distance_rule;",
        """DistanceRule distance_rule;
    std::unique_ptr<boundary_forecast::Engine> lbm;
    uint64_t lbm_index=0,lbm_start=0,lbm_stop=0;
    std::unordered_set<uint64_t> lbm_pages;
    std::string lbm_mode;
    Log lbm_grants;""",
    )
    s = replace(
        s,
        "    loads.open(",
        """    const char* table=std::getenv("LBM_FIT");
    lbm.reset(new boundary_forecast::Engine(table));
    std::ifstream range(table); if(!(range>>lbm_start>>lbm_stop))throw std::runtime_error("bad LBM range");
    lbm_mode=std::getenv("LBM_MODE");
    lbm_grants.open("lbm_grants","index,instr,operand,base,cycle,choice,aux_offer,native_offer,actual_offer",compressed);
    loads.open(""",
    )
    key = "    arrival_history.write("
    s = replace(
        s,
        key,
        """    if(measured() && roi_id(p.instr_id)){
        uint64_t ix=lbm_index++; lbm_pages.insert(p.address/64);
        auto next=lbm->step(p.ip,p.address,current_core_cycle[p.cpu]);
        if(ix>=lbm_start && ix<lbm_stop && boundary_forecast::sample(ix)){
            int64_t candidate=next;
            if(lbm_mode=="plus1")candidate=int64_t(p.address)+1;
            if(lbm_mode=="plus5")candidate=int64_t(p.address)+5;
            if((lbm_mode=="plus1" || lbm_mode=="plus5") && uint64_t(candidate)/64!=p.address/64)candidate=-1;
            bool legal=candidate>0 && uint64_t(candidate)!=p.address && lbm_pages.count(uint64_t(candidate)/64);
            auto native_offer=native_match ? int64_t(p.address)+stride_delta : -1;
            if(lbm_mode!="shadow" && legal && candidate!=native_offer)decision={candidate-int64_t(p.address),"auxiliary"};
            lbm_grants.write("%llu,%llu,%u,%llu,%llu,%lld,%lld,%lld,%lld\\n",U(ix),U(ordinal(p.cpu,p.instr_id)),p.data_index,U(p.address),U(current_core_cycle[p.cpu]),(long long)next,(long long)(legal?candidate:-1),(long long)native_offer,(long long)(decision.delta?int64_t(p.address)+decision.delta:-1));
        }
    }
"""
        + key,
    )
    s = replace(
        s,
        "else if (item.base / 64 != item.target / 64)",
        'else if (item.base / 64 != item.target / 64 && (std::strcmp(decision.source,"auxiliary") || !lbm_pages.count(item.target/64)))',
    )
    s = replace(
        s, "    arrival_history.close();", "    lbm_grants.close();\n    arrival_history.close();"
    )
    (dest / "src/hybrid.cc").write_text(s)
    shutil.copy(HERE.parent / "predictability/transition_rule.hh", dest / "inc/transition_rule.hh")
    config = dest / "Makefile"
    txt = config.read_text()
    if "-std=c++11" in txt:
        config.write_text(replace(txt, "-std=c++11", "-std=c++14"))
    cc = (
        os.environ.get("CXX")
        or (shutil.which("g++-15") if platform.system() == "Darwin" else None)
        or "g++"
    )
    subprocess.run(["make", "-j6", "CXX=" + cc], cwd=dest, check=True)
    return dest / "bin/champsim"


def run(job):
    binary, mode, rep = job
    tag = "lbm_" + mode + ("_" + rep if mode == "shadow" else "")
    folder = RUNS / (CASE + "__" + tag)
    folder.mkdir(parents=True, exist_ok=True)
    base = original(CASE, "stride")
    s = json.loads((base / "run.json").read_text())
    src = HERE.parent / "raw/predictability" / ("lbm-" + rep + ".txt")
    meta = dict(mode=mode, representation=rep, spec=s["run_id"], source=src.name)
    if (folder / "complete.json").is_file():
        assert json.loads((folder / "complete.json").read_text()) == meta
        return
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("HYBRID_", "TRANSITION_", "LLC_PF_", "STUDY_", "LBM_"))
    }
    env.update(
        HYBRID_DETAIL="1",
        HYBRID_COMPRESS="1",
        LLC_PF_POLICY="stride",
        TRANSITION_RULE="stride",
        HYBRID_OUT=str(folder),
        HYBRID_ACTIVE_CORES="1",
        HYBRID_PAGE_SLOTS="0",
        HYBRID_SKIPS=str(s["skip"]),
        LBM_MODE=mode,
        LBM_FIT=str(src),
    )
    args = [
        str(binary),
        f"--warmup_instructions={s['warmup']}",
        f"--simulation_instructions={s['roi']}",
        "-traces",
        str(BASE / "raw/traces" / (s["traces"][0] + ".champsimtrace")),
    ]
    with (folder / "stdout.log").open("w") as f:
        subprocess.run(args, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    if mode == "shadow":
        assert stats(folder) == stats(base) and csvs(folder / "cores.csv") == csvs(base / "cores.csv")
        expected = csvs(src.with_name("lbm-" + rep + "-expected.csv"))
        with gzip.open(folder / "lbm_grants.csv.gz", "rt") as f:
            out = list(csv.DictReader(f))
        assert len(out) == len(expected)
        assert all(
            all(int(actual[k]) == int(ref[k]) for k in ref) for actual, ref in zip(out, expected)
        ), (mode, rep)
    (folder / "complete.json").write_text(json.dumps(meta, indent=2) + "\n")


def outputs():
    modes = ["lbm_shadow_page", "lbm_plus1", "lbm_plus5"] + ["lbm_" + r for r in REPS]
    result = [summarize((CASE, m)) for m in modes]
    for j, title in enumerate(("cohorts", "examples", "unique_targets", "performance")):
        save(HERE / ("lbm_" + title + ".csv"), [row for item in result for row in item[j]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--skip-summary", action="store_true")
    args = ap.parse_args()
    binary = build()
    jobs = [(binary, "shadow", rep) for rep in REPS] + [
        (binary, m, "page" if m in ("plus1", "plus5") else m) for m in MODES if m != "shadow"
    ]
    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(run, jobs))
    if not args.skip_summary:
        outputs()
