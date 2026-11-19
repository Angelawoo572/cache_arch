"""Run full fixed-work, fixed-placement causal transition-policy experiments."""

import argparse
import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
RAW = HERE.parent / "raw/replay"
RUNS = RAW / "runs"
CASES = (
    "mcf-cactus-r1",
    "mcf-r2",
    "lbm-r1",
    "lbm-xalan-r1",
    "gap_bfs3_late",
    "gap_bfs8",
    "gap_bfs8_lbm",
    "row_period2_paced",
    "row_period5_paced",
    "streams_three_identified_paced",
    "streams_three_switch_paced",
    "compiled_rows",
)
POLICIES = ("stride", "look2", "paced", "turn")
MIXES = ("mcf-cactus-r1", "lbm-xalan-r1", "gap_bfs8_lbm")


def original(case, policy):
    legacy = BASE / "raw/qualified" / f"{case}-{policy}"
    if (legacy / "complete.json").is_file():
        return legacy
    return BASE / "opportunity_study/raw/runs" / f"{case}-{policy}"


def csvs(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def stats(folder):
    return {r["key"]: int(r["value"]) for r in csvs(folder / "stats.csv")}


def run(job):
    case, p = job
    old = original(case, "stride")
    spec = json.loads((old / "run.json").read_text())
    cores = len(spec["traces"])
    slots = [stats(old)["page_slot_core_" + str(i)] for i in range(cores)]
    src = BASE / ("raw/traces" if old.parent.name == "qualified" else "opportunity_study/raw/traces")
    target = RUNS / f"{case}__{p}"
    target.mkdir(parents=True, exist_ok=True)
    args = [
        str(RAW / "build/pythia/bin/champsim"),
        f"--warmup_instructions={spec['warmup']}",
        f"--simulation_instructions={spec['roi']}",
        "-traces",
    ] + [str(src / (t + ".champsimtrace")) for t in spec["traces"]]
    env = dict(
        HYBRID_DETAIL="1",
        HYBRID_COMPRESS="1",
        HYBRID_COMPACT="0",
        LLC_PF_POLICY="stride",
        TRANSITION_RULE=p,
        HYBRID_OUT=str(target),
        HYBRID_ACTIVE_CORES=str(cores),
        HYBRID_PAGE_SLOTS=",".join(map(str, slots)),
        HYBRID_SKIPS=",".join([str(spec["skip"])] * cores),
    )
    if case in MIXES:
        env["TRANSITION_VICTIMS"] = "1"
    meta = dict(
        case=case,
        policy=p,
        active_cores=cores,
        slots=slots,
        warmup=spec["warmup"],
        roi=spec["roi"],
        skip=spec["skip"],
        traces=spec["traces"],
        victim_probe=case in MIXES,
    )
    if (target / "complete.json").is_file():
        assert json.loads((target / "complete.json").read_text()) == meta
        return
    inherited = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("HYBRID_", "TRANSITION_", "LLC_PF_", "STUDY_"))
    }
    with (target / "stdout.log").open("w") as f:
        subprocess.run(args, env=dict(inherited, **env), stdout=f, stderr=subprocess.STDOUT, check=True)
    core = csvs(target / "cores.csv")
    assert len(core) == cores
    assert all(
        int(r["instructions"]) == spec["roi"] and int(r["warmup"]) == spec["warmup"] for r in core
    )
    if p == "stride":
        assert csvs(old / "cores.csv") == core and stats(old) == stats(target), (case, p)
    (target / "complete.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(case, p, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    ap.add_argument("--policies", nargs="+", choices=POLICIES, default=POLICIES)
    arg = ap.parse_args()
    # Small arms finish first. Original SPEC windows can run alongside each other.
    order = arg.cases[4:] + arg.cases[:4] if arg.cases == CASES else arg.cases
    with ThreadPoolExecutor(arg.jobs) as pool:
        list(pool.map(run, [(c, p) for c in order for p in arg.policies]))
