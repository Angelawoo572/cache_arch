"""Run registered discovery traces on the original shared-LLC native controls."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
PROJECT = STUDY.parent
DEFAULT_OUTPUT = STUDY / "raw/screen"
BINARY = PROJECT / "opportunity_study/raw/conventional/build/pythia/bin/champsim"
POLICIES = ("nopf", "stride", "multi", "spp")
WARM, ROI = 100_000, 250_000


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def select(cases, only_ready=False, prefix_dir=None):
    rows = table(STUDY / "intake/registry.csv")
    available = [r for r in rows if r["split"] == "discovery"]
    if cases:
        unknown = set(cases) - {r["case"] for r in available}
        if unknown:
            raise ValueError(f"Unknown or reserved case(s): {sorted(unknown)}")
        available = [r for r in available if r["case"] in cases]
    if prefix_dir is not None:
        prefix_dir = prefix_dir.resolve()
        if not prefix_dir.is_relative_to(STUDY):
            raise ValueError(
                "Stage alternate prefixes under the workload expansion study"
            )
        for row in available:
            row["prefix_path"] = str(
                (prefix_dir / (row["case"] + ".champsimtrace")).relative_to(STUDY)
            )
    if only_ready:
        available = [r for r in available if (STUDY / r["prefix_path"]).is_file()]
    return available


def run(job):
    row, policy, root, skip, warmup, roi = job
    case = row["case"]
    trace = STUDY / row["prefix_path"]
    if (
        not trace.is_file()
        or trace.stat().st_size % 64
        or trace.stat().st_size // 64 < skip + warmup + roi + 256
    ):
        raise ValueError(f"Missing/short/incomplete registered trace {case}: {trace}")
    folder = root / "runs" / case / policy
    folder.mkdir(exist_ok=True, parents=True)
    env = dict(
        LLC_PF_POLICY=("nopf" if policy == "nopf" else "stride"),
        STUDY_POLICY=("stride" if policy == "nopf" else policy),
        STUDY_EPOCH_LIMIT="0",
        STUDY_LATENCY="0",
        STUDY_FAST_NATIVE="0",
        STUDY_CHOICES="0",
        HYBRID_ACTIVE_CORES="1",
        HYBRID_COMPRESS="1",
        HYBRID_DETAIL="1",
        HYBRID_PAGE_SLOTS="0",
        HYBRID_SKIPS=str(skip),
        HYBRID_OUT=str(folder),
    )
    spec = dict(
        case=case,
        traces=[case],
        policy=policy,
        skip=skip,
        warmup=warmup,
        roi=roi,
        trace_path=row["prefix_path"],
        geometry_cores=2,
        active_cores=1,
        page_slot=0,
        epoch_limit=0,
        decision_delay=0,
    )
    marker = folder / "complete.json"
    if marker.is_file():
        if json.loads(marker.read_text()) != spec:
            raise ValueError(f"Different configuration already ran in {folder}")
        return
    args = [
        str(BINARY),
        f"--warmup_instructions={warmup}",
        f"--simulation_instructions={roi}",
        "-traces",
        str(trace),
    ]
    (folder / "run.json").write_text(
        json.dumps(dict(spec, command=args, environment=env), indent=2) + "\n"
    )
    active_env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("STUDY_", "LLC_PF_", "HYBRID_"))
    }
    with (folder / "run.log").open("w") as log:
        subprocess.run(
            args,
            env=dict(active_env, **env),
            cwd=folder,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    c = table(folder / "cores.csv")
    assert (
        len(c) == 1
        and int(c[0]["instructions"]) == roi
        and int(c[0]["warmup"]) == warmup
    )
    tmp = folder / "complete.tmp"
    tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.replace(marker)
    print(case, policy, "completed", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", nargs="+")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--warmup", type=int, default=WARM)
    p.add_argument("--roi", type=int, default=ROI)
    p.add_argument(
        "--prefix-dir",
        type=Path,
        help="Alternate input directory of discovery <case>.champsimtrace prefixes under this study",
    )
    p.add_argument(
        "--ready-only",
        action="store_true",
        help="Start only staged discovery inputs; rerun without this flag to require all.",
    )
    args = p.parse_args()
    if args.skip < 0 or args.warmup <= 0 or args.roi <= 0:
        p.error("Require a nonnegative skip and positive warmup/ROI")
    if (
        args.skip or args.prefix_dir or args.warmup != WARM or args.roi != ROI
    ) and args.output.resolve() == DEFAULT_OUTPUT:
        p.error("Use a separate output directory for an alternate measurement window.")
    jobs = [
        (r, policy, args.output.resolve(), args.skip, args.warmup, args.roi)
        for r in select(args.case, args.ready_only, args.prefix_dir)
        for policy in POLICIES
    ]
    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(run, jobs))
