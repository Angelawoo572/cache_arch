#!/usr/bin/env python3
"""Take checkpoints and run measured arms for the registered workloads.

Run on Sacramento after `source sim/env.sh`:

    python sim/np_runs.py checkpoint micro_list micro_tree
    python sim/np_runs.py run --arms nopf,fast micro_list micro_tree
    python sim/np_runs.py status

Outputs go under gem5_neural_prefetch/raw/ (git-ignored):
    raw/checkpoints/<workload>/cpt_<offset>/
    raw/runs/<workload>__o<offset>__<arm>/{m5out,capture,stdout.txt,gem5.log}
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STUDY = Path(os.environ["GEM5_STUDY"])
CACHE = Path(os.environ["CACHE_ROOT"])
GEM5 = os.environ["GEM5_BIN"]
REGISTRY = json.loads((STUDY / "sim" / "workloads.json").read_text())
CONFIGS = STUDY / "sim" / "configs"


def workload(name):
    if name not in REGISTRY:
        sys.exit(f"unknown workload {name!r}; known: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[name]


def launch(command, log_path):
    """Run one gem5 process at the lowest priority from the cache root, so
    that relative paths in workload arguments resolve the same way everywhere."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w") as log:
        code = subprocess.call(
            ["nice", "-n", "19"] + [str(c) for c in command],
            cwd=CACHE,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
    return code, time.time() - started


def take_checkpoints(name, force):
    spec = workload(name)
    out = STUDY / "raw" / "checkpoints" / name
    if (out / "checkpoints.json").exists() and not force:
        return name, "exists", 0.0
    command = [
        GEM5,
        f"--outdir={out / 'm5out'}",
        CONFIGS / "checkpoint.py",
        "--binary",
        CACHE / spec["binary"],
        "--args",
        spec["args"],
        "--checkpoints",
        out,
        "--offsets",
        ",".join(str(o) for o in spec["offsets"]),
        "--stdout",
        out / "stdout.txt",
    ]
    code, seconds = launch(command, out / "gem5.log")
    done = (out / "checkpoints.json").exists()
    return name, "ok" if code == 0 and done else f"FAILED ({code})", seconds


def run_dir(name, offset, arm):
    return STUDY / "raw" / "runs" / f"{name}__o{offset}__{arm}"


def run_arm(name, offset, arm, warmup, measure, with_uops, force):
    spec = workload(name)
    out = run_dir(name, offset, arm)
    label = out.name
    if (out / "capture" / "capture_counts.json").exists() and not force:
        return label, "exists", 0.0
    checkpoint = STUDY / "raw" / "checkpoints" / name / f"cpt_{offset}"
    if not checkpoint.exists():
        return label, "NO CHECKPOINT", 0.0
    command = [
        GEM5,
        f"--outdir={out / 'm5out'}",
        CONFIGS / "run.py",
        "--binary",
        CACHE / spec["binary"],
        "--args",
        spec["args"],
        "--checkpoint",
        checkpoint,
        "--arm",
        arm,
        "--warmup",
        warmup,
        "--measure",
        measure,
        "--capture",
        out / "capture",
        "--stdout",
        out / "stdout.txt",
    ]
    if not with_uops:
        command.append("--no-uops")
    code, seconds = launch(command, out / "gem5.log")
    done = (out / "capture" / "capture_counts.json").exists()
    return label, "ok" if code == 0 and done else f"FAILED ({code})", seconds


def report(results):
    failed = 0
    for label, state, seconds in results:
        print(f"{state:>14}  {seconds:8.0f}s  {label}", flush=True)
        failed += state not in ("ok", "exists")
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)

    p = sub.add_parser("checkpoint")
    p.add_argument("workloads", nargs="+")
    p.add_argument("-j", "--jobs", type=int, default=4)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("run")
    p.add_argument("workloads", nargs="+")
    p.add_argument("--arms", default="nopf,fast")
    p.add_argument("--offsets", default=None, help="default: all registered")
    p.add_argument("--warmup", type=int, default=5_000_000)
    p.add_argument("--measure", type=int, default=20_000_000)
    p.add_argument(
        "--uops-arm",
        default="nopf",
        help="the one arm that also records the architectural streams",
    )
    p.add_argument("-j", "--jobs", type=int, default=6)
    p.add_argument("--force", action="store_true")

    sub.add_parser("status")
    args = parser.parse_args()

    if args.action == "status":
        for name in sorted(REGISTRY):
            cpts = STUDY / "raw" / "checkpoints" / name
            have = sorted(p.name for p in cpts.glob("cpt_*")) if cpts.exists() else []
            runs = sorted(
                p.parent.parent.name.split("__", 1)[1]
                for p in (STUDY / "raw" / "runs").glob(
                    f"{name}__*/capture/capture_counts.json"
                )
            )
            print(f"{name}: checkpoints {have or 'none'}; captures {runs or 'none'}")
        return 0

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        if args.action == "checkpoint":
            jobs = [
                pool.submit(take_checkpoints, name, args.force)
                for name in args.workloads
            ]
        else:
            arms = args.arms.split(",")
            jobs = []
            for name in args.workloads:
                offsets = (
                    [int(o) for o in args.offsets.split(",")]
                    if args.offsets
                    else workload(name)["offsets"]
                )
                for offset in offsets:
                    for arm in arms:
                        jobs.append(
                            pool.submit(
                                run_arm,
                                name,
                                offset,
                                arm,
                                args.warmup,
                                args.measure,
                                arm == args.uops_arm,
                                args.force,
                            )
                        )
        return 1 if report(job.result() for job in jobs) else 0


if __name__ == "__main__":
    sys.exit(main())
