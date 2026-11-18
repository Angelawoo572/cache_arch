"""Run/resume registered paired SPEC and synthetic measurements; each run owns its directory."""
import argparse
import csv
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from matrix import ROOT, TRACE_DIR, run_specs


def write_json(target, data):
    partial = target.with_suffix(".tmp")
    partial.write_text(json.dumps(data, indent=2) + "\n")
    partial.replace(target)


def checked_completion(folder, spec):
    with (folder / "cores.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    return (list(range(len(spec["traces"]))) == [int(row["cpu"]) for row in rows]
            and all(int(row["warmup"]) == spec["warmup"]
                    and int(row["instructions"]) == spec["roi"] and int(row["cycles"]) > 0
                    for row in rows))


def one(spec, options):
    out = options.out / spec["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return run_locked(out, spec, options)


def run_locked(out, spec, options):
    complete = out / "complete.json"
    paths = [TRACE_DIR / (stem + ".champsimtrace") for stem in spec["traces"]]
    for path in paths:
        # Include one read-ahead instruction beyond the measurement.
        if path.stat().st_size < 64 * (spec["skip"] + spec["warmup"] + spec["roi"] + 1):
            raise RuntimeError("insufficient trace prefix: " + str(path))
    command = [str(options.bin), "--warmup_instructions=" + str(spec["warmup"]),
               "--simulation_instructions=" + str(spec["roi"]), "-traces"] + list(map(str, paths))
    slots = [0, 1] if len(paths) == 2 else [int(spec["workloads"][0] in ("cactus", "xalan"))]
    env = dict(HYBRID_PAGE_SLOTS=",".join(map(str, slots)), HYBRID_OUT=str(out), LLC_PF_POLICY=spec["policy"],
               HYBRID_ACTIVE_CORES=str(len(paths)), HYBRID_SKIPS=",".join([str(spec["skip"])] * len(paths)),
               HYBRID_COMPRESS=str(options.compress))
    inherited = {k: v for k, v in os.environ.items()
                 if k.startswith("HYBRID_") or k.startswith("LLC_PF_")}
    inherited.update(options.overrides)
    inherited.update(env)
    env = inherited
    if complete.is_file():
        previous = json.loads((out / "run.json").read_text())
        operational = {"HYBRID_OUT", "HYBRID_COMPRESS"}
        recorded = {k: v for k, v in previous["environment"].items() if k not in operational}
        requested = {k: v for k, v in env.items() if k not in operational}
        if (any(previous.get(k) != v for k,v in spec.items())
                or recorded != requested or not checked_completion(out,spec)):
            raise ValueError("conflicting or incomplete previous result; use a fresh --out directory: " + str(out))
        return True
    write_json(out / "run.json", dict(spec, command=command, environment=env,
                                      machine="two-core geometry, one idle core for solo",
                                      host=socket.gethostname()))
    start = time.monotonic()
    print("start " + spec["run_id"], flush=True)
    with (out / "stdout.txt").open("w") as log:
        try:
            status = subprocess.call(command, cwd=str(out), stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, **env))
        except OSError as exc:
            log.write(str(exc) + "\n")
            status = 126
    seconds = round(time.monotonic() - start, 3)
    if status == 0:
        try:
            status = 0 if checked_completion(out, spec) else 65
        except (OSError, ValueError, KeyError):
            status = 65
    result = dict(exit_code=status, duration_s=seconds, run_id=spec["run_id"])
    failed = out / "failed.json"
    write_json(complete if status == 0 else failed, result)
    if status == 0 and failed.exists():
        failed.unlink()
    print("{} {} ({:.1f}s)".format("done" if status == 0 else "failed", spec["run_id"], seconds), flush=True)
    return status == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path, default=ROOT / "raw/build-2/pythia/bin/champsim")
    parser.add_argument("--out", type=Path, default=ROOT / "raw/runs")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--compress", type=int, choices=[0, 1], default=1)
    parser.add_argument("--group", action="append", choices=["spec", "warmup", "synthetic"])
    parser.add_argument("--run", action="append", help="specific run IDs, may repeat")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                        help="record and apply intervention-specific simulator settings")
    parser.add_argument("--list", action="store_true")
    options = parser.parse_args()
    options.overrides = {}
    for value in options.env:
        key, sep, content = value.partition("=")
        if not sep or not key:
            parser.error("--env expects KEY=VALUE")
        if key in ("HYBRID_OUT", "LLC_PF_POLICY", "HYBRID_ACTIVE_CORES", "HYBRID_SKIPS", "HYBRID_COMPRESS", "HYBRID_PAGE_SLOTS"):
            parser.error("matrix field cannot be overridden by --env: " + key)
        options.overrides[key] = content
    options.out = options.out.resolve()
    options.bin = options.bin.resolve()
    specs = [s for s in run_specs() if (not options.group or s["group"] in options.group)
             and (not options.run or s["run_id"] in options.run)]
    if not specs:
        parser.error("selection matched no runs")
    if options.list:
        print("\n".join(s["run_id"] for s in specs))
    else:
        if not options.bin.is_file():
            parser.error("build the simulation binary first: " + str(options.bin))
        with ThreadPoolExecutor(max_workers=options.jobs) as executor:
            success = list(executor.map(lambda s: one(s, options), specs))
        sys.exit(0 if all(success) else 1)
