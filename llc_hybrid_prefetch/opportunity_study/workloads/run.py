"""Run the registered opportunity cases using the validated Phase I simulator."""

import argparse, csv, fcntl, json, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from cases import BINARY, RAW, TRACES, specs


def run(task):
    spec, pol = task
    rid = spec["case"] + "-" + pol
    dest = RAW / "runs" / rid
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (dest / "complete.json").exists():
            return
        files = [TRACES / (n + ".champsimtrace") for n in spec["traces"]]
        for p in files:
            if p.stat().st_size < 64 * (spec["skip"] + spec["warmup"] + spec["roi"] + 1):
                raise ValueError("short " + str(p))
        env = dict(
            HYBRID_OUT=str(dest),
            LLC_PF_POLICY=pol,
            HYBRID_ACTIVE_CORES=str(len(files)),
            HYBRID_PAGE_SLOTS=",".join(map(str, spec["slots"])),
            HYBRID_COMPRESS="1",
            HYBRID_SKIPS=",".join([str(spec["skip"])] * len(files)),
        )
        cmd = [
            str(BINARY),
            "--warmup_instructions=" + str(spec["warmup"]),
            "--simulation_instructions=" + str(spec["roi"]),
            "-traces",
        ] + [str(f) for f in files]
        (dest / "run.json").write_text(
            json.dumps(
                dict(spec, run_id=rid, policy=pol, command=cmd, environment=env, region_rank=None),
                indent=2,
            )
            + "\n"
        )
        start = time.monotonic()
        with (dest / "stdout.txt").open("w") as output:
            subprocess.run(
                cmd,
                env=dict(os.environ, **env),
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
                cwd=dest,
            )
        with (dest / "cores.csv").open() as src:
            cs = list(csv.DictReader(src))
        if len(cs) != len(files) or any(
            int(r["warmup"]) != spec["warmup"]
            or int(r["instructions"]) != spec["roi"]
            or int(r["cycles"]) <= 0
            for r in cs
        ):
            raise ValueError("retirement mismatch " + rid)
        tmp = dest / "complete.tmp"
        tmp.write_text(
            json.dumps(dict(run_id=rid, exit_code=0, duration_s=round(time.monotonic() - start, 3)))
            + "\n"
        )
        tmp.replace(dest / "complete.json")
        print("finished " + rid, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--group", action="append")
    p.add_argument("--case", action="append")
    p.add_argument("--jobs", type=int, default=6)
    opt = p.parse_args()
    tasks = [
        (s, p)
        for s in specs()
        if (not opt.group or s["group"] in opt.group) and (not opt.case or s["case"] in opt.case)
        for p in ("nopf", "stride")
    ]
    if not tasks:
        raise ValueError("no matching cases")
    with ThreadPoolExecutor(opt.jobs) as executor:
        list(executor.map(run, tasks))
