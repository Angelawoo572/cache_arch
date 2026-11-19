"""Run matched, reset fixed-work interventions on certified complete-archive inputs."""

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pattern import HERE, RAW, table
from build import BIN

POLICIES = (
    "nopf",
    "stride",
    "multi",
    "spp",
    "temporal",
    "look2",
    "sparse1",
    "local",
    "ordered",
    "shape",
    "shadow",
    "look4",
    "sparse4",
)
CASES = (
    "mcf17_discovery",
    "pr_heldout",
    "bfs_discovery",
    "mcf17_transfer",
    "bfs_transfer",
    "mcf06_discovery",
    "soplex06_heldout",
)
SKIP, WARM, ROI = 2_000_000, 1_000_000, 500_000


def run(job):
    case, policy = job
    trace = HERE.parent / "raw/traces" / (case + ".champsimtrace")
    if not trace.is_file() or trace.stat().st_size < (SKIP + WARM + ROI + 256) * 64:
        raise ValueError(f"missing verified native input {trace}")
    folder = RAW / "runs" / case / policy
    folder.mkdir(exist_ok=True, parents=True)
    meta = dict(
        traces=[case],
        mode=policy,
        skip=SKIP,
        warmup=WARM,
        roi=ROI,
        active_cores=1,
        geometry_cores=2,
    )
    marker = folder / "complete.json"
    if marker.is_file():
        if json.loads(marker.read_text()) != meta:
            raise ValueError(f"configuration mismatch in {folder}")
        return
    native = policy in ("nopf", "stride", "multi", "spp", "look2", "look4", "temporal")
    environment = dict(
        HYBRID_ACTIVE_CORES="1",
        HYBRID_PAGE_SLOTS="0",
        HYBRID_COMPRESS="1",
        HYBRID_DETAIL="1",
        LLC_PF_POLICY="nopf" if policy == "nopf" else "stride",
        STUDY_EPOCH_LIMIT="0",
        STUDY_LATENCY="0",
        STUDY_FAST_NATIVE="0",
        STUDY_CHOICES="1",
        HYBRID_SKIPS=str(SKIP),
        HYBRID_OUT=str(folder),
        STUDY_POLICY=policy if native and policy != "nopf" else "stride",
    )
    if not native:
        environment["DISCOVERY_RULE"] = policy
        if policy in ("local", "ordered", "shape"):
            environment["DISCOVERY_FIT"] = str(HERE / (policy + ".rules"))
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("HYBRID_", "STUDY_", "LLC_PF_", "DISCOVERY_"))
    }
    args = [
        str(BIN),
        f"--warmup_instructions={WARM}",
        f"--simulation_instructions={ROI}",
        "-traces",
        str(trace),
    ]
    (folder / "run.json").write_text(
        json.dumps(dict(meta, environment=environment, command=args), indent=2) + "\n"
    )
    with (folder / "stdout.log").open("w") as f:
        subprocess.run(
            args,
            cwd=folder,
            env=dict(env, **environment),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    cores = table(folder / "cores.csv")
    assert (
        len(cores) == 1
        and int(cores[0]["warmup"]) == WARM
        and int(cores[0]["instructions"]) == ROI
    )
    marker.write_text(json.dumps(meta, indent=2) + "\n")
    print(case, policy, cores[0]["cycles"], flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    cli.add_argument("--policies", nargs="+", choices=POLICIES, default=POLICIES)
    cli.add_argument("--jobs", type=int, choices=[1, 2], default=2)
    a = cli.parse_args()
    with ThreadPoolExecutor(a.jobs) as pool:
        list(pool.map(run, [(c, p) for c in a.cases for p in a.policies]))
