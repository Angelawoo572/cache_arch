"""Replay sparse earlier and near-demand offers through normal SPP admission and service."""

from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json
import os
import subprocess
from select_targets import CASES, FULL, HERE, RAW, RUNS, read


def replay(job):
    case, mode = job
    ref = json.loads((RUNS / (case + "__spp") / "complete.json").read_text())
    folder = RAW / (case + "__" + mode)
    folder.mkdir(exist_ok=True, parents=True)
    marker = folder / "complete.json"
    plan = HERE / (case + "_" + mode + ".csv")
    data = dict(
        case=case,
        mode=mode,
        reference="spp",
        plan=list(read(plan)),
        config=ref["config"],
    )
    if marker.exists():
        if json.loads(marker.read_text()) != data:
            raise ValueError("A different diagnostic already exists")
        return
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("STUDY_", "FULL_", "HYBRID_", "LLC_PF_"))
    }
    env.update(ref["environment"], HYBRID_PLAN=str(plan), HYBRID_OUT=str(folder))
    trace = FULL.parents[1] / ref["config"]["traces"][0]
    with (folder / "run.log").open("w") as log:
        subprocess.run(
            [
                str(FULL / "raw/native/build/pythia/bin/champsim"),
                "--warmup_instructions=50000000",
                "--simulation_instructions=200000000",
                "-traces",
                str(trace),
            ],
            env=env,
            stdout=log,
            stderr=log,
            check=True,
        )
    if int(next(read(folder / "cores.csv"))["instructions"]) != 200_000_000:
        raise ValueError("Partial diagnostic execution")
    marker.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(replay, product(CASES, ("early", "near"))))
