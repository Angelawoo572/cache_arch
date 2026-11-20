"""Apply one prospectively chosen held-out forecast to an otherwise unchanged Stride run."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY / "trials"))
from select_targets import dump, read

BASE = STUDY / "raw/capture/runs/bfs_discovery__stride"
RAW = STUDY / "raw/one_offer"
OFFERS = STUDY / "probes/prospective_offers.csv"


def replay(row, baseline=BASE, destination=RAW):
    case = row["scenario"]
    if not case.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Invalid experiment name")
    src, target = int(row["cue_line"]), int(row["predicted_line"])
    if src >> 6 != target >> 6 or src == target:
        raise ValueError(
            "Native single-override control requires a nonzero same-page displacement"
        )
    folder = destination / case
    folder.mkdir(exist_ok=True, parents=True)
    plan = folder / "offer.csv"
    meta = json.loads((baseline / "complete.json").read_text())
    config = dict(source=meta["config"], offer=row, inference_cycles=0)
    marker = folder / "complete.json"
    if marker.is_file():
        if json.loads(marker.read_text()) != config:
            raise ValueError("Forecast differs from completed run")
        return
    dump(
        plan,
        [
            dict(
                cpu=0,
                instr_id=int(row["instr_id"]),
                operand=int(row["operand"]),
                delta=target - src,
            )
        ],
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("HYBRID_", "LLC_PF_", "FULL_", "STUDY_"))
    }
    env.update(meta["environment"], HYBRID_PLAN=str(plan), HYBRID_OUT=str(folder))
    cfg = meta["config"]
    cmd = [
        str(STUDY / "raw/capture/build/pythia/bin/champsim"),
        f"--warmup_instructions={cfg['warmup']}",
        f"--simulation_instructions={cfg['roi']}",
        "-traces",
        str(STUDY.parents[1] / cfg["traces"][0]),
    ]
    with (folder / "run.log").open("w") as f:
        subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    stats = {r["key"]: int(r["value"]) for r in read(folder / "stats.csv")}
    if int(next(read(folder / "cores.csv"))["instructions"]) != cfg["roi"] or (
        stats["plan_anchors"],
        stats["plan_matched"],
    ) != (1, 1):
        raise ValueError("Forecast injection or fixed-work validation failed")
    marker.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(replay, read(OFFERS)))
