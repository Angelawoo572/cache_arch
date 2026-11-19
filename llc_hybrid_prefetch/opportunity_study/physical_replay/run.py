"""Run each fixed-work physical-layout control, recording its native lifecycle."""

import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parent
RAW = STUDY / "raw"
CASES = ("nopf", "stride", "plus128", "recent", "recent16")


def run(arm):
    perm, mode = arm
    dest = (
        Path(os.environ.get("WIDE_RESULTS", RAW / "physical_runs"))
        / ("perm" if perm else "original")
        / mode
    )
    if (dest / "complete.json").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    src = ROOT / "raw/runs" / ("cactus-r2-" + ("nopf" if mode == "nopf" else "stride"))
    meta = json.loads((src / "run.json").read_text())
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("WIDE_", "HYBRID_", "LLC_PF_"))
    }
    env.update(meta["environment"], HYBRID_OUT=str(dest))
    if perm:
        env["WIDE_PERMUTE"] = "1"
    if mode not in ("nopf", "stride"):
        env.update(
            WIDE_MODE="recent" if mode == "recent16" else mode,
            WIDE_TABLE=str(RAW / "history/cactus_live/r1_table.txt"),
        )
    if mode == "recent16":
        env["WIDE_BOUND"] = "16"
    args = [
        str(RAW / "physical_build/native/pythia/bin/champsim"),
        f"--warmup_instructions={meta['warmup']}",
        f"--simulation_instructions={meta['roi']}",
        "-traces",
        str(ROOT / "raw/traces" / (meta["traces"][0] + ".champsimtrace")),
    ]
    with (dest / "stdout.txt").open("w") as f:
        subprocess.run(
            args, cwd=dest, env=env, stdout=f, stderr=subprocess.STDOUT, check=True
        )
    with (dest / "cores.csv").open() as f:
        a = list(csv.DictReader(f))
    if len(a) != 1 or int(a[0]["instructions"]) != meta["roi"]:
        raise ValueError("work mismatch")
    (dest / "complete.json").write_text(
        json.dumps(dict(perm=perm, mode=mode, roi=meta["roi"])) + "\n"
    )


if __name__ == "__main__":
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(run, [(p, m) for p in (False, True) for m in CASES]))
