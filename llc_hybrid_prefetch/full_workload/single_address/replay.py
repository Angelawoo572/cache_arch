"""One explicitly retrospective issue intervention on two complete fixed-work baselines."""

import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
FULL = HERE.parent
CACHE = FULL.parents[1]
RAW = FULL / "raw/single_address"
RUNS = FULL / "raw/native/runs"
BIN = FULL / "raw/native/build/pythia/bin/champsim"
ANCHOR, TARGET = 100100888, 100101506


def table(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def run(arm):
    base = RUNS / ("bfs_discovery__" + arm)
    orig = json.loads((base / "complete.json").read_text())
    cfg = orig["config"]
    folder = RAW / arm
    folder.mkdir(exist_ok=True, parents=True)
    config = dict(
        cfg, override=dict(trigger_instruction=ANCHOR, operand=0, signed_delta=3)
    )
    marker = folder / "complete.json"
    if not marker.is_file():
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("STUDY_", "FULL_", "HYBRID_", "LLC_PF_"))
        }
        env.update(
            orig["environment"],
            HYBRID_PLAN=str(HERE / "anchor.csv"),
            HYBRID_OUT=str(folder),
        )
        cmd = [
            str(BIN),
            "--warmup_instructions=50000000",
            "--simulation_instructions=200000000",
            "-traces",
            str(CACHE / cfg["traces"][0]),
        ]
        with (folder / "run.log").open("w") as f:
            subprocess.run(
                cmd, cwd=folder, env=env, stdout=f, stderr=subprocess.STDOUT, check=True
            )
        stats = {r["key"]: int(r["value"]) for r in table(folder / "stats.csv")}
        if (stats["plan_matched"], stats["plan_anchors"]) != (1, 1) or int(
            table(folder / "cores.csv")[0]["instructions"]
        ) != 200_000_000:
            raise ValueError("One matched override and fixed-work retirement required")
        marker.write_text(json.dumps(config, indent=2) + "\n")
    if json.loads(marker.read_text()) != config:
        raise ValueError("Different intervention configuration")

    def witness(p):
        with gzip.open(p / "candidates.csv.gz", "rt") as f:
            events = [
                r
                for r in csv.DictReader(f)
                if r["instr_id"] == str(ANCHOR) and r["operand"] == "0"
            ]
        native_offer = next(
            (r["target"] for r in events if r["stage"] == "proposed"), ""
        )
        lifecycle = ",".join(r["stage"] for r in events)
        with gzip.open(p / "head.csv.gz", "rt") as f:
            wait = sum(
                int(r["cycles"])
                for r in csv.DictReader(f)
                if (int(r["instr_id"]), int(r["operand"])) == (TARGET, 0)
            )
        return native_offer, lifecycle, wait

    fixed = table(folder / "cores.csv")[0]
    original_cycles = int(table(base / "cores.csv")[0]["cycles"])
    ref_offer, ref_life, ref_head = witness(base)
    actual_offer, actual_life, actual_head = witness(folder)
    if int(actual_offer) != 3500606:
        raise ValueError("Issued wrong physical target in intervention")
    paths = [RUNS / "bfs_discovery__nopf", base, folder]
    readers = [gzip.open(p / "outcomes_core0.u16le.gz", "rb") for p in paths]
    gains = losses = 0
    try:
        for start in range(0, 200_000_000, 131072):
            n = min(131072, 200_000_000 - start)
            blocks = [f.read(2 * n) for f in readers]
            prior = min(n, max(0, 50_000_000 - start))
            if prior and blocks[1][: 2 * prior] != blocks[2][: 2 * prior]:
                raise ValueError(
                    "Dynamic demand outcomes changed before the intervention quarter"
                )
            if any(len(x) != 2 * n for x in blocks):
                raise EOFError("Incomplete outcomes")
            a, b, c = [int.from_bytes(block, "little") for block in blocks]
            original = (a & int.from_bytes(b"\x22" * (2 * n), "little")) >> 1
            old = (b >> 3) & original
            new = (c >> 3) & original
            gains += (new & ~old).bit_count()
            losses += (old & ~new).bit_count()
            pos = TARGET - 1 - 50_000_000 - start
            if 0 <= pos < n:
                states = [block[2 * pos] & 15 for block in blocks]
        if any(f.read(1) for f in readers):
            raise ValueError("Excess outcomes")
    finally:
        for f in readers:
            f.close()
    base_quarters = table(base / "phases.csv")
    trial_quarters = table(folder / "phases.csv")
    savings_by_quarter = [
        int(b["cycles"]) - int(a["cycles"])
        for b, a in zip(base_quarters, trial_quarters)
    ]
    if savings_by_quarter[0] or sum(savings_by_quarter) != original_cycles - int(
        fixed["cycles"]
    ):
        raise ValueError("Cycle totals disagree with the pre-intervention reference")
    return dict(
        quarter_cycle_savings=" ".join(map(str, savings_by_quarter)),
        reference=arm,
        anchor=ANCHOR,
        target=TARGET,
        original_offer=ref_offer,
        override_offer=actual_offer,
        reference_lifecycle=ref_life,
        override_lifecycle=actual_life,
        nopf_target_mask=states[0],
        reference_target_mask=states[1],
        override_target_mask=states[2],
        reference_target_head=ref_head,
        override_target_head=actual_head,
        original_timely_gained=gains,
        original_timely_lost=losses,
        baseline_cycles=original_cycles,
        intervention_cycles=fixed["cycles"],
        cycles_saved=original_cycles - int(fixed["cycles"]),
    )


if __name__ == "__main__":
    with ThreadPoolExecutor(2) as pool:
        rows = list(pool.map(run, ("stride", "spp")))
    with (HERE / "results.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
