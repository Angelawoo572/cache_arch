#!/usr/bin/env python3
"""Check detailed logging changes neither cycle counts nor summary counters."""

import argparse
import csv
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--binary", type=Path, default=root / "raw/build-2/pythia/bin/champsim"
)
parser.add_argument("--output", type=Path, default=root / "results/logging_control.csv")
args = parser.parse_args()
traces = root / "raw/traces"
workloads = [
    traces / "605.mcf_s-665B.champsimtrace",
    traces / "607.cactuBSSN_s-2421B.champsimtrace",
]
records = []
for n in (1, 2):
    for policy in ("nopf", "stride"):
        summaries = []
        for detail in ("1", "0"):
            dest = root / "raw/logging_control" / f"{n}core-{policy}-detail{detail}"
            dest.mkdir(parents=True, exist_ok=True)
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith(("HYBRID_", "LLC_PF_"))
            }
            env.update(
                HYBRID_OUT=str(dest),
                HYBRID_ACTIVE_CORES=str(n),
                HYBRID_COMPRESS="1",
                HYBRID_DETAIL=detail,
                LLC_PF_POLICY=policy,
            )
            with (dest / "stdout.txt").open("w") as log:
                subprocess.run(
                    [
                        str(args.binary.resolve()),
                        "--warmup_instructions=100000",
                        "--simulation_instructions=250000",
                        "-traces",
                        *map(str, workloads[:n]),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            with (dest / "cores.csv").open() as f:
                cores = list(csv.DictReader(f))
            with (dest / "stats.csv").open() as f:
                totals = {r["key"]: int(r["value"]) for r in csv.DictReader(f)}
            assert (dest / "loads.csv.gz").is_file() == (detail == "1")
            summaries.append((cores, totals))
        on, off = summaries
        if on != off:
            raise ValueError(f"{n} core(s), {policy}: logging changed simulation state")
        for first, second in zip(on[0], off[0]):
            records.append(
                dict(
                    active_cores=n,
                    policy=policy,
                    core=first["cpu"],
                    warmup=first["warmup"],
                    roi=first["instructions"],
                    cycles_logging_on=first["cycles"],
                    cycles_logging_off=second["cycles"],
                    equal_summary_counters=len(on[1]),
                    accepted_prefetches=on[1]["prefetch_accepted"],
                    dram_bytes=sum(
                        on[1][k]
                        for k in (
                            "dram_demand_bytes",
                            "dram_prefetch_bytes",
                            "dram_write_bytes",
                        )
                    ),
                )
            )
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=records[0])
    w.writeheader()
    w.writerows(records)
