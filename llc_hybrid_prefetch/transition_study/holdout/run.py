"""Replay the two unexamined BFS windows on the archived LLC machine and form paired records."""

from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
PROJECT = STUDY.parent
RAW = STUDY / "raw"
OLD = PROJECT / "opportunity_study"
sys.path.insert(0, str(OLD / "history"))
from extract import FIELDS, parse_arm


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def run(case, mode):
    name = case + "_mid-" + mode
    dest = RAW / "runs" / name
    if (dest / "complete.json").is_file():
        return
    dest.mkdir(parents=True, exist_ok=True)
    trace = case.replace("gap_bfs", "bfs-") + "-long"
    file = OLD / "raw/traces" / (trace + ".champsimtrace")
    if not file.exists() or file.stat().st_size // 64 < 27_000_001:
        raise ValueError("short input")
    args = [
        str(PROJECT / "raw/build-2/pythia/bin/champsim"),
        "--warmup_instructions=1000000",
        "--simulation_instructions=2000000",
        "-traces",
        str(file),
    ]
    settings = dict(
        HYBRID_ACTIVE_CORES="1",
        HYBRID_PAGE_SLOTS="0",
        HYBRID_SKIPS="24000000",
        HYBRID_COMPRESS="1",
        HYBRID_OUT=str(dest),
        LLC_PF_POLICY=mode,
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("HYBRID_", "LLC_PF_", "STUDY_"))
    }
    meta = dict(
        case=case + "_mid",
        run_id=name,
        group="real",
        policy=mode,
        traces=[trace],
        workloads=["bfs"],
        slots=[0],
        warmup=1_000_000,
        roi=2_000_000,
        skip=24_000_000,
        environment=settings,
        command=args,
    )
    (dest / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    with (dest / "stdout.txt").open("w") as log:
        subprocess.run(
            args,
            env=dict(env, **settings),
            cwd=dest,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    cores = table(dest / "cores.csv")
    assert (
        len(cores) == 1
        and int(cores[0]["instructions"]) == 2_000_000
        and int(cores[0]["warmup"]) == 1_000_000
    )
    (dest / "complete.json").write_text(
        json.dumps(dict(run_id=name, exit_code=0)) + "\n"
    )


def pages(folder):
    with gzip.open(folder / "pages.csv.gz", "rt") as f:
        return {
            (r["cpu"], r["virtual_page"]): (r["page_slot"], r["physical_page"])
            for r in csv.DictReader(f)
        }


def source_operands(folder):
    with gzip.open(folder / "loads.csv.gz", "rt") as f:
        lines = csv.reader(f)
        h = {key: i for i, key in enumerate(next(lines))}
        for row in lines:
            if row[h["level"]] == "CORE" and row[h["stage"]] == "trace":
                yield tuple(
                    row[h[k]]
                    for k in ("cpu", "instr_id", "operand", "pc", "virtual_line")
                )


def summarize(case):
    from itertools import zip_longest

    nopf, stride = (
        RAW / "runs" / (case + "_mid-" + mode) for mode in ("nopf", "stride")
    )
    if pages(nopf) != pages(stride):
        raise ValueError("page mapping mismatch")
    count = 0
    for a, b in zip_longest(source_operands(nopf), source_operands(stride)):
        if a != b:
            raise ValueError("input identity mismatch")
        count += 1
    baseline = parse_arm(nopf)
    records, traces = parse_arm(stride, baseline)
    z = np.array(records[0], dtype=np.int64).reshape(-1, len(FIELDS))
    data = {k: z[:, i] for i, k in enumerate(FIELDS)}
    d = RAW / "history"
    d.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(d / (case + "_mid-c0.npz"), trace=traces[0], **data)

    def cycles(dest):
        return int(table(dest / "cores.csv")[0]["cycles"])

    return dict(
        case=case + "_mid",
        aligned_input_loads=count,
        virtual_pages=len(pages(nopf)),
        nopf_demand_llc_misses=sum(1 for _, _, hit in baseline[0].values() if hit == 0),
        stride_ingresses=len(z),
        stride_distinct_trigger_pcs=len(set(data["pc"])),
        attributed_timely=int(
            (
                (data["baseline"] == 0) & (data["hit"] == 1) & (data["pf_kind"] == 1)
            ).sum()
        ),
        attributed_late=int(
            (
                (data["baseline"] == 0) & (data["hit"] == 0) & (data["pf_kind"] == 2)
            ).sum()
        ),
        nopf_cycles=cycles(nopf),
        stride_cycles=cycles(stride),
    )


if __name__ == "__main__":
    cases = ("gap_bfs3", "gap_bfs8")
    with ThreadPoolExecutor(4) as pool:
        tasks = [pool.submit(run, c, m) for c in cases for m in ("nopf", "stride")]
        for job in tasks:
            job.result()
    with ThreadPoolExecutor(2) as pool:
        data = list(pool.map(summarize, cases))
    with (HERE / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(data)
    print("Verified four fixed-work arms and the two input-operand streams")
