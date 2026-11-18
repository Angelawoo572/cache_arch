#!/usr/bin/env python3
"""Run bounded-window, mapping, lifecycle and intervention checks on synthetic traces."""

import argparse
import csv
from collections import Counter
from pathlib import Path
import os
import struct
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--binary", type=Path, default=root / "raw/build-2/pythia/bin/champsim"
)
args = parser.parse_args()
record = struct.Struct("<QBB2B4B2Q4Q")


def write_trace(path, base, jump):
    with path.open("wb") as f:
        for i in range(30000):
            address = base + (i // 4 // 8 * jump + i // 4 % 8) * 64 if i % 4 == 1 else 0
            f.write(
                record.pack(
                    0x400000 + (i % 4) * 4,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    address,
                    0,
                    0,
                    0,
                )
            )


def rows(path, name):
    with (path / (name + ".csv")).open() as f:
        return list(csv.DictReader(f))


def run(
    directory,
    name,
    traces,
    policy,
    warm=5000,
    count=10000,
    skip=0,
    slots=None,
    plan=None,
):
    output = directory / name
    output.mkdir()
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("HYBRID_", "LLC_PF_"))
    }
    env.update(
        HYBRID_OUT=str(output),
        HYBRID_ACTIVE_CORES=str(len(traces)),
        LLC_PF_POLICY=policy,
        HYBRID_SKIPS=",".join([str(skip)] * len(traces)),
    )
    if slots:
        env["HYBRID_PAGE_SLOTS"] = slots
    if plan:
        env["HYBRID_PLAN"] = str(plan)
    with (output / "run.log").open("w") as log:
        subprocess.run(
            [
                str(args.binary.resolve()),
                "--warmup_instructions=" + str(warm),
                "--simulation_instructions=" + str(count),
                "-traces",
                *map(str, traces),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    assert all(
        int(r["warmup"]) == warm and int(r["instructions"]) == count
        for r in rows(output, "cores")
    )
    assert len(rows(output, "cores")) == len(traces)
    for r in rows(output, "loads"):
        assert skip + warm < int(r["instr_id"]) <= skip + warm + count, r
    return output


def mapping(run_dir, core):
    return {
        r["virtual_page"]: r["physical_page"]
        for r in rows(run_dir, "pages")
        if r["cpu"] == str(core)
    }


def stream(run_dir):
    return [
        (r["cpu"], r["instr_id"], r["operand"], r["pc"], r["virtual_line"])
        for r in rows(run_dir, "loads")
        if r["level"] == "CORE"
    ]


with tempfile.TemporaryDirectory(dir=root / "raw", prefix="check-") as tmp:
    directory = Path(tmp)
    a, b = directory / "stream-a.trace", directory / "stream-b.trace"
    write_trace(a, 0x40000000, 16)
    write_trace(b, 0x60000000, 64)
    baseline = run(directory, "noPF", [a, b], "nopf")
    stride = run(directory, "stride", [a, b], "stride")
    short = run(directory, "short", [a, b], "nopf", warm=2500, skip=2500)
    solo = run(directory, "solo", [b], "nopf", slots="1")
    # Cross-core issue order can differ; every core's ordered dynamic demand stream must agree.
    assert sorted(stream(stride)) == sorted(stream(baseline)) == sorted(stream(short))
    assert not rows(baseline, "candidates")
    assert mapping(short, 0) == mapping(baseline, 0) == mapping(stride, 0)
    assert (
        mapping(short, 1)
        == mapping(baseline, 1)
        == mapping(stride, 1)
        == mapping(solo, 0)
    )
    assert not set(mapping(stride, 0).values()) & set(mapping(stride, 1).values())
    proposals = rows(stride, "candidates")
    r = next(
        r for r in proposals if r["stage"] == "proposed" and r["reason"] == "stride"
    )
    plan = directory / "plan.csv"
    plan.write_text(
        "cpu,instr_id,operand,delta\n"
        + ",".join([r["cpu"], r["instr_id"], r["operand"], "2"])
        + "\n"
    )
    intervention = run(directory, "intervention", [a, b], "stride", plan=plan)
    counts = Counter(r["stage"] for r in proposals)
    assert counts["accepted"] > 0
    assert counts["accepted"] == sum(
        counts[s] for s in ("timely", "late", "evicted", "redundant", "unresolved")
    )
    assert (
        sum(
            r["stage"] == "proposed" and r["reason"] == "intervention"
            for r in rows(intervention, "candidates")
        )
        == 1
    )
    print(
        "Passed fixed ROI, stable mappings, active-core isolation, complete Stride lifecycle, disabled NoPF, and intervention checks."
    )
