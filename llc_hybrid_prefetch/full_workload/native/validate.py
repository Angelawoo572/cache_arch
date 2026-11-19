"""Differentially test passive instrumentation against the original simulator's fully logged execution."""

import argparse
import csv
import gzip
import json
import os
import subprocess
from array import array
from build import SOURCE
from run import BASE, RAW, HERE, run, rows

TRACE = BASE / "raw/traces/605.mcf_s-994B.champsimtrace"
WARM, ROI, SKIP = 100000, 350000, 4800000


def gzrows(path):
    with gzip.open(path, "rt") as f:
        yield from csv.DictReader(f)


def validate(policy):
    case = "parity_registered"
    detailed = run(case, policy, [TRACE], [0], WARM, ROI, SKIP, ROI)
    old = RAW / "reference" / policy
    old.mkdir(exist_ok=True, parents=True)
    settings = json.loads((detailed / "complete.json").read_text())["environment"]
    clean = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("STUDY_", "HYBRID_", "FULL_", "LLC_PF_"))
    }
    with (old / "run.log").open("w") as log:
        subprocess.run(
            [
                str(SOURCE / "bin/champsim"),
                f"--warmup_instructions={WARM}",
                f"--simulation_instructions={ROI}",
                "-traces",
                str(TRACE),
            ],
            env={**clean, **settings, "HYBRID_OUT": str(old)},
            stdout=log,
            stderr=log,
            check=True,
        )
    for name in (
        "cores.csv",
        "stats.csv",
        "loads.csv.gz",
        "candidates.csv.gz",
        "head.csv.gz",
        "pages.csv.gz",
    ):
        if (old / name).read_bytes() != (detailed / name).read_bytes():
            raise ValueError(
                f"Original and passive native instrumentation disagree: {name} ({policy})"
            )
    expected = array("H", [0]) * ROI
    actual = gzip.decompress((detailed / "outcomes_core0.u16le.gz").read_bytes())
    phases = rows(detailed / "phases.csv")
    counts = [[0] * 5 for _ in range(4)]
    for r in gzrows(old / "loads.csv.gz"):
        pos = int(r["instr_id"]) - SKIP - WARM - 1
        q = pos * 4 // ROI
        if r["stage"] == "lookup" and r["level"] in ("LLC", "L2"):
            hit = int(r["hit"])
            ix = (0 if r["level"] == "L2" else 2) + 1 - hit
            counts[q][ix] += 1
            if r["level"] == "LLC":
                expected[pos] |= (1 if hit else 2) << (4 * int(r["operand"]))
        if r["stage"] == "ingress" and r["level"] == "LLC":
            counts[q][4] += 1
    for q, r in enumerate(phases):
        keys = (
            "l2_tag_hits",
            "l2_tag_misses",
            "llc_tag_hits",
            "llc_tag_misses",
            "llc_ingress",
        )
        if counts[q] != [int(r[k]) for k in keys]:
            raise ValueError("Quarter counts disagree with reference")
    for r in gzrows(old / "candidates.csv.gz"):
        k = r["stage"]
        if k in ("timely", "late"):
            pos = int(r["demand_instr_id"]) - SKIP - WARM - 1
            expected[pos] |= (8 if k == "timely" else 4) << (
                4 * int(r["demand_operand"])
            )
    if __import__("sys").byteorder == "big":
        expected.byteswap()
    if actual != expected.tobytes():
        raise ValueError("Dynamic operand stream diverged from fully detailed log")
    return dict(
        policy=policy,
        warmup=WARM,
        roi=ROI,
        skip=SKIP,
        cycles=rows(old / "cores.csv")[0]["cycles"],
        llc_tag_misses=sum(x[3] for x in counts),
        original_core_stats_and_logs_identical=True,
        all_u16le_operands_match_reference=True,
    )


def temporal_positive_control():
    # The existing phase-one workload repeatedly walks nineteen previously mapped conflict lines.
    conventional = BASE / "opportunity_study"
    trace = conventional / "raw/conventional/traces/region_conflict_cycle.champsimtrace"
    reference = next(
        r
        for r in rows(conventional / "conventional/temporal_fixture.csv")
        if r["policy"] == "temporal"
    )
    result = run("temporal_positive_control", "temporal", [trace], [0], 285572, 357504)
    stats = {r["key"]: int(r["value"]) for r in rows(result / "stats.csv")}
    observed = dict(
        cycles=int(rows(result / "cores.csv")[0]["cycles"]),
        accepted=stats["prefetch_accepted"],
        crosspage_accepted=stats["study_accepted_crosspage"],
        timely=stats["study_pf_timely"],
        late=stats.get("study_pf_late", 0),
        dram_bytes=sum(
            stats[f"dram_{traffic}_bytes"]
            for traffic in ("demand", "prefetch", "write")
        ),
    )
    if not observed["timely"] or any(
        value != int(reference[key]) for key, value in observed.items()
    ):
        raise ValueError(
            "Positive temporal control diverged from committed measurement"
        )
    return dict(
        case="region_conflict_cycle",
        policy="temporal",
        matched_prior_result=True,
        **observed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-fixture", action="store_true")
    args = parser.parse_args()
    if args.temporal_fixture:
        data = [temporal_positive_control()]
        filename = "temporal_validation.csv"
    else:
        data = [validate(p) for p in ("nopf", "stride", "multi", "spp")]
        filename = "parity.csv"
    with (HERE / filename).open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
