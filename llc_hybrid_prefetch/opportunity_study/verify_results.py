"""Verify the paired experiments and their publication tables from the archived runs."""

import argparse
import csv
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parent
RAW = STUDY / "raw"


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main(repeat):
    cases = json.loads((STUDY / "workloads/cases.json").read_text())
    inventory = {case["case"]: case for case in cases}
    if len(cases) != 31 or len(inventory) != len(cases):
        raise ValueError("registered matrix changed")
    for name, spec in inventory.items():
        for mode in ("nopf", "stride"):
            run = RAW / "runs" / (name + "-" + mode)
            assert json.loads((run / "complete.json").read_text())["exit_code"] == 0
            cores = table(run / "cores.csv")
            assert len(cores) == len(spec["traces"])
            assert all(
                int(c["instructions"]) == spec["roi"]
                and int(c["warmup"]) == spec["warmup"]
                for c in cores
            )
    print(
        f"{2*len(inventory)} paired-policy workload runs met their registered retirement bounds"
    )

    rows = table(STUDY / "deadlines/replay_results.csv")
    assert len(rows) == 24
    for row in rows:
        assert (
            int(row["overrides"])
            >= int(row["accepted"])
            >= int(row["timely"]) + int(row["late"])
        )
        assert (
            int(row["timely"])
            >= int(row["direct_timely_on_paired_misses"])
            >= int(row["original_late_now_timely"])
            >= int(row["confirmed_earlier_anchor"])
        )
        assert int(row["cycles_saved"]) == int(row["stride_makespan"]) - int(
            row["makespan"]
        )
        if row["memory"] == "noop":
            assert int(row["cycles_saved"]) == int(row["overrides"]) == 0
    print("24 frozen-cue runs passed timing and first-use accounting")

    rows = table(STUDY / "physical_replay/placement_results.csv")
    assert len(rows) == 10
    for row in rows:
        p = RAW / "physical_runs" / row["layout"] / row["mode"]
        assert int(table(p / "cores.csv")[0]["cycles"]) == int(row["cycles"])
        assert int(row["direct_timely_on_original_misses"]) <= int(
            row["paired_miss_to_hit"]
        )
        if repeat:
            for name in ("cores.csv", "stats.csv"):
                if table(p / name) != table(
                    repeat / row["layout"] / row["mode"] / name
                ):
                    raise ValueError(
                        f"{repeat.name} disagrees: {row['layout']}, {row['mode']}, {name}"
                    )
    print(
        "10 physical-layout controls passed"
        + (" including independent repeat" if repeat else "")
    )

    modes = {r["mode"]: r for r in table(STUDY / "page_memory/results.csv")}
    shadow = modes["shadow"]
    assert int(shadow["grants"]) == 1460 and int(shadow["cycles_saved_vs_stride"]) == 0
    for mode, row in modes.items():
        if mode not in ("shadow", "nopf"):
            assert int(row["grants"]) == int(shadow["grants"])
            assert int(row["cycles_saved_vs_stride"]) == int(shadow["cycles"]) - int(
                row["cycles"]
            )
    print(
        f"{len(modes)} page-memory results passed fixed-grant and timing checks, including the canonical NoPF row"
    )

    native = table(STUDY / "conventional/native_validation.csv")
    assert len(native) == 58 and all(
        r["cycles"] == r["canonical_cycles"] for r in native
    )
    controls = table(STUDY / "conventional/policy_results.csv")
    assert len(controls) == 352
    crit = table(STUDY / "conventional/criticality.csv")
    assert {r["group"] for r in crit} == {"high", "zero"}
    assert all(
        int(r["pairs"])
        == int(r["accepted_overrides"])
        == int(r["timely_intended"]) + int(r["late_intended"])
        == 96
        for r in crit
    )
    print(
        "352 conventional configurations, 58 baseline core comparisons, and two groups of 96 matched interventions verified"
    )


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--repeat", type=Path)
    main(a.parse_args().repeat)
