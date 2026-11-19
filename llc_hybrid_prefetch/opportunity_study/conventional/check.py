"""Validate unchanged native timing and one-physical-line conflict handling on the new binary."""

import csv
import os
import subprocess

from run import BASE, HERE, RAW, csvs, grid, totals

binary = RAW / "build/pythia/bin/champsim"
fixture = RAW / "dual_issuer"
fixture.mkdir(exist_ok=True, parents=True)
s1 = BASE / "raw/traces/synthetic-steady.champsimtrace"
s2 = BASE / "raw/traces/synthetic-nested_samepage.champsimtrace"
env = {k: v for k, v in os.environ.items() if not k.startswith(("STUDY_", "HYBRID_", "LLC_PF_"))}
env.update(
    HYBRID_OUT=str(fixture),
    HYBRID_DETAIL="1",
    HYBRID_ACTIVE_CORES="2",
    LLC_PF_POLICY="nopf",
    STUDY_DUAL_ISSUER_TEST="1",
)
with (fixture / "run.log").open("w") as log:
    subprocess.run(
        [
            str(binary),
            "--warmup_instructions=1000",
            "--simulation_instructions=20000",
            "-traces",
            str(s1),
            str(s2),
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        check=True,
    )
s = totals(fixture)
m = {stage: s["study_fixture_" + stage + "_cycle"] for stage in ("accepted", "pq", "mshr", "resident")}
assert s["prefetch_accepted"] == 1 and m["accepted"] == m["pq"] < m["mshr"] < m["resident"]
log = csvs(fixture / "candidates.csv")
trace = [r for r in log if r["stage"] in ("accepted", "rejected")]
assert (
    len(trace) == 4
    and len({r["target"] for r in trace}) == 1
    and [r["cpu"] for r in trace] == ["0", "1", "1", "1"]
)
with (HERE / "duplicate_fixture.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=("core", "line", "cycle", "result", "reason"))
    w.writeheader()
    for r in trace:
        w.writerow(
            dict(core=r["cpu"], line=r["target"], cycle=r["cycle"], result=r["stage"], reason=r["reason"])
        )
out = []
groups = ("open", "newreal", "synthetic", "followup", "lategraph")
cases = sorted({case for group in groups for case, policy, *unused in grid(group) if policy == "nopf"})
for case in cases:
    for policy in ("nopf", "stride"):
        canonical = (
            BASE / "raw/qualified"
            if case in ("mcf-cactus-r1", "mcf-r2", "lbm-xalan-r1")
            else HERE.parent / "raw/runs"
        ) / f"{case}-{policy}"
        replay = RAW / "runs" / f"{case}__{policy}__open"
        if not (canonical / "cores.csv").is_file() or not (replay / "complete.json").is_file():
            raise ValueError("missing completed native replay: " + str(replay))
        a, b = csvs(canonical / "cores.csv"), csvs(replay / "cores.csv")
        assert a == b, (case, policy, a, b)
        x, y = totals(canonical), totals(replay)
        matched = [k for k in x if not k.startswith("study_")]
        assert set(matched) <= y.keys()
        assert all(x[k] == y[k] for k in matched), (
            case,
            policy,
            {k: (x[k], y[k]) for k in matched if x[k] != y[k]},
        )
        out.extend(
            dict(
                case=case,
                policy=policy,
                cpu=r["cpu"],
                cycles=r["cycles"],
                canonical_cycles=q["cycles"],
                equal_native_counters=len(matched),
            )
            for r, q in zip(a, b)
        )
with (HERE / "native_validation.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=out[0])
    w.writeheader()
    w.writerows(out)
print("Validated live shared issuer and", len(out), "native core comparisons.")

# The complete comparison must include every declared arm exactly once.
from analyze import generate

generate()
public = {(r["case"], r["policy"], r["setting"]): r for r in csvs(HERE / "policy_results.csv")}
expected = {
    (c, p, setting)
    for group in (*groups, "budget", "timing", "strong_timing", "async", "promotion")
    for c, p, setting, limit, delay in grid(group)
}
for c in ("mcf-cactus-r1", "lbm-xalan-r1", "mcf-r2"):
    for p in ("stride", "look2", "multi", "spp", "union", "priority"):
        expected.add((c, p, "exact"))
for p in ("nopf", "stride", "temporal_sp", "temporal"):
    expected.add(("region_conflict_cycle", p, "open"))
assert expected == public.keys(), (expected - public.keys(), public.keys() - expected)
exact_arms = [r for r in csvs(HERE / "policy_by_core.csv") if r["setting"] == "exact"]
assert len(exact_arms) == 30 and all(r["cap_met"] == "1" for r in exact_arms)
for c, policy, setting in expected:
    if policy != "promote":
        continue
    p = public[c, policy, setting]
    resource = next(
        r
        for r in csvs(HERE / "resources.csv")
        if (r["case"], r["policy"], r["setting"]) == (c, policy, setting)
    )
    if int(resource["promotion_eligible"]) == 0:
        u = public[c, "union", setting]
        assert (p["cycles"], p["accepted"], p["bytes"]) == (u["cycles"], u["accepted"], u["bytes"])
targets = csvs(HERE / "criticality_examples.csv")
assert len(targets) == 192 and {(r["group"], r["pair"]) for r in targets} == {
    (g, str(i)) for g in ("high", "zero") for i in range(1, 97)
}
for r in targets:
    assert int(r["ready"]) <= int(r["accepted"]) <= int(r["dispatch"]) <= int(r["fill"])
    assert int(r["timely"]) + int(r["late"]) == 1
for r in csvs(HERE / "conventional_examples.csv"):
    assert tuple(map(int, (r["nopf_lookup"], r["stride_lookup"], r["policy_lookup"]))) == (0, 0, 1)
    assert int(r["accepted"]) <= int(r["dispatch"]) < int(r["fill"]) <= int(r["demand_hit"])
for r in csvs(HERE / "pollution.csv"):
    assert int(r["nopf_hit_policy_miss"]) == sum(
        int(r[k])
        for k in (
            "last_evictor_prefetch",
            "last_evictor_other",
            "last_change_refill",
            "no_line_change_observed",
        )
    )
print(
    "Validated",
    len(public),
    "comparison arms,",
    len(exact_arms),
    "per-core acceptance caps, timed interventions and paired examples.",
)
