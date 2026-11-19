"""Check committed summaries, actual first-use timing and fixed-work identity; no refitting."""

import argparse
from collections import Counter, defaultdict
from pattern import HERE, load, mix, table
from replay import CASES, POLICIES


def verify(partial=False):
    count = Counter()
    rules = load()
    for mode, r in rules.items():
        assert len(r) <= 256
        if mode in ("local", "ordered", "shape"):
            exported = {
                int(k): int(v)
                for k, v in (
                    line.split()
                    for line in (HERE / (mode + ".rules")).read_text().splitlines()
                )
            }
            assert exported == {mix(k): v[0] for k, v in r.items()}
        assert all(n >= 4 and y >= 2 and 5 * y >= n for _, y, n in r.values())
    full_cases = (*CASES, "pr_heldout_entire")
    for scope, expected, arms in (
        ("diagnostic", CASES, POLICIES),
        ("full", full_cases, ("nopf", "stride", "multi", "spp", "shape")),
    ):
        recorded = defaultdict(dict)
        for r in table(HERE / (scope + "_performance.csv")):
            c = r["case"]
            p = r["arm"]
            assert p not in recorded[c]
            recorded[c][p] = r
            n = int(r["instructions"])
            assert n == (
                500000
                if scope == "diagnostic"
                else (450000001 if c == "pr_heldout_entire" else 200000000)
            )
            assert int(r["cycles"]) > 0
            assert abs(n / int(r["cycles"]) - float(r["ipc"])) < 1e-12
        assert set(recorded) <= set(expected)
        if not partial:
            assert set(recorded) == set(expected), (
                scope,
                set(expected) - set(recorded),
            )
        for row in recorded.values():
            assert set(row) == set(arms)
            if scope == "diagnostic":
                assert row["stride"]["cycles"] == row["shadow"]["cycles"]
                for p in ("local", "ordered", "shape", "sparse1", "sparse4"):
                    r = row[p]
                    val = lambda k: int(r.get(k) or 0)
                    assert val("independent_legal") == val("agrees") + val(
                        "native_was_silent"
                    ) + val("different_native_proposal_displaced")
                    if "independent_different_proposals" in r:
                        assert val("independent_different_proposals") == val(
                            "different_native_proposal_displaced"
                        ) + val("native_was_silent")
                    assert val("different_native_proposal_displaced") == val(
                        "displaced_then_accepted"
                    ) + val("displaced_then_rejected")
        count[scope + "_native_arms"] = sum(map(len, recorded.values()))
        for r in table(HERE / (scope + "_first_uses.csv")):
            assert int(r["actual_timely"]) == int(r["kept"]) + int(r["gained"])
            assert int(r["ref_timely"]) == int(r["kept"]) + int(r["lost"])
        for r in table(HERE / (scope + "_native_examples.csv")):
            gen, dispatch, fill, lookup = [
                int(r[k]) for k in ("generation", "dispatch", "fill", "lookup")
            ]
            assert int(r["demand_ingress"]) - gen == int(r["lead"])
            if dispatch >= 0:
                assert dispatch >= gen
            if fill >= 0:
                assert fill <= lookup and fill >= gen
            assert int(r["cue_line"]) != int(r["target_line"])
            count[scope + "_native_examples"] += 1
    for r in table(HERE / "full_complete_load_alignment.csv"):
        assert int(r["instructions_joined"]) == (
            450000001 if r["case"] == "pr_heldout_entire" else 200000000
        )
        assert int(r["original_late_to_timely"]) <= int(r["original_timely_gained"])
        assert int(r["original_misses"]) >= sum(
            int(r[k])
            for k in (
                "original_timely_kept",
                "original_timely_gained",
                "original_timely_lost",
            )
        )
        count["whole_operand_joins"] += 1
    for r in table(HERE / "phase_reach.csv"):
        n = lambda k: int(r.get(k) or 0)
        assert (
            n("enough_lead")
            <= n("expressible")
            <= n("within_displacement")
            <= n("original_miss_targets")
        )
        count["phase_reach_rows"] += 1
    phase_cases = {r["case"] for r in table(HERE / "phase_reach.csv")}
    if not partial:
        assert phase_cases == set(
            (*full_cases, "bfs_discovery_far", "mcf17_discovery_far")
        )
    ledger = table(HERE / "run_registry.csv")
    if not partial:
        assert len(ledger) == len(CASES) * len(POLICIES) + len(full_cases) + 1
        assert all(r["status"] == "complete" for r in ledger)
    joint = table(HERE / "full_joint_coverage.csv")
    if not partial:
        assert {r["case"] for r in joint} == set(full_cases)
    for r in joint:
        v = lambda k: int(r[k])
        assert (
            v("hybrid_timely_all_reference_persistent_or_late")
            <= v("hybrid_timely_none_reference_timely")
            <= v("hybrid_original_timely")
        )
        assert v("hybrid_original_timely") - v(
            "hybrid_timely_none_reference_timely"
        ) == v("any_reference_original_timely") - v(
            "any_reference_timely_hybrid_became_non_timely"
        )
    return count


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--partial", action="store_true", help="during a still-running matrix"
    )
    args = p.parse_args()
    for key, n in verify(args.partial).items():
        print(key, n)
