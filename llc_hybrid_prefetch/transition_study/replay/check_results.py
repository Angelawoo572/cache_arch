"""Validate fixed cohorts and preserved native behavior across all three experimental blocks."""

import csv
import gzip

from prepare_tables import TABLES
from run import HERE, RAW, RUNS, csvs
from score import LABELS


def check_block(prefix, expected):
    perf = csvs(HERE / (prefix + "performance.csv"))
    cohorts = csvs(HERE / (prefix + "cohorts.csv"))
    group = {(r["case"], r["policy"], r["core"]) for r in perf}
    assert len({(r["case"], r["policy"]) for r in perf}) == expected
    assert len(cohorts) == len(group) * 16
    for r in cohorts:
        n = int(r["episodes"])
        prefix_count = int(r["prefix_loads"])
        timely = int(r["timely"])
        assert sum(int(r[s]) for s in LABELS) == n
        assert (
            int(r["converted_offer_before_last_ingress"])
            + int(r["converted_offer_at_last_ingress"])
            <= int(r["converted_with_last_ingress_observed"])
            <= timely
        )
        assert (
            int(r["converted_with_entire_prefix_preserved"])
            <= int(r["converted_with_last_timely_preserved"])
            <= timely
        )
        assert int(r["entire_prefix_preserved"]) <= int(r["last_timely_preserved"]) <= n
        assert int(r["prefix_timely_preserved"]) <= prefix_count and prefix_count >= 2 * n
        assert int(r["recovery_id_still_timely"]) <= int(r["baseline_recoveries"]) <= n
        if r["policy"] in ("stride", "shadow", "lbm_shadow_page"):
            assert int(r["late" if r["reference_failure"] == "L" else "miss"]) == n
            assert int(r["prefix_timely_preserved"]) == prefix_count and int(
                r["recovery_id_still_timely"]
            ) == int(r["baseline_recoveries"])
    for r in perf:
        assert (r["case"], r["policy"], r["core"]) in group
        assert int(r["accepted"]) == sum(
            int(r[x]) for x in ("timely", "late", "unused_evicted", "redundant", "unresolved")
        )
    return len(perf), len(cohorts)


def shadow(csv_file, live_file, old_format):
    ref = csvs(csv_file)
    with gzip.open(live_file, "rt") as f:
        actual = list(csv.DictReader(f))
    if old_format:
        ref = [r for r in ref if int(r["choice"]) >= 0]
        cols = {
            "instr": "instr_id",
            "operand": "operand",
            "base": "base",
            "cycle": "cycle",
            "choice": "offered",
        }
    else:
        cols = {k: k for k in ref[0]}
    assert len(ref) == len(actual)
    assert all(all(int(a[c]) == int(b[d]) for c, d in cols.items()) for a, b in zip(ref, actual))
    return len(ref)


if __name__ == "__main__":
    print(
        "Validated per-core rows and cohort rows",
        check_block("", 48),
        check_block("supplement_", 6),
        check_block("lbm_", 7),
    )
    fit = HERE.parent / "raw/predictability"
    used_grants = 0
    for src, dst, name, rep in TABLES:
        path = fit / (name + "-expected.csv")
        if not path.exists():
            path = RAW / "regenerated" / (name + "-expected.csv")
        is_lbm = dst == "lbm-r2"
        live = (
            RUNS
            / (dst + "__lbm_shadow_" + rep if is_lbm else dst + "__shadow")
            / ("lbm_grants.csv.gz" if is_lbm else "aux_grants.csv.gz")
        )
        used_grants += shadow(path, live, not is_lbm)
    for r in csvs(HERE / "pollution.csv"):
        assert int(r["prefetch_last_evictor"]) == int(r["other_core"]) + int(r["same_core"])
    for r in csvs(HERE / "lbm_transfer_witness.csv"):
        assert (
            int(r["training_positive_targets_for_offset"]) > 0
            and int(r["raw_contexts_sharing_fitted_hash"]) == 1
        )
        if r["context"] == "exact":
            assert (
                int(r["target_hit"]) == 1
                and r["first_use"] == "timely"
                and int(r["claim_trigger_instr"]) == int(r["cue_instr"])
            )
            assert (
                int(r["proposal"]) <= int(r["dispatch"]) < int(r["fill"]) <= int(r["target_lookup"])
            )
    survivors = {
        r["control"]: r
        for r in csvs(HERE / "lbm_witness_controls.csv")
        if r["selection"] == "four_open_controls_without_timely"
    }
    assert set(survivors) == {"spp", "multi", "xpc", "temporal"}
    assert all(r["tag_hit"] == "0" for r in survivors.values())
    for control in ("spp", "multi"):
        r = survivors[control]
        assert r["classification"] == "late" and int(r["fill"]) > int(r["demand_ingress"])
    for control in ("xpc", "temporal"):
        assert survivors[control]["prior_same_line_proposals"] == "0"
    print(
        "Verified six original-stream candidate exports on",
        used_grants,
        "offered/registered slots, plus pollution and R1/R2 witness consistency.",
    )
