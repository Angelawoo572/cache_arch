"""Summarize validated solo baselines with causal first-use attribution to aligned NoPF misses."""

import argparse
import csv
import io
import json
import subprocess
from pathlib import Path
from run import FULL, RAW, HERE, rows
from joint import summarize_joint

POLICIES = ("nopf", "stride", "multi", "spp", "temporal")
FAR = ("bfs_discovery_far", "mcf17_discovery_far")


def ratio(x, y):
    return round(x / y, 8) if y else ""


def write(path, data):
    if data:
        with path.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)


def masks(root, folder, roi, raw=RAW):
    args = [
        str(raw / "join"),
        str(root / "outcomes_core0.u16le.gz"),
        str(folder / "outcomes_core0.u16le.gz"),
        str(roi),
    ]
    proc = subprocess.run(args, check=True, capture_output=True, text=True)
    result = []
    for r in csv.DictReader(io.StringIO(proc.stdout)):
        q, before, after, count = map(
            int, (r["quarter"], r["nopf_status"], r["arm_status"], r["operands"])
        )
        if before not in (0, 1, 2) or after not in (0, 1, 2, 6, 9):
            raise ValueError(
                f"Invalid lookup/claim mask {(before, after, count)} in {folder}"
            )
        result.append((q, before, after, count))
    return result


def summarize(out, only_complete, cases=None, raw=RAW, policies=POLICIES):
    subprocess.run(
        [
            "c++",
            "-std=c++11",
            "-O3",
            str(HERE / "join.cc"),
            "-lz",
            "-o",
            str(raw / "join"),
        ],
        check=True,
    )
    summaries, phases, transitions, reasons, joint = [], [], [], [], []
    roster = cases or [r["case"] for r in rows(FULL / "intake/selection.csv")] + list(
        FAR
    )
    for case in roster:
        root = raw / "runs" / (case + "__nopf")
        for policy in policies:
            directory = raw / "runs" / (case + "__" + policy)
            if not all((d / "complete.json").exists() for d in (root, directory)):
                if only_complete:
                    continue
                raise ValueError("Missing completed native arm: " + str(directory))
            meta = json.loads((directory / "complete.json").read_text())
            roi = meta["config"]["roi"]
            st = {r["key"]: int(r["value"]) for r in rows(directory / "stats.csv")}
            for key, value in st.items():
                if key.startswith("study_rejected_"):
                    reasons.append(
                        dict(
                            case=case,
                            policy=policy,
                            reason=key.removeprefix("study_rejected_"),
                            rejected_proposals=value,
                        )
                    )
            qs = rows(directory / "phases.csv")
            base_qs = rows(root / "phases.csv")
            pp = rows(directory / "cores.csv")[0]
            if int(pp["instructions"]) != roi:
                raise ValueError("Fixed work did not retire")
            hist = masks(root, directory, roi, raw)
            # Physical placement follows trace-order registrations and must agree for every policy.
            if (root / "pages.csv.gz").read_bytes() != (
                directory / "pages.csv.gz"
            ).read_bytes():
                raise ValueError(
                    "Source page placement drifted across matched policy arms"
                )

            def count(q, old, new):
                return sum(
                    n
                    for phase, a, b, n in hist
                    if (phase == q or q is None) and a in old and b in new
                )

            allold, allnew = {0, 1, 2}, {0, 1, 2, 6, 9}
            for i, q in enumerate(qs):
                for value, check in [
                    (int(base_qs[i]["llc_tag_misses"]), count(i, {2}, allnew)),
                    (int(base_qs[i]["llc_tag_hits"]), count(i, {1}, allnew)),
                    (int(q["llc_tag_hits"]), count(i, allold, {1, 9})),
                    (int(q["llc_tag_misses"]), count(i, allold, {2, 6})),
                    (int(q["late"]), count(i, allold, {6})),
                    (int(q["timely"]), count(i, allold, {9})),
                ]:
                    if value != check:
                        raise ValueError(
                            f"Dynamic alignment totals diverge: {case}, {policy}, Q{i}, {value}/{check}"
                        )
                phases.append(dict(case=case, policy=policy, **q))
            for phase, a, b, n in hist:
                transitions.append(
                    dict(
                        case=case,
                        policy=policy,
                        quarter=phase,
                        nopf_mask=a,
                        arm_mask=b,
                        operands=n,
                    )
                )
            c = int(pp["cycles"])
            base_cycles = int(rows(root / "cores.csv")[0]["cycles"])
            orig = count(None, {2}, allnew)
            tl = count(None, {2}, {9})
            late = count(None, {2}, {6})
            taghit = count(None, allold, {1, 9})
            tagmiss = count(None, allold, {2, 6})
            l2miss = sum(int(q["l2_tag_misses"]) for q in qs)
            l2hit = sum(int(q["l2_tag_hits"]) for q in qs)
            accepted = st["prefetch_accepted"]
            firsts = st.get("study_pf_timely", 0)
            lates = st.get("study_pf_late", 0)
            dur = st["pressure_cycles"]
            proposals = sum(v for k, v in st.items() if k.startswith("study_chosen_"))
            rejected = sum(v for k, v in st.items() if k.startswith("study_rejected_"))
            if proposals != accepted + rejected:
                raise ValueError(
                    "Proposals must partition into accepted and rejected under immediate issue"
                )
            ingress = sum(int(q["llc_ingress"]) for q in qs)
            summaries.append(
                dict(
                    case=case,
                    policy=policy,
                    warmup=meta["config"]["warmup"],
                    measured_instructions=roi,
                    cycles=c,
                    ipc=ratio(roi, c),
                    cycles_saved_vs_nopf=base_cycles - c,
                    time_reduction_vs_nopf=ratio(base_cycles - c, base_cycles),
                    llc_hits=taghit,
                    llc_misses=tagmiss,
                    llc_tag_hit_rate=ratio(taghit, taghit + tagmiss),
                    llc_tag_miss_rate=ratio(tagmiss, taghit + tagmiss),
                    llc_mpki=ratio(tagmiss * 1000, roi),
                    l2_hits=l2hit,
                    l2_misses=l2miss,
                    l2_tag_hit_rate=ratio(l2hit, l2hit + l2miss),
                    l2_tag_miss_rate=ratio(l2miss, l2hit + l2miss),
                    l2_mpki=ratio(l2miss * 1000, roi),
                    nopf_original_misses=orig,
                    orig_to_timely=tl,
                    orig_to_late=late,
                    orig_to_ordinary_hit=count(None, {2}, {1}),
                    orig_to_persistent_miss=count(None, {2}, {2}),
                    orig_without_arm_tag=count(None, {2}, {0}),
                    timely_coverage=ratio(tl, orig),
                    net_miss_coverage=ratio(orig - tagmiss, orig),
                    llc_demand_ingress=ingress,
                    proposed=proposals,
                    rejected=rejected,
                    accepted=accepted,
                    accepted_crosspage=st.get("study_accepted_crosspage", 0),
                    timely_first_use=firsts,
                    late_merge=lates,
                    timeliness_among_claims=ratio(firsts, firsts + lates),
                    inclusive_coverage=ratio(tl + late, orig),
                    accepted_per_ingress=ratio(accepted, ingress),
                    unused_evicted=st.get("study_pf_evicted", 0),
                    redundant=st.get("study_pf_redundant", 0),
                    unresolved=st.get("study_pf_unresolved", 0),
                    useful_accuracy=ratio(firsts + lates, accepted),
                    timely_accuracy=ratio(firsts, accepted),
                    head_union_cycles=st["head_unique_cycles_core_0"],
                    head_union_fraction=ratio(st["head_unique_cycles_core_0"], c),
                    avg_llc_mshr=ratio(st["llc_mshr_cycle_sum"], dur),
                    avg_dram_rq=ratio(st["dram_rq_cycle_sum"], dur),
                    avg_dram_wq=ratio(st["dram_wq_cycle_sum"], dur),
                    dram_demand_bytes=st["dram_demand_bytes"],
                    dram_prefetch_bytes=st["dram_prefetch_bytes"],
                    dram_write_bytes=st["dram_write_bytes"],
                    wall_seconds=meta["wall_seconds"],
                )
            )
        if all(
            (raw / "runs" / (case + "__" + p) / "complete.json").exists()
            for p in policies
        ):
            quarter_misses = [
                int(r["llc_tag_misses"]) for r in rows(root / "phases.csv")
            ]
            joint.extend(summarize_joint(case, policies, roi, quarter_misses, raw))
    out.mkdir(exist_ok=True, parents=True)
    write(out / "performance.csv", summaries)
    write(out / "joint_coverage.csv", joint)
    write(out / "quarters.csv", phases)
    write(out / "rejections.csv", reasons)
    write(out / "transitions.csv", transitions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed-only", action="store_true")
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument(
        "--cases", nargs="+", help="explicit view IDs for a separate secondary panel"
    )
    args = parser.parse_args()
    summarize(args.out, args.completed_only, args.cases)
