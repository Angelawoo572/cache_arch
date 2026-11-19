"""Export fixed-work performance, per-core costs, arbitration cohorts, and first-reason counts."""

import csv
import json

from run import HERE, RAW, csvs, totals


def save(name, rows):
    if not rows:
        return
    with (HERE / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def generate():
    overall = []
    individual = []
    cohorts = []
    resources = []
    for finished in sorted((RAW / "runs").glob("*/complete.json")):
        d = finished.parent
        spec = json.loads(finished.read_text())
        case = spec["case"]
        p = spec["policy"]
        label = spec["label"]
        cs = csvs(d / "cores.csv")
        st = totals(d)
        if label.startswith("head-") or label in ("evidence", "victim_probe", "four_targets"):
            continue
        rf = RAW / "runs" / f"{case}__stride__{label}"
        if not (rf / "complete.json").exists():
            rf = RAW / "runs" / f"{case}__stride__open"
        if not (rf / "complete.json").exists():
            continue
        reference = csvs(rf / "cores.csv")
        rs = totals(rf)
        cycles = max(int(r["cycles"]) for r in cs)
        rc = max(int(r["cycles"]) for r in reference)
        traffic = sum(st[k] for k in ("dram_demand_bytes", "dram_prefetch_bytes", "dram_write_bytes"))
        rtraffic = sum(rs[k] for k in ("dram_demand_bytes", "dram_prefetch_bytes", "dram_write_bytes"))
        issued = st["prefetch_accepted"]
        timely = st.get("study_pf_timely", 0)
        late = st.get("study_pf_late", 0)
        identity = dict(case=case, policy=p, setting=label)
        overall.append(
            dict(
                **identity,
                cycles=cycles,
                reference_stride_cycles=rc,
                saved_cycles=rc - cycles,
                time_saved_pct=round(100 * (rc - cycles) / rc, 5),
                instructions=sum(int(r["instructions"]) for r in cs),
                accepted=issued,
                timely=timely,
                late=late,
                unused_evictions=st.get("study_pf_evicted", 0),
                redundant=st.get("study_pf_redundant", 0),
                unresolved=st.get("study_pf_unresolved", 0),
                timely_per_accepted_pct=round(100 * timely / issued, 3) if issued else "",
                bytes=traffic,
                dram_bytes_per_kinst=round(1000 * traffic / sum(int(r["instructions"]) for r in cs), 3),
                bytes_delta_from_stride=traffic - rtraffic,
                demand_read_bytes=st["dram_demand_bytes"],
                prefetch_read_bytes=st["dram_prefetch_bytes"],
                write_bytes=st["dram_write_bytes"],
                delayed_engine_drops=st.get("study_rejected_engine_full", 0),
                engine_pending_at_end=st.get("study_engine_pending_end", 0),
                rate_budget_drops=st.get("study_rejected_budget", 0),
                pressure_suppressed=st.get("study_silent_pressure_gate", 0),
                agreement_required_suppressed=st.get("study_silent_uncorroborated", 0),
                crosspage_issued=st.get("study_accepted_crosspage", 0),
                agree_raw=st.get("study_agreed_anchors", 0),
                disagree_raw=st.get("study_disagreed_anchors", 0),
                accepted_agreed=st.get("study_accepted_agreement", 0),
            )
        )
        assert issued == timely + late + sum(
            st.get("study_pf_" + k, 0) for k in ("evicted", "redundant", "unresolved")
        ), (d, issued)
        for a, b in zip(cs, reference):
            i = int(a["cpu"])
            v = int(a["cycles"])
            n = int(a["instructions"])
            nominal = spec["per_core_acceptance_cap"]
            accepted = st.get("study_accepted_core_" + str(i), 0)
            individual.append(
                dict(
                    **identity,
                    cpu=i,
                    instructions=n,
                    cycles=v,
                    ipc=round(n / v, 8),
                    reference_cycles=b["cycles"],
                    saved_cycles=int(b["cycles"]) - v,
                    accepted=accepted,
                    total_cap=nominal[i] if nominal else "",
                    cap_met=int(accepted == nominal[i]) if nominal else "",
                    head_cycles=st["head_unique_cycles_core_" + str(i)],
                )
            )
        for g in range(8):
            data = {
                k: st.get(f"study_cohort_{g}_{k}", 0)
                for k in ("accepted", "timely", "late", "evicted", "redundant", "unresolved")
            }
            if data["accepted"] and label == "open" and p == "stride":
                cohorts.append(
                    dict(
                        **identity,
                        agreement=g // 4,
                        high_pressure=(g >> 1) & 1,
                        run_at_least_four=g & 1,
                        **data,
                        timely_fraction=round(data["timely"] / data["accepted"], 6),
                        use_fraction=round((data["timely"] + data["late"]) / data["accepted"], 6),
                    )
                )
        if label in ("open", "paced8", "lat32"):
            resources.append(
                dict(
                    **identity,
                    anchors=st.get("study_anchors", 0),
                    under_pressure=st.get("study_pressure_anchors", 0),
                    **{
                        reason: st.get("study_rejected_" + reason, 0)
                        for reason in (
                            "page",
                            "resident",
                            "mshr",
                            "rq",
                            "wq",
                            "pq",
                            "outstanding",
                            "pq_full",
                            "budget",
                            "engine_full",
                        )
                    },
                    confirmations_of_original_native=st.get("study_confirm_existing_stride", 0),
                    promotion_eligible=st.get("study_promoted_ready_entry", 0),
                    already_head=st.get("study_confirm_already_head", 0),
                    already_in_mshr=st.get("study_confirm_already_mshr", 0),
                    already_resident=st.get("study_confirm_already_resident", 0),
                    not_ready=st.get("study_confirm_not_ready", 0),
                )
            )
    save("policy_results.csv", overall)
    save("policy_by_core.csv", individual)
    save("agreement.csv", cohorts)
    save("resources.csv", resources)


if __name__ == "__main__":
    generate()
