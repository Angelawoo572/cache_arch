"""Inspect the already selected deep-window Google Charlie forecast in all native arms."""

import csv
from transfer_witnesses import STUDY, HERE, candidates, read_history, table


def measure(policy, observed):
    case = observed["case"]
    root = STUDY / "raw/long_google"
    folder = root / "runs" / case / policy
    z = read_history(root, case, policy)
    ids = {(int(i), int(op)): j for j, (i, op) in enumerate(zip(z["instr"], z["op"]))}
    anchor = int(observed["trigger_instr"]), int(observed["trigger_operand"])
    target = int(observed["target_instr"]), int(observed["target_operand"])
    ci, ti = ids[anchor], ids[target]
    line = int(z["pa"][ti])
    base = int(z["pa"][ci])
    start, stop = int(z["t"][ci]), int(z["t"][ti])
    assert (int(z["pc"][ci]), base, line) == (
        int(observed["trigger_pc"]),
        int(observed["trigger_line"]),
        int(observed["target_line"]),
    )
    records, claim = candidates(folder, {line})
    q = claim.get(target, {})
    offers = records[line]
    prior = [c for c in offers if c["gen"] < start]
    previous = [int(i) for i, p in enumerate(z["pa"][:ci]) if p == line]
    # Counter is a lower bound on all shared-cache transactions: stores, I-side and warm-up are not fully logged.
    pending_prefetch = sum(
        c.get("accepted", start + 1) <= start and c.get("fill", start + 1) > start
        for c in prior
    )
    pending_demand = sum(
        z["dispatch"][j] >= 0
        and z["dispatch"][j] <= start
        and (z["fill"][j] < 0 or z["fill"][j] > start)
        for j in previous
    )
    issue = q.get("gen")
    dispatch = q.get("dispatch")
    fill = q.get("fill")
    tag, dd, df = (int(z[k][ti]) for k in ("tag_t", "dispatch", "fill"))
    return dict(
        case=case,
        policy=policy,
        cue_instr=anchor[0],
        target_instr=target[0],
        cue_line=base,
        target_line=line,
        previous_three_pcs=" ".join(map(str, z["pc"][ci - 3 : ci])),
        same_page_candidate=int(line // 64 == base // 64),
        cue_ingress=start,
        target_ingress=stop,
        target_lookup=tag,
        ingress_distance=ti - ci,
        available_cycles=stop - start,
        training_dispatch_to_fill_reference=observed["service_reference"],
        prior_tracked_target_ingresses=len(previous),
        earlier_tracked_target_proposals=len(prior),
        tracked_demand_in_service_at_cue=pending_demand,
        tracked_prefetch_in_service_at_cue=pending_prefetch,
        occupancy_tag_at_cue="unobserved",
        proposals_from_cue_through_demand_lookup=sum(
            start <= c["gen"] <= tag for c in offers
        ),
        native_claim_producer=q.get("producer", ""),
        native_proposal=issue,
        native_pf_dispatch=dispatch,
        native_pf_fill=fill,
        native_proposal_after_cue=issue - start if q else "",
        native_proposal_lead=stop - issue if q else "",
        native_dispatch_to_fill=fill - dispatch if q else "",
        native_fill_minus_demand_ingress=fill - stop if q else "",
        ordinary_demand_dispatch=dd if dd >= 0 else "",
        demand_completion_or_fill=df if df >= 0 else "",
        ordinary_demand_dispatch_to_fill=df - dd if dd >= 0 and df >= 0 else "",
        result=(
            "timely"
            if "timely" in q
            else "late"
            if "late" in q
            else "ordinary_hit"
            if int(z["hit"][ti]) == 1
            else "miss"
        ),
        target_operand_head_wait=int(z["head"][ti]),
    )


if __name__ == "__main__":
    rows = [
        r
        for r in table(STUDY / "patterns/long_google/window_examples.csv")
        if r["case"] == "gg_charlie"
        and r["rule"] == "previous_pc"
        and r["target_instr"] == "4658539"
    ]
    assert len(rows) == 1
    data = [measure(p, rows[0]) for p in ("nopf", "stride", "multi", "spp")]
    with (HERE / "charlie_probe.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(data)
