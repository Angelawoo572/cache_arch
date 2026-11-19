"""Retrospectively align all frozen, strict branch examples to actual native requests."""

from collections import defaultdict
import csv
import gzip
import sys
import numpy as np
from run import HERE, STUDY, table

sys.path.append(str(STUDY / "transfer"))
from evaluate import snapshots, fit, legal, read_history


def candidates(folder, lines):
    requests, by_line, claim = {}, defaultdict(list), {}
    with gzip.open(folder / "candidates.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            line, pid, stage = int(r["target"]), int(r["id"]), r["stage"]
            if line not in lines:
                continue
            if stage == "proposed":
                q = dict(
                    line=line,
                    id=pid,
                    producer=int(r["instr_id"]),
                    producer_op=int(r["operand"]),
                    gen=int(r["cycle"]),
                    why=r["reason"],
                )
                requests[pid] = q
                by_line[line].append(q)
            if pid not in requests:
                continue
            q = requests[pid]
            if stage == "rejected":
                q["rejection"] = r["reason"]
            if stage in (
                "rejected",
                "accepted",
                "dispatch",
                "fill",
                "timely",
                "late",
                "evicted",
                "unresolved",
                "redundant",
            ):
                q[stage] = int(r["cycle"])
            if stage in ("timely", "late"):
                k = int(r["demand_instr_id"]), int(r["demand_operand"])
                assert k not in claim
                claim[k] = q
    return by_line, claim


def stats(rows, root, case, policy):
    z = read_history(root, case, policy)
    source = read_history(STUDY / "raw/screen", case, "stride")
    sk, sp = snapshots(source)
    fit_by_h = {
        h: fit(source, sk, sp, int(0.6 * len(source["t"])), h)[2]
        for h in {int(r["horizon"]) for r in rows}
    }
    keys, pages = snapshots(z)
    ix = {(int(i), int(op)): j for j, (i, op) in enumerate(zip(z["instr"], z["op"]))}
    line_set = {int(r["target_line"]) for r in rows}
    by_line, claim = candidates(root / "runs" / case / policy, line_set)
    results = []
    for r in rows:
        h = int(r["horizon"])
        ck = (int(r["cue_instr"]), int(r["cue_operand"]))
        tk = (int(r["target_instr"]), int(r["target_operand"]))
        target, base = int(r["target_line"]), int(r["cue_line"])
        ci = ix.get(ck)
        ti = ix.get(tk)
        t = int(z["t"][ti]) if ti is not None else None
        tag = int(z["tag_t"][ti]) if ti is not None else None
        gen = int(z["t"][ci]) if ci is not None else None
        old_key = tuple(
            int(r[n])
            for n in ("local_pc", "offset", "same_pc_delta", "preceding_same_pc_delta")
        ) + tuple(map(int, r["previous_three_pcs"].split()))
        if ci is not None:
            assert int(z["pa"][ci]) == base
        if ti is not None:
            assert int(z["pa"][ti]) == target
        prior_line = (
            np.flatnonzero((z["pa"] == target) & (np.arange(len(z["t"])) < ti))
            if ti is not None
            else []
        )
        prev = int(z["t"][prior_line[-1]]) if len(prior_line) else -1
        # Window is strictly after previous same-line ingress, strictly before target ingress.
        proposals = [
            q for q in by_line[target] if t is not None and prev < q["gen"] < t
        ]
        first = min(proposals, key=lambda q: q["gen"]) if proposals else {}
        c = claim.get(tk, {})
        outcome = r[policy]
        assert (outcome == "timely") == ("timely" in c) and (outcome == "late") == (
            "late" in c
        )
        native_choice = fit_by_h[h].get(keys[ci][2]) if ci is not None else None
        reaches = legal(base, target, pages, ci) if ci is not None else False
        row = dict(
            case=case,
            horizon=h,
            anchor_instr=ck[0],
            anchor_op=ck[1],
            target_instr=tk[0],
            target_op=tk[1],
            cue_line=base,
            target_line=target,
            delta=target - base,
            policy=policy,
            outcome=outcome,
            all_three_miss=r["missed_by_all"],
            cue_has_ingress=int(ci is not None),
            target_has_ingress=int(ti is not None),
            cue_time=gen,
            target_ingress=t,
            target_lookup=tag,
            cue_to_demand_cycles=t - gen if gen is not None and t is not None else "",
            target_legal_at_cue=int(reaches) if ci is not None else "",
            same_ordered_context=int(keys[ci][2] == old_key) if ci is not None else "",
            actual_frozen_delta=native_choice[0] if native_choice else "",
            frozen_would_offer_exact=int(
                native_choice is not None
                and native_choice[0] == target - base
                and reaches
            )
            if ci is not None
            else "",
            intervening_ingresses=ti - ci - 1
            if ci is not None and ti is not None
            else "",
            intervening_same_line=int(np.count_nonzero(z["pa"][ci + 1 : ti] == target))
            if ci is not None and ti is not None and ci < ti
            else "",
            n_prior_line_proposals=len(proposals),
            proposals_at_ingress_or_through_tag=(
                sum(t <= q["gen"] <= tag for q in by_line[target])
                if t is not None and tag is not None
                else ""
            ),
            first_proposal=first.get("gen", ""),
            first_producer=first.get("producer", ""),
            first_relation=(
                "no_cue"
                if gen is None
                else "no_target_ingress"
                if t is None
                else "no_proposal"
                if not first
                else "before_cue"
                if first["gen"] < gen
                else "after_cue"
                if first["gen"] > gen
                else "same_cycle"
            ),
            first_accepted=first.get("accepted", ""),
            first_rejection=first.get("rejection", ""),
            direct_claim_producer=c.get("producer", ""),
            direct_claim_generation=c.get("gen", ""),
            direct_claim_dispatch=c.get("dispatch", ""),
            direct_claim_fill=c.get("fill", ""),
            direct_claim_proposal_to_ingress=t - c["gen"]
            if c and t is not None
            else "",
            direct_claim_fill_minus_ingress=c["fill"] - t
            if c and "fill" in c and t is not None
            else "",
            head_wait=int(z["head"][ti]) if ti is not None else "",
        )
        if policy == "stride":
            assert (
                row["same_ordered_context"]
                and row["frozen_would_offer_exact"]
                and t - gen == int(r["lead_cycles"])
            )
        results.append(row)
    return results


if __name__ == "__main__":
    data = []
    for name, root in (("B", "replication"), ("deep", "long_google")):
        path = (
            STUDY
            / "transfer"
            / ("long_google" if name == "deep" else "")
            / "paired_futures.csv"
        )
        rows = table(path)
        for case in sorted({r["case"] for r in rows}):
            same = [r for r in rows if r["case"] == case]
            for policy in ("stride", "multi", "spp"):
                for row in stats(same, STUDY / "raw" / root, case, policy):
                    data.append(dict(window=name, **row))
    with (HERE / "transfer_witnesses.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(data)
    print("Verified", len(data), "existing cue / target / policy combinations.")
