"""Check the selected LBM demands against each full-rate SPP, multi-delta, cross-PC and temporal run."""

import bisect
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from lbm_witness import rows
from run import HERE, csvs
from score import save


def classify(job):
    arm, records = job
    folder = HERE.parent / "raw/strong_controls/verified" / ("open-" + arm)
    keys = {(int(r["target_instr"]), int(r["target_op"])) for r in records}
    lines = {int(r["target_line"]) for r in records}
    arrivals = defaultdict(list)
    demand = {}
    head = {}
    requests = {}
    use = {}
    for r in rows(folder, "loads"):
        if r["level"] != "LLC" or int(r["physical_line"]) not in lines:
            continue
        key = int(r["instr_id"]), int(r["operand"])
        if r["stage"] == "ingress":
            arrivals[int(r["physical_line"])].append(int(r["cycle"]))
            if key in keys:
                demand.setdefault(key, {})["ingress"] = int(r["cycle"])
        if r["stage"] == "lookup" and key in keys:
            d = demand.setdefault(key, {})
            d.update(
                hit=int(r["hit"]),
                lookup=int(r["cycle"]),
                pc=int(r["pc"]),
                virtual=int(r["virtual_line"]),
                physical=int(r["physical_line"]),
            )
    for r in rows(folder, "head"):
        key = int(r["instr_id"]), int(r["operand"])
        if key in keys:
            head[key] = int(r["cycles"])
    for r in rows(folder, "candidates"):
        if int(r["target"]) not in lines:
            continue
        i = int(r["id"])
        stage = r["stage"]
        if stage == "proposed":
            requests[i] = dict(
                target=int(r["target"]),
                trigger=int(r["instr_id"]),
                trigger_operand=int(r["operand"]),
                source=r["reason"],
            )
        if i not in requests:
            continue
        req = requests[i]
        req[stage] = int(r["cycle"])
        if stage == "rejected":
            req["why"] = r["reason"]
        if stage in ("timely", "late"):
            k = int(r["demand_instr_id"]), int(r["demand_operand"])
            if k in keys:
                use[k] = (stage, req)
    output = []
    for r in records:
        key = int(r["target_instr"]), int(r["target_op"])
        d = demand[key]
        line = int(r["target_line"])
        assert (d["pc"], d["physical"], d["virtual"]) == (
            int(r["target_pc"]),
            line,
            int(r["target_virtual_line"]),
        )
        seq = sorted(arrivals[line])
        at = bisect.bisect_left(seq, d["ingress"])
        since = seq[at - 1] if at else -1
        recent = [
            req
            for req in requests.values()
            if req["target"] == line and since < req["proposed"] < d["ingress"]
        ]
        prior = max(recent, key=lambda tx: tx["proposed"], default={})
        event, claim = use.get(key, ("none", {}))
        desc = (
            event
            if event != "none"
            else "ordinary_tag_hit"
            if d["hit"]
            else "no_target_proposal_since_last_ingress"
            if not prior
            else "proposed_then_rejected_" + prior["why"]
            if "why" in prior
            else "prior_proposal_other_outcome"
        )
        output.append(
            dict(
                selection=r["selection"],
                control=arm,
                target_instr=key[0],
                operand=key[1],
                target_line=line,
                tag_hit=d["hit"],
                classification=desc,
                head_wait=head.get(key, 0),
                demand_ingress=d["ingress"],
                tag_lookup=d["lookup"],
                prior_same_line_proposals=len(recent),
                last_prior_trigger=prior.get("trigger", ""),
                last_prior_reason=prior.get("why", ""),
                claim_source=claim.get("source", ""),
                claim_trigger=claim.get("trigger", ""),
                proposed=claim.get("proposed", ""),
                dispatch=claim.get("dispatch", ""),
                fill=claim.get("fill", ""),
                first_use_cycle=claim.get(event, ""),
            )
        )
    return output


if __name__ == "__main__":
    selected = [r for r in csvs(HERE / "lbm_transfer_witness.csv") if r["context"] == "exact"]
    with ThreadPoolExecutor(4) as pool:
        results = list(
            pool.map(classify, [(name, selected) for name in ("spp", "multi", "xpc", "temporal")])
        )
    save(HERE / "lbm_witness_controls.csv", [r for result in results for r in result])
