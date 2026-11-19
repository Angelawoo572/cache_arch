"""Find a transferred physical-page offset with original training occurrences and a directly timed R2 first use."""

import csv
import gzip
import sys
from collections import Counter, defaultdict

from run import HERE, RUNS
from score import save

# The fitting code is imported in a subprocess-free local namespace; it supplies the archived training labels.
sys.path.insert(0, str(HERE.parent / "predictability"))
from predict import case, guarded, measured_service, outcomes


def rows(folder, stem):
    with gzip.open(folder / (stem + ".csv.gz"), "rt") as f:
        yield from csv.DictReader(f)


def get(folder, failure_ids):
    ingress = {}
    at = {}
    tx = {}
    by = {}
    head = {}
    for r in rows(folder, "ingress_history"):
        if r["measured"] == "1":
            k = int(r["instr_id"]), int(r["operand"])
            ingress[k] = (int(r["pc"]), int(r["line"]), int(r["cycle"]), int(r["native_match"]))
    for r in rows(folder, "loads"):
        if r["level"] != "LLC" or r["stage"] != "lookup":
            continue
        k = int(r["instr_id"]), int(r["operand"])
        if k in failure_ids:
            at[k] = (int(r["hit"]), int(r["cycle"]), int(r["virtual_line"]))
    for r in rows(folder, "head"):
        k = int(r["instr_id"]), int(r["operand"])
        if k in failure_ids:
            head[k] = int(r["cycles"])
    for r in rows(folder, "candidates"):
        cid = int(r["id"])
        stage = r["stage"]
        if stage == "proposed":
            tx[cid] = dict(
                source=r["reason"],
                anchor=(int(r["instr_id"]), int(r["operand"])),
                target=int(r["target"]),
            )
        if cid not in tx:
            continue
        t = tx[cid]
        t[stage] = int(r["cycle"])
        if stage in ("timely", "late"):
            k = int(r["demand_instr_id"]), int(r["demand_operand"])
            if k in failure_ids:
                by[k] = (stage, t)
    return ingress, at, by, head


def main():
    z, _, g = case("lbm-r1", "ingress")
    cut = int(0.6 * len(z["pc"]))
    tr = guarded(z, 0, cut)
    service = measured_service(z, cut)
    _, ok, _ = outcomes(z, g, tr, service, "1_64")
    train_counts = Counter()
    positives = Counter()
    example = {}
    bucket = defaultdict(set)
    for n, j in enumerate(tr):
        pc, o = int(z["pc"][j]), int(z["pa"][j] % 64)
        site = (pc, o)
        train_counts[site] += 1
        bucket[(pc % 4096) ^ o].add((pc, o))
        if ok[n] and 0 <= g["code"][j] < 64:
            row = (*site, int(g["code"][j]))
            positives[row] += 1
            example.setdefault(row, (j, g["q"][j]))
    ep_file = HERE.parent / "raw/episodes/lbm-r2/episode_ids.csv.gz"
    with gzip.open(ep_file, "rt") as f:
        episodes = {
            (int(r["first_miss_instr"]), int(r["first_miss_op"])): r
            for r in csv.DictReader(f)
            if r["view"] == "ingress" and r["failure_class"] == "P"
        }
    base = get(RUNS / "lbm-r2__lbm_shadow_page", episodes)
    full = get(RUNS / "lbm-r2__lbm_exact", episodes)
    selected = []
    for k, (stage, action) in full[2].items():
        if stage != "timely" or action["source"] != "auxiliary":
            continue
        anchor = action["anchor"]
        pc, line, cycle, _ = full[0][anchor]
        o = line % 64
        offset = action["target"] % 64
        site = (pc, o)
        entry = (*site, offset)
        support = positives[entry]
        if not support:
            continue
        demand = full[0][k][2]
        prefix_end = (int(episodes[k]["last_instr"]), int(episodes[k]["last_op"]))
        is_earlier = base[0][anchor][2] < base[0][prefix_end][2]
        if not is_earlier or demand - cycle > 4096 or len(bucket[(pc % 4096) ^ o]) != 1:
            continue
        w = base[3].get(k, 0) - full[3].get(k, 0)
        if w <= 0:
            continue
        selected.append((support, support / train_counts[site], w, -k[0], k, anchor))
    assert selected
    fixed5 = get(RUNS / "lbm-r2__lbm_plus5", episodes)
    distinct = [
        hit
        for hit in selected
        if (full[2][hit[4]][1]["target"] - full[0][hit[5]][1]) not in (1, 5)
        and fixed5[1][hit[4]][0] == 0
    ]
    assert distinct
    choices = [
        ("most_repeated_qualifying_key", max(selected)),
        ("fixed_plus5_still_misses", max(distinct)),
    ]
    strong_names = ("spp", "multi", "xpc", "temporal")
    strong_dir = HERE.parent / "raw/strong_controls/verified"
    strong_claims = {
        arm: {
            (int(r["demand_instr_id"]), int(r["demand_operand"])): r["stage"]
            for r in rows(strong_dir / ("open-" + arm), "candidates")
            if r["stage"] in ("timely", "late")
        }
        for arm in strong_names
    }
    strong = set().union(
        *(
            {k for k, stage in records.items() if stage == "timely"}
            for records in strong_claims.values()
        )
    )
    all_t = {k for k, (stage, _) in full[2].items() if stage == "timely"}
    surviving = all_t - strong
    exact_timely = {
        (int(r["demand_instr_id"]), int(r["demand_operand"]))
        for r in rows(RUNS / "lbm-r2__lbm_exact", "candidates")
        if r["stage"] == "timely"
    }
    import numpy as np

    with np.load(HERE.parent / "raw/episodes/lbm-r2/c0.npz") as original:
        ordered = original["ids"][original["ingress"]]
    index = {int(integral): i for i, integral in enumerate(ordered)}
    prefix = {}
    for k in episodes:
        meta = episodes[k]
        begin = 256 * int(meta["start_instr"]) + int(meta["start_op"])
        end = 256 * k[0] + k[1]
        seq = ordered[index[begin] : index[end]]
        members = [(int(v) // 256, int(v) % 256) for v in seq]
        prefix[k] = (
            len(members),
            sum(v in exact_timely for v in members),
            int(members[-1] in exact_timely),
        )
    candidates = [obj for obj in selected if obj[-2] in surviving]
    if candidates:
        strongest = max(
            candidates,
            key=lambda obj: (
                prefix[obj[-2]][0] == prefix[obj[-2]][1],
                fixed5[1][obj[-2]][0] == 0,
                *obj,
            ),
        )
        choices.append(("four_open_controls_without_timely", strongest))
    save(
        HERE / "lbm_witness_selection.csv",
        [
            dict(
                primary_original_persistent=len(episodes),
                exact_timely=len(all_t),
                exact_timely_without_timely_from_four_open=len(surviving),
                eligible_with_raw_training_support_sub4096_early_and_reduced_wait=len(candidates),
            )
        ],
    )
    witness_keys = {hit[-2] for _, hit in choices}
    cached = {"shadow_page": base, "exact": full, "plus5": fixed5}
    for mode in ("plus1", "page", "unrelated_older", "last_global_pc"):
        cached[mode] = get(RUNS / ("lbm-r2__lbm_" + mode), witness_keys)
    ledger = []
    for label, (*_, k, anchor) in choices:
        pc, line, _, _ = full[0][anchor]
        target = full[2][k][1]["target"]
        site = (pc, line % 64)
        entry = (*site, target % 64)
        j, q = example[entry]
        evidence = [
            dict(
                recording="lbm-r1",
                context="training observed occurrence",
                cue_instr=z["instr"][j],
                cue_operand=z["op"][j],
                cue_pc=z["pc"][j],
                cue_low12=pc % 4096,
                cue_line=z["pa"][j],
                cue_offset=int(z["pa"][j]) % 64,
                cue_ingress=z["t"][j],
                target_instr=z["instr"][q],
                target_op=z["op"][q],
                target_pc=z["pc"][q],
                target_virtual_line=z["va"][q],
                target_line=z["pa"][q],
                target_offset=int(z["pa"][q]) % 64,
                target_ingress=z["t"][q],
                target_lookup=z["tag_t"][q],
                target_hit=z["hit"][q],
                head_wait=z["head"][q],
                first_use="",
                first_use_source="",
                proposal="",
                accepted="",
                dispatch="",
                fill="",
                prior_timely_instr="",
                training_contexts=train_counts[site],
                training_positive_targets_for_offset=positives[entry],
                raw_contexts_sharing_fitted_hash=len(bucket[(pc % 4096) ^ site[1]]),
            )
        ]
        for policy in (
            "shadow_page",
            "plus1",
            "plus5",
            "page",
            "exact",
            "unrelated_older",
            "last_global_pc",
        ):
            arm = cached[policy]
            p, anchor_line, t, _ = arm[0][anchor]
            target_pc, ln, tt, _ = arm[0][k]
            hit, lookup, va = arm[1][k]
            s, tx = arm[2].get(k, ("", {}))
            row = dict(
                recording="lbm-r2",
                context=policy,
                cue_instr=anchor[0],
                cue_operand=anchor[1],
                cue_pc=p,
                cue_low12=p % 4096,
                cue_line=anchor_line,
                cue_offset=anchor_line % 64,
                cue_ingress=t,
                target_instr=k[0],
                target_op=k[1],
                target_pc=target_pc,
                target_virtual_line=va,
                target_line=ln,
                target_offset=ln % 64,
                target_ingress=tt,
                target_lookup=lookup,
                target_hit=hit,
                head_wait=arm[3].get(k, 0),
                first_use=s,
                first_use_source=tx.get("source", ""),
                proposal=tx.get("proposed", ""),
                accepted=tx.get("accepted", ""),
                dispatch=tx.get("dispatch", ""),
                fill=tx.get("fill", ""),
                prior_timely_instr=episodes[k]["last_instr"],
                training_contexts=train_counts[site],
                training_positive_targets_for_offset=positives[entry],
                raw_contexts_sharing_fitted_hash=len(bucket[(pc % 4096) ^ site[1]]),
            )
            row["claim_trigger_instr"] = tx["anchor"][0] if tx else ""
            row["claim_trigger_op"] = tx["anchor"][1] if tx else ""
            evidence.append(row)
        evidence[0].update(claim_trigger_instr="", claim_trigger_op="")
        for r in evidence:
            r["selection"] = label
            (
                r["original_prefix_loads"],
                r["exact_prefix_still_timely"],
                r["exact_last_still_timely"],
            ) = prefix[k]
        ledger.extend(evidence)
    save(HERE / "lbm_transfer_witness.csv", ledger)


if __name__ == "__main__":
    main()
