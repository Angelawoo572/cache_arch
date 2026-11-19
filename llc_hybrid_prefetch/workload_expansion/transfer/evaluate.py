"""Freeze descriptive tables on window A and test conditional futures on window B."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import argparse
import csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
RULES = ("local", "previous_pc", "ordered", "unordered", "relative", "prior", "next_line")
COLD = 65


def read_history(root, case, policy):
    with np.load(root / "history" / case / (policy + ".npz")) as z:
        return {k: v for k, v in z.items()}


def snapshots(z):
    """Store only prefix-derived keys and first-observed physical page positions."""
    previous = {}
    page_first = {}
    last_pcs = (-1, -1, -1)
    keys = []
    for i, (ip, addr) in enumerate(zip(z["pc"], z["pa"])):
        pc, line = int(ip), int(addr)
        page_first.setdefault(line // 64, i)
        prior, prev_delta = previous.get(pc, (None, COLD))
        delta = COLD if prior is None else max(-64, min(64, line - prior))
        previous[pc] = (line, delta)
        local = (pc, line % 64, delta, prev_delta)
        seq = (*last_pcs, pc)
        relative = tuple(-1 if x == -1 else seq.index(x) for x in seq)
        keys.append((local, local + last_pcs[-1:], local + last_pcs,
                     local + tuple(sorted(last_pcs)), local[1:] + relative, ()))
        last_pcs = (*last_pcs[1:], pc)
    return keys, page_first


def legal(base, line, pages, anchor):
    return (line > 0 and 0 < abs(line - base) <= 512 and
            (line // 64 == base // 64 or pages.get(line // 64, anchor + 1) <= anchor))


def future(z, i, horizon, pages):
    base = int(z["pa"][i])
    j = i + horizon
    target = int(z["pa"][j])
    first_use = not any(z["pa"][i+1:j] == target)
    return target - base, first_use, legal(base, target, pages, i)


def fit(z, keys, pages, end, horizon):
    visits = [Counter() for _ in keys[0]] if keys else [Counter() for _ in RULES[:-1]]
    actions = [defaultdict(Counter) for _ in visits]
    for i in range(max(0, end-horizon)):
        d, first, reachable = future(z, i, horizon, pages)
        for r, key in enumerate(keys[i]):
            visits[r][key] += 1
            if reachable and first:
                actions[r][key][d] += 1
    tables = []
    for counts, candidates in zip(visits, actions):
        table = {}
        trained = [k for k,n in counts.items() if n >= 3 and candidates[k]]
        for k in sorted(trained, key=lambda x: (-counts[x], x))[:256]:
            d, n = min(candidates[k].items(), key=lambda item: (-item[1], abs(item[0]), item[0]))
            table[k] = (d, counts[k], n)
        tables.append(table)
    return tables


def reference_service(z):
    end = int(.6 * len(z["t"]))
    if not end:
        return None
    deadline = z["t"][end-1]
    good = (z["dispatch"] >= 0) & (z["fill"] >= z["dispatch"]) & (z["fill"] <= deadline)
    values = z["fill"][good] - z["dispatch"][good]
    return float(np.median(values)) if len(values) else None


def lookup(z):
    return {(int(n), int(op)): (int(hit), int(kind), int(pa)) for n, op, hit, kind, pa
            in zip(z["instr"], z["op"], z["hit"], z["pf_kind"], z["pa"])}


def native_class(z, j, lookups):
    key = int(z["instr"][j]), int(z["op"][j])
    statuses = []
    for table in lookups:
        entry = table.get(key)
        if entry is None:
            statuses.append("upstream")
            continue
        hit, kind, line = entry
        assert line == z["pa"][j]
        statuses.append("timely" if kind == 1 else "late" if kind == 2 else
                        "miss" if hit == 0 else "hit" if hit == 1 else "no_tag")
    return statuses


def scan(case, source, test, sources, keys, pages, h, service, lookups):
    tables = fit(source, *sources, int(.6 * len(source["t"])), h)
    count = [Counter() for _ in RULES]
    ambiguity = defaultdict(dict)
    wins = []
    for i in range(max(0, len(keys)-h)):
        k = keys[i]
        j = i+h
        base = int(test["pa"][i])
        d, first, reachable = future(test, i, h, pages)
        lead = int(test["t"][j] - test["t"][i])
        adequate = bool(first and reachable and service is not None and lead >= service and lead > 0)
        original = int(test["baseline"][j]) == 0
        statuses = native_class(test, j, lookups)
        all_missed = original and all(x in ("miss","late") for x in statuses)
        outputs = [table.get(ctx) for table, ctx in zip(tables, k)] + [(1,0,0)]
        credited = []
        for r, proposal in enumerate(outputs):
            c = count[r]
            c["anchors"] += 1
            c["original_misses"] += original
            c["all_three_missed"] += all_missed
            c["legal_first_with_lead"] += adequate
            c["original_legal_with_lead"] += adequate and original
            c["gap_legal_with_lead"] += adequate and all_missed
            stored = proposal is not None
            offer = stored and legal(base, base+proposal[0], pages, i)
            if r==len(RULES)-1 and offer:
                offer = base // 64 == (base+1) // 64
            correct = bool(offer and proposal[0] == d and first)
            useful = bool(correct and adequate)
            credited.append(useful)
            c["supported"] += stored
            c["offered"] += offer
            c["first_use_address_matches"] += correct
            c["matches_with_lead"] += useful
            c["original_matches_with_lead"] += useful and original
            c["gap_matches_with_lead"] += useful and all_missed
            c["gap_positive_head"] += useful and all_missed and test["head"][j] > 0
            if r:
                c["gap_gained_vs_local"] += useful and not credited[0] and all_missed
                c["gap_lost_vs_local"] += credited[0] and not useful and all_missed
        # Store the earliest B occurrence per distinct trained ordered branch.
        if credited[2] and abs(k[0][2]) < 64 and abs(k[0][3]) < 64:
            record = dict(case=case, horizon=h, local_pc=k[0][0], offset=k[0][1],
                          same_pc_delta=k[0][2], preceding_same_pc_delta=k[0][3],
                          previous_three_pcs=" ".join(map(str,k[2][-3:])), chosen_delta=d,
                          A_context_count=outputs[2][1], A_action_count=outputs[2][2],
                          cue_instr=int(test["instr"][i]), cue_operand=int(test["op"][i]),
                          target_instr=int(test["instr"][j]), target_operand=int(test["op"][j]),
                          cue_line=base, target_line=base+d, lead_cycles=lead,
                          service_reference=service, baseline_miss=int(original),
                          missed_by_all=int(all_missed), stride=statuses[0],
                          multi=statuses[1], spp=statuses[2], target_head=int(test["head"][j]))
            branches = ambiguity[k[0]]
            previous = branches.get(d)
            if previous is None or record["missed_by_all"] > previous["missed_by_all"] or (
                    record["missed_by_all"] == previous["missed_by_all"] and
                    record["baseline_miss"] > previous["baseline_miss"]):
                branches[d] = record
        if credited[2] and not credited[0] and all_missed:
            wins.append((int(test["instr"][j]),int(test["op"][j])))
    witness = []
    for branches in ambiguity.values():
        if len(branches)>=2 and any(b["baseline_miss"] for b in branches.values()):
            witness.extend(branches.values())
    fields = ("anchors original_misses all_three_missed legal_first_with_lead original_legal_with_lead "
              "gap_legal_with_lead supported offered first_use_address_matches matches_with_lead "
              "original_matches_with_lead gap_matches_with_lead gap_positive_head gap_gained_vs_local gap_lost_vs_local").split()
    output=[]
    for idx, name in enumerate(RULES):
        output.append(dict(case=case,horizon=h,rule=name,
                           training_ingresses=int(.6*len(source["t"])),context_entries=len(tables[idx]) if idx<len(tables) else 0,
                           service_reference=service,**{f:int(count[idx][f]) for f in fields}))
    return output,witness,wins


def analyze(job):
    case, test_root = job
    root = STUDY / "raw"
    source = read_history(root/"screen",case,"stride")
    test = read_history(test_root,case,"stride")
    sources = snapshots(source)
    keys, pages = snapshots(test)
    lookups = [lookup(read_history(test_root,case,p)) for p in ("stride","multi","spp")]
    service = reference_service(read_history(root/"screen",case,"nopf"))
    scores=[]; examples=[]; targets=set()
    for h in (1,4,16):
        rows,wins,ids=scan(case,source,test,sources,keys,pages,h,service,lookups)
        scores.extend(rows); examples.extend(wins);targets.update(ids)
    return scores,examples,dict(case=case,distinct_ordered_only_gap_targets=len(targets),
                                strict_ambiguous_local_context_horizons=len({(r["local_pc"],r["offset"],r["same_pc_delta"],
                                    r["preceding_same_pc_delta"],r["horizon"]) for r in examples}),
                                contexts_with_branch_missed_by_all=len({(r["local_pc"],r["offset"],r["same_pc_delta"],
                                    r["preceding_same_pc_delta"],r["horizon"]) for r in examples if r["missed_by_all"]}))


def write(path, records, fields=None):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields or list(records[0]),lineterminator="\n")
        w.writeheader();w.writerows(records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs",type=int,default=4)
    parser.add_argument("--test-root", type=Path, default=STUDY/"raw/replication")
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--case", nargs="+")
    args=parser.parse_args()
    with (STUDY/"intake/registry.csv").open() as f:
        cases=[r["case"] for r in csv.DictReader(f) if r["split"]=="discovery"]
    if args.case:
        if set(args.case)-set(cases):
            raise ValueError("unregistered or held-out cases")
        cases=[c for c in cases if c in args.case]
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.jobs) as pool:
        results=list(pool.map(analyze,[(c,args.test_root) for c in cases]))
    write(args.output/"rules.csv",[r for s,_,_ in results for r in s])
    fields=("case horizon local_pc offset same_pc_delta preceding_same_pc_delta previous_three_pcs chosen_delta "
            "A_context_count A_action_count cue_instr cue_operand target_instr target_operand cue_line target_line "
            "lead_cycles service_reference baseline_miss missed_by_all stride multi spp target_head").split()
    write(args.output/"paired_futures.csv",[r for _,ex,_ in results for r in ex],fields)
    write(args.output/"opportunities.csv",[stats for _,_,stats in results])


if __name__=="__main__":
    main()
