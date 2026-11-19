"""Causal absolute-target forecasts from finite physical-address association tables."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
POLICIES = ("nopf", "stride", "multi", "spp")
RULES = ("next_line", "pc256", "line256", "pc_line256", "line_unlimited")


def mixed(x):
    # Fold upper bits into the low bits before indexing a 256-entry direct-mapped table.
    x ^= x >> 30
    x = (x * 0xbf58476d1ce4e5b9) & ((1 << 64) - 1)
    x ^= x >> 27
    x = (x * 0x94d049bb133111eb) & ((1 << 64) - 1)
    return (x ^ (x >> 31)) & 255


def issue(z, horizon, rule):
    line, pc = z["pa"].tolist(), z["pc"].tolist()
    n = len(line)
    target = np.full(n, -1, np.int64)
    source = np.full(n, -1, np.int64)
    if rule == "next_line":
        a = z["pa"]
        return np.where(a % 64 < 63, a + 1, -1), source
    table = {} if rule == "line_unlimited" else [None] * 256
    if rule == "pc256":
        keys = pc
        slots = [mixed(x) for x in pc]
    elif rule == "pc_line256":
        keys = list(zip(pc, line))
        slots = [mixed(x ^ ((y * 0x9e3779b97f4a7c15) & ((1 << 64)-1))) for x,y in keys]
    else:
        keys = line
        slots = line if rule == "line_unlimited" else [mixed(x) for x in line]
    for i, key in enumerate(keys):
        row = table.get(key) if rule == "line_unlimited" else table[slots[i]]
        if row is not None and row[0] == key:
            previous_cue = row[1]
            candidate = line[previous_cue + horizon]
            if candidate != line[i]:
                assert previous_cue + horizon < i
                target[i] = candidate
                source[i] = previous_cue
        if i >= horizon:
            old = i - horizon
            table[slots[old]] = (keys[old], old)
    return target, source


def trajectory(case, panel):
    folder = STUDY / "raw" / panel / "history" / case
    with np.load(folder / "stride.npz") as f:
        z = {k:f[k] for k in f if k != "trace"}
    line, pc = z["pa"], z["pc"]
    ids = list(zip(z["instr"].tolist(),z["op"].tolist()))
    assert len(ids) == len(set(ids)) and np.all(np.diff(z["t"]) >= 0)
    ref = {}
    for name in POLICIES:
        with np.load(folder / (name + ".npz")) as f:
            ref[name] = {(i, o):(h, p) for i,o,h,p in zip(f["instr"].tolist(), f["op"].tolist(),
                                                          f["hit"].tolist(), f["pa"].tolist())}
    all_miss = np.ones(len(line), bool)
    for name in POLICIES:
        aligned = [ref[name].get(key, (-1, -1)) for key in ids]
        for i, (hit, pa) in enumerate(aligned):
            if hit >= 0:
                assert pa == line[i]
        status = np.array([a[0] for a in aligned], dtype=int)
        if name == "nopf":
            assert np.array_equal(status == 0, z["baseline"] == 0)
        all_miss &= status == 0
    timely_pcs = set(pc[(z["baseline"]==0) & (z["hit"]==1) & (z["pf_kind"]==1)].tolist())
    z["target_never_T"] = np.array([x not in timely_pcs for x in pc.tolist()],bool)
    z["all_miss"] = all_miss
    cut = int(.6 * len(line))
    timer = np.zeros(cut,bool) if not cut else (
        (z["dispatch"][:cut] >= 0) & (z["fill"][:cut] > z["dispatch"][:cut]) & (z["fill"][:cut] <= z["t"][cut-1]))
    service = int(np.ceil(np.median((z["fill"][:cut]-z["dispatch"][:cut])[timer]))) if timer.any() else 128
    previous = {}; first = {}; first_page = {}; recent = []; history = []
    prev = np.empty(len(line),int)
    for i, x in enumerate(line.tolist()):
        prev[i] = previous.get(x,-1)
        previous[x] = i
        first.setdefault(x,i); first_page.setdefault(x//64,i)
        pg = x//64
        recent = [pg] + [k for k in recent if k != pg][:2]
        history.append(tuple(recent))
    return z, prev, first, first_page, history, service, int(timer.sum())


def analyze(item):
    panel, case = item
    z, prev, first, first_pg, recent, service, completed = trajectory(case,panel)
    n = len(z["t"]); result=[]; examples=[]; cold=[]
    for h in (1,4):
        ix = np.arange(int(.8*n), n-h, dtype=int)
        j = ix+h; base = z["pa"][ix]; actual = z["pa"][j]; page = actual//64
        first_use = (actual!=base)&(prev[j]<=ix)
        early = z["t"][j]-z["t"][ix]>=service
        original = z["baseline"][j]==0
        strong = z["all_miss"][j]
        seen = np.array([first[int(x)]<=i for i,x in zip(ix, actual)],bool)
        known_page = np.array([first_pg[int(x)]<=i for i,x in zip(ix, page)],bool)
        nonlocal_page = np.array([int(p) not in recent[i] for i,p in zip(ix,page)],bool)
        distant = np.abs(actual-base)>512
        never = z["target_never_T"][j]
        def count(mask): return int(np.count_nonzero(mask))
        meta=dict(panel=panel,case=case,horizon=h)
        for label,sel in (("all",np.ones(len(ix),bool)),("never_T_pc", never),("had_T_pc",~never)):
            eligible = first_use & sel
            valuable = eligible & strong & early
            cold.append(dict(**meta, target_pc_cohort=label, test_triggers=count(sel), first_uses=count(eligible),
                requested_line_first_seen_after_cue=count(eligible&~seen), requested_page_first_seen_after_cue=count(eligible&~known_page),
                original_miss_first_uses=count(eligible&original), all_controls_miss_with_lead=count(valuable),
                all_controls_miss_with_lead_seen_target=count(valuable&seen),
                all_controls_miss_with_lead_known_page=count(valuable&known_page),
                all_controls_miss_with_lead_same_page=count(valuable&(page==base//64)),
                all_controls_miss_with_lead_known_nonrecent_page=count(valuable&known_page&nonlocal_page),
                all_controls_miss_with_lead_seen_target_distant=count(valuable&seen&distant),
                all_controls_miss_with_lead_seen_target_outside_recent_pages=count(valuable&seen&nonlocal_page),
                all_controls_miss_with_lead_seen_target_distant_outside=count(valuable&seen&distant&nonlocal_page)))
        baseline_good = np.zeros(len(ix),bool)
        for rule in RULES:
            choices, source = issue(z,h,rule)
            offered = choices[ix] > 0
            correct = offered & (choices[ix] == actual) & first_use
            lead = correct & early
            good = lead & strong
            if rule=="next_line":
                baseline_good = good
            for label,sel in (("all",np.ones(len(ix),bool)),("never_T_pc",never),("had_T_pc",~never)):
                result.append(dict(**meta, rule=rule, target_pc_cohort=label, service_cycles=service,
                    training_services_completed=completed, trajectory_ingresses=n, test_triggers=count(sel),
                    offers=count(offered&sel), correct_first_use=count(correct&sel),
                    original_misses_correct=count(correct&original&sel),
                    correct_distant_nonrecent=count(correct&distant&nonlocal_page&sel),
                    correct_nopf_hit=count(correct&(z["baseline"][j]==1)&sel),
                    lead_first_use=count(lead&sel),
                    lead_original_miss=count(lead&original&sel), lead_all_controls_miss=count(good&sel),
                    leads_unique_over_next=count(good&~baseline_good&sel), lead_misses_head_positive=count(good&sel&(z["head"][j]>0)),
                    lead_all_controls_miss_distant=count(good&distant&sel),
                    lead_all_controls_miss_outside_recent_pages=count(good&nonlocal_page&sel),
                    lead_all_controls_miss_distant_outside=count(good&distant&nonlocal_page&sel),
                    median_correct_lead_cycles=float(np.median(z["t"][j][correct&sel]-z["t"][ix][correct&sel])) if count(correct&sel) else ""))
            if rule=="next_line":
                continue
            for kind, mask in (("four_policy_miss_distant_outside_recent_pages",good&nonlocal_page&distant),
                               ("four_policy_miss_outside_recent_pages",good&nonlocal_page),
                               ("four_policy_miss",good), ("original_miss",lead&original),
                               ("correct_distant_outside_recent_pages",correct&distant&nonlocal_page),
                               ("correct_first_use",correct)):
                matches=np.flatnonzero(mask)
                if not len(matches):continue
                k=matches[0];i=ix[k];q=j[k];old=source[i];then=old+h
                assert then<i and z["pa"][then] == z["pa"][q]
                examples.append(dict(**meta,rule=rule,selection=kind,occurrences=len(matches),
                    previous_cue_instruction=int(z["instr"][old]),previous_target_instruction=int(z["instr"][then]),
                    previous_cue_index=int(old),previous_target_index=int(then),cue_index=int(i),target_index=int(q),
                    old_cue_pc=int(z["pc"][old]),old_cue_line=int(z["pa"][old]),
                    cue_instruction=int(z["instr"][i]),cue_operand=int(z["op"][i]),cue_pc=int(z["pc"][i]),cue_line=int(base[k]),
                    cue_cycle=int(z["t"][i]),target_instruction=int(z["instr"][q]),target_operand=int(z["op"][q]),
                    target_pc=int(z["pc"][q]),target_line=int(actual[k]),target_cycle=int(z["t"][q]),target_first_in_roi=int(first[int(actual[k])]),
                    target_head=int(z["head"][q]),original_tag=int(z["baseline"][q]),stride_tag=int(z["hit"][q]),original_miss=int(original[k]),all_four_miss=int(strong[k]),lead=int(early[k]),
                    target_outside_recent=int(nonlocal_page[k]),delta_lines=int(actual[k]-base[k])))
    return result,cold,examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    args=parser.parse_args()
    with (STUDY / "intake/registry.csv").open() as f:
        roster=[r for r in csv.DictReader(f) if r["split"]=="discovery"]
    jobs = [(p,r["case"]) for p in ("screen","replication","long_google") for r in roster
            if p!="long_google" or r["collection"]=="gtrace_v2"]
    with ProcessPoolExecutor(args.jobs) as pool:
        rows=list(pool.map(analyze,jobs))
    for col,name in enumerate(("forecasts","reach","examples")):
        data=[row for entry in rows for row in entry[col]]
        with (HERE/(name+".csv")).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator="\n")
            w.writeheader();w.writerows(data)
    print(f"Completed {len(jobs)} recordings/windows, both horizons and all five rules.")


if __name__=="__main__":
    main()
