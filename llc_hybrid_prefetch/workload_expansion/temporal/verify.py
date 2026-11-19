"""Independently replay prefix-only table operations and check reach/score populations."""
import csv

import numpy as np

from study import HERE, RULES, issue, mixed


def manual(z,h,rule):
    slots = {}
    arrivals = []
    predictions = []
    for line, pc in zip(z["pa"].tolist(), z["pc"].tolist()):
        if rule == "next_line":
            predictions.append(line+1 if line%64!=63 else -1)
        else:
            if rule == "pc256":
                key, address = pc, mixed(pc)
            elif rule == "pc_line256":
                key = (pc,line)
                address = mixed(pc ^ ((line * 0x9e3779b97f4a7c15) & ((1<<64)-1)))
            else:
                key = line
                address = mixed(line) if rule == "line256" else line
            hit = slots.get(address)
            predictions.append(hit[1] if hit is not None and hit[0] == key and hit[1] != line else -1)
            if len(arrivals)>=h:
                prev_key, prev_slot = arrivals[-h]
                slots[prev_slot] = (prev_key,line)
            arrivals.append((key,address))
    return predictions


def tests():
    rng=np.random.default_rng(447)
    z={"pa":np.tile([13,5500,19,20000,13,5500,19],30), "pc":rng.integers(1,800,210)}
    for h in (1,4):
        for rule in RULES:
            a,_=issue(z,h,rule)
            assert np.array_equal(a,manual(z,h,rule))
            for stop in (11,90,158):
                short={k:v[:stop] for k,v in z.items()}
                b,_=issue(short,h,rule)
                assert np.array_equal(a[:stop],b)
    a,src=issue(z,1,"line_unlimited")
    assert a[4]==5500 and src[4]==0
    with (HERE/"forecasts.csv").open() as f: scores=list(csv.DictReader(f))
    with (HERE/"reach.csv").open() as f: bounds=list(csv.DictReader(f))
    key=lambda r:(r["panel"],r["case"],r["horizon"],r["target_pc_cohort"])
    ceilings={key(r):r for r in bounds}
    assert len(ceilings)==39*2*3 and len(scores)==len(ceilings)*len(RULES)
    for r in scores:
        b=ceilings[key(r)]
        assert int(r["test_triggers"])==int(b["test_triggers"])
        assert int(r["lead_all_controls_miss"])<=int(r["lead_original_miss"])<=int(r["correct_first_use"])<=int(r["offers"])<=int(r["test_triggers"])
        if r["rule"]!="next_line":
            assert int(r["lead_all_controls_miss"])<=int(b["all_controls_miss_with_lead_seen_target"])
    print("Matched independent simulations of all five rules at both horizons, causal truncation, and 1,170 score rows.")


if __name__=="__main__":
    tests()
