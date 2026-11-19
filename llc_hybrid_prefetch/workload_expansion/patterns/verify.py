"""Check target legality, causal feature invariance, censored supervision and recorded witnesses."""
import argparse
import ast
import csv
from pathlib import Path
import numpy as np
from discovery import HERE, ROOT, MODES, RULES, TASKS, HORIZON, Stream, Recording, counts, freeze


def csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def toy(lines,pcs):
    n=len(lines)
    return dict(pa=np.asarray(lines),pc=np.asarray(pcs),t=np.arange(n)*8,baseline=np.zeros(n,dtype=int),
                dispatch=np.arange(n)*8,fill=np.arange(n)*8+3)


def verify_causality():
    lines=[1024+i%43 if i%5 else 2048+i%11 for i in range(160)]
    pcs=[7 if i%3 else 11 for i in range(160)]
    a=Stream(toy(lines,pcs));z=toy(lines[:96]+[3100+j*128 for j in range(64)],pcs[:96]+[17]*64)
    z["baseline"]=np.ones(len(lines),dtype=int);b=Stream(z)
    for name in a.keys:assert a.keys[name][:96]==b.keys[name][:96]
    for task in TASKS:
        for name in a.keys:
            tr1=counts(a,a.task(task),96,name,task);tr2=counts(b,b.task(task),96,name,task)
            assert tr1==tr2
            assert sum(sum(c.values()) for c in tr1.values())==96-HORIZON[task]
    assert a.prior_pages[:96]==b.prior_pages[:96]
    for i in range(96):
        for target in lines[:i+1]+[0,6000]:
            code=a.action(i,target)
            assert code==b.action(i,target)
            if code is not None:
                assert a.decode(i,code)==target
            assert a.decode(i,("line",6000)) is None
    for name in a.keys:
        fit=freeze(counts(a,a.task("next"),96,name,"next"))
        assert len(fit.rows)<=256
    print("Causal prefix, horizon, positive physical address and page legality checks passed.")


def verify_data(table_root,history):
    reg={r["case"]:r for r in csv_rows(ROOT/"intake/registry.csv") if r["split"]=="discovery"}
    summaries=csv_rows(table_root/"scores.csv");cases={r["case"] for r in summaries}
    assert cases and cases<=reg.keys()
    pop=csv_rows(table_root/"sequence_populations.csv")
    assert len(pop)==len(cases)*3*3
    for r in pop:
        assert sum(int(r[k]) for k in ("T","L","P","H","other"))==int(r["ingresses"])
        assert int(r["never_T_original_misses"])<=int(r["never_T_events"])<=int(r["ingresses"])
    for r in summaries:
        n=lambda k:int(r[k])
        assert n("table_keys")<=256 and n("eligible")>=n("offers")>=n("exact")>=n("original_with_lead")
        assert n("original_with_lead")>=n("both_strong_tag_miss")>=n("both_strong_tag_miss_head_positive")
    w=csv_rows(table_root/"recurring_contexts.csv")
    loaded={}
    for r in w:
        n=lambda k:int(r[k])
        rec=loaded.get(r["case"])
        if rec is None:
            rec=loaded[r["case"]]=Recording(reg[r["case"]],history)
        st=rec.st
        ix=np.flatnonzero((st.z["instr"]==n("trigger_instr"))&(st.z["op"]==n("trigger_op")))
        assert len(ix)==1
        i=int(ix[0]);j=rec.targets[r["horizon"]][i]
        assert j>i and st.splits()["test"][0]<=i
        assert st.z["instr"][j]==n("target_instr") and st.z["op"][j]==n("target_op")
        assert st.action(i,st.line[j])==ast.literal_eval(r["chosen_action"])
        assert st.keys[r["conditional_rule"]][i]==ast.literal_eval(r["conditional_key"])
        assert st.line[j]==n("target_line") and st.t[j]-st.t[i]==n("lead_cycles")
        assert all(rec.out[mode][j]==r[mode] for mode in ("stride","spp","multi"))
        if r["version"]=="primary":
            assert st.z["baseline"][j]==0 and st.t[j]-st.t[i]>=st.service()
    lookup=csv_rows(table_root/"window_examples.csv")
    for row in lookup:
        rec=loaded.get(row["case"])
        if rec is None:
            rec=loaded[row["case"]]=Recording(reg[row["case"]],history)
        st=rec.st;z=st.z
        find=lambda prefix:int(np.flatnonzero((z["instr"]==int(row[prefix+"_instr"]))&(z["op"]==int(row[prefix+"_operand"])))[0])
        i,j=find("trigger"),find("target");act=ast.literal_eval(row["chosen_action"])
        assert 1<=j-i<=16 and st.prev[j]<=i and st.decode(i,act)==st.line[j]
        assert st.t[j]-st.t[i]>=st.service() and z["baseline"][j]==0
        assert st.t[j]-st.t[i]==int(row["lead_cycles"]) and st.pc[i]==int(row["trigger_pc"])
        assert all(rec.out[pol][j]==row[pol+"_class"] for pol in ("stride","multi","spp"))
        assert rec.out["spp"][j] in ("L","P") and rec.out["multi"][j] in ("L","P")
    for r in csv_rows(table_root/"window_additions.csv"):
        assert int(r["added_vs_local"])-int(r["lost_vs_local"])==int(r["net_vs_local"])
        assert int(r["new_strong_miss_outside_union_of_four_simple"])<=int(r["new_both_strong_missed"])<=int(r["added_vs_local"])
    print(f"Verified {len(cases)} discovery recordings, {len(summaries)} rule/split/horizon rows, {len(w)} fixed-horizon witness roles and {len(lookup)} flexible-window examples.")


if __name__=="__main__":
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument("--output",type=Path,default=HERE)
    args.add_argument("--history-root",type=Path,default=ROOT/"raw/screen/history")
    opt=args.parse_args()
    verify_causality()
    verify_data(opt.output,opt.history_root)
