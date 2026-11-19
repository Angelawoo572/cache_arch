"""Fixed-distance and per-PC cadence controls; training-selected interleaved motifs."""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
from statistics import median
from discovery import ROOT, HERE, Recording, TASKS, HORIZON, write, pct, mid

RULES=("fixed_k","stride_times_k","pc_cadence","one_trained_delta")


def select(args):
    with (ROOT/"intake/registry.csv").open() as f:
        chosen=[r for r in csv.DictReader(f) if r["split"]=="discovery"]
    if args.case:
        if set(args.case)-{r["case"] for r in chosen}:
            raise ValueError("Only discovery cases are eligible")
        chosen=[r for r in chosen if r["case"] in args.case]
    return [Recording(r,args.history_root) for r in chosen]


def cadence(st):
    back={}; period=[]; repeats=[]; prev_d={}
    for i,pc in enumerate(st.pc):
        period.append(i-back[pc] if pc in back else 0)
        d=st.stride[i];repeats.append(d is not None and d!=0 and d==prev_d.get(pc))
        back[pc]=i;prev_d[pc]=d
    return period,repeats


def motif(rec,stable):
    st=rec.st
    tr=st.splits()["train"][1];lo,hi=st.splits()["test"]
    def words(lo,hi):
        for i in range(lo+3,hi):
            if not all(stable[i-3:i+1]) or len(set(st.pc[i-3:i+1]))<2:
                continue
            yield i,tuple(zip(st.pc[i-3:i+1],st.stride[i-3:i+1]))
    train=Counter(w for i,w in words(0,tr))
    m=min(train,key=lambda w:(-train[w],w)) if train else None
    ids=[i for i,w in words(lo,hi) if w==m and train[m]>=3]
    i=ids[0] if ids else -1
    return dict(case=rec.name,train_windows=max(0,tr-3),test_windows=max(0,hi-lo-3),
                trained_motif=repr(m) if m is not None else "",training_occurrences=train.get(m,0),
                suffix_occurrences=len(ids),suffix_outcome_timely=sum(rec.out["stride"][j]=="T" for j in ids),
                suffix_outcome_late=sum(rec.out["stride"][j]=="L" for j in ids),
                suffix_outcome_persistent=sum(rec.out["stride"][j]=="P" for j in ids),
                suffix_strong_missed=sum(all(rec.out[p][j] in ("P","L") for p in ("multi","spp")) for j in ids),
                example_instr_sequence=repr(st.z["instr"][i-3:i+1].tolist()) if i>=0 else "",
                example_physical_lines=repr(st.line[i-3:i+1]) if i>=0 else "",
                example_interarrival_cycles=repr([st.t[j]-st.t[j-1] for j in range(i-2,i+1)]) if i>=0 else "")


def study(rec):
    st=rec.st;period,stable=cadence(st);service=st.service(); rows=[];witness=[]
    for task in TASKS:
        q=rec.targets[task];tr=st.splits()["train"][1];delta=Counter()
        for i in range(max(0,tr-HORIZON[task])):
            j=q[i]
            if j>i and st.prev[j]<=i and st.action(i,st.line[j]) is not None:
                delta[st.line[j]-st.line[i]]+=1
        fixed=min(delta,key=lambda d:(-delta[d],abs(d),d)) if delta else None
        for part in ("dev","test"):
            lo,hi=st.splits()[part];counts={r:Counter() for r in RULES}
            for i in range(lo,max(lo,hi-HORIZON[task])):
                j=q[i];h=1 if task=="same_pc" else HORIZON[task]
                d=st.stride[i]
                lag=period[i]
                if task=="same_pc":
                    proj=d
                else:
                    proj=d*(h//lag) if d is not None and lag and h%lag==0 else None
                targets={"fixed_k":h,"one_trained_delta":fixed,"stride_times_k":d*h if d is not None else None,
                         "pc_cadence":proj}
                for name,disp in targets.items():
                    c=counts[name];c["eligible"]+=1
                    if disp is None or st.action(i,st.line[i]+disp) is None:continue
                    c["offers"]+=1
                    if j<0 or st.prev[j]>i or st.line[j]!=st.line[i]+disp:continue
                    c["correct"]+=1
                    if st.z["baseline"][j]!=0 or st.t[j]-st.t[i]<service:continue
                    c["good"]+=1
                    strong=all(rec.out[p][j] in ("P","L") for p in ("multi","spp"))
                    c["strong"]+=strong
                    if strong and st.z["head"][j]>0:
                        c["head"]+=1
                    if part=="test" and strong and c["strong"]==1:
                        witness.append(dict(case=rec.name,horizon=task,rule=name,train_selected_delta=fixed,
                                            cue_instr=st.z["instr"][i],cue_pc=st.pc[i],cue_line=st.line[i],
                                            previous_same_PC_interval=lag,previous_same_PC_delta=d,target_instr=st.z["instr"][j],
                                            target_line=st.line[j],cycles=st.t[j]-st.t[i],service=service,
                                            stride=rec.out["stride"][j],multi=rec.out["multi"][j],spp=rec.out["spp"][j]))
            for name,c in counts.items():
                rows.append(dict(case=rec.name,horizon=task,split=part,rule=name,
                                 train_selected_delta=fixed if name=="one_trained_delta" else "",
                                 eligible=c["eligible"],offers=c["offers"],correct=c["correct"],
                                 original_with_lead=c["good"],both_strong_tag_miss=c["strong"],
                                 both_strong_tag_miss_head_positive=c["head"],precision=pct(c["correct"],c["offers"])))
    return rows,witness,motif(rec,stable)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case",nargs="+");ap.add_argument("--history-root",type=Path,default=ROOT/"raw/screen/history")
    ap.add_argument("--output",type=Path,default=HERE)
    args=ap.parse_args();output=[];examples=[];motifs=[]
    for rec in select(args):
        r,w,m=study(rec);output+=r;examples+=w;motifs.append(m)
    args.output.mkdir(exist_ok=True,parents=True)
    write("distance_controls.csv",output,folder=args.output)
    write("interleaved_motifs.csv",motifs,folder=args.output)
    write("distance_examples.csv",examples,("case","horizon","rule","train_selected_delta","cue_instr","cue_pc","cue_line",
        "previous_same_PC_interval","previous_same_PC_delta","target_instr","target_line","cycles","service","stride","multi","spp"),args.output)
