"""Diagnostic multi-target window and decomposition of finite-table support."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from controls import select
from discovery import HERE, ROOT, HORIZON, TASKS, write, counts, freeze, choice, pct

EXAMPLE=("case","rule","target_instr","target_operand","target_pc","target_line","trigger_instr","trigger_operand","trigger_pc","trigger_line","local_key","previous_three_pc","conditional_train_observations","conditional_train_match_queries","chosen_action","lead_cycles","lead_events","service_reference","covered_by_offset_or_fixed_controls","stride_class","multi_class","spp_class","stride_head")

RULES=("next_line","one_trained_delta","offset","local","previous_pc","ordered_pc3","unordered_pc3")


def support(rec):
    st=rec.st;out=[]
    for h,q in rec.targets.items():
        end=st.splits()["train"][1];lo,hi=st.splits()["test"]
        for name in ("local","previous_pc","ordered_pc3","relative_order"):
            raw=counts(st,q,end,name,h); fitted=freeze(raw)
            priority=sorted(raw,key=lambda k:(-sum(raw[k].values()),k))
            top=set(priority[:256])
            score=Counter()
            for i in range(lo,max(lo,hi-HORIZON[h])):
                key=st.keys[name][i];score["eligible"]+=1
                if key not in raw: why="unseen"
                elif sum(raw[key].values())<3: why="training_support_below_three"
                elif not any(a is not None for a in raw[key]): why="no_legal_training_target"
                elif key not in top: why="outside_256"
                else: why="represented"
                score[why]+=1
            cols=("eligible","unseen","training_support_below_three","no_legal_training_target","outside_256","represented")
            assert score["eligible"]==sum(score[c] for c in cols[1:])
            out.append(dict(case=rec.name,horizon=h,rule=name,**{c:score[c] for c in cols}))
    return out


def candidates(st,i,service):
    c={}
    for j in range(i+1,i+17):
        target=st.line[j]
        if st.prev[j]>i or st.t[j]-st.t[i]<service:
            continue
        act=st.action(i,target)
        if act is not None:c[act]=j
    return c


def run(rec):
    st=rec.st;t=st.splits();service=st.service()
    train_end=t["train"][1]
    active=RULES[2:]
    raw={r:defaultdict(Counter) for r in active}; counts_={r:Counter() for r in active}
    most=Counter()
    for i in range(max(0,train_end-16)):
        c=candidates(st,i,service)
        # Count support once per query even when many candidate actions occur.
        for r in active:
            key=st.keys[r][i]; counts_[r][key]+=1;raw[r][key].update(c.keys())
        for j in c.values():most[st.line[j]-st.line[i]]+=1
    constant=min(most,key=lambda k:(-most[k],abs(k),k)) if most else None
    tables={}
    for name in active:
        order=sorted(counts_[name],key=lambda k:(-counts_[name][k],k))[:256]
        tab={}
        for k in order:
            cc=raw[name][k]
            if cc and counts_[name][k]>=3:
                tab[k]=min(cc,key=lambda a:(-cc[a],a))
        tables[name]=tab
    rows=[];pair=[];examples=[]
    for part in ("dev","test"):
        lo,hi=t[part]
        obs={r:Counter() for r in RULES};anchors={r:{} for r in RULES}
        hits={r:set() for r in RULES};strong={r:set() for r in RULES}
        for i in range(lo,max(lo,hi-16)):
            legal=candidates(st,i,service);byline={st.line[j]:j for j in legal.values()}
            for name in RULES:
                c=obs[name];c["eligible"]+=1
                if name in active:
                    act=tables[name].get(st.keys[name][i])
                    target=st.decode(i,act)
                elif name=="next_line":
                    target=st.decode(i,("delta",1))
                else:
                    target=st.line[i]+constant if constant is not None else None
                    if target is not None and st.action(i,target) is None:target=None
                if target is None:continue
                c["offer"]+=1
                if target not in byline:continue
                c["match"]+=1;j=byline[target]
                if st.z["baseline"][j]!=0:continue
                hits[name].add(j)
                anchors[name].setdefault(j,i)
                if all(rec.out[p][j] in ("L","P") for p in ("spp","multi")):strong[name].add(j)
        for name in RULES:
            c=obs[name]
            rows.append(dict(case=rec.name,split=part,rule=name,service=service,eligible=c["eligible"],offers=c["offer"],
                             timely_address_matches=c["match"],offer_match_pct=pct(c["match"],c["offer"]),
                             distinct_original_misses_named=len(hits[name]),both_strong_missed=len(strong[name]),
                             both_strong_missed_with_head=sum(int(st.z["head"][j]>0) for j in strong[name]),
                             train_delta=constant if name=="one_trained_delta" else ""))
            if part=="test" and name in ("previous_pc","ordered_pc3","unordered_pc3"):
                a,b=hits[name],hits["local"];new=strong[name]-hits["local"]
                simple=hits["offset"]|hits["one_trained_delta"]|hits["next_line"]|b
                exclusive=new-simple
                for j in sorted(new,key=lambda k:(k not in exclusive,-int(st.z["head"][k]),k))[:2]:
                    i=anchors[name][j];key=st.keys[name][i];act=tables[name][key]
                    vals=(rec.name,name,st.z["instr"][j],st.z["op"][j],st.pc[j],st.line[j],
                          st.z["instr"][i],st.z["op"][i],st.pc[i],st.line[i],str(st.keys["local"][i]),str(st.older_pcs[i]),
                          counts_[name][key],raw[name][key][act],str(act),st.t[j]-st.t[i],j-i,service,
                          int(j in simple),rec.out["stride"][j],rec.out["multi"][j],rec.out["spp"][j],st.z["head"][j])
                    examples.append(dict(zip(EXAMPLE,vals)))
                pair.append(dict(case=rec.name,rule=name,added_vs_local=len(a-b),lost_vs_local=len(b-a),
                                 new_strong_miss_outside_union_of_four_simple=len(exclusive),
                                 net_vs_local=len(a)-len(b),new_both_strong_missed=len(new),
                                 new_strong_miss_with_head=sum(int(st.z["head"][j]>0) for j in new)))
    return rows,pair,examples


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--history-root",type=Path,default=ROOT/"raw/screen/history")
    ap.add_argument("--output",type=Path,default=HERE);ap.add_argument("--case",nargs="+")
    args=ap.parse_args();totals=[];pairs=[];diagnostic=[];witness=[]
    for rec in select(args):
        r,p,w=run(rec);totals+=r;pairs+=p;witness+=w;diagnostic+=support(rec)
    write("any_in_window.csv",totals,folder=args.output);write("window_additions.csv",pairs,folder=args.output)
    write("key_support.csv",diagnostic,folder=args.output)
    write("window_examples.csv",witness,EXAMPLE,args.output)
