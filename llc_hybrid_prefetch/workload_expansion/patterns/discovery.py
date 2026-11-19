"""Discover context-dependent physical addresses on positive-list discovery runs."""
from collections import Counter, defaultdict
import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import median
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODES = ("nopf", "stride", "multi", "spp")
HORIZON = {"next":1, "fourth":4, "sixteenth":16, "same_pc":64}
TASKS = tuple(HORIZON)
WIDE = (-512, -256, -128, -64, 64, 128, 256, 512)
RULES = ("next_line", "repeat_delta", "temporal", "offset", "local", "global_delta",
         "previous_pc", "ordered_pc3", "unordered_pc3", "relative", "relative_order")
TAGS = ("T", "L", "P", "H", "other")


def write(name, rows, fields=None, folder=HERE):
    fields = fields or (list(rows[0]) if rows else None)
    if fields is None:
        raise ValueError("Missing schema for empty " + name)
    with (folder / name).open("w", newline="") as handle:
        w = csv.DictWriter(handle, fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def pct(a, b):
    return round(100*a/b, 3) if b else ""


def mid(values):
    return round(float(median(values)), 1) if len(values) else ""


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k].astype(np.int64) for k in archive.files if k != "trace"}


def ids(z):
    return z["instr"]*256 + z["op"]


def labels(z):
    return np.select(((z["baseline"]==0)&(z["hit"]==1)&(z["pf_kind"]==1),
                      (z["baseline"]==0)&(z["hit"]==0)&(z["pf_kind"]==2),
                      (z["baseline"]==0)&(z["hit"]==0),
                      (z["baseline"]==0)&(z["hit"]==1)), TAGS[:4], default="other")


def clipped(value):
    return max(-64, min(64, value))


def identity_pattern(seq):
    named = {}
    return tuple(-1 if v == -1 else named.setdefault(v, len(named)) for v in seq)


class Stream:
    def __init__(self, z):
        self.z = z
        self.line, self.pc, self.t = [z[k].astype(int).tolist() for k in ("pa","pc","t")]
        self.n = len(self.t)
        self.prev = []
        self.prior_pages = []
        self.next_pc = np.full(self.n, -1, dtype=int)
        self.stride = []
        self.older_pcs = []
        self.keys = defaultdict(list)
        last_pc, last_line, self.seen = {}, {}, {}
        recent = []
        for i, (line, pc) in enumerate(zip(self.line, self.pc)):
            pg, off = divmod(line,64)
            self.seen.setdefault(pg, i)
            self.prior_pages.append(tuple(q for q in recent if q != pg)[:2])
            recent = [pg] + [q for q in recent if q != pg][:2]
            self.prev.append(last_line.get(line,-1))
            last_line[line] = i
            lp = last_pc.get(pc,-1)
            if lp >= 0:
                self.next_pc[lp] = i
            d = line - self.line[lp] if lp >= 0 else None
            self.stride.append(d)
            last_pc[pc] = i
            gd = clipped(line-self.line[i-1]) if i else 65
            local = (pc, off, clipped(d) if d is not None else 65)
            prefix = tuple(self.pc[j] if j>=0 else -1 for j in range(i-3,i))
            rel = (off, local[2], gd)
            self.older_pcs.append(prefix)
            for key, value in (("offset", (off,)), ("local",local), ("global_delta", (*local,gd)),
                               ("previous_pc", (*local,prefix[-1])), ("ordered_pc3",(*local,*prefix)),
                               ("unordered_pc3",(*local,*sorted(prefix))), ("relative",rel),
                               ("relative_order",(*rel,*identity_pattern((*prefix,pc)))),
                               ("temporal",(line,))):
                self.keys[key].append(value)

    def action(self,i,target):
        line = self.line[i]
        if line == target or target<=0:
            return None
        page, off = divmod(target,64)
        if page == line//64:
            return ("delta", target-line)
        pages = self.prior_pages[i]
        if page in pages:
            return ("page"+str(pages.index(page)+1), off)
        delta = target-line
        if delta in WIDE and self.seen.get(page,self.n)<=i:
            return ("wide",delta)
        return None

    def decode(self,i,act):
        if act is None:
            return None
        code, val = act
        line = self.line[i]
        if code=="delta":
            target = line+val
            return target if target>0 and target!=line and target//64==line//64 else None
        if code.startswith("page"):
            j = int(code[-1])-1
            target=self.prior_pages[i][j]*64+val if j<len(self.prior_pages[i]) else 0
            return target if target>0 else None
        target = val if code=="line" else line+val
        return target if target>0 and target!=line and self.seen.get(target//64,self.n)<=i else None

    def task(self,kind):
        if kind=="same_pc":
            return [int(v) if i<v<=i+64 else -1 for i,v in enumerate(self.next_pc)]
        gap={"next":1,"fourth":4,"sixteenth":16}[kind]
        return [i+gap if i+gap<self.n else -1 for i in range(self.n)]

    def splits(self):
        a,b=int(.6*self.n),int(.8*self.n)
        return {"train":(0,a),"dev":(a,b),"test":(b,self.n)}

    def service(self):
        end = self.splits()["train"][1]
        if not end:
            return 128
        z=self.z
        good = (z["dispatch"][:end]>=0)&(z["fill"][:end]>z["dispatch"][:end])&(z["fill"][:end]<=self.t[end-1])
        vals = (z["fill"][:end]-z["dispatch"][:end])[good]
        return int(np.ceil(np.median(vals))) if len(vals) else 128


@dataclass
class Fitted:
    rows: dict
    raw: dict


def counts(st, q, stop, rule, task):
    counts = defaultdict(Counter)
    for i in range(max(0,stop-HORIZON[task])):
        j=q[i]
        if j<0:
            counts[st.keys[rule][i]][None]+=1
            continue
        assert j<stop
        target = st.line[j]
        act = st.action(i,target) if rule!="temporal" else (("line",target) if target!=st.line[i] else None)
        if st.prev[j] > i:
            act=None
        counts[st.keys[rule][i]][act]+=1
    return counts


def freeze(raw):
    """Keep the 256 most frequently observed keys; count all, including no legal target."""
    popular=sorted(raw, key=lambda k:(-sum(raw[k].values()),k))[:256]
    out={}
    for key in popular:
        c=raw[key]
        if sum(c.values())<3:
            continue
        valid=[a for a in c if a is not None]
        if valid:
            out[key]=min(valid,key=lambda a:(-c[a],a))
    return Fitted(out,raw)


def choice(st,i,rule,fit):
    if rule=="next_line":
        return st.decode(i,("delta",1))
    if rule=="repeat_delta":
        return st.decode(i,("delta",st.stride[i])) if st.stride[i] else None
    return st.decode(i,fit.rows.get(st.keys[rule][i]))


class Recording:
    def __init__(self, rec, history_root):
        self.name,self.family = rec["case"],rec["collection"]
        path=history_root/self.name
        self.arms={p:load(path/(p+".npz")) for p in MODES}
        self.st=Stream(self.arms["stride"])
        z=self.st.z
        base=ids(z)
        self.out={}
        for p,other in self.arms.items():
            by={int(k):i for i,k in enumerate(ids(other))}
            state=labels(other)
            result=[]
            for i,k in enumerate(base):
                j=by.get(k)
                if j is not None:
                    assert all(z[f][i]==other[f][j] for f in ("pc","va","pa")), (p,k)
                result.append(state[j] if j is not None else "absent")
            self.out[p]=result
        self.targets={kind:self.st.task(kind) for kind in TASKS}

    def contexts(self):
        summaries=[]
        for pol in MODES[1:]:
            z=self.arms[pol]; tags=labels(z)
            total=len(tags)
            for view,group in (("global",np.zeros(total,dtype=int)),("pc",z["pc"]),("page",z["pa"]//64)):
                streak=Counter(); per_group=defaultdict(Counter); counts=Counter()
                last={};last_delta={};switch_returns=0;repeat_after_switch=0
                for i,(k,tag) in enumerate(zip(group,tags)):
                    k=int(k)
                    if streak[k]>=2:
                        counts["after_two_"+tag]+=1
                    streak[k]=(streak[k]+1 if tag=="T" else 0)
                    c=per_group[k];c[tag]+=1
                    c["head_miss"]+=int(tag in ("L","P") and z["head"][i]>0)
                    c["original_miss"]+=int(z["baseline"][i]==0)
                    if k in last:
                        j=last[k];delta=int(z["pa"][i]-z["pa"][j])
                        if i-j>1:
                            switch_returns+=1
                            repeat_after_switch+=int(last_delta.get(k)==delta)
                        last_delta[k]=delta
                    last[k]=i
                without=[c for c in per_group.values() if not c["T"]]
                rows=dict(case=self.name,policy=pol,view=view,ingresses=total,streams=len(per_group),
                          same_group_return_after_interruption=switch_returns,repeat_delta_on_return=repeat_after_switch,
                          global_pc_switches=int(np.sum(z["pc"][1:]!=z["pc"][:-1])),
                          global_page_switches=int(np.sum(z["pa"][1:]//64!=z["pa"][:-1]//64)),
                          never_T_streams=len(without),never_T_events=sum(sum(c[k] for k in TAGS) for c in without),
                          never_T_original_misses=sum(c["original_miss"] for c in without),
                          never_T_head_exposed_paired_misses=sum(c["head_miss"] for c in without))
                rows.update({k:int(sum(tags==k)) for k in TAGS})
                rows.update({"after_two_"+k:counts["after_two_"+k] for k in TAGS})
                summaries.append(rows)
        return summaries

    def earliest(self):
        st=self.st; service=st.service()
        categories={k:Counter() for k in ("strong_missed","other")}; gaps=defaultdict(lambda:([],[]))
        for j in range(st.n):
            if st.z["baseline"][j]!=0:
                continue
            group="strong_missed" if all(self.out[p][j] in ("L","P") for p in ("spp","multi")) else "other"
            c=categories[group]; c["targets"]+=1
            limit=max(0,j-64,st.prev[j]+1)
            legal=False
            margin=False
            for i in range(limit,j):
                if st.t[j]-st.t[i]>=service:
                    margin=True
                act=st.action(i,st.line[j])
                if act is None:
                    continue
                legal=True
                if st.t[j]-st.t[i]>=service:
                    c["feasible"]+=1
                    c[act[0]]+=1
                    gaps[group][0].append(st.t[j]-st.t[i]);gaps[group][1].append(j-i)
                    break
            c["has_lead"]+=int(margin)
            c["legal"]+=int(legal)
            c["positive_head"]+=int(st.z["head"][j]>0)
        return [dict(case=self.name,cohort=group,original_demand_misses=c["targets"],head_exposed=c["positive_head"],
                     reference_service=service,has_early_observation=c["has_lead"],has_legal_address=c["legal"],
                     has_legal_address_and_lead=c["feasible"],earliest_samepage=c["delta"],earliest_previous_page=c["page1"],
                     earliest_second_page=c["page2"],earliest_wide=c["wide"],median_max_cycles=mid(gaps[group][0]),
                     median_max_events=mid(gaps[group][1])) for group,c in categories.items()]


def evaluate(rec,task,part,rule,table,service,transfer="within"):
    st=rec.st; z=st.z;q=rec.targets[task]
    lo,hi=st.splits()[part]
    c=Counter(); correct=set(); good=set(); offers=set()
    for i in range(lo,max(lo,hi-HORIZON[task])):
        j=q[i]
        # Labels crossing the right boundary are censored; unknown outcomes receive no training credit.
        assert j==-1 or i<j<hi
        c["eligible"]+=1
        target=st.line[j] if j>=0 else None
        distinct=(j>=0 and target!=st.line[i] and st.prev[j]<=i)
        miss=j>=0 and z["baseline"][j]==0
        legal=distinct and st.action(i,target) is not None
        lead=st.t[j]-st.t[i] if j>=0 else 0
        c["original_miss"]+=int(miss)
        c["reachable_miss"]+=int(miss and legal and lead>=service)
        if rule not in ("next_line","repeat_delta"):
            c["supported"]+=int(st.keys[rule][i] in table.rows)
        pred=choice(st,i,rule,table)
        if pred is None:
            continue
        c["offered"]+=1; offers.add(i)
        if not distinct or pred!=target:
            continue
        correct.add(i);c["exact"]+=1
        if not miss or lead<service:
            continue
        good.add(i)
        c["exposed"]+=int(z["head"][j]>0)
        for mode in ("stride","multi","spp"):
            c["native_"+mode]+=int(rec.out[mode][j]=="T")
        if all(rec.out[p][j] in ("L","P") for p in ("multi","spp")):
            c["strong_miss"]+=1
            c["strong_head"]+=int(z["head"][j]>0)
    row=dict(case=rec.name,collection=rec.family,fit=transfer,split=part,horizon=task,rule=rule,
             reference_service=service,eligible=c["eligible"],original_misses=c["original_miss"],
             reachable_original_with_lead=c["reachable_miss"],table_keys=len(table.rows) if table else 0,
             supported_anchors=c["supported"],offers=c["offered"],exact=c["exact"],offer_precision_pct=pct(c["exact"],c["offered"]),
             eligible_accuracy_pct=pct(c["exact"],c["eligible"]),original_with_lead=len(good),
             original_with_lead_head_positive=c["exposed"],native_stride_timely=c["native_stride"],
             native_multi_timely=c["native_multi"],native_spp_timely=c["native_spp"],
             both_strong_tag_miss=c["strong_miss"],both_strong_tag_miss_head_positive=c["strong_head"])
    return row,good,correct,offers


WITNESS=("case","horizon","conditional_rule","local_key","version","chosen_action","other_trained_action",
         "local_train_n","local_train_action_n","conditional_key","conditional_train_n","conditional_train_action_n",
         "trigger_instr","trigger_op","trigger_pc","trigger_line","older_three_pcs","target_instr","target_op","target_pc",
         "target_line","lead_cycles","lead_events","prev_global_pc","service_reference","baseline_miss","stride","multi","spp",
         "stride_operand_head","target_page_seen_before_cue","other_future_trigger_id")


def witness(rec,task,rule,local,rich,good,correct):
    """For a pair to qualify, both training alternatives must recur in the untouched suffix."""
    st=rec.st;q=rec.targets[task];service=st.service()
    variants=defaultdict(lambda:defaultdict(list))
    for i in sorted(correct):
        lk=st.keys["local"][i]; key=st.keys[rule][i]
        act=rich.rows[key]
        if lk not in local.rows:
            continue
        # Exclude same action encoded on different pages as evidence of two futures.
        variants[lk][act].append(i)
    pairs=[]; counts_=Counter(); positives=set(); strong=set()
    for lk,acts in variants.items():
        if len(acts)<2:
            continue
        counts_["contexts"]+=1
        for act, anchors in acts.items():
            valid=[i for i in anchors if i in good]
            for i in valid:
                positives.add(i)
                if all(rec.out[p][q[i]] in ("L","P") for p in ("multi","spp")):
                    strong.add(i)
            if not valid:
                continue
            # Choose the earliest example in each explanatory class, prioritizing true strong misses.
            a=min(valid,key=lambda i:(i not in strong,i))
            b=min(i for other,ii in acts.items() if act!=other for i in ii)
            pairs.append((a not in strong, a, b,lk))
    rows=[]
    for _,a,b,lk in sorted(pairs)[:2]:
        for i,alt in ((a,b),(b,a)):
            j=q[i];other=q[alt];key=st.keys[rule][i]; act=rich.rows[key]
            qa=st.action(i,st.line[j]); assert qa==act
            ca=rich.raw[key];lc=local.raw[lk]
            values=(rec.name,task,rule,str(lk),"primary" if i==a else "alternative",str(act),
                    str(st.action(alt,st.line[other])),sum(lc.values()),lc[act],str(key),sum(ca.values()),ca[act],
                    st.z["instr"][i],st.z["op"][i],st.pc[i],st.line[i],str(st.older_pcs[i]),st.z["instr"][j],
                    st.z["op"][j],st.pc[j],st.line[j],st.t[j]-st.t[i],j-i,st.older_pcs[i][-1],service,
                    int(st.z["baseline"][j]==0),*(rec.out[p][j] for p in ("stride","multi","spp")),st.z["head"][j],
                    int(st.seen[st.line[j]//64]<=i),st.z["instr"][alt])
            rows.append(dict(zip(WITNESS,values)))
    return rows,dict(case=rec.name,horizon=task,conditional_rule=rule,ambiguous_local_keys=len([k for k,x in variants.items() if len(x)>=2]),
                     candidate_anchors=len(positives),candidate_strong_misses=len(strong))


def run_case(rec):
    summaries=[]; witnesses=[]; ambiguous=[]; lift=[]
    st=rec.st
    for task,q in rec.targets.items():
        tr=st.splits()["train"][1]
        fits={rule:freeze(counts(st,q,tr,rule,task)) for rule in RULES if rule not in ("next_line","repeat_delta")}
        for part in ("dev","test"):
            details={}
            for rule in RULES:
                row,good,correct,offers=evaluate(rec,task,part,rule,fits.get(rule),st.service())
                summaries.append(row); details[rule]=(good,correct,offers)
            if part=="dev":
                continue
            for rule in ("global_delta","previous_pc","ordered_pc3","unordered_pc3","relative_order"):
                prior="relative" if rule=="relative_order" else "local"
                good=details[rule][0]
                bg=details[prior][0]
                extra=good-bg
                survives={i for i in extra if all(rec.out[p][q[i]] in ("L","P") for p in ("spp","multi"))}
                lift.append(dict(case=rec.name,horizon=task,conditional_rule=rule,control=prior,
                                 added=len(extra),lost=len(bg-good),net=len(good)-len(bg),both=len(good&bg),
                                 added_on_both_strong_miss=len(survives),
                                 added_on_both_strong_miss_with_head=sum(int(st.z["head"][q[i]]>0) for i in survives)))
                if rule in ("previous_pc","ordered_pc3","global_delta"):
                    w,c=witness(rec,task,rule,fits["local"],fits[rule],good,details[rule][1])
                    witnesses+=w;ambiguous.append(c)
    return summaries,witnesses,ambiguous,lift


def cross_family(recs):
    out=[]
    families=sorted({r.family for r in recs})
    for fam in families:
        train=[r for r in recs if r.family!=fam]
        if not train:
            continue
        # Median of source-record calibration values weights each source once.
        service=int(np.ceil(np.median([r.st.service() for r in train])))
        for task in TASKS:
            for rule in ("offset","relative","relative_order"):
                merged=defaultdict(Counter)
                for r in train:
                    c=counts(r.st,r.targets[task],r.st.splits()["train"][1],rule,task)
                    for k,v in c.items():
                        merged[k].update(v)
                fit=freeze(merged)
                for r in recs:
                    if r.family==fam:
                        row,*_=evaluate(r,task,"test",rule,fit,service,"other_collections")
                        out.append(row)
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--available",action="store_true",help="omit discovery cases without all four completed history files")
    ap.add_argument("--history-root",type=Path,default=ROOT/"raw/screen/history")
    ap.add_argument("--output",type=Path,default=HERE)
    ap.add_argument("--case",nargs="+",help="subset of positive discovery IDs")
    args=ap.parse_args()
    with (ROOT/"intake/registry.csv").open() as f:
        selected=[r for r in csv.DictReader(f) if r["split"]=="discovery"]
    if args.case:
        illegal=set(args.case)-{r["case"] for r in selected}
        if illegal: raise ValueError("Case outside discovery registry: "+str(illegal))
        selected=[r for r in selected if r["case"] in args.case]
    recs=[];pending=[]
    for r in selected:
        files=[args.history_root/r["case"]/(p+".npz") for p in MODES]
        if all(f.is_file() for f in files):
            recs.append(Recording(r,args.history_root))
        else:
            pending.append(r["case"])
    if pending and not args.available:
        raise ValueError("Incomplete discovery screen: "+", ".join(pending))
    if not recs:
        raise ValueError("No completed discovery cases.")
    scores=[];examples=[];ambiguity=[];lift=[];pop=[];earliest=[]
    for rec in recs:
        s,w,a,l=run_case(rec)
        scores+=s;examples+=w;ambiguity+=a;lift+=l
        pop+=rec.contexts();earliest+=rec.earliest()
        print(rec.name,rec.st.n,"ingresses",len(w),"example roles",flush=True)
    scores.extend(cross_family(recs))
    args.output.mkdir(parents=True,exist_ok=True)
    write("scores.csv",scores,folder=args.output);write("recurring_contexts.csv",examples,WITNESS,args.output)
    write("ambiguity.csv",ambiguity,folder=args.output);write("added_targets.csv",lift,folder=args.output)
    write("sequence_populations.csv",pop,folder=args.output);write("candidate_reach.csv",earliest,folder=args.output)
    print("Executed:",", ".join(r.name for r in recs),"Pending:",", ".join(pending))


if __name__=="__main__":
    main()
