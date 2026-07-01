#!/usr/bin/env python3
"""Summarize existing DEMAND_EVENT_LOG resource counters for normal and NN runs."""
from __future__ import annotations
import argparse, csv, gzip, json
from pathlib import Path

def op(path, mode="rt"):
    return gzip.open(str(path), mode, newline="") if str(path).endswith(".gz") else open(str(path), mode, newline="")
def f(v):
    try: return float(v) if str(v).strip() else 0.0
    except (TypeError, ValueError): return 0.0
def q(xs, frac):
    if not xs:return 0.0
    xs=sorted(xs); return xs[min(len(xs)-1, int(frac*(len(xs)-1)))]
def desc(xs, name):
    return {name+"mean":sum(xs)/len(xs) if xs else 0.0,name+"p50":q(xs,.5),name+"p95":q(xs,.95),name+"max":max(xs) if xs else 0.0}
def one(path, family, variant, trace):
    dp=[]; dm=[]; pp=[]; pm=[]; loads=attempts=accepted=dups=timely=late=0
    with op(path) as h:
        for r in csv.DictReader(h):
            if str(r.get("cache","")).upper() != "L2C":continue
            e=str(r.get("event","")).upper(); pq=f(r.get("pq_occ")); m=f(r.get("mshr_occ"))
            if e=="DEMAND":
                loads+=1; dp.append(pq); dm.append(m); timely+=int(f(r.get("hit"))>0 and f(r.get("was_prefetch"))>0); late+=int(f(r.get("late"))>0)
            elif e=="PF":
                attempts+=1; pp.append(pq); pm.append(m); accepted+=int(f(r.get("accepted"))>0); dups+=int(f(r.get("duplicate"))>0)
    out={"trace":trace,"family":family,"variant":variant,"event_file":str(path),"demand_l2_loads":loads,"prefetch_attempts":attempts,"prefetch_accepted":accepted,"prefetch_duplicate":dups,"timely_prefetch_demand":timely,"late_prefetch_demand":late,"prefetch_attempts_per_l2_load":attempts/loads if loads else 0.0,"prefetch_accepted_per_l2_load":accepted/loads if loads else 0.0,"prefetch_reject_fraction":1-accepted/attempts if attempts else 0.0}
    out.update(desc(dp,"demand_pq_occ_"));out.update(desc(dm,"demand_mshr_occ_"));out.update(desc(pp,"pf_pq_occ_"));out.update(desc(pm,"pf_mshr_occ_"));return out
def main():
    p=argparse.ArgumentParser();p.add_argument("--event-root",required=True,type=Path);p.add_argument("--out",required=True,type=Path);a=p.parse_args();rows=[]
    for path in sorted((a.event_root/"normal"/"events").glob("*.events.csv.gz")):
        s=path.name[:-len(".events.csv.gz")];t,v=s.rsplit(".",1);rows.append(one(path,"normal",v,t))
    root=a.event_root/"lstm"
    if root.is_dir():
        for d in sorted(root.iterdir()):
            if d.is_dir():
                for path in sorted((d/"events").glob("*.events.csv.gz")):rows.append(one(path,"standalone",d.name,path.name[:-len(".events.csv.gz")]))
    if not rows:raise RuntimeError("no event files")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    fields=sorted({k for r in rows for k in r})
    with a.out.open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(sorted(rows,key=lambda r:(r["trace"],r["family"],r["variant"])))
    a.out.with_suffix(".json").write_text(json.dumps({"event_root":str(a.event_root),"rows":len(rows)},indent=2)+"\n")
    print("[resource summary]",a.out)
if __name__=="__main__":main()
