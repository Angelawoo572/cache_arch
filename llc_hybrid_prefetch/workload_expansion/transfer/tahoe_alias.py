"""Diagnose the exploratory transferred-count spike at unchanged physical addresses."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import subprocess
import numpy as np
from frozen_lbm import RAW, PARENT, prepare, LIMIT
from evaluate import HERE, STUDY, read_history, lookup, native_class, write


def family_shift(shift, z, maps, pc_pairs, buckets, engine):
    # XOR relabels the source sites bijectively; all line addresses/order/cycles remain fixed.
    rows = "".join(f"{int(pc)^shift} {int(line)} {int(t)}\n" for pc,line,t in zip(z["pc"],z["pa"],z["t"]))
    out = subprocess.run(engine,input=rows,text=True,capture_output=True,check=True).stdout.splitlines()
    counts=Counter()
    found=set()
    trace=Counter()
    offset_controls = {d:set() for d in (1,5,8,16,32)}
    for row in out:
        i, candidate, _, _=map(int,row.split())
        if i+64>=len(z["t"]):
            continue
        line=int(z["pa"][i]); pc=int(z["pc"][i])^shift
        counts["slots"]+=1
        counts["donor_full_pc_offset"]+=(pc,line%64) in pc_pairs
        bucket=(((pc%4096)^(line%64))*11400714819323198485)%(2**64)
        counts["donor_trained_bucket"]+=bucket in buckets
        choices=[(candidate,found)] + [(line+d,s) for d,s in offset_controls.items() if (line+d)//64==line//64]
        for target,s in choices:
            match=np.flatnonzero(z["pa"][i+1:i+65]==target)
            if not len(match):continue
            j=i+1+int(match[0]);lead=int(z["t"][j]-z["t"][i])
            if LIMIT<=lead<=4096 and z["baseline"][j]==0 and all(v in ("miss","late") for v in native_class(z,j,maps)):
                s.add(j)
                if s is found:
                    trace[pc,target-line]+=1
    top=trace.most_common(1)
    return dict(pc_xor=shift,slots=counts["slots"],real_donor_full_pc_offset=counts["donor_full_pc_offset"],
                supported_mixed_buckets=counts["donor_trained_bucket"],distinct_qualified_targets=len(found),
                most_common_receiver_pc_in_original_labels=(top[0][0][0]^shift if top else ""),
                most_common_displacement=(top[0][0][1] if top else ""),matching_grants_at_this_pc_displacement=(top[0][1] if top else 0),
                **{f"fixed_plus_{d}":len(s) for d,s in offset_controls.items()})


def main():
    engine,buckets=prepare()
    data=PARENT.parent/"opportunity_study/raw/history/lbm-r1-c0.npz"
    with np.load(data) as train:
        n=int(.6*len(train["pc"]))
        pairs={(int(pc),int(line)%64) for pc,line in zip(train["pc"][:n],train["pa"][:n])}
    root=STUDY/"raw/replication"
    z=read_history(root,"gg_tahoe","stride")
    maps=[lookup(read_history(root,"gg_tahoe",p)) for p in ("stride","multi","spp")]
    masks=[0]+[i*257 for i in range(1,16)]
    with ThreadPoolExecutor(4) as pool:
        results=list(pool.map(partial(family_shift,z=z,maps=maps,pc_pairs=pairs,buckets=buckets,engine=engine),masks))
    write(HERE/"tahoe_alias.csv",results)


if __name__=="__main__":
    main()
