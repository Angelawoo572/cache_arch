"""Score native shadow's three rules and equal-grant controls on the same frozen labels."""
import csv
import numpy as np
from prepare import OUT,HERE,openz,targets,signals

def main():
    z=openz('gap_bfs3_late-c0')
    with (OUT/'runs/shadow/decisions.csv').open() as f:shadow=list(csv.DictReader(f))
    cut=int(.6*len(z['t']))
    mask=(z['dispatch']>=0)&(z['fill']>=z['dispatch'])&(z['fill']<z['t'][cut])
    service=int(np.median((z['fill']-z['dispatch'])[mask]))
    _,nxt=signals(z)
    next_target=targets(z,nxt,service)
    cases=('next','repeat','current','page_offset_only','page_link')
    rows=[]
    for case in cases:
        generated=correct=persistent=timely=other_hits=0
        unique=set()
        for r in shadow:
            i=int(r['index']);line=int(r['base']);j=int(next_target[i])
            if case=='next': predicted=line+1
            elif case=='repeat':
                age=int(r['prior_age'])
                if age<0:continue
                predicted=2*line-int(z['pa'][i-age])
            else:predicted=int(r[case])
            if predicted==line or predicted//64!=line//64:continue
            generated+=1
            if j<0 or predicted!=int(z['pa'][j]):continue
            correct+=1; unique.add(j)
            if z['hit'][j]==0:persistent+=1
            elif z['pf_kind'][j]==1:timely+=1
            else:other_hits+=1
        assert correct==len(unique)
        rows.append(dict(rule=case,grants=len(shadow),choices=generated,matched_next_qualifying_offset=correct,
                         matched_original_stride_miss=persistent,matched_original_stride_direct_timely=timely,
                         matched_original_stride_other_hit=other_hits,service=service))
    with (HERE/'offline.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
