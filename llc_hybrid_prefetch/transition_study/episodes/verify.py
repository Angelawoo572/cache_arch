"""Independently validate exact episodes against ordered stream indices and report exposure."""
import csv
import json
import numpy as np
import pandas as pd
from extract import CASES,DEST,HERE,PATHS,SUPPLEMENTAL_CASES,runpath
from report import VIEWS,labels

def code(k):return k[0].astype('int64')*256+k[1].astype('int64')
def location(d,prefix,ids):
    keys=code((d[prefix+'_instr'].to_numpy(),d[prefix+'_op'].to_numpy()))
    ix=np.searchsorted(ids,keys)
    if not np.array_equal(ids[ix],keys):raise ValueError('unknown ID')
    return ix
def main():
    episodes=pd.read_csv(HERE/'episode_ids.csv.gz')
    matches=pd.read_csv(HERE/'matched_ids.csv.gz')
    summary=[];paths=[];source=[];roster=[];peek={}
    for case in CASES:
        meta=json.loads((runpath(case,'stride')/'run.json').read_text())
        panel=('supplemental_LBM_R2' if case in SUPPLEMENTAL_CASES else 'fresh_BFS_midpoint' if case.endswith('_mid') else 'initial_archive')
        roster.append(dict(case=case,panel=panel,warmup=meta['warmup'],roi=meta['roi'],traces=' + '.join(meta['traces'])))
        for f in sorted((DEST/case).glob('c*.npz')):
            cpu=int(f.stem[1:])
            with np.load(f) as z:
                ids=z['ids'];out=z['outcome'];ing=z['ingress'];pc=z['pc'];pa=z['pa'];locs=np.arange(len(ids))
                packed=z['base_path'].astype(int)*16+z['path'];head=z['head']
            lab=labels(case,None)
            if case in ('gap_bfs8','gap_bfs8_lbm') and cpu==0:peek[case]=(ids,out)
            for kind,n in zip(*np.unique(packed,return_counts=True)):
                paths.append(dict(case=case,core=cpu,nopf_path=PATHS[kind//16],stride_path=PATHS[kind%16],demands=n))
            for view in VIEWS:
                ix=ing if view.startswith('ingress') else locs
                K=pc[ix] if view.endswith('_pc') else (pa[ix]//64 if view.endswith('_page') else pa[ix]//256 if view.endswith('_region4') else np.zeros(len(ix),int))
                order=np.argsort(K,kind='stable');seq=ix[order];K=K[order];y=out[seq];T=y==5
                inv=np.full(len(ids),-1,dtype='int32');inv[seq]=np.arange(len(seq))
                available=np.flatnonzero(T[:-2]&T[1:-1]&(K[:-2]==K[1:-1])&(K[1:-1]==K[2:]))+1
                fail=available[np.isin(y[available+1],[6,7])]
                d=episodes[(episodes.case==case)&(episodes.core==cpu)&(episodes.view==view)]
                m=matches[(matches.case==case)&(matches.core==cpu)&(matches.view==view)]
                saved=inv[location(d,'last',ids)];q=inv[location(d,'first_miss',ids)]
                assert np.array_equal(fail,saved) and np.array_equal(q,saved+1)
                first=inv[location(d,'start',ids)]
                assert np.array_equal(d.run_length.to_numpy(),1+saved-first)
                assert np.all((first==0)|(K[first]!=K[first-1])|~T[first-1])
                sums=np.r_[0,np.cumsum(T)]
                assert np.all(sums[saved+1]-sums[first]==1+saved-first)
                ok=d.recovery_censored==0
                rec=inv[location(d[ok],'recovery',ids)]
                assert np.all(T[rec]) and np.all(K[rec]==K[saved[ok]])
                assert np.all(sums[rec]-sums[q[ok]]==0)
                if len(m):
                    a=inv[location(m,'control_anchor',ids)];b=inv[location(m,'control_next',ids)]
                    assert len(np.unique(a))==len(a) and np.all(b==a+1) and np.all(T[a]&T[b])
                if lab and view=='ingress':
                    pops=CounterValues([lab.get(int(ids[j]),'unlabelled') for j in seq[np.isin(y,(6,7))]])
                    hits=CounterValues([lab.get(int(ids[j]),'unlabelled') for j in seq[q]])
                    for tag,val in pops.items():
                        source.append(dict(case=case,core=cpu,target_label=tag,paired_misses=val,breaks_after_streak=hits.get(tag,0)))
                ans=y[available+1]
                summary.append(dict(case=case,core=cpu,view=view,core_loads=len(ids),projection_events=len(seq),direct_timely=int(T.sum()),
                    streak_anchors=len(available),continued_T=int((ans==5).sum()),late=int((ans==6).sum()),persistent=int((ans==7).sum()),
                    all_paired_late=int(np.count_nonzero(y==6)),all_paired_persistent=int(np.count_nonzero(y==7)),
                    episode_operand_head_sum=int(head[seq[q]].sum()),all_paired_miss_operand_head_sum=int(head[seq[np.isin(y,(6,7))]].sum()),
                    episodes_with_positive_head=int(np.count_nonzero(head[seq[q]]>0)),other_next_event=int((~np.isin(ans,[5,6,7])).sum()),matched_boundaries=len(m),recovered=int(ok.sum()),
                    median_run_length=d.run_length.median() if len(d) else '',median_gap_to_recovery=d.recovery_gap.median() if ok.any() else '',
                    median_target_operand_head=d.target_head.median() if len(d) else ''))
    pd.DataFrame(roster).to_csv(HERE/'case_roster.csv',index=False)
    pd.DataFrame(source).to_csv(HERE/'synthetic_labels.csv',index=False)
    pd.DataFrame(paths).to_csv(HERE/'path_pairs.csv',index=False)
    pd.DataFrame(summary).to_csv(HERE/'summary.csv',index=False)
    aa,ac=peek['gap_bfs8'];ba,bc=peek['gap_bfs8_lbm']
    if not np.array_equal(aa,ba):raise ValueError('peer identities')
    subset=episodes[(episodes.case=='gap_bfs8_lbm')&(episodes.core==0)&(episodes.view=='ingress')]
    ix=location(subset,'first_miss',aa)
    print('mixed BFS8 boundary miss outcome codes in solo',CounterValues(ac[ix]),'mixed late alone timely',int(np.count_nonzero((bc[ix]==6)&(ac[ix]==5))))
    print(len(summary),'validated view populations;',len(episodes),'episodes and',len(matches),'controls')
def CounterValues(a):
    return {k:int(v) for k,v in zip(*np.unique(a,return_counts=True))}
if __name__=='__main__':main()
