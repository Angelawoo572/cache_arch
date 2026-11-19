"""Sensitivity: can an indexed earlier cue identify the next useful offset on this page?"""
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import csv
import numpy as np
from evaluate import HERE,openz,source_labels,guarded,grant,table,rates,writer
from stream import mix,MAX_CYCLES,MAX_EVENTS
from run import jobs


def signals(z):
    n=len(z['pc']);i=np.arange(n);pg=z['pa']//64;o=z['pa']%64;p=z['pc']%4096
    prev=np.full(n,-1,'int32');nxt=prev.copy();last={}
    for j,page in enumerate(pg):
        k=last.get(page,-1)
        if k>=0:
            nxt[k]=j
            if j-k<=MAX_EVENTS and z['t'][j]-z['t'][k]<=MAX_CYCLES:prev[j]=k
        last[page]=j
    # None of these features uses the future links. Current offset and
    # low PC bits can generalize across a page-aligned executable displacement.
    curr=mix(p,o)
    recent2=mix(mix(curr,np.r_[4096,p[:-1]]),np.r_[4096,4096,p[:-2]])
    prior_o=np.where(prev>=0,o[np.maximum(prev,0)],64)
    prior_p=np.where(prev>=0,p[np.maximum(prev,0)],4096)
    linked=mix(mix(curr,prior_o),prior_p)
    # A matched-depth alternate reads one earlier but unrelated event.
    other=np.maximum(0,np.where(prev>=0,prev-1,i-1))
    alternate=mix(mix(curr,o[other]),p[other])
    return dict(current=curr,last_two_pcs=recent2,page_offset_only=mix(curr,prior_o),page_link=linked,
                alternative_older_event=alternate),nxt


def targets(z,nxt,service):
    ii=np.arange(len(nxt));q=np.maximum(nxt,0)
    return np.where((nxt>ii)&(nxt-ii<=MAX_EVENTS)&(z['t'][q]-z['t']>=service)&(z['t'][q]-z['t']<=MAX_CYCLES)
              &(z['pa'][q]!=z['pa'])&(z['baseline'][q]==0)&(z['hit'][q]>=0),nxt,-1)


def evaluate(item):
    train,test,split=item
    z=openz(train);t=z if train==test else openz(test)
    src=source_labels(t,test);prefix=train.startswith(('mcf','lbm','xalan','cactus','gap','compiled'))
    n=len(z['t']);end=int((.6 if prefix else .4)*n);dev=int((.8 if prefix else .5)*n)
    observed=(z['fill']-z['dispatch'])[(z['dispatch']>=0)&(z['fill']>=z['dispatch'])&(z['fill']<z['t'][end])]
    service=int(np.median(observed)) if len(observed) else 200
    alltr=guarded(z,0,end);tr=alltr[grant(alltr,2,131)]
    lo=dev if split=='within' else 0;ids=guarded(t,lo,len(t['pc']));g=ids[grant(ids)]
    sk,sn=signals(z);tk,tn=signals(t)
    target=targets(t,tn,service);tt=targets(z,sn,service)
    off=(z['pa'][np.maximum(tt[tr],0)]%64)
    lbl=((tt[tr]>=0)[:,None]&(np.arange(64)==off[:,None]))
    prior=lbl.mean(0); ref=target[g];dest=np.maximum(ref,0);valid=ref>=0;results=[]; examples=[]
    for name in sk:
        tab=table(sk[name][tr],lbl)
        rates_,support,_=rates(tab,tk[name][g],prior if name=='current' else base)
        if name=='current':base=rates_
        allowed=(t['pa'][g]%64)[:,None]!=np.arange(64)
        pick=np.where(allowed,rates_,-1).argmax(1)
        matched=valid&(t['pa'][dest]%64==pick)
        for label in (['all']+list(sorted(np.unique(src[dest[valid]]))) if src is not None else ['all']):
            subgroup=valid if label=='all' else valid&(src[dest]==label)
            hit=np.flatnonzero(matched&subgroup)
            if not subgroup.any() and label!='all':continue
            results.append(dict(train=train,test=test,split=split,rule=name,target_label=label,
                                budget_grants=len(g),eligible_ground_truth_given_grant=int(subgroup.sum()),
                                matched=len(hit),supported_matches=int((support[hit]>=8).sum()),service=service))
            if len(hit) and label in ('joint_target','choice_target','subrow_start','band_start','row_start'):
                j=hit[0];a,q=g[j],dest[j]
                examples.append(dict(train=train,test=test,rule=name,target_label=label,
                    cue_instr=int(t['instr'][a]),cue_pc=int(t['pc'][a]),cue_line=int(t['pa'][a]),
                    target_instr=int(t['instr'][q]),target_pc=int(t['pc'][q]),target_line=int(t['pa'][q]),
                    target_offset=int(pick[j]),lead_cycles=int(t['t'][q]-t['t'][a]),lead_events=int(q-a),
                    train_context_count=int(support[j]),head_cycles=int(t['head'][q])))
    return results,examples


if __name__=='__main__':
    picks=['xor_','crosspc_','row_period','row_hierarchy','motif_noise','mcf','cactus','lbm','gap','xalan']
    matrix=[j for j in jobs() if any(j[1].startswith(v) for v in picks)]
    with ProcessPoolExecutor(4) as pool:
        data=list(pool.map(evaluate,matrix))
    writer(HERE/'page_association.csv',[r for sheet,_ in data for r in sheet])
    writer(HERE/'page_association_examples.csv',[r for _,examples in data for r in examples])
