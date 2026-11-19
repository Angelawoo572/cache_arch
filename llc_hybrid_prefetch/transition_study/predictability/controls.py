"""Matched current offset and storage-free controls for the frozen page table."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from predict import (HERE,case,group,guarded,measured_service,fit_all,frequencies,fit_keys,
                     outcomes,eligible_scores,score,writer,ACTIONS,WIDE)
from evaluate import grant


def keys(z,f):
    ix=np.arange(len(z['t']));off=z['pa']%64;pc=z['pc']%4096
    pg=f['previous_page'];ok=f['page_ok'];lasti=np.where(ok,np.maximum(0,pg),0)
    val=np.where(ok,off[lasti],64)
    other=np.maximum(0,np.where(ok,lasti-1,ix-1))
    # Controls keep the same exact current PC+offset and backoff. Older position
    # is chosen without looking at future labels or demanding another page.
    last_pc=np.r_[4096,pc[:-1]]
    tag=np.flatnonzero(z['tag_t']>=0)
    ordered=tag[np.argsort(z['tag_t'][tag],kind='stable')]
    cut=np.searchsorted(z['tag_t'][ordered],z['t'],side='left')-1
    # Native observed first-use and tag results arrive after lookup. No paired
    # NoPF demand outcome ever participates in this input.
    streak=np.zeros(len(ordered),int);run=0
    for j,i in enumerate(ordered):
        run=min(3,run+1) if z['hit'][i]==1 and z['pf_kind'][i]==1 else 0
        streak[j]=run
    feedback=np.where(cut>=0,streak[np.maximum(0,cut)],0) if len(tag) else np.zeros(len(ix),int)
    return {'completed_timely_streak':group(group(pc,off),feedback),'context_present':group(group(pc,off),ok.astype(int)),
            'last_global_pc':group(group(pc,off),last_pc),
            'exact':group(pc,off),'offset_only':off,'offset_pair':group(off,val),
            'unrelated_older':group(group(pc,off),off[other]),'page':f['keys']['page']}


def evaluate(item):
    a,b,view=item
    z,f,g=case(a,view);t,u,h=case(b,view)
    end=int(.6*len(z['t']));lo=int(.8*len(t['t'])) if a==b else 0
    tr=guarded(z,0,end);allids=guarded(t,lo,len(t['t']));service=measured_service(z,end)
    if not len(tr) or not len(allids): return [],[]
    _,valid,q=outcomes(z,g,tr,service,'1_64')
    y=np.where(valid,g['code'][tr],-1);prior=np.bincount(y[y>=0],minlength=ACTIONS)/len(tr)
    nz=np.where((y>=0)&(y<64))[0];dd=z['pa'][q[nz]]-z['pa'][tr[nz]]
    const=int(np.argmax(np.bincount(dd+63,minlength=127)))-63 if len(dd) else 0
    ids=allids
    ks=keys(z,f);kt=keys(t,u)
    base=fit_keys(f['keys']['low12'][tr],y)
    current=t['pa'][ids];pg=u['previous_page'][ids];delta=current-t['pa'][np.maximum(0,pg)]
    data=[];audit=[];outcomes_by_rule={}
    for rule in ('fixed_best_delta','repeat_last_page_delta','exact','offset_only','offset_pair','context_present','completed_timely_streak','last_global_pc','unrelated_older','page'):
        if rule in ('fixed_best_delta','repeat_last_page_delta'):
            d=const if rule=='fixed_best_delta' else delta
            target=current+d;ok=(target//64==current//64)&(target!=current)
            if rule=='repeat_last_page_delta':ok &= u['page_ok'][ids]
            target=np.where(ok,target,-1);sup=np.full(len(ids),-1);sc=np.ones(len(ids))
        else:
            tab=fit_keys(ks[rule][tr],y);dest=[];scores=[];counts=[]
            for start in range(0,len(ids),4096):
                ix=ids[start:start+4096]
                backup,_=frequencies(base,u['keys']['low12'][ix],prior)
                s,n=frequencies(tab,kt[rule][ix],backup)
                s=eligible_scores(s,t,ix,u['wide_legal'])
                code=s.argmax(1);v=np.where(code<64,(t['pa'][ix]//64)*64+code,t['pa'][ix]+WIDE[np.maximum(0,code-64)])
                dest.extend(v);scores.extend(s[np.arange(len(ix)),code]);counts.extend(n)
            target=np.array(dest);sc=np.array(scores);sup=np.array(counts)
        for mode,gate in [('all',np.ones(len(ids),bool)),('fixed_1in8',grant(ids))]:
            issue=gate&(target>=0)&(sc>0);meta=dict(train=a,test=b,split='within' if a==b else 'transfer',view=view)
            so,_=score(meta,t,u,h,ids,rule,mode,issue,target,sc,sup,service,'1_64',0)
            so['train_selected_constant']=const if rule=='fixed_best_delta' else ''
            data.append(so)
            if mode=='fixed_1in8':
                _,eligible,q=outcomes(t,h,ids,service,'1_64')
                hit=issue&eligible&(target==t['pa'][q])
                outcomes_by_rule[rule]=(target[gate].copy(),set(q[hit]),hit.copy())
    yt,yt_ids,anchor_hits=outcomes_by_rule['page']
    for rule,(choice,targets,hit) in outcomes_by_rule.items():
        audit.append(dict(train=a,test=b,view=view,comparison=rule,
                    fixed_grants=len(yt),equal_proposed_physical_lines=int(sum((yt==choice)&(yt>=0))),
                    equal_addresses_or_abstentions=int(sum(yt==choice)),
                    page_only_targets=len(yt_ids-targets),common_targets=len(yt_ids&targets),control_only_targets=len(targets-yt_ids),
                    same_grant_qualifying_matches=int((hit&anchor_hits).sum()),page_qualifying_grants=int(anchor_hits.sum()),
                    page_qualifying_with_observed_prior_page=int((anchor_hits&u['page_ok'][ids]).sum()),
                    page_qualifying_WITHOUT_prior_page=int((anchor_hits&~u['page_ok'][ids]).sum())))
    return data,audit

if __name__=='__main__':
    from predict import WITHIN,TRANSFER
    panels=[(k,k) for k in WITHIN]+TRANSFER+[(a,b) for a in ('gap_bfs3','gap_bfs3_late') for b in ('gap_bfs3_mid','gap_bfs8_mid')]
    with ProcessPoolExecutor(4) as ex:
        parts=list(ex.map(evaluate,[(a,b,v) for a,b in panels for v in ('ingress','ingress_pc')]))
    writer(HERE/'matched_controls.csv',[r for group,_ in parts for r in group])
    writer(HERE/'control_overlap.csv',[r for _,group in parts for r in group])
