"""Frozen frequency tables, identical causally scheduled degree-one grants and held-out tests."""
import argparse
import csv
import gzip
import json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from stream import build,contexts,MAX_EVENTS,MAX_CYCLES,LABELS,B1,B2,B3
from extract import CACHE

HERE=Path(__file__).resolve().parent
EVAL=CACHE/'evaluation'
ELIGIBILITY=('same_page_any','same_page_14','extended_seen_page','wide_24','recent_page_any','recent_page_16','successor_3','stride_extrap_2','all_enumerated','all_enumerated_grant','actual_stride_proposed')
EMPIRICAL=('recent','pc_low12','recent_with_outcome_stride','transition_flags','skip_pc_association','pc_local_w16_ordered','cross_pc_w16_ordered','cross_pc_older_jump')
GRANT=8
SAMPLE=48


def grant(ids, size=8, seed=447):
    # Choose a data-independent slot per consecutive block; fixed-phase
    # sampling aliases the exact period of several generated controls.
    ids=np.asarray(ids)
    x=(ids//size).astype('uint64')+np.uint64(seed)
    with np.errstate(over='ignore'):
        x=(x^(x>>30))*np.uint64(0xbf58476d1ce4e5b9)
        x=(x^(x>>27))*np.uint64(0x94d049bb133111eb)
    return ids%size==((x^(x>>31))%size)


def source_labels(z,test):
    path=HERE.parent/'raw'/'traces'/(str(z['trace'])+'.labels.csv.gz')
    if not path.exists():return None
    byid={}
    with gzip.open(path,'rt') as f:
        for r in csv.DictReader(f):
            if r['measurement']=='roi':
                byid[(int(r['instr_id']),int(r['operand']))]=r['label']
    return np.array([byid.get((int(i),int(o)),'unlabelled') for i,o in zip(z['instr'],z['op'])])



def openz(name):
    with np.load(CACHE/(name+'.npz')) as f:
        return {k:f[k] for k in f.files}


def guarded(z,lo,hi):
    ids=np.arange(lo,hi,dtype=int)
    if not len(ids):return ids
    # Discard anchors whose complete label horizon might straddle the split.
    return ids[(ids+MAX_EVENTS<hi)&(z['t'][ids]+MAX_CYCLES<=z['t'][hi-1])]


def table(keys,labels):
    unique,index,count=np.unique(keys,return_inverse=True,return_counts=True)
    yes=np.array([np.bincount(index,weights=a,minlength=len(unique)) for a in labels.T],dtype='float32').T
    return unique,count,yes


def rates(tab,keys,prior):
    u,n,yes=tab
    pos=np.minimum(np.searchsorted(u,keys),len(u)-1)
    seen=u[pos]==keys
    support=np.where(seen,n[pos],0)
    valid=support>=8
    result=np.where(valid[:,None],(yes[pos]+8*prior)/(support[:,None]+8),prior)
    return result,support,yes[pos]*seen[:,None]


def positive(z,b,lead):
    follows=b['nxt']; safe=np.maximum(follows,0)
    return ((follows>=0)&(z['baseline'][safe]==0)&(z['hit'][safe]>=0)&
            (z['t'][safe]-z['t'][:,None]>=lead))


def Wilson_lower(yes,n):
    n=np.maximum(n,1);q=yes/n
    return (q+1.96**2/(2*n)-1.96*np.sqrt(q*(1-q)/n+1.96**2/(4*n*n)))/(1+1.96**2/n)


def choices(scores,b,ids):
    legal=b['cand'][ids]>=0
    pick=np.argmax(np.where(legal,scores,-3),axis=1)
    if not np.all(legal[np.arange(len(ids)),pick]): raise ValueError('no legal displacement')
    return pick


def assess_slice(z,b,name,mode,ids,allids,lo,hi,support,raw_yes,scores,base_misses,seen_pc,threshold,label,lead):
    index=np.arange(len(ids)); pick=choices(scores,b,ids)
    reached=b['nxt'][ids,pick]
    caught=label[ids,pick]
    actual=reached[caught]
    well=(support>=8)&(raw_yes[index,pick]>=3)&(Wilson_lower(raw_yes[index,pick],support)>=0.10)
    valid=well&caught
    valid_tally=[]
    for part in np.array_split(np.arange(len(ids)),3):
        eligible=well[part]
        valid_tally.append((int(eligible.sum()),int(valid[part].sum())))
    counts=Counter(int(k) for k in actual)
    distinct=np.array(sorted(counts.keys()),dtype=int)
    persistent=int((z['hit'][distinct]==0).sum())
    return (dict(model=name,split=mode,lead_cycles=lead,eligible_ingresses=len(allids),grants=len(ids),
              supported_grants=int((support>=8).sum()),unseen_pc_grants=int((~np.isin(z['pc'][ids],seen_pc)).sum()),
              within_horizon_future_touches=int((reached>=0).sum()),timely_original_miss_emissions=int(caught.sum()),
              distinct_original_misses=len(distinct),distinct_persistent_misses=persistent,original_miss_denominator=base_misses,
              train_supported_grants=int(well.sum()),train_supported_matches=int(valid.sum()),
              qualified_dev=threshold,qualified_distinct_misses=int(len(np.unique(reached[valid]))) if threshold else 0,
              early_tercile_hits=int(np.sum(caught&(ids<lo+(hi-lo)/3))),
              middle_tercile_hits=int(np.sum(caught&(ids>=lo+(hi-lo)/3)&(ids<lo+2*(hi-lo)/3))),
              late_tercile_hits=int(np.sum(caught&(ids>=lo+2*(hi-lo)/3)))),
              valid_tally,(ids[valid],reached[valid]))


def sample_targets(z,lo,hi):
    ix=np.arange(lo,hi); orig=ix[(z['baseline'][ix]==0)&(z['hit'][ix]>=0)]
    strata={'persistent_head':(z['hit'][orig]==0)&(z['head'][orig]>0),
            'persistent_other':(z['hit'][orig]==0)&(z['head'][orig]==0),
            'direct_timely':(z['hit'][orig]==1)&(z['pf_kind'][orig]==1),
            'other_hit':(z['hit'][orig]==1)&(z['pf_kind'][orig]!=1)}
    rng=np.random.default_rng(447)
    return {k:(len(orig[v]),np.sort(rng.choice(orig[v],min(SAMPLE,int(v.sum())),replace=False))) for k,v in strata.items()}


def writer(path,rows):
    if not rows:
        path.unlink(missing_ok=True)
        return
    with path.open('w',newline='') as f:
        out=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');out.writeheader();out.writerows(rows)


def ceilings(z,b,lo,hi,split,service,scope,empirical):
    samples=sample_targets(z,lo,hi)
    lookup=defaultdict(list)
    scoreable=np.zeros(len(z['t']),bool);scoreable[guarded(z,lo,hi)]=True
    byline=defaultdict(list)
    for j,line in enumerate(z['pa']):
        byline[int(line)].append(j)
    import bisect
    rows=[]; example=[]
    for stratum,(N,targets) in samples.items():
        details=defaultdict(list)
        for q in targets:
            q=int(q);line=int(z['pa'][q]);page=line//64;t=int(z['t'][q]);prior=byline[line]
            o=bisect.bisect_left(prior,q)
            # Stop before any intervening demand that would consume the line.
            start=max(lo,q-MAX_EVENTS,np.searchsorted(z['t'],t-MAX_CYCLES),prior[o-1]+1 if o else lo)
            ix=np.arange(start,q,dtype=int);avail=ix[z['t'][ix]<t]
            family={}
            if len(avail):
                current=z['pa'][avail];known=b['first_page'].get(page,10**18)<=avail
                candidates=b['cand'][avail]==line
                tests=((current//64==page)&(current!=line),
                       candidates[:,:B1].any(1),(current//64!=page)&(abs(current-line)<=1024)&known,
                       candidates[:,B1:B2].any(1),(b['older'][avail]==page).any(1),
                       candidates[:,B2:B3].any(1),candidates[:,B3:B3+3].any(1),
                       candidates[:,B3+3:].any(1),candidates.any(1),candidates.any(1)&grant(avail)&scoreable[avail],
                       z['proposed'][avail]==line)
                for name,ok in zip(ELIGIBILITY,tests):
                    hits=avail[ok];family[name]=int(hits[0]) if len(hits) else -1
            for name in EMPIRICAL:
                eligible=empirical.get(name,{}).get(q,[])
                family['supported_'+name]=min((j for j in eligible if j>=start),default=-1)
            stride=family.get('actual_stride_proposed',-1)
            for name in ELIGIBILITY+tuple('supported_'+x for x in EMPIRICAL):
                j=family.get(name,-1)
                lead=t-int(z['t'][j]) if j>=0 else -1
                details[name].append((q,j,lead))
                nrow=dict(case=scope,split=split,stratum=stratum,method=name,
                    target_instr=int(z['instr'][q]),target_operand=int(z['op'][q]),target_pc=int(z['pc'][q]),target_physical_line=line,
                    target_ingress=t,target_tag=int(z['tag_t'][q]),stride_tag_hit=int(z['hit'][q]),
                    head_operand_cycles=int(z['head'][q]),cue_instr=int(z['instr'][j]) if j>=0 else '',
                    cue_operand=int(z['op'][j]) if j>=0 else '',cue_pc=int(z['pc'][j]) if j>=0 else '',
                    cue_physical_line=int(z['pa'][j]) if j>=0 else '',cue_cycle=int(z['t'][j]) if j>=0 else '',
                    lead_cycles=lead if j>=0 else '',lead_events=q-j if j>=0 else '',service_proxy=service,
                    observed_own_dispatch_fill=int(z['fill'][q]-z['dispatch'][q]) if z['dispatch'][q]>=0 and z['fill'][q]>=z['dispatch'][q] else '',
                    actual_stride_inwindow_lead=t-int(z['t'][stride]) if stride>=0 else '',
                    actual_prefetch_generation=int(z['pf_gen'][q]) if z['pf_gen'][q]>=0 else '',
                    actual_prefetch_fill=int(z['pf_fill'][q]) if z['pf_fill'][q]>=0 else '')
                rows.append(nrow)
                if j>=0 and lead>=service and (name=='all_enumerated' or name.startswith('supported_')):
                    example.append(nrow)
        for name,entries in details.items():
            leads=[v[2] for v in entries if v[1]>=0];events=[v[0]-v[1] for v in entries if v[1]>=0]
            lookup[stratum].append(dict(case=scope,split=split,stratum=stratum,population=N,sample=len(targets),method=name,
                reached=len(leads),at_least_service=int(sum(x>=service for x in leads)),service_cycles=service,
                median_earliest_cycles=float(np.median(leads)) if leads else '',
                median_earliest_events=float(np.median(events)) if events else '',
                p90_earliest_cycles=float(np.percentile(leads,90)) if leads else ''))
    return rows,[r for l in lookup.values() for r in l],example


def run(train,test,kind):
    EVAL.mkdir(exist_ok=True,parents=True)
    scope=train+'__'+test+'__'+kind
    out=EVAL/scope;out.mkdir(exist_ok=True)
    z=openz(train);b=build(z); n=len(z['t'])
    other=z if train==test else openz(test); otherb=b if train==test else build(other)
    synthetic=not train.startswith(('mcf','cactus','xalan','lbm','gap','compiled'))
    end=int((.40 if synthetic else .60)*n);devend=int((.50 if synthetic else .80)*n)
    training=guarded(z,0,end);idx=training[grant(training,2,131)]   # Frequency training on deterministic alternating eligible ingresses.
    serv=(z['fill']-z['dispatch'])[(z['dispatch']>=0)&(z['fill']>=z['dispatch'])&(z['fill']<z['t'][end])]
    if not len(serv):
        # Explicit fixed diagnostic when primary training demand samples are unavailable.
        lead=200
    else:lead=max(1,int(np.median(serv)))
    train_labels=positive(z,b,lead)
    test_labels=train_labels if test==train else positive(other,otherb,lead)
    pc_seen=np.unique(z['pc'][idx]);prior=train_labels[idx].mean(axis=0)
    lo=devend if kind=='within' else 0
    dest=other;db=otherb;hi=len(other['t'])
    dev_all=guarded(z,end,devend);test_all=guarded(dest,lo,hi)
    did=dev_all[grant(dev_all)];tid=test_all[grant(test_all)]
    if not len(idx) or not len(did) or not len(tid):
        print('insufficient eligible ingresses',scope,flush=True);return
    basegt=int(np.sum((other['baseline'][lo:hi]==0)&(other['hit'][lo:hi]>=0)))
    devgt=int(np.sum((z['baseline'][end:devend]==0)&(z['hit'][end:devend]>=0)))
    records=[];empirical={}; verified=[]; classes=[]; sources=source_labels(other,test)
    ctx=contexts(z,b);otherctx=contexts(other,otherb)
    for (name,key),(oname,okey) in zip(ctx,otherctx):
        assert name==oname
        tab=table(key[idx],train_labels[idx])
        dr,ds,dy=rates(tab,key[did],prior if name in ('recent','physical_offset_delta','global_frequency','pc_low12') else base_dr)
        tr,ts,ty=rates(tab,okey[tid],prior if name in ('recent','physical_offset_delta','global_frequency','pc_low12') else base_tr)
        if name=='recent':
            base_dr,base_tr=dr,tr
            hold=test_all[test_all%GRANT==0]
            sr,sc,sy=rates(tab,okey[hold],prior)
            rr,_,_=assess_slice(other,otherb,'recent_fixed_grant_phase0',kind,hold,test_all,lo,hi,sc,sy,sr,basegt,pc_seen,False,test_labels,lead)
            rr.update(train=train,test=test,training_anchors=len(idx),training_service_samples=len(serv));records.append(rr)
            for cap,labelname in ((B1,'same14'),(B2,'plus_wide'),(B3,'plus_seen_pages')):
                # All masks retain legal same-page candidates at every grant.
                for split,zz,bb,ii,eligible,l,h,su,yy,scores,ng,labels0 in (
                        ('dev',z,b,did,dev_all,end,devend,ds,dy,dr,devgt,train_labels),
                        (kind,other,otherb,tid,test_all,lo,hi,ts,ty,tr,basegt,test_labels)):
                    restricted=scores.copy(); restricted[:,cap:]=-2
                    rr,_,_=assess_slice(zz,bb,'recent_'+labelname,split,ii,eligible,l,h,su,yy,restricted,ng,pc_seen,False,labels0,lead)
                    rr.update(train=train,test=test,training_anchors=len(idx),training_service_samples=len(serv))
                    records.append(rr)
        dv,tally,_=assess_slice(z,b,name,'dev',did,dev_all,end,devend,ds,dy,dr,devgt,pc_seen,False,train_labels,lead)
        # Confirmation must recur in two disjoint development thirds, with
        # at least eight predictions and a 10% observed useful-address rate.
        qualify=sum(n>=8 and hits/n>=0.10 for n,hits in tally)>=2
        te,_,(anchors,targets)=assess_slice(other,otherb,name,kind,tid,test_all,lo,hi,ts,ty,tr,basegt,pc_seen,qualify,test_labels,lead)
        for r in (dv,te):
            r.update(train=train,test=test,training_anchors=len(idx),training_service_samples=len(serv))
            records.append(r)
        if name in EMPIRICAL and qualify:
            mp=defaultdict(list)
            for a,q in zip(anchors,targets):mp[int(q)].append(int(a))
            empirical[name]=mp
        # Source labels are attached after all predictions have been made.
        chosen=choices(tr,otherb,tid); loc=np.arange(len(tid))
        true=test_labels[tid,chosen]; follows=otherb['nxt'][tid,chosen]
        family=np.where(chosen<B1,'same',np.where(chosen<B2,'wide',np.where(chosen<B3,'seen_page','successor_or_stride')))
        for family_name in ('same','wide','seen_page','successor_or_stride'):
            te['hits_'+family_name]=len(np.unique(follows[true&(family==family_name)]))
        if name in ('recent','cross_pc_w16_ordered','cross_pc_w16_shuffled','pc_local_w16_ordered','skip_pc_association','transition_flags') and sources is not None:
            q=np.arange(lo,hi);q=q[(other['baseline'][q]==0)&(other['hit'][q]>=0)]
            for source in sorted(np.unique(sources[q])):
                matched=np.unique(follows[true]); matched=matched[sources[matched]==source]
                classes.append(dict(train=train,test=test,split=kind,model=name,source_label=source,
                                    original_misses=int((sources[q]==source).sum()),distinct_misses_reached=len(matched)))
        if name in ('recent','recent_with_outcome_stride','pc_low12','transition_flags','cross_pc_w16_ordered') and qualify:
            good=true&(ts>=8)&(ty[loc,chosen]>=3)&(Wilson_lower(ty[loc,chosen],ts)>=.10)
            seen=set()
            for i in np.flatnonzero(good):
                j,q=int(tid[i]),int(follows[i])
                comp=('same_page' if other['pa'][j]//64==other['pa'][q]//64 else 'cross_page',
                      'persistent_head' if other['hit'][q]==0 and other['head'][q]>0 else 'other')
                if comp in seen:continue
                seen.add(comp)
                prior_idx=np.arange(max(lo,q-MAX_EVENTS,0),q)
                previous_touch=prior_idx[other['pa'][prior_idx]==other['pa'][q]]
                if len(previous_touch):prior_idx=prior_idx[prior_idx>previous_touch[-1]]
                stride=prior_idx[(other['proposed'][prior_idx]==other['pa'][q])&(other['t'][q]-other['t'][prior_idx]<=MAX_CYCLES)]
                verified.append(dict(train=train,test=test,split=kind,model=name,context=comp[0],target_cohort=comp[1],
                    target_instr=int(other['instr'][q]),target_operand=int(other['op'][q]),target_pc=int(other['pc'][q]),target_line=int(other['pa'][q]),
                    cue_instr=int(other['instr'][j]),cue_operand=int(other['op'][j]),cue_pc=int(other['pc'][j]),cue_line=int(other['pa'][j]),
                    cue_cycle=int(other['t'][j]),target_cycle=int(other['t'][q]),lead_cycles=int(other['t'][q]-other['t'][j]),lead_events=q-j,
                    candidate=LABELS[chosen[i]],training_context_count=int(ts[i]),training_positive_count=int(ty[i,chosen[i]]),
                    dev_supported_n=sum(x for x,y in tally),dev_supported_positive=sum(y for x,y in tally),service_proxy=lead,
                    earliest_stride_inwindow_lead=int(other['t'][q]-other['t'][stride[0]]) if len(stride) else '',
                    actual_pf_gen=int(other['pf_gen'][q]) if other['pf_gen'][q]>=0 else '',
                    head_cycles=int(other['head'][q])))
    # Common schema for calibrated, direct and budget-sensitivity rows.
    for r in records:
        for family in ('same','wide','seen_page','successor_or_stride'):
            r.setdefault('hits_'+family,'')
    writer(out/'verified.csv',verified);writer(out/'classes.csv',classes)
    detail,summaries,ex=ceilings(other,otherb,lo,hi,kind,lead,scope,empirical)
    writer(out/'models.csv',records); writer(out/'ceiling.csv',summaries)
    if detail:
        with gzip.open(out/'sampled_misses.csv.gz','wt',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(detail[0]));wr.writeheader();wr.writerows(detail)
    # Retain representative supported examples before filling out legal-only ones.
    pick=[];seen=set()
    for r in sorted(ex,key=lambda r:(not r['method'].startswith('supported_'),r['target_instr'],r['method'])):
        cat=(r['stratum'],r['method'])
        if cat not in seen:
            seen.add(cat);pick.append(r)
    writer(out/'examples.csv',pick)
    (out/'complete').write_text('done\n')
    print(scope,len(idx),len(tid),lead,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('train');ap.add_argument('test');ap.add_argument('kind',choices=['within','blind'])
    a=ap.parse_args();run(a.train,a.test,a.kind)
