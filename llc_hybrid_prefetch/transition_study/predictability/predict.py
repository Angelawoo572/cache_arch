"""Frozen first-boundary address forecasts from every causal LLC demand ingress."""
import argparse
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1]/'opportunity_study/history'
sys.path.insert(0,str(OLD))
from evaluate import openz as archive, guarded, writer
from stream import mix

WIDE = np.array([-512,-256,-128,-64,-32,-16,16,32,64,128,256,512])
ACTIONS = 64+len(WIDE)
BANDS = {'1_4':(1,4),'5_16':(5,16),'17_64':(17,64),'1_64':(1,64)}
RULES = ('next','repeat_pc','look2_pc','current','local','low12','short_local','short_global','page')
WITHIN = ['mcf-r1','mcf-r2','mcf994_adjacent','gap_bfs3_late','gap_bfs8_late','lbm-r1','lbm-r2',
          'row_period2_paced','row_period5_paced','crosspc_cue_paced','crosspc_mask_paced']
TRANSFER = [('mcf-r1','mcf-r2'),('mcf-r2','mcf994_adjacent'),('gap_bfs3_late','gap_bfs8_late'),
            ('lbm-r1','lbm-r2'),('row_period2_paced','row_period5_paced'),('crosspc_cue_paced','crosspc_mask_paced')]
MODELS = {'current':None,'local':'current','low12':None,'short_local':'local','short_global':'low12','page':'low12'}


def delta_class(x):
    mag=np.abs(x)
    return np.sign(x)*np.where(mag<=4,mag,np.where(mag<=16,5,np.where(mag<64,6,7)))+7


def openz(name):
    later=HERE.parent/'raw/history'/(name+'.npz')
    if later.exists():
        with np.load(later) as f:
            return {k:f[k] for k in f.files}
    return archive(name)


def fixed(name):
    return name in ('next','repeat_pc','look2_pc')


def group(a,b):
    return mix(np.asarray(a,dtype=np.uint64),np.asarray(b,dtype=np.uint64))


def features(z):
    line=z['pa']; pc=z['pc']; tick=z['t']; n=len(line)
    prev_pc={};prev_page={};prev_line={};seen={}
    lp=np.full(n,-1,int); pg=np.full(n,-1,int); prev=np.full(n,-1,int)
    for i,(physical,ip) in enumerate(zip(line,pc)):
        a=int(physical); p=a//64
        lp[i]=prev_pc.get(ip,-1); prev_pc[ip]=i
        pg[i]=prev_page.get(p,-1);prev_page[p]=i
        prev[i]=prev_line.get(a,-1);prev_line[a]=i
        seen.setdefault(p,i)
    local=np.where(lp>=0,line-line[np.maximum(lp,0)],0); lt=delta_class(local)
    gt=delta_class(line-np.r_[line[:1],line[:-1]])
    page_ok=(pg>=0)&(np.arange(n)-pg<=64)&(tick-tick[np.maximum(pg,0)]<=4096)
    last_offset=np.where(page_ok,line[np.maximum(pg,0)]%64,64)
    base=group(pc,(line%64)//8)
    masked=group(pc%4096,(line%64)//8)
    current=group(base,lt); norm=group(masked,lt)
    keys={'current':base,'local':current,'low12':norm,
          'short_local':group(current,np.where(lp>=0,lt[np.maximum(lp,0)],15)),
          'short_global':group(group(norm,np.r_[4096,(pc%4096)[:-1]]),gt),
          'page':group(group(pc%4096,line%64),last_offset)}
    # Normalized output keys never hash a future virtual or physical address.
    wide_legal=np.zeros((n,len(WIDE)),bool)
    ix=np.arange(n)
    for j,d in enumerate(WIDE):
        pp=(line+d)//64
        wide_legal[:,j]=(pp!=line//64)&(np.fromiter((seen.get(int(k),n) for k in pp),int,count=n)<=ix)
    return dict(keys=keys,previous_line=prev,local=local,known=seen,previous_page=pg,page_ok=page_ok,wide_legal=wide_legal)


def cohort(z,projection):
    n=len(z['t']); end=np.zeros(n,bool); last=np.full(n,-1,int); stable=np.zeros(n,bool)
    timely=(z['baseline']==0)&(z['hit']==1)&(z['pf_kind']==1)
    failure=(z['baseline']==0)&(z['hit']==0)
    streak={};prior={};start={};first=np.full(n,-1,int)
    for i,ip in enumerate(z['pc']):
        k=int(ip) if projection=='ingress_pc' else 0
        count=streak.get(k,0)
        if count>=2 and failure[i]:
            end[i]=True;last[i]=prior[k];first[i]=start[k]
        if count>=2 and timely[i]:
            stable[prior[k]]=True
        if timely[i] and not count:start[k]=i
        streak[k]=count+1 if timely[i] else 0
        prior[k]=i
    return end,last,stable,first


def next_event(z,end,projection):
    n=len(end); nxt={}; q=np.full(n,-1,int)
    for i in range(n-1,-1,-1):
        k=int(z['pc'][i]) if projection=='ingress_pc' else 0
        q[i]=nxt.get(k,-1)
        if end[i]:nxt[k]=i
    return q


def action(line,target,known,idx):
    n=len(line); out=np.full(n,-1,int)
    same=(line//64==target//64)&(line!=target)
    out[same]=target[same]%64
    delta=target-line
    # Positive unknown-page times are target metadata for retrospective legal reach only.
    first=np.fromiter((known.get(int(pg),n) for pg in target//64),dtype=np.int64,count=n)
    allowed=(~same)&(target//64!=line//64)&(first<=idx)
    for j,v in enumerate(WIDE):out[allowed&(delta==v)] = 64+j
    return out


def earliest_native(z,prev):
    n=len(z['t']);history={};earliest=np.full(n,-1,int)
    from bisect import bisect_right
    for i,(line,prop) in enumerate(zip(z['pa'],z['proposed'])):
        cand=history.get(int(line),[])
        p=bisect_right(cand,prev[i])
        if p<len(cand):earliest[i]=cand[p]
        if prop>=0:history.setdefault(int(prop),[]).append(i)
    return earliest


@lru_cache(maxsize=4)
def case(name,projection):
    z=openz(name+'-c0');f=features(z);e,last,stable,first_T=cohort(z,projection)
    ix=np.arange(len(e));q=next_event(z,e,projection);v=np.maximum(q,0)
    codes=action(z['pa'],z['pa'][v],f['known'],ix)
    return z,f,dict(end=e,last=last,stable=stable,first_T=first_T,q=q,nxt=v,code=codes,
                    first=(q>=0)&(f['previous_line'][v]<=ix),native=earliest_native(z,f['previous_line']))


def measured_service(z,cut):
    cutoff=z['t'][cut]
    obs=(z['dispatch']>=0)&(z['fill']>=z['dispatch'])&(z['fill']<cutoff)
    a=z['fill'][obs]-z['dispatch'][obs]
    return int(np.median(a)) if len(a) else 200


def outcomes(z,g,ids,service,band):
    a,b=BANDS[band];q=g['nxt'][ids];age=g['q'][ids]-ids;dt=z['t'][q]-z['t'][ids]
    in_window=(age>=a)&(age<=b)&(dt<=4096)&g['first'][ids]
    return in_window,(in_window&(dt>=service)),q


def fit_keys(keys,y):
    u,inv,n=np.unique(keys,return_inverse=True,return_counts=True)
    mask=y>=0
    cnt=np.bincount(inv[mask]*ACTIONS+y[mask],minlength=len(u)*ACTIONS).reshape(len(u),ACTIONS).astype('int32')
    return u,n,cnt

def frequencies(fit,keys,prior):
    u,n,cts=fit; pos=np.minimum(np.searchsorted(u,keys),len(u)-1)
    support=np.where(u[pos]==keys,n[pos],0)
    sc=np.broadcast_to(prior,(len(keys),ACTIONS)).copy()
    use=support>=8
    pri=prior[use] if prior.ndim==2 else prior
    sc[use]=(cts[pos[use]]+8*pri)/(support[use,None]+8)
    return sc,support


def eligible_scores(scores,z,ids,legal):
    # A model picks its highest supported legal code; degree one.
    base=z['pa'][ids]
    scores[np.arange(len(ids)),base%64]=-1
    scores[:,64:]=np.where(legal[ids],scores[:,64:],-1)
    return scores


def predict(name,z,f,ids,table,global_prior):
    if fixed(name):
        delta={'next':1,'repeat_pc':f['local'][ids],'look2_pc':2*f['local'][ids]}[name]
        target=z['pa'][ids]+delta
        ok=(target!=z['pa'][ids])&(target//64==z['pa'][ids]//64)
        return np.where(ok,target,-1),np.ones(len(ids)),np.full(len(ids),-1)
    def get(r):
        fallback=get(MODELS[r])[0] if MODELS[r] else global_prior
        return frequencies(table[r],f['keys'][r][ids],fallback)
    scores,support=get(name)
    scores=eligible_scores(scores,z,ids,f['wide_legal'])
    code=scores.argmax(1);confidence=scores[np.arange(len(scores)),code]
    target=np.where(code<64,(z['pa'][ids]//64)*64+code,z['pa'][ids]+WIDE[np.maximum(0,code-64)])
    return target,confidence,support


def fit_all(z,f,g,train,service,band):
    _,timely,_=outcomes(z,g,train,service,band)
    y=np.where(timely,g['code'][train],-1)
    return {r:fit_keys(f['keys'][r][train],y) for r in MODELS}, np.bincount(y[y>=0],minlength=ACTIONS)/len(train)


def predictions(r,z,f,ids,tabs,prior):
    parts=[predict(r,z,f,ids[j:j+4096],tabs,prior) for j in range(0,len(ids),4096)]
    return tuple(np.concatenate(columns) for columns in zip(*parts))


def threshold(conf,target):
    for v in (.005,.01,.025,.05,.10,.20,.40,.80,1.01):
        if np.mean((conf>=v)&(target>=0))<=.125:
            return v
    return 1.01


def median(x):
    return round(float(np.median(x)),1) if len(x) else ''


def earliest(ids,q,mask):
    pos=np.flatnonzero(mask)
    if not len(pos): return pos
    _,first=np.unique(q[pos],return_index=True)
    return pos[first]


def score(meta,z,f,g,ids,rule,mode,issue,target,confidence,support,service,band,threshold):
    raw,timely,q=outcomes(z,g,ids,service,band)
    end=g['end']; true=z['pa'][q]; pa=z['pa']; cycles=z['t']
    correct=(target==true)&raw&issue; good=correct&timely
    represent=timely&(g['code'][ids]>=0)
    prior=good&(ids<g['last'][q]);at=good&(ids==g['last'][q])
    native=g['native'][q]; none=native<0
    stable=g['stable'][ids]
    # Count targets reachable from fully evaluated anchors only. May include valid
    # future endpoints in the guarded tail. The population starts at first anchor.
    in_pop=np.arange(len(end))>=ids[0]
    pop=sum(end&in_pop)
    na=earliest(ids,q,good); pr=earliest(ids,q,prior)
    out={**meta,'band':band,'rule':rule,'gate':mode,'service_cycles':service,'dev_cutoff':threshold,
         'eligible_ingresses':len(ids),'issued':int(issue.sum()),'endpoints_in_interval':int(pop),
         'endpoints_reachable_in_time':len(np.unique(q[timely])),
         'endpoints_representable_in_time':len(np.unique(q[represent])),
         'endpoints_named':len(np.unique(q[correct])),'endpoints_named_on_time':len(na),
         'endpoints_named_strictly_before_last_T':len(pr),'endpoints_named_at_last_T':len(np.unique(q[at])),
         'target_late_in_interval':int((end&in_pop&(z['pf_kind']==2)).sum()),
         'target_P_in_interval':int((end&in_pop&(z['pf_kind']!=2)).sum()),
         'named_late_target':len(np.unique(q[good&(z['pf_kind'][q]==2)])),
         'named_P_target':len(np.unique(q[good&(z['pf_kind'][q]!=2)])),
         'named_from_earlier_T_in_ending_run':len(np.unique(q[prior&(ids>=g['first_T'][q])])),
         'named_from_before_ending_run_began':len(np.unique(q[prior&(ids<g['first_T'][q])])),
         'endpoints_named_only_short_lead':len(np.setdiff1d(q[correct],q[good])),
         'correct_opportunities':int(good.sum()),'short_lead_opportunities':int((correct&~timely).sum()),
         'false_against_endpoint':int((issue&~good).sum()),'stable_continuations':int(stable.sum()),
         'issued_on_stable':int((issue&stable).sum()),'stable_false_against_endpoint':int((issue&stable&~good).sum()),
         'stable_without_endpoint_in_band':int((stable&~raw).sum()),'issued_on_stable_without_endpoint_in_band':int((issue&stable&~raw).sum()),
         'named_with_no_native_proposal':len(np.unique(q[good&none])),
         'named_earlier_than_existing_native':len(np.unique(q[good&~none&(ids<native)])),
         'named_no_earlier_than_existing_native':len(np.unique(q[good&~none&(ids>=native)])),
         'median_earliest_cycles':median(cycles[q[na]]-cycles[ids[na]]),
         'median_earliest_events':median(q[na]-ids[na]),'median_before_lastT_cycles':median(cycles[g['last'][q[pr]]]-cycles[ids[pr]]),
         'explicit_context_unseen_ingresses':int((support==0).sum()),'context_supported_ingresses':int((support>=8).sum())}
    ex=[]
    for reason,allowed in (('prelast_vs_native',prior&~none&(ids<native)),('prelast_no_native',prior&none),
                           ('at_last_no_native',at&none)):
        jj=np.flatnonzero(allowed)
        if not len(jj):continue
        k=jj[0];a=ids[k];b=q[k];h=g['last'][b];gen=native[k]
        ex.append({**meta,'band':band,'rule':rule,'gate':mode,'reason':reason,
                   'trigger_instr':int(z['instr'][a]),'trigger_op':int(z['op'][a]),'trigger_pc':int(z['pc'][a]),'trigger_line':int(pa[a]),
                   'last_T_instr':int(z['instr'][h]),'last_T_op':int(z['op'][h]),'target_instr':int(z['instr'][b]),'target_op':int(z['op'][b]),
                   'target_line':int(pa[b]),'target_class':'L' if z['pf_kind'][b]==2 else 'P','train_context_obs':int(support[k]),
                   'score':round(float(confidence[k]),5),'cycles_to_last_T':int(cycles[h]-cycles[a]),
                   'cycles_to_target':int(cycles[b]-cycles[a]),'ingresses_to_target':int(b-a),
                   'native_same_line_trigger_instr':int(z['instr'][gen]) if gen>=0 else '',
                   'earlier_than_native_cycles':int(cycles[gen]-cycles[a]) if gen>=0 else '',
                   'target_head_wait':int(z['head'][b])})
    return out,ex


def run(job):
    from evaluate import grant
    train_name,test_name,proj=job
    z,f,g=case(train_name,proj);t,ft,gt=case(test_name,proj)
    nt=len(z['t']); end=int(.6*nt);dev_end=int(.8*nt)
    tr=guarded(z,0,end);dev=guarded(z,end,dev_end)
    within=train_name==test_name
    lo=int(.8*len(t['t'])) if within else 0
    te=guarded(t,lo,len(t['t']))
    if min(len(tr),len(dev),len(te))==0:return [],[],[]
    service=measured_service(z,end)
    meta=dict(train=train_name,test=test_name,split='within' if within else 'transfer',view=proj)
    summary=[];examples=[];ceil=[]
    for band in BANDS:
        tabs,prior=fit_all(z,f,g,tr,service,band)
        for r in RULES:
            yt,sc,sup=predictions(r,t,ft,te,tabs,prior)
            if fixed(r): cut=0.
            else:
                yd,sd,_=predictions(r,z,f,dev,tabs,prior)
                cut=threshold(sd,yd)
            for mode,allow in [('all',np.ones(len(te),bool)),('fixed_1in8',grant(te)),('dev_gate',sc>=cut)]:
                issue=allow&(yt>=0)&(sc>0)
                so,ex=score(meta,t,ft,gt,te,r,mode,issue,yt,sc,sup,service,band,cut)
                summary.append(so)
                if band=='1_64' and mode=='fixed_1in8' and r in ('current','short_local','short_global','page'): examples.extend(ex)
        raw,ok,q=outcomes(t,gt,te,service,band)
        pre=te<gt['last'][q];code=gt['code'][te];fits=(code>=0)
        short=raw&~ok;pop=int(gt['end'][lo:].sum());possible=np.unique(q[ok]);rep=np.unique(q[ok&fits])
        ceil.append({**meta,'band':band,'train_anchors':len(tr),'dev_anchors':len(dev),'test_anchors':len(te),
                     'endpoints_in_test':pop,'any_cue_in_window':len(np.unique(q[raw])),
                     'any_cue_meets_reference':len(possible),'reachable_same_page_in_time':len(np.unique(q[ok&(code>=0)&(code<64)])),
                     'reachable_wide_seen_page_in_time':len(np.unique(q[ok&(code>=64)])),
                     'any_representable_in_time':len(rep),'no_timely_cue':pop-len(possible),'timely_but_no_address_in_menu':len(possible)-len(rep),
                     'representable_pre_last_T_in_time':len(np.unique(q[ok&fits&pre])),
                     'representable_at_last_T_in_time':len(np.unique(q[ok&fits&(te==gt['last'][q])])),
                     'train_service_cycles':service})
    return summary,examples,ceil


def main():
    from concurrent.futures import ProcessPoolExecutor
    args=argparse.ArgumentParser()
    args.add_argument('--jobs',type=int,default=3); args.add_argument('--mid',action='store_true')
    args.add_argument('--only',nargs=2)
    args=args.parse_args()
    panels=[(v,v) for v in WITHIN]+TRANSFER
    if args.only:panels=[tuple(args.only)]
    if args.mid:panels=[(s,t) for s in ('gap_bfs3','gap_bfs3_late') for t in ('gap_bfs3_mid','gap_bfs8_mid')]
    jobs=[(a,b,p) for a,b in panels for p in ('ingress','ingress_pc')]
    dest='middle_' if args.mid else ''
    if args.only:dest=args.only[1]+'_pilot_'
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        groups=list(ex.map(run,jobs))
    for i,fname in enumerate(('scores.csv','examples.csv','reach.csv')):
        rows=[r for group in groups for r in group[i]]
        if rows:writer(HERE/(dest+fname),rows)


if __name__=='__main__':
    main()
