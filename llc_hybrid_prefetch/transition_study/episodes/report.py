"""Count maximal attributed-timely runs, preserve interruptions, and pair continuations."""
import argparse
import bisect
import csv
import gzip
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from extract import ACTIONS,CASES,CLASSES,DEST,HERE,PATHS
VIEWS=('core','core_pc','core_page','core_region4','ingress','ingress_pc','ingress_page','ingress_region4')
STATES=('unobserved','cold','delta_changed','zero_delta','stride','intervention')
EP=('case','core','view','stream','start_instr','start_op','last_instr','last_op','first_miss_instr','first_miss_op',
    'recovery_instr','recovery_op','run_length','failure_class','left_censored','recovery_censored','recovery_gap',
    'anchor_pc','target_pc','anchor_line','target_line','anchor_ingress','anchor_lookup','target_ingress','target_lookup',
    'anchor_state','target_state','target_prior_action','anchor_proposal','anchor_admission','target_pf_generation','target_pf_fill',
    'core_positions_to_target','ingresses_to_target','anchor_resolved_before_target','target_head')
MATCH=('case','core','view','failure_class','miss_anchor_instr','miss_anchor_op','miss_instr','miss_op','control_anchor_instr',
       'control_anchor_op','control_next_instr','control_next_op','case_run_length','control_prefix','same_stream','absolute_core_distance')
METRIC=('pc_changed','delta_changed','cross_page','absolute_delta','core_position_gap','signed_ingress_gap','ingress_reordered',
        'anchor_tag_known_before_target','anchor_named_next','anchor_admission_accepted','target_mshr','target_dram_rq','anchor_mshr',
        'anchor_dram_rq','target_head','anchor_prior_gap','same_page_previous_history')
def ids(z,x):
    return divmod(int(z['ids'][x]),256) if x>=0 else ('','')
def quant(x):
    a=np.asarray(x)
    if not len(a):return ('','','')
    return tuple(round(float(k),3) for k in np.percentile(a,[25,50,75]))
def qrow(x):
    return (len(x),*(quant(x)),round(float(np.mean(x)),4) if len(x) else '')
def interval_group(v):return '1' if v==1 else '2-3' if v<4 else '4-7' if v<8 else '8-15' if v<16 else '16+'
def writers(folder):
    specs={'episode_ids':EP,'matched_ids':MATCH,'population':('case','core','view','outcome','events'),
     'endings':('case','core','view','run_bin','following','runs','left_censored','no_recovery','gap_p25','gap_p50','gap_p75','recovered_median_core_distance'),
     'comparison':('case','core','view','failure_class','role','metric','n','p25','median','p75','mean'),
     'reasons':('case','core','view','failure_class','metric','value','events'),
     'timeline':('case','core','view','example','position','role','instr','op','pc','virtual_line','physical_line','NoPF_path','stride_path',
                 'outcome','ingress','lookup','pf_gen','pf_fill','own_state','own_proposal','own_admission','last_target_action','head','source_label')}
    handles={};ws={}
    for name,cols in specs.items():
        f=gzip.open(folder/(name+'.csv.gz'),'wt',newline='') if name in ('episode_ids','matched_ids') else (folder/(name+'.csv')).open('w',newline='')
        handles[name]=f;ws[name]=csv.writer(f,lineterminator='\n');ws[name].writerow(cols)
    return handles,ws

def labels(case,z):
    from extract import ROOT,runpath
    import json
    meta=json.loads((runpath(case,'stride')/'run.json').read_text())
    if meta.get('group') not in ('synthetic','compiled'):return {}
    p=ROOT/'opportunity_study/raw/traces'/(meta['traces'][0]+'.labels.csv.gz')
    if not p.exists():return {}
    with gzip.open(p,'rt') as f:return {int(r['instr_id'])*256+int(r['operand']):r['label'] for r in csv.DictReader(f) if r['measurement']=='roi'}

def project(case,cpu,z,view,ws,lab):
    base=(case,cpu,view)
    ixs=z['ingress'] if view.startswith('ingress') else np.arange(len(z['ids']))
    if view.endswith('_pc'):groups=z['pc'][ixs]
    elif view.endswith('_page'):groups=z['pa'][ixs]//64
    elif view.endswith('_region4'):groups=z['pa'][ixs]//256
    else:groups=np.zeros(len(ixs),dtype=int)
    sort=np.argsort(groups,kind='stable');ixs=ixs[sort];groups=groups[sort]
    outcomes=z['outcome'][ixs];N=len(ixs)
    if not N:return
    for i,n in enumerate(np.bincount(outcomes,minlength=len(CLASSES))):
        ws['population'].writerow((*base,CLASSES[i],n))
    T=outcomes==5
    begins=np.r_[True,groups[1:]!=groups[:-1]]
    ends=np.r_[begins[1:],True]
    starts=np.flatnonzero(T&(begins|np.r_[True,~T[:-1]]))
    end=np.flatnonzero(T&(ends|np.r_[~T[1:],True]))
    if len(starts)!=len(end):raise ValueError('unpaired runs')
    lengths=end-starts+1
    categories=defaultdict(list);reasons=Counter();stats=defaultdict(list)
    groupend=ends[end]
    conts=np.flatnonzero(T[:-1]&T[1:]&~ends[:-1])
    cr=np.searchsorted(starts,conts,side='right')-1
    prefix=conts-starts[cr]+1
    pools=defaultdict(list)
    def matchkey(pos,k):
        a=ixs[pos]
        return (int(z['pc'][a]),int(z['pa'][a]%64)//16,min(8,int(k)),min(3,4*int(a)//len(z['ids'])))
    for pos,count in zip(conts[prefix>=2],prefix[prefix>=2]):
        pools[matchkey(pos,count)].append((int(ixs[pos]),int(pos),int(count)))
    for arr in pools.values():arr.sort()
    seen=set()
    def metrics(pos):
        prev,a,q=ixs[pos-1:pos+2];pa=z['pa'];t=z['t'];order=z['iorder']
        aq=int(t[q])-int(t[a]);ph=int(t[a])-int(t[prev])
        return (z['pc'][a]!=z['pc'][q],pa[q]-pa[a]!=pa[a]-pa[prev],pa[q]//64!=pa[a]//64,abs(pa[q]-pa[a]),
                int(q)-int(a),aq,aq<0,z['tag_t'][a]<t[q],z['proposed'][a]==pa[q],z['own_admission'][a]==2,
                z['mshr'][order[q]],z['dram_rq'][order[q]],z['mshr'][order[a]],z['dram_rq'][order[a]],z['head'][q],ph,pa[prev]//64==pa[a]//64)
    for r,(s,e,k,cens) in enumerate(zip(starts,end,lengths,groupend)):
        s,e,k=int(s),int(e),int(k)
        follow='right_censored' if cens else CLASSES[outcomes[e+1]]
        nr=r+1
        recovered=nr<len(starts) and groups[starts[nr]]==groups[e]
        at=int(starts[nr]) if recovered else -1
        categories[interval_group(k),follow].append((int(begins[s]),int(not recovered),
                    at-e-1 if recovered else -1,abs(int(ixs[at])-int(ixs[e])) if recovered else -1))
        if k<2 or follow not in ('L','P'):continue
        a,q=map(int,ixs[e:e+2]);ret=int(ixs[at]) if recovered else -1
        state=lambda j:STATES[max(0,int(z['state'][j]))]
        action=ACTIONS[z['last_action'][q]]
        pstart,pend=int(ixs[s]),int(ixs[e])
        ws['episode_ids'].writerow((*base,int(groups[e]),*ids(z,pstart),*ids(z,pend),*ids(z,q),*ids(z,ret),k,follow,
              int(begins[s]),int(not recovered),at-e-1 if recovered else '',int(z['pc'][a]),int(z['pc'][q]),int(z['pa'][a]),int(z['pa'][q]),
              z['t'][a],z['tag_t'][a],z['t'][q],z['tag_t'][q],state(a),state(q),action,z['proposed'][a],ACTIONS[z['own_admission'][a]],
              z['pf_gen'][q],z['pf_fill'][q],q-a,z['iorder'][q]-z['iorder'][a],int(z['tag_t'][a]<z['t'][q]),z['head'][q]))
        for name,value in (('last_action',action),('anchor_stride_state',state(a)),('target_stride_state',state(q)),
                           ('anchor_admission',ACTIONS[z['own_admission'][a]]),('anchor_named_target',str(int(z['proposed'][a]==z['pa'][q])))):
            reasons[follow,name,value]+=1
        pool=pools[matchkey(e,k)]
        if pool:
            idx=bisect.bisect_left(pool,(a,-1,-1))
            candidates=[j for j in (idx-1,idx) if 0<=j<len(pool)]
            j=min(candidates,key=lambda c:(abs(pool[c][0]-a),pool[c][0]))
            c,loc,count=pool.pop(j);d=int(ixs[loc+1])
            ws['matched_ids'].writerow((*base,follow,*ids(z,a),*ids(z,q),*ids(z,c),*ids(z,d),k,count,int(groups[e]==groups[loc]),abs(a-c)))
            for role,position in (('failure',e),('continued_T',loc)):
                for metric,val in zip(METRIC,metrics(position)):stats[follow,role,metric].append(float(val))
        if (view in ('ingress','core_pc') and follow not in seen):
            seen.add(follow)
            times=list(range(max(s,e-2),e+2))
            if recovered:times+=list(range(max(e+2,at-1),min(N,at+1)))
            for pos in times:
                event=int(ixs[pos]);role='return_T' if pos==at else 'failure' if pos==e+1 else 'last_T' if pos==e else 'prior_T' if pos<e else 'intervening'
                ws['timeline'].writerow((*base,follow,pos-e,role,*ids(z,event),z['pc'][event],z['va'][event],z['pa'][event],PATHS[z['base_path'][event]],
                    PATHS[z['path'][event]],CLASSES[z['outcome'][event]],z['t'][event],z['tag_t'][event],z['pf_gen'][event],z['pf_fill'][event],
                    state(event),z['proposed'][event],ACTIONS[z['own_admission'][event]],ACTIONS[z['last_action'][event]],z['head'][event],lab.get(int(z['ids'][event]),'')))
    for (bucket,f),vals in sorted(categories.items()):
        ar=np.asarray(vals);ok=ar[:,2]>=0
        ws['endings'].writerow((*base,bucket,f,len(vals),ar[:,0].sum(),ar[:,1].sum(),*quant(ar[ok,2]),quant(ar[ok,3])[1]))
    for (f,name,value),n in sorted(reasons.items()):ws['reasons'].writerow((*base,f,name,value,n))
    for key,values in sorted(stats.items()):ws['comparison'].writerow((*base,*key,*qrow(values)))

def one(case):
    folder=DEST/case
    if not (folder/'complete').exists():raise FileNotFoundError(case)
    hs,ws=writers(folder)
    try:
        for path in sorted(folder.glob('c*.npz')):
            cpu=int(path.stem[1:])
            with np.load(path) as f:z={k:f[k] for k in f.files}
            z['iorder']=np.full(len(z['ids']),-1);z['iorder'][z['ingress']]=np.arange(len(z['ingress']))
            lab=labels(case,z)
            for view in VIEWS:project(case,cpu,z,view,ws,lab)
    finally:
        for h in hs.values():h.close()
    print(case,flush=True)

def combine(cases):
    for f in sorted((DEST/cases[0]).glob('*.csv*')):
        op=gzip.open if f.name.endswith('.gz') else open
        temp=DEST/(f.name+'.pending')
        with op(temp,'wt',newline='') as out:
            for j,case in enumerate(cases):
                with op(DEST/case/f.name,'rt') as stream:
                    first=next(stream)
                    if not j:out.write(first)
                    for ln in stream:out.write(ln)
        temp.replace(HERE/f.name)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--jobs',type=int,default=3);p.add_argument('--cases',nargs='+',default=CASES)
    args=p.parse_args()
    with ProcessPoolExecutor(args.jobs) as ex:list(ex.map(one,args.cases))
    combine(args.cases)
