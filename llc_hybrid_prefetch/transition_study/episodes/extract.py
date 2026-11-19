"""Align every core load operand and both cache paths to measured ingress feedback."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
HISTORY=ROOT/'opportunity_study/raw/history'
DEST=HERE.parent/'raw/episodes'
ORIGINAL_CASES=('row_period2_paced','row_period5_paced','streams_three_identified_paced','streams_three_switch_paced',
       'compiled_rows','gap_bfs3_late','gap_bfs8_late','gap_bfs3','gap_bfs8','gap_bfs8_lbm',
       'cactus-r2','lbm-r1','mcf-r1','mcf-r2','mcf-cactus-r1','lbm-xalan-r1','gap_bfs3_mid','gap_bfs8_mid')
PATHS=('no_lookup','upper_merge_forward','l1_hit','l2_hit','llc_merge','llc_forward','llc_hit','llc_miss')
CLASSES=('upper_other','ordinary_llc_hit','ordinary_miss','paired_hit_to_miss','paired_miss_to_other_hit',
         'T','L','P','base_miss_no_stride_lookup','attributed_hit_other_baseline','late_other_baseline')
ACTIONS=('none','proposed','accepted','dispatch','fill','timely','late','evicted','redundant','unresolved',
         'rejected_page','rejected_resident','rejected_mshr','rejected_rq','rejected_wq','rejected_pq',
         'rejected_outstanding','rejected_pq_full','rejected_self','rejected_enqueue')
SUPPLEMENTAL_CASES=('lbm-r2',)
CASES=ORIGINAL_CASES+SUPPLEMENTAL_CASES

def runpath(case,policy):
    for r in (ROOT/'raw/runs',ROOT/'opportunity_study/raw/runs',HERE.parent/'raw/runs'):
        p=r/(case+'-'+policy)
        if (p/'complete.json').is_file():return p
    raise FileNotFoundError((case,policy))

def read_paths(p,pressure):
    pieces={}
    columns=['cpu','instr_id','operand','pc','virtual_line','level','stage','hit']
    if pressure:columns+=['mshr','rq','pq','dram_rq','dram_wq']
    for df in pd.read_csv(p/'loads.csv.gz',usecols=columns,chunksize=300000):
        for cpu,g in df.groupby('cpu',sort=False):
            ids=g.instr_id.to_numpy()*256+g.operand.to_numpy()
            item=pieces.setdefault(cpu,dict(core=[],path=[],ing=[]))
            core=(g.level=='CORE').to_numpy()
            if core.any():item['core'].append(np.column_stack((ids[core],g.pc.to_numpy()[core],g.virtual_line.to_numpy()[core])))
            lookup=(g.stage=='lookup');hit=g.hit==1;merge=g.stage.isin(('merged','forwarded'));llc=g.level=='LLC'
            status=np.select((llc&lookup&~hit,llc&lookup&hit,llc&(g.stage=='forwarded'),llc&(g.stage=='merged'),
                              (g.level=='L2')&lookup&hit,(g.level=='L1')&lookup&hit,~llc&merge),(7,6,5,4,3,2,1),0)
            ok=status>0
            if ok.any():item['path'].append(np.column_stack((ids[ok],status[ok])))
            if pressure:
                at=(llc&(g.stage=='ingress')).to_numpy()
                if at.any():item['ing'].append(np.column_stack([ids[at]]+[g[s].to_numpy()[at] for s in columns[-5:]]))
    out={}
    for cpu,item in pieces.items():
        a=np.concatenate(item['core']);ids=a[:,0];path=np.zeros(len(a),dtype='int8')
        if np.any(np.diff(ids)<=0):raise ValueError('CORE order')
        for v in item['path']:
            q=np.searchsorted(ids,v[:,0]);assert np.array_equal(ids[q],v[:,0])
            np.maximum.at(path,q,v[:,1].astype('int8'))
        out[cpu]=(a,path,np.concatenate(item['ing']) if item['ing'] else None)
    return out

def last_reason(p,cores):
    d=pd.read_csv(p/'candidates.csv.gz',usecols=['id','instr_id','operand','cpu','target','stage','reason','cycle'])
    d=d[d.stage!='silent'].copy()
    d['kind']=np.where(d.stage=='rejected','rejected_'+d.reason,d.stage)
    action_index={v:j for j,v in enumerate(ACTIONS)}
    for cpu,(z,idx,__) in cores.items():
        own=d[(d.cpu==cpu)&d.stage.isin(('rejected','accepted'))]
        q=np.searchsorted(z['ids'],own.instr_id.to_numpy()*256+own.operand.to_numpy())
        z['own_admission'][q]=own.kind.map(action_index).fillna(0).to_numpy(dtype='int8')
        # Stable input row order breaks same-line/same-cycle ties; strict cutoff excludes all target-cycle actions.
        t=z['tag_t'][idx];valid=t>=0;pos=idx[valid];t=t[valid]
        if not len(t) or d.empty:continue
        line=z['pa'][pos]
        scale=max(int(t.max()),int(d.cycle.max()))+1
        token=d.target.to_numpy()*scale+d.cycle.to_numpy()
        order=np.argsort(token,kind='stable')
        ref=np.searchsorted(token[order],line*scale+t,side='left')-1
        si=order[np.maximum(ref,0)]
        sort=np.lexsort((t,line))
        previous=np.full(len(t),-1,dtype='int64')
        previous[sort[1:]]=np.where(line[sort[1:]]==line[sort[:-1]],t[sort[:-1]],-1)
        ok=(ref>=0)&(d.target.to_numpy()[si]==line)&(d.cycle.to_numpy()[si]>previous)
        mapped=d.kind.map(action_index)
        if mapped.isna().any():raise ValueError(d.kind[mapped.isna()].unique())
        z['last_action'][pos[ok]]=mapped.to_numpy(dtype='int8')[si[ok]]

def extract(case):
    out=DEST/case
    if (out/'complete').is_file():return
    out.mkdir(exist_ok=True,parents=True)
    no,pf=runpath(case,'nopf'),runpath(case,'stride')
    n=read_paths(no,False);s=read_paths(pf,True)
    maps=pd.read_csv(pf/'pages.csv.gz')
    if not maps.equals(pd.read_csv(no/'pages.csv.gz')):
        a=maps.sort_values(['cpu','virtual_page']).reset_index(drop=True)
        b=pd.read_csv(no/'pages.csv.gz').sort_values(['cpu','virtual_page']).reset_index(drop=True)
        if not a.equals(b):raise ValueError('placement differs')
    history=HISTORY
    if (DEST.parent/'history'/(case+'-c0.npz')).exists():
        history=DEST.parent/'history'
    if not (history/(case+'-c0.npz')).exists():
        import importlib.util
        src=ROOT/'opportunity_study/history/extract.py'
        spec=importlib.util.spec_from_file_location('history_extractor',src)
        mod=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        history=DEST.parent/'history'
        history.mkdir(exist_ok=True,parents=True)
        mod.CACHE=history
        mod.extract((case,pf.parent))
    data={}
    for cpu,(a,path,ing) in s.items():
        b,base,_=n[cpu]
        if not np.array_equal(a,b):raise ValueError('unaligned core identity')
        z=dict(ids=a[:,0],pc=a[:,1],va=a[:,2],base_path=base,path=path)
        mp=maps[maps.cpu==cpu].sort_values('virtual_page');vpn=z['va']//64
        ind=np.searchsorted(mp.virtual_page.to_numpy(),vpn)
        if not np.array_equal(mp.virtual_page.to_numpy()[ind],vpn):raise ValueError('missing frame')
        z['pa']=mp.physical_page.to_numpy()[ind]*64+z['va']%64
        z['outcome']=np.select((path==6,(path==7)&(base==6),path==7,(base==7)&(path<6)),(1,3,2,8),0).astype('int8')
        for k in ('last_action','own_admission'):z[k]=np.zeros(len(a),dtype='int8')
        with np.load(history/(case+'-c'+str(cpu)+'.npz')) as h:
            hi=h['instr']*256+h['op'];idx=np.searchsorted(z['ids'],hi)
            if not np.array_equal(z['ids'][idx],hi) or not np.array_equal(h['pa'],z['pa'][idx]):raise ValueError('ingress identity')
            if not np.array_equal(ing[:,0],hi):raise ValueError('ingress chronology')
            for j,k in enumerate(('mshr','rq','pq','dram_rq','dram_wq')):z[k]=ing[:,j+1].astype('int16')
            for k in ('t','tag_t','pf_gen','pf_fill','proposed','state','pf_kind','head'):
                z[k]=np.full(len(a),-1,dtype='int64'); z[k][idx]=h[k]
            z['ingress']=idx
            paired=(base[idx]==7)&(path[idx]==7)
            kind=h['pf_kind']; hit=path[idx]==6
            o=z['outcome'][idx]
            o[paired]=7;o[paired&(kind==2)]=6
            o[(base[idx]==7)&hit]=4
            o[hit&(kind==1)&(base[idx]!=7)]=9
            o[(kind==2)&~paired]=10
            o[(base[idx]==7)&hit&(kind==1)]=5
            z['outcome'][idx]=o
        data[cpu]=(z,idx,path)
    last_reason(pf,data)
    for cpu,(z,_,__) in data.items():np.savez_compressed(out/('c'+str(cpu)+'.npz'),**z)
    (out/'complete').write_text('complete\n')
    print(case,[(c,len(z['ids']),len(ix)) for c,(z,ix,_) in data.items()],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--jobs',type=int,default=3);p.add_argument('--cases',nargs='+',default=CASES)
    a=p.parse_args()
    with ProcessPoolExecutor(a.jobs) as ex:list(ex.map(extract,a.cases))
