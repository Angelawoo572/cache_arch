"""Verify the frozen online C++ rule and reconstructed archived transition identities."""
import csv,gzip,subprocess,tempfile
from pathlib import Path
import numpy as np
from predict import HERE,MODELS,case,cohort,features,guarded,fit_all,measured_service,predictions
from evaluate import grant
from export import export,MODES

PROGRAM='''
#include "transition_rule.hh"
#include <iostream>
int main(int n,char** v){
    boundary_forecast::Engine rule(v[1]);
    unsigned long long pc,line,t;
    while(std::cin>>pc>>line>>t) std::cout<<rule.step(pc,line,t)<<'\\n';
}
'''


def shadow(name,target,exe,folder,mode='page'):
    z,f,g=case(name,'ingress');t,ft,gt=case(target,'ingress')
    nt=len(z['t']);cut=int(.6*nt)
    tr=guarded(z,0,cut);tm=measured_service(z,cut)
    tabs,prior=fit_all(z,f,g,tr,tm,'1_64')
    lo=int(.8*len(t['t'])) if name==target else 0;te=guarded(t,lo,len(t['t']))
    tab=folder/'frozen.txt';expected=folder/'expected.csv';export(name,target,tab,mode,expected)
    ref=np.full(len(t['t']),-1,int)
    with expected.open() as f:
        for r in csv.DictReader(f):
            ref[int(r['index'])]=int(r['choice'])
    if mode=='page':
        physically_selected,score,_=predictions('page',t,ft,te,tabs,prior)
        ok=grant(te)&(score>0)
        np.testing.assert_array_equal(ref[te[ok]],physically_selected[ok])
    entries=''.join(f'{ip} {line} {tim}\n' for ip,line,tim in zip(t['pc'],t['pa'],t['t']))
    result=subprocess.run([str(exe),str(tab)],input=entries,capture_output=True,text=True,check=True)
    real=np.fromstring(result.stdout,dtype=np.int64,sep=' ')
    np.testing.assert_array_equal(real,ref)
    print(name,target,mode,'checked all',len(ref),'ingresses and',sum(ref>=0),'positive choices')
    # A suffix change must leave the prefix's keys and selected online action unchanged.
    k=len(t['t'])//2
    altered=dict(t); altered['pc']=t['pc'].copy();altered['pa']=t['pa'].copy()
    altered['pc'][k:]^=1971;altered['pa'][k:]+=192
    changed=features(altered)
    for rule in MODELS:
        np.testing.assert_array_equal(changed['keys'][rule][:k],ft['keys'][rule][:k])
    np.testing.assert_array_equal(changed['wide_legal'][:k],ft['wide_legal'][:k])

def episode_check():
    by={}
    path=HERE.parent/'episodes/episode_ids.csv.gz'
    if not path.exists():return
    with gzip.open(path,'rt') as f:
        for r in csv.DictReader(f):
            if r['view'] in ('ingress','ingress_pc') and r['core']=='0':
                by.setdefault((r['case'],r['view']),set()).add((int(r['last_instr']),int(r['last_op']),int(r['first_miss_instr']),int(r['first_miss_op'])))
    rows=[]
    from predict import WITHIN
    names=set(WITHIN+['gap_bfs3','gap_bfs8','gap_bfs3_mid','gap_bfs8_mid'])
    raw=HERE.parent/'raw/episodes'
    for name,view in sorted((k,v) for k in names if (raw/k).is_dir() for v in ('ingress','ingress_pc')):
        observed=by.get((name,view),set())
        z,f,g=case(name,view)
        expect={(int(z['instr'][g['last'][i]]),int(z['op'][g['last'][i]]),int(z['instr'][i]),int(z['op'][i])) for i in np.flatnonzero(g['end'])}
        if expect!=observed:raise ValueError((name,view,len(expect),len(observed),list(expect^observed)[:3]))
        rows.append((name,view,len(expect)))
    with (HERE/'cohort_checks.csv').open('w') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(('case','view','identical_episode_pairs'));w.writerows(rows)

if __name__=='__main__':
    with tempfile.TemporaryDirectory() as d:
        folder=Path(d); src=folder/'main.cc';exe=folder/'shadow'
        src.write_text(PROGRAM)
        subprocess.run(['c++','-std=c++14','-O2','-I',str(HERE),str(src),'-o',str(exe)],check=True)
        for name in ('gap_bfs3_late','row_period2_paced'):shadow(name,name,exe,folder)
        for mode in MODES:shadow('lbm-r1','lbm-r2',exe,folder,mode)
    episode_check()
