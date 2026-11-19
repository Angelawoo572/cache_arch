"""Run standalone full-rate conventional controls for the supplemental 3766B region."""
import csv
import gzip
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RAW=HERE.parent/'raw/strong_controls'
BINARY=ROOT/'opportunity_study/raw/conventional/build/pythia/bin/champsim'
ORIGINAL=ROOT/'raw/runs/lbm-r2-stride'
MODES=('stride','spp','multi','xpc','temporal')

def stats(folder):
    with (folder/'stats.csv').open() as f:return {r['key']:int(r['value']) for r in csv.DictReader(f)}
def cores(folder):
    with (folder/'cores.csv').open() as f:return list(csv.DictReader(f))
def page_map(folder):
    with gzip.open(folder/'pages.csv.gz','rt') as f:return sorted((r['cpu'],r['virtual_page'],r['physical_page']) for r in csv.DictReader(f))
def run(arm):
    mode,setting=arm
    folder=RAW/'verified'/(setting+'-'+mode);folder.mkdir(exist_ok=True,parents=True)
    meta=json.loads((ORIGINAL/'run.json').read_text())
    cap=stats(ORIGINAL)['prefetch_accepted']
    args=[str(BINARY),f"--warmup_instructions={meta['warmup']}",f"--simulation_instructions={meta['roi']}",'-traces',
          str(ROOT/'raw/traces'/(meta['traces'][0]+'.champsimtrace'))]
    env={k:v for k,v in os.environ.items() if not k.startswith(('HYBRID_','LLC_PF_','STUDY_'))}
    env.update(meta['environment'],STUDY_POLICY=mode,STUDY_EPOCH_LIMIT='0',STUDY_LATENCY='0',STUDY_FAST_NATIVE='0',STUDY_CHOICES='0',
               HYBRID_DETAIL='1',HYBRID_OUT=str(folder))
    if setting=='capped':env['STUDY_CAPS']=str(cap)
    if not (folder/'complete.json').is_file():
        with (folder/'run.log').open('w') as f:
            subprocess.run(args,cwd=folder,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        c=cores(folder)
        if len(c)!=1 or int(c[0]['instructions'])!=meta['roi'] or int(c[0]['warmup'])!=meta['warmup']:
            raise ValueError('wrong retirement window')
        if page_map(folder)!=page_map(ORIGINAL):raise ValueError('page placement mismatch')
        (folder/'complete.json').write_text(json.dumps(dict(policy=mode,setting=setting,cap=cap if setting=='capped' else None,trace=meta['traces'][0]))+'\n')
    c=cores(folder);s=stats(folder);ref=stats(ORIGINAL);orig=cores(ORIGINAL)
    if mode=='stride':
        if c!=orig:raise ValueError('fixed-work baseline mismatch')
        if any(s[k]!=v for k,v in ref.items()):raise ValueError([(k,v,s.get(k)) for k,v in ref.items() if s.get(k)!=v])
    duration=int(c[0]['cycles']);before=int(orig[0]['cycles']);accepted=s['prefetch_accepted']
    parts={k:s.get('study_pf_'+k,0) for k in ('timely','late','evicted','redundant','unresolved')}
    if sum(parts.values())!=accepted:raise ValueError('incomplete lifecycle')
    traffic=sum(s['dram_'+x+'_bytes'] for x in ('demand','prefetch','write'))
    native_traffic=sum(ref['dram_'+x+'_bytes'] for x in ('demand','prefetch','write'))
    return dict(policy=mode,setting=setting,source=meta['traces'][0],roi=meta['roi'],cycles=duration,cycles_saved_vs_stride=before-duration,
                time_saved_pct=round(100*(before-duration)/before,6),accepted_ceiling=cap if setting=='capped' else '',
                accepted=accepted,**parts,budget_rejections=s.get('study_rejected_budget',0),cross_page_accepted=s.get('study_accepted_crosspage',0),
                dram_bytes=traffic,dram_bytes_vs_stride=traffic-native_traffic,demand_bytes=s['dram_demand_bytes'],
                prefetch_bytes=s['dram_prefetch_bytes'],write_bytes=s['dram_write_bytes'],head_unique_cycles=s['head_unique_cycles_core_0'])
if __name__=='__main__':
    with ThreadPoolExecutor(5) as exe:rows=list(exe.map(run,[(mode,setting) for setting in ('open','capped') for mode in MODES]))
    with (HERE/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
