"""Run shadow and seven fixed-work, degree-one arms for the later BFS3 region."""
import argparse
import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
DATA=STUDY/'raw/page_memory'
BASE=STUDY/'raw/runs'
MODES=('shadow','next','repeat','current','offset','link','offset16','link16')

def run(mode):
    meta=json.loads((BASE/'gap_bfs3_late-stride/run.json').read_text())
    out=DATA/'runs'/mode
    if (out/'complete.json').exists():return
    out.mkdir(exist_ok=True,parents=True)
    exe=DATA/'build/native/pythia/bin/champsim'
    trace=STUDY/'raw/traces'/(meta['traces'][0]+'.champsimtrace')
    env={k:v for k,v in os.environ.items() if not k.startswith(('PAGE_','HYBRID_','LLC_PF_'))}
    env.update(meta['environment'],HYBRID_OUT=str(out),PAGE_MODE=mode,PAGE_TABLE=str(DATA/'prefix.txt'))
    cmds=[str(exe),f"--warmup_instructions={meta['warmup']}",f"--simulation_instructions={meta['roi']}",'-traces',str(trace)]
    with (out/'stdout.txt').open('w') as f:
        subprocess.run(cmds,cwd=out,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    with (out/'cores.csv').open() as f:rows=list(csv.DictReader(f))
    if len(rows)!=1 or int(rows[0]['instructions'])!=meta['roi']:raise ValueError('work mismatch')
    (out/'complete.json').write_text(json.dumps(dict(mode=mode,region=meta['case'],roi=meta['roi']))+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--jobs',type=int,default=4)
    ap.add_argument('--modes',nargs='+',choices=MODES,default=MODES)
    args=ap.parse_args()
    with ThreadPoolExecutor(args.jobs) as ex:list(ex.map(run,args.modes))
