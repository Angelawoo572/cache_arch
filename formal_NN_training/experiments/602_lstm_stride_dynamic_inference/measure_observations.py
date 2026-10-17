#!/usr/bin/env python3
"""Supplement only missing replay timelines and frozen history/latency observations."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import time
import run as driver

EXP=Path(__file__).resolve().parent
CONFIG=driver.CONFIG
RAW=driver.RAW/'observation_study'

def specs():
    rows=[]
    def add(name,arm,h=0,budget=None,macs=0,capacity=0,protocol='historical',replay=None):
        rows.append(dict(run_id=name,arm=arm,h=h,hidden_size=h,checkpoint_tag=budget,
                         seed=7,macs=macs,state_capacity=capacity,protocol=protocol,phase='final',replay=replay))
    add('historical_none','none');add('historical_stride','stride')
    for h in (8,16):
        for budget in ('i1m','i20m'):
            tape=Path(CONFIG['checkpoints'])/('h'+str(h))/budget/'seed7/offline/offline_lstm.replay.csv'
            add(f'historical_replay_h{h}_{budget}','summer_replay',h,budget,replay=str(tape))
    tape=Path(CONFIG['checkpoints'])/'h8/i20m/seed7/offline/offline_stride.replay.csv'
    add('historical_stride_replay','stride_replay',replay=str(tape))
    add('cold_history_h8_i1m','frozen_i1m',8,'i1m',protocol='cold')
    for macs in (4,16):add(f'cold_history_h8_i1m_mac{macs}','frozen_i1m',8,'i1m',macs,64,'cold')
    return rows

def execute(row):
    p=RAW/row['run_id']
    if p.exists():
        prior=json.loads((p/'run.json').read_text())
        if prior.get('status')=='complete' and 'ChampSim completed all CPUs' in (p/'run.log').read_text():
            print('SKIP complete',row['run_id'],flush=True);return
        if prior.get('status')=='running' and driver.alive(prior.get('pid')):
            raise RuntimeError('already running '+str(p))
        backup=driver.RAW/'failed'/(row['run_id']+'-'+str(time.time_ns()))
        backup.parent.mkdir(parents=True,exist_ok=True);p.rename(backup)
    p.mkdir(parents=True)
    cold=row['protocol']=='cold'
    spec=dict(row,trace=CONFIG['trace'],skip_records=25000000 if cold else 0,
              simulation_instructions=25000000,warmup_instructions=0 if cold else 25000000,
              output_limit=32 if cold else 0,status='preparing')
    binary=json.loads((driver.RAW/'build.json').read_text())['binary']
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    env.update(DYNAMIC602_METHOD='frozen' if spec['arm'].startswith('frozen') else spec['arm'],
               DYNAMIC602_STATE_CAPACITY=str(spec['state_capacity']),DYNAMIC602_MACS=str(spec['macs']),
               DYNAMIC602_SKIP_RECORDS=str(spec['skip_records']),DYNAMIC602_MAX_RECORDS='25000000' if cold else '0',
               DYNAMIC602_OUTPUT_LIMIT=str(spec['output_limit']),DYNAMIC602_STATS=str(p/'snapshots.jsonl'))
    if spec['h']:
        point=Path(CONFIG['checkpoints'])/('h'+str(spec['h']))/spec['checkpoint_tag']/'seed7'
        spec['checkpoint']=str(point/'offline/model.pt');spec['model_binary']=str(point/'export/model.bin')
        env['STRIDE_LSTM_MODEL_BIN']=spec['model_binary']
    if spec['replay']:
        prefetcher='list_replayer';env['PFETCH_LIST_PATH']=spec['replay']
    else:
        prefetcher='dynamic602'
        if spec['h']:env['DYNAMIC602_HISTORY_LOG']=str(p/'history.jsonl')
    command=[binary,'--l2c_prefetcher_types='+prefetcher,'--stride_num_trackers=64','--stride_pref_degree=2',
             '--warmup_instructions='+str(spec['warmup_instructions']),'--simulation_instructions=25000000','-traces',CONFIG['trace']]
    spec['command']=command;driver.write(p/'run.json',spec)
    started=time.monotonic();print('START',spec['run_id'],flush=True)
    with (p/'run.log').open('w') as output:
        job=subprocess.Popen(['/usr/bin/time','-v','-o',str(p/'host_time.txt')]+command,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        spec.update(status='running',pid=job.pid);driver.write(p/'run.json',spec);code=job.wait()
    complete=code==0 and 'ChampSim completed all CPUs' in (p/'run.log').read_text()
    spec.update(status='complete' if complete else 'failed',exit_code=code,host_seconds=time.monotonic()-started)
    driver.write(p/'run.json',spec);print(spec['status'].upper(),spec['run_id'],round(spec['host_seconds'],2),flush=True)
    if not complete:raise RuntimeError(str(p))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['run','resume','status']);a=parser.parse_args()
    if a.action=='status':
        for p in sorted(RAW.glob('*/run.json')):
            r=json.loads(p.read_text());print(r['run_id'],r['status'])
        return
    errors=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(execute,row) for row in specs()]):
            try:future.result()
            except Exception as error:errors.append(str(error));print('ERROR',error,flush=True)
    if errors:raise RuntimeError('\n'.join(errors))

if __name__=='__main__':main()
