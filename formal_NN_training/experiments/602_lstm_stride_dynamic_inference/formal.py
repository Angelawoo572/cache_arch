#!/usr/bin/env python3
"""Formal frozen L2 study. No pilot; every simulator row is validation or test.

Source/config equality and immutable build directories prevent stale output reuse.
No content hashes or old protocol results are used for completion decisions.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import gzip
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[2]
C = json.loads((EXP / 'formal_config.json').read_text())
RAW = Path(C['raw'])
LIVE = ROOT / 'formal_NN_training/experiments/602_lstm_stride_live_inference'
TRAIN = ROOT / 'formal_NN_training/experiments/602_offline_lstm_stride/python/train_and_offline_infer.py'
PY = C['python']
THREAD_ENV = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def source_files():
    paths = []
    for directory in (EXP/'runtime', EXP/'measurement', EXP/'patches', LIVE/'runtime'):
        paths += [p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    return sorted(paths)


def assert_build_current():
    b = json.loads((RAW/'build.json').read_text())
    if b['protocol']!=C['protocol'] or b['base']!=C['simulator_base']:
        raise RuntimeError('build protocol/dependency base changed')
    saved = Path(b['source_snapshot'])
    actual={str(p.relative_to(ROOT)) for p in source_files()}
    stored={str(p.relative_to(saved)) for p in saved.rglob('*') if p.is_file()}
    if actual!=stored: raise RuntimeError('build source file set changed')
    for p in source_files():
        q = saved / p.relative_to(ROOT)
        if not q.is_file() or q.read_bytes() != p.read_bytes():
            raise RuntimeError('source changed after build; build again before running: '+str(p))
    if Path(b['binary']).read_bytes()!=Path(b['binary_snapshot']).read_bytes():
        raise RuntimeError('built executable changed')
    return b


def build():
    destination = RAW/'builds'/('build_'+time.strftime('%Y%m%d_%H%M%S'))
    sim = destination/'simulator'
    sim.mkdir(parents=True, exist_ok=False)
    archive = subprocess.check_output(['git','-C',C['simulator'],'archive',C['simulator_base']])
    with tarfile.open(fileobj=io.BytesIO(archive)) as f:
        f.extractall(sim)
    subprocess.run(['patch','-p1','-i',str(EXP/'patches/inherited_cache.patch')],cwd=sim,check=True)
    shutil.copytree(Path(C['simulator'])/'libbf',sim/'libbf')
    (sim/'.dynamic602_isolated').write_text(C['protocol']+'\n')
    subprocess.run([PY,str(EXP/'runtime/prepare_forward.py'),str(sim)],check=True)
    subprocess.run([PY,str(EXP/'patches/install.py'),str(sim)],check=True)
    with (destination/'build.log').open('w') as log:
        subprocess.run(['bash','build_champsim.sh','no','dynamic602','no','1'],cwd=sim,
                       env=dict(os.environ,**THREAD_ENV),stdout=log,stderr=subprocess.STDOUT,check=True)
    binary=sim/'bin/perceptron-no-dynamic602-no-ship-1core'
    if not binary.is_file(): raise RuntimeError('missing binary')
    saved=destination/'source_snapshot'
    for p in source_files():
        q=saved/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    frozen_binary=destination/'simulator_binary_snapshot'
    shutil.copy2(binary,frozen_binary)
    write(RAW/'build.json',dict(binary=str(binary),simulator=str(sim),source_snapshot=str(saved),
                              binary_snapshot=str(frozen_binary),build_id=destination.name,
                              protocol=C['protocol'],base=C['simulator_base']))
    print('BUILD COMPLETE',binary,flush=True)


def spec(stage, workload='602.gcc_s-734B', roi=0, arm='none', seed=7, loss='balanced',
         macs=0, budget=0, silent=0, queue='fifo', history=None):
    if stage=='validation': start,end=C['validation_records'];warmup=4000000
    else: start,end=C['test_records'][roi];warmup=5000000
    history=C['state_history_instructions'] if history is None else history
    history=min(history,warmup)
    run_id='{}_{}_r{}_{}_s{}_{}_m{}_b{}_sil{}_q{}_hist{}'.format(
        stage,workload.split('.')[0],roi,arm,seed,loss,macs,budget,silent,queue,history)
    model=RAW/'models'/('seed'+str(seed))/loss/'export/model.bin' if arm=='nn' else None
    return dict(run_id=run_id,stage=stage,workload=workload,roi=roi,arm=arm,seed=seed,loss=loss,
                macs=macs,budget=budget,silent=silent,queue=queue,history=history,
                warmup=warmup,skip=start-warmup,instructions=end-start,
                trace=str(Path(C['trace_root'])/(workload+'.champsimtrace.xz')),
                model=str(model) if model else None,protocol=C['protocol'])


def final_snapshot(path):
    result=None
    with path.open() as f:
        for line in f:
            row=json.loads(line)
            if row.get('event')=='snapshot' and row.get('final'): result=row
    if result is None: raise RuntimeError('missing final ROI snapshot: '+str(path))
    return result


def run_one(s):
    b=assert_build_current()
    directory=RAW/s['stage']/s['run_id']
    expected=dict(s,build_id=b['build_id'],controls={k:C[k] for k in (
        'state_capacity','input_queue','output_queue','decoder_limit','nonlinear_cycles','page_policy')})
    if s['model']:
        meta=Path(s['model']).with_name('model_metadata.json')
        identity=json.loads(meta.read_text())
        if identity['seed']!=s['seed'] or identity['emit_loss']!=s['loss']:
            raise RuntimeError('checkpoint identity mismatch')
        if Path(s['model']).read_bytes()!=Path(s['model']).with_name('immutable_model.bin').read_bytes():
            raise RuntimeError('exported model changed after verification')
        expected['model_metadata']=identity
    if directory.exists():
        prior=json.loads((directory/'run.json').read_text())
        if prior.get('status')=='complete' and prior.get('spec')==expected:
            final_snapshot(directory/'snapshots.jsonl')
            print('COMPLETE EXISTS',s['run_id'],flush=True);return
        if prior.get('status')=='running':
            try: os.kill(prior['pid'],0)
            except ProcessLookupError: pass
            else: raise RuntimeError('active run still exists; it is not a completed result: '+s['run_id'])
        old=RAW/'preserved_attempts'/(s['run_id']+'_'+str(time.time_ns()))
        old.parent.mkdir(parents=True,exist_ok=True);directory.rename(old)
    directory.mkdir(parents=True)
    env=dict(os.environ,**THREAD_ENV)
    method='frozen' if s['arm']=='nn' else 'stride' if s['arm'].startswith('stride') else 'none'
    env.update(DYNAMIC602_METHOD=method,DYNAMIC602_MACS=str(s['macs']),
        DYNAMIC602_STATE_CAPACITY=str(C['state_capacity']),DYNAMIC602_INPUT_QUEUE=str(C['input_queue']),
        DYNAMIC602_OUTPUT_QUEUE=str(C['output_queue']),DYNAMIC602_OUTPUT_LIMIT=str(C['decoder_limit']),
        DYNAMIC602_NONLINEAR_CYCLES=str(C['nonlinear_cycles']),DYNAMIC602_PAGE_POLICY=C['page_policy'],
        DYNAMIC602_QUEUE_POLICY=s['queue'],DYNAMIC602_WARMUP_INSTRUCTIONS=str(s['warmup']),
        DYNAMIC602_HISTORY_INSTRUCTIONS=str(s['history']),DYNAMIC602_SILENT_SUPPRESS=str(s['silent']),
        DYNAMIC602_REQUESTS_PER_MILLION_INSTRUCTIONS=str(s['budget']),DYNAMIC602_KEEP_PPM='1000000',
        DYNAMIC602_SKIP_RECORDS=str(s['skip']),DYNAMIC602_MAX_RECORDS=str(s['warmup']+s['instructions']+1024),
        DYNAMIC602_STATS=str(directory/'snapshots.jsonl'),DYNAMIC602_HISTORY_LOG=str(directory/'events.jsonl'),
        DYNAMIC602_HISTORY_LOG_LIMIT='4096',DYNAMIC602_TRACE_NAME=s['workload'])
    if s['arm'].startswith('stride_'): env['DYNAMIC602_STRIDE_MODE']=s['arm'][7:]
    if s['model']: env['STRIDE_LSTM_MODEL_BIN']=s['model']
    if s['stage']=='validation' and s['arm']=='none': env['DYNAMIC602_INPUT_LOG']=str(directory/'inputs.csv')
    l2=['--l2c_prefetcher_types=dynamic602']+(['--l2c_prefetcher_types=spp_dev2'] if s['arm']=='spp' else [])
    cmd=[b['binary']]+l2+['--stride_num_trackers=64','--stride_pref_degree=2',
         '--warmup_instructions='+str(s['warmup']),'--simulation_instructions='+str(s['instructions']),'-traces',s['trace']]
    record=dict(spec=expected,status='preparing',environment={k:v for k,v in env.items() if k.startswith(('DYNAMIC602_','STRIDE_LSTM_'))},command=cmd)
    write(directory/'run.json',record)
    print('START',s['run_id'],flush=True);start=time.monotonic()
    with (directory/'run.log').open('w') as log:
        proc=subprocess.Popen(['/usr/bin/time','-v','-o',str(directory/'host_time.txt')]+cmd,
                              stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        record.update(status='running',pid=proc.pid);write(directory/'run.json',record);code=proc.wait()
    record.update(exit_code=code,wall_seconds=time.monotonic()-start)
    try:
        if code: raise RuntimeError('simulator exit '+str(code))
        if 'ChampSim completed all CPUs' not in (directory/'run.log').read_text(): raise RuntimeError('incomplete log')
        final=final_snapshot(directory/'snapshots.jsonl')
        if abs(final['instructions']-s['instructions'])>4: raise RuntimeError('ROI count mismatch')
        if final.get('numerical_failures',0): raise RuntimeError('numerical inference failure')
        record['status']='complete';write(directory/'summary.json',final)
    except Exception as e:
        record.update(status='failed',error=str(e));write(directory/'run.json',record);raise
    write(directory/'run.json',record)
    print('COMPLETE',s['run_id'],round(record['wall_seconds'],2),flush=True)


def run_many(rows):
    errors=[]
    with ThreadPoolExecutor(max_workers=C['jobs']) as pool:
        for f in as_completed([pool.submit(run_one,s) for s in rows]):
            try:f.result()
            except Exception as e: errors.append(str(e));print('FAILED',e,flush=True)
    if errors: raise RuntimeError('\n'.join(errors))


def validation_reference():
    rows=[spec('validation',arm=a) for a in ('none','stride_legacy','stride_remove_current','stride_forward1','stride_forward2','spp')]
    rows += [spec('validation',workload=w,arm='stride_forward1') for w in C['cross_workloads']]
    return rows


def train():
    manifest=json.loads(Path(C['train_stream']).with_name('training_manifest.json').read_text())
    if (C['train_records']!=[0,1000000] or manifest.get('training_warmup_instructions')!=0
        or manifest.get('training_simulation_instructions')!=1000000
        or manifest.get('collection_semantics')!='trace_start_to_exact_retired_instruction_budget'
        or manifest.get('trace')!='602.gcc_s-734B' or manifest.get('status')!='trainable'):
        raise RuntimeError('training prefix provenance does not establish the declared split')
    with gzip.open(C['train_stream'],'rt') as stream:
        if sum(1 for _ in csv.DictReader(stream))!=manifest['decision_rows']:
            raise RuntimeError('training prefix row count changed')
    assert C['train_records'][1]<=C['validation_records'][0]<C['validation_records'][1]<=C['test_records'][0][0]
    assert C['test_records'][0][1]<C['test_records'][1][0]
    write(RAW/'training_split.json',{k:v for k,v in manifest.items() if 'sha256' not in k})
    val=RAW/'validation'/spec('validation')['run_id']/'inputs.csv'
    if not val.is_file(): raise RuntimeError('formal NoPF validation stream required')
    compressed=val.with_suffix('.csv.gz')
    if not compressed.is_file() or gzip.open(compressed,'rb').read()!=val.read_bytes():
        with val.open('rb') as source, gzip.open(compressed,'wb') as target:
            shutil.copyfileobj(source,target)
    val=compressed
    frozen=RAW/'models/training_source_snapshot'
    train_sources=[TRAIN]+sorted((ROOT/'formal_NN_training/common').glob('*.py'))+[LIVE/'python/export_live_model.py',LIVE/'python/live_model_format.py']
    for source in train_sources:
        target=frozen/source.relative_to(ROOT)
        if target.exists() and target.read_bytes()!=source.read_bytes():
            raise RuntimeError('training source changed; use a new immutable run root')
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    val_snapshot=RAW/'models/validation_inputs.csv.gz'
    if val_snapshot.exists() and gzip.open(val_snapshot,'rb').read()!=gzip.open(val,'rb').read():
        raise RuntimeError('validation stream changed; use a new immutable run root')
    if not val_snapshot.exists(): shutil.copy2(val,val_snapshot)
    train_snapshot=RAW/'models/train_inputs.csv.gz'
    if train_snapshot.exists() and gzip.open(train_snapshot,'rb').read()!=gzip.open(C['train_stream'],'rb').read():
        raise RuntimeError('training stream changed; use a new immutable run root')
    if not train_snapshot.exists(): shutil.copy2(C['train_stream'],train_snapshot)
    models=[]
    for seed in C['seeds']:
        for loss in ('balanced','unweighted'):
            base=RAW/'models'/('seed'+str(seed))/loss
            offline=base/'offline';export=base/'export'
            checkpoint=offline/'model.pt'
            train_spec=dict(seed=seed,loss=loss,hidden=C['hidden_size'],epochs=C['epochs'],
                chunk_len=256,pc_batch_size=128,learning_rate=.002,train_records=C['train_records'],
                validation_records=C['validation_records'],protocol=C['protocol'],device='cpu')
            identity_path=base/'training_spec.json'
            if identity_path.exists() and json.loads(identity_path.read_text())!=train_spec:
                raise RuntimeError('training controls changed; refuse stale model reuse')
            if not identity_path.exists(): write(identity_path,train_spec)
            if not checkpoint.is_file():
                offline.mkdir(parents=True,exist_ok=True)
                cmd=[PY,str(TRAIN),'--train-stream',C['train_stream'],'--eval-stream',str(val),
                    '--out-dir',str(offline),'--seed',str(seed),'--trace','602.gcc_s-734B','--emit-loss',loss,
                    '--epochs',str(C['epochs']),'--chunk-len','256','--pc-batch-size','128',
                    '--learning-rate','0.002','--hidden-size',str(C['hidden_size']),'--device','cpu','--omit-content-fingerprints']
                with (base/'training.log').open('w') as log:
                    subprocess.run(cmd,env=dict(os.environ,**THREAD_ENV),stdout=log,stderr=subprocess.STDOUT,check=True)
            if not (export/'model.bin').is_file():
                subprocess.run([PY,str(LIVE/'python/export_live_model.py'),'--checkpoint',str(checkpoint),
                    '--run-metadata',str(offline/'run_metadata.json'),'--out-dir',str(export),
                    '--hidden-size',str(C['hidden_size']),'--instruction-budget','1000000',
                    '--seed',str(seed),'--emit-loss',loss],env=dict(os.environ,**THREAD_ENV),check=True)
            immutable=export/'immutable_model.bin'
            if immutable.exists() and immutable.read_bytes()!=(export/'model.bin').read_bytes():
                raise RuntimeError('export changed after original verification')
            if not immutable.exists(): shutil.copy2(export/'model.bin',immutable)
            models.append(str(checkpoint));print('TRAINED',seed,loss,flush=True)
    subprocess.run([PY,str(LIVE/'validation/validate_formal_pairs.py'),'--checkpoints']+models+
                   ['--output',str(RAW/'training_pairs.json')],check=True)


def select_budget():
    # A single common cumulative cap is chosen ONLY from validation forward1.
    caps={};evidence={}
    for workload in ['602.gcc_s-734B']+C['cross_workloads']:
        r=spec('validation',workload=workload,arm='stride_forward1')
        v=final_snapshot(RAW/'validation'/r['run_id']/'snapshots.jsonl')
        requests=v.get('P',0)
        if requests<=0: raise RuntimeError('validation reference has no requests')
        caps[workload]=int(requests*1000000/v['instructions'])
        evidence[workload]=dict(source_run=r['run_id'],requests=requests,instructions=v['instructions'])
    result=dict(protocol=C['protocol'],selection='floor(workload validation forward1 requests / retired instructions * 1e6)',
                requests_per_million_instructions=caps,evidence=evidence)
    write(RAW/'budget_selection.json',result);return caps


def parity():
    binary=RAW/'parity/stride_lstm_standalone'
    subprocess.run(['bash',str(LIVE/'linux/build_standalone_runtime.sh')],
                   env=dict(os.environ,OUT=str(binary),FORCE='1',**THREAD_ENV),check=True)
    for seed in C['seeds']:
        for loss in ('balanced','unweighted'):
            base=RAW/'models'/('seed'+str(seed))/loss
            destination=RAW/'parity'/('seed'+str(seed))/loss
            subprocess.run([PY,str(LIVE/'validation/compare_python_cpp_outputs.py'),
                '--cpp-runner',str(binary),'--work-dir',str(destination),
                '--model-bin',str(base/'export/model.bin'),'--checkpoint',str(base/'offline/model.pt'),
                '--stream',str(RAW/'models/validation_inputs.csv.gz'),'--max-events','2000',
                '--output',str(destination/'parity.json')],env=dict(os.environ,**THREAD_ENV),check=True)


def matrix(caps):
    cap=caps['602.gcc_s-734B']
    rows=[]
    baselines=('none','stride_legacy','stride_remove_current','stride_forward1','stride_forward2','spp')
    rows += [spec('test',arm=a) for a in baselines]
    rows += [spec('test',arm=a,budget=cap) for a in baselines if a!='none']
    for seed in C['seeds']:
        for loss in ('balanced','unweighted'):
            for macs in (0,4):
                for budget in (0,cap):rows.append(spec('test',arm='nn',seed=seed,loss=loss,macs=macs,budget=budget))
        for macs in (0,4):rows.append(spec('test',arm='nn',seed=seed,macs=macs,silent=1))
        rows.append(spec('test',arm='nn',seed=seed,macs=4,queue='drop_oldest'))
        for history in (0,100000):rows.append(spec('test',arm='nn',seed=seed,macs=4,history=history))
    rows += [spec('test',roi=1,arm=a) for a in ('none','stride_legacy','stride_forward1','stride_forward2','spp')]
    rows += [spec('test',roi=1,arm=a,budget=cap) for a in ('stride_forward1','spp')]
    for seed in C['seeds']:
        for macs in (0,4):rows.append(spec('test',roi=1,arm='nn',seed=seed,macs=macs))
        rows.append(spec('test',roi=1,arm='nn',seed=seed,macs=4,budget=cap))
        rows.append(spec('test',roi=1,arm='nn',seed=seed,macs=4,silent=1))
    for history in (0,100000):rows.append(spec('test',roi=1,arm='nn',macs=4,history=history))
    for workload in C['cross_workloads']:
        rows += [spec('test',workload=workload,arm=a) for a in ('none','stride_legacy','stride_forward2','spp')]
        rows += [spec('test',workload=workload,arm=a,budget=caps[workload]) for a in ('stride_forward2','spp')]
        for macs in (0,4):rows.append(spec('test',workload=workload,arm='nn',macs=macs))
        rows.append(spec('test',workload=workload,arm='nn',macs=4,budget=caps[workload]))
    assert len(rows)==len({r['run_id'] for r in rows})
    return rows


def status():
    counts={}
    for p in sorted(RAW.glob('*/*/run.json')):
        r=json.loads(p.read_text());k=r['status'];counts[k]=counts.get(k,0)+1
        if k!='complete':print(p.parent.name,k,r.get('error',''),flush=True)
    print(json.dumps(counts),flush=True)


def pipeline():
    RAW.mkdir(parents=True,exist_ok=True)
    lock=(RAW/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert_build_current()
    run_many(validation_reference())
    train()
    parity()
    run_many([spec('validation',arm='nn',loss=l,macs=m) for l in ('balanced','unweighted') for m in (0,4)])
    cap=select_budget()
    rows=matrix(cap);write(RAW/'formal_matrix.json',rows)
    run_many(rows)
    write(RAW/'pipeline_result.json',dict(status='complete',test_rows=len(rows),validation_rows=12,
          training_models=6,protocol=C['protocol']))
    print('ALL FORMAL ROWS COMPLETE',len(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['build','pipeline','status','validation','train','test'])
    a=p.parse_args().action
    {'build':build,'pipeline':pipeline,'status':status,'validation':lambda:run_many(validation_reference()),
     'train':train,'test':lambda:run_many(matrix(json.loads((RAW/'budget_selection.json').read_text())['requests_per_million_instructions']))}[a]()
