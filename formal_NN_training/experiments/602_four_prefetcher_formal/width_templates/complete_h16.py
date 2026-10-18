#!/usr/bin/env python3
"""Resume the missing H16 conditions using the unchanged certified simulator."""
import argparse, csv, gzip, json, os, platform, shutil, subprocess, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
C=json.loads((HERE/'complete_h16.json').read_text())
RAW=Path(C['raw'])

def canonical(value):
    if isinstance(value,str):
        for day in ('20261016','20261017'):
            value=value.replace('/cache/formal_study_'+day,'/cache/formal_study/archive/'+day)
        return value
    if isinstance(value,dict):return {k:canonical(v) for k,v in value.items()}
    if isinstance(value,list):return [canonical(v) for v in value]
    return value

def read(path):return canonical(json.loads(Path(path).read_text()))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def env():
    e={k:v for k,v in os.environ.items() if not k.startswith(('FOUR602_','DYNAMIC602_','STRIDE_LSTM_'))}
    e.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    for k,name in (('TMPDIR','tmp'),('PYTHONPYCACHEPREFIX','pycache'),('MPLCONFIGDIR','mpl')):
        p=RAW/'runtime'/name;p.mkdir(parents=True,exist_ok=True);e[k]=str(p)
    return e

def build():
    b=read(Path(C['reference_raw'])/'build.json')
    if b['base']!=C['base'] or b['compiler_flags']!='-O3 -ffp-contract=off':raise RuntimeError('different actual machine/build protocol')
    if Path(b['binary']).read_bytes()!=Path(b['binary_snapshot']).read_bytes():raise RuntimeError('changed actual executable')
    sim=Path(b['simulator'])
    for p in Path(b['compiled_source']).rglob('*'):
        if p.is_file() and p.read_bytes()!=(sim/p.relative_to(b['compiled_source'])).read_bytes():raise RuntimeError('changed compiled source '+str(p))
    return b

def bind(source,snapshot):
    source=Path(source);snapshot=Path(snapshot);snapshot.parent.mkdir(parents=True,exist_ok=True)
    if snapshot.exists():
        if source.read_bytes()!=snapshot.read_bytes():raise RuntimeError('frozen source/model changed '+str(source))
    else:shutil.copy2(source,snapshot)
    return dict(source=str(source),snapshot=str(snapshot),bytes=source.stat().st_size)

def frozen():
    state=read(RAW/'bindings.json');b=build()
    if state['protocol']!=C['protocol'] or state['build_id']!=b['build_id']:raise RuntimeError('wrong frozen run identity')
    for item in state['bindings']:
        if Path(item['source']).read_bytes()!=Path(item['snapshot']).read_bytes():raise RuntimeError('frozen model/config/parity changed '+item['source'])
    return state,b

def prepare(certificates):
    b=build();bindings=[]
    for source in (HERE/'complete_h16.py',HERE/'complete_h16.json'):
        bindings.append(bind(source,RAW/'protocol_source'/source.name))
    for policy,path in C['models'].items():
        bindings.append(bind(path,RAW/'models'/policy/'model.bin'))
    for policy,path in certificates.items():
        report=read(path)
        if report.get('status') not in ('PASS','verified'):raise RuntimeError('parity not passed '+policy)
        bindings.append(bind(path,RAW/'correctness_bound'/(policy+'.json')))
    if set(certificates)!=set(('stride','streamer')):raise RuntimeError('both new widths require actual C++ parity')
    for policy in ('ampm','spp'):
        template=Path(C['width_raw'])/'templates'/(policy+'_h16.txt')
        bindings.append(bind(template,RAW/'templates'/(policy+'_h16.txt')))
    for field in ('trace','seed','macs','page_policy','validation_records','test_records','warmup_instructions','phase_instructions'):
        if C[field]!=read(Path(C['reference_raw'])/'runs/test_r0_stride_stride_student_full_cap0/run.json')['spec']['contract'][field]:raise RuntimeError('reference measurement mismatch '+field)
    if read(Path(C['reference_raw'])/'selection.json')['common_cap']!=C['common_cap']:raise RuntimeError('cap must remain validation-frozen')
    write(RAW/'bindings.json',dict(protocol=C['protocol'],status='PASS',build_id=b['build_id'],host=platform.node(),bindings=bindings,certificates=certificates))
    write(RAW/'matrix.json',matrix());print('PREPARE PASS',flush=True)

def model(policy):return RAW/'models'/policy/'model.bin'
def template(policy):return RAW/'templates'/(policy+'_h16.txt')
def spec(stage,roi,policy,method='student',mode='full',cap=0):
    start,end=C['validation_records'] if stage=='validation' else C['test_records'][roi]
    return dict(run_id=f'{stage}_r{roi}_{policy}_h16_{method}_{mode}_cap{cap}',protocol=C['protocol'],stage=stage,roi=roi,policy=policy,hidden_size=16,method=method,mode=mode,budget=cap,seed=7,macs=0,page_policy=C['page_policy'],skip=start-C['warmup_instructions'],warmup=C['warmup_instructions'],instructions=end-start,measurement_records=[start,end])
def matrix():
    rows=[spec('validation',0,p,'stride_student' if p=='stride' else 'student') for p in ('stride','streamer')]
    for p in ('stride','streamer'):
        for roi in (0,1):
            for cap in (0,C['common_cap']):
                rows.append(spec('test',roi,p,'stride_student' if p=='stride' else 'student',cap=cap))
                rows.append(spec('test',roi,p,'template',cap=cap))
    for p in ('streamer','ampm','spp'):
        for roi in (0,1):
            for mode in ('fixed_payload','reset_state'):rows.append(spec('test',roi,p,mode=mode))
    return rows

def final(path):
    last=None
    with Path(path).open() as f:
        for line in f:
            x=json.loads(line)
            if x.get('event')=='snapshot' and x.get('final'):last=x
    if last is None:raise RuntimeError('missing complete final ROI snapshot')
    return last

def equivalent_template(s,b):
    if s['method']!='template' or s['policy']=='stride':return None
    p=s['policy'];path=Path(C['reference_raw'])/'templates'/(p+'.txt')
    if not path.exists() or path.read_bytes()!=template(p).read_bytes():return None
    sources=[Path(C['reference_raw'])/'runs'/f'test_r{s["roi"]}_{p}_template_full_cap{s["budget"]}',Path(C['width_raw'])/'runs'/f'test_r{s["roi"]}_{p}_h8_template_full_cap{s["budget"]}']
    for folder in sources:
        if not (folder/'run.json').exists():continue
        record=read(folder/'run.json');old=record['spec']
        if record['status']!='complete' or old['build_id']!=b['build_id']:continue
        if any(old[k]!=s[k] for k in ('stage','roi','policy','method','mode','budget','skip','warmup','instructions','measurement_records','page_policy')):continue
        if old.get('frozen_template')!=template(p).read_text():continue
        return dict(source_run=str(folder),original_run_id=old['run_id'],original_protocol=old['protocol'],reason='byte-identical frozen NN-free template, same actual executable/controls; not an extra simulated run')
    return None

def run_one(s):
    state,b=frozen();dest=RAW/'runs'/s['run_id'];expected=dict(s,build_id=b['build_id'],contract=C,trace=C['trace'])
    if s['method'] in ('student','stride_student'):expected['model']=str(model(s['policy']))
    if s['method']=='template' or s['mode']=='fixed_payload':expected['frozen_template']=template(s['policy']).read_text()
    if dest.exists():
        record=read(dest/'run.json')
        if record.get('status') in ('complete','reused') and record['spec']==expected:
            if record['status']=='complete':final(dest/'snapshots.jsonl')
            return
        if record.get('status')=='running':
            try:os.kill(record['pid'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError('existing active run '+s['run_id'])
        saved=RAW/'preserved_attempts'/(s['run_id']+'_'+str(time.time_ns()));saved.parent.mkdir(exist_ok=True);dest.rename(saved)
    dest.mkdir(parents=True)
    equivalent=equivalent_template(s,b)
    if equivalent:
        write(dest/'run.json',dict(status='reused',spec=expected,reuse=equivalent));print('REUSED',s['run_id'],flush=True);return
    e=env();p=s['policy'];stride=s['method']=='stride_student'
    e.update(FOUR602_POLICY=p,FOUR602_METHOD='monitor' if stride else s['method'],FOUR602_MODE=s['mode'],DYNAMIC602_METHOD='frozen' if stride else 'none',DYNAMIC602_MACS='0',DYNAMIC602_STATE_CAPACITY='64',DYNAMIC602_INPUT_QUEUE='16',DYNAMIC602_OUTPUT_QUEUE='32',DYNAMIC602_OUTPUT_LIMIT='32',DYNAMIC602_NONLINEAR_CYCLES='4',DYNAMIC602_BANDWIDTH_BYTES_PER_CYCLE='16',DYNAMIC602_PAGE_POLICY=s['page_policy'],DYNAMIC602_WARMUP_INSTRUCTIONS=str(s['warmup']),DYNAMIC602_HISTORY_INSTRUCTIONS=str(s['warmup']),DYNAMIC602_SKIP_RECORDS=str(s['skip']),DYNAMIC602_MAX_RECORDS=str(s['warmup']+s['instructions']+1024),DYNAMIC602_ROI_INSTRUCTIONS=str(s['instructions']),DYNAMIC602_PHASE_INSTRUCTIONS=str(C['phase_instructions']),DYNAMIC602_REQUESTS_PER_MILLION_INSTRUCTIONS=str(s['budget']),DYNAMIC602_FILTER_CURRENT='1' if p=='stride' else '0',DYNAMIC602_ARM=s['run_id'],DYNAMIC602_STATS=str(dest/'snapshots.jsonl'))
    if s['method'] in ('student','stride_student'):
        e['FOUR602_MODEL']=str(model(p));shutil.copy2(model(p),dest/'model.bin')
        if stride:e['STRIDE_LSTM_MODEL_BIN']=str(model(p))
        elif s['mode']=='full' and not s['budget']:e['FOUR602_INPUT_LOG']=str(dest/'callbacks.csv')
    if 'frozen_template' in expected:e['FOUR602_TEMPLATE']=str(template(p))
    l2=['--l2c_prefetcher_types=four602']+(['--l2c_prefetcher_types=dynamic602'] if stride else [])
    cmd=[b['binary']]+l2+['--streamer_num_trackers=64','--streamer_pref_degree=5','--ampm_pb_size=64','--ampm_pred_degree=4','--ampm_pref_degree=4','--ampm_enable_pref_buffer=false','--ampm_pref_buffer_size=256','--ampm_max_delta=16','--stride_num_trackers=64','--stride_pref_degree=2','--spp_dev2_fill_threshold=90','--spp_dev2_pf_threshold=40','--warmup_instructions='+str(s['warmup']),'--simulation_instructions='+str(s['instructions']),'-traces',C['trace']]
    record=dict(status='preparing',spec=expected,host=platform.node(),command=cmd,environment={k:v for k,v in e.items() if k.startswith(('FOUR602_','DYNAMIC602_','STRIDE_LSTM_'))})
    write(dest/'run.json',record);started=time.monotonic();print('START',s['run_id'],flush=True)
    with (dest/'run.log').open('w') as log:
        proc=subprocess.Popen(['/usr/bin/time','-v','-o',str(dest/'host_time.txt')]+cmd,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record.update(status='running',pid=proc.pid);write(dest/'run.json',record);code=proc.wait()
    record.update(exit_code=code,wall_seconds=time.monotonic()-started)
    try:
        if code or 'ChampSim completed all CPUs' not in (dest/'run.log').read_text():raise RuntimeError('failed/unfinished actual simulator')
        x=final(dest/'snapshots.jsonl')
        if abs(x['instructions']-s['instructions'])>4 or x.get('numerical_failures',0):raise RuntimeError('wrong full ROI or numerical failure')
        if s['method'] in ('student','stride_student') and (dest/'model.bin').read_bytes()!=model(p).read_bytes():raise RuntimeError('weights changed while running')
        if stride:
            if x.get('h')!=16 or x.get('nn_completed')!=x.get('eligible_l2_callbacks') or x.get('weight_bytes')!=20880:raise RuntimeError('wrong Stride H16 state/forward count')
        elif s['method']=='student':
            c=x['four602_counters']
            if x.get('four602_state_bytes')!=128 or c['completed_updates']!=c['demand_callbacks']+c.get('fill_callbacks',0):raise RuntimeError('wrong global H16 updates/state')
        if s['method']=='template' and x.get('four602_method')!='template':raise RuntimeError('NN-free arm mapped incorrectly')
        write(dest/'summary.json',x);path=dest/'callbacks.csv'
        if path.exists():
            with path.open('rb') as src,gzip.open(path.with_suffix('.csv.gz'),'wb') as dst:shutil.copyfileobj(src,dst)
            path.unlink()
        record['status']='complete'
    except Exception as error:
        record.update(status='failed',error=str(error));write(dest/'run.json',record);raise
    write(dest/'run.json',record);print('COMPLETE',s['run_id'],flush=True)

def run_many(rows):
    frozen();errors=[]
    with ThreadPoolExecutor(max_workers=C['jobs']) as pool:
        for f in as_completed([pool.submit(run_one,s) for s in rows]):
            try:f.result()
            except Exception as error:errors.append(str(error))
    if errors:raise RuntimeError('\n'.join(errors))

def select():
    frozen();selected={}
    for p in ('stride','streamer'):
        s=spec('validation',0,p,'stride_student' if p=='stride' else 'student');x=read(RAW/'runs'/s['run_id']/'summary.json');counts=Counter();slots={}
        source=x['mechanism_distributions_exact'] if p=='stride' else x['four602_distributions']
        for name,hist in source.items():
            suffix=name.split('_',2)[2]
            if suffix=='raw_K':counts.update({int(k):v for k,v in hist.items()})
            if p=='stride' and suffix.startswith('raw_delta_slot_'):
                slot=int(suffix[len('raw_delta_slot_'):]);slots.setdefault(slot,Counter()).update({(int(k),2):v for k,v in hist.items()})
            elif p!='stride' and suffix.startswith('raw_delta_fill_'):
                fill,slot=map(int,suffix[len('raw_delta_fill_'):].split('_slot_'));slots.setdefault(slot,Counter()).update({(int(k),fill):v for k,v in hist.items()})
        if not counts:raise RuntimeError('missing complete validation profile')
        modal=min(counts,key=lambda k:(-counts[k],k))
        if modal and not slots:raise RuntimeError('positive modal count missing deltas')
        payload=[list(min(slots[i],key=lambda k:(-slots[i][k],k))) for i in range(max(slots)+1)] if slots else [[0,2]]
        # Match the existing Stride deployment decoder cap without changing K profiling.
        deployed_modal=min(modal,32) if p=='stride' else modal
        text='modal_count {}\n'.format(deployed_modal)+''.join('{} {}\n'.format(*v) for v in payload)
        if template(p).exists() and template(p).read_text()!=text:raise RuntimeError('frozen validation selection changed')
        template(p).parent.mkdir(exist_ok=True);template(p).write_text(text)
        selected[p]=dict(source_run=s['run_id'],modal_count=modal,deployed_count=deployed_modal,slots=payload,selector='validation-only modal raw K; joint per-slot delta/fill, numeric ascending ties',current_line_filter=p=='stride')
    value=dict(protocol=C['protocol'],common_cap=C['common_cap'],cap_source=str(Path(C['reference_raw'])/'selection.json'),validation_records=C['validation_records'],templates=selected)
    if (RAW/'selection.json').exists() and read(RAW/'selection.json')!=value:raise RuntimeError('selection identity changed')
    write(RAW/'selection.json',value);print('VALIDATION CONTROLS FROZEN',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','validation','select','test','all','status','matrix']);parser.add_argument('--stride-certificate');parser.add_argument('--streamer-certificate');a=parser.parse_args()
    if a.command=='status':print(Counter(read(p)['status'] for p in (RAW/'runs').glob('*/run.json')));return
    if a.command=='matrix':print(json.dumps(matrix(),indent=2));return
    if a.command=='prepare':prepare(dict(stride=a.stride_certificate,streamer=a.streamer_certificate));return
    if a.command=='validation':run_many([s for s in matrix() if s['stage']=='validation']);return
    if a.command=='select':select();return
    if a.command=='test':run_many([s for s in matrix() if s['stage']=='test']);return
    write(RAW/'controller.json',dict(status='running',pid=os.getpid(),protocol=C['protocol']))
    try:
        run_many([s for s in matrix() if s['stage']=='validation']);select();run_many([s for s in matrix() if s['stage']=='test'])
        write(RAW/'controller.json',dict(status='complete',protocol=C['protocol']))
    except Exception as error:
        write(RAW/'controller.json',dict(status='failed',protocol=C['protocol'],error=str(error)));raise
if __name__=='__main__':main()
