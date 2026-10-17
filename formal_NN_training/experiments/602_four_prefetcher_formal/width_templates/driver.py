#!/usr/bin/env python3
"""Separate frozen H16 + H8 capped-template formal supplement; no simulator pilot."""
import argparse,csv,gzip,importlib.util,json,math,os,platform,shutil,subprocess,time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PARENT=HERE.parent
C=json.loads((HERE/'config.json').read_text());RAW=Path(C['raw']);RUNTIME=RAW/'runtime'
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def env():
    value={k:v for k,v in os.environ.items() if not k.startswith(('FOUR602_','DYNAMIC602_','STRIDE_LSTM_'))}
    value.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    for k,name in [('TMPDIR','tmp'),('PYTHONPYCACHEPREFIX','pycache'),('MPLCONFIGDIR','mpl')]:
        p=RUNTIME/name;p.mkdir(parents=True,exist_ok=True);value[k]=str(p)
    return value
def bind(source,snapshot):
    source=Path(source);snapshot=Path(snapshot);snapshot.parent.mkdir(parents=True,exist_ok=True)
    if not source.is_file():raise RuntimeError('missing binding source '+str(source))
    if snapshot.exists():
        if source.read_bytes()!=snapshot.read_bytes():raise RuntimeError('frozen binding changed '+str(source))
    else:shutil.copy2(source,snapshot)
    return dict(source=str(source.resolve()),snapshot=str(snapshot.resolve()),bytes=source.stat().st_size)
def verify(bindings):
    for b in bindings:
        if Path(b['source']).read_bytes()!=Path(b['snapshot']).read_bytes():raise RuntimeError('bound bytes changed '+b['source'])
def model_key(policy,h):return policy+'_h'+str(h)
def model_path(policy,h=16):return RAW/'models'/model_key(policy,h)/'model.bin'
def template_path(policy,h):return RAW/'templates'/(model_key(policy,h)+'.txt')
def checkpoint(policy):return Path(C['model_root'])/C['models'][policy]['checkpoint']
def header(path):
    import sys;sys.path.insert(0,str(ROOT))
    from formal_NN_training.common.live602_runtime.model_format import read_model
    return read_model(path)[0]
def actual_build():
    b=read(Path(C['reference_raw'])/'build.json')
    if b['protocol']!='four602_frozen_live_20261017_v2' or b['base']!=C['base'] or b['compiler_flags']!='-O3 -ffp-contract=off':raise RuntimeError('incompatible archived actual build')
    if Path(b['binary']).read_bytes()!=Path(b['binary_snapshot']).read_bytes():raise RuntimeError('archived binary changed')
    sim=Path(b['simulator']);saved=Path(b['compiled_source'])
    for p in saved.rglob('*'):
        if p.is_file() and (sim/p.relative_to(saved)).read_bytes()!=p.read_bytes():raise RuntimeError('actual compiled source changed '+str(p))
    return b
def protocol_sources():return [HERE/'config.json',HERE/'driver.py',HERE/'ampm_h16.py']
def frozen():
    state=read(RAW/'bindings.json')
    if state['protocol']!=C['protocol']:raise RuntimeError('wrong supplemental protocol')
    verify(state['bindings']);b=actual_build()
    if state['build_id']!=b['build_id']:raise RuntimeError('actual build identity changed')
    return state,b
def execute(cmd,log):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as handle:subprocess.run([str(v) for v in cmd],env=env(),cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
def prepare():
    RAW.mkdir(parents=True,exist_ok=True);b=actual_build();bindings=[]
    old=read(Path(C['reference_source'])/'formal_NN_training/experiments/602_four_prefetcher_formal/config.json')
    for key in ('trace','seed','macs','page_policy','validation_records','test_records','warmup_instructions','phase_instructions'):
        if C[key]!=old[key]:raise RuntimeError('reference control mismatch '+key)
    audit_path=Path(C['reference_raw'])/'analysis_final/audit.json'
    audit=read(audit_path)
    if audit.get('status')!='PASS' or audit.get('complete_count')!=64 or audit.get('failure_count')!=0:raise RuntimeError('archived strict reference audit not PASS64/0')
    bindings.append(bind(audit_path,RAW/'reference_snapshot/audit.json'))
    old_selection=read(Path(C['reference_raw'])/'selection.json')
    if old_selection['common_cap']!=C['common_cap']:raise RuntimeError('fixed reference cap mismatch')
    bindings.append(bind(Path(C['reference_raw'])/'selection.json',RAW/'reference_snapshot/selection.json'))
    for p in protocol_sources():bindings.append(bind(p,RAW/'protocol_source'/p.name))
    cpp=Path(b['simulator'])/'prefetcher/live602_runtime.cc';h=Path(b['simulator'])/'inc/live602_runtime.h'
    for p in (cpp,h,ROOT/'formal_NN_training/common/live602_runtime/probe.cc',ROOT/'formal_NN_training/common/live602_runtime/model_format.py',
              ROOT/'formal_NN_training/experiments/602_lstm_ampm_live_inference/python/export_model.py',
              ROOT/'formal_NN_training/experiments/602_lstm_spp_live_inference/python/export_and_validate.py',
              ROOT/'formal_NN_training/experiments/602_offline_lstm_spp/python/train_and_offline_infer.py',
              ROOT/'formal_NN_training/common/threshold_free_policy.py'):
        bindings.append(bind(p,RAW/'source_bindings'/p.name))
    bindings.append(bind(Path(b['binary']),RAW/'actual_binary_snapshot'))
    references=[]
    for path in sorted((Path(C['reference_raw'])/'runs').glob('*/run.json')):
        record=read(path)
        if record.get('status')!='complete':continue
        s=record['spec'];summary=path.with_name('summary.json')
        if s['protocol']!=old['protocol'] or s.get('build_id')!=b['build_id']:raise RuntimeError('reference protocol/build mismatch '+path.parent.name)
        for p in (path,summary):bindings.append(bind(p,RAW/'reference_snapshot'/path.parent.name/p.name))
        fields={k:s.get(k) for k in ('stage','roi','policy','method','mode','budget','seed','skip','warmup','instructions','measurement_records','page_policy','build_id')}
        references.append(dict(old_run_id=path.parent.name,raw_run_directory=str(path.parent.resolve()),original_protocol=s['protocol'],fields=fields,compatibility='same actual binary and common machine/trace/cache/ROI/page/native-knob controls; new supplemental protocol remains distinct',machine_evidence='same configured Sacramento host; old per-run records have no separate hardware identity telemetry'))
    if len(references)!=64:raise RuntimeError('expected 64 complete archived reference rows')
    for policy in ('streamer','ampm','spp'):
        old_model=Path(C['reference_raw'])/'models'/policy/'model.bin'
        if old_model.read_bytes()!=old_model.with_name('immutable_model.bin').read_bytes():raise RuntimeError('archived H8 model changed '+policy)
        bindings.append(bind(old_model,RAW/'reference_models'/model_key(policy,8)/'model.bin'))
        original=Path(C['reference_raw'])/'templates'/(policy+'.txt')
        bindings.append(bind(original,template_path(policy,8)))
    for policy in ('ampm','spp'):
        cp=checkpoint(policy);key=model_key(policy,16);bindings.append(bind(cp,RAW/'checkpoints'/key/'model.pt'))
        meta=cp.with_name('run_metadata.json');training_metadata=read(meta)
        if training_metadata.get('seed')!=7 or training_metadata.get('matched_normal_prefetcher')!=policy or training_metadata.get('parameter_count')!=C['models'][policy]['parameters']:raise RuntimeError('retained H16 training metadata mismatch '+key)
        if policy=='spp':
            for field,value in dict(model_size=16,model_family='lstm',runtime_feature_count=65,model_revision='compact_empirical_prior_hurdle_autoregressive_gmm_fill_v2',decoder_mixture_components=4).items():
                if training_metadata.get(field)!=value:raise RuntimeError('SPP H16 metadata mismatch '+field)
        bindings.append(bind(meta,RAW/'checkpoints'/key/'run_metadata.json'))
        target=model_path(policy);target.parent.mkdir(parents=True,exist_ok=True)
        if policy=='ampm':
            command=[C['python'],HERE/'ampm_h16.py','export','--checkpoint',cp,'--metadata',meta,'--model',target]
        else:command=[C['python'],ROOT/'formal_NN_training/experiments/602_lstm_spp_live_inference/python/export_and_validate.py','--checkpoint',cp,'--model',target]
        if target.exists():
            if not target.with_name('immutable_model.bin').exists() or target.read_bytes()!=target.with_name('immutable_model.bin').read_bytes():raise RuntimeError('unsafe prior model state '+key)
        else:execute(command,RAW/'prepare_logs'/(key+'.log'))
        cfg=header(target)
        if cfg['hidden_size']!=16 or cfg['parameter_count']!=C['models'][policy]['parameters']:raise RuntimeError('not expected actual H16 weights '+key)
        if policy=='spp' and (cfg['feature_count'],cfg['kind_feature'],cfg['fill_classes'],cfg['count_mode'])!=(65,1,2,1):raise RuntimeError('wrong SPP schema')
        if policy=='ampm' and (cfg['feature_count'],cfg['kind_feature'],cfg['fill_classes'],cfg['count_mode'])!=(64,0,0,0):raise RuntimeError('wrong AMPM schema')
        bindings.append(bind(target,target.with_name('immutable_model.bin')))
    write(RAW/'bindings.json',dict(status='prepared',protocol=C['protocol'],build_id=b['build_id'],actual_build=b,host=platform.node(),bindings=bindings))
    write(RAW/'reference_bindings.json',dict(protocol=C['protocol'],reference_protocol=old['protocol'],common_controls={k:C[k] for k in ('trace','seed','macs','page_policy','validation_records','test_records','warmup_instructions','phase_instructions','common_cap')},actual_build_id=b['build_id'],actual_binary_byte_equal=True,references=references,scope='explicit reference join only; reference rows not relabeled as new runs'))
    write(RAW/'matrix.json',matrix());print('PREPARE PASS',flush=True)
def certify():
    state,b=frozen();cert=RAW/'correctness';cert.mkdir(exist_ok=True);sim=Path(b['simulator']);probe=cert/'probe'
    execute(['g++','-std=c++11','-O3','-ffp-contract=off','-I'+str(sim/'inc'),sim/'prefetcher/live602_runtime.cc',ROOT/'formal_NN_training/common/live602_runtime/probe.cc','-o',probe],cert/'compile.log')
    reports={}
    for policy in ('ampm','spp'):
        cp=checkpoint(policy)
        if policy=='ampm':
            stream=cp.parents[2]/'colab_input/602.gcc_s-734B.guard_stream.csv.gz'
            execute([C['python'],HERE/'ampm_h16.py','parity','--checkpoint',cp,'--model',model_path(policy),'--probe',probe,'--stream',stream,'--out',cert/'ampm_h16','--events','4096'],cert/'ampm_h16.log')
            reports[policy]=read(cert/'ampm_h16/parity.json')
        else:
            # Export to a separate candidate; never overwrite frozen formal weights.
            candidate=cert/'spp_h16/model.bin';candidate.parent.mkdir(exist_ok=True)
            stream=cp.parents[2]/'colab_input/602.gcc_s-734B.spp.train_stream.csv.gz'
            execute([C['python'],ROOT/'formal_NN_training/experiments/602_lstm_spp_live_inference/python/export_and_validate.py','--checkpoint',cp,'--model',candidate,'--probe',probe,'--stream',stream,'--events','4096','--report',cert/'spp_h16/parity.json'],cert/'spp_h16.log')
            if candidate.read_bytes()!=model_path(policy).read_bytes():raise RuntimeError('SPP certified export differs from formal H16 model')
            reports[policy]=read(cert/'spp_h16/parity.json')
    if any(v.get('status') not in ('PASS','verified') for v in reports.values()):raise RuntimeError('H16 parity failed')
    frozen();certificate=dict(status='PASS',protocol=C['protocol'],build_id=b['build_id'],reports=reports,probe=str(probe),scope='actual compiled generic runtime; real H16 checkpoint states/counts/all actions on causal correctness fixtures and boundaries; no simulator pilot')
    write(RAW/'correctness_gate.json',certificate);print('H16 CORRECTNESS PASS',flush=True)
def require_gate():
    state,b=frozen();gate=read(RAW/'correctness_gate.json')
    if gate['status']!='PASS' or gate['protocol']!=C['protocol'] or gate['build_id']!=b['build_id']:raise RuntimeError('H16 gate not passed for actual build')
def spec(stage,roi,policy,h,method,budget=0):
    begin,end=C['validation_records'] if stage=='validation' else C['test_records'][roi]
    rid='{}_r{}_{}_h{}_{}_full_cap{}'.format(stage,roi,policy,h,method,budget)
    model=model_path(policy,h) if h==16 else RAW/'reference_models'/model_key(policy,h)/'model.bin'
    return dict(run_id=rid,protocol=C['protocol'],stage=stage,roi=roi,policy=policy,hidden_size=h,method=method,mode='full',budget=budget,seed=7,macs=0,page_policy=C['page_policy'],skip=begin-C['warmup_instructions'],warmup=C['warmup_instructions'],instructions=end-begin,measurement_records=[begin,end],model_provenance=str(model),template_source=str(template_path(policy,h)) if method=='template' else None)
def matrix():
    rows=[spec('validation',0,p,16,'student') for p in ('ampm','spp')]
    rows += [spec('test',r,p,16,m,cap) for p in ('ampm','spp') for r in (0,1) for m in ('student','template') for cap in (0,C['common_cap'])]
    rows += [spec('test',r,p,8,'template',C['common_cap']) for p in ('streamer','ampm','spp') for r in (0,1)]
    return rows
def final_snapshot(path):
    last=None
    with Path(path).open() as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('event')=='snapshot' and row.get('final'):last=row
    if last is None:raise RuntimeError('no final snapshot')
    return last
def run_one(s):
    state,b=frozen();dest=RAW/'runs'/s['run_id'];expected=dict(s,build_id=b['build_id'],contract=C)
    model=model_path(s['policy']) if s['method']=='student' else None
    if model:expected['model']=str(model)
    template=Path(s['template_source']) if s['template_source'] else None
    if template:expected['frozen_template']=template.read_text()
    if dest.exists():
        old=read(dest/'run.json')
        if old.get('status')=='complete' and old.get('spec')==expected:
            if model and model.read_bytes()!=(dest/'model.bin').read_bytes():raise RuntimeError('completed model changed')
            final_snapshot(dest/'snapshots.jsonl');return
        if old.get('status')=='running':
            try:os.kill(old['pid'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError('existing active run '+s['run_id'])
        saved=RAW/'preserved_attempts'/(s['run_id']+'_'+str(time.time_ns()));saved.parent.mkdir(exist_ok=True);dest.rename(saved)
    dest.mkdir(parents=True);e=env()
    e.update(FOUR602_POLICY=s['policy'],FOUR602_METHOD=s['method'],FOUR602_MODE='full',DYNAMIC602_METHOD='none',DYNAMIC602_MACS='0',
        DYNAMIC602_STATE_CAPACITY='64',DYNAMIC602_INPUT_QUEUE='16',DYNAMIC602_OUTPUT_QUEUE='32',DYNAMIC602_OUTPUT_LIMIT='32',
        DYNAMIC602_NONLINEAR_CYCLES='4',DYNAMIC602_BANDWIDTH_BYTES_PER_CYCLE='16',DYNAMIC602_PAGE_POLICY=s['page_policy'],
        DYNAMIC602_WARMUP_INSTRUCTIONS=str(s['warmup']),DYNAMIC602_HISTORY_INSTRUCTIONS=str(s['warmup']),
        DYNAMIC602_SKIP_RECORDS=str(s['skip']),DYNAMIC602_MAX_RECORDS=str(s['warmup']+s['instructions']+1024),
        DYNAMIC602_ROI_INSTRUCTIONS=str(s['instructions']),DYNAMIC602_PHASE_INSTRUCTIONS=str(C['phase_instructions']),
        DYNAMIC602_REQUESTS_PER_MILLION_INSTRUCTIONS=str(s['budget']),DYNAMIC602_FILTER_CURRENT='0',
        DYNAMIC602_ARM=s['run_id'],DYNAMIC602_STATS=str(dest/'snapshots.jsonl'))
    if model:
        e['FOUR602_MODEL']=str(model);shutil.copy2(model,dest/'model.bin')
        if not s['budget']:e['FOUR602_INPUT_LOG']=str(dest/'callbacks.csv')
    if template:e['FOUR602_TEMPLATE']=str(template)
    cmd=[b['binary'],'--l2c_prefetcher_types=four602','--streamer_num_trackers=64','--streamer_pref_degree=5','--ampm_pb_size=64','--ampm_pred_degree=4','--ampm_pref_degree=4','--ampm_enable_pref_buffer=false','--ampm_pref_buffer_size=256','--ampm_max_delta=16','--stride_num_trackers=64','--stride_pref_degree=2','--spp_dev2_fill_threshold=90','--spp_dev2_pf_threshold=40','--warmup_instructions='+str(s['warmup']),'--simulation_instructions='+str(s['instructions']),'-traces',C['trace']]
    record=dict(status='preparing',spec=expected,command=cmd,host=platform.node(),environment={k:v for k,v in e.items() if k.startswith(('FOUR602_','DYNAMIC602_','STRIDE_LSTM_'))})
    write(dest/'run.json',record);start=time.monotonic();print('START',s['run_id'],flush=True)
    with (dest/'run.log').open('w') as log:
        process=subprocess.Popen(['/usr/bin/time','-v','-o',str(dest/'host_time.txt')]+cmd,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record.update(status='running',pid=process.pid);write(dest/'run.json',record);code=process.wait()
    record.update(exit_code=code,wall_seconds=time.monotonic()-start)
    try:
        if code or 'ChampSim completed all CPUs' not in (dest/'run.log').read_text():raise RuntimeError('simulator failed or unfinished')
        summary=final_snapshot(dest/'snapshots.jsonl')
        if abs(summary['instructions']-s['instructions'])>4:raise RuntimeError('wrong full ROI length')
        if summary.get('numerical_failures',0):raise RuntimeError('numerical failure')
        c=summary.get('four602_counters',{})
        if model:
            if model.read_bytes()!=(dest/'model.bin').read_bytes():raise RuntimeError('model changed in run')
            if c.get('completed_updates',0)!=c.get('demand_callbacks',0)+c.get('fill_callbacks',0):raise RuntimeError('global callback/state update mismatch')
            if summary.get('four602_state_bytes')!=128:raise RuntimeError('wrong live H16 state payload')
            if s['policy']=='spp' and not c.get('fill_callbacks'):raise RuntimeError('SPP raw fills missing')
        write(dest/'summary.json',summary)
        path=dest/'callbacks.csv'
        if path.exists():
            with path.open('rb') as source,gzip.open(path.with_suffix('.csv.gz'),'wb') as target:shutil.copyfileobj(source,target)
            path.unlink()
        record['status']='complete'
    except Exception as error:
        record.update(status='failed',error=str(error));write(dest/'run.json',record);raise
    write(dest/'run.json',record);print('COMPLETE',s['run_id'],flush=True)
def run_many(rows):
    require_gate()
    for path in (RAW/'runs').glob('*/run.json'):
        record=read(path)
        if record.get('status')=='running':
            try:os.kill(record['pid'],0)
            except ProcessLookupError:continue
            raise RuntimeError('active prior row '+path.parent.name)
    errors=[]
    with ThreadPoolExecutor(max_workers=C['jobs']) as pool:
        for future in as_completed([pool.submit(run_one,s) for s in rows]):
            try:future.result()
            except Exception as error:errors.append(str(error))
    if errors:raise RuntimeError('\n'.join(errors))
def validation():run_many([s for s in matrix() if s['stage']=='validation'])
def select():
    require_gate();templates={}
    for policy in ('ampm','spp'):
        s=spec('validation',0,policy,16,'student');record=read(RAW/'runs'/s['run_id']/'run.json')
        if record['status']!='complete':raise RuntimeError('H16 full validation incomplete')
        x=read(RAW/'runs'/s['run_id']/'summary.json');counts=Counter();slots={}
        for name,hist in x['four602_distributions'].items():
            suffix=name.split('_',2)[2]
            if suffix=='raw_K':counts.update({int(k):v for k,v in hist.items()})
            if suffix.startswith('raw_delta_fill_'):
                fill,slot=map(int,suffix[len('raw_delta_fill_'):].split('_slot_'));slots.setdefault(slot,Counter()).update({(int(k),fill):v for k,v in hist.items()})
        if not counts or not slots:raise RuntimeError('validation profile missing')
        modal=min(counts,key=lambda k:(-counts[k],k));selected=[list(min(slots[i],key=lambda k:(-slots[i][k],k))) for i in range(max(slots)+1)]
        text='modal_count {}\n'.format(modal)+''.join('{} {}\n'.format(*v) for v in selected)
        path=template_path(policy,16);path.parent.mkdir(exist_ok=True)
        if path.exists() and path.read_text()!=text:raise RuntimeError('H16 validation template changed')
        path.write_text(text)
        templates[model_key(policy,16)]=dict(policy=policy,hidden_size=16,modal_count=modal,slots=selected,source_run=s['run_id'],selector='joint per-slot modal delta/fill; ties numeric ascending; overflow repeats final slot; no PC')
    result=dict(protocol=C['protocol'],common_cap=C['common_cap'],cap_source=dict(path=str(Path(C['reference_raw'])/'selection.json'),rule='fixed archived cap; no reselection'),validation_records=C['validation_records'],templates=templates)
    path=RAW/'selection.json'
    if path.exists() and read(path)!=result:raise RuntimeError('frozen supplemental selection changed')
    write(path,result);print('H16 VALIDATION SELECTION FROZEN',flush=True)
def test():
    require_gate()
    if read(RAW/'selection.json')['common_cap']!=C['common_cap']:raise RuntimeError('selection cap mismatch')
    run_many([s for s in matrix() if s['stage']=='test'])
def status():
    counts=Counter(read(p).get('status') for p in (RAW/'runs').glob('*/run.json'));print(json.dumps(dict(counts)))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','certify','validation','select','test','all','status','matrix']);a=parser.parse_args()
    if a.command=='matrix':print(json.dumps(matrix(),indent=2));return
    if a.command=='all':
        state=dict(protocol=C['protocol'],status='running',matrix_rows=24);write(RAW/'pipeline_result.json',state)
        try:
            for name in ('prepare','certify','validation','select','test'):globals()[name]()
            state['status']='complete';write(RAW/'pipeline_result.json',state)
        except Exception as error:
            state.update(status='failed',error=str(error));write(RAW/'pipeline_result.json',state);raise
    else:globals()[a.command]()
if __name__=='__main__':main()
