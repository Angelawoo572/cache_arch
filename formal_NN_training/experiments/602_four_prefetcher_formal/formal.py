#!/usr/bin/env python3
"""Recoverable full-ROI four-prefetcher study; no simulator pilot entry point."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
import csv
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[2]
C=json.loads((EXP/'config.json').read_text());RAW=Path(C['raw']);RUNTIME=Path(C['runtime']);PY=C['python']
OLD=EXP.parent/'602_lstm_stride_dynamic_inference'


def environment():
    inherited={k:v for k,v in os.environ.items() if not k.startswith(('FOUR602_','DYNAMIC602_','STRIDE_LSTM_'))}
    d=dict(inherited,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1',TMPDIR=str(RUNTIME/'tmp'),
        PYTHONPYCACHEPREFIX=str(RUNTIME/'pycache'),MPLCONFIGDIR=str(RUNTIME/'mpl'))
    for key in ['TMPDIR','PYTHONPYCACHEPREFIX','MPLCONFIGDIR']:Path(d[key]).mkdir(parents=True,exist_ok=True)
    return d


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def source_files():
    files=[]
    folders=[EXP/'runtime',ROOT/'formal_NN_training/common/live602_runtime',OLD/'runtime',OLD/'measurement',
        OLD/'patches',OLD.parent/'602_lstm_stride_live_inference/runtime']
    for policy in ('streamer','ampm','spp'):
        folders += [EXP.parent/('602_lstm_'+policy+'_live_inference')/'python',
                    EXP.parent/('602_lstm_'+policy+'_live_inference')/'validation',
                    EXP.parent/('602_offline_lstm_'+policy)/'python']
    for folder in folders:
        files += [p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.h','.cc','.py','.patch') and '__pycache__' not in str(p)]
    files += [ROOT/'formal_NN_training/common/threshold_free_policy.py',
              ROOT/'formal_NN_training/common/direct_action_lstm.py']
    return sorted(set(files+[EXP/'install.py',EXP/'formal.py',EXP/'config.json',
        EXP/'validation/test_cohort.cc',EXP/'validation/test_final_flush.py']))


def compiled_files(sim):
    sim=Path(sim);files=[]
    for folder in ('src','inc','prefetcher','branch','replacement'):
        files += [p for p in (sim/folder).rglob('*') if p.is_file() and p.suffix in ('.cc','.h')]
    return sorted(files+[sim/'Makefile',sim/'build_champsim.sh',sim/'build_four602.sh'])


def prepare_build_script(sim):
    sim=Path(sim);script=(sim/'build_champsim.sh').read_text()
    marker='# Restore to the default configuration'
    if script.count(marker)!=1:raise RuntimeError('isolated build restore marker changed')
    # Retain selected module copies and configuration as actually compiled.
    (sim/'build_four602.sh').write_text(script.split(marker)[0]+'\n# Isolated build retains actual selected modules.\n')
    makefile=sim/'Makefile';text=makefile.read_text()
    anchor='CFlags = -Wall -O3 -std=c++11 -D_DEFAULT_SOURCE -I./libbf/'
    if text.count(anchor)!=1 or '-ffp-contract=' in text:raise RuntimeError('unexpected compiler flag schema')
    makefile.write_text(text.replace(anchor,anchor+' -ffp-contract=off',1))


def copy_equal(source,target):
    source=Path(source);target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    if source.read_bytes()!=target.read_bytes():raise RuntimeError('snapshot byte mismatch: '+str(source))
    return dict(source=str(source),snapshot=str(target),bytes=source.stat().st_size)


def verify_bindings(bindings):
    for binding in bindings:
        a=Path(binding['source']);b=Path(binding['snapshot'])
        if not a.is_file() or not b.is_file() or a.read_bytes()!=b.read_bytes():
            raise RuntimeError('certification bytes changed: '+str(a))


def build():
    dest=RAW/'builds'/('build_'+str(time.time_ns()));sim=dest/'simulator';sim.mkdir(parents=True)
    data=subprocess.check_output(['git','-C',C['simulator'],'archive',C['simulator_base']])
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:archive.extractall(sim,filter='data')
    subprocess.run(['patch','-p1','-i',str(OLD/'patches/inherited_cache.patch')],cwd=sim,check=True)
    shutil.copytree(Path(C['simulator'])/'libbf',sim/'libbf',ignore=shutil.ignore_patterns('.git'))
    (sim/'.dynamic602_isolated').write_text(C['protocol']+'\n')
    subprocess.run([PY,str(OLD/'runtime/prepare_forward.py'),str(sim)],check=True,env=environment())
    subprocess.run([PY,str(EXP/'install.py'),str(sim)],check=True,env=environment())
    prepare_build_script(sim)
    build_env=environment();build_env['CXX']='g++'
    with (dest/'build.log').open('w') as log:
        subprocess.run(['bash','build_four602.sh','no','four602','no','1'],cwd=sim,env=build_env,stdout=log,stderr=subprocess.STDOUT,check=True)
    binary=sim/'bin/perceptron-no-four602-no-ship-1core'
    if not binary.is_file():raise RuntimeError('missing actual binary')
    if (sim/'prefetcher/l2c_prefetcher.cc').read_bytes()!=(sim/'prefetcher/four602.l2c_pref').read_bytes():
        raise RuntimeError('actual compiled L2 registry is not four602')
    snapshot=dest/'source_snapshot'
    for source in source_files():copy_equal(source,snapshot/source.relative_to(ROOT))
    copy_equal(binary,dest/'binary_snapshot')
    compiled=dest/'compiled_source'
    compiled_records=[copy_equal(source,compiled/source.relative_to(sim)) for source in compiled_files(sim)]
    module_path=OLD/'runtime/prepare_forward.py'
    spec=importlib.util.spec_from_file_location('prepared',module_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    forward=module.verify_prepared(ROOT,sim,require_marker=True)
    compiler=subprocess.check_output(['g++','--version'],text=True,env=build_env).splitlines()[0]
    write(RAW/'build.json',dict(protocol=C['protocol'],build_id=dest.name,binary=str(binary),binary_snapshot=str(dest/'binary_snapshot'),
        simulator=str(sim),source_snapshot=str(snapshot),compiled_source=str(compiled),
        compiled_source_files=len(compiled_records),compiled_source_byte_equal=True,
        generated_forward_verification=forward,compiler=compiler,cxx='g++',
        compiler_flags='-O3 -ffp-contract=off',base=C['simulator_base']))
    print('BUILD COMPLETE',str(binary),flush=True)


def current_build():
    b=json.loads((RAW/'build.json').read_text())
    if b['protocol']!=C['protocol'] or b['base']!=C['simulator_base']:raise RuntimeError('protocol/dependency changed')
    actual={str(p.relative_to(ROOT)) for p in source_files()}
    stored={str(p.relative_to(Path(b['source_snapshot']))) for p in Path(b['source_snapshot']).rglob('*') if p.is_file()}
    if actual!=stored:raise RuntimeError('critical source file set changed')
    for p in source_files():
        q=Path(b['source_snapshot'])/p.relative_to(ROOT)
        if not q.is_file() or q.read_bytes()!=p.read_bytes():raise RuntimeError('source changed; rebuild before formal run: '+str(p))
    if Path(b['binary']).read_bytes()!=Path(b['binary_snapshot']).read_bytes():raise RuntimeError('actual binary changed')
    sim=Path(b['simulator']);compiled=Path(b['compiled_source'])
    actual_compiled={str(p.relative_to(sim)) for p in compiled_files(sim)}
    saved_compiled={str(p.relative_to(compiled)) for p in compiled.rglob('*') if p.is_file()}
    if actual_compiled!=saved_compiled:raise RuntimeError('actual compiled source file set changed')
    for relative in sorted(actual_compiled):
        if (sim/relative).read_bytes()!=(compiled/relative).read_bytes():
            raise RuntimeError('actual compiled source bytes changed: '+relative)
    return b


def model_path(policy):return RAW/'models'/policy/'model.bin'


def certify():
    """Rerun real-checkpoint parity against a probe built from actual simulator source."""
    b=current_build();sim=Path(b['simulator'])
    cert=RAW/'certifications'/('cert_'+str(time.time_ns()));cert.mkdir(parents=True)
    env=environment();env['CXX']='g++';bindings=[];reports={};commands=[]
    state=dict(status='running',protocol=C['protocol'],build_id=b['build_id'],certification_root=str(cert))
    write(cert/'certification.json',state)
    def execute(command,log):
        verify_bindings(bindings);commands.append([str(v) for v in command])
        with (cert/log).open('w') as output:
            subprocess.run(commands[-1],cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
        verify_bindings(bindings)
    try:
        model_sources={policy:model_path(policy) for policy in ('streamer','ampm','spp')}
        model_sources['stride']=Path(C['stride_model'])
        for policy,source in model_sources.items():
            immutable=source.with_name('immutable_model.bin')
            if immutable.exists() and immutable.read_bytes()!=source.read_bytes():raise RuntimeError('frozen model replaced: '+policy)
            if not immutable.exists():copy_equal(source,immutable)
            bindings.append(copy_equal(source,cert/'models'/policy/'model.bin'))
        for policy in ('streamer','ampm','spp'):
            cp=checkpoint_path(policy)
            bindings.append(copy_equal(cp,cert/'checkpoints'/policy/'model.pt'))
        streamer_cp=checkpoint_path('streamer');metadata=streamer_cp.parent/'run_metadata.json'
        bindings.append(copy_equal(metadata,cert/'checkpoints/streamer/run_metadata.json'))
        cpp=sim/'prefetcher/live602_runtime.cc';header=sim/'inc/live602_runtime.h'
        probe_source=ROOT/'formal_NN_training/common/live602_runtime/probe.cc'
        for source,relative in [(cpp,'simulator/prefetcher/live602_runtime.cc'),
                                (header,'simulator/inc/live602_runtime.h'),(probe_source,'probe.cc')]:
            bindings.append(copy_equal(source,cert/'source'/relative))
        probe=cert/'probe'
        execute(['g++','-std=c++11','-O3','-ffp-contract=off','-I'+str(sim/'inc'),str(cpp),str(probe_source),'-o',str(probe)],'probe_compile.log')
        bindings.append(copy_equal(probe,cert/'probe_snapshot'))
        execute([PY,str(EXP.parent/'602_lstm_streamer_live_inference/validation/compare_python_cpp.py'),
            '--seed',str(C['seed']),'--allow-legacy-metadata','--model',str(model_path('streamer')),
            '--checkpoint',str(streamer_cp),'--run-metadata',str(metadata),'--probe',str(probe),
            '--sequence',str(streamer_cp.parents[2]/'colab_input/602.gcc_s-734B.train_stream.csv.gz'),
            '--max-events','4096','--out-dir',str(cert/'streamer'),'--runtime-dir',str(RUNTIME)],'streamer_parity.log')
        reports['streamer']=json.loads((cert/'streamer/parity_report.json').read_text())
        ampm_cp=checkpoint_path('ampm');guard=ampm_cp.parents[2]/'colab_input/602.gcc_s-734B.guard_stream.csv.gz'
        fixture=cert/'ampm/guard4096.csv';fixture.parent.mkdir()
        with gzip.open(guard,'rt',newline='') as source,fixture.open('w',newline='') as target:
            reader=csv.DictReader(source);writer=csv.DictWriter(target,fieldnames=reader.fieldnames);writer.writeheader();count=0
            for row in reader:
                writer.writerow(row);count+=1
                if count==4096:break
        if count!=4096:raise RuntimeError('AMPM real guard has fewer than4096 correctness rows')
        execute([PY,str(EXP.parent/'602_lstm_ampm_live_inference/validation/validate_parity.py'),
            '--checkpoint',str(ampm_cp),'--model',str(model_path('ampm')),'--probe',str(probe),
            '--stream',str(fixture),'--tmp',str(cert/'ampm/runtime'),'--output',str(cert/'ampm/parity.json')],'ampm_parity.log')
        reports['ampm']=json.loads((cert/'ampm/parity.json').read_text())
        spp_cp=checkpoint_path('spp');spp_model=cert/'spp/model.bin';spp_model.parent.mkdir()
        execute([PY,str(EXP.parent/'602_lstm_spp_live_inference/python/export_and_validate.py'),
            '--checkpoint',str(spp_cp),'--model',str(spp_model),'--probe',str(probe),
            '--stream',str(spp_cp.parents[2]/'colab_input/602.gcc_s-734B.spp.train_stream.csv.gz'),
            '--events','4096','--report',str(cert/'spp/parity.json')],'spp_parity.log')
        if spp_model.read_bytes()!=model_path('spp').read_bytes():raise RuntimeError('certified SPP re-export differs from formal model')
        reports['spp']=json.loads((cert/'spp/parity.json').read_text())
        if any(report.get('status') not in ('PASS','verified') for report in reports.values()):raise RuntimeError('real checkpoint parity failed')
        cohort=cert/'cohort_test'
        execute(['g++','-std=c++11','-O3','-ffp-contract=off','-I'+str(sim/'inc'),str(EXP/'validation/test_cohort.cc'),'-o',str(cohort)],'cohort_compile.log')
        execute([str(cohort)],'cohort_test.log')
        result=json.loads((cert/'cohort_test.log').read_text())
        if result.get('status')!='PASS':raise RuntimeError('measurement cohort test failed')
        execute([PY,str(EXP/'validation/test_final_flush.py'),'--source',str(sim/'prefetcher/four602_engine.cc'),
            '--out',str(cert/'final_flush'),'--cxx','g++'],'final_flush_test.log')
        flush_report=json.loads((cert/'final_flush/verification.json').read_text())
        if flush_report.get('status')!='PASS':raise RuntimeError('final statistics callback flush failed')
        current_build();verify_bindings(bindings)
        state.update(final_statistics_flush=flush_report,status='PASS',actual_checkpoint_parity=reports,measurement_test=result,bindings=bindings,commands=commands,
            actual_probe_source=str(cpp),probe=str(probe),compiler_flags='-O3 -ffp-contract=off',
            ampm_fixture_source=str(guard),ampm_fixture_rows=count,formal_models_unchanged=True,
            scope='4096 real causal correctness rows per global model plus script boundary/reset cases; no simulator pilot/IPC sample')
        write(cert/'certification.json',state);write(RAW/'correctness_gate.json',state)
    except Exception as error:
        state.update(status='FAIL',error=str(error),commands=commands);write(cert/'certification.json',state);raise
    print('CORRECTNESS GATE PASS',str(cert),flush=True)


def require_correctness():
    gate=json.loads((RAW/'correctness_gate.json').read_text());b=current_build()
    if gate['status']!='PASS' or gate['protocol']!=C['protocol'] or gate['build_id']!=b['build_id']:
        raise RuntimeError('correctness certificate does not describe the current build')
    verify_bindings(gate['bindings'])
    for policy in ('streamer','ampm','spp'):
        p=model_path(policy)
        if p.read_bytes()!=p.with_name('immutable_model.bin').read_bytes():raise RuntimeError('frozen weights changed')


def checkpoint_path(policy):return Path(C['model_root'])/('602_offline_lstm_'+policy)/'runs'/C['checkpoint_runs'][policy]


def spec(stage,roi,policy,method,mode='full',budget=0):
    start,end=C['validation_records'] if stage=='validation' else C['test_records'][roi]
    warm=C['warmup_instructions'];run_id='{}_r{}_{}_{}_{}_cap{}'.format(stage,roi,policy,method,mode,budget)
    return dict(run_id=run_id,protocol=C['protocol'],stage=stage,roi=roi,policy=policy,method=method,mode=mode,budget=budget,
        skip=start-warm,warmup=warm,instructions=end-start,measurement_records=[start,end],seed=C['seed'],macs=0,page_policy=C['page_policy'])


def final_snapshot(path):
    last=None
    with Path(path).open() as f:
        for line in f:
            row=json.loads(line)
            if row.get('event')=='snapshot' and row.get('final'):last=row
    if last is None:raise RuntimeError('no final ROI snapshot')
    return last


def run_one(s):
    b=current_build();dest=RAW/'runs'/s['run_id']
    expected=dict(s,build_id=b['build_id'],contract=json.loads((EXP/'config.json').read_text()))
    model=model_path(s['policy']) if s['method']=='student' else Path(C['stride_model']) if s['method']=='stride_student' else None
    if model:
        if not model.is_file():raise RuntimeError('missing actual model: '+str(model))
        if model.read_bytes()!=model.with_name('immutable_model.bin').read_bytes():raise RuntimeError('frozen checkpoint replaced')
        expected['model']=str(model)
    template=RAW/'templates'/(s['policy']+'.txt') if s['method']=='template' or s['mode']=='fixed_payload' else None
    if template:expected['frozen_template']=template.read_text()
    if dest.exists():
        prior=json.loads((dest/'run.json').read_text())
        if prior['status']=='complete' and prior['spec']==expected:
            if model and model.read_bytes()!=(dest/'model.bin').read_bytes():raise RuntimeError('completed run used different weights')
            final_snapshot(dest/'snapshots.jsonl');print('COMPLETE EXISTS',s['run_id'],flush=True);return
        if prior.get('status')=='running':
            try:os.kill(prior['pid'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError('run still active '+s['run_id'])
        old=RAW/'preserved_attempts'/(s['run_id']+'_'+str(time.time_ns()));old.parent.mkdir(exist_ok=True);dest.rename(old)
    dest.mkdir(parents=True)
    env=environment()
    env.update(FOUR602_POLICY=s['policy'],FOUR602_METHOD='monitor' if s['method'].startswith('stride_') else s['method'],FOUR602_MODE=s['mode'],
        DYNAMIC602_METHOD='none',DYNAMIC602_MACS='0',DYNAMIC602_STATE_CAPACITY='64',DYNAMIC602_INPUT_QUEUE='16',DYNAMIC602_OUTPUT_QUEUE='32',
        DYNAMIC602_OUTPUT_LIMIT='32',DYNAMIC602_NONLINEAR_CYCLES='4',DYNAMIC602_BANDWIDTH_BYTES_PER_CYCLE='16',
        DYNAMIC602_PAGE_POLICY=s['page_policy'],DYNAMIC602_WARMUP_INSTRUCTIONS=str(s['warmup']),DYNAMIC602_HISTORY_INSTRUCTIONS=str(s['warmup']),
        DYNAMIC602_SKIP_RECORDS=str(s['skip']),DYNAMIC602_MAX_RECORDS=str(s['warmup']+s['instructions']+1024),
        DYNAMIC602_ROI_INSTRUCTIONS=str(s['instructions']),DYNAMIC602_PHASE_INSTRUCTIONS=str(C['phase_instructions']),
        DYNAMIC602_REQUESTS_PER_MILLION_INSTRUCTIONS=str(s['budget']),DYNAMIC602_FILTER_CURRENT='0',
        DYNAMIC602_ARM=s['run_id'],DYNAMIC602_STATS=str(dest/'snapshots.jsonl'))
    if model:env['FOUR602_MODEL']=str(model)
    if template:env['FOUR602_TEMPLATE']=str(template)
    if s['method']=='student' and s['mode']=='full' and not s['budget']:
        env['FOUR602_INPUT_LOG']=str(dest/'callbacks.csv')
    l2=['--l2c_prefetcher_types=four602']
    if s['method'].startswith('stride_'):
        l2.append('--l2c_prefetcher_types=dynamic602')
        if s['method']=='stride_student':env.update(DYNAMIC602_METHOD='frozen',STRIDE_LSTM_MODEL_BIN=str(model),DYNAMIC602_FILTER_CURRENT='1')
        else:env.update(DYNAMIC602_METHOD='stride',DYNAMIC602_STRIDE_MODE=s['method'][7:])
    cmd=[b['binary']]+l2+['--streamer_num_trackers=64','--streamer_pref_degree=5','--ampm_pb_size=64','--ampm_pred_degree=4',
        '--ampm_pref_degree=4','--ampm_enable_pref_buffer=false','--ampm_pref_buffer_size=256','--ampm_max_delta=16',
        '--stride_num_trackers=64','--stride_pref_degree=2','--spp_dev2_fill_threshold=90','--spp_dev2_pf_threshold=40','--warmup_instructions='+str(s['warmup']),
        '--simulation_instructions='+str(s['instructions']),'-traces',C['trace']]
    record=dict(status='preparing',spec=expected,command=cmd,environment={k:v for k,v in env.items() if k.startswith(('FOUR602_','DYNAMIC602_','STRIDE_LSTM_'))})
    write(dest/'run.json',record)
    if model:shutil.copy2(model,dest/'model.bin')
    started=time.monotonic();print('START',s['run_id'],flush=True)
    with (dest/'run.log').open('w') as log:
        proc=subprocess.Popen(['/usr/bin/time','-v','-o',str(dest/'host_time.txt')]+cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        record.update(status='running',pid=proc.pid);write(dest/'run.json',record);code=proc.wait()
    record.update(exit_code=code,wall_seconds=time.monotonic()-started)
    try:
        if code:raise RuntimeError('simulator exit '+str(code))
        if 'ChampSim completed all CPUs' not in (dest/'run.log').read_text():raise RuntimeError('unfinished simulator log')
        last=final_snapshot(dest/'snapshots.jsonl')
        if abs(last['instructions']-s['instructions'])>4:raise RuntimeError('wrong final ROI length')
        if model and model.read_bytes()!=(dest/'model.bin').read_bytes():raise RuntimeError('weights changed while running')
        if last.get('numerical_failures',0):raise RuntimeError('numerical failure')
        counters=last.get('four602_counters',{})
        if s['method']=='student':
            if counters.get('completed_updates',0)!=counters.get('demand_callbacks',0)+counters.get('fill_callbacks',0):raise RuntimeError('missing global recurrent callback update')
            if s['policy']=='spp' and not counters.get('fill_callbacks'):raise RuntimeError('SPP actual fill callbacks absent')
            if last.get('four602_method')!='student':raise RuntimeError('student mapped to another method')
        write(dest/'summary.json',last);record['status']='complete'
        csv_path=dest/'callbacks.csv'
        if csv_path.exists():
            with csv_path.open('rb') as source,gzip.open(csv_path.with_suffix('.csv.gz'),'wb') as target:shutil.copyfileobj(source,target)
            csv_path.unlink()
    except Exception as e:
        record.update(status='failed',error=str(e));write(dest/'run.json',record);raise
    write(dest/'run.json',record);print('COMPLETE',s['run_id'],round(record['wall_seconds'],2),flush=True)


def no_live_prior_runs():
    live=[]
    for path in (RAW/'runs').glob('*/run.json'):
        prior=json.loads(path.read_text())
        if prior.get('status')!='running':continue
        try:os.kill(prior['pid'],0)
        except ProcessLookupError:continue
        except PermissionError:pass
        live.append(dict(run=path.parent.name,pid=prior['pid']))
    if live:raise RuntimeError('existing simulator jobs still active; preserve and wait before a new batch: '+json.dumps(live))


def run_many(rows):
    no_live_prior_runs()
    errors=[]
    with ThreadPoolExecutor(max_workers=C['jobs']) as pool:
        for future in as_completed([pool.submit(run_one,s) for s in rows]):
            try:future.result()
            except Exception as e:errors.append(str(e));print('FAILED',e,flush=True)
    if errors:raise RuntimeError('\n'.join(errors))


def validation():
    require_correctness()
    rows=[spec('validation',0,'none','none')]
    for policy in ['stride','streamer','ampm','spp']:rows.append(spec('validation',0,policy,'native'))
    for policy in ['streamer','ampm','spp']:rows.append(spec('validation',0,policy,'student'))
    run_many(rows)


def select():
    require_correctness()
    n=json.loads((RAW/'runs'/spec('validation',0,'none','none')['run_id']/'summary.json').read_text())
    cap=int(n['four602_counters']['demand_callbacks']*1000000/n['instructions'])
    if cap<=0:raise RuntimeError('invalid validation NoPF callback cap')
    templates={}
    for policy in ['streamer','ampm','spp']:
        s=json.loads((RAW/'runs'/spec('validation',0,policy,'student')['run_id']/'summary.json').read_text())
        distributions=s['four602_distributions'];counts=Counter();slots={}
        for name,hist in distributions.items():
            suffix=name.split('_',2)[2]
            if suffix=='raw_K':counts.update({int(k):v for k,v in hist.items()})
            if suffix.startswith('raw_delta_fill_'):
                fill,slot=map(int,suffix[len('raw_delta_fill_'):].split('_slot_'))
                c=slots.setdefault(slot,Counter());c.update({(int(k),fill):v for k,v in hist.items()})
        if not counts or not slots:raise RuntimeError('no full validation profile '+policy)
        modal=min(counts,key=lambda k:(-counts[k],k));selected=[]
        for i in range(max(slots)+1):
            delta,fill=min(slots[i],key=lambda k:(-slots[i][k],k));selected.append([delta,fill])
        path=RAW/'templates'/(policy+'.txt');path.parent.mkdir(exist_ok=True)
        content='modal_count {}\n'.format(modal)+''.join('{} {}\n'.format(d,f) for d,f in selected)
        if path.exists() and path.read_text()!=content:raise RuntimeError('frozen template changed')
        path.write_text(content)
        templates[policy]=dict(modal_count=modal,slots=selected,source_run=spec('validation',0,policy,'student')['run_id'],
            selector='joint per-slot modal delta/fill; ties numeric ascending; overflow slot repeats final observed slot; no PC')
    selection=dict(protocol=C['protocol'],validation_records=C['validation_records'],common_cap=cap,
        cap_rule=C['request_cap_selection'],same_actual_requests_guaranteed=False,templates=templates)
    selection_file=RAW/'selection.json'
    if selection_file.exists() and json.loads(selection_file.read_text())!=selection:raise RuntimeError('frozen validation selection changed')
    write(selection_file,selection)
    print('FROZEN VALIDATION SELECTION',cap,templates,flush=True)


def test_matrix():
    selection=json.loads((RAW/'selection.json').read_text());cap=selection['common_cap'];rows=[]
    for roi in range(2):
        rows.append(spec('test',roi,'none','none'))
        for budget in [0,cap]:
            for policy in ['stride','streamer','ampm','spp']:rows.append(spec('test',roi,policy,'native',budget=budget))
            rows.append(spec('test',roi,'stride','stride_student',budget=budget))
            for policy in ['streamer','ampm','spp']:rows.append(spec('test',roi,policy,'student',budget=budget))
        for policy in ['streamer','ampm','spp']:
            for method,mode in [('student','fixed_payload'),('template','full'),('student','reset_state')]:rows.append(spec('test',roi,policy,method,mode))
        for mode in ['forward1','forward2']:rows.append(spec('test',roi,'stride','stride_'+mode))
    return rows


def test():
    require_correctness();run_many(test_matrix())


def status():
    counts=Counter()
    for p in (RAW/'runs').glob('*/run.json'):counts[json.loads(p.read_text())['status']]+=1
    print(json.dumps(dict(counts),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['build','certify','validation','select','test','status','matrix']);a=p.parse_args()
    if a.command=='matrix':print(json.dumps(test_matrix(),indent=2))
    else:globals()[a.command]()
