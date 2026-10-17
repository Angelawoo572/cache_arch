#!/usr/bin/env python3
"""Full formal gate/payload mechanism study; no performance pilot or hash ledger."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import filecmp
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

EXP=Path(__file__).resolve().parent
ROOT=EXP.parents[2]
DYNAMIC=EXP.parent/'602_lstm_stride_dynamic_inference'
LIVE=EXP.parent/'602_lstm_stride_live_inference'
C=json.loads((EXP/'gate_config.json').read_text())
RAW=Path(C['raw'])
OLD=Path(C['reference_raw'])
module_spec=importlib.util.spec_from_file_location('formal_gate_base',DYNAMIC/'formal.py')
base=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(base)
base.C=C;base.RAW=RAW;base.PY=C['python']
THREAD_ENV=base.THREAD_ENV
write=base.write

def source_files():
    paths=[]
    for directory in (DYNAMIC/'runtime',DYNAMIC/'measurement',DYNAMIC/'patches',LIVE/'runtime'):
        paths += [p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    paths += [DYNAMIC/'formal.py']
    # Postprocessing, documentation and unit tests may evolve while jobs run.
    # Only these three local files affect actual fitting/build/launch semantics.
    paths += [EXP/name for name in ('run_gate.py','gate_config.json','score_pc.cc')]
    return sorted(paths)
base.source_files=source_files

def assert_build_current():
    b=base.assert_build_current()
    if not filecmp.cmp(b['scorer'],b['scorer_snapshot'],shallow=False):
        raise RuntimeError('standalone scorer changed after build')
    return b

def build():
    base.build()
    b=json.loads((RAW/'build.json').read_text());sim=Path(b['simulator'])
    scorer=sim.parent/'score_pc'
    command=['g++','-std=c++14','-O2','-ffp-contract=off','-I'+str(sim/'inc'),
             str(EXP/'score_pc.cc'),str(sim/'prefetcher/stride_lstm_runtime.cc'),
             str(sim/'prefetcher/stride_lstm_model_loader.cc'),'-o',str(scorer)]
    with (sim.parent/'scorer_build.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    frozen=scorer.with_name('score_pc_snapshot');shutil.copy2(scorer,frozen)
    b.update(scorer=str(scorer),scorer_snapshot=str(frozen),scorer_command=command)
    write(RAW/'build.json',b)

def model(seed):
    return OLD/'models'/('seed'+str(seed))/'balanced/export/model.bin'

def model_identity(seed):
    path=model(seed);identity=json.loads(path.with_name('model_metadata.json').read_text())
    if identity['seed']!=seed or identity['emit_loss']!='balanced':
        raise RuntimeError('independent balanced checkpoint identity mismatch')
    if not filecmp.cmp(path,path.with_name('immutable_model.bin'),shallow=False):
        raise RuntimeError('verified checkpoint bytes changed')
    return identity

def spec(roi=0,arm='nn',seed=7,macs=0,budget=0,filtered=0,payload='learned',silent=0,execution='fresh'):
    start,end=C['test_records'][roi]
    method=('frozen' if arm in {'nn','nn_gate_fixed_plus2'} else
            arm if arm in {'fixed_plus2','pc_gate_plus2'} else
            'stride' if arm.startswith('stride_') else 'none')
    run_id=f'test_602_r{roi}_{arm}_s{seed}_m{macs}_b{budget}_f{filtered}_sil{silent}'
    return dict(run_id=run_id,stage='test',workload='602.gcc_s-734B',roi=roi,arm=arm,
                seed=seed,loss='balanced',macs=macs,budget=budget,silent=silent,queue='fifo',
                history=C['state_history_instructions'],warmup=C['warmup'],skip=start-C['warmup'],
                instructions=end-start,method=method,payload=payload,filter_current=filtered,
                execution=execution,trace=str(Path(C['trace_root'])/'602.gcc_s-734B.champsimtrace.xz'),
                model=str(model(seed)) if method=='frozen' else None,
                gate_table=str(RAW/'pc_gate'/f'seed{seed}'/'pc_gate.tsv') if method=='pc_gate_plus2' else None,
                protocol=C['protocol'])

def matrix():
    rows=[];cap=C['budget']
    for roi in (0,1):
        for seed in C['seeds']:
            for budget in (0,cap):
                for filtered in (0,1):rows.append(spec(roi,seed=seed,budget=budget,filtered=filtered))
                rows.append(spec(roi,'nn_gate_fixed_plus2',seed,budget=budget,filtered=1,payload='fixed_plus2'))
                rows.append(spec(roi,'pc_gate_plus2',seed,budget=budget,filtered=1,payload='fixed_plus2'))
        for budget in (0,cap):rows.append(spec(roi,'fixed_plus2',budget=budget,filtered=1,payload='fixed_plus2'))
        for filtered in (0,1):rows.append(spec(roi,seed=7,macs=4,budget=cap,filtered=filtered))
        rows.append(spec(roi,'none'))
        for budget in (0,cap):rows.append(spec(roi,'spp',budget=budget))
        for arm in ('stride_forward1','stride_forward2'):
            for budget in (0,cap):
                execution='fresh' if (roi,arm,budget)==(1,'stride_forward2',cap) else 'compatible_reference'
                rows.append(spec(roi,arm,budget=budget,execution=execution))
    rows.append(spec(1,seed=7,silent=1))
    assert len(rows)==71 and len({r['run_id'] for r in rows})==71
    assert Counter(r['execution'] for r in rows)=={'fresh':64,'compatible_reference':7}
    return rows

def verify_budget():
    selection=json.loads((OLD/'budget_selection.json').read_text())
    if (selection['protocol']!=C['reference_protocol'] or
        selection['requests_per_million_instructions']['602.gcc_s-734B']!=C['budget']):
        raise RuntimeError('fixed validation-selected common budget changed')
    write(RAW/'budget_selection.json',dict(protocol=C['protocol'],fixed_from_protocol=C['reference_protocol'],
        source=str(OLD/'budget_selection.json'),selection=selection))

def integer(s):
    return int(s,16) if str(s).lower().startswith('0x') else int(s,10)

def normalize_split(name):
    source=Path(C[name+'_stream']);folder=RAW/'fit_inputs'/name
    folder.mkdir(parents=True,exist_ok=True);snapshot=folder/'source.csv.gz'
    if snapshot.exists():
        if not filecmp.cmp(source,snapshot,shallow=False):raise RuntimeError('fit source changed: '+name)
    else:shutil.copy2(source,snapshot)
    destination=folder/'pc_line.csv';temporary=destination.with_suffix('.tmp')
    events=0
    with gzip.open(snapshot,'rt',newline='') as f,temporary.open('w',newline='') as out:
        reader=csv.DictReader(f)
        if not {'pc','line'}.issubset(reader.fieldnames or []):raise RuntimeError('fit stream needs pc,line')
        out.write('pc,line\n')
        for row in reader:
            pc,line=integer(row['pc']),integer(row['line'])
            if not 0<=pc<2**64 or not 0<=line<2**58:raise RuntimeError('fit address units/domain invalid')
            out.write(f'{pc},{line}\n');events+=1
    if not events:raise RuntimeError('empty fit input')
    if destination.exists():
        if not filecmp.cmp(destination,temporary,shallow=False):raise RuntimeError('normalized fit input changed')
        temporary.unlink()
    else:temporary.replace(destination)
    if name=='train':
        manifest=json.loads((OLD/'training_split.json').read_text())
        if manifest['decision_rows']!=events or manifest.get('training_simulation_instructions')!=1000000:
            raise RuntimeError('training input does not match complete original 1M stream')
    else:
        directory=OLD/'validation/validation_602_r0_none_s7_balanced_m0_b0_sil0_qfifo_hist4000000'
        record=json.loads((directory/'run.json').read_text());summary=json.loads((directory/'summary.json').read_text())
        if (record['status']!='complete' or record['spec']['protocol']!=C['reference_protocol']
            or summary['eligible_l2_callbacks']!=events):raise RuntimeError('validation input is not complete NoPF ROI stream')
    return dict(name=name,source=str(source),snapshot=str(snapshot),canonical=str(destination),events=events,
                initial_state='independent_zero',state_capacity=64,source_records=C[name+'_records'])

def read_counts(path):
    rows={}
    with path.open() as f:
        for r in csv.DictReader(f):
            pc=int(r['pc']);count=int(r['count']);positive=int(r['positives'])
            if pc in rows or count<=0 or not 0<=positive<=count:raise RuntimeError('invalid scored PC counts')
            rows[pc]=(count,positive,int(r['emit_k_zero']))
    return rows

def make_table(combined):
    selected=sorted(combined,key=lambda pc:(-combined[pc][0],pc))[:64]
    gates={pc:int(2*combined[pc][1]>=combined[pc][0]) for pc in selected}
    contents='pc gate count positives\n'+''.join(f'{pc} {gates[pc]} {combined[pc][0]} {combined[pc][1]}\n' for pc in selected)
    return contents,gates

def read_gate_table(path):
    """Independent strict parse of the engine ABI; engine harness also loads it."""
    lines=path.read_text(encoding='ascii').splitlines()
    if not lines or lines[0]!='pc gate count positives':raise RuntimeError('invalid PC table header')
    rows={}
    for line in lines[1:]:
        fields=line.split()
        if len(fields)!=4 or any(not s.isascii() or not s.isdecimal() for s in fields):
            raise RuntimeError('PC table fields must be four unsigned decimal integers')
        pc,gate,count,positive=map(int,fields)
        if (pc>=2**64 or count<=0 or count>=2**64 or positive>count or gate not in (0,1)
                or gate!=int(2*positive>=count) or pc in rows):raise RuntimeError('invalid PC table values')
        rows[pc]=(gate,count,positive)
    if not rows or len(rows)>64:raise RuntimeError('PC table must contain 1..64 unique entries')
    return rows

def fit_tables():
    b=assert_build_current();verify_budget()
    contract=dict(protocol=C['protocol'],fit_splits=C['pc_gate_fit_splits'],capacity=64,
                  selection='descending combined event count; ascending unsigned PC tie',
                  majority='2 * NN_emit_positive_count >= combined_event_count',threshold=.5,
                  split_state=C['pc_gate_split_state'],emit_definition='raw result.emit, independent of K/addresses',
                  unseen_pc='silent',test_used=False,build_id=b['build_id'])
    contract_path=RAW/'pc_gate/fit_contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text())!=contract:
        raise RuntimeError('PC fit contract changed; use a new run root')
    write(contract_path,contract)
    inputs=[normalize_split(name) for name in ('train','validation')]
    write(RAW/'pc_gate/input_records.json',inputs)
    for seed in C['seeds']:
        identity=model_identity(seed);folder=RAW/'pc_gate'/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
        source_model=folder/'model_snapshot.bin'
        if source_model.exists() and not filecmp.cmp(source_model,model(seed),shallow=False):raise RuntimeError('fit model changed')
        if not source_model.exists():shutil.copy2(model(seed),source_model)
        expected=dict(contract=contract,model_metadata=identity,inputs=inputs,seed=seed)
        record_path=folder/'fit.json'
        prior=json.loads(record_path.read_text()) if record_path.exists() else None
        if prior and prior.get('spec')!=expected:raise RuntimeError('fit identity changed')
        split_counts={}
        for source in inputs:
            counts=folder/(source['name']+'_counts.csv');summary=folder/(source['name']+'_score.json')
            if not(prior and prior.get('status')=='complete' and counts.is_file() and summary.is_file()):
                with (folder/(source['name']+'_score.log')).open('w') as log:
                    subprocess.run([b['scorer'],str(model(seed)),source['canonical'],str(counts),str(summary)],
                                   stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,**THREAD_ENV))
            score=json.loads(summary.read_text());rr=read_counts(counts)
            if score['events']!=source['events'] or sum(v[0] for v in rr.values())!=source['events'] or score['full_stream'] is not True:
                raise RuntimeError('scorer did not consume complete split')
            split_counts[source['name']]=rr
        combined={}
        for rr in split_counts.values():
            for pc,v in rr.items():combined[pc]=tuple(a+b for a,b in zip(combined.get(pc,(0,0,0)),v))
        table,gates=make_table(combined)
        table_path=folder/'pc_gate.tsv'
        if table_path.exists() and table_path.read_text()!=table:raise RuntimeError('fitted PC table changed')
        table_path.write_text(table)
        roundtrip=read_gate_table(table_path)
        if roundtrip!={pc:(gate,combined[pc][0],combined[pc][1]) for pc,gate in gates.items()}:
            raise RuntimeError('written PC table does not roundtrip exactly')
        agreement={}
        for name,rr in dict(split_counts,combined=combined).items():
            total=sum(v[0] for v in rr.values());correct=sum(v[1] if gates.get(pc,0) else v[0]-v[1] for pc,v in rr.items())
            agreement[name]=dict(events=total,pcs=len(rr),agreement_events=correct,agreement=correct/total,
                table_miss_events=sum(v[0] for pc,v in rr.items() if pc not in gates),
                table_miss_pcs=sum(pc not in gates for pc in rr),
                unseen_in_combined_events=sum(v[0] for pc,v in rr.items() if pc not in combined),
                nn_emit_positive_events=sum(v[1] for v in rr.values()),nn_emit_k_zero_events=sum(v[2] for v in rr.values()))
        write(record_path,dict(status='complete',spec=expected,table=str(table_path),table_entries=len(gates),
                               gate_positive_entries=sum(gates.values()),agreement=agreement,
                               python_table_roundtrip='PASS',engine_table_roundtrip='required_in_correctness_ready'))
        print('PC_GATE_FIT_COMPLETE',seed,agreement['combined'],flush=True)

def correctness_ready(b):
    path=RAW/'correctness_ready.json'
    if not path.is_file():raise RuntimeError('formal launch awaits correctness_ready.json from coordinated unit audits')
    ready=json.loads(path.read_text())
    if ready.get('status')!='PASS' or ready.get('protocol')!=C['protocol'] or ready.get('build_id')!=b['build_id'] or not ready.get('checks'):
        raise RuntimeError('correctness certificate does not authorize this build/protocol')
    return ready

def expected_spec(s,b):
    expected=dict(s,build_id=b['build_id'],controls={k:C[k] for k in (
        'state_capacity','input_queue','output_queue','decoder_limit','nonlinear_cycles','page_policy')})
    if s['model']:expected['model_metadata']=model_identity(s['seed'])
    if s['gate_table']:
        folder=Path(s['gate_table']).parent;fit=json.loads((folder/'fit.json').read_text())
        if fit['status']!='complete' or fit['spec']['seed']!=s['seed'] or fit['spec']['contract']['build_id']!=b['build_id']:
            raise RuntimeError('PC table fit is not compatible with this run')
        expected['gate_table_contents']=Path(s['gate_table']).read_text()
        expected['gate_fit']=fit
    return expected

def validate_final(s,directory):
    final=base.final_snapshot(directory/'snapshots.jsonl')
    if abs(final['instructions']-s['instructions'])>4 or final.get('numerical_failures',0):
        raise RuntimeError('ROI/numerical check failed')
    if final.get('engine_method')!=s['method'] or final.get('engine_payload_mode')!=s['payload']:
        raise RuntimeError('engine did not receive requested method/payload ABI')
    if 'ChampSim completed all CPUs' not in (directory/'run.log').read_text():
        raise RuntimeError('simulator completion marker missing')
    return final

def run_one(s):
    if s['execution']!='fresh':raise RuntimeError('reused reference cannot execute as a fresh row')
    b=assert_build_current();correctness_ready(b);expected=expected_spec(s,b)
    directory=RAW/'test'/s['run_id']
    if directory.exists():
        prior=json.loads((directory/'run.json').read_text())
        if prior.get('status')=='complete' and prior.get('spec')==expected:
            try:
                final=validate_final(s,directory)
                if json.loads((directory/'summary.json').read_text())!=final:
                    raise RuntimeError('summary differs from final snapshot')
                print('COMPLETE_EXISTS',s['run_id'],flush=True);return
            except (RuntimeError,FileNotFoundError,json.JSONDecodeError):pass
        if prior.get('status')=='running':
            try:os.kill(prior['pid'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError('active process is not a completed result: '+s['run_id'])
        preserved=RAW/'preserved_attempts'/(s['run_id']+'_'+str(time.time_ns()))
        preserved.parent.mkdir(parents=True,exist_ok=True);directory.rename(preserved)
    directory.mkdir(parents=True)
    env={k:v for k,v in os.environ.items() if not k.startswith(('DYNAMIC602_','STRIDE_LSTM_'))}
    env.update(THREAD_ENV)
    env.update(DYNAMIC602_METHOD=s['method'],DYNAMIC602_PAYLOAD=s['payload'],
        DYNAMIC602_ARM=s['arm'],DYNAMIC602_ROI_INSTRUCTIONS=str(s['instructions']),
        DYNAMIC602_FILTER_CURRENT=str(s['filter_current']),DYNAMIC602_MACS=str(s['macs']),
        DYNAMIC602_STATE_CAPACITY=str(C['state_capacity']),DYNAMIC602_INPUT_QUEUE=str(C['input_queue']),
        DYNAMIC602_OUTPUT_QUEUE=str(C['output_queue']),DYNAMIC602_OUTPUT_LIMIT=str(C['decoder_limit']),
        DYNAMIC602_NONLINEAR_CYCLES=str(C['nonlinear_cycles']),DYNAMIC602_PAGE_POLICY=C['page_policy'],
        DYNAMIC602_QUEUE_POLICY=s['queue'],DYNAMIC602_WARMUP_INSTRUCTIONS=str(s['warmup']),
        DYNAMIC602_HISTORY_INSTRUCTIONS=str(s['history']),DYNAMIC602_SILENT_SUPPRESS=str(s['silent']),
        DYNAMIC602_PHASE_INSTRUCTIONS='5000000',
        DYNAMIC602_REQUESTS_PER_MILLION_INSTRUCTIONS=str(s['budget']),DYNAMIC602_KEEP_PPM='1000000',
        DYNAMIC602_SKIP_RECORDS=str(s['skip']),DYNAMIC602_MAX_RECORDS=str(s['warmup']+s['instructions']+1024),
        DYNAMIC602_STATS=str(directory/'snapshots.jsonl'),DYNAMIC602_HISTORY_LOG=str(directory/'events.jsonl'),
        DYNAMIC602_HISTORY_LOG_LIMIT='4096',DYNAMIC602_TRACE_NAME=s['workload'])
    if s['arm'].startswith('stride_'):env['DYNAMIC602_STRIDE_MODE']=s['arm'][7:]
    if s['model']:env['STRIDE_LSTM_MODEL_BIN']=s['model']
    if s['gate_table']:env['DYNAMIC602_GATE_TABLE']=s['gate_table']
    l2=['--l2c_prefetcher_types=dynamic602']+(['--l2c_prefetcher_types=spp_dev2'] if s['arm']=='spp' else [])
    cmd=[b['binary']]+l2+['--stride_num_trackers=64','--stride_pref_degree=2',
        '--warmup_instructions='+str(s['warmup']),'--simulation_instructions='+str(s['instructions']),'-traces',s['trace']]
    record=dict(status='preparing',spec=expected,command=cmd,
                environment={k:v for k,v in env.items() if k.startswith(('DYNAMIC602_','STRIDE_LSTM_'))})
    write(directory/'run.json',record);print('START',s['run_id'],flush=True);start=time.monotonic()
    with (directory/'run.log').open('w') as log:
        proc=subprocess.Popen(['/usr/bin/time','-v','-o',str(directory/'host_time.txt')]+cmd,
                              stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        record.update(status='running',pid=proc.pid);write(directory/'run.json',record);code=proc.wait()
    record.update(exit_code=code,wall_seconds=time.monotonic()-start)
    try:
        if code or 'ChampSim completed all CPUs' not in (directory/'run.log').read_text():raise RuntimeError('simulator failed/incomplete')
        final=validate_final(s,directory)
        write(directory/'summary.json',final);record['status']='complete'
    except Exception as e:
        record.update(status='failed',error=str(e));write(directory/'run.json',record);raise
    write(directory/'run.json',record);print('COMPLETE',s['run_id'],flush=True)

def reference_manifest(rows):
    references=[]
    for s in rows:
        if s['execution']!='compatible_reference':continue
        old_id=f"test_602_r{s['roi']}_{s['arm']}_s7_balanced_m0_b{s['budget']}_sil0_qfifo_hist5000000"
        folder=OLD/'test'/old_id;record=json.loads((folder/'run.json').read_text())
        if record['status']!='complete' or record['spec']['protocol']!=C['reference_protocol']:
            raise RuntimeError('old forward reference is not complete/compatible')
        for k in ('roi','arm','budget','warmup','skip','instructions','history'):
            if record['spec'][k]!=s[k]:raise RuntimeError('old forward reference differs: '+k)
        local=RAW/'compatible_references'/old_id;local.mkdir(parents=True,exist_ok=True)
        for name in ('run.json','summary.json','snapshots.jsonl'):
            target=local/name
            if target.exists() and not filecmp.cmp(folder/name,target,shallow=False):
                raise RuntimeError('preserved reference changed: '+old_id+'/'+name)
            if not target.exists():shutil.copy2(folder/name,target)
        references.append(dict(planned_spec=s,source_run=str(folder),original_record=record,
                               original_summary=str(local/'summary.json'),
                               relative_reference=str(local.relative_to(RAW)),
                               original_summary_relative=str((local/'summary.json').relative_to(RAW)),
                               status='awaiting_fresh_NoPF_compatibility_audit'))
    write(RAW/'compatible_reference_manifest.json',dict(protocol=C['protocol'],count=len(references),references=references,
          policy='Preserve original protocol/build/spec; never relabel as fresh. Final compatibility requires new NoPF vs old exact common-field audit.'))

def pipeline():
    RAW.mkdir(parents=True,exist_ok=True);lock=(RAW/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    b=assert_build_current();correctness_ready(b);fit_tables()
    rows=matrix();write(RAW/'formal_matrix.json',rows);reference_manifest(rows)
    errors=[]
    with ThreadPoolExecutor(max_workers=C['jobs']) as pool:
        for future in as_completed([pool.submit(run_one,s) for s in rows if s['execution']=='fresh']):
            try:future.result()
            except Exception as e:errors.append(str(e));print('FAILED',e,flush=True)
    if errors:raise RuntimeError('\n'.join(errors))
    write(RAW/'pipeline_result.json',dict(status='fresh_complete_reference_audit_required',protocol=C['protocol'],
          fresh_rows=64,compatible_reference_rows=7,build_id=b['build_id']))

def status():
    counts=Counter(json.loads(p.read_text())['status'] for p in RAW.glob('test/*/run.json'))
    print(json.dumps(dict(counts),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['build','fit','matrix','pipeline','status'])
    action=parser.parse_args().action
    if action=='matrix':print(json.dumps(matrix(),indent=2))
    else:{'build':build,'fit':fit_tables,'pipeline':pipeline,'status':status}[action]()
