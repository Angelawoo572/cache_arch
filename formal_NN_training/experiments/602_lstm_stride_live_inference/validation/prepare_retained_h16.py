#!/usr/bin/env python3
"""Verify/re-export the retained compatible Stride H16; certify actual build code.

This command does not train or run ChampSim. It uses real recorded inputs only
as a numerical correctness fixture and preserves all existing checkpoints.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import csv
import gzip
import importlib.util
import json
import os
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / 'formal_NN_training/experiments/602_lstm_stride_live_inference'
PREFIX = EXP/'runs/602_gcc_stride_prefix_seed7'
POINT = PREFIX/'points/h16/i1m/seed7'
DEST = EXP/'models/h16'
RAW = ROOT/'formal_study/runs/602_width_templates_20261017/stride_streamer_h16'
BASE = ROOT/'formal_study/archive/20261016/research_output/raw_v2'


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')
def inside(path):
    path=path.resolve()
    if path!=ROOT and ROOT not in path.parents:raise RuntimeError('path outside project: '+str(path))
    return path

def copy_equal(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if destination.read_bytes()!=source.read_bytes():raise RuntimeError('preserve conflicting file: '+str(destination))
    else:shutil.copy2(source,destination)
    return dict(source=str(source),destination=str(destination),bytes=destination.stat().st_size,byte_equal=True)

def canonical(path):
    return Path(str(path).replace('/home/qianruw/cache/formal_study_20261017','/home/qianruw/cache/formal_study/archive/20261017'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build-json',type=Path,default=ROOT/'formal_study/archive/20261017/raw_v2/build.json')
    ap.add_argument('--out',type=Path,default=RAW/'correctness/stride')
    args=ap.parse_args();cert=inside(args.out);cert.mkdir(parents=True,exist_ok=True)
    for key,rel in (('TMPDIR','.runtime/tmp'),('TMP','.runtime/tmp'),('TEMP','.runtime/tmp'),('XDG_CACHE_HOME','.runtime/xdg')):
        directory=ROOT/rel;directory.mkdir(parents=True,exist_ok=True);os.environ[key]=str(directory)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(EXP/'python'))
    from formal_NN_training.common.stride_direct_action_model import load_checkpoint
    from live_model_format import read_model,TENSOR_ORDER
    checkpoint=POINT/'offline/model.pt';meta=read(POINT/'offline/run_metadata.json');point=read(POINT/'point_metadata.json')
    controls=read(EXP/'config/training_prefix_sweep.json')['fixed_controls']
    manifest=read(PREFIX/'training_prefixes/i1m/training_manifest.json');current_split=read(BASE/'training_split.json')
    expected={'seed':7,'hidden_size':16,'parameter_count':5220,'model_revision':'compact_shared_pc_hurdle_delta_v9'}
    for k,v in expected.items():
        if meta.get(k)!=v or point.get(k)!=v:raise RuntimeError('retained training identity mismatch: '+k)
    if meta.get('data_derived_class_balancing_used') is not True:raise RuntimeError('retained checkpoint is not balanced')
    for k,v in dict(epochs=12,chunk_length=256,pc_batch_size=128,learning_rate=.002,optimizer='Adam').items():
        if point.get(k,controls.get(k))!=v or controls.get(k)!=v:raise RuntimeError('recipe mismatch: '+k)
    if point['instruction_budget']!=1000000 or manifest['instruction_budget']!=1000000 or manifest['training_warmup_instructions']!=0:raise RuntimeError('prefix contract mismatch')
    history=list(csv.DictReader((POINT/'offline/training_history.csv').open()))
    if len(history)!=12 or [int(r['epoch']) for r in history]!=list(range(1,13)) or sum(int(r['optimizer_steps'])for r in history)!=228:raise RuntimeError('12-epoch training history incomplete')
    for key in ('decision_rows','action_atoms','k_histogram','unique_pcs','training_simulation_instructions','training_warmup_instructions'):
        if manifest[key]!=current_split[key]:raise RuntimeError('current H8 training-data contract mismatch: '+key)
    stream=PREFIX/'training_prefixes/i1m/602.gcc_s-734B.i1m.train_stream.csv.gz'
    if gzip.open(stream,'rb').read()!=gzip.open(BASE/'models/train_inputs.csv.gz','rb').read():raise RuntimeError('current H8 and retained H16 training stream differs')
    model,payload=load_checkpoint(checkpoint,expected_hidden_size=16)
    if sum(t.numel()for t in model.parameters())!=5220:raise RuntimeError('tensor parameter count differs')
    if any(t.dtype!=torch.float32 or not torch.isfinite(t).all()for t in payload['state_dict'].values()):raise RuntimeError('nonfinite/non-FP32 checkpoint')
    bindings=[copy_equal(checkpoint,DEST/'original_checkpoint.pt'),copy_equal(POINT/'offline/run_metadata.json',DEST/'source_training_metadata.json'),copy_equal(POINT/'point_metadata.json',DEST/'source_point_metadata.json'),copy_equal(POINT/'offline/training_history.csv',DEST/'training_history.csv')]
    export=DEST
    command=[sys.executable,str(EXP/'python/export_live_model.py'),'--checkpoint',str(DEST/'original_checkpoint.pt'),'--run-metadata',str(DEST/'source_training_metadata.json'),'--out-dir',str(export),'--hidden-size','16','--instruction-budget','1000000','--seed','7','--emit-loss','balanced','--allow-legacy-checkpoint']
    if not (export/'model.bin').exists():
        with (cert/'export.log').open('w')as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    binary=read_model(export/'model.bin')
    for name in TENSOR_ORDER:
        observed=binary['tensors'][name]
        if not (observed==payload['state_dict'][name].numpy()).all():raise RuntimeError('export tensor mismatch: '+name)
    if (POINT/'export/model.bin').read_bytes()!=(export/'model.bin').read_bytes():raise RuntimeError('new export differs from original portable tensor file')
    b=read(inside(args.build_json));sim=canonical(b['simulator']);inside(sim)
    generator_path=ROOT/'formal_NN_training/experiments/602_lstm_stride_dynamic_inference/runtime/prepare_forward.py'
    module_spec=importlib.util.spec_from_file_location('stride_forward_for_h16_certificate',generator_path)
    generator=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(generator)
    forward=generator.verify_prepared(ROOT,sim,require_marker=True)
    for name in ('stride_lstm_runtime.h','stride_lstm_model_loader.h'):
        bindings.append(copy_equal(sim/'inc'/name,cert/'actual_build_source/inc'/name))
    for name in ('stride_lstm_runtime.cc','stride_lstm_model_loader.cc'):
        bindings.append(copy_equal(sim/'prefetcher'/name,cert/'actual_build_source/prefetcher'/name))
    bindings.append(copy_equal(EXP/'runtime/standalone_runner.cc',cert/'actual_build_source/standalone_runner.cc'))
    probe=cert/'stride_lstm_standalone'
    compile_command=['g++','-std=c++11','-O3','-ffp-contract=off','-I'+str(sim/'inc'),str(sim/'prefetcher/stride_lstm_runtime.cc'),str(sim/'prefetcher/stride_lstm_model_loader.cc'),str(cert/'actual_build_source/standalone_runner.cc'),'-o',str(probe)]
    with (cert/'compile.log').open('w')as log:subprocess.run(compile_command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    parity_script=EXP/'validation/compare_python_cpp_outputs.py'
    commands=[]
    for label,extra in [('synthetic',['--synthetic']),('recorded',['--checkpoint',str(DEST/'original_checkpoint.pt'),'--model-bin',str(export/'model.bin'),'--stream',str(BASE/'models/validation_inputs.csv.gz'),'--max-events','4096'])]:
        cmd=[sys.executable,str(parity_script),'--cpp-runner',str(probe),'--work-dir',str(cert/label),'--output',str(cert/(label+'_parity.json'))]+extra
        commands.append(cmd)
        with (cert/(label+'_parity.log')).open('w')as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    recorded=read(cert/'recorded_parity.json');synthetic=read(cert/'synthetic_parity.json')
    report=dict(status='PASS',policy='stride',hidden_size=16,checkpoint_reused=True,training_performed=False,seed=7,emit_loss='balanced',training_instruction_interval=[0,1000000],training_rows=16172,training_epochs=12,optimizer_steps=228,training_stream_byte_equal_current_H8=True,legacy_identity_source='original run/point metadata and complete history; legacy checkpoint has no embedded seed/loss',parameter_count=5220,FP32_parameter_bytes=20880,recurrent_hc_bytes_per_pc=128,weight_and_decoder_semantics='shared projected128-feature v9 weights; exact-PC h/c; hurdle exp count; deterministic free-running GRU delta; L2 fill',deployment_controls='same current H8 per-PC64/decoder32/current-line-filter; correctness fixture compares full mathematical runtime',original_H8_note='retained prefix H8 and current formal CPU-retrained H8 have small weight differences; recipe agreement is not checkpoint identity',model_bin=str(export/'model.bin'),checkpoint=str(DEST/'original_checkpoint.pt'),metadata=str(DEST/'model_metadata.json'),original_checkpoint=str(checkpoint),actual_build_id=b['build_id'],actual_simulator=str(sim),forward_generation=forward,compile_command=compile_command,parity_commands=commands,recorded=recorded,synthetic=synthetic,bindings=bindings)
    write(cert/'compatibility_and_parity.json',report)
    # Keep the lightweight route-side compatibility record independent of raw logs.
    summary={k:v for k,v in report.items()if k not in ('bindings','compile_command','parity_commands')}
    write(DEST/'compatibility.json',summary)
    print(json.dumps(dict(status='PASS',model=str(export/'model.bin'),parameters=5220,certificate=str(cert/'compatibility_and_parity.json'),recorded_events=recorded['events'],max_hidden_error=recorded['max_hidden_state_error'],max_cell_error=recorded['max_cell_state_error'])))

if __name__=='__main__':main()
