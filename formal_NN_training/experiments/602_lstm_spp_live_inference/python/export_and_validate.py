#!/usr/bin/env python3
"""Export the actual frozen SPP student and compare its complete causal state/actions.

The correctness fixture is not a simulator pilot or an IPC sample. All formal
simulator rows run the full stored warmup and measurement instruction budgets.
"""
import argparse
import csv
import gzip
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from formal_NN_training.common.live602_runtime.model_format import write_model


def source_module():
    path=ROOT/'formal_NN_training/experiments/602_offline_lstm_spp/python/train_and_offline_infer.py'
    spec=importlib.util.spec_from_file_location('real_spp_training',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def checkpoint(path):
    import torch
    data=torch.load(path,map_location='cpu',weights_only=True)
    if (data.get('model_family')!='lstm' or data.get('runtime_features')!=65
        or tuple(data.get('fill_levels',()))!=(2,4) or data.get('mixture_components')!=4
        or data.get('model_revision')!='compact_empirical_prior_hurdle_autoregressive_gmm_fill_v2'):
        raise ValueError('Not the real empirical-prior SPP checkpoint')
    return data


def export(path,out):
    data=checkpoint(path);state=data['state_dict']
    mapping={
        'lstm.weight_ih_l0':'encoder.weight_ih','lstm.weight_hh_l0':'encoder.weight_hh',
        'lstm.bias_ih_l0':'encoder.bias_ih','lstm.bias_hh_l0':'encoder.bias_hh',
        'decoder.positive_log_count.weight':'count.weight','decoder.positive_log_count.bias':'count.bias',
        'decoder.emit_head.weight':'emit.weight','decoder.emit_head.bias':'emit.bias',
        'decoder.action_cell.weight_ih':'gru.weight_ih','decoder.action_cell.weight_hh':'gru.weight_hh',
        'decoder.action_cell.bias_ih':'gru.bias_ih','decoder.action_cell.bias_hh':'gru.bias_hh',
        'decoder.delta_head.weight':'delta.weight','decoder.delta_head.bias':'delta.bias',
        'decoder.fill_head.weight':'fill.weight','decoder.fill_head.bias':'fill.bias'}
    if set(mapping)!=set(state):raise ValueError('SPP tensor set changed')
    config=write_model(out,{v:state[k] for k,v in mapping.items()},hidden_size=data['model_size'],
        feature_count=65,mixture_components=4,fill_classes=2,count_mode=1,
        decoder_initial=1,kind_feature=1,demand_kind=1,delta_limit_bits=57,delta_mode=0)
    metadata=dict(policy='spp',checkpoint=str(Path(path).resolve()),model=str(Path(out).resolve()),
        architecture='global LSTM65 + empirical-prior categorical gate + positive exp-count + autoregressive GMM4/GRU3/fill2',
        frozen_weights=True,pc_is_neural_input=False,state_organization='one chronological global h/c pair',
        training_records=[0,20000000],old_guard_records=[20000000,25000000],old_observed_eval_records=[25000000,50000000],
        callback_contract='all actual raw CACHE L2-fill call sites (evicted address), plus eligible L2 LOAD demand address',
        native_feedback_contract='native registry forwards SPP FILTER feedback only for prefetch fills; preserved separately',
        original_fill_levels=[2,4],new_page_filter=False,new_degree_limit=None,**config)
    Path(out).with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def sequence(path,limit):
    result=[]
    with gzip.open(path,'rt',newline='') as f:
        for row in csv.DictReader(f):
            result.append(('D' if row['event_kind']=='DEMAND' else 'F',int(row['event_address'],0)))
            if len(result)>=limit:break
    # Address boundary checks are explicit correctness inputs, not test fitting.
    edge=[0,64,4096-64,4096,(1<<63),((1<<64)-64),64,0]
    result += [(kind,address) for address in edge for kind in ('F','D')]
    return result


def validate(path,model_bin,probe,stream,out,limit=4096):
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.backends.mkldnn.enabled=False
    module=source_module();data=checkpoint(path)
    model=module.CompactSPPLSTM(65,data['model_size']);model.load_state_dict(data['state_dict'],strict=True);model.eval()
    events=sequence(stream,limit)
    text=''.join('{} {}\n'.format(kind,address) for kind,address in events)
    run=subprocess.run([str(probe),str(model_bin)],input=text,text=True,capture_output=True,check=True)
    actual=[json.loads(line) for line in run.stdout.splitlines() if line.strip()]
    if len(actual)!=len(events):raise ValueError('C++ callback result count differs')
    state=None;max_h=0.0;max_c=0.0;actions=0;fills=0;demands=0;differences=[]
    with torch.no_grad():
        for index,((kind,address),cpp) in enumerate(zip(events,actual)):
            address &= ((1<<64)-64)
            bits=module.runtime_bits([0],[address],False)
            runtime=np.concatenate([bits,np.asarray([[float(kind=='D')]],dtype=np.float32)],axis=1)
            encoded,state=model.encode(torch.from_numpy(runtime).reshape(1,1,65),state)
            h=state[0].reshape(-1).numpy();c=state[1].reshape(-1).numpy()
            ch=np.asarray(cpp['hidden']);cc=np.asarray(cpp['cell'])
            max_h=max(max_h,float(np.max(np.abs(h-ch))));max_c=max(max_c,float(np.max(np.abs(c-cc))))
            expected=[];count=0
            if kind=='D':
                demands+=1;context=encoded.reshape(1,-1)
                gate=model.decoder.emit_head(context).numpy()
                positive=model.decoder.positive_log_count(context).numpy().reshape(-1)
                counts,lines,levels=module.decode_actions(model,gate,positive,context.numpy(),[address>>6],torch.device('cpu'))
                count=int(counts[0]);expected=[(int(line)<<6,int(fill)) for line,fill in zip(lines[0],[module.FILL_LEVELS[k] for k in levels[0]])]
            else:fills+=1
            observed=[(int(a['address']),int(a['fill'])) for a in cpp['actions']]
            if int(cpp['count'])!=count or expected!=observed:
                differences.append(dict(index=index,kind=kind,address=address,python=expected,cpp=observed,python_count=count,cpp_count=cpp['count']))
            actions+=len(expected)
    report=dict(status='PASS' if max(max_h,max_c)<2e-5 and not differences else 'FAIL',
        policy='spp',checkpoint=str(path),model=str(model_bin),sequence_source=str(stream),
        callback_count=len(events),demand_callbacks=demands,fill_callbacks=fills,actions_compared=actions,
        max_abs_hidden_error=max_h,max_abs_cell_error=max_c,absolute_tolerance=2e-5,
        exact_action_sequence_equal=not differences,mismatches=differences[:16],
        scope='complete states and all ordered address/fill actions for a chronological correctness fixture plus uint64 boundary inputs; not an IPC sample')
    Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(report,indent=2)+'\n')
    if report['status']!='PASS':raise RuntimeError(json.dumps(report))
    print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--probe',type=Path);p.add_argument('--stream',type=Path);p.add_argument('--report',type=Path);p.add_argument('--events',type=int,default=4096)
    a=p.parse_args();export(a.checkpoint,a.model)
    if a.probe:validate(a.checkpoint,a.model,a.probe,a.stream,a.report,a.events)
