#!/usr/bin/env python3
"""Retained real AMPM H16 exporter/PyTorch parity, isolated from archived H8."""
import argparse,importlib.util,json,math,subprocess,sys,csv,gzip,itertools
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))
from formal_NN_training.common.live602_runtime.model_format import canonicalize_poisson_lstm,write_model,read_model
IDENTITY=dict(hidden_size=16,feature_count=64,parameters=6653,matched_normal_prefetcher='ampm',experiment_revision='source_input_variable_delta_free_running_v7')
def checkpoint_reader():
    path=ROOT/'formal_NN_training/experiments/602_lstm_ampm_live_inference/python/export_model.py'
    spec=importlib.util.spec_from_file_location('archived_ampm_checkpoint_reader',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def export(checkpoint,metadata,output):
    data=checkpoint_reader().load_checkpoint(checkpoint);meta=json.loads(Path(metadata).read_text())
    for key,value in IDENTITY.items():
        if data.get(key)!=value:raise ValueError('not the retained real AMPM H16 checkpoint: '+key)
    if meta.get('seed')!=7 or meta.get('matched_normal_prefetcher')!='ampm' or meta.get('parameter_count')!=6653:raise ValueError('AMPM H16 metadata mismatch')
    tensors=canonicalize_poisson_lstm(data['state_dict'])
    config=write_model(output,tensors,hidden_size=16,feature_count=64)
    reread,actual=read_model(output)
    if reread['hidden_size']!=16 or reread['parameter_count']!=6653:raise ValueError('H16 export header mismatch')
    for name,t in tensors.items():
        if tuple(t['shape'])!=actual[name]['shape'] or list(t['data'])!=actual[name]['data']:raise ValueError('FP32 tensor roundtrip mismatch '+name)
    result=dict(status='PASS',policy='ampm',hidden_size=16,seed=7,checkpoint=str(Path(checkpoint).resolve()),source_metadata=str(Path(metadata).resolve()),model=str(Path(output).resolve()),checkpoint_identity=IDENTITY,binary_config=config,all_tensors_exact_roundtrip=True,training_records=[0,20000000],validation_records=[20000000,25000000],state='one global chronological h/c pair',count='float64 softplus of FP32 affine, then floor; no degree cap',fill_levels=[2],global_hc_bytes=128)
    Path(output).with_name('model_metadata.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
def integer(value):
    value=str(value);return int(value,16) if value.lower().startswith('0x') else int(value,10)
def sequence(stream,events):
    opener=gzip.open if str(stream).endswith('.gz') else open
    with opener(stream,'rt',newline='') as handle:
        for index,row in enumerate(itertools.islice(csv.DictReader(handle),events)):
            if 'demand_idx' in row and integer(row['demand_idx'])!=index:raise ValueError('noncausal fixture index')
            yield integer(row['line'])<<6 if 'line' in row else integer(row['address'])
class Reference:
    def __init__(self,path):
        import torch
        from formal_NN_training.common.threshold_free_policy import build_model,_coordinate_to_delta,apply_signed_line_delta
        torch.set_num_threads(1);torch.backends.mkldnn.enabled=False
        data=torch.load(path,map_location='cpu',weights_only=True)
        for key,value in IDENTITY.items():
            if data.get(key)!=value:raise ValueError('wrong true PyTorch AMPM H16 checkpoint '+key)
        self.torch=torch;self.delta=_coordinate_to_delta;self.apply=apply_signed_line_delta
        self.model,_=build_model('lstm',64,16,fill_classes=0,mixture_components=4)
        self.model.load_state_dict(data['state_dict'],strict=True);self.model.eval()
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        self.Reset()
    def Reset(self):self.state=None
    def Infer(self,address):
        torch=self.torch
        if address<0 or address>=2**64 or address&63:raise ValueError('aligned uint64 byte address required')
        bits=torch.tensor([[(address>>i)&1 for i in range(64)]],dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            context,self.state=self.model.encode(bits,self.state);context=context[0,0].unsqueeze(0)
            raw=float(self.model.decoder.count_raw(context).item())
            rate=max(raw,0.0)+math.log1p(math.exp(-abs(raw)))
            if not math.isfinite(rate) or rate>=2**63:raise ValueError('Poisson count outside int64 domain')
            count=math.floor(rate);actions=[]
            if count:
                decoder=self.model.decoder.begin(context)
                for _ in range(count):
                    mix,means,_,fill=self.model.decoder.distribution(decoder)
                    coordinate=means[:,int(mix.argmax(-1).item())];value=float(coordinate.item());delta=self.delta(value)
                    actions.append((self.apply(address>>6,delta)<<6,delta,2,-1,value))
                    decoder=self.model.decoder.advance(decoder,coordinate)
        return dict(hidden=self.state[0][0,0].tolist(),cell=self.state[1][0,0].tolist(),count=count,emit=bool(count),actions=actions,count_raw=raw)
def parity(checkpoint,model,probe,stream,out,events):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    config,_=read_model(model)
    if config['hidden_size']!=16 or config['parameter_count']!=6653:raise ValueError('not H16 binary')
    inputs=list(sequence(stream,events))
    if len(inputs)!=events:raise ValueError('insufficient real correctness callbacks')
    inputs += [0,64,4032,4096,1<<63,(1<<64)-64,64,0]
    commands=''.join('D '+str(v)+'\n' for v in inputs);(out/'commands.txt').write_text(commands)
    process=subprocess.run([str(probe),str(model)],input=commands,text=True,capture_output=True,check=True)
    (out/'cpp.jsonl').write_text(process.stdout)
    actual=[json.loads(v) for v in process.stdout.splitlines()]
    if len(actual)!=len(inputs):raise AssertionError('wrong C++ result count')
    ref=Reference(checkpoint);max_h=max_c=max_coord=0.;actions=0
    for index,(address,cpp) in enumerate(zip(inputs,actual)):
        expected=ref.Infer(address)
        for name in ('hidden','cell'):
            if len(cpp[name])!=16 or len(expected[name])!=16:raise AssertionError('wrong H16 state shape')
            error=max(abs(x-y) for x,y in zip(cpp[name],expected[name]))
            if name=='hidden':max_h=max(max_h,error)
            else:max_c=max(max_c,error)
            if error>2e-5:raise AssertionError((index,name,error))
        if cpp['count']!=expected['count'] or cpp['emit']!=expected['emit'] or cpp['truncated']:raise AssertionError((index,'count/emit/cap'))
        observed=[(x['address'],x['delta'],x['fill'],x['fill_index']) for x in cpp['actions']]
        wanted=[x[:4] for x in expected['actions']]
        if observed!=wanted:raise AssertionError((index,'actions',observed,wanted))
        for x,y in zip(cpp['actions'],expected['actions']):
            error=abs(x['coordinate']-y[4]);max_coord=max(max_coord,error)
            if error>2e-5:raise AssertionError((index,'coordinate',error))
        k=expected['count']
        if cpp['macs']!=5136+256*bool(k)+1008*k or cpp['nonlinears']!=81+16*bool(k)+53*k:raise AssertionError('H16 actual operation count mismatch')
        actions+=k
    reset=subprocess.run([str(probe),str(model)],input='D 64\nR\nD 64\n',text=True,capture_output=True,check=True)
    first,_,again=map(json.loads,reset.stdout.splitlines())
    if first!=again:raise AssertionError('Reset state mismatch')
    result=dict(status='PASS',policy='ampm',hidden_size=16,seed=7,checkpoint=str(checkpoint),model=str(model),source_stream=str(stream),real_contiguous_callbacks=events,boundary_callbacks=8,callbacks=len(inputs),actions=actions,exact_counts_actions=True,all_hidden_cell_compared=True,max_hidden_abs_error=max_h,max_cell_abs_error=max_c,max_coordinate_abs_error=max_coord,absolute_tolerance=2e-5,reset_verified=True,actual_operation_counts_verified=True,torch_mkldnn=False,scope='causal correctness fixture and address boundaries only, no simulator pilot, no training')
    (out/'parity.json').write_text(json.dumps(result,indent=2)+'\n');return result
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['export','parity']);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--metadata',type=Path);p.add_argument('--probe',type=Path);p.add_argument('--stream',type=Path);p.add_argument('--out',type=Path);p.add_argument('--events',type=int,default=4096);a=p.parse_args()
    result=export(a.checkpoint,a.metadata,a.model) if a.command=='export' else parity(a.checkpoint,a.model,a.probe,a.stream,a.out,a.events)
    print(json.dumps(result))
if __name__=='__main__':main()
