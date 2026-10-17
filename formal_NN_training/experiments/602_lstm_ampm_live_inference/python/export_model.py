#!/usr/bin/env python3
"""Export the real retained AMPM v7 checkpoint without importing Torch."""
import argparse
import array
import collections
import io
import json
import math
from pathlib import Path
import pickle
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from formal_NN_training.common.live602_runtime.model_format import canonicalize_poisson_lstm,write_model,read_model


class CheckpointReader(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
        if (module,name)==('torch._utils','_rebuild_tensor_v2'):
            return lambda storage,offset,size,stride,*unused:dict(storage=storage,offset=offset,shape=tuple(size),stride=tuple(stride))
        if (module,name)==('torch','FloatStorage'):return 'FloatStorage'
        raise ValueError('unsupported checkpoint symbol '+module+'.'+name)
    def persistent_load(self,key):
        if key[0]!='storage' or key[1]!='FloatStorage':raise ValueError('checkpoint is not plain FP32 storage')
        return dict(key=key[2],location=key[3],count=key[4])


def load_checkpoint(path):
    with zipfile.ZipFile(path) as archive:
        name=next(name for name in archive.namelist() if name.endswith('/data.pkl'))
        prefix=name[:-len('data.pkl')]
        checkpoint=CheckpointReader(io.BytesIO(archive.read(name))).load()
        tensors={}
        for name,value in checkpoint['state_dict'].items():
            shape=value['shape'];stride=[];running=1
            for dimension in reversed(shape):stride.insert(0,running);running*=dimension
            if tuple(stride)!=value['stride']:raise ValueError('noncontiguous source tensor '+name)
            data=array.array('f');data.frombytes(archive.read(prefix+'data/'+str(value['storage']['key'])))
            if sys.byteorder!='little':data.byteswap()
            if len(data)!=value['storage']['count']:raise ValueError('truncated source storage')
            data=data[value['offset']:value['offset']+math.prod(shape)]
            tensors[name]=dict(shape=shape,data=data)
        checkpoint['state_dict']=tensors
        return checkpoint


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();checkpoint=load_checkpoint(args.checkpoint);metadata=json.loads(args.metadata.read_text())
    required=dict(hidden_size=8,feature_count=64,parameters=2821,matched_normal_prefetcher='ampm',experiment_revision='source_input_variable_delta_free_running_v7')
    for key,value in required.items():
        if checkpoint.get(key)!=value:raise ValueError('wrong real AMPM checkpoint identity: '+key)
    if metadata['seed']!=7 or metadata['matched_normal_prefetcher']!='ampm' or metadata['parameter_count']!=2821:
        raise ValueError('wrong AMPM retained metadata')
    tensors=canonicalize_poisson_lstm(checkpoint['state_dict'])
    spec=write_model(args.output,tensors,hidden_size=8,feature_count=64)
    reread,actual=read_model(args.output)
    if reread['parameter_count']!=2821:raise ValueError('export parameter mismatch')
    for name,value in tensors.items():
        if actual[name]['shape']!=tuple(value['shape']) or actual[name]['data']!=list(value['data']):raise ValueError('export tensor roundtrip failed')
    result=dict(model_family='real_ampm_v7_global_lstm_poisson_mixture',matched_normal_prefetcher='ampm',seed=7,
        checkpoint=str(args.checkpoint.resolve()),source_metadata=str(args.metadata.resolve()),model=str(args.output.resolve()),
        checkpoint_identity=required,binary_config=spec,all_tensors_exact_roundtrip=True,
        original_training_instruction_records=[0,20000000],original_guard_instruction_records=[20000000,25000000],
        original_evaluation_instruction_records=[25000000,50000000],new_validation_instruction_records=[20000000,25000000],
        source_train_rows=metadata['train_rows'],source_guard_rows=metadata['guard_rows'],source_eval_rows=metadata['eval_rows'],
        state='one global chronological FP32 hidden/cell pair; decoder reset from context per event; noPC state',
        runtime_input='64 LSB-first bits of aligned uint64 byte address (line<<6), low6 bits naturallyzero',
        count='floor(float64 softplus(FP32 count affine)), permits0; reject>=2**63',
        delta='4-mixture logits argmax first; selected mean; signed expm1 then ties-even round; modulo2**58 line domain',
        fill='no learned fill head; fill_level0=FILL_L2(2) at external adapter',
        old_results='keyed address-list replay; not live recurrent or finite-cost inference evidence')
    args.output.with_name('model_metadata.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='exported_real_ampm_v7',parameters=2821,weight_bytes=11284,model=str(args.output))))

if __name__=='__main__':main()
