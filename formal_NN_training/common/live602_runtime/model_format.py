"""Shared LV602R1 portable FP32 tensor container; no Torch import required."""
import array
import math
from pathlib import Path
import struct
import sys

MAGIC=b'LV602R1\0'
HEADER=struct.Struct('<8s16IQ')
TENSOR_HEADER=struct.Struct('<HHQ')
FIELDS=('version','endian','float_type','tensor_count','hidden_size','feature_count',
        'mixture_components','fill_classes','count_mode','decoder_initial',
        'kind_feature','demand_kind','delta_limit_bits','fill_level0','fill_level1','delta_mode')
DEFAULTS=dict(version=1,endian=0x01020304,float_type=1,hidden_size=8,feature_count=64,
              mixture_components=4,fill_classes=0,count_mode=0,decoder_initial=0,
              kind_feature=0,demand_kind=0,delta_limit_bits=57,fill_level0=2,fill_level1=4,delta_mode=0)


def expected_shapes(config):
    h=config['hidden_size'];f=config['feature_count'];d=1+config['fill_classes']
    pairs=[('encoder.weight_ih',(4*h,f)),('encoder.weight_hh',(4*h,h)),
           ('encoder.bias_ih',(4*h,)),('encoder.bias_hh',(4*h,)),
           ('count.weight',(1,h)),('count.bias',(1,))]
    if config['count_mode']==1:pairs += [('emit.weight',(2,h)),('emit.bias',(2,))]
    if config['decoder_initial']==0:pairs += [('initial.weight',(h,h)),('initial.bias',(h,))]
    pairs += [('gru.weight_ih',(3*h,d)),('gru.weight_hh',(3*h,h)),
              ('gru.bias_ih',(3*h,)),('gru.bias_hh',(3*h,))]
    rows=3*config['mixture_components'] if config['delta_mode']==0 else 1
    pairs += [('delta.weight',(rows,h)),('delta.bias',(rows,))]
    if config['fill_classes']:pairs += [('fill.weight',(config['fill_classes'],h)),('fill.bias',(config['fill_classes'],))]
    return dict(pairs)


def _parts(value):
    if isinstance(value,dict):shape=tuple(value['shape']);data=array.array('f',value['data'])
    else:
        if hasattr(value,'detach'):value=value.detach().cpu().contiguous().numpy()
        shape=tuple(value.shape);data=array.array('f',value.reshape(-1).tolist())
    if math.prod(shape)!=len(data) or not all(math.isfinite(v) for v in data):
        raise ValueError('invalid tensor dimensions or nonfinite data')
    if sys.byteorder!='little':data.byteswap()
    return shape,data.tobytes()


def write_model(path,tensors,**configuration):
    config=dict(DEFAULTS,**configuration);shapes=expected_shapes(config)
    if set(tensors)!=set(shapes):raise ValueError('tensor set mismatch: '+str(set(tensors)^set(shapes)))
    values=[];parameters=0
    for name,expected in shapes.items():
        shape,data=_parts(tensors[name])
        if shape!=expected:raise ValueError('tensor shape mismatch '+name+': '+str(shape)+' != '+str(expected))
        values.append((name,shape,data));parameters+=len(data)//4
    config['tensor_count']=len(values)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as handle:
        handle.write(HEADER.pack(MAGIC,*[config[key] for key in FIELDS],parameters))
        for name,shape,data in values:
            name=name.encode('ascii');count=len(data)//4
            handle.write(TENSOR_HEADER.pack(len(name),len(shape),count))
            handle.write(struct.pack('<'+'I'*len(shape),*shape));handle.write(name);handle.write(data)
    return dict(config,parameter_count=parameters,weight_bytes=parameters*4)


def read_model(path):
    with Path(path).open('rb') as handle:
        values=HEADER.unpack(handle.read(HEADER.size))
        if values[0]!=MAGIC:raise ValueError('bad LV602R1 magic')
        config=dict(zip(FIELDS,values[1:17]));config['parameter_count']=values[17]
        tensors={}
        for _ in range(config['tensor_count']):
            name_size,rank,count=TENSOR_HEADER.unpack(handle.read(TENSOR_HEADER.size))
            shape=struct.unpack('<'+'I'*rank,handle.read(rank*4));name=handle.read(name_size).decode('ascii')
            data=array.array('f');data.frombytes(handle.read(count*4))
            if sys.byteorder!='little':data.byteswap()
            if math.prod(shape)!=len(data):raise ValueError('truncated tensor')
            tensors[name]=dict(shape=shape,data=list(data))
        if handle.read(1):raise ValueError('trailing model bytes')
    return config,tensors


def canonicalize_poisson_lstm(state_dict):
    mapping={'lstm.weight_ih_l0':'encoder.weight_ih','lstm.weight_hh_l0':'encoder.weight_hh',
             'lstm.bias_ih_l0':'encoder.bias_ih','lstm.bias_hh_l0':'encoder.bias_hh',
             'decoder.count_head.weight':'count.weight','decoder.count_head.bias':'count.bias',
             'decoder.initial_state.weight':'initial.weight','decoder.initial_state.bias':'initial.bias',
             'decoder.action_cell.weight_ih':'gru.weight_ih','decoder.action_cell.weight_hh':'gru.weight_hh',
             'decoder.action_cell.bias_ih':'gru.bias_ih','decoder.action_cell.bias_hh':'gru.bias_hh',
             'decoder.delta_head.weight':'delta.weight','decoder.delta_head.bias':'delta.bias'}
    if set(state_dict)!=set(mapping):raise ValueError('not the real14-tensor Poisson/mixture LSTM checkpoint')
    return {canonical:state_dict[source] for source,canonical in mapping.items()}
