#!/usr/bin/env python3
"""Tensor-verified parameter/state accounting and reader diagrams; no simulation.

Default: verify all eight actual exports. --verify-checkpoints additionally
compares every exported FP32 byte with its retained Torch checkpoint.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import json
import math
import os
from pathlib import Path
import struct
import tempfile

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'formal_study/results/602_width_templates'
for variable,relative in (('TMPDIR','.runtime/tmp'),('TEMP','.runtime/tmp'),('TMP','.runtime/tmp'),('MPLCONFIGDIR','.runtime/mpl'),('XDG_CACHE_HOME','.runtime/xdg'),('TORCH_HOME','.runtime/torch')):
    directory=ROOT/relative;directory.mkdir(parents=True,exist_ok=True);os.environ[variable]=str(directory)
tempfile.tempdir=str(ROOT/'.runtime/tmp')
POLICIES=('stride','streamer','ampm','spp')
LAYERS=('input_projection','encoder_lstm','emit_gate','count_head','decoder_initial','decoder_gru','delta_head','fill_head')
NAMES=dict(stride='Stride',streamer='Streamer',ampm='AMPM',spp='SPP')
STRIDE_H8='formal_study/archive/20261016/research_output/raw_v2/models/seed7/balanced'


def specs():
    result=[]
    for policy in POLICIES:
        for h in (8,16):
            base=Path('formal_NN_training/experiments/602_lstm_'+policy+'_live_inference/models')
            if h==16:base=base/'h16'
            if policy=='stride' and h==8:
                model=Path(STRIDE_H8)/'export/model.bin';checkpoint=Path(STRIDE_H8)/'offline/model.pt'
            else:model=base/'model.bin';checkpoint=base/'original_checkpoint.pt'
            result.append(dict(policy=policy,hidden_size=h,model_path=str(model),checkpoint_path=str(checkpoint)))
    return result


def read_exact(stream,n):
    b=stream.read(n)
    if len(b)!=n:raise ValueError('truncated model')
    return b


def expected_shapes(policy,h):
    if policy=='stride':
        return {'input_projection.weight':(h,128),'input_projection.bias':(h,),
                'encoder_lstm.weight_ih_l0':(4*h,h),'encoder_lstm.weight_hh_l0':(4*h,h),
                'encoder_lstm.bias_ih_l0':(4*h,),'encoder_lstm.bias_hh_l0':(4*h,),
                'emit_head.weight':(2,h),'emit_head.bias':(2,),
                'log_count_mean.weight':(1,h),'log_count_mean.bias':(1,),
                'action_decoder.action_cell.weight_ih':(3*h,1),'action_decoder.action_cell.weight_hh':(3*h,h),
                'action_decoder.action_cell.bias_ih':(3*h,),'action_decoder.action_cell.bias_hh':(3*h,),
                'action_decoder.delta_head.weight':(1,h),'action_decoder.delta_head.bias':(1,)}
    f=65 if policy=='spp' else 64;d=3 if policy=='spp' else 1
    shapes={'encoder.weight_ih':(4*h,f),'encoder.weight_hh':(4*h,h),
            'encoder.bias_ih':(4*h,),'encoder.bias_hh':(4*h,),
            'count.weight':(1,h),'count.bias':(1,),
            'gru.weight_ih':(3*h,d),'gru.weight_hh':(3*h,h),'gru.bias_ih':(3*h,),'gru.bias_hh':(3*h,),
            'delta.weight':(12,h),'delta.bias':(12,)}
    if policy=='spp':shapes.update({'emit.weight':(2,h),'emit.bias':(2,),'fill.weight':(2,h),'fill.bias':(2,)})
    else:shapes.update({'initial.weight':(h,h),'initial.bias':(h,)})
    return shapes


def parse_model(path,policy,h):
    with path.open('rb') as stream:
        magic=read_exact(stream,8)
        if magic==b'STRLSTM1':
            values=struct.unpack('<6IQ',read_exact(stream,32))
            version,endian,float_type,count,width,features,parameters=values
            config=dict(format='STRLSTM1',version=version,endian=endian,float_type=float_type,tensor_count=count,hidden_size=width,feature_count=features,parameter_count=parameters)
            assert policy=='stride'
        elif magic==b'LV602R1\0':
            values=struct.unpack('<16IQ',read_exact(stream,72))
            keys=('version','endian','float_type','tensor_count','hidden_size','feature_count','mixture_components','fill_classes','count_mode','decoder_initial','kind_feature','demand_kind','delta_limit_bits','fill_level0','fill_level1','delta_mode','parameter_count')
            config=dict(zip(keys,values));config['format']='LV602R1'
            assert policy!='stride'
            assert config['mixture_components']==4 and config['delta_mode']==0
            assert config['fill_classes']==(2 if policy=='spp' else 0)
            assert config['count_mode']==(1 if policy=='spp' else 0)
            assert config['decoder_initial']==(1 if policy=='spp' else 0)
        else:raise ValueError('unrecognized portable model: '+str(path))
        assert config['version']==1 and config['endian']==0x01020304 and config['float_type']==1
        assert config['hidden_size']==h
        assert config['feature_count']==(128 if policy=='stride' else 65 if policy=='spp' else 64)
        tensors={}
        for _ in range(config['tensor_count']):
            name_length,rank,count=struct.unpack('<HHQ',read_exact(stream,12))
            assert name_length>0 and 0<rank<=4
            shape=struct.unpack('<'+'I'*rank,read_exact(stream,rank*4))
            name=read_exact(stream,name_length).decode('ascii');data=read_exact(stream,count*4)
            assert name not in tensors and math.prod(shape)==count
            assert all(math.isfinite(x[0]) for x in struct.iter_unpack('<f',data)),name
            tensors[name]=dict(shape=shape,count=count,data=data)
        assert stream.read(1)==b''
    assert {n:t['shape'] for n,t in tensors.items()}==expected_shapes(policy,h)
    assert sum(t['count'] for t in tensors.values())==config['parameter_count']
    return config,tensors


def layer(name):
    if name.startswith('input_projection.'):return 'input_projection'
    if name.startswith(('encoder.','encoder_lstm.')):return 'encoder_lstm'
    if name.startswith(('emit.','emit_head.')):return 'emit_gate'
    if name.startswith(('count.','log_count_mean.')):return 'count_head'
    if name.startswith('initial.'):return 'decoder_initial'
    if name.startswith(('gru.','action_decoder.action_cell.')):return 'decoder_gru'
    if name.startswith(('delta.','action_decoder.delta_head.')):return 'delta_head'
    if name.startswith('fill.'):return 'fill_head'
    raise ValueError('unclassified tensor '+name)


def canonical_checkpoint_name(name,policy):
    if policy=='stride':return name
    if name.startswith('lstm.'):
        return 'encoder.'+name[len('lstm.'):].replace('_l0','')
    replacements=(('decoder.count_head.','count.'),('decoder.positive_log_count.','count.'),
                  ('decoder.emit_head.','emit.'),('decoder.initial_state.','initial.'),
                  ('decoder.action_cell.','gru.'),('decoder.delta_head.','delta.'),('decoder.fill_head.','fill.'))
    for source,dest in replacements:
        if name.startswith(source):return dest+name[len(source):]
    raise ValueError('unrecognized checkpoint tensor '+name)


def verify_checkpoint(path,policy,tensors):
    import torch
    torch.set_num_threads(1)
    try:checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    except TypeError:checkpoint=torch.load(path,map_location='cpu')
    state=checkpoint['state_dict'];canonical={canonical_checkpoint_name(name,policy):value for name,value in state.items()}
    assert set(canonical)==set(tensors)
    for name,value in canonical.items():
        assert value.dtype==torch.float32 and tuple(value.shape)==tensors[name]['shape'],name
        assert value.detach().cpu().contiguous().numpy().astype('<f4',copy=False).tobytes()==tensors[name]['data'],name
    return dict(status='PASS',tensors=len(canonical),parameter_count=sum(v.numel() for v in canonical.values()),all_FP32_tensor_bytes_equal_export=True)


def formula(policy,h):
    return 11*h*h+150*h+4 if policy=='stride' else 7*h*h+300*h+17 if policy=='spp' else 8*h*h+287*h+13


def dump(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def inventory(verify_checkpoints=False,allow_missing=False):
    models=[];tensor_rows=[];layer_rows=[];checks=[]
    for spec in specs():
        policy=spec['policy'];h=spec['hidden_size'];path=ROOT/spec['model_path']
        if not path.exists():
            if allow_missing:
                checks.append(dict(**spec,status='MISSING_EXPORT'));continue
            raise FileNotFoundError('Actual compatible export is required: '+str(path))
        config,tensors=parse_model(path,policy,h);parameters=config['parameter_count']
        assert parameters==formula(policy,h)
        groups={name:0 for name in LAYERS}
        for name,tensor in tensors.items():
            group=layer(name);groups[group]+=tensor['count']
            tensor_rows.append(dict(policy=policy,hidden_size=h,layer=group,tensor=name,shape='×'.join(str(x) for x in tensor['shape']),parameters=tensor['count'],FP32_bytes=4*tensor['count'],model_path=spec['model_path']))
        for name in LAYERS:
            layer_rows.append(dict(policy=policy,hidden_size=h,layer=name,parameters=groups[name],FP32_bytes=4*groups[name],scope='actual exported tensors; zero means no learned layer'))
        nstates=64 if policy=='stride' else 1
        parameter_bytes=4*parameters;hc_pair=8*h;hc_total=nstates*hc_pair
        row=dict(policy=policy,hidden_size=h,feature_count=config['feature_count'],tensor_count=len(tensors),parameters=parameters,FP32_parameter_bytes=parameter_bytes,
                 h_bytes_per_state=4*h,c_bytes_per_state=4*h,hc_bytes_per_state=hc_pair,state_organization='per-PC' if policy=='stride' else 'global',configured_state_entries=nstates,
                 configured_hc_payload_bytes=hc_total,shared_weight_copies=1,weights_plus_configured_hc_payload_bytes=parameter_bytes+hc_total,
                 stride_tag_control_reservation_bytes=32*64 if policy=='stride' else 0,
                 stride_source_state_table_reservation_bytes=(32+hc_pair)*64 if policy=='stride' else '',
                 model_file_bytes=path.stat().st_size,model_file_nonparameter_bytes=path.stat().st_size-parameter_bytes,
                 checkpoint_file_bytes=(ROOT/spec['checkpoint_path']).stat().st_size if (ROOT/spec['checkpoint_path']).exists() else '',
                 model_path=spec['model_path'],checkpoint_path=spec['checkpoint_path'],tensor_verification='PASS',checkpoint_byte_verification='NOT_RUN',
                 scope='weights + configured h/c only; excludes scratch, outputs, queues, metadata, allocator overhead and hardware area')
        check=dict(**spec,status='PASS',format=config['format'],parameters=parameters,shape_contract_exact=True,FP32_values_finite=True,header_count_equals_tensor_sum=True)
        if verify_checkpoints:
            result=verify_checkpoint(ROOT/spec['checkpoint_path'],policy,tensors);check['checkpoint']=result;row['checkpoint_byte_verification']='PASS'
        checks.append(check);models.append(row)
    if not allow_missing:assert len(models)==8
    dump(OUT/'model_size.csv',models);dump(OUT/'model_size_layers.csv',layer_rows);dump(OUT/'model_size_tensors.csv',tensor_rows)
    report=dict(status='PASS' if len(models)==8 else 'INCOMPLETE',models_verified=len(models),checkpoint_bytes_verified=verify_checkpoints,
                no_simulation_or_training=True,checks=checks,
                equations=dict(affine='out × in + out bias',lstm='4H(D + H + 2): input/recurrent weights plus two biases',gru='3H(D_decoder + H + 2)',stride='11H² + 150H + 4',streamer_ampm='8H² + 287H + 13',spp='7H² + 300H + 17',FP32_parameter_bytes='4 × sum(product(tensor shape))',hc_payload='4H hidden + 4H cell = 8H bytes per state'),
                boundaries=['A model.bin file includes headers, tensor names/shapes and payload; checkpoint serialization may include additional training data. Neither file size is resident FP32 parameter payload.',
                            'Stride shares one weight copy across up to 64 PC states; only h/c and state metadata multiply by capacity. Global routes retain one chronological h/c pair.',
                            'Stride 32-byte tag/controller/lifetime reservation per entry is a source-field accounting convention, not measured map-node storage; actual ABI/padding/STL allocation remains separate.',
                            'A shorter observation history changes state contents/update count, not fixed H, parameter count or configured state reservation. Fewer active PCs may change occupied entries, not the configured capacity.',
                            'Weights plus h/c is partial storage, not complete deployment memory and not synthesized area/energy/PPA. Scratch, queues, output, controller and allocator overhead need separate implementation accounting.'])
    (OUT/'model_size_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return models


def diagrams(models):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
    plt.rcParams.update({'font.size':11,'svg.fonttype':'none'})
    grey='#777777';blue='#0072B2';orange='#D55E00';ink='#22313f';light='#f3f6f8'
    def box(ax,x,y,w,h,title,lines,edge=ink,face=light):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012,rounding_size=.02',facecolor=face,edgecolor=edge,lw=1.5))
        ax.text(x+w/2,y+h-.045,title,ha='center',va='top',fontweight='bold',color=edge,fontsize=12)
        ax.text(x+w/2,y+h/2-.025,'\n'.join(lines),ha='center',va='center',fontsize=10,color=ink,linespacing=1.45)
    def arrow(ax,start,end,label='',color=ink,rad=0):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,lw=1.5,color=color,connectionstyle='arc3,rad='+str(rad)))
        if label:ax.text((start[0]+end[0])/2,(start[1]+end[1])/2+.022,label,ha='center',va='bottom',fontsize=9,color=color)
    def save(fig,name):
        for ext in ('svg','png'):
            path=OUT/(name+'.'+ext);fig.savefig(path,dpi=180,bbox_inches='tight',facecolor='white')
            if ext=='svg':path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
        plt.close(fig)
    # Both route organization and decoder shape are explicit; no weights updated online.
    fig,ax=plt.subplots(figsize=(14,6.6));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    ax.text(.5,.98,'Frozen model: inputs → encoder/state → count and autoregressive actions',ha='center',va='top',fontsize=16,fontweight='bold',color=ink)
    box(ax,.02,.50,.22,.31,'Route inputs',['Stride: 64 PC + 64 byte bits','Streamer/AMPM: 64 byte bits','SPP: 64 byte bits + D/F kind'],grey)
    box(ax,.30,.50,.23,.31,'Encoder, hidden width H',['Stride: tanh projection → LSTM','Others: direct LSTM','H = 8 (blue) or 16 (orange)'])
    box(ax,.59,.50,.18,.31,'Count / emit heads',['Stride/SPP: hurdle gate','Streamer/AMPM: count','SPP fill callbacks: no output'])
    box(ax,.82,.50,.16,.31,'Action decoder',['GRU, own feedback','Stride: scalar delta','Others: GMM4','SPP: L2/LLC fill'])
    arrow(ax,(.24,.66),(.30,.66));arrow(ax,(.53,.66),(.59,.66));arrow(ax,(.77,.66),(.82,.66))
    box(ax,.30,.10,.23,.24,'Persistent h/c',['Stride: one state per PC','Others: one global state','Weights are shared and frozen'],blue)
    ax.add_patch(FancyArrowPatch((.415,.50),(.415,.34),arrowstyle='<->',mutation_scale=13,lw=1.5,color=blue))
    ax.text(.45,.42,'read / publish h/c',ha='left',va='center',fontsize=10,color=blue)
    box(ax,.59,.10,.39,.24,'Frozen inference',['P includes all learned tensors','projection, LSTM, heads and GRU','No optimizer or gradient runs online'])
    ax.text(.5,.015,'H is vector width, not history length. The finite engine and external request/page gates are separate from this model diagram.',ha='center',fontsize=10,color=grey)
    save(fig,'model_size_architecture')
    fig,ax=plt.subplots(figsize=(14,6.1));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    ax.text(.5,.98,'From tensor shapes to parameter bytes — and separate recurrent-state storage',ha='center',va='top',fontsize=16,fontweight='bold',color=ink)
    box(ax,.02,.56,.25,.28,'1 · Count parameters',['P = Σ product(tensor shape)','Affine: O × I + O','LSTM: 4H(D + H + 2)','GRU: 3H(Ddecoder + H + 2)'])
    box(ax,.37,.56,.24,.28,'2 · FP32 parameters',['4 bytes per parameter','Resident weights = 4P bytes','One shared weight copy','Not checkpoint-file size'])
    box(ax,.71,.56,.27,.28,'3 · Add state separately',['One h/c pair = 2 × H × 4 = 8H B','H8: 64 B; H16: 128 B','Global: one pair','Stride: capacity 64 PC pairs'])
    arrow(ax,(.27,.70),(.37,.70));arrow(ax,(.61,.70),(.71,.70))
    box(ax,.05,.10,.39,.26,'Configured h/c payload',['Global: 64 B (H8) / 128 B (H16)','Stride: 4,096 B (H8) / 8,192 B (H16)','Stride weights are not copied 64 times'],blue)
    box(ax,.55,.10,.40,.26,'Separate accounting boundaries',['Tag/control + ABI/padding + allocator','Scratch + output + queue/controller','Partial storage is not synthesized area'],orange)
    ax.text(.5,.015,'Shorter history changes h/c values and completed updates; it does not shrink fixed weights or configured state capacity.',ha='center',fontsize=10,color=grey)
    save(fig,'model_size_accounting')
    # Same data source and colors as the current H8/H16 reader figures.
    if len(models)==8:
        fig,axes=plt.subplots(1,2,figsize=(12,4.8));labels=[NAMES[p] for p in POLICIES]
        lookup={(r['policy'],r['hidden_size']):r for r in models}
        for h,offset,color in ((8,-.18,blue),(16,.18,orange)):
            values=[lookup[p,h]['FP32_parameter_bytes']/1024 for p in POLICIES]
            axes[0].bar([i+offset for i in range(4)],values,.33,label='H'+str(h),color=color)
            for i,y in enumerate(values):axes[0].text(i+offset,y,f'{y:.2f}',ha='center',va='bottom',fontsize=9)
            values=[lookup[p,h]['configured_hc_payload_bytes'] for p in POLICIES]
            axes[1].bar([i+offset for i in range(4)],values,.33,label='H'+str(h),color=color)
            for i,y in enumerate(values):axes[1].text(i+offset,y,str(y),ha='center',va='bottom',fontsize=9)
        for ax in axes:ax.set_xticks(range(4),labels);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False);ax.legend()
        axes[0].set_ylabel('FP32 parameter payload (KiB)');axes[1].set_ylabel('Configured h/c payload (bytes)');axes[1].set_yscale('log');axes[1].set_ylim(32,20000)
        fig.suptitle('Tensor-verified weights and separate configured recurrent state',fontsize=14)
        fig.text(.04,.015,'State uses a log axis: Stride capacity = 64 PC pairs; other routes = one global pair. Metadata/scratch/control excluded; no area claim.',fontsize=9)
        fig.tight_layout(rect=(0,.05,1,.94));save(fig,'model_size_storage')


def reader_section():
    """Concise section for the existing main results README; no extra MD file."""
    with (OUT/'model_size.csv').open(newline='') as stream:models=list(csv.DictReader(stream))
    with (OUT/'model_size_layers.csv').open(newline='') as stream:layers=list(csv.DictReader(stream))
    audit=json.loads((OUT/'model_size_audit.json').read_text())
    assert len(models)==8 and audit['status']=='PASS' and audit['checkpoint_bytes_verified']
    labels=[NAMES[row['policy']]+' H'+row['hidden_size'] for row in models]
    result=['## Model size and inference state','',
        '- **Hidden width H:** the number of elements in each hidden/cell vector, not the number of PCs or observations.',
        '- **Parameters → bytes:** sum every tensor’s element count to obtain **P**; FP32 parameter payload is **4P bytes**. One h/c pair stores **2 × H × 4 = 8H bytes**. [Tensor-by-tensor verification](model_size_tensors.csv) · [checkpoint/export audit](model_size_audit.json).','',
        '| Model | Parameters P | FP32 parameter bytes | h/c per state (B) | Configured states | Configured h/c bytes |',
        '| --- | --- | --- | --- | --- | --- |']
    for label,row in zip(labels,models):
        result.append('| '+label+' | '+' | '.join(format(int(row[key]),',') for key in ('parameters','FP32_parameter_bytes','hc_bytes_per_state','configured_state_entries','configured_hc_payload_bytes'))+' |')
    result+=['',
        '- **Layer calculation:** affine = O×I + O; LSTM = 4H(D+H+2); GRU = 3H(Ddecoder+H+2). Both cells include two bias vectors. Encoder D = H for projected Stride, 64 for Streamer/AMPM and 65 for SPP; decoder input = 1 except SPP’s 3. Counts below come from actual checkpoint/export tensors; 0 means that learned layer is absent. [Per-layer CSV](model_size_layers.csv).','',
        '| Layer parameters | '+' | '.join(labels)+' |',
        '| --- | '+' | '.join('---' for _ in models)+' |']
    index={(row['policy'],int(row['hidden_size']),row['layer']):int(row['parameters']) for row in layers}
    layer_names=dict(input_projection='Input projection',encoder_lstm='Encoder LSTM',emit_gate='Emit gate',count_head='Count head',decoder_initial='Decoder initialization',decoder_gru='Decoder GRU',delta_head='Delta head',fill_head='Fill head')
    for name in LAYERS:
        result.append('| '+layer_names[name]+' | '+' | '.join(format(index[row['policy'],int(row['hidden_size']),name],',') for row in models)+' |')
    result.append('| **Total P** | '+' | '.join('**'+format(int(row['parameters']),',')+'**' for row in models)+' |')
    result+=['',
        '- **Why SPP H16 is smaller:** it omits the 272-parameter learned decoder initialization; extra input, gate, fill and GRU parameters add 228, leaving **44 fewer parameters** than Streamer/AMPM H16.',
        '- **State organization:** Stride shares one weight copy across its 64-PC capacity; only h/c and metadata scale with PC entries. Streamer/AMPM/SPP each keep one chronological global h/c pair. Stride’s source-field tag/control reservation adds 32 B/entry (2,048 B at capacity); actual ABI/padding/map allocation is separate.',
        '- **Measured host example:** Streamer H16 object + persistent requested heap is **28,596 B**, already including its 26,612 B weights and 128 B h/c. Extra inference heap peak is **948 B** on the passed 8,236-callback fixture; this is not a worst-case bound. [Actual host fields and source/fixture bindings](model_size_host_streamer_h16.json).',
        '- **Observation history:** shorter history changes h/c values and update counts, not fixed P, H or configured capacity. Fewer active PCs may reduce occupied states, not reserved capacity. Action-dependent scratch/output is separate.',
        '- **Accounting boundary:** weights + h/c is partial storage. Checkpoint/container file lengths include serialization and headers. Scratch, outputs, queues, control, allocator overhead and synthesized hardware area require separate accounting. [Full model-size CSV](model_size.csv) · [global runtime cost details](../../../formal_NN_training/experiments/602_four_prefetcher_formal/report/COST.md) · [Stride field/ABI boundaries](../../../docs/602_gate_mechanism/STORAGE.md).','',
        '![Frozen model architecture and recurrent-state organization](model_size_architecture.svg)','',
        '![Parameter count to FP32 size, with separate state and implementation boundaries](model_size_accounting.svg)','',
        '- [H8/H16 parameter and state storage plot](model_size_storage.svg) · [PNG architecture](model_size_architecture.png) · [PNG accounting](model_size_accounting.png).','']
    return result


def host_probe_streamer(simulator):
    """Actual requested heap on the complete passed reset-separated fixture."""
    import subprocess
    simulator=Path(simulator).resolve()
    if ROOT not in simulator.parents:raise ValueError('actual simulator must stay inside cache')
    raw=ROOT/'formal_study/runs/602_width_templates_20261017/stride_streamer_h16'
    parity_dir=raw/'correctness/streamer/parity'
    parity=json.loads((parity_dir/'parity_report.json').read_text())
    assert parity['status']=='PASS' and parity['count_and_all_action_fields_required_exact']
    cost=raw/'cost/streamer_h16_host';cost.mkdir(parents=True,exist_ok=True)
    runtime=cost/'runtime';runtime.mkdir(exist_ok=True)
    source=cost/'source';source.mkdir(exist_ok=True)
    bindings=[]
    def bind(path,destination):
        data=Path(path).read_bytes();destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():assert destination.read_bytes()==data,'preserve conflicting cost source: '+str(destination)
        else:destination.write_bytes(data)
        bindings.append(dict(source=str(path),preserved_copy=str(destination),bytes=len(data),byte_equal=True))
    core=simulator/'prefetcher/live602_runtime.cc';header=simulator/'inc/live602_runtime.h'
    canonical=ROOT/'formal_NN_training/common/live602_runtime'
    assert core.read_bytes()==(canonical/'live602_runtime.cc').read_bytes()
    assert header.read_bytes()==(canonical/'live602_runtime.h').read_bytes()
    passed_source=raw/'correctness/streamer/source'
    assert core.read_bytes()==(passed_source/'live602_runtime.cc').read_bytes()
    assert header.read_bytes()==(passed_source/'live602_runtime.h').read_bytes()
    probe=ROOT/'formal_NN_training/experiments/602_four_prefetcher_formal/validation/storage_probe.cc'
    bind(core,source/'live602_runtime.cc');bind(probe,source/'storage_probe.cc')
    for name in ('live602_runtime.h','streamer.h','ampm.h','bitmap.h','prefetcher.h','spp_dev2.h','spp_dev2_helper.h'):
        bind(simulator/'inc'/name,source/'native_headers'/name)
    model=ROOT/'formal_NN_training/experiments/602_lstm_streamer_live_inference/models/h16/model.bin'
    bind(model,source/'model.bin')
    for name in ('commands.txt','cpp.jsonl','parity_report.json'):bind(parity_dir/name,source/name)
    command=['g++','-std=c++11','-O3','-ffp-contract=off','-I'+str(simulator/'inc'),str(core),str(probe),'-o',str(runtime/'storage_probe')]
    process=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
    (cost/'compile.log').write_text(process.stdout)
    if process.returncode:raise RuntimeError('host-only probe compile failed; see '+str(cost/'compile.log'))
    compiler=subprocess.check_output(['g++','--version'],universal_newlines=True).splitlines()[0]
    commands=[];expected=[]
    for line in (parity_dir/'commands.txt').read_text().splitlines():
        if line=='R':commands.append([])
        else:
            fields=line.split();assert len(fields)==2 and fields[0]=='D' and int(fields[1])&63==0
            commands[-1].append(line)
    for line in (parity_dir/'cpp.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row.get('reset'):expected.append([])
        else:expected[-1].append(row)
    assert len(commands)==len(expected)==parity['reset_records']
    reports=[]
    for i,(segment,cpp) in enumerate(zip(commands,expected)):
        assert len(segment)==len(cpp)
        text='\n'.join(segment)+'\n';(cost/('segment_'+str(i)+'_commands.txt')).write_text(text)
        process=subprocess.run([str(runtime/'storage_probe'),str(model)],input=text,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
        (cost/('segment_'+str(i)+'_probe.log')).write_text(process.stdout+process.stderr)
        if process.returncode:raise RuntimeError('host-only probe failed on passed segment '+str(i))
        report=json.loads(process.stdout)
        wanted=dict(callbacks=len(cpp),actions=sum(len(x['actions']) for x in cpp),macs=sum(x['macs'] for x in cpp),nonlinears=sum(x['nonlinears'] for x in cpp),demand_callbacks=len(cpp),fill_callbacks=0)
        for key,value in wanted.items():assert report['fixture'][key]==value,(i,key,report['fixture'][key],value)
        assert report['model']['parameters']==6653 and report['model']['fp32_weight_bytes']==26612 and report['model']['global_hc_bytes']==128
        assert report['host_allocations']['residual_after_runtime_destruction']==0
        report['segment']=i;report['passed_fixture_match']=wanted;reports.append(report)
    assert sum(r['fixture']['callbacks'] for r in reports)==parity['actual_global_callbacks']
    assert sum(r['fixture']['actions'] for r in reports)==parity['compared_all_actions']
    constant=('runtime_object_bytes','persistent_requested_bytes','object_plus_persistent_requested_bytes','constructor_peak_requested_bytes','constructor_allocation_calls')
    for r in reports:
        assert r['sizeof']==reports[0]['sizeof']
        assert all(r['host_allocations'][key]==reports[0]['host_allocations'][key] for key in constant)
    host={key:reports[0]['host_allocations'][key] for key in constant}
    for key in ('max_infer_incremental_peak_requested_bytes','max_returned_result_requested_bytes','max_actions_vector_capacity'):
        host[key]=max(r['host_allocations'][key] for r in reports)
    host['residual_after_runtime_destruction']=0
    fixture=dict(callbacks=sum(r['fixture']['callbacks'] for r in reports),actions=sum(r['fixture']['actions'] for r in reports),reset_segments=len(reports),segment_lengths=[len(s) for s in commands],max_count=max(r['fixture']['max_count'] for r in reports),macs=sum(r['fixture']['macs'] for r in reports),nonlinears=sum(r['fixture']['nonlinears'] for r in reports),bound_to_passed_parity=True)
    result=dict(schema='model_size_streamer_h16_actual_host_v1',status='PASS',policy='streamer',hidden_size=16,simulator=str(simulator),compiler=compiler,compile_command=command,model=reports[0]['model'],sizeof=reports[0]['sizeof'],host_allocations=host,fixture=fixture,source_parity=str(parity_dir/'parity_report.json'),bindings=bindings,segments=reports,
                scope='Actual scalar-runtime object + ordinary C++ new/new[] requested heap; includes weights and global h/c, so do not add them again. Excludes stack, C malloc internals, allocator metadata/alignment, outer Four602 controller, observer/log/fixture storage, RSS and hardware area/PPA.',
                peak_scope='Observed across the complete passed 8236-callback parity fixture with four independent reset segments. This is not a worst-case K/heap bound or full-ROI peak.',no_simulator_executed=True)
    (cost/'host_memory.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'model_size_host_streamer_h16.json').write_text(json.dumps(result,indent=2)+'\n')
    row=dict(policy='streamer',hidden_size=16,FP32_parameter_bytes=26612,global_hc_payload_bytes=128,runtime_object_bytes=host['runtime_object_bytes'],persistent_requested_heap_bytes=host['persistent_requested_bytes'],object_plus_persistent_requested_bytes=host['object_plus_persistent_requested_bytes'],observed_infer_incremental_heap_peak_bytes=host['max_infer_incremental_peak_requested_bytes'],observed_max_action_capacity=host['max_actions_vector_capacity'],observed_max_K=fixture['max_count'],fixture_callbacks=fixture['callbacks'],fixture_reset_segments=fixture['reset_segments'],source_parity=str(parity_dir/'parity_report.json'),scope=result['scope'],peak_scope=result['peak_scope'])
    dump(OUT/'model_size_host_streamer_h16.csv',[row])
    print(json.dumps(dict(status='PASS',host_allocations=host,fixture=fixture,no_simulator_executed=True)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host-probe-streamer',type=Path,metavar='ACTUAL_SIMULATOR',help='run the existing requested-allocation probe on the complete passed Streamer H16 fixture; no simulation')
    parser.add_argument('--verify-checkpoints',action='store_true')
    parser.add_argument('--allow-missing',action='store_true',help='inventory available actual exports; never infer missing-model tensor counts')
    parser.add_argument('--no-plots',action='store_true')
    parser.add_argument('--plots-only',action='store_true',help='render SVG/PNG from the existing checkpoint-verified inventory without rewriting its certificate')
    args=parser.parse_args()
    if args.host_probe_streamer:
        if args.verify_checkpoints or args.allow_missing or args.plots_only:parser.error('host probe is independent of inventory flags')
        host_probe_streamer(args.host_probe_streamer);return
    if args.plots_only:
        if args.verify_checkpoints or args.allow_missing or args.no_plots:parser.error('--plots-only is independent of inventory flags')
        with (OUT/'model_size.csv').open(newline='') as stream:models=list(csv.DictReader(stream))
        for row in models:
            for key in ('hidden_size','parameters','FP32_parameter_bytes','configured_hc_payload_bytes'):row[key]=int(row[key])
            assert row['tensor_verification']=='PASS'
        assert len(models)==8
    else:models=inventory(args.verify_checkpoints,args.allow_missing)
    if not args.no_plots:diagrams(models)
    print(json.dumps(dict(status='PASS' if len(models)==8 else 'INCOMPLETE',models=len(models),checkpoint_bytes_verified=all(r['checkpoint_byte_verification']=='PASS' for r in models),verification_source='retained_inventory' if args.plots_only else 'this_run' if args.verify_checkpoints else 'exports_only',no_training_or_simulation=True)))

if __name__=='__main__':main()
