#!/usr/bin/env python3
"""Collect actual scalar-runtime host allocations; never starts ChampSim."""
import argparse,csv,gzip,itertools,json,shutil,subprocess
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--probe',type=Path,required=True)
p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--cost-directory',type=Path);p.add_argument('--models-root',type=Path)
p.add_argument('--build-metadata',type=Path);p.add_argument('--optimization',default='-O2');a=p.parse_args()
cost=a.cost_directory or a.study/'cost';cost.mkdir(parents=True,exist_ok=True)
models_root=a.models_root or a.study/'raw/models'
headers=cost/'native_headers';headers.mkdir(exist_ok=True)
for name in ('streamer.h','ampm.h','bitmap.h','prefetcher.h','spp_dev2.h','spp_dev2_helper.h'):
    shutil.copy2(a.build/'inc'/name,headers/name)
legacy=a.study.parents[0]/'formal_NN_training'/'experiments'
rows={
 'ampm':legacy/'602_offline_lstm_ampm/runs/602_offline_lstm_ampm_variable_delta_free_running_v7_seed7/colab_input/602.gcc_s-734B.guard_stream.csv.gz',
 'streamer':legacy/'602_offline_lstm_streamer/runs/602_offline_lstm_streamer_variable_delta_free_running_v7_seed7/colab_input/602.gcc_s-734B.train_stream.csv.gz',
 'spp':legacy/'602_offline_lstm_spp/runs/602_offline_lstm_spp_empirical_prior_hurdle_free_running_v2_seed7/colab_input/602.gcc_s-734B.spp.train_stream.csv.gz'}
models={};common_sizes=None
for policy,stream in rows.items():
    commands=[]
    with gzip.open(stream,'rt',newline='') as handle:
        for row in itertools.islice(csv.DictReader(handle),4096):
            if policy=='spp':
                kind='D' if row['event_kind']=='DEMAND' else 'F'
                value=row['event_address'];address=int(value,16) if value.lower().startswith('0x') else int(value,10)
            else:kind='D';address=int(row['line'],10)<<6
            commands.append(f'{kind} {address}')
    if len(commands)!=4096:raise AssertionError('fixture must contain 4096 real contiguous rows')
    fixture=cost/(policy+'_commands.txt');fixture.write_text('\n'.join(commands)+'\n')
    model=models_root/policy/'model.bin'
    completed=subprocess.run([str(a.probe),str(model)],input=fixture.read_text(),text=True,capture_output=True,check=True)
    data=json.loads(completed.stdout);data['model_path']=str(model);data['fixture']['source']=str(stream);data['fixture']['rows']=[0,4096]
    data['model']['weight_plus_global_hc_payload_bytes']=data['model']['fp32_weight_bytes']+data['model']['global_hc_bytes']
    data['model']['payload_qualifier']='FP32 weights plus one h/c pair only; excludes decoder scratch/output/control, not a complete deployment total'
    (cost/(policy+'_host_memory.json')).write_text(json.dumps(data,indent=2)+'\n')
    if common_sizes is not None and common_sizes!=data['sizeof']:raise AssertionError('sizeof disagrees between invocations')
    common_sizes=data['sizeof'];models[policy]=data

sizes=common_sizes
native=dict(
    streamer=dict(capacity=64,declared_payload_bits_per_tracker=64+32+32+8,
        declared_payload_bytes_at_capacity=64*17,cpp_tracker_bytes=sizes['Stream_Tracker'],
        known_object_entries_pointer_bytes_at_capacity_lower_bound=sizes['Streamer']+64*(sizes['Stream_Tracker']+8),
        field_inventory=dict(page='uint64_t',last_offset='uint32_t',last_dir='int32_t',conf='uint8_t'),
        qualifier='source-width fields; deque blocks/map allocation and allocator overhead not fully measured; class sizeof includes diagnostic counters/base object'),
    ampm=dict(capacity=64,bitmap_bits=64,declared_payload_bytes_per_page=16,
        declared_payload_bytes_at_capacity=1024,cpp_page_entry_bytes=sizes['AMPM_PB_Entry'],
        known_object_entries_pointer_bytes_at_capacity_lower_bound=sizes['AMPM']+64*(sizes['AMPM_PB_Entry']+8),
        prefetch_buffer_enabled=False,prefetch_buffer_configured_capacity=256,
        field_inventory=dict(page_id='uint64_t',bitmap='std::bitset<64>'),
        qualifier='disabled optional request buffer is not a 256-entry architectural allocation; empty std::deque may still allocate in host implementation; constructor allocations not measured'),
    spp=dict(signature_entries=256,pattern_sets=512,pattern_ways=4,filter_entries=1024,ghr_entries=8,
        cpp_inline_helper_bytes=sum(sizes[k] for k in ('SIGNATURE_TABLE','PATTERN_TABLE','PREFETCH_FILTER','GLOBAL_REGISTER')),
        cpp_inline_components={k:sizes[k] for k in ('SIGNATURE_TABLE','PATTERN_TABLE','PREFETCH_FILTER','GLOBAL_REGISTER')},
        declared_source_width_payload_bytes=dict(signature_table=256*(1+4*4),pattern_table=512*(4*4+4*4+4),prefetch_filter=1024*(8+1+1),global_register=3*8+8*(1+4+4+4+4)),
        qualifier='helper arrays inline in SPP_dev2; native histograms are diagnostics and dynamically unbounded host allocations, excluded from deployment claims; sizeof SPP_dev2 retains their empty descriptors'))

formulas=dict(
    encoder_macs='4*H*(F+H) for every callback',encoder_nonlinears='5*H for every callback',
    poisson_demand_macs='H count + (H*H initial if K>0) + K*(3*M*H + 3*H*(1+H))',
    poisson_demand_nonlinears='1 count softplus + (H initial if K>0) + K*(M scale-softplus + 1 delta-expm1 + 3*H GRU)',
    spp_demand_macs='2*H emit; if emit positive add H count; then K*(3*M*H + 2*H fill + 3*H*(3+H)); no initial affine',
    spp_demand_nonlinears='if emit positive add 1 count-exp; then K*(M scale-softplus + 1 delta-expm1 + 3*H GRU)',
    spp_fill='encoder only; no count/emit/delta/fill head or decoder',
    mac_definition='dense FP32 multiply-accumulate pairs; bias additions, argmax, bit extraction, queue/control work excluded',
    nonlinear_definition='number of primitive sigmoid/tanh/softplus/exp/expm1 evaluations, not calibrated cycles',
    measured_runtime='synchronous MAC0: no modeled service delay, resource queue, memory bandwidth, weight transfer, nonlinear unit throughput or hardware energy',
    weight_traffic='weights resident in host vectors; no simulated weight DRAM traffic; 4*MAC is not measured weight traffic and omits biases/reuse/cache behavior',
    raw_payload='full K/delta/fill decoder runs before external page/current/request gates or fixed-payload replacement; these gates do not reduce counted inference operations')

result=dict(schema='four602_actual_source_memory_cost_v1',status='verified',
    build_headers=str(a.build/'inc'),compiler='server c++ -std=c++11 '+a.optimization+' -ffp-contract=off; 64-bit libstdc++ ABI',
    byte_interpretation='sizeof includes descriptors/padding; allocation counters are requested C++ heap payload bytes, excluding malloc metadata, probe tracking headers and automatic stack frames; not RSS, synthesized hardware area, or complete deployment SRAM',
    nn_models=models,native=native,cost_model=formulas,
    exclusions=['outer Four602 adapter object/control/template allocation (scalar-runtime scope only)','measurement observers and counters','input fixture/raw logs','automatic stack frames/scalar register spills','C-library malloc internal storage','allocator metadata/alignment gaps/tracking headers','native histogram dynamic memory'],
    scope='persistent scalar runtime tracking captures ordinary new/new[] map nodes/string/vector allocations; direct malloc internals and baseline constructor heaps are not claimed complete. Peak scratch is observed for the listed 4096-callback correctness fixture, not an all-input upper bound. Global NN state payload is exactly 64B for H8; host allocation overhead is reported separately.')
if a.build_metadata:result['formal_build']=json.loads(a.build_metadata.read_text())
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
(cost/'cost_fields.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({policy:dict(weights=v['model']['fp32_weight_bytes'],hc=v['model']['global_hc_bytes'],host=v['host_allocations'],fixture=v['fixture']) for policy,v in models.items()},indent=2))
