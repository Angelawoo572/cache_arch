#!/usr/bin/env python3
"""Independent all-completed-row histogram, +2 payload and identity checks."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--raw',type=Path,required=True)
p.add_argument('--old-raw',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
matrix=json.loads((a.raw/'formal_matrix.json').read_text())
complete=[];pending=[];original=[];fixed=[]
build=json.loads((a.raw/'build.json').read_text())['build_id']

def histogram(x,name,phase=None):
    result=Counter()
    for key,value in x['mechanism_distributions_exact'].items():
        match=re.fullmatch(r'phase_([0-4])_(.+)',key)
        assert match,key
        if match[2]==name and (phase is None or int(match[1])==phase):
            result.update({int(k):v for k,v in value.items()})
    return result

def same(a,b):return {k:v for k,v in a.items() if v}=={k:v for k,v in b.items() if v}

for planned in matrix:
    if planned['execution']!='fresh':continue
    folder=a.raw/'test'/planned['run_id']
    if not (folder/'run.json').is_file():pending.append(planned['run_id']);continue
    record=json.loads((folder/'run.json').read_text())
    if record['status']!='complete':pending.append(planned['run_id']);continue
    s=record['spec'];x=json.loads((folder/'summary.json').read_text())
    assert all(s[k]==v for k,v in planned.items()) and s['build_id']==build
    assert abs(x['instructions']-25000000)<=4 and not x['numerical_failures']
    assert x['engine_method']==s['method'] and x['engine_payload_mode']==s['payload']
    assert record['environment']['DYNAMIC602_METHOD']==s['method']
    assert record['environment']['DYNAMIC602_FILTER_CURRENT']==str(s['filter_current'])
    assert s['warmup']==s['history']==5000000
    assert s['skip']==(20000000 if s['roi']==0 else 70000000)
    if s['method']=='frozen':
        metadata=json.loads((a.old_raw/'models'/f"seed{s['seed']}"/'balanced/export/model_metadata.json').read_text())
        assert metadata==s['model_metadata'] and metadata['seed']==s['seed'] and metadata['emit_loss']=='balanced'
        assert Path(s['model']).parent.parent.parent.name==f"seed{s['seed']}"
        raw_k=histogram(x,'raw_K');raw_emit=histogram(x,'raw_emit')
        assert set(raw_emit).issubset({0,1}) and sum(raw_emit.values())==x['nn_completed']==sum(raw_k.values())
        assert sum(min(k,32)*n for k,n in raw_k.items())==x['mechanism_raw_generated_addresses']
        assert sum(sum(values.values()) for key,values in x['mechanism_distributions_exact'].items()
                   if '_raw_delta_slot_' in key)==x['mechanism_raw_generated_addresses']
        if not s['macs']:assert x['nn_completed']==x['eligible_l2_callbacks'] and not x['inference_dropped']
        for phase in range(5):
            rk=histogram(x,'raw_K',phase);re_=histogram(x,'raw_emit',phase);pk=histogram(x,'payload_K',phase)
            assert sum(rk.values())==sum(re_.values())==sum(pk.values())
            if s['payload']=='fixed_plus2':assert same(pk,{0:re_[0],1:re_[1]})
            else:
                decoded=Counter()
                for k,n in rk.items():decoded[min(k,32)]+=n
                assert same(pk,decoded)
        if not s['macs'] and not s['budget'] and not s['silent'] and not s['filter_current'] and s['arm']=='nn':
            original.append(dict(run_id=s['run_id'],roi=s['roi'],seed=s['seed'],decisions=x['nn_completed'],
                raw_K=dict(raw_k),raw_emit=dict(raw_emit),raw_slot0_delta=dict(histogram(x,'raw_delta_slot_0')),
                raw_slot1_delta=dict(histogram(x,'raw_delta_slot_1')),
                raw_generated=x['mechanism_raw_generated_addresses'],raw_current=x['mechanism_raw_current_line_addresses'],
                raw_list_duplicates=x['mechanism_raw_list_duplicate_addresses']))
    elif s['method']=='pc_gate_plus2':
        table=a.raw/'pc_gate'/f"seed{s['seed']}"/'pc_gate.tsv'
        assert table.read_text()==s['gate_table_contents']
        fit=json.loads(table.with_name('fit.json').read_text())
        assert fit==s['gate_fit'] and fit['spec']['seed']==s['seed']
        assert x['pc_gate_entries']==fit['table_entries']==55 and x['pc_gate_capacity']==64
        assert x['pc_gate_seen_events']+x['pc_gate_unseen_events']==x['eligible_l2_callbacks']
        assert x['generated_addresses']==x['pc_gate_emit_events']
        assert x['nn_completed']==x['inference_macs']==x['mechanism_raw_generated_addresses']==0
    if s['payload']=='fixed_plus2':
        pk=histogram(x,'payload_K');pe=histogram(x,'payload_emit');delta=Counter()
        for key,value in x['mechanism_distributions_exact'].items():
            if '_payload_delta_slot_' in key:
                assert key.endswith('_slot_0')
                delta.update({int(k):v for k,v in value.items()})
        assert set(pk).issubset({0,1}) and same(pk,pe)
        assert set(delta)=={2} and delta[2]==pk[1]==x['generated_addresses']==x['mechanism_payload_generated_addresses']
        assert x['mechanism_payload_current_line_addresses']==x['mechanism_payload_list_duplicate_addresses']==0
        if s['method']=='frozen':assert pk[1]==histogram(x,'raw_emit')[1]
        elif s['method']=='fixed_plus2':assert pk[1]==x['eligible_l2_callbacks'] and not pk[0] and x['nn_completed']==0
        else:assert sum(pk.values())==x['eligible_l2_callbacks'] and pk[1]==x['pc_gate_emit_events']
        fixed.append(dict(run_id=s['run_id'],roi=s['roi'],arm=s['arm'],seed=s['seed'],budget=s['budget'],
                          payload_K=dict(pk),payload_delta_slot0=dict(delta),nn_completed=x['nn_completed'],
                          pc_gate_unseen_events=x.get('pc_gate_unseen_events',0)))
    complete.append(s['run_id'])
expected={(roi,seed) for roi in (0,1) for seed in (7,17,27)}
assert {(r['roi'],r['seed']) for r in original}==expected and len(original)==6
result=dict(status='PASS_COMPLETED_SCOPE' if pending else 'PASS',protocol='gate_mechanism_20261016_v1',
    build_id=build,planned_fresh=64,completed_fresh=len(complete),pending_fresh=pending,
    scope='All currently completed fresh rows; all six uncapped original MAC0 seed/ROI controls are complete. Final matrix outcome audit remains separate.',
    model_and_gate_identities_verified=True,full_phase_histogram_conservation=True,
    fixed_plus2_completed_count=len(fixed),fixed_plus2_K_0_or_1=True,fixed_plus2_all_deltas_exactly_2=True,
    original_mac0_six_full_profiles=original,fixed_payload_controls=fixed)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in {'original_mac0_six_full_profiles','fixed_payload_controls'}},indent=2))
for r in original:print(json.dumps(r))
