#!/usr/bin/env python3
"""Independent full-source/count/table audit; optional exact scorer recomputation."""
import argparse
import csv
import filecmp
import gzip
import json
import os
from pathlib import Path
import subprocess
import tempfile

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--raw',type=Path,required=True)
p.add_argument('--reference-root',type=Path,required=True)
p.add_argument('--recompute',action='store_true')
a=p.parse_args();raw=a.raw;old=a.reference_root
b=json.loads((raw/'build.json').read_text())
assert filecmp.cmp(b['scorer'],b['scorer_snapshot'],shallow=False)
inputs=json.loads((raw/'pc_gate/input_records.json').read_text())
assert [r['name'] for r in inputs]==['train','validation']
source_rows={}
for item in inputs:
    name=item['name'];assert item['source_records']==({'train':[0,1000000],'validation':[5000000,10000000]}[name])
    assert item['initial_state']=='independent_zero' and item['state_capacity']==64
    source=old/'models'/('train_inputs.csv.gz' if name=='train' else 'validation_inputs.csv.gz')
    assert filecmp.cmp(source,item['snapshot'],shallow=False)
    count=0
    with gzip.open(source,'rt',newline='') as f,Path(item['canonical']).open(newline='') as canonical:
        source_iter=csv.DictReader(f);canonical_iter=csv.DictReader(canonical)
        for r in source_iter:
            actual=next(canonical_iter)
            def num(s):return int(s,16) if s.lower().startswith('0x') else int(s,10)
            assert (num(r['pc']),num(r['line']))==(int(actual['pc']),int(actual['line']))
            count+=1
        assert next(canonical_iter,None) is None
    assert count==item['events'] and count>4096
    source_rows[name]=count
manifest=json.loads((old/'training_split.json').read_text())
assert manifest['decision_rows']==source_rows['train']
val=old/'validation/validation_602_r0_none_s7_balanced_m0_b0_sil0_qfifo_hist4000000'
assert json.loads((val/'run.json').read_text())['status']=='complete'
assert json.loads((val/'summary.json').read_text())['eligible_l2_callbacks']==source_rows['validation']
results=[];models=[]
for seed in (7,17,27):
    folder=raw/'pc_gate'/f'seed{seed}';fit=json.loads((folder/'fit.json').read_text())
    contract=fit['spec']['contract'];assert fit['status']=='complete'
    assert fit['spec']['seed']==seed and contract['build_id']==b['build_id']
    assert contract['fit_splits']==['train','validation'] and contract['test_used'] is False
    assert contract['capacity']==64 and contract['threshold']==.5
    original=old/'models'/f'seed{seed}'/'balanced/export/model.bin'
    metadata=json.loads(original.with_name('model_metadata.json').read_text())
    assert metadata['seed']==seed and metadata['emit_loss']=='balanced'
    assert metadata==fit['spec']['model_metadata']
    assert filecmp.cmp(original,original.with_name('immutable_model.bin'),shallow=False)
    assert filecmp.cmp(original,folder/'model_snapshot.bin',shallow=False)
    models.append(original)
    by_split={};recomputed=[]
    for item in inputs:
        name=item['name'];rr={}
        with (folder/(name+'_counts.csv')).open() as f:
            for r in csv.DictReader(f):
                pc,n,pos,k0=map(int,(r['pc'],r['count'],r['positives'],r['emit_k_zero']))
                assert pc not in rr and n>0 and 0<=k0<=pos<=n
                rr[pc]=(n,pos,k0)
        assert sum(v[0] for v in rr.values())==source_rows[name]
        score=json.loads((folder/(name+'_score.json')).read_text())
        assert score['events']==source_rows[name] and score['full_stream'] is True
        assert score['split_initial_state']=='zero' and score['state_capacity']==64
        assert score['positive_emits']==sum(v[1] for v in rr.values())
        if a.recompute:
            with tempfile.TemporaryDirectory(prefix='gate_fit_audit_') as td:
                counts=Path(td)/'counts.csv';summary=Path(td)/'summary.json'
                subprocess.run([b['scorer'],str(original),item['canonical'],str(counts),str(summary)],
                    check=True,stdout=subprocess.DEVNULL,env=dict(os.environ,OMP_NUM_THREADS='1'))
                assert filecmp.cmp(counts,folder/(name+'_counts.csv'),shallow=False)
                assert json.loads(summary.read_text())==score
                recomputed.append(name)
        by_split[name]=rr
    combined={}
    for rr in by_split.values():
        for pc,v in rr.items():combined[pc]=tuple(x+y for x,y in zip(combined.get(pc,(0,0,0)),v))
    expected=sorted(combined,key=lambda pc:(-combined[pc][0],pc))[:64]
    lines=(folder/'pc_gate.tsv').read_text(encoding='ascii').splitlines()
    assert lines[0]=='pc gate count positives' and len(lines)==len(expected)+1
    table={}
    for line,expected_pc in zip(lines[1:],expected):
        fields=line.split();assert len(fields)==4 and all(x.isdecimal() for x in fields)
        pc,gate,n,pos=map(int,fields)
        assert pc==expected_pc and pc not in table
        assert (n,pos)==combined[pc][:2] and gate==int(2*pos>=n)
        table[pc]=gate
    agreement={}
    for name,rr in dict(by_split,combined=combined).items():
        n=sum(v[0] for v in rr.values());correct=sum(v[1] if table.get(pc,0) else v[0]-v[1] for pc,v in rr.items())
        misses=sum(v[0] for pc,v in rr.items() if pc not in table)
        reported=fit['agreement'][name]
        assert reported['events']==n and reported['agreement_events']==correct
        assert abs(reported['agreement']-correct/n)<1e-15 and reported['table_miss_events']==misses
        agreement[name]=dict(events=n,correct=correct,agreement=correct/n,table_miss_events=misses,
                             unique_pcs=len(rr))
    results.append(dict(seed=seed,table_entries=len(table),positive_table_entries=sum(table.values()),
                        agreement=agreement,recomputed_full_splits=recomputed))
assert all(not filecmp.cmp(models[i],models[j],shallow=False) for i in range(3) for j in range(i+1,3))
print(json.dumps(dict(status='PASS',protocol='gate_mechanism_20261016_v1',build_id=b['build_id'],
    independent_audit=True,test_inputs_used=False,source_records=source_rows,
    combined_events=sum(source_rows.values()),three_distinct_original_models=True,seeds=results),indent=2))
