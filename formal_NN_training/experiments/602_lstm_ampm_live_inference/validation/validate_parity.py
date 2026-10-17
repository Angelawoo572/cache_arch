#!/usr/bin/env python3
"""Full supplied causal sequences, every h/c and action; no ChampSim pilot."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys
EXP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXP/'python'))
from reference import AMPMReference


def integer(value):
    value=str(value);return int(value,16) if value.lower().startswith('0x') else int(value,10)

def addresses(paths):
    for path in paths:
        opener=gzip.open if str(path).endswith('.gz') else open
        with opener(path,'rt',newline='') as handle:
            for index,row in enumerate(csv.DictReader(handle)):
                if 'demand_idx' in row and integer(row['demand_idx'])!=index:raise ValueError('stream row ordering failure')
                address=(integer(row['line'])<<6) if 'line' in row else integer(row['address'])
                yield address


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--stream',type=Path,action='append',required=True)
    p.add_argument('--tmp',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--atol',type=float,default=2e-5);args=p.parse_args()
    args.tmp.mkdir(parents=True,exist_ok=True)
    inputs=args.tmp/'ampm_parity_inputs.txt';actual=args.tmp/'ampm_parity_cpp.jsonl'
    with inputs.open('w') as output:
        for address in addresses(args.stream):output.write('D '+str(address)+'\n')
    with inputs.open() as inp,actual.open('w') as out:
        subprocess.run([str(args.probe),str(args.model)],stdin=inp,stdout=out,check=True)
    ref=AMPMReference(args.checkpoint);count=0;actions=0;max_hidden=0.;max_cell=0.
    with actual.open() as handle:
        for count,(address,line) in enumerate(zip(addresses(args.stream),handle),1):
            cpp=json.loads(line);expected=ref.Infer(address)
            for name in ('hidden','cell'):
                error=max(abs(a-b) for a,b in zip(cpp[name],expected[name]))
                if name=='hidden':max_hidden=max(max_hidden,error)
                else:max_cell=max(max_cell,error)
                if len(cpp[name])!=8 or error>args.atol:raise AssertionError((count,name,error,cpp[name],expected[name]))
            if cpp['count']!=expected['count'] or cpp['emit']!=expected['emit']:raise AssertionError((count,'count',cpp['count'],expected['count']))
            left=[(a['address'],a['delta'],a['fill'],a['fill_index']) for a in cpp['actions']]
            right=[(a['address'],a['delta'],a['fill'],a['fill_index']) for a in expected['actions']]
            if left!=right:raise AssertionError((count,'actions',left,right))
            if cpp['truncated']:raise AssertionError('unexpected action truncation')
            actions+=len(left)
            if count%100000==0:print('matched',count,'callbacks',flush=True)
        if handle.readline():raise AssertionError('extra C++ callbacks')
    expected_count=sum(1 for _ in addresses(args.stream))
    if count!=expected_count:raise AssertionError(('missing C++ callbacks',count,expected_count))
    # Reset restarts the actual global state, including zero-count events.
    first=next(addresses(args.stream));reset=subprocess.run([str(args.probe),str(args.model)],input=f'D {first}\nR\nD {first}\n',text=True,capture_output=True,check=True)
    a,_,b=map(json.loads,reset.stdout.splitlines())
    if a!=b:raise AssertionError('Reset did not restore zero h/c')
    result=dict(status='verified',model_family='real_ampm_v7_global_lstm_poisson_mixture',seed=7,
        checkpoint=str(args.checkpoint),model=str(args.model),streams=[str(path) for path in args.stream],
        callbacks=count,actions=actions,all_actions_exact=True,all_hidden_cell_compared=True,
        max_hidden_abs_error=max_hidden,max_cell_abs_error=max_cell,atol=args.atol,reset_verified=True,
        no_degree_cap=True,no_policy_budget=True,no_teacher_at_inference=True,
        scope='full supplied ordered sequences; global h/c carried between supplied streams, no weight training; no ChampSim performance pilot')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
