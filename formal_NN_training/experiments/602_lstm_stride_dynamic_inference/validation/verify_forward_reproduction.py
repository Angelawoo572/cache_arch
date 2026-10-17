#!/usr/bin/env python3
'''Compare generated forward bytes to actual archived simulator sources.'''
import argparse
import importlib.util
import json
from pathlib import Path

EXP=Path(__file__).resolve().parents[1]
ROOT=EXP.parents[2]
SPEC=importlib.util.spec_from_file_location('checked_prepare_forward',EXP/'runtime/prepare_forward.py')
FORWARD=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FORWARD)


def verify(repo, builds):
    checks=[]
    for build in builds:
        build=Path(build)
        sim=build/'simulator'
        generated=FORWARD.verify_prepared(repo,sim)
        archived=FORWARD.verify_prepared(build/'source_snapshot',sim)
        canonical=[]
        for name in FORWARD.NAMES:
            a=Path(repo)/FORWARD.SOURCE/name
            b=build/'source_snapshot'/FORWARD.SOURCE/name
            if a.read_bytes()!=b.read_bytes():
                raise RuntimeError('canonical live source differs from archived source: '+name)
            canonical.append(dict(path=str(FORWARD.SOURCE/name),byte_equal=True))
        binary=sim/'bin/perceptron-no-dynamic602-no-ship-1core'
        frozen=build/'simulator_binary_snapshot'
        if binary.read_bytes()!=frozen.read_bytes():
            raise RuntimeError('actual simulator binary differs from immutable snapshot')
        checks.append(dict(build_id=build.name,actual_build=str(build),
            current_generation_vs_actual=generated,archived_canonical_vs_actual=archived,
            canonical_source_comparisons=canonical,binary_snapshot_equal=True,
            note='Legacy build may lack the new generation marker; exact source bytes are still checked.'))
    return dict(status='PASS',scope='Forward source reproduction only; historical observer/runner protocols remain separate.',
                current_repo=str(Path(repo).resolve()),builds=checks,
                generated_artifact_comparisons=4*len(checks),canonical_source_comparisons=4*len(checks),
                no_simulation_or_retraining=True,no_completed_sources_modified=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=ROOT)
    p.add_argument('--build',type=Path,action='append',required=True,
                   help='Actual immutable build directory containing simulator/ and source_snapshot/')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=verify(a.repo,a.build)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','generated_artifact_comparisons','canonical_source_comparisons','no_simulation_or_retraining']}))
