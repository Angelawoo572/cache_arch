#!/usr/bin/env python3
"""Copy the completed first-formal lightweight artifacts, verifying every byte.

No simulation, result recomputation, hashes, logs, models or traces are involved.
Original source evidence and completed raw/build directories stay unchanged.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import shutil


def archive(study, destination):
    study=Path(study).resolve(); destination=Path(destination).resolve()
    source=study/'research_output'
    analysis=source/'analysis_final'
    entries=[]
    for path in sorted(analysis.iterdir()):
        if path.is_file() and path.name!='results.json':
            if path.suffix not in {'.csv','.png','.svg','.json','.md'}:
                raise RuntimeError('Unrecognized analysis artifact: '+str(path))
            entries.append((path,Path('analysis')/path.name))
    for name in ['analysis_final_verification.json','data_audit_final.json','hc_final_verification.json']:
        entries.append((source/name,Path('verification')/name))
    entries.append((source/'delivery/findings.json',Path('findings.json')))
    for name in ['pipeline_result.json','formal_matrix.json','training_pairs.json','budget_selection.json',
                 'training_split.json','resource_envelope.json','build.json']:
        entries.append((source/'raw_v2'/name,Path('metadata')/name))
    for name in ['storage_audit.json','storage_audit.md','numerical_domain_audit.json','spp_bounds_certificate.json']:
        entries.append((study/'research_audit/engine'/name,Path('audits')/name))
    with (analysis/'metrics_per_seed.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    stages=Counter(row['stage'] for row in rows)
    if stages!={'test':89,'validation':12}:
        raise RuntimeError('Unexpected completed first-formal matrix: '+str(stages))
    if {row['protocol'] for row in rows}!={'frozen_formal_20261016_v2'}:
        raise RuntimeError('Mixed protocols in first-formal table')
    if sum(p.suffix=='.csv' for p,_ in entries)!=11 or sum(p.suffix=='.png' for p,_ in entries)!=25 or sum(p.suffix=='.svg' for p,_ in entries)!=25:
        raise RuntimeError('Expected exactly 11 CSVs and 25 PNG/SVG figure pairs')
    # Check the complete source set and all existing destinations before writing.
    for path,relative in entries:
        if not path.is_file(): raise RuntimeError('Missing source artifact: '+str(path))
        target=destination/relative
        if target.exists() and target.read_bytes()!=path.read_bytes():
            raise RuntimeError('Refusing to overwrite a different archived artifact: '+str(target))
    records=[]
    for path,relative in entries:
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): shutil.copy2(path,target)
        if target.read_bytes()!=path.read_bytes():
            raise RuntimeError('Archived bytes differ: '+str(target))
        records.append(dict(path=str(relative),source_relative_path=str(path.relative_to(study)),
                            bytes=path.stat().st_size,byte_equal=True))
    result=dict(status='PASS',protocol='frozen_formal_20261016_v2',
                source_study_root=str(study),archive_root=str(destination),
                completed_rows=len(rows),test_rows=stages['test'],validation_rows=stages['validation'],
                copied_artifacts=len(records),total_copied_bytes=sum(r['bytes'] for r in records),
                figures={'png':25,'svg':25},csv_files=11,
                excluded=['analysis_final/results.json (redundant nested copy)',
                          'raw traces, models, binaries, large logs, raw simulator event streams'],
                no_source_changes=True,no_new_simulation=True,artifacts=records)
    (destination/'archive_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();r=archive(args.study_root,args.out)
    print(json.dumps({k:r[k] for k in ['status','completed_rows','copied_artifacts','total_copied_bytes','csv_files','figures']}))
