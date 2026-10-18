#!/usr/bin/env python3
"""Archive only the complete, strictly verified v2 four-route formal study.

This reads completed analysis/raw metadata and creates fresh result folders. It
never runs a simulator, retrains, replaces an archive, or copies large raw data.
"""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import shutil
import time

EXP=Path(__file__).resolve().parents[1]
REPO=EXP.parents[2]
CACHE=next(p for p in REPO.parents if p.name=='cache')
PROTOCOL='four602_frozen_live_20261017_v2'
REQUIRED_CSV=('per_run.csv','raw_flat.csv','paired_comparisons.csv','storage_cost.csv',
              'route_status.csv','action_histograms.csv','action_histograms_full_roi.csv',
              'phase_progress.csv','progress_snapshots.csv','classified_requests.csv')
SUBSET_CSV=('per_run.csv','action_histograms.csv','action_histograms_full_roi.csv',
            'phase_progress.csv','progress_snapshots.csv','classified_requests.csv')
REQUIRED_JSON=('audit.json','metric_definitions.json','selection_verification.json',
               'callback_verification.json','cost_fields.json','inventory.json')
MAX_LIGHT_FILE_BYTES=64*1024*1024


def fail(message):
    raise RuntimeError(message)


def inside_cache(path):
    path=Path(path).resolve()
    if not path.is_relative_to(CACHE.resolve()):fail('All archive inputs/outputs must stay inside cache: '+str(path))
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='') as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames:fail('Missing CSV schema: '+str(path))
        return reader.fieldnames,list(reader)


def source_config(raw,build):
    # Metadata retains original server paths after a local sync. Prefer the
    # actual source snapshot underneath the supplied raw root, without rewriting
    # any historical path inside the original build/config files.
    relative=EXP.relative_to(REPO)/'config.json'
    candidates=[raw/'builds'/build['build_id']/'source_snapshot'/relative]
    if build.get('source_snapshot'):candidates.append(Path(build['source_snapshot'])/relative)
    candidates.append(raw/'config.json')
    for candidate in candidates:
        if candidate.is_file():return inside_cache(candidate)
    fail('Actual built config snapshot absent; sync the raw build/source evidence before archiving')


def validate(analysis,raw):
    for name in (*REQUIRED_CSV,*REQUIRED_JSON,'report.md','method.svg'):
        if not (analysis/name).is_file():fail('Required completed analysis artifact absent: '+name)
    audit=read_json(analysis/'audit.json')
    required={'status':'PASS','protocol':PROTOCOL,'complete_count':64,'expected_count':64,
              'resolved_expected_count':64,'failure_count':0,'plot_status':'generated',
              'callback_logs_checked':True,'actual_full_ROI_only':True,'raw_inputs_modified':False,
              'unresolved_test_selection':False}
    for key,value in required.items():
        if audit.get(key)!=value:fail('Final audit gate requires '+key+'='+str(value))
    if set(audit.get('stages',[]))!={'test','validation'}:fail('Archive must contain both complete formal stages')
    if audit.get('failures') or audit.get('missing_or_incomplete_run_ids'):fail('Final audit has missing/failed rows')
    if not audit.get('checks') or any(c.get('pass_') is not True for c in audit['checks']):fail('Not every explicit analysis check passed')
    if audit.get('run_status_counts')!={'complete':64}:fail('Expected exactly64 complete run records')
    fields,rows=read_csv(analysis/'per_run.csv')
    if len(rows)!=64 or len({r['run_id'] for r in rows})!=64:fail('Per-run table must contain64 unique complete rows')
    by_id={r['run_id']:r for r in rows}
    if Counter(r['stage'] for r in rows)!={'test':56,'validation':8}:fail('Expected56 test and8 validation rows')
    if Counter(r['roi'] for r in rows if r['stage']=='test')!={'0':28,'1':28}:fail('Expected28 test rows per ROI')
    if {r['policy'] for r in rows}!={'none','stride','streamer','ampm','spp'}:fail('Unexpected workload/policy scope')
    if {r['protocol'] for r in rows}!={PROTOCOL} or {r['status'] for r in rows}!={'complete'}:fail('Mixed protocol/incomplete per-run table')
    for r in rows:
        nominal=25000000 if r['stage']=='test' else 5000000
        if int(r['warmup'])!=5000000 or abs(int(r['instructions'])-nominal)>4:fail('Wrong full-ROI budget: '+r['run_id'])
        baseline=by_id.get(r.get('no_pf_run'))
        if not baseline or baseline['policy']!='none' or baseline['method']!='none':fail('Missing matched NoPF: '+r['run_id'])
        for key in ('protocol','build_id','stage','roi','skip','warmup','measurement_records','page_policy','macs'):
            if baseline.get(key)!=r.get(key):fail('Mismatched NoPF '+key+': '+r['run_id'])
    callbacks=read_json(analysis/'callback_verification.json')
    if not isinstance(callbacks,list) or len(callbacks)!=64:fail('Strict callback verification must cover every row')
    callback_by_id={c['run_id']:c for c in callbacks}
    if set(callback_by_id)!=set(by_id):fail('Callback verification identities differ from full formal table')
    required_callbacks={r['run_id'] for r in rows if r['method']=='student' and r['mode']=='full' and int(r['budget'])==0}
    if len(required_callbacks)!=9:fail('Expected9 real uncapped full-model callback logs')
    for rid,c in callback_by_id.items():
        if rid in required_callbacks and c.get('available') is not True:fail('Required actual callback log missing: '+rid)
        if c.get('available'):
            if c.get('complete_log') is not True or c.get('recorded_prefix_valid') is not True or c.get('missing_complete_rows')!=0 or c.get('malformed_rows'):
                fail('Callback log is not strictly complete: '+rid)
    for name in ('build.json','correctness_gate.json','selection.json'):
        if not (raw/name).is_file():fail('Original raw provenance absent: '+name)
    build=read_json(raw/'build.json');gate=read_json(raw/'correctness_gate.json');selection=read_json(raw/'selection.json')
    config_path=source_config(raw,build);config=read_json(config_path)
    if any(x.get('protocol')!=PROTOCOL for x in (build,gate,selection,config)):fail('Raw metadata protocol mismatch')
    if gate.get('status')!='PASS' or gate.get('build_id')!=build['build_id']:fail('Unverified current build certificate')
    if {r['build_id'] for r in rows}!={build['build_id']}:fail('Per-run table does not describe the current single build')
    selected=read_json(analysis/'selection_verification.json')
    if selected.get('available') is not True or selected.get('source')!='validation_only' or selected.get('selected')!=selection or selected.get('cap_recomputed')!=selection['common_cap']:
        fail('Validation-only frozen selection is not independently verified')
    if set(selected.get('templates',{}))!={'streamer','ampm','spp'}:fail('Validation template certificates absent')
    cap=int(selection['common_cap'])
    required_png=[]
    for roi in range(2):
        required_png += ['ipc_traffic_roi'+str(roi), 'action_distribution_roi'+str(roi),
                         'progress_axes_roi'+str(roi), 'phase_observations_roi'+str(roi)]
        required_png += ['requests_ipc_roi{}_cap{}'.format(roi,b) for b in (0,cap)]
    for stem in required_png:
        png=analysis/(stem+'.png');svg=analysis/(stem+'.svg')
        if not png.is_file() or png.read_bytes()[:8]!=b'\x89PNG\r\n\x1a\n':fail('Required generated PNG absent/invalid: '+str(png))
        if not svg.is_file() or '<svg' not in svg.read_text():fail('Required generated SVG absent/invalid: '+str(svg))
    for name in REQUIRED_CSV:
        _,table=read_csv(analysis/name)
        for row in table:
            if 'run_id' in row and row['run_id'] not in by_id:fail('Unknown/new-old mixed run ID in '+name)
            if 'policy' in row and row['policy'] not in {'none','stride','streamer','ampm','spp'}:fail('Unknown policy in '+name)
    return dict(audit=audit,fields=fields,rows=rows,by_id=by_id,build=build,gate=gate,
                selection=selection,config_path=config_path,required_callbacks=sorted(required_callbacks))


def copied_file(source,target):
    if source.stat().st_size>MAX_LIGHT_FILE_BYTES:fail('Refusing oversized lightweight artifact; preserve it at raw source: '+str(source))
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    if target.read_bytes()!=source.read_bytes():fail('Copied artifact bytes differ: '+str(target))
    return dict(path=str(target),source=str(source),bytes=source.stat().st_size,byte_equal=True)


def subset_csv(source,target,ids):
    fields,rows=read_csv(source);selected=[r for r in rows if r['run_id'] in ids]
    if 'run_id' not in fields:fail('Subset CSV lacks run_id schema: '+str(source))
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(selected)
    actual_fields,actual_rows=read_csv(target)
    if actual_fields!=fields or actual_rows!=selected:fail('Subset field content differs: '+str(target))
    return dict(path=target.name,source=str(source),source_rows=len(rows),retained_rows=len(selected),
                fields_and_values_exact=True)


def relative_link(source_dir,target):
    return Path(os.path.relpath(target,source_dir)).as_posix()


def archive(analysis,raw,destination):
    analysis=inside_cache(analysis);raw=inside_cache(raw);destination=inside_cache(destination)
    if analysis==destination or raw==destination:fail('Archive destination cannot replace source evidence')
    data=validate(analysis,raw)
    sibling_names={'stride':'602_lstm_stride_dynamic_inference','streamer':'602_lstm_streamer_live_inference',
                   'ampm':'602_lstm_ampm_live_inference','spp':'602_lstm_spp_live_inference'}
    sibling_dest={p:inside_cache(EXP.parent/name/'results/formal_20261017') for p,name in sibling_names.items()}
    targets=[destination,*sibling_dest.values()]
    if len(set(targets))!=5:fail('Full archive and four sibling destinations must be distinct')
    for target in targets:
        if target.exists():fail('Preserve existing archive; a fresh destination is required: '+str(target))
    # Validate all copies and subsets before publishing any final directory.
    stages={}
    for target in targets:
        target.parent.mkdir(parents=True,exist_ok=True)
        stage=target.with_name('.'+target.name+'.staging_'+str(time.time_ns()));stage.mkdir()
        stages[target]=stage
    stage=stages[destination];copies=[]
    for source in sorted(analysis.rglob('*')):
        if not source.is_file() or source.name=='raw_snapshots.json':continue
        if source.suffix.lower() not in ('.csv','.png','.svg','.md','.json','.html'):continue
        relative=source.relative_to(analysis)
        if relative==Path('README.md'):relative=Path('analysis_README.md')
        record=copied_file(source,stage/relative);record['path']=str(relative);copies.append(record)
    for name in ('build.json','correctness_gate.json','selection.json'):
        record=copied_file(raw/name,stage/'metadata'/name);record['path']='metadata/'+name;copies.append(record)
    for source,name in [(data['config_path'],'config.json'),(EXP/'route_manifest.json','route_manifest.json')]:
        record=copied_file(source,stage/'metadata'/name);record['path']='metadata/'+name;copies.append(record)
    subsets={}
    for policy,target in sibling_dest.items():
        own={r['run_id'] for r in data['rows'] if r['policy']==policy}
        baselines={data['by_id'][rid]['no_pf_run'] for rid in own}
        ids=own|baselines;part=stages[target];tables=[]
        for name in SUBSET_CSV:tables.append(subset_csv(analysis/name,part/name,ids))
        summary=dict(protocol=PROTOCOL,policy=policy,own_run_ids=sorted(own),matched_NoPF_run_ids=sorted(baselines),
                     run_ids=sorted(ids),own_rows=len(own),matched_NoPF_rows=len(baselines),rows=len(ids),
                     source_full_archive=str(destination),source_raw=str(raw),tables=tables,
                     no_old_protocol_rows=True,no_metric_recomputation=True)
        (part/'subset_manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
        full=relative_link(target,destination/'README.md')
        text=['# '+policy+' formal v2 result subset','',
              'Protocol: `'+PROTOCOL+'`. This is an exact row/field subset of the complete64-row archive,',
              'containing this policy and its same-build, same-split, same-ROI matched NoPF controls.',
              'It does not import any old replay, old live, first-formal or superseded-v1 row.','',
              'Own policy rows: {}; matched NoPF rows: {}; total: {}.'.format(len(own),len(baselines),len(ids)),'',
              '[Full results, figures, definitions and certificates]('+full+').','',
              'Tables preserve original fields and values without recomputing averages or hiding negative results.','']
        text += ['- ['+name+']('+name+').' for name in SUBSET_CSV]
        text += ['', 'Sparse action/classification tables need not contain NoPF rows when no matching event exists.',
                 'L2/LLC cohorts and their denominators remain separate; MAC0 is synchronous, cap is a ceiling,',
                 'and SPP effective fill-update inputs differ from its native PF-only feedback.','',
                 'Complete raw result root: `'+str(raw)+'`.','']
        (part/'README.md').write_text('\n'.join(text));subsets[policy]=summary
    manifest=dict(status='PASS',protocol=PROTOCOL,complete_rows=64,test_rows=56,validation_rows=8,
        source_analysis=str(analysis),source_raw=str(raw),destination=str(destination),build_id=data['build']['build_id'],
        input_audit_status='PASS',strict_callback_logs=True,required_full_callback_logs=data['required_callbacks'],
        plots_generated_and_required_pairs_present=True,all_copied_artifact_bytes_equal=True,
        copied_files=copies,total_copied_bytes=sum(r['bytes'] for r in copies),subsets=subsets,
        excluded=['raw_snapshots.json','raw traces','model/binary files','large simulator logs','raw callback/snapshot streams'],
        source_evidence_unchanged=True,no_hash_ledger=True,no_simulator_execution=True)
    (stage/'archive_verification.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# Complete four-route formal v2 archive','',
           'Protocol: `'+PROTOCOL+'`;64 completed full runs (56 test +8 validation), strict callback checks,',
           'generated PNG/SVG figures, one actual build and a fresh real-checkpoint correctness gate.','',
           '[Paper report](report.md), [per-run metrics](per_run.csv), [paired contrasts](paired_comparisons.csv),',
           '[Reviewed findings](FINDINGS.md), [H8 interpretation](FINDINGS.md#interpretation),',
           '[audit](audit.json), [metric definitions](metric_definitions.json),',
           '[callback verification](callback_verification.json), [cost fields](cost_fields.json),',
           '[storage/cost table](storage_cost.csv), [source metadata](metadata/),',
           '[copied-byte and subset certificate](archive_verification.json).','',
           'All lightweight CSV/PNG/SVG/Markdown/JSON/HTML analysis artifacts are preserved byte for byte.',
           'The large redundant `raw_snapshots.json`, raw event streams, models, binaries, traces and large logs',
           'remain at the full raw source. No old protocol is pooled into these tables.','',
           'Complete local raw root: `'+str(raw)+'`.','',
           'Recorded server raw root: `'+str(read_json(data['config_path'])['raw'])+'`.','',
           '## Policy subsets','']
    lines += ['- ['+p+']('+relative_link(destination,target/'README.md')+').' for p,target in sibling_dest.items()]
    lines += ['', 'ROI0 was previously observed; ROI1 is future relative to the original training/evaluation.',
              'Equal ceilings do not imply equal actual traffic. Native/student SPP effective fill inputs differ.',
              'The three global students are complete MAC0 forwards; Stride keeps its per-PC/decoder controls.',
              'Negative and mixed findings are retained. No finite-cost IPC or novelty claim is established by this archive.','']
    (stage/'README.md').write_text('\n'.join(lines))
    for target,temporary in stages.items():
        if target.exists():fail('Destination appeared during staging; preserved staged files at '+str(temporary))
    for target,temporary in stages.items():temporary.rename(target)
    print(json.dumps(dict(status='PASS',destination=str(destination),complete_rows=64,
        copied_files=len(copies),copied_bytes=manifest['total_copied_bytes'],
        sibling_destinations={p:str(target) for p,target in sibling_dest.items()}),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--destination',type=Path,default=EXP/'results/formal_20261017')
    args=parser.parse_args();archive(args.analysis,args.raw,args.destination)
