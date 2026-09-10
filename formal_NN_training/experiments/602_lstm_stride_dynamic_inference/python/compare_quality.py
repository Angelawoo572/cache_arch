#!/usr/bin/env python3
"""Compare three legacy quality metrics at matched, retained observations.

No interpolation, thresholds, model changes, or future-dependent state resets.
An EXCEEDS segment describes sampled metrics; it is not minimum history evidence.
"""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as a

QUALITY = ('useful_prefetch_coverage', 'selected_accuracy', 'legacy_timeliness')
COHORT = ('enqueued','timely','late','unused','redundant_cache','redundant_inflight','pending','resident_unused')


def compare_metrics(nn, ref):
    differences = {k: None if nn.get(k) is None or ref.get(k) is None else nn[k] - ref[k] for k in QUALITY}
    if any(v is None for v in differences.values()):
        status = 'NA'
    elif all(v > 0 for v in differences.values()):
        status = 'EXCEEDS'
    elif all(v >= 0 for v in differences.values()):
        status = 'NONLOWER_WITH_TIE'
    else:
        status = 'NOT_EXCEEDS'
    return status, differences


def exact_quality(c,baseline):
    """Use integer counter fractions for strict/equal classification."""
    def frac(n,d):
        return None if n is None or d in (None,0) else Fraction(int(n),int(d))
    u,i,q,l=(c.get(k) for k in ('pf_useful','pf_issued','pq_merged_duplicate_proxy','pf_late'))
    return dict(zip(QUALITY,(frac(u,baseline.get('l2_load_miss')),
        frac(u,None if i is None or q is None else i-q),
        frac(u,None if u is None or l is None else u+l))))


def segments(rows):
    """All maximal consecutive recorded states; keep the first exit observation."""
    output = []
    for index, row in enumerate(rows):
        if not output or row['quality_status'] != output[-1]['quality_status']:
            if output:
                output[-1]['exit_at_instructions'] = row['end_instructions']
                output[-1]['exit_status'] = row['quality_status']
                output[-1]['exit_limiting_metrics'] = row.get('limiting_metrics','')
            output.append({
                'quality_status': row['quality_status'],
                'first_recorded_instructions': row['end_instructions'],
                'start_window_instructions': row['start_instructions'],
                'first_eligible_l2': row.get('total_eligible_callbacks'),
                'first_admitted_l2': row.get('total_admitted_decisions'),
                'first_completed_l2': row.get('total_completed_decisions'),
                'preceding_recorded_instructions': rows[index-1]['end_instructions'] if index else None,
                'recorded_points': 0,
                'exit_at_instructions': None, 'exit_status': None,
            })
        current = output[-1]
        current['last_recorded_instructions'] = row['end_instructions']
        current['last_eligible_l2'] = row.get('total_eligible_callbacks')
        current['last_admitted_l2'] = row.get('total_admitted_decisions')
        current['last_completed_l2'] = row.get('total_completed_decisions')
        current['recorded_points'] += 1
        current['sampled_instruction_span'] = current['last_recorded_instructions'] - current['first_recorded_instructions']
        current['covered_window_instructions'] = None if row.get('scale') == 'cumulative' else current['last_recorded_instructions'] - current['start_window_instructions']
    return output


def matching_protocol(lhs, rhs):
    # State/service limits apply to NN treatment, not the hardware Stride baseline.
    if not a.match_protocol(lhs['identity'],rhs['identity']): return False
    lo=lhs['identity'].get('observed_trace_origin_records');ro=rhs['identity'].get('observed_trace_origin_records')
    return lo is None or ro is None or lo == ro


def checkpoint_tag(ident):
    value = str(ident['checkpoint'])
    if 'i20m' in value: return 'i20m'
    if 'i1m' in value: return 'i1m'
    return value


def select_reference(nn, runs, role):
    candidates = [r for r in runs if r['complete'] and matching_protocol(nn,r)]
    if role in ('summer','offline_same_checkpoint'):
        candidates = [r for r in candidates if r['identity']['arm'] == 'summer_replay'
                      and str(r['identity']['hidden_size']) == str(nn['identity']['hidden_size'])
                      and checkpoint_tag(r['identity']) == ('i20m' if role == 'summer' else checkpoint_tag(nn['identity']))]
    else:
        candidates = [r for r in candidates if r['identity']['arm'] == role]
    if len(candidates)>1:
        raise ValueError(f"ambiguous {role} reference for {nn['identity']['run_id']}: {[r['identity']['run_id'] for r in candidates]}")
    return candidates[0] if candidates else None


def series(run, scale):
    if run is None: return {}
    if scale == 'cumulative':
        return {(0,r['instructions']): {**r,'start_instructions':0,'end_instructions':r['instructions'],'snapshot':r,'cumulative_cycles':r['cycles']} for r in run['snapshots'] if r['instructions'] > 0}
    width = 100_000 if scale == 'interval_100k' else 1_000_000
    return {(r['start_instructions'],r['end_instructions']):r for r in a.windows(run['snapshots'],width)}


def counts_fields(c, baseline):
    return {
        'instructions':c.get('instructions'),'NL':c.get('l2_loads'),'M':c.get('l2_load_miss'),'M0':baseline.get('l2_load_miss') if baseline else None,
        'P':c.get('pf_requested'),'Ipf':c.get('pf_issued'),'Q':c.get('pq_merged_duplicate_proxy'),
        'U':c.get('pf_useful'),'L':c.get('pf_late'),
        'accuracy_denominator':None if c.get('pf_issued') is None or c.get('pq_merged_duplicate_proxy') is None else c['pf_issued']-c['pq_merged_duplicate_proxy'],
        'timeliness_denominator':None if c.get('pf_useful') is None or c.get('pf_late') is None else c['pf_useful']+c['pf_late'],
        'cycles':c.get('cycles'), **a.metrics(c,baseline),
    }


def cohort_fields(run,start,end):
    # Cohorts are final follow-up of requests issued in fixed existing 100k bins.
    # Small early milestones cut a bin: no exact cohort denominator is available.
    if start % 100_000 or (end % 100_000 and end != run['snapshots'][-1]['instructions']):
        return {'cohort_followup':'UNAVAILABLE: boundary cuts retained 100k issue cohort'}
    rows=[r for r in run['cohorts'] if start <= r['start_instructions'] < end]
    if not rows:return {'cohort_followup':'UNAVAILABLE'}
    totals={f'issue_cohort_{k}':sum(r.get(k,0) for r in rows) for k in COHORT}
    totals['issue_cohort_censored']=totals['issue_cohort_pending']+totals['issue_cohort_resident_unused']
    totals['cohort_followup']='outcomes observed by run end; censored pending/resident-unused; separate from legacy interval counters'
    return totals


def deployment_total(run,field='configured_max_bytes'):
    entries=a.storage_categories(run)
    values=[e[field] for e in entries]
    return sum(values)/1024 if values and all(v is not None for v in values) else None


def retirement_axes(ident,end):
    origin=ident.get('observed_trace_origin_records');skip=ident.get('skip_records')
    return {'roi_retired_instructions':end,
        'total_simulated_retired_instructions':origin-skip+end if origin is not None and skip is not None else None,
        'trace_record_ordinal':origin+end if origin is not None else None}


def compare_series(nn, reference, baseline, stride, scale, role):
    ns,rs,bs,ss=(series(r,scale) for r in (nn,reference,baseline,stride))
    totals={r['instructions']:r for r in nn['snapshots']}
    output=[]
    for key,c in ns.items():
        start,end=key; ref=rs.get(key); base=bs.get(key); st=ss.get(key); snap=totals[end]
        row={
            'run_id':nn['identity']['run_id'],'protocol':nn['identity']['protocol'],
            'hidden_size':nn['identity']['hidden_size'],'checkpoint':checkpoint_tag(nn['identity']),
            'state_capacity':nn['identity']['state_capacity'],'macs_per_cycle':nn['identity']['service_case'],
            'comparison':role,'reference_run_id':reference['identity']['run_id'] if reference else None,
            'no_pref_run_id':baseline['identity']['run_id'] if baseline else None,
            'stride_run_id':stride['identity']['run_id'] if stride else None,
            'scale':scale,'start_instructions':start,'end_instructions':end,
            'trace_slice_origin':nn['identity']['slice_origin_instructions'],
            'trace_origin_records_observed':nn['identity'].get('observed_trace_origin_records'),
            **retirement_axes(nn['identity'],end),
            'total_eligible_callbacks':snap.get('eligible_callbacks'),
            'total_admitted_decisions':snap.get('admitted_decisions'),
            'total_completed_decisions':snap.get('completed_decisions'),
            'interval_eligible_callbacks':c.get('eligible_callbacks'),
            'interval_completed_decisions':c.get('completed_decisions'),
            'occupancy_lines':snap.get('occupancy_lines'), 'capacity_lines':snap.get('l2_capacity_lines'),
            'total_input_drops':snap.get('input_drops'),
            'total_inference_service_cycles':snap.get('inference_service_cycles'),
            'total_nn_started':snap.get('nn_started'),
            'total_inference_macs':snap.get('inference_macs'),
            'total_fills_demand':snap.get('fills_load'),'total_fills_prefetch':snap.get('fills_prefetch'),
            'total_fills_writeback':snap.get('fills_writeback'),'total_replacements':snap.get('replacements'),
            'deployment_configured_kib':deployment_total(nn),
            'deployment_component_peaks_kib':deployment_total(nn,'observed_peak_bytes'),
            'mean_wait_cycles':a.ratio(snap.get('inference_wait_cycles_sum'),snap.get('nn_started')),
            'max_wait_cycles':snap.get('inference_wait_cycles_max'),
            'mean_completion_cycles':a.ratio(snap.get('inference_completion_cycles_sum'),snap.get('completed_decisions')),
            'max_completion_cycles':snap.get('inference_completion_cycles_max'),
            'mean_prior_state_updates':a.ratio(snap.get('state_history_prior_sum'),snap.get('state_history_completed')),
            'max_prior_state_updates':snap.get('state_history_prior_max'),
            'state_bytes_observed_peak':snap.get('state_bytes_peak'),
            'weight_kib':a.ratio(snap.get('weight_bytes'),1024),
            'snapshot_cohort_redundant_cache':snap.get('cohort_redundant_cache'),
            'snapshot_cohort_redundant_inflight':snap.get('cohort_redundant_inflight'),
            'snapshot_cohort_pending':snap.get('cohort_pending'),
            'snapshot_cohort_resident_unused':snap.get('cohort_resident_unused'),
            **counts_fields(c,base), **cohort_fields(nn,start,end),
        }
        row['cumulative_cycles']=c['cumulative_cycles']
        row['reference_cumulative_cycles']=ref['cumulative_cycles'] if ref else None
        row['cumulative_cycles_saved_vs_reference']=ref['cumulative_cycles']-c['cumulative_cycles'] if ref else None
        row['interval_cycles_saved_vs_reference']=ref['cycles']-c['cycles'] if ref else None
        row['cumulative_cycles_saved_vs_stride']=ss[key]['cumulative_cycles']-c['cumulative_cycles'] if key in ss else None
        row['interval_cycles_saved_vs_stride']=st['cycles']-c['cycles'] if st else None
        if reference is None: status='MISSING_REFERENCE';differences={k:None for k in QUALITY}
        elif ref is None or base is None: status='UNALIGNED_OR_MISSING_BASELINE';differences={k:None for k in QUALITY}
        else:
            reference_fields=counts_fields(ref,base)
            row.update({'reference_'+k:v for k,v in reference_fields.items()})
            status,exact_differences=compare_metrics(exact_quality(c,base),exact_quality(ref,base))
            differences={k:None if v is None else float(v) for k,v in exact_differences.items()}
        row.update(quality_status=status)
        row.update({'difference_'+k:v for k,v in differences.items()})
        row['tied_metrics']=';'.join(k for k,v in differences.items() if v==0)
        row['limiting_metrics']=';'.join(k for k,v in differences.items() if v is None or v<=0)
        output.append(row)
    return output


def joint_series(left,right):
    """Both references must pair to this same NN run and exact measurement scope."""
    output=[]
    for l,r in zip(left,right):
        keys=('run_id','scale','start_instructions','end_instructions')
        if any(l[k]!=r[k] for k in keys):raise ValueError('joint comparison mismatches NN run or interval')
        row={k:v for k,v in l.items() if not k.startswith(('reference_','difference_'))}
        row.update(comparison='stride_and_summer')
        row.pop('cumulative_cycles_saved_vs_reference',None);row.pop('interval_cycles_saved_vs_reference',None)
        for label,source in (('stride',l),('summer',r)):
            row[label+'_reference_run_id']=source.get('reference_run_id')
            row[label+'_quality_status']=source['quality_status']
            row[label+'_cumulative_cycles_saved_vs_reference']=source.get('cumulative_cycles_saved_vs_reference')
            row[label+'_interval_cycles_saved_vs_reference']=source.get('interval_cycles_saved_vs_reference')
            row.update({label+'_'+k:v for k,v in source.items() if k.startswith(('reference_','difference_')) and k!='reference_run_id'})
        statuses=[l['quality_status'],r['quality_status']]
        if all(s=='EXCEEDS' for s in statuses):status='EXCEEDS'
        elif any(s in ('MISSING_REFERENCE','UNALIGNED_OR_MISSING_BASELINE','NA') for s in statuses):status='MISSING_OR_UNDEFINED_REFERENCE'
        elif all(s in ('EXCEEDS','NONLOWER_WITH_TIE') for s in statuses):status='NONLOWER_WITH_TIE'
        else:status='NOT_EXCEEDS'
        row['quality_status']=status
        row['limiting_metrics']=';'.join(label+':'+metric for label,source in (('stride',l),('summer',r)) for metric in source['limiting_metrics'].split(';') if metric)
        row['tied_metrics']=';'.join(label+':'+metric for label,source in (('stride',l),('summer',r)) for metric in source.get('tied_metrics','').split(';') if metric)
        output.append(row)
    return output


def load_runs(roots):
    oracle=a.load_oracle();runs={}
    for root in roots:
        directories=([root] if (root/'run.json').is_file() else sorted(p.parent for p in root.glob('*/run.json')))
        for directory in directories:
            cfg=a.read_json(directory/'run.json');method=a.arm(cfg)
            if method not in ('no_pref','stride','frozen','summer_replay','stride_replay'):continue
            run=a.inspect_run(directory,oracle)
            run['identity']['skip_records']=cfg.get('skip_records',cfg.get('skip_instructions',cfg.get('skip',0)))
            if not run['complete']:continue
            rid=run['identity']['run_id']
            if rid in runs and runs[rid]['directory']!=run['directory']:raise ValueError('duplicate run id: '+rid)
            runs[rid]=run
    return list(runs.values())


def analyze(roots,out):
    runs=load_runs(roots);points=[];intervals=[];summary=[];first_points=[]
    for nn in sorted(runs,key=lambda r:r['identity']['run_id']):
        if nn['identity']['arm']!='frozen' or nn['identity']['run_id'].startswith('cold_history_'):continue
        baseline=select_reference(nn,runs,'no_pref');stride=select_reference(nn,runs,'stride');summer=select_reference(nn,runs,'summer');offline=select_reference(nn,runs,'offline_same_checkpoint')
        for scale in ('cumulative','interval_100k','interval_1m'):
            sr=compare_series(nn,stride,baseline,stride,scale,'stride')
            mr=compare_series(nn,summer,baseline,stride,scale,'summer')
            offline_rows=compare_series(nn,offline,baseline,stride,scale,'offline_same_checkpoint')
            comparisons=(sr,mr,joint_series(sr,mr),offline_rows)
            for rows in comparisons:
                if not rows:continue
                context={k:rows[0][k] for k in ('run_id','protocol','hidden_size','checkpoint','state_capacity','macs_per_cycle','scale','comparison')}
                spans=segments(rows)
                if not all(r['quality_status'] in ('MISSING_REFERENCE','UNALIGNED_OR_MISSING_BASELINE','MISSING_OR_UNDEFINED_REFERENCE') for r in rows):
                    points.extend(rows)
                intervals.extend({**context,**s} for s in spans)
                win=next((r for r in rows if r['quality_status']=='EXCEEDS'),None)
                if win: first_points.append(win)
                summary.append({**context,'sampled_points':len(rows),'strictly_exceeding_points':sum(r['quality_status']=='EXCEEDS' for r in rows),
                    'joint_win_intervals':sum(s['quality_status']=='EXCEEDS' for s in spans),
                    'status':'OBSERVED' if win else 'INSUFFICIENT_REFERENCE_DATA' if all(r['quality_status'] in ('MISSING_REFERENCE','UNALIGNED_OR_MISSING_BASELINE','MISSING_OR_UNDEFINED_REFERENCE') for r in rows) else 'NOT_REACHED_IN_MEASURED_RANGE',
                    'first_recorded_instructions':win['end_instructions'] if win else None,
                    'first_eligible_l2':win['total_eligible_callbacks'] if win else None,
                    'first_admitted_l2':win['total_admitted_decisions'] if win else None,
                    'first_completed_l2':win['total_completed_decisions'] if win else None,
                    'last_recorded_instructions':rows[-1]['end_instructions'],
                    'final_quality_status':rows[-1]['quality_status'],
                    'limiting_metric_counts':dict(Counter(metric for r in rows for metric in r['limiting_metrics'].split(';') if metric)),
                    'measurement':'exact matching retained milestones; strict > in coverage/selected accuracy/legacy timeliness; no minimum-history claim'})
    reference_summary=[]
    for run in runs:
        if run['identity']['arm']=='frozen':continue
        base=select_reference(run,runs,'no_pref')
        reference_summary.append({**run['identity'],**counts_fields(run['counters'],base['counters'] if base else None),
            'raw_directory':run['directory'],'complete':run['complete'],
            'retained_snapshot_count':len(run['snapshots'])})
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in (('quality_points',points),('quality_intervals',intervals),('quality_summary',summary),('quality_first_points',first_points),('quality_reference_summary',reference_summary)):
        a.write_csv(out/(name+'.csv'),rows)
    print(json.dumps({'completed_compatible_runs':len(runs),'comparison_points':len(points),'status_segments':len(intervals),'output':str(out)}))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,action='append',required=True);p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args();analyze(args.run_dir,args.out_dir)

if __name__=='__main__':main()
