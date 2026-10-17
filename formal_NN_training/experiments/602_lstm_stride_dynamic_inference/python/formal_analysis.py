#!/usr/bin/env python3
"""Aggregate only the new formal protocol; preserve every raw counter and seed.

Read-only inputs. No replay/historical result import, imputed missing runs,
per-request causal attribution, or issued-times-line-size DRAM estimate.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
import re
from pathlib import Path
import statistics

EXP = Path(__file__).resolve().parents[1]
DEFAULT_RAW = Path('/home/qianruw/cache_formal_20261016_runs')
IDENTITY = ('run_id', 'stage', 'workload', 'roi', 'arm', 'seed', 'loss', 'macs',
            'budget', 'silent', 'queue', 'history', 'protocol', 'build_id')
PROTOCOL_FIELDS = ('protocol', 'build_id', 'stage', 'workload', 'roi', 'trace',
                   'warmup', 'skip', 'instructions', 'controls')
PAIR_FIELDS = PROTOCOL_FIELDS + ('arm', 'seed', 'loss', 'macs', 'budget',
                                'silent', 'queue', 'history')
DEFINITIONS = {
    'ipc': 'ROI retired instructions / ROI cycles.',
    'l2_mpki_load': '1000 * M / ROI retired instructions; M is L2 LOAD misses.',
    'load_miss_rate': 'M / NL, with LOAD-only cache counters; undefined ratios are null.',
    'speedup_vs_nopf': 'IPC / IPC of same-build, same-workload, same-ROI, same machine/protocol NoPF.',
    'timely_load_coverage': 'cohort_timely_load / matched NoPF M0. LOAD-only timely accepted-request cohort outcomes.',
    'all_useful_load_coverage': '(cohort_timely_load + cohort_late_load) / matched NoPF M0; includes late usefulness, distinct from miss coverage.',
    'load_useful_accuracy_enqueued': '(cohort_timely_load + cohort_late_load) / cohort_enqueued. Denominator includes accepted lifecycle requests, duplicates and censored requests.',
    'load_useful_accuracy_submitted': '(cohort_timely_load + cohort_late_load) / request_gate_allowed; includes cache merges and cache rejections in the submitted-request denominator.',
    'timely_load_accuracy_enqueued': 'cohort_timely_load / cohort_enqueued.',
    'load_useful_timeliness': 'cohort_timely_load / (cohort_timely_load + cohort_late_load). RFO usefulness excluded.',
    'load_useful_accuracy_resolved': '(cohort_timely_load + cohort_late_load) / (cohort_enqueued - cohort_pending - cohort_resident_unused); conditional on resolution, reported separately.',
    'load_miss_reduction': '(M0 - M) / M0 with matched NoPF LOAD misses; distinct from usefulness coverage.',
    'legacy_accuracy_U_over_Ipf': 'Legacy U / Ipf; U can include RFO, retained for compatibility only.',
    'legacy_timeliness_U_over_UplusL': 'Legacy U / (U + L); not the corrected LOAD-only headline.',
    'gate_requests_per_million_instructions': '1e6 * request_gate_allowed / instructions; actual common-gate request submissions after page/budget gates.',
    'cache_enqueued_per_million_instructions': '1e6 * request_cache_enqueued / instructions; distinct from generated, merged and rejected.',
    'common_budget': 'Shared cumulative maximum selected on validation only. Equal configured caps do not guarantee equal realized request counts.',
    'lifecycle': 'enqueued = timely + late + evicted-unused + redundant-cache + redundant-inflight + pending + resident-unused. Pending/resident-unused are right-censored, never silently classified as useless.',
    'pollution': 'Not measured. Evicted-unused and 1-accuracy are not a counterfactual pollution estimate.',
    'dram_queue_accepted_payload_bytes': 'Actual DRAM queue enqueue events times active line_bytes, labeled accepted payload; merges/forwarding are separate, not bus transfers.',
    'dram_started_bytes': 'Actual instrumented DRAM bus start events times active line_bytes.',
    'dram_completed_bytes': 'Recorded ROI-started DRAM bus transfers whose modeled finish time has arrived. Initial pending queue entries and initial busy channels are separate boundary state.',
    'nn_weight_traffic': 'weight_bytes is resident FP32 model storage. inference_weight_read_bytes=4*inference_macs is analytical dense matrix operand-read payload, not measured SRAM or DRAM traffic; source bias reads are excluded. Dense MAC and weight-read phases overlap by max(MAC time, bandwidth time); nonlinear/control/scratch phases serialize. Bias traffic and physical ports/energy are not separately charged. nn_weight_dram_bytes=0 assumes resident weights.',
    'histograms': 'Exact counts in log2 upper-bound bins 0,1,3,7,...; U=0 covers[0,0], U>0 covers[(U+1)/2,U], even when intermediate empty bins are omitted. Histograms retain tails; plots show binned CDF, not interpolated latency quantiles.',
    'history': 'Main v2 compares operational predictor warmup history: cache warmup and future ROI match, but h/c AND PC tags/LRU/occupancy at the ROI boundary may differ. This does not isolate recurrent h/c history. Per-PC completions/evictions remain actual measurements. Separate fixed-metadata h/c-only full formal controls are pending and must be reported under their own build/protocol. No first crossing establishes minimum history.',
    'storage': 'Raw configured/peak counters remain untouched. The NN raw deployment_bytes_configured=16808 labels core arrays/events/scratch reservations, omitting controller bookkeeping. For the audited h8/state64/queue16/32 configuration, complete conservative source-field budgets are NN16920B, Stride2136B and nativeSPP5938B; the common request-cap schema adds16B to each (16936/2152/5954B). Values are applied only to matching audited protocol/build/resources; storage_source_field_audit.json preserves full inventory. These are source-field reservations, not RTL/SRAM macro area or measured physical hardware.',
    'storage_host_boundaries': 'host_cpp_sizeof_* is actual server GCC/libstdc++ host object sizeof, excludes pointed-to allocations/call stack and includes diagnostic representations. host_peak_rss_kib is the entire simulator process peak RSS from host_time.txt, never predictor SRAM. Neither is substituted for logical source-field budgets. Component peak sums are an envelope, not simultaneous allocation.',
    'storage_diagnostics': 'Shadow teacher, teacher address/category snapshots, timestamps/histograms, per-PC/lifecycle maps, CSV occurrence maps, SPP diagnostic histograms, logs and offline optimizer/training memory are excluded from deployment budget and remain measurement overhead. NativeSPP logical storage is audited; finite service timing is unmodeled. Any candidate-array guard certificate applies only to its listed completed formal runs and is not an unbounded source-code proof.',
    'seed_reporting': 'Each completed seed appears separately in raw_per_seed.csv, metrics_per_seed.csv, paired_comparisons.csv and distribution/history tables. Main charts retain each actual run point; optional mean/sample-SD marks do not replace seed data. Service CDFs show individual seed/loss curves rather than pooled averages.',
    'paired_delta': 'Treatment IPC minus exact-matching control IPC, with only the declared comparison field(s) varied. A silent-output ablation estimates that whole simulation intervention, not the causal value of individual request labels.',
    'cross_workload': 'NN model_metadata.training_trace identifies the training workload. A 602 checkpoint on another workload is transfer, not per-workload retraining.',
}


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def ratio(n, d):
    return n / d if finite(n) and finite(d) and d != 0 else None


def add(*values):
    return sum(values) if all(finite(v) for v in values) else None


def subtract(a, b):
    return a - b if finite(a) and finite(b) else None


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = prefix + str(key)
        if isinstance(item, dict):
            # Preserve nested values in a JSON cell as well as scalar columns.
            result[name] = item
            result.update(flatten(item, name + '__'))
        else:
            result[name] = item
    return result


def identity(run):
    result={name:run['spec'].get(name) for name in IDENTITY}
    result['training_seed']=run['spec'].get('seed') if run['spec'].get('arm')=='nn' else None
    return result


def key(spec, fields=PROTOCOL_FIELDS, omit=()):
    return json.dumps({k: spec.get(k) for k in fields if k not in omit},
                      sort_keys=True, separators=(',', ':'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: '' if v is None else json.dumps(v, sort_keys=True)
                             if isinstance(v, (dict, list)) else v for k, v in row.items()})


def inspect(path, protocol):
    try:
        record = read_json(path)
    except (ValueError, OSError) as error:
        return dict(spec={'run_id': path.parent.name}, status='invalid_metadata',
                    errors=[str(error)], directory=str(path.parent), summary={})
    spec = record.get('spec', {})
    run = dict(spec=spec, status=record.get('status', 'unknown'), errors=[],
               directory=str(path.parent), summary={}, record=record)
    if spec.get('protocol') != protocol:
        run['status'] = 'excluded_protocol'
        run['errors'].append('Different or absent formal protocol; never imported into comparisons.')
        return run
    if run['status'] != 'complete':
        return run
    summary_path = path.parent / 'summary.json'
    if not summary_path.is_file():
        run['status'] = 'incomplete_missing_summary'
        run['errors'].append('run.json claimed complete but summary.json is absent.')
        return run
    try:
        summary = read_json(summary_path)
        run['summary'] = summary
        if summary.get('event') != 'snapshot' or summary.get('final') is not True:
            raise ValueError('Summary is not a final observer snapshot.')
        if not finite(summary.get('cycles')) or summary['cycles'] <= 0:
            raise ValueError('Invalid ROI cycle denominator.')
        if not finite(summary.get('instructions')) or abs(summary['instructions'] - spec['instructions']) > 4:
            raise ValueError('ROI instruction count differs by more than the allowed retirement batch.')
        if summary.get('numerical_failures', 0):
            raise ValueError('Numerical inference failures are present.')
        lifecycle_fields=('cohort_timely','cohort_late','cohort_unused','cohort_redundant_cache',
                          'cohort_redundant_inflight','cohort_pending','cohort_resident_unused')
        if (all(finite(summary.get(k)) for k in lifecycle_fields)
                and finite(summary.get('cohort_enqueued'))
                and sum(summary[k] for k in lifecycle_fields)!=summary['cohort_enqueued']):
            raise ValueError('Request lifecycle conservation failed.')
        run['validated'] = True
    except (ValueError, KeyError, OSError) as error:
        run['status'] = 'invalid_complete_result'
        run['errors'].append(str(error))
    return run


def audit_budget(raw_root, protocol):
    path = raw_root / 'budget_selection.json'
    report = {'status': 'unavailable', 'source': str(path), 'workloads': {}}
    if not path.is_file():
        return report
    selection = read_json(path)
    report['selection'] = selection
    report['status'] = 'valid' if selection.get('protocol') == protocol else 'invalid_protocol'
    for workload, cap in selection.get('requests_per_million_instructions', {}).items():
        item = {'configured_cap': cap, 'status': 'invalid_evidence'}
        report['workloads'][workload] = item
        try:
            evidence = selection['evidence'][workload]
            source_id = evidence['source_run']
            if Path(source_id).name != source_id:
                raise ValueError('Validation source must be a run identifier.')
            directory = raw_root / 'validation' / source_id
            source = read_json(directory / 'run.json')
            snapshot = read_json(directory / 'summary.json')
            spec = source['spec']
            if (source.get('status') != 'complete' or spec.get('stage') != 'validation'
                    or spec.get('protocol') != protocol or spec.get('workload') != workload
                    or spec.get('arm') != 'stride_forward1' or snapshot.get('final') is not True):
                raise ValueError('Budget source is not a completed same-protocol validation forward1 run.')
            if not finite(snapshot.get('instructions')) or snapshot['instructions']<=0 or not finite(snapshot.get('P')) or snapshot['P']<=0:
                raise ValueError('Validation budget source has no positive instruction/request denominator.')
            expected = int(snapshot['P'] * 1000000 // snapshot['instructions'])
            if (cap != expected or evidence.get('requests') != snapshot['P']
                    or evidence.get('instructions') != snapshot['instructions']):
                raise ValueError('Cap or evidence differs from validation request counts.')
            item.update(status='valid', source_run=source_id, requests=snapshot['P'],
                        instructions=snapshot['instructions'], recomputed_cap=expected)
        except (KeyError, ValueError, OSError) as error:
            item['error'] = str(error)
    if report['status'] == 'valid' and any(v['status'] != 'valid' for v in report['workloads'].values()):
        report['status'] = 'invalid_evidence'
    return report


def derive(run, reference):
    c = run['summary']
    b = reference['summary'] if reference else {}
    t, l = c.get('cohort_timely_load'), c.get('cohort_late_load')
    useful = add(t, l)
    enqueued = c.get('cohort_enqueued')
    unresolved = add(c.get('cohort_pending'), c.get('cohort_resident_unused'))
    m0 = b.get('M')
    ipc = ratio(c.get('instructions'), c.get('cycles'))
    d = dict(ipc=ipc, nopf_reference=reference['spec']['run_id'] if reference else None,
             nopf_load_misses=m0,
             l2_mpki_load=ratio(1000 * c['M'], c.get('instructions')) if finite(c.get('M')) else None,
             load_miss_rate=ratio(c.get('M'), c.get('NL')),
             speedup_vs_nopf=ratio(ipc, ratio(b.get('instructions'), b.get('cycles'))),
             cycles_saved_vs_nopf=subtract(b.get('cycles'), c.get('cycles')),
             timely_load_coverage=ratio(t, m0), all_useful_load_coverage=ratio(useful, m0),
             load_miss_reduction=ratio(subtract(m0, c.get('M')), m0),
             load_useful_accuracy_enqueued=ratio(useful, enqueued),
             load_useful_accuracy_submitted=ratio(useful, c.get('request_gate_allowed')),
             timely_load_accuracy_enqueued=ratio(t, enqueued),
             load_useful_accuracy_resolved=ratio(useful, subtract(enqueued, unresolved)),
             load_useful_timeliness=ratio(t, useful),
             lifecycle_unresolved_fraction=ratio(unresolved, enqueued),
             legacy_accuracy_U_over_Ipf=ratio(c.get('U'), c.get('Ipf')),
             legacy_timeliness_U_over_UplusL=ratio(c.get('U'), add(c.get('U'), c.get('L'))),
             generated_per_eligible_load=ratio(c.get('generated_addresses'), c.get('eligible_l2_callbacks')),
             gate_requests_per_million_instructions=ratio(1000000*c['request_gate_allowed'], c.get('instructions')) if finite(c.get('request_gate_allowed')) else None,
             cache_enqueued_per_million_instructions=ratio(1000000*c['request_cache_enqueued'], c.get('instructions')) if finite(c.get('request_cache_enqueued')) else None,
             input_drop_fraction=ratio(c.get('inference_dropped'), c.get('eligible_l2_callbacks')),
             mean_queue_wait_cycles=ratio(c.get('inference_wait_cycles_sum'), c.get('nn_started')),
             mean_completion_age_cycles=ratio(c.get('inference_completion_cycles_sum'), c.get('nn_completed')),
             mean_completed_service_cycles=ratio(c.get('inference_completed_service_cycles_sum'), c.get('nn_completed')),
             mean_prior_completed_updates=ratio(c.get('state_history_prior_sum'), c.get('state_history_completed')),
             trained_workload=(run['spec'].get('model_metadata', {}).get('training_trace')
                 or run['spec'].get('model_metadata', {}).get('training_config', {}).get('train_trace')
                 or run['spec'].get('model_metadata', {}).get('trace')))
    if run['spec'].get('arm') == 'nn':
        d['workload_training_mode'] = ('same_workload_training' if d['trained_workload'] == run['spec'].get('workload')
                                       else 'cross_workload_same_checkpoint' if d['trained_workload'] else 'training_workload_unavailable')
    outcomes = [c.get(k) for k in ('cohort_timely', 'cohort_late', 'cohort_unused',
                'cohort_redundant_cache', 'cohort_redundant_inflight', 'cohort_pending', 'cohort_resident_unused')]
    d['lifecycle_conservation_residual'] = subtract(enqueued, add(*outcomes))
    if d['lifecycle_conservation_residual'] not in (0, None):
        run['errors'].append('Request lifecycle does not conserve.')
    if reference and b.get('instructions') != c.get('instructions'):
        run['errors'].append('Actual retired counts differ within tolerated batch; IPC is normalized, raw cycle saving has differing counts.')
    return d


def paired(runs, default_history):
    results = []
    for treatment in runs:
        s = treatment['spec']
        wanted = []
        if s.get('arm') == 'nn':
            if s.get('silent'): wanted.append(('silent_output_suppression', {'silent': 0}))
            if s.get('queue') != 'fifo': wanted.append(('queue_policy', {'queue': 'fifo'}))
            if s.get('loss') == 'unweighted': wanted.append(('emit_loss', {'loss': 'balanced'}))
            if s.get('macs', 0) > 0: wanted.append(('finite_service', {'macs': 0}))
            long = min(default_history, s.get('warmup', default_history))
            if s.get('history', 0) < long: wanted.append(('state_history_vs_long', {'history': long}))
        if s.get('budget', 0) > 0: wanted.append(('common_cap_vs_original', {'budget': 0}))
        if s.get('arm') == 'stride_remove_current': wanted.append(('remove_current_only', {'arm': 'stride_legacy'}))
        if s.get('arm') == 'stride_forward1': wanted.append(('forward1_vs_remove_current', {'arm': 'stride_remove_current'}))
        if s.get('arm') == 'stride_forward2': wanted.append(('additional_lookahead', {'arm': 'stride_forward1'}))
        for comparison, changes in wanted:
            target = dict(s, **changes)
            candidates = [r for r in runs if key(r['spec'], PAIR_FIELDS) == key(target, PAIR_FIELDS)]
            row = dict(identity(treatment), comparison=comparison, changed_fields=changes,
                       treatment=s['run_id'], control=None, comparison_status='missing_control')
            if len(candidates) == 1:
                control = candidates[0]
                row.update(control=control['spec']['run_id'], comparison_status='paired',
                           treatment_ipc=treatment['metrics']['ipc'], control_ipc=control['metrics']['ipc'],
                           delta_ipc=subtract(treatment['metrics']['ipc'], control['metrics']['ipc']),
                           ipc_ratio=ratio(treatment['metrics']['ipc'], control['metrics']['ipc']))
                for counter in ('request_gate_allowed', 'request_cache_enqueued', 'generated_addresses',
                                'nn_admitted', 'nn_completed', 'state_evictions', 'dram_total_bus_bytes'):
                    row['delta_' + counter] = subtract(treatment['summary'].get(counter), control['summary'].get(counter))
            elif len(candidates) > 1:
                row['comparison_status'] = 'ambiguous_control'
            row['interpretation'] = DEFINITIONS['paired_delta']
            if comparison=='state_history_vs_long':
                row['history_comparison_scope']='operational_predictor_history_hc_and_tags_LRU_occupancy'
                row['interpretation']=DEFINITIONS['history']
            results.append(row)
    return results


def request_gaps(runs):
    rows = []
    for run in runs:
        s = run['spec']
        if s.get('arm') != 'nn' or not s.get('budget', 0): continue
        refs = [r for r in runs if key(s) == key(r['spec']) and r['spec'].get('budget') == s['budget']
                and r['spec'].get('arm') in ('stride_forward1', 'stride_forward2', 'spp')]
        for ref in refs:
            c, b = run['summary'], ref['summary']
            row = dict(identity(run), comparison_run=ref['spec']['run_id'],
                       comparison_arm=ref['spec']['arm'], configured_cap_equal=True,
                       realized_submission_gap=subtract(c.get('request_gate_allowed'), b.get('request_gate_allowed')),
                       realized_submission_ratio=ratio(c.get('request_gate_allowed'), b.get('request_gate_allowed')),
                       realized_enqueued_gap=subtract(c.get('request_cache_enqueued'), b.get('request_cache_enqueued')),
                       realized_enqueued_ratio=ratio(c.get('request_cache_enqueued'), b.get('request_cache_enqueued')))
            row['realized_submission_counts_exactly_equal'] = row['realized_submission_gap'] == 0
            row['interpretation'] = DEFINITIONS['common_budget']
            rows.append(row)
    return rows


def load_storage_audit(explicit=None):
    candidates=[explicit] if explicit else []
    if not explicit:
        candidates += [EXP/'storage_audit.json']
        candidates += [parent/'research_audit/engine/storage_audit.json'
                       for parent in (Path.cwd(),)+tuple(EXP.parents)]
    source=next((p for p in candidates if p and p.is_file()),None)
    if source is None:
        return {'status':'unavailable','error':'Provide --storage-audit with the source-field inventory JSON.'}
    try:
        audit=read_json(source)
        if not audit.get('accounting_summary') or not audit.get('boundaries'):
            raise ValueError('Storage audit lacks accounting summary or explicit domain boundaries.')
        return {'status':'available','source':str(source.resolve()),'audit':audit}
    except (ValueError,OSError) as error:
        return {'status':'invalid','source':str(source),'error':str(error)}


def source_storage_fields(run, evidence):
    spec,c=run['spec'],run['summary']
    result={'source_field_budget_status':'audit_unavailable',
            'source_field_budget_is_rtl_or_silicon_area':False,
            'host_rss_scope':DEFINITIONS['storage_host_boundaries']}
    directory=run.get('directory')
    host_path=Path(directory)/'host_time.txt' if directory else None
    host_text=host_path.read_text(errors='replace') if host_path and host_path.is_file() else ''
    match=re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',host_text)
    result['host_peak_rss_kib']=int(match.group(1)) if match else None
    result['host_wall_seconds']=run.get('record',{}).get('wall_seconds')
    if not evidence or evidence.get('status')!='available':return result
    audit=evidence['audit']
    result.update(source_audit_date=audit.get('audit_date'),source_audit_protocol=audit.get('protocol'),
                  source_audit_build_id=audit.get('build_id'))
    if spec.get('protocol')!=audit.get('protocol'):
        result['source_field_budget_status']='audit_protocol_mismatch';return result
    if spec.get('build_id')!=audit.get('build_id'):
        result['source_field_budget_status']='audit_build_mismatch';return result
    cap_bytes=audit['common_budget_gate']['chosen_register_schema_bytes'] if spec.get('budget',0)>0 else 0
    arm=spec.get('arm');parts={};fixed=None;capped=None
    if arm=='nn':
        design=audit['nn_h8_state64']
        required={'h':design['hidden_size'],'state_capacity':design['state_capacity'],
                  'input_queue_capacity':design['input_queue_capacity'],
                  'output_queue_capacity':design['output_queue_capacity']}
        if any(c.get(k)!=v for k,v in required.items()) or spec.get('controls',{}).get('decoder_limit')!=design['decoder_capacity']:
            result['source_field_budget_status']='audited_nn_resource_shape_mismatch';return result
        parts=dict(design['components_bytes'])
        parts['serial_fifo_state_controller']=design['controller_schema']['total_bytes']
        fixed=design['complete_fixed_design_source_field_budget_bytes']
        capped=design['complete_budget_matched_source_field_budget_bytes']
        result.update(source_core_arrays_events_scratch_bytes=design['array_payload_and_forward_scratch_bytes'],
                      source_controller_schema=design['controller_schema'],
                      source_raw_counter_scope_erratum=design['raw_counter_label_correction'],
                      host_cpp_sizeof_components=design['host_sizeof'],
                      host_cpp_sizeof_object_bytes=design['host_sizeof']['Dynamic602_object_bytes'],
                      source_optional_programmable_knobs_additional_bytes=design['controller_schema']['programmable_resource_registers_if_supported']['additional_bytes'],
                      source_dense_weight_traffic_caveats=audit['nn_dense_weight_traffic'])
    elif arm and arm.startswith('stride'):
        design=audit['stride'];schema=design['complete_control_scratch_schema']
        parts={'tracker_entry_reservations':schema['entry_storage_bytes'],
               'existing_global_controller':schema['existing_global_controller_bytes'],
               'source_scalar_reservation':schema['word_aligned_scalar_reservation_bytes'],
               'immediate_output_payload':schema['immediate_output_payload_bytes']}
        fixed=schema['complete_fixed_design_source_field_budget_bytes']
        capped=schema['complete_budget_matched_source_field_budget_bytes']
        result.update(source_stride_control_scratch_schema=schema,
                      host_cpp_sizeof_object_bytes=design['cpp_predictor_object_bytes'],
                      host_cpp_tracker_payload64_bytes=design['cpp_tracker_payload_for64_bytes'],
                      host_cpp_object_plus_tracker_lower_bound_bytes=design['cpp_object_plus_tracker_payload_lower_bound_bytes'],
                      host_cpp_dynamic_exclusions=design['additional_cpp_allocations'],
                      source_forward_helper_representation_note=design['forward_helper'])
    elif arm=='spp':
        design=audit['spp_native'];schema=design['source_scalar_control_schema']
        parts={item['component']:item['logical_bytes'] for item in design['persistent_components']}
        parts.update(candidate_arrays_native_width=schema['candidate_arrays_native_width_bytes'],
                     source_scalar_control_reservation=schema['total_reserved_bytes'])
        fixed=schema['complete_fixed_design_source_field_budget_bytes']
        capped=schema['complete_budget_matched_source_field_budget_bytes']
        result.update(source_spp_persistent_inventory=design['persistent_components'],
                      source_spp_scalar_control_schema=schema,
                      host_cpp_sizeof_object_bytes=design['cpp_predictor_object_bytes'],
                      host_cpp_persistent_tables_bytes=design['cpp_persistent_tables_bytes'],
                      host_cpp_dynamic_exclusions=design['cpp_dynamic_exclusions'],
                      source_optional_programmable_knobs_additional_bytes=schema['programmable_thresholds_if_supported_additional_bytes'],
                      source_spp_finite_cost_status=design['cost_model'],
                      source_spp_candidate_array_safety_source_flag=design['lookahead_limits'].get('array_safety_verified'),
                      source_spp_candidate_array_safety_evidence=design['lookahead_limits'],
                      source_spp_candidate_array_safety_scope='Use the full guard-certificate scope/status in the audit; this source flag is not a universal unbounded-code proof.',
                      source_spp_candidate_array_safety_limitation=design['lookahead_limits'].get('reason'))
    elif arm=='none':
        fixed=audit['noprefetch']['additional_predictor_storage_bytes'];capped=fixed
        cap_bytes=0
    else:
        result['source_field_budget_status']='arm_not_audited';return result
    parts['common_request_cap_controller']=cap_bytes
    selected=capped if cap_bytes else fixed
    if sum(parts.values())!=selected:
        raise ValueError('Storage audit component total does not match declared schema for '+str(arm))
    result.update(source_field_budget_status='audited_fixed_design_schema',
                  source_field_components_bytes=parts,
                  source_field_fixed_design_bytes=fixed,source_field_common_cap_design_bytes=capped,
                  source_field_for_this_run_bytes=selected,source_request_cap_controller_bytes=cap_bytes,
                  source_field_budget_scope=audit['accounting_summary']['all_are'],
                  source_excluded_diagnostics=audit['excluded_diagnostics'])
    result.update({'source_field_component__'+k:v for k,v in parts.items()})
    return result


def ancillary(runs, storage_evidence=None):
    tables = {k: [] for k in ('lifecycle', 'dram_traffic', 'storage', 'service_distributions',
                              'history_per_pc', 'resident_state_lifetimes')}
    for run in runs:
        c = run['summary']; ident = identity(run)
        tables['lifecycle'].append(dict(ident, **{k: v for k, v in c.items() if k.startswith(
            ('cohort_', 'request_', 'teacher_', 'shadow_', 'cache_request_')) or k in
            ('generated_addresses', 'eligible_l2_callbacks', 'nn_admitted', 'nn_started', 'nn_completed',
             'inference_dropped', 'queue_evicted_oldest', 'output_dropped_addresses', 'decoder_dropped_addresses',
             'page_filtered_addresses', 'silent_suppressed_addresses', 'budget_suppressed_addresses',
             'input_queue', 'output_queue', 'inference_inflight', 'pf_useless', 'pf_dropped')}))
        traffic = dict(ident, **{k: v for k, v in c.items() if k.startswith('dram_') or k in
                               ('nn_weight_dram_bytes', 'inference_weight_read_bytes', 'weight_bytes', 'line_bytes')})
        line_bytes = c.get('line_bytes')
        for direction in ('read', 'write'):
            traffic['dram_' + direction + '_queue_accepted_payload_bytes'] = (
                c['dram_' + direction + '_enqueued'] * line_bytes
                if finite(c.get('dram_' + direction + '_enqueued')) and finite(line_bytes) else None)
        traffic['boundary_note'] = DEFINITIONS['dram_completed_bytes']
        tables['dram_traffic'].append(traffic)
        storage=dict(ident, **{k: v for k, v in c.items() if
            ('bytes' in k and not k.startswith('dram_') and k not in ('inference_weight_read_bytes', 'inference_scratch_traffic_bytes', 'nn_weight_dram_bytes')) or 'capacity' in k or k in ('h', 'dense_weight_copies', 'nonlinear_units',
              'macs_per_cycle', 'service_architecture', 'queue_policy', 'page_policy',
              'configured_state_unbounded', 'peak_state_entries')})
        has_nn=bool(c.get('h',0))
        storage['corrected_nn_state_bytes_configured']=c.get('state_bytes_configured') if has_nn else 0
        storage['corrected_nn_state_bytes_peak']=c.get('state_bytes_peak') if has_nn else 0
        storage['corrected_nn_core_arrays_events_scratch_bytes']=c.get('deployment_bytes_configured') if has_nn else 0
        storage['raw_deployment_counter_scope']='Core arrays/event/scratch reservations only; full source-field controller schema is separate.'
        storage['raw_counter_erratum']=(
            'h=0 has no NN forward/state allocation; raw state_bytes_configured includes capacity*32 metadata anyway. Corrected NN storage is zero; raw value retained.'
            if not has_nn and c.get('state_bytes_configured',0) else None)
        storage.update(source_storage_fields(run,storage_evidence))
        if run['spec'].get('arm')=='spp':
            storage['strong_baseline_storage_status']=('logical_source_field_audit_available_functional_timing_only'
                if storage['source_field_budget_status']=='audited_fixed_design_schema' else 'source_field_audit_unavailable')
        tables['storage'].append(storage)
        for name, buckets in c.get('distributions_log2_upper_bound', {}).items():
            total = sum(buckets.values()); cumulative = 0
            for upper, count in sorted(buckets.items(), key=lambda item: int(item[0])):
                upper = int(upper); cumulative += count
                tables['service_distributions'].append(dict(ident, distribution=name, lower_bound=0 if upper==0 else (upper+1)//2,
                    upper_bound=upper, count=count, total_events=total, cumulative_fraction=ratio(cumulative, total)))
        per_pc = c.get('per_pc_roi_updates', {})
        if not per_pc:
            completed, evictions = c.get('pc_completed_updates', {}), c.get('pc_state_evictions', {})
            per_pc = {pc: {'completed': completed.get(pc, 0), 'evictions': evictions.get(pc, 0)}
                      for pc in set(completed) | set(evictions)}
        for pc, values in sorted(per_pc.items(), key=lambda item: int(item[0])):
            tables['history_per_pc'].append(dict(ident, pc=pc, **values))
        for pc, values in sorted(c.get('resident_pc_lifetime_updates', {}).items(), key=lambda item: int(item[0])):
            tables['resident_state_lifetimes'].append(dict(ident, pc=pc, **values))
    return tables


def main_rows(runs, default_history):
    return [r for r in runs if r['spec'].get('silent', 0) == 0
            and r['spec'].get('queue', 'fifo') == 'fifo'
            and r['spec'].get('history', 0) == min(default_history, r['spec'].get('warmup', default_history))]


def plot_results(runs, directory, default_history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi': 160, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    created = []
    def save(fig, name):
        fig.savefig(directory/name)
        vector_name=str(Path(name).with_suffix('.svg'))
        fig.savefig(directory/vector_name)
        return [name, vector_name]
    display={'none':'NoPF','spp':'SPP-dev2','stride_legacy':'Legacy Stride',
             'stride_remove_current':'Remove current only','stride_forward1':'Forward Stride d1',
             'stride_forward2':'Forward Stride d2'}
    standard = main_rows(runs, default_history)
    panels = defaultdict(list)
    for r in standard:
        s = r['spec']; panels[(s.get('stage'), s.get('workload'), s.get('roi'), s.get('budget', 0))].append(r)
    for (stage, workload, roi, budget), rows in sorted(panels.items()):
        stage_tag='' if stage=='test' else str(stage)+'_'
        rows = [r for r in rows if finite(r['metrics'].get('speedup_vs_nopf'))]
        if not rows: continue
        categories = defaultdict(list)
        for r in rows:
            s = r['spec']; label = display.get(s['arm'],s['arm']) if s['arm'] != 'nn' else 'NN {} {}'.format(s['loss'], 'zero cost' if not s['macs'] else str(s['macs'])+' MAC/c')
            categories[label].append(r)
        labels = sorted(categories, key=lambda v: (v!='NoPF', v.startswith('NN'), v))
        fig, ax = plt.subplots(figsize=(max(7, len(labels)*.85), 4.3))
        seed_ids=sorted({r['spec']['seed'] for r in rows if r['spec']['arm']=='nn'})
        seed_colors={seed:plt.get_cmap('tab10')(i) for i,seed in enumerate(seed_ids)}
        shown=set()
        for x, label in enumerate(labels):
            actual=sorted(categories[label],key=lambda r:r['spec'].get('seed',0))
            values=[r['metrics']['speedup_vs_nopf'] for r in actual]
            for i,r in enumerate(actual):
                is_nn=r['spec']['arm']=='nn'
                legend='training seed '+str(r['spec']['seed']) if is_nn else 'fixed conventional reference'
                ax.scatter(x+(i-(len(values)-1)/2)*.075,values[i],s=25,
                           color=seed_colors.get(r['spec']['seed'],'#4a5964') if is_nn else '#4a5964',
                           marker='o' if is_nn else 'D',label=legend if legend not in shown else None)
                shown.add(legend)
            if len(values)>1:
                ax.errorbar(x,statistics.mean(values),yerr=statistics.stdev(values),color='#132a3a',capsize=3,fmt='_')
        ax.legend(fontsize=7,loc='best')
        ax.axhline(1, color='#888888', lw=.9)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=28, ha='right')
        ax.set_ylabel('IPC / matched NoPF IPC')
        ax.set_title('{} · {} ROI {} · {}\nDots = actual runs; bars = mean ± sample SD when multiple seeds exist'.format(
            workload, stage, roi, 'original policy' if not budget else 'common cap {} requests/M instructions'.format(budget)))
        fig.tight_layout(); name = 'ipc_{}{}_roi{}_budget{}.png'.format(stage_tag,workload.split('.')[0], roi, budget)
        created.extend(save(fig,name)); plt.close(fig)
    groups = defaultdict(list)
    for r in standard: groups[(r['spec'].get('stage'),r['spec'].get('workload'), r['spec'].get('roi'))].append(r)
    colors = {'none': '#888888', 'spp': '#8b54a1', 'nn': '#c45535',
              'stride_legacy':'#244e74','stride_remove_current':'#689db0',
              'stride_forward1':'#1f937e','stride_forward2':'#83a83c'}
    for (stage, workload, roi), rows in sorted(groups.items()):
        stage_tag='' if stage=='test' else str(stage)+'_'
        rows=[r for r in rows if finite(r['metrics'].get('gate_requests_per_million_instructions'))
              and finite(r['metrics'].get('speedup_vs_nopf'))]
        if not rows:continue
        fig, ax = plt.subplots(figsize=(7, 4.4)); labels_seen = set()
        for r in rows:
            x, y = r['metrics'].get('gate_requests_per_million_instructions'), r['metrics'].get('speedup_vs_nopf')
            if not finite(x) or not finite(y): continue
            s = r['spec']; label = display.get(s['arm'],s['arm']) if s['arm'] != 'nn' else 'NN {} {}'.format('zero cost' if not s['macs'] else str(s['macs'])+' MAC/c',s['loss'])
            color=colors.get(s['arm'], '#245d8f')
            if s['arm']=='nn':
                color={('balanced',0):'#df7f22',('unweighted',0):'#bd4c9c',
                       ('balanced',4):'#b6383a',('unweighted',4):'#9e6c3b'}.get((s['loss'],s['macs']),color)
            ax.scatter(x, y, s=30, marker='o' if not s.get('budget') else 's', alpha=.75,
                       color=color, label=label if label not in labels_seen else None)
            labels_seen.add(label)
        ax.axhline(1, color='#888888', lw=.8); ax.set_xlabel('Actual allowed requests / million retired instructions')
        ax.set_ylabel('IPC / matched NoPF IPC'); ax.set_title('{} · {} ROI {}\nCircle = original; square = validation-selected common cap'.format(workload, stage, roi))
        ax.legend(fontsize=7, loc='best'); fig.tight_layout()
        name='request_ipc_{}{}_roi{}.png'.format(stage_tag,workload.split('.')[0], roi)
        created.extend(save(fig,name)); plt.close(fig)
    hist_groups = defaultdict(list)
    for r in standard:
        if r['spec'].get('arm') == 'nn' and r['spec'].get('macs', 0) > 0:
            s=r['spec']; hist_groups[(s['stage'],s['workload'], s['roi'], s['budget'], s['macs'])].append(r)
    for (stage, workload, roi, budget, macs), rows in sorted(hist_groups.items()):
        stage_tag='' if stage=='test' else str(stage)+'_'
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
        for ax, distribution in zip(axes, ('completed_service_cycles', 'queue_wait_cycles', 'completion_age_cycles')):
            seed_ids=sorted({r['spec']['seed'] for r in rows})
            seed_colors={seed:plt.get_cmap('tab10')(i) for i,seed in enumerate(seed_ids)}
            for r in sorted(rows,key=lambda v:(v['spec']['seed'],v['spec']['loss'])):
                buckets={int(u):n for u,n in r['summary'].get('distributions_log2_upper_bound',{}).get(distribution,{}).items()}
                total=sum(buckets.values());count=0;xs=[0];ys=[0]
                for upper,n in sorted(buckets.items()):
                    count+=n;xs.append(upper);ys.append(count/total)
                if total:
                    xs.append(xs[-1]*1.05+1);ys.append(1.0)
                    ax.step(xs,ys,where='post',marker='o',markersize=2.5,label='{} s{}'.format(r['spec']['loss'],r['spec']['seed']),
                            color=seed_colors[r['spec']['seed']],linestyle='-' if r['spec']['loss']=='balanced' else '--')
            ax.set_xscale('symlog',linthresh=1);ax.set_ylim(0,1.02);ax.set_xlabel('Cycles, bin upper bound')
            ax.set_title(distribution.replace('_',' '));ax.set_ylabel('Binned cumulative fraction');ax.legend(fontsize=7)
        fig.suptitle('{} {} ROI{} m{} cap{} · separate seed/loss curves, no pooling'.format(workload,stage,roi,macs,budget),fontsize=9)
        fig.tight_layout();name='service_{}{}_roi{}_budget{}_m{}.png'.format(stage_tag,workload.split('.')[0],roi,budget,macs)
        created.extend(save(fig,name));plt.close(fig)
    history_groups=defaultdict(list)
    for r in runs:
        s=r['spec']
        if s.get('arm')=='nn' and s.get('macs',0)>0 and not s.get('budget') and not s.get('silent') and s.get('queue')=='fifo' and s.get('loss')=='balanced':
            history_groups[(s['stage'],s['workload'],s['roi'],s['macs'])].append(r)
    for (stage,workload,roi,macs),rows in sorted(history_groups.items()):
        stage_tag='' if stage=='test' else str(stage)+'_'
        histories=sorted({r['spec']['history'] for r in rows})
        if len(histories)<2:continue
        fig,axes=plt.subplots(1,2,figsize=(9,3.8))
        for seed in sorted({r['spec']['seed'] for r in rows}):
            selected=sorted([r for r in rows if r['spec']['seed']==seed],key=lambda r:r['spec']['history'])
            xs=[histories.index(r['spec']['history']) for r in selected]
            axes[0].plot(xs,[r['metrics']['speedup_vs_nopf'] for r in selected],'-o',ms=3,label='seed '+str(seed))
            axes[1].plot(xs,[r['summary'].get('nn_completed') for r in selected],'-o',ms=3,label='seed '+str(seed))
        for ax in axes:
            ax.set_xticks(range(len(histories)));ax.set_xticklabels([str(h) for h in histories]);ax.set_xlabel('Operational predictor warmup, instructions');ax.legend(fontsize=7)
        axes[0].set_ylabel('IPC / matched NoPF IPC');axes[0].ticklabel_format(axis='y',style='plain',useOffset=False);axes[1].set_ylabel('Actual ROI completed NN updates')
        fig.suptitle('{} {} ROI{} · operational predictor history\nh/c + tag/LRU/occupancy may differ; isolated h/c-only supplement pending'.format(workload,stage,roi),fontsize=9)
        fig.tight_layout();name='history_{}{}_roi{}_m{}.png'.format(stage_tag,workload.split('.')[0],roi,macs)
        created.extend(save(fig,name));plt.close(fig)
    return created


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=None)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--protocol',default=None)
    parser.add_argument('--stages',nargs='+',choices=['test','validation'],default=['test'])
    parser.add_argument('--expected-matrix',type=Path,help='Optional JSON list of planned spec objects; missing runs stay explicit.')
    parser.add_argument('--no-plots',action='store_true')
    parser.add_argument('--storage-audit',type=Path,help='Full source-field inventory JSON; auto-discovered in research_audit/engine when available.')
    args=parser.parse_args()
    config=read_json(EXP/'formal_config.json')
    protocol=args.protocol or config['protocol']
    storage_evidence=load_storage_audit(args.storage_audit)
    args.run_dir=args.run_dir or Path(config.get('raw',str(DEFAULT_RAW)))
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise RuntimeError('Output directory already contains files; choose a fresh directory to retain prior analyses.')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    runs=[]
    for stage in args.stages:
        runs.extend(inspect(p,protocol) for p in sorted((args.run_dir/stage).glob('*/run.json')))
    if args.expected_matrix:
        expected=read_json(args.expected_matrix)
        if isinstance(expected,dict):expected=expected.get('specs',expected.get('runs',[]))
        seen={r['spec'].get('run_id') for r in runs}
        for spec in expected:
            if spec.get('stage') in args.stages and spec.get('run_id') not in seen:
                runs.append(dict(spec=spec,status='not_submitted_or_missing',errors=[],directory=None,summary={}))
    budget_audit=audit_budget(args.run_dir,protocol)
    for run in runs:
        if run.get('validated') and run['spec'].get('budget',0)>0:
            evidence=budget_audit.get('workloads',{}).get(run['spec'].get('workload'),{})
            if (budget_audit['status']!='valid' or evidence.get('status')!='valid'
                    or evidence.get('configured_cap')!=run['spec']['budget']):
                run['validated']=False
                run['status']='invalid_budget_selection'
                run['errors'].append('Configured cap lacks matching validation-only selection evidence.')
    complete=[r for r in runs if r.get('validated')]
    for run in complete:
        candidates=[r for r in complete if r['spec'].get('arm')=='none' and key(r['spec'])==key(run['spec'])]
        reference=candidates[0] if len(candidates)==1 else None
        if reference is None:run['errors'].append('No unique completed matched NoPF reference; baseline-derived metrics unavailable.')
        run['metrics']=derive(run,reference)
    raw_rows=[];metric_rows=[];inventory=[]
    for run in runs:
        ident=identity(run)
        inventory.append(dict(ident,status=run['status'],directory=run['directory'],errors=run['errors']))
        if not run.get('summary'):continue
        raw_rows.append(dict(ident,status=run['status'],directory=run['directory'],spec=run['spec'],
                             **flatten(run['summary'],'raw__')))
        if not run.get('validated'):continue
        metric_rows.append(dict(ident,**run['metrics'],warnings=run['errors']))
    write_csv(args.out_dir/'inventory.csv',inventory)
    write_csv(args.out_dir/'raw_per_seed.csv',raw_rows)
    write_csv(args.out_dir/'metrics_per_seed.csv',metric_rows)
    comparisons=paired(complete,config['state_history_instructions'])
    write_csv(args.out_dir/'paired_comparisons.csv',comparisons)
    gaps=request_gaps(complete)
    write_csv(args.out_dir/'realized_request_gap.csv',gaps)
    for name,rows in ancillary(complete,storage_evidence).items():write_csv(args.out_dir/(name+'.csv'),rows)
    write_json(args.out_dir/'metric_definitions.json',DEFINITIONS)
    write_json(args.out_dir/'budget_audit.json',budget_audit)
    write_json(args.out_dir/'storage_source_field_audit.json',storage_evidence)
    plot_status={'status':'not_requested','files':[]}
    if not args.no_plots:
        try:plot_status={'status':'complete','files':plot_results(complete,args.out_dir,config['state_history_instructions'])}
        except ImportError as error:plot_status={'status':'unavailable_dependency','error':str(error),'files':[]}
    write_json(args.out_dir/'plot_status.json',plot_status)
    status_counts=dict(Counter(r['status'] for r in runs))
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),protocol=protocol,
        raw_root=str(args.run_dir.resolve()),stages=args.stages,complete_results=len(complete),
        status_counts=status_counts,planned_matrix_available=bool(args.expected_matrix),
        inventory_scope='Expected matrix plus discovered runs' if args.expected_matrix else 'Discovered run.json files only; unscheduled matrix rows are not inferred.',
        plots=plot_status,budget_audit=budget_audit,storage_audit=storage_evidence,definitions=DEFINITIONS,runs=runs)
    write_json(args.out_dir/'results.json',manifest)
    notes=['# Formal results analysis','',
           'Protocol: '+protocol+'. All tables use only this formal protocol and matching build/ROI controls.',
           '', 'Completed and validated results: '+str(len(complete))+'. Status counts: '+json.dumps(status_counts,sort_keys=True)+'.',
           '',manifest['inventory_scope'], '',
           'Headline metrics use LOAD-only cohort outcomes. Legacy U/L metrics remain separately labeled. Pending and resident-unused requests remain censored. Unused is not pollution.', '',
           'The common budget is a shared cap selected on validation. See realized_request_gap.csv before claiming matched realized traffic.', '',
           'Teacher request categories are descriptive. Use paired_comparisons.csv silent-output interventions for simulation deltas; do not infer per-request causality from tags.', '',
           'DRAM tables separate queue acceptance, bus start and completion, boundary pending work, and NN resident weights/internal SRAM reads.', '',
           'All seed rows remain separate in raw/metric/pair/history/distribution tables. Mean/SD overlays never replace individual observations.', '',
           'Storage: audited fixed-design source-field budgets are NN16920B, Stride2136B and nativeSPP5938B; common-cap control adds16B (16936/2152/5954B). Applied only when audit protocol/build/resources match. The original NN16808B counter is retained and relabeled core arrays/events/scratch, not full hardware storage.', '',
           'Host object sizeof, whole-process peak RSS and conservative logical source-field reservations are separate. None is an RTL/SRAM-macro area result. NativeSPP finite compute cost is unmodeled. Candidate-array guard evidence is limited to its listed formal runs, not a universal source-code proof. Full inventory: storage_source_field_audit.json; per-run values: storage.csv.', '',
           'NN dense weight-read bytes are analytical FP32 multiply-operand reads, exclude bias loads and are not measured SRAM/DRAM traffic. Weights are modeled resident; physical ports/energy are not established.', '',
           'Main v2 history arms are operational predictor-history comparisons: h/c plus tag/LRU/occupancy can differ at the ROI boundary. They do not isolate h/c history. Cache warmup and future ROI match. A separate immutable-build fixed-metadata h/c-only full formal supplement is pending; do not claim its result from these rows. State updates/evictions are actual observations, not a minimum-history threshold.', '',
           'Plots: '+plot_status['status']+'. No cross-protocol historical IPC comparison or interpolated missing result is included.']
    missing_references=[r for r in complete if not r['metrics'].get('nopf_reference')]
    notes += ['', 'NoPF denominator availability: '+str(len(missing_references))+' analyzed rows lack a matched NoPF: '
              +json.dumps([{'run_id':r['spec']['run_id'],'stage':r['spec']['stage'],'workload':r['spec']['workload']} for r in missing_references])
              +'. Their NoPF-derived metrics are null; raw IPC/cycles and direct counters remain valid. This is not a raw-result validation failure.']
    if storage_evidence.get('status')=='available':
        limits=storage_evidence['audit']['spp_native']['lookahead_limits']
        certificate=limits.get('guard_certificate',{})
        notes += ['', 'SPP scoped array-write guard certificate: '+str(certificate.get('status','unavailable'))
                  +'; completed '+str(certificate.get('completed_repetitions',0))+'/'+str(certificate.get('expected_repetitions',8))
                  +'; guard trips '+str(certificate.get('guard_trip_count','unavailable'))
                  +'; exact original-summary comparisons '+str(certificate.get('total_exact_field_comparisons','unavailable'))
                  +'. This covers only the listed complete formal cases, not a general unbounded static proof.']
    (args.out_dir/'analysis_notes.md').write_text('\n'.join(notes)+'\n')
    print(json.dumps(dict(output=str(args.out_dir.resolve()),complete_results=len(complete),
                          status_counts=status_counts,plots=plot_status['status']),sort_keys=True))


if __name__=='__main__':main()
