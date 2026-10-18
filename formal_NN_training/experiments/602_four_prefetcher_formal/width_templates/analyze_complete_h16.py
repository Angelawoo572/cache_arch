#!/usr/bin/env python3
"""Audit the complete H16 extension; read original evidence, publish only new views.

No training, simulator, frozen driver main, or historical analyzer main is run.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter, defaultdict
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
CACHE = next(p for p in HERE.parents if p.name == 'cache')
CANONICAL = CACHE / 'formal_study/results/602_width_templates'
PROTOCOL = 'four602_all_h16_20261017_v1'
REFERENCE = 'four602_frozen_live_20261017_v2'
SYMBOLS = importlib.util.spec_from_file_location('four602_read_only_helpers', HERE.parent/'analyze.py')
OLD = importlib.util.module_from_spec(SYMBOLS)
SYMBOLS.loader.exec_module(OLD)
sys.path.insert(0, str(CACHE))


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def dump(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, columns, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v, sort_keys=True) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def path(value):
    text = str(value)
    for prefix in ('/home/qianruw/cache/', '/Users/angelawoo/Documents/CMU/ece/cache/'):
        if text.startswith(prefix):
            text = text[len(prefix):]
            for day in ('20261016','20261017'):
                text = text.replace('formal_study_'+day+'/', 'formal_study/archive/'+day+'/')
            return CACHE/text
    return Path(text)


def inside(value, root=CACHE):
    value, root = Path(value).resolve(), Path(root).resolve()
    return value == root or root in value.parents


def expected(config):
    rows = []
    def one(stage, roi, policy, method='student', mode='full', cap=0):
        lo,hi = config['validation_records'] if stage=='validation' else config['test_records'][roi]
        rows.append(dict(run_id='{}_r{}_{}_h16_{}_{}_cap{}'.format(stage,roi,policy,method,mode,cap),
                         protocol=PROTOCOL,stage=stage,roi=roi,policy=policy,hidden_size=16,method=method,mode=mode,
                         budget=cap,seed=7,macs=0,page_policy=config['page_policy'],skip=lo-config['warmup_instructions'],
                         warmup=config['warmup_instructions'],instructions=hi-lo,measurement_records=[lo,hi]))
    for policy in ('stride','streamer'):
        one('validation',0,policy,'stride_student' if policy=='stride' else 'student')
        for roi in (0,1):
            for cap in (0,config['common_cap']):
                one('test',roi,policy,'stride_student' if policy=='stride' else 'student',cap=cap)
                one('test',roi,policy,'template',cap=cap)
    for policy in ('streamer','ampm','spp'):
        for roi in (0,1):
            for mode in ('fixed_payload','reset_state'):
                one('test',roi,policy,mode=mode)
    assert len(rows)==30
    return {r['run_id']:r for r in rows}


def verify_bindings(raw, config, audit):
    state=read(raw/'bindings.json');b=read(path(config['reference_raw'])/'build.json')
    audit.check(state['protocol']==PROTOCOL and state['status']=='PASS' and state['build_id']==b['build_id'], 'frozen_extension_build_identity')
    audit.check(b['protocol']==REFERENCE and b['base']==config['base'] and b['compiler_flags']=='-O3 -ffp-contract=off', 'compatible_actual_formal_compiler_machine')
    audit.check(path(b['binary']).read_bytes()==path(b['binary_snapshot']).read_bytes(), 'actual_executable_preserved')
    files=[p for p in path(b['compiled_source']).rglob('*') if p.is_file()]
    changed=[str(p.relative_to(path(b['compiled_source']))) for p in files if p.read_bytes()!=(path(b['simulator'])/p.relative_to(path(b['compiled_source']))).read_bytes()]
    audit.check(bool(files) and not changed, 'actual_compiled_source_bytes_preserved', checked_files=len(files),different=changed)
    audit.check(bool(state['bindings']), 'frozen_source_model_parity_bindings_present')
    for item in state['bindings']:
        source,snapshot=path(item['source']),path(item['snapshot'])
        audit.check(inside(source) and inside(snapshot,raw) and source.is_file() and snapshot.is_file()
                    and source.stat().st_size==item['bytes'] and source.read_bytes()==snapshot.read_bytes(),
                    'frozen_source_model_parity_exact_bytes',source=str(source))
    audit.check(set(state['certificates'])=={'stride','streamer'}, 'both_new_H16_correctness_certificates_bound')
    for policy,certificate in state['certificates'].items():
        report=read(path(certificate))
        audit.check(report.get('status') in ('PASS','verified'), 'real_checkpoint_actual_CPP_parity_PASS', policy=policy)
    return state,b


def baseline(raw, stage, roi, config, build, audit):
    rid='{}_r{}_none_none_full_cap0'.format(stage,roi);folder=raw/'runs'/rid
    record=read(folder/'run.json');spec=record['spec'];snap=read(folder/'summary.json')
    lo,hi=config['validation_records'] if stage=='validation' else config['test_records'][roi]
    audit.check(record['status']=='complete' and record['exit_code']==0 and spec['protocol']==REFERENCE
                and spec['method']=='none' and spec['policy']=='none' and spec['build_id']==build['build_id']
                and spec['measurement_records']==[lo,hi] and spec['skip']==lo-config['warmup_instructions']
                and spec['warmup']==config['warmup_instructions'] and spec['instructions']==hi-lo
                and spec['page_policy']==config['page_policy'] and spec['macs']==0 and spec['seed']==7,
                'exact_matching_original_NoPF_reference',rid)
    audit.check(snap['final'] and spec['instructions']<=snap['instructions']<=spec['instructions']+4 and snap['cycles']>0,
                'matching_NoPF_complete_ROI',rid)
    return dict(spec=spec,summary=snap,dir=folder,record=record)


def load_run(folder, wanted, config, build, audit):
    record=read(folder/'run.json');spec=record['spec'];rid=spec['run_id']
    audit.check(folder.name==rid and all(spec.get(k)==v for k,v in wanted.items()), 'exact_planned_H16_condition',rid)
    audit.check(spec.get('contract')==config and spec.get('trace')==config['trace'] and spec['build_id']==build['build_id'], 'exact_frozen_contract_and_actual_build',rid)
    if record['status']=='reused':
        source=path(record['reuse']['source_run']);source_record=read(source/'run.json');old=source_record['spec']
        same=('stage','roi','policy','method','mode','budget','skip','warmup','instructions','measurement_records','page_policy')
        audit.check(spec['method']=='template' and source_record['status']=='complete' and source_record['exit_code']==0
                    and old['build_id']==spec['build_id'] and all(old.get(k)==spec.get(k) for k in same)
                    and old.get('frozen_template')==spec.get('frozen_template'), 'identical_NN_free_control_reused_not_new_run',rid)
        audit.check(inside(source,path(config['reference_raw'])) or inside(source,path(config['width_raw'])), 'reuse_source_in_retained_protocol_roots',rid)
        return dict(spec=spec,summary=read(source/'summary.json'),dir=source,record=source_record,
                    logical_record=record,source_spec=old,reused=True)
    audit.check(record['status']=='complete' and record.get('exit_code')==0, 'successful_complete_actual_simulator',rid)
    return dict(spec=spec,summary=read(folder/'summary.json'),dir=folder,record=record,source_spec=spec,reused=False)


def verify_run(run, config, audit):
    s,x,record=run['spec'],run['summary'],run['record'];rid=s['run_id'];c=OLD.counters(x)
    audit.check(x.get('final') is True and s['instructions']<=x['instructions']<=s['instructions']+4 and x['cycles']>0, 'complete_full_ROI_not_prefix',rid)
    audit.check(x.get('numerical_failures',0)==0 and x.get('measurement_unconsumed_context_labels',0)==0, 'no_numerical_or_request_context_failures',rid)
    audit.check(x.get('measurement_cohort_scope')=='separate_L2_and_LLC_target_cohorts', 'separate_tracked_target_cohorts',rid)
    env=record['environment'];stride=s['method']=='stride_student'
    actual_method='monitor' if stride else s['method']
    audit.check(x.get('four602_method')==actual_method and x.get('four602_policy')==s['policy'] and x.get('four602_mode')==s['mode'], 'actual_registered_policy_method_mode',rid)
    wanted_env=dict(FOUR602_POLICY=s['policy'],FOUR602_METHOD=actual_method,FOUR602_MODE=s['mode'],DYNAMIC602_MACS='0',
                    DYNAMIC602_PAGE_POLICY=s['page_policy'],DYNAMIC602_SKIP_RECORDS=str(s['skip']),
                    DYNAMIC602_WARMUP_INSTRUCTIONS=str(s['warmup']),DYNAMIC602_HISTORY_INSTRUCTIONS=str(s['warmup']),
                    DYNAMIC602_ROI_INSTRUCTIONS=str(s['instructions']),DYNAMIC602_PHASE_INSTRUCTIONS=str(config['phase_instructions']),
                    DYNAMIC602_REQUESTS_PER_MILLION_INSTRUCTIONS=str(s['budget']),DYNAMIC602_FILTER_CURRENT='1' if s['policy']=='stride' else '0')
    audit.check(all(env.get(k)==v for k,v in wanted_env.items()), 'actual_environment_ROI_history_filter_credit_MAC0_contract',rid)
    command=record['command'];registry=[v for v in command if v.startswith('--l2c_prefetcher_types=')]
    audit.check(registry==(['--l2c_prefetcher_types=four602','--l2c_prefetcher_types=dynamic602'] if stride else ['--l2c_prefetcher_types=four602']), 'actual_prefetcher_registry_not_NoPF',rid)
    knobs=dict(streamer_num_trackers='64',streamer_pref_degree='5',ampm_pb_size='64',ampm_pred_degree='4',ampm_pref_degree='4',
               ampm_enable_pref_buffer='false',ampm_pref_buffer_size='256',ampm_max_delta='16',stride_num_trackers='64',stride_pref_degree='2',
               spp_dev2_fill_threshold='90',spp_dev2_pf_threshold='40',warmup_instructions=str(s['warmup']),simulation_instructions=str(s['instructions']))
    actual={item[2:].split('=',1)[0]:item.split('=',1)[1] for item in command if item.startswith('--') and '=' in item and not item.startswith('--l2c_prefetcher_types=')}
    audit.check(actual==knobs and command[-2:]==['-traces',config['trace']], 'actual_native_knobs_trace_and_full_ROI_CLI',rid)
    expected_binary=path(read(path(config['reference_raw'])/'build.json')['binary'])
    audit.check(path(command[0]).resolve()==expected_binary.resolve() and 'ChampSim completed all CPUs' in (run['dir']/'run.log').read_text(), 'actual_unchanged_binary_and_completed_log',rid)
    for target in ('l2','llc'):OLD.check_cohort(audit,OLD.cohort(x,target),target,rid)
    if x.get('llc_only_lifecycle_accuracy_available') is True:
        audit.check(x.get('llc_target_prefetch_queue_rejected_untracked')==0 and x.get('llc_target_shortcut_missing_pending_id')==0, 'LLC_quality_requires_full_handoff_tracking',rid)
    audit.check(x['request_gate_generated']==sum(x['request_gate_'+key]for key in ('allowed','budget_suppressed','page_suppressed','current_suppressed')), 'global_request_gate_conservation',rid)
    audit.check(x['request_gate_allowed']==sum(x['request_cache_'+key]for key in ('enqueued','merged','rejected')), 'cache_accept_merge_reject_conservation',rid)
    audit.check(x['request_cache_pq_inserted']==x['cohort_enqueued']+x['llc_target_cohort']['enqueued'], 'target_enqueue_partition',rid)
    for suffix in ('generated','allowed','cache_merged','cache_rejected'):
        total='request_gate_'+suffix if suffix in ('generated','allowed') else 'request_'+suffix
        audit.check(x['request_l2_target_'+suffix]+x['request_llc_only_'+suffix]==x[total], 'L2_LLC_actual_request_partition',rid,field=suffix)
    if s['budget']:audit.check(x['request_gate_allowed']<=x['instructions']*s['budget']//1000000, 'cumulative_credit_ceiling_respected',rid)
    audit.check(x['dram_total_bus_bytes']==x['dram_read_bus_bytes']+x['dram_write_bus_bytes']
                and x['dram_read_bus_bytes']==x['line_bytes']*x['dram_read_bus_completed_transactions']
                and x['dram_write_bus_bytes']==x['line_bytes']*x['dram_write_bus_completed_transactions'], 'actual_DRAM_read_write_transaction_bytes',rid)
    audit.check(x['dram_read_bus_started_bytes']+x['dram_write_bus_started_bytes']-x['dram_total_bus_bytes']==x['line_bytes']*x['dram_bus_transfers_in_progress'], 'DRAM_started_completed_pending_boundary_conservation',rid)
    if s['method'] in ('student','stride_student'):
        model=run['dir']/'model.bin';frozen=path(s['model'])
        audit.check(model.is_file() and model.read_bytes()==frozen.read_bytes() and frozen.read_bytes()==(path(config['raw'])/'models'/s['policy']/'model.bin').read_bytes(), 'per_run_actual_H16_weights_exact_frozen_bytes',rid)
        if stride:
            metadata=read(path(config['models']['stride']).with_name('model_metadata.json'))
            audit.check(x.get('h')==16 and x.get('weight_bytes')==20880 and metadata['hidden_size']==16 and metadata['parameter_count']==5220, 'actual_Stride_H16_model_width_parameters',rid)
            audit.check(x.get('nn_completed')==x.get('eligible_l2_callbacks')==c.get('demand_callbacks') and x.get('inference_dropped')==0, 'Stride_MAC0_every_eligible_LOAD_completes',rid)
            audit.check(x.get('state_capacity')==64 and x.get('state_entry_bytes')==160, 'Stride_retains_64_PC_state_ABI',rid)
        else:
            from formal_NN_training.common.live602_runtime.model_format import read_model
            header,tensors=read_model(model);wanted=6609 if s['policy']=='spp' else 6653
            audit.check(header['hidden_size']==16 and header['parameter_count']==wanted and sum(len(t['data']) for t in tensors.values())==wanted, 'actual_global_H16_header_tensor_parameters',rid)
            schema=(65,1,2,1) if s['policy']=='spp' else (64,0,0,0)
            audit.check(tuple(header[k]for k in ('feature_count','kind_feature','fill_classes','count_mode'))==schema, 'original_input_count_fill_schema',rid)
            audit.check(x.get('four602_global_state_entries')==1 and x.get('four602_state_bytes')==128 and x.get('four602_state_evictions')==0, 'original_global_H16_state_organization',rid)
            audit.check(c.get('completed_updates')==c.get('demand_callbacks',0)+c.get('fill_callbacks',0)
                        and c.get('demand_inferences')==c.get('demand_callbacks') and c.get('fill_inferences',0)==c.get('fill_callbacks',0), 'complete_forward_per_actual_callback',rid)
            if s['policy']=='spp':audit.check(c.get('fill_callbacks',0)>0, 'SPP_actual_all_raw_fill_updates',rid)
            if s['mode']=='full':
                ds=x['four602_distributions'];phases={OLD.PROFILE.fullmatch(k).group(1)for k in ds if OLD.PROFILE.fullmatch(k)}
                audit.check(bool(phases) and all(ds.get('phase_'+p+'_learned_K')==ds.get('phase_'+p+'_raw_K') and ds.get('phase_'+p+'_learned_emit')==ds.get('phase_'+p+'_raw_emit')for p in phases), 'complete_learned_K_emit_no_decoder_truncation',rid)
    else:
        audit.check(s['method']=='template' and c.get('completed_updates',0)==0 and c.get('inference_macs',0)==0 and c.get('inference_nonlinears',0)==0, 'NN_free_template_has_no_NN_work',rid)


def selected_template(run):
    s,x=run['spec'],run['summary'];counts=Counter();slots=defaultdict(Counter)
    distributions=x['mechanism_distributions_exact'] if s['policy']=='stride' else x['four602_distributions']
    for name,hist in distributions.items():
        match=OLD.PROFILE.fullmatch(name)
        if not match:raise ValueError('unknown validation histogram '+name)
        _,stream,stat=match.groups()
        if stream!='raw':continue
        if stat=='K':counts.update({int(k):v for k,v in hist.items()})
        if s['policy']=='stride' and stat.startswith('delta_slot_'):
            slots[int(stat[len('delta_slot_'):])].update({(int(k),2):v for k,v in hist.items()})
        elif s['policy']!='stride' and stat.startswith('delta_fill_'):
            fill,slot=map(int,stat[len('delta_fill_'):].split('_slot_'));slots[slot].update({(int(k),fill):v for k,v in hist.items()})
    if not counts:raise ValueError('missing whole-validation K distribution')
    modal=min(counts,key=lambda k:(-counts[k],k))
    if modal and not slots:raise ValueError('positive validation K missing slots')
    payload=[list(min(slots[i],key=lambda k:(-slots[i][k],k)))for i in range(max(slots)+1)] if slots else [[0,2]]
    deployed=min(modal,32) if s['policy']=='stride' else modal
    return dict(modal_count=modal,deployed_count=deployed,slots=payload,source_run=s['run_id'],
                text='modal_count {}\n'.format(deployed)+''.join('{} {}\n'.format(*v)for v in payload))


def verify_selection(raw, runs, config, audit):
    selection=read(raw/'selection.json');old=read(path(config['reference_raw'])/'selection.json')
    audit.check(selection['protocol']==PROTOCOL and selection['common_cap']==config['common_cap']==old['common_cap']==16144
                and selection['validation_records']==[20000000,25000000], 'validation_frozen_common_cap_not_test_selected')
    for policy in ('stride','streamer'):
        rid='validation_r0_{}_h16_{}_full_cap0'.format(policy,'stride_student' if policy=='stride' else 'student')
        actual=selected_template(runs[rid]);stored=selection['templates'][policy]
        audit.check(all(stored.get(k)==actual[k]for k in ('modal_count','deployed_count','slots','source_run'))
                    and (raw/'templates'/(policy+'_h16.txt')).read_text()==actual['text'], 'H16_template_recomputed_from_entire_own_validation',rid)
    for run in runs.values():
        s=run['spec']
        if s['method']=='template' or s['mode']=='fixed_payload':
            frozen=(raw/'templates'/(s['policy']+'_h16.txt')).read_text()
            audit.check(s['frozen_template']==frozen, 'actual_run_uses_frozen_validation_template_bytes',s['run_id'])
    return selection


def annotate(row, run, raw):
    s=run['spec'];original=run['source_spec'];reused=run['reused']
    row.update(result_id=original['protocol']+'::'+original['run_id'],run_id=original['run_id'],protocol=original['protocol'],
               logical_run_id=s['run_id'],logical_protocol=PROTOCOL,evidence_origin='compatible_NN_free_control_reused' if reused else 'new_H16_completion_run',
               source_raw=str(run['dir'].parent.parent),source_run=str(run['dir']),hidden_size=16,no_pf_protocol=REFERENCE,
               original_run_id=original['run_id'],original_protocol=original['protocol'],baseline_reuse_is_explicit=True,
               model_width_scope='compatible retained width checkpoint; no independent-seed claim',
               hardware_reuse_scope='same configured Sacramento and exact actual simulator binary; no hardware-area measurement')
    if s['method']=='stride_student':
        x=run['summary'];row.update(model_parameter_count=5220,recurrent_hc_bytes_per_pc=128,
            recurrent_hc_configured_bytes=128*x['state_capacity'],recurrent_hc_peak_bytes=128*x['peak_state_entries'],
            recurrent_hc_resident_bytes=128*x['state_entries'],model_weight_plus_configured_hc_payload_bytes=20880+128*x['state_capacity'],
            state_organization='shared_frozen_weights; exact_PC_hc;64_existing_entries;decoder32',
            storage_scope='FP32 weights plus separately counted per_PC_hc; state entry metadata/queues/control/decoder scratch separate')
    if s['method']=='template':
        text=s['frozen_template'].split();n=(len(text)-2)//2
        row.update(model_parameter_count=0,model_weight_bytes=0,recurrent_hc_bytes=0,inference_macs=0,inference_nonlinears=0,
                   model_weight_dram_bytes=0,state_organization='unconditional_validation_frozen_delta_fill_table; no NN state',
                   template_modal_count=int(text[1]),template_stored_slots=n,template_prediction_fields_bytes=8+12*n,
                   template_declared_Action_fields_bytes=8+28*n,template_logical_Action_ABI_plus_modal_bytes=8+32*n,
                   actual_template_Action_ABI_bytes_per_slot=32,storage_scope='template scalar payload and measured Action ABI; excludes vector/control/allocator')
    row['mac4_aggregate_dense_over_measured_MAC0_window']=OLD.ratio(row['mac4_aggregate_dense_lower_bound_cycles'],row['cycles'])
    return row


def auxiliary(run, metadata):
    x,s=run['summary'],run['spec'];timing=[];states=[]
    for metric,hist in x.get('distributions_log2_upper_bound',{}).items():
        for upper,count in sorted(hist.items(),key=lambda p:int(p[0])):
            timing.append(dict(**metadata,metric=metric,bucket_upper_bound=int(upper),count=count,scope='actual_log2_upper_bound_histogram; MAC0 not finite_cost_service'))
    for pc,values in x.get('per_pc_roi_updates',{}).items():
        states.append(dict(**metadata,pc=pc,completed_updates=values['completed'],evictions=values['evictions'],scope='actual_ROI_per_PC_updates'))
    host=dict(**metadata,wall_seconds=run['record'].get('wall_seconds'),timing_scope='host_seconds_not_simulated_cycles',
              simulated_service_scope=x.get('service_architecture','synchronous_MAC0_no_finite_service_model'))
    hostfile=run['dir']/'host_time.txt'
    if hostfile.is_file():
        for line in hostfile.read_text().splitlines():
            for key,field in (('User time (seconds):','host_user_seconds'),('System time (seconds):','host_system_seconds'),('Maximum resident set size (kbytes):','host_maximum_RSS_kbytes')):
                if line.strip().startswith(key):host[field]=float(line.strip().split(':',1)[1])
    return timing,states,host



def paired_controls(new_metrics, config, audit):
    def typed(value):
        if value=='':return None
        try:return json.loads(value)
        except (ValueError,TypeError):return value
    with (CANONICAL/'per_run.csv').open(newline='') as stream:
        old=[{k:typed(v) for k,v in row.items()} for row in csv.DictReader(stream)]
    audit.check(len(old)==88, 'original_88_source_rows_read_without_modification')
    prior=read(path(config['width_raw'])/'analysis_final/audit.json')
    audit.check(prior['status']=='PASS' and prior['complete_count']==24 and prior['failure_count']==0, 'retained_H16_reference_controls_complete_audit')
    rows=[r for r in old+new_metrics if r.get('stage')=='test' and r.get('hidden_size')==16]
    checked=set()
    def verify_source(row):
        if row['result_id'] in checked:return
        checked.add(row['result_id']);folder=path(row['source_run']);record=read(folder/'run.json');x=read(folder/'summary.json');spec=record['spec']
        audit.check(record['status']=='complete' and record['exit_code']==0 and x['final'] is True
                    and row['run_id']==spec['run_id'] and row['protocol']==spec['protocol']
                    and row['cycles']==x['cycles'] and row['instructions']==x['instructions']
                    and row['raw__request_gate_allowed']==x['request_gate_allowed']
                    and row['raw__dram_total_bus_bytes']==x['dram_total_bus_bytes']
                    and math.isclose(row['ipc'],x['instructions']/x['cycles'],rel_tol=1e-14),
                    'paired_control_source_metrics_match_actual_complete_run',row['run_id'])
    def find(policy,roi,budget,method,mode):
        matches=[r for r in rows if (r['policy'],r['roi'],r['budget'],r['method'],r['mode'])==(policy,roi,budget,method,mode)]
        audit.check(len(matches)==1,'unique_H16_full_or_control_condition',policy=policy,roi=roi,budget=budget,method=method,mode=mode,count=len(matches))
        if len(matches)!=1:raise ValueError('missing/ambiguous H16 control '+str((policy,roi,budget,method,mode)))
        verify_source(matches[0]);return matches[0]
    pairs=[]
    for policy in ('stride','streamer','ampm','spp'):
        method='stride_student' if policy=='stride' else 'student'
        for roi in (0,1):
            for cap in (0,config['common_cap']):
                nn=find(policy,roi,cap,method,'full')
                controls=[('template','full','NN_free_template')]
                if policy!='stride' and cap==0:controls += [('student','fixed_payload','fixed_payload'),('student','reset_state','reset_state')]
                for control_method,mode,label in controls:
                    control=find(policy,roi,cap,control_method,mode)
                    same=('stage','roi','policy','budget','seed','macs','page_policy','skip','warmup','measurement_records','build_id')
                    audit.check(all(nn.get(k)==control.get(k) for k in same), 'paired_control_same_ROI_warmup_budget_actual_build',nn['run_id'],control_run_id=control['run_id'])
                    if control_method=='student':
                        audit.check(nn['model_weight_bytes']==control['model_weight_bytes']==(26436 if policy=='spp' else 26612)
                                    and control['model_parameter_count']==nn['model_parameter_count'], 'same_H16_weights_in_NN_mechanism_control',nn['run_id'],control_run_id=control['run_id'])
                    nreq=nn['requests_per_million_instructions'];creq=control['requests_per_million_instructions']
                    nd=nn['raw__dram_total_bus_bytes'];cd=control['raw__dram_total_bus_bytes']
                    row=dict(policy=policy,hidden_size=16,stage='test',roi=roi,budget=cap,control=label,
                             full_nn_result_id=nn['result_id'],full_nn_run_id=nn['run_id'],full_nn_protocol=nn['protocol'],full_nn_source_run=nn['source_run'],
                             control_result_id=control['result_id'],control_run_id=control['run_id'],control_protocol=control['protocol'],control_source_run=control['source_run'],
                             full_nn_ipc=nn['ipc'],control_ipc=control['ipc'],control_relative_IPC_pct=100*(control['ipc']/nn['ipc']-1),
                             full_nn_requests_per_million=nreq,control_requests_per_million=creq,requests_change_pct=100*(creq/nreq-1) if nreq else None,
                             full_nn_dram_bus_bytes=nd,control_dram_bus_bytes=cd,dram_change_pct=100*(cd/nd-1) if nd else None,
                             delta_allowed_requests=control['raw__request_gate_allowed']-nn['raw__request_gate_allowed'],delta_dram_bus_bytes=cd-nd,
                             delta_cycles=control['cycles']-nn['cycles'],delta_instructions=control['instructions']-nn['instructions'],
                             delta_l2_load_misses=control['l2_load_misses']-nn['l2_load_misses'],
                             delta_llc_load_misses=control['raw__LLC_LOAD_misses']-nn['raw__LLC_LOAD_misses'],
                             cycles_equal=control['cycles']==nn['cycles'],instructions_equal=control['instructions']==nn['instructions'],
                             allowed_requests_equal=control['raw__request_gate_allowed']==nn['raw__request_gate_allowed'],
                             requests_per_million_equal=creq==nreq,dram_bus_bytes_equal=cd==nd,
                             l2_load_misses_equal=control['l2_load_misses']==nn['l2_load_misses'],
                             llc_load_misses_equal=control['raw__LLC_LOAD_misses']==nn['raw__LLC_LOAD_misses'],
                             scope='actual whole_ROI intervention outcomes; changed cache feedback can change future callbacks; no policy_identity or finite_cost claim')
                    pairs.append(row)
    audit.check(len(pairs)==28 and Counter(r['control']for r in pairs)==Counter(NN_free_template=16,fixed_payload=6,reset_state=6), 'all_28_H16_matched_control_comparisons')
    return pairs

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=HERE/'complete_h16.json')
    parser.add_argument('--raw',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--publish',action='store_true',help='Copy only named lightweight extension artifacts into the existing results directory')
    args=parser.parse_args();config=read(args.config);raw=path(args.raw or config['raw']).resolve();out=path(args.out or raw/'analysis').resolve()
    if not inside(raw,CACHE/'formal_study/runs') or not inside(out,raw):raise ValueError('All analysis output must remain under the canonical raw child')
    if out.exists():raise ValueError('Preserve previous analysis; choose a fresh --out directory')
    for variable,name in (('TMPDIR','tmp'),('PYTHONPYCACHEPREFIX','pycache'),('MPLCONFIGDIR','mpl')):
        folder=raw/'runtime/analysis'/name;folder.mkdir(parents=True,exist_ok=True);os.environ[variable]=str(folder)
    audit=OLD.Audit();audit.check(config['protocol']==PROTOCOL and config['seed']==7 and config['macs']==0 and config['common_cap']==16144, 'fixed_H16_completion_protocol')
    audit.check(config['validation_records']==[20000000,25000000] and config['test_records']==[[25000000,50000000],[75000000,100000000]] and config['warmup_instructions']==5000000, 'fixed_validation_full_ROIs_and_warmup')
    planned=expected(config);supplied=read(raw/'matrix.json')
    audit.check(len(supplied)==30 and {s['run_id']for s in supplied}==set(planned), 'exact_30_condition_matrix')
    for s in supplied:audit.check(all(s.get(k)==v for k,v in planned.get(s['run_id'],{}).items()), 'independent_expected_matrix_spec',s['run_id'])
    state,build=verify_bindings(raw,config,audit)
    ref=path(config['reference_raw']);prior=read(ref/'analysis_final/audit.json')
    audit.check(prior['status']=='PASS' and prior['complete_count']==64 and prior['failure_count']==0 and prior['callback_logs_checked'] is True, 'original_reference_strict_complete_audit')
    baselines={(stage,roi):baseline(ref,stage,roi,config,build,audit)for stage,roi in (('validation',0),('test',0),('test',1))}
    runs={};inventory=[];statuses=Counter()
    for folder in sorted((raw/'runs').iterdir()):
        if not folder.is_dir() or not (folder/'run.json').is_file():continue
        record=read(folder/'run.json');rid=record['spec']['run_id'];statuses[record['status']]+=1
        audit.check(rid in planned and folder.name==rid, 'registered_run_directory_identity',rid)
        inventory.append(dict(run_id=rid,status=record['status'],source=str(folder/'run.json')))
        if record['status'] in ('complete','reused') and rid in planned:
            run=load_run(folder,planned[rid],config,build,audit);verify_run(run,config,audit);runs[rid]=run
    missing=sorted(set(planned)-set(runs));controller=read(raw/'controller.json')
    audit.check(not missing and controller['status']=='complete' and controller['protocol']==PROTOCOL, 'all_full_conditions_finished',missing=missing)
    if missing:
        print(json.dumps(dict(status='INCOMPLETE',missing=missing,run_status_counts=dict(statuses))));return 2
    selection=verify_selection(raw,runs,config,audit)
    metrics=[];hists=[];classified=[];snapshots=[];phases=[];callbacks=[];aliases=[];timings=[];state_updates=[];hosts=[]
    for rid,run in sorted(runs.items()):
        s=run['spec'];base=baselines[(s['stage'],s['roi'])]
        row=annotate(OLD.derive(run,base,raw),run,raw)
        audit.check(math.isclose(row['ipc'],run['summary']['instructions']/run['summary']['cycles'],rel_tol=1e-15)
                    and math.isclose(row['requests_per_million_instructions'],1e6*run['summary']['request_gate_allowed']/run['summary']['instructions'],rel_tol=1e-15,abs_tol=1e-15), 'IPC_and_actual_requests_recomputed',rid)
        metrics.append(row);meta={k:row[k]for k in ('result_id','logical_run_id','protocol','evidence_origin','hidden_size')}
        if run['reused']:aliases.append(dict(**meta,source_run=row['source_run'],original_run_id=row['run_id'],logical_protocol=PROTOCOL,reason=run['logical_record']['reuse']['reason']))
        def tagged(items):
            return [dict(item,**meta)for item in items]
        hists.extend(tagged(OLD.profile_rows(run['summary'],s,audit)))
        classified.extend(tagged(OLD.classification_rows(run['summary'],s,audit)))
        seq=OLD.snapshot_rows(run,audit);snapshots.extend(tagged(seq));phases.extend(tagged(OLD.phase_progress(seq,s,config)))
        callbacks.append(dict(OLD.verify_callbacks(run,audit),**meta))
        t,st,host=auxiliary(run,dict(meta,run_id=row['run_id'],policy=s['policy'],stage=s['stage'],roi=s['roi'],method=s['method'],mode=s['mode'],budget=s['budget']))
        timings.extend(t);state_updates.extend(st);hosts.append(host)
    pairs=paired_controls(metrics,config,audit)
    aggregate=Counter()
    for h in hists:aggregate[(h['result_id'],h['logical_run_id'],h['stream'],h['statistic'],h['value'])]+=h['count']
    totals=Counter()
    for (result_id,rid,stream,stat,value),count in aggregate.items():totals[(result_id,rid,stream,stat)]+=count
    full=[dict(result_id=result_id,logical_run_id=rid,run_id=rid,hidden_size=16,stream=stream,statistic=stat,value=value,count=count,
               fraction=OLD.ratio(count,totals[(result_id,rid,stream,stat)]),scope='exact_sum_of_all_complete_ROI_phases')
          for (result_id,rid,stream,stat,value),count in sorted(aggregate.items())]
    record=dict(status='FAIL' if audit.failures else 'PASS',protocol=PROTOCOL,expected_count=30,complete_count=len(runs),
                new_actual_run_count=sum(not r['reused']for r in runs.values()),reused_control_count=sum(r['reused']for r in runs.values()),
                new_test_count=sum(not r['reused'] and r['spec']['stage']=='test'for r in runs.values()),
                new_validation_count=sum(not r['reused'] and r['spec']['stage']=='validation'for r in runs.values()),
                run_status_counts=dict(statuses),failure_count=len(audit.failures),checks=audit.checks,failures=audit.failures,notes=audit.notes,
                callback_logs_checked=True,actual_full_ROI_only=True,raw_inputs_modified=False,raw_root=str(raw),reference_raw=str(ref),
                original_canonical_per_run_untouched=True,plot_status='not_requested; reader plots generated separately')
    out.mkdir(parents=True)
    outputs={'per_run_h16_completion.csv':metrics,'paired_controls_h16_completion.csv':pairs,'action_histograms_h16_completion.csv':hists,'action_histograms_full_roi_h16_completion.csv':full,
             'classified_requests_h16_completion.csv':classified,'progress_snapshots_h16_completion.csv':snapshots,'phase_progress_h16_completion.csv':phases,
             'timing_histograms_h16_completion.csv':timings,'state_updates_h16_completion.csv':state_updates,'host_timings_h16_completion.csv':hosts,
             'control_aliases_h16_completion.csv':aliases}
    for name,rows in outputs.items():
        if name=='control_aliases_h16_completion.csv' and not rows:
            (out/name).write_text('logical_run_id,logical_protocol,result_id,original_run_id,protocol,source_run,reason,hidden_size,evidence_origin\n')
        else:dump(out/name,rows)
    save(out/'audit_h16_completion.json',record);save(out/'callback_verification_h16_completion.json',callbacks)
    save(out/'selection_verification_h16_completion.json',selection);save(out/'inventory_h16_completion.json',inventory)
    if args.publish:
        if audit.failures:raise ValueError('Refuse publishing failed audit')
        for name in (*outputs,'audit_h16_completion.json','callback_verification_h16_completion.json','selection_verification_h16_completion.json'):
            source,target=out/name,CANONICAL/name
            if target.exists() and target.read_bytes()!=source.read_bytes():raise ValueError('Preserve different published extension '+str(target))
            if not target.exists():shutil.copy2(source,target)
    print(json.dumps({k:record[k]for k in ('status','complete_count','expected_count','new_actual_run_count','reused_control_count','failure_count')}))
    return int(bool(audit.failures))


if __name__=='__main__':raise SystemExit(main())
