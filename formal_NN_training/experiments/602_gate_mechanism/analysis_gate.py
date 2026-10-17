#!/usr/bin/env python3
"""Analyze full formal gate/payload runs and explicitly compatible references."""
import argparse,csv,json,math,re,statistics
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone

def read(p):return json.loads(Path(p).read_text())
def div(a,b):return a/b if a is not None and b is not None and b else None
def put(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvout(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def leaves(d,prefix=''):
 for k,v in (d.items() if isinstance(d,dict) else enumerate(d)):
  key=prefix+'/'+str(k)
  if isinstance(v,(dict,list)):yield from leaves(v,key)
  else:yield key,v

PROFILE_KEY=re.compile(r'phase_(\d+)/slot_(\d+|unknown)/current_([01])/duplicate_([01]|unknown)/teacher_([012]|unknown)')
COHORT_FIELDS=('enqueued','timely','late','timely_load','late_load','timely_rfo','late_rfo','unused','redundant_cache','redundant_inflight','pending','resident_unused')

def classified_audit(c,q):
 """Counter identities, not estimates from prefixes or projected request counts."""
 errors=[]
 def ck(ok,why):
  if not ok:errors.append(why)
 groups=c.get('mechanism_classified_requests')
 ck(isinstance(groups,list),'classified request profile is not an ARRAY')
 if not isinstance(groups,list):return errors
 width=c.get('mechanism_phase_instructions',5000000);last=(q['instructions']-1)//width
 totals=Counter();cohorts=Counter();scopes=defaultdict(Counter);keys=set()
 generated=c.get('mechanism_request_profile',{})
 ck(isinstance(generated,dict),'engine generated profile is not an object')
 for g in groups:
  ident=(g.get('arm'),g.get('fill_scope'),g.get('key'))
  ck(ident not in keys,'duplicate classified group');keys.add(ident)
  ck(g.get('arm')==q['arm'],'classified group wrong arm')
  scope=g.get('fill_scope');ck(scope in ('l2_target','llc_only'),'unknown fill scope')
  match=PROFILE_KEY.fullmatch(g.get('key',''))
  ck(match is not None,'malformed classified key')
  if match:
   phase,slot,current,duplicate,teacher=match.groups()
   expected=(int(phase),-1 if slot=='unknown' else int(slot),int(current),-1 if duplicate=='unknown' else int(duplicate),-1 if teacher=='unknown' else int(teacher))
   ck(tuple(g.get(k) for k in ('origin_phase','slot','current_line','list_duplicate','teacher_category'))==expected,'classified metadata/key disagree')
   ck(0<=int(phase)<=last,'origin phase outside planned ROI')
  names=('cache_attempted','allowed','current_suppressed','page_suppressed','budget_suppressed','cache_accepted_nonmerged','cache_merged','cache_rejected','pq_enqueued')
  if not all(isinstance(g.get(k),int) and not isinstance(g.get(k),bool) and g[k]>=0 for k in names):
   errors.append('classified count missing or outside nonnegative integer domain');continue
  totals.update({k:g[k] for k in names});scopes[scope].update({k:g[k] for k in names})
  ck(g['cache_attempted']==sum(g[k] for k in ('allowed','current_suppressed','page_suppressed','budget_suppressed')),'classified gate conservation')
  ck(g['allowed']==sum(g[k] for k in ('cache_accepted_nonmerged','cache_merged','cache_rejected')),'classified cache-result conservation')
  for field,counter in [('actual_attempt_phase_counts','cache_attempted'),('actual_allowed_phase_counts','allowed')]:
   counts=g.get(field,{})
   valid=isinstance(counts,dict) and all(str(k).isdigit() and 0<=int(k)<=last and isinstance(v,int) and v>=0 for k,v in counts.items())
   ck(valid,'bad actual phase bins')
   if not valid:continue
   ck(sum(counts.values())==g[counter],'actual phase sum differs from profile count')
   ck(all(int(k)>=g['origin_phase'] for k in counts),'gate phase predates generation phase')
  if q.get('filter_current') and g.get('current_line'):
   ck(g['allowed']==0 and g['current_suppressed']==g['cache_attempted'],'filtered current request consumed a credit')
  if q['arm']!='spp':
   ck(g.get('slot',-1)>=0 and g.get('teacher_category',-1) in (0,1,2),'engine-labeled request has unknown slot/teacher')
   ck(g['cache_attempted']<=generated.get(g['key'],0),'cache attempts exceed originating generated payload group')
  else:ck(g.get('slot')==g.get('list_duplicate')==g.get('teacher_category')==-1,'native SPP has invented slot/duplicate/teacher labels')
  v=g.get('l2_cohort')
  if scope=='llc_only':
   ck(v is None and not g.get('l2_lifecycle_fully_tracked'),'LLC-only request falsely has a tracked L2 cohort')
   ck(g.get('l2_load_accuracy_enqueued') is None and g.get('l2_load_timeliness') is None,'LLC-only accuracy/timeliness must be null')
   continue
  ck(isinstance(v,dict) and g.get('l2_lifecycle_fully_tracked') is True,'L2 target lacks complete lifecycle representation')
  if not isinstance(v,dict):continue
  valid=all(isinstance(v.get(k),int) and v[k]>=0 for k in COHORT_FIELDS)
  ck(valid,'invalid cohort count')
  if not valid:continue
  cohorts.update({k:v[k] for k in COHORT_FIELDS})
  ck(v['enqueued']==sum(v[k] for k in ('timely','late','unused','redundant_cache','redundant_inflight','pending','resident_unused')),'classified lifecycle conservation')
  ck(v['enqueued']==g['pq_enqueued'],'L2 tracked cohort does not equal actual PQ insertions')
  ck(v.get('useful')==v['timely']+v['late'] and v.get('useful_load')==v['timely_load']+v['late_load'],'classified useful definition')
  for field,expected in [('l2_load_accuracy_enqueued',div(v['timely_load']+v['late_load'],v['enqueued'])),('l2_load_timeliness',div(v['timely_load'],v['timely_load']+v['late_load']))]:
   ck(g.get(field) is None if expected is None else g.get(field) is not None and math.isclose(g[field],expected,rel_tol=1e-12,abs_tol=1e-15),'classified '+field+' denominator')
 for field,counter in [('cache_attempted','request_gate_generated'),('allowed','request_gate_allowed'),('current_suppressed','request_gate_current_suppressed'),('page_suppressed','request_gate_page_suppressed'),('budget_suppressed','request_gate_budget_suppressed'),('cache_accepted_nonmerged','request_cache_enqueued'),('cache_merged','request_cache_merged'),('cache_rejected','request_cache_rejected'),('pq_enqueued','request_cache_pq_inserted')]:
  ck(totals[field]==c.get(counter),'profile total differs from '+counter)
 for field in COHORT_FIELDS:ck(cohorts[field]==c.get('cohort_'+field),'profile lifecycle total differs from cohort_'+field)
 for scope,prefix in [('l2_target','request_l2_target_'),('llc_only','request_llc_only_')]:
  for field,suffix in [('cache_attempted','generated'),('allowed','allowed'),('cache_merged','cache_merged'),('cache_rejected','cache_rejected')]:ck(scopes[scope][field]==c.get(prefix+suffix),'fill-scope total differs from '+prefix+suffix)
 ck(scopes['llc_only']['pq_enqueued']==c.get('cohort_llc_only_excluded'),'LLC-only excluded insertion count')
 ck(c.get('request_cache_pq_inserted')==c['cohort_enqueued']+c.get('cohort_llc_only_excluded',0),'PQ partition into tracked L2 / excluded LLC')
 ck(c.get('measurement_unconsumed_context_labels')==0,'unconsumed diagnostic labels')
 ck(c.get('measurement_cohort_scope')=='l2_target_fully_tracked_only','wrong lifecycle scope')
 ck(c.get('llc_only_lifecycle_accuracy_available') is False,'LLC lifecycle availability falsely claimed')
 ck(c.get('cohort_all_enqueued_requests_fully_tracked')==(c.get('cohort_llc_only_excluded')==0),'all-enqueued tracking availability disagrees with LLC exclusion')
 if isinstance(generated,dict):
  valid=all(PROFILE_KEY.fullmatch(k) and isinstance(v,int) and v>=0 for k,v in generated.items())
  ck(valid,'malformed engine generated request profile')
  if valid:
   ck(sum(generated.values())==c.get('mechanism_payload_generated_addresses'),'engine generated-profile total')
   for token,counter in [('/current_1/','mechanism_payload_current_line_addresses'),('/duplicate_1/','mechanism_payload_list_duplicate_addresses')]:ck(sum(v for k,v in generated.items() if token in k)==c.get(counter),'engine generated-profile '+counter)
 return errors

def distributions_audit(c,q):
 errors=[];counts=Counter();weighted=Counter();phase_payload=Counter();delta_counts=Counter()
 width=c.get('mechanism_phase_instructions',5000000);last=(q['instructions']-1)//width
 for name,bins in c.get('mechanism_distributions_exact',{}).items():
  match=re.fullmatch(r'phase_(\d+)_(raw|payload)_(emit|K|delta_slot_\d+)',name)
  if not match:errors.append('unknown full-ROI distribution '+name);continue
  phase,prefix,kind=int(match[1]),match[2],match[3]
  if not 0<=phase<=last:errors.append('distribution phase outside planned ROI')
  if not isinstance(bins,dict) or not all(re.fullmatch(r'-?\d+',str(k)) and isinstance(v,int) and v>=0 for k,v in bins.items()):
   errors.append('distribution invalid integer/count domain');continue
  counts[(prefix,kind)]+=sum(bins.values());weighted[(prefix,kind)]+=sum(int(k)*v for k,v in bins.items())
  if kind=='emit' and any(int(k) not in (0,1) for k in bins):errors.append('emit distribution not boolean')
  if kind=='K' and any(int(k)<0 for k in bins):errors.append('negative count distribution')
  if kind=='K' and prefix=='payload':phase_payload[phase]+=sum(int(k)*v for k,v in bins.items())
  if kind.startswith('delta_slot_'):delta_counts[prefix]+=sum(bins.values())
 expected=c.get('nn_completed',0) if q['method']=='frozen' else c.get('eligible_l2_callbacks',0) if q['method'] in ('fixed_plus2','pc_gate_plus2','stride') else 0
 for kind in ('emit','K'):
  if counts[('payload',kind)]!=expected:errors.append('all-completed payload '+kind+' event count')
  if q['method']=='frozen' and counts[('raw',kind)]!=expected:errors.append('all-completed raw '+kind+' event count')
 if weighted[('payload','K')]!=c.get('mechanism_payload_generated_addresses'):errors.append('payload K weighted sum differs from all generated addresses')
 for prefix in ('raw','payload'):
  if delta_counts[prefix]!=c.get('mechanism_'+prefix+'_generated_addresses'):errors.append(prefix+' delta-slot histograms do not cover all actual generated addresses')
 generated_phases=Counter()
 for key,count in c.get('mechanism_request_profile',{}).items():
  match=PROFILE_KEY.fullmatch(key)
  if match:generated_phases[int(match[1])]+=count
 if +generated_phases!=+phase_payload:errors.append('per-phase generated profile differs from payload K weighted histogram')
 return errors

FIELDS=['instructions','cycles','NL','M','P','Ipf','request_gate_generated','request_gate_allowed',
 'request_gate_budget_suppressed','request_gate_page_suppressed','request_cache_enqueued','request_cache_merged','request_cache_rejected',
 'cohort_enqueued','cohort_timely_load','cohort_late_load','cohort_timely_rfo','cohort_late_rfo','cohort_unused',
 'cohort_redundant_cache','cohort_redundant_inflight','cohort_pending','cohort_resident_unused',
 'dram_read_bus_completed_transactions','dram_write_bus_completed_transactions','dram_total_bus_bytes',
 'dram_read_bus_transfers','dram_write_bus_transfers','dram_bus_transfers_in_progress','dram_read_queue_pending','dram_write_queue_pending',
 'dram_initial_read_queue_pending','dram_initial_write_queue_pending','dram_initial_busy_channels',
 'nn_admitted','nn_started','nn_completed','inference_dropped','input_queue','output_queue','inference_inflight',
 'inference_macs','inference_nonlinears','inference_weight_read_bytes','nn_weight_dram_bytes','weight_bytes',
 'state_evictions','warmup_completed_updates','warmup_state_evictions','warmup_discarded_work','warmup_discarded_outputs',
 'inference_wait_cycles_sum','inference_completion_cycles_sum','inference_completed_service_cycles_sum',
 'pc_gate_seen_events','pc_gate_unseen_events','pc_gate_emit_events','pc_gate_entries','pc_gate_capacity',
 'mechanism_raw_generated_addresses','mechanism_raw_current_line_addresses','mechanism_raw_list_duplicate_addresses',
 'mechanism_payload_generated_addresses','mechanism_payload_current_line_addresses','mechanism_payload_list_duplicate_addresses',
 'request_l2_target_allowed','request_llc_only_allowed','cohort_llc_only_excluded','measurement_cohort_scope',
 'request_gate_current_suppressed','request_cache_pq_inserted','request_l2_target_generated','request_llc_only_generated',
 'request_l2_target_cache_merged','request_llc_only_cache_merged','request_l2_target_cache_rejected','request_llc_only_cache_rejected',
 'cohort_all_enqueued_requests_fully_tracked','llc_only_lifecycle_accuracy_available','l2_cohort_coverage_scope','measurement_unconsumed_context_labels',
 'engine_method','engine_payload_mode','h','macs_per_cycle','weight_bandwidth_bytes_per_cycle','nonlinear_cycles_per_operation',
 'state_capacity','input_queue_capacity','output_queue_capacity','deployment_bytes_configured','measurement_shadow_bytes']

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--prior-raw',type=Path);p.add_argument('--partial',action='store_true');p.add_argument('--no-plots',action='store_true');a=p.parse_args()
 if a.out.exists() and any(a.out.iterdir()):raise RuntimeError('Use a fresh output directory to preserve previous analyses.')
 a.out.mkdir(parents=True,exist_ok=True);matrix=read(a.raw/'formal_matrix.json');manifest=read(a.raw/'compatible_reference_manifest.json')
 refs={r['planned_spec']['run_id']:r for r in manifest['references']};runs=[];pending=[];fail=[]
 def check(ok,r,why):
  if not ok:fail.append(dict(run_id=r,reason=why))
 for planned in matrix:
  rid=planned['run_id'];reuse=planned['execution']=='compatible_reference'
  if reuse:
   info=refs[rid];folder=a.raw/info['relative_reference']
  else:folder=a.raw/'test'/rid
  if not (folder/'run.json').exists():pending.append(rid);continue
  record=read(folder/'run.json')
  if record.get('status')!='complete':pending.append(rid);continue
  c=read(folder/'summary.json');s=record['spec'];check(c.get('final') is True and c.get('event')=='snapshot',rid,'nonfinal summary')
  check(abs(c['instructions']-planned['instructions'])<=4 and c['cycles']>0,rid,'ROI denominator')
  for k in ['roi','arm','seed','macs','budget','history','warmup','skip','instructions']:
   check(s[k]==planned[k],rid,'incompatible spec '+k)
  if not reuse:
   for k,v in planned.items():check(s.get(k)==v,rid,'planned identity '+k)
   check(s['protocol']=='gate_mechanism_20261016_v1',rid,'wrong protocol')
   check(c.get('engine_method')==s['method'] and c.get('engine_payload_mode')==s['payload'],rid,'wrong registered method/payload')
   check(c.get('numerical_failures',0)==0,rid,'numerical failure')
   for reason in classified_audit(c,planned):check(False,rid,reason)
   for reason in distributions_audit(c,planned):check(False,rid,reason)
  if planned['budget']:check(c['request_gate_allowed']<=c['instructions']*planned['budget']//1000000,rid,'cap violated')
  check(c['request_gate_allowed']==c['request_cache_enqueued']+c['request_cache_merged']+c['request_cache_rejected'],rid,'cache request conservation')
  check(c['cohort_enqueued']==sum(c[k] for k in ['cohort_timely','cohort_late','cohort_unused','cohort_redundant_cache','cohort_redundant_inflight','cohort_pending','cohort_resident_unused']),rid,'L2 lifecycle conservation')
  check(c['dram_read_bus_transfers']+c['dram_write_bus_transfers']==c['dram_read_bus_completed_transactions']+c['dram_write_bus_completed_transactions']+c['dram_bus_transfers_in_progress'],rid,'DRAM flow')
  check(c['dram_total_bus_bytes']==64*(c['dram_read_bus_completed_transactions']+c['dram_write_bus_completed_transactions']),rid,'DRAM bytes')
  if planned['method']=='frozen':
   check(s['model_metadata']['seed']==s['seed'] and s['model_metadata']['emit_loss']=='balanced',rid,'checkpoint identity')
   check(c['nn_completed']==c['state_history_completed']==sum(c['pc_completed_updates'].values()),rid,'completed per-PC state updates')
   check(c['nn_admitted']==c['nn_started']+c['input_queue'],rid,'input queue')
   check(c['nn_started']==c['nn_completed']+c['inference_inflight'],rid,'serial active')
   check(c['weight_bytes']==7632 and c['h']==8,rid,'frozen h8 checkpoint')
  elif planned['arm'] in ['fixed_plus2','pc_gate_plus2']:
   check(c['nn_completed']==0 and c['h']==0,rid,'simple arm unexpectedly uses NN or is NoPF')
   if planned['arm']=='pc_gate_plus2':check(c['pc_gate_seen_events']+c['pc_gate_unseen_events']==c['eligible_l2_callbacks'],rid,'PC table seen/miss partition')
  runs.append(dict(planned=planned,spec=s,summary=c,directory=str(folder),record=record))
 refs_nopf={r['planned']['roi']:r for r in runs if r['planned']['arm']=='none'}
 compatibility=[]
 if a.prior_raw:
  for roi,run in refs_nopf.items():
   old=a.prior_raw/'test'/f'test_602_r{roi}_none_s7_balanced_m0_b0_sil0_qfifo_hist5000000'/'summary.json'
   oldc=read(old);newc=run['summary'];common=oldc.keys()&newc.keys();diff={k:[oldc[k],newc[k]] for k in common if oldc[k]!=newc[k]}
   compatibility.append(dict(kind='NoPF',roi=roi,fields=len(common),differences=diff,original=str(old)))
   check(not diff,run['planned']['run_id'],'fresh NoPF differs from original v2')
  for run in runs:
   q=run['planned']
   if q['arm']=='spp':
    old=a.prior_raw/'test'/f'test_602_r{q["roi"]}_spp_s7_balanced_m0_b{q["budget"]}_sil0_qfifo_hist5000000'/'summary.json'
    if not old.exists():continue
    oldc=read(old);newc=run['summary']
    stable=['instructions','cycles','M','NL','P','request_gate_allowed']+sorted(k for k in oldc.keys()&newc.keys() if k.startswith('dram_'))
    diff={k:[oldc[k],newc[k]] for k in stable if oldc[k]!=newc[k]}
    compatibility.append(dict(kind='native_SPP_functional_only',run=q['run_id'],fields=len(stable),differences=diff,original=str(old),excluded='Mixed-fill old cohort metrics are invalid as whole-SPP accuracy/timeliness/coverage. New L2-target subset is a distinct denominator.'))
    check(not diff,q['run_id'],'native SPP functional IPC/miss/request/DRAM behavior changed');continue
   if q['arm']!='nn' or q['filter_current'] or q['silent']:continue
   old=a.prior_raw/'test'/f'test_602_r{q["roi"]}_nn_s{q["seed"]}_balanced_m{q["macs"]}_b{q["budget"]}_sil0_qfifo_hist5000000'/'summary.json'
   if not old.exists():continue
   oldc=read(old);stable=['instructions','cycles','M','NL','P','request_gate_allowed','dram_total_bus_bytes','nn_completed','inference_macs','nn_admitted','nn_started','inference_dropped','state_evictions','inference_weight_read_bytes']
   diff={k:[oldc[k],run['summary'][k]] for k in stable if oldc[k]!=run['summary'][k]}
   compatibility.append(dict(kind='unfiltered_NN',run=q['run_id'],fields=len(stable),differences=diff,original=str(old)));check(not diff,q['run_id'],'unfiltered NN changed functional behavior')
 metrics=[];dists=[];profiles=[];classified_rows=[];classified_phases=[];pc_updates=[];curves=[];all_numeric=[];raw_rows=[]
 for run in runs:
  q,s,c=run['planned'],run['spec'],run['summary'];rid=q['run_id'];ref=refs_nopf.get(q['roi']);b=ref['summary'] if ref else {}
  ident={k:q[k] for k in ['run_id','roi','arm','seed','macs','budget','filter_current','payload','silent','execution']}
  ident.update(source_protocol=s['protocol'],source_build=s['build_id'])
  raw_rows.append(dict(ident,**{'raw__'+k:v for k,v in c.items()}))
  t,l=c['cohort_timely_load'],c['cohort_late_load'];en=c['cohort_enqueued'];allowed_l2=c.get('request_l2_target_allowed',c['request_gate_allowed'] if q['arm']!='spp' else None)
  known_l2=c.get('measurement_cohort_scope')=='l2_target_fully_tracked_only' or (q['execution']=='compatible_reference' and q['arm'].startswith('stride_'))
  tracked_t,tracked_l,tracked_en=(t,l,en) if known_l2 else (None,None,None)
  useful=tracked_t+tracked_l if known_l2 else None
  ipc=c['instructions']/c['cycles'];unresolved=c['cohort_pending']+c['cohort_resident_unused']
  m=dict(ident,ipc=ipc,relative_ipc_vs_nopf=div(ipc,div(b.get('instructions'),b.get('cycles'))),
   requests_per_million=1e6*c['request_gate_allowed']/c['instructions'],l2_load_mpki=1000*c['M']/c['instructions'],
   l2_load_miss_reduction=div(b.get('M',0)-c['M'],b.get('M')),l2_target_timely_coverage=div(tracked_t,b.get('M')),
   l2_target_useful_accuracy_enqueued=div(useful,tracked_en),l2_target_tracked_load_uses_per_submission=div(useful,allowed_l2),
   l2_target_useful_accuracy_resolved=div(useful,en-unresolved if known_l2 else None),l2_target_timeliness=div(tracked_t,useful),
   l2_cohort_denominator_verified=known_l2,classified_request_profile_available=isinstance(c.get('mechanism_classified_requests'),list),
   llc_only_accuracy=None,llc_only_timeliness=None,
   coverage_scope='L2-target tracked subset only; LLC-only usefulness unavailable' if q['arm']=='spp' else 'Fully tracked L2-target prefetch cohort',
   mean_queue_wait_cycles=div(c.get('inference_wait_cycles_sum'),c.get('nn_started')),
   mean_completion_age_cycles=div(c.get('inference_completion_cycles_sum'),c.get('nn_completed')),
   **{k:c.get(k) for k in FIELDS})
  metrics.append(m)
  for path,value in leaves(c):
   if isinstance(value,(int,float,str,bool)) and any(token in path for token in ['mechanism','request_profile','cohort_scope','l2_target','llc_only','filter_current']):all_numeric.append(dict(ident,path=path,value=value))
  for name,bins in c.get('mechanism_distributions_exact',{}).items():
   mm=re.fullmatch(r'phase_(\d+)_(.*)',name)
   if not mm:raise RuntimeError('Unknown full-distribution schema '+name)
   for value,count in bins.items():dists.append(dict(ident,phase=int(mm[1]),distribution=mm[2],value=int(value),count=count))
  for field,object in c.items():
   if isinstance(object,dict) and ('profile' in field or 'mechanism_request' in field):
    for path,value in leaves(object):profiles.append(dict(ident,profile=field,key=path,value=value))
  for group in c.get('mechanism_classified_requests',[]):
   classified_rows.append(dict(ident,**group))
   for field,counter in [('actual_attempt_phase_counts','cache_attempted'),('actual_allowed_phase_counts','allowed')]:
    for actual_phase,count in group[field].items():classified_phases.append(dict(ident,fill_scope=group['fill_scope'],key=group['key'],origin_phase=group['origin_phase'],slot=group['slot'],current_line=group['current_line'],list_duplicate=group['list_duplicate'],teacher_category=group['teacher_category'],counter=counter,actual_phase=int(actual_phase),count=count))
  for pc,updates in c.get('pc_completed_updates',{}).items():pc_updates.append(dict(ident,pc=pc,completed_updates=updates,state_evictions=c.get('pc_state_evictions',{}).get(pc,0),warmup_updates=c.get('pc_warmup_updates',{}).get(pc,0)))
  previous=None
  for line in (Path(run['directory'])/'snapshots.jsonl').read_text().splitlines():
   v=json.loads(line)
   if v.get('event')!='snapshot' or not v['cycles']:continue
   if previous and v['instructions']==previous['instructions']:continue
   di=v['instructions']-(previous['instructions'] if previous else 0);dc=v['cycles']-(previous['cycles'] if previous else 0)
   curves.append(dict(ident,instructions=v['instructions'],cycles=v['cycles'],cumulative_ipc=v['instructions']/v['cycles'],interval_instructions=di,interval_cycles=dc,interval_ipc=div(di,dc),final=v['final']))
   previous=v
 # Explicit whole-ROI sums over every disjoint phase, never prefix estimates.
 sums=Counter()
 for r in dists:sums[(r['run_id'],r['distribution'],r['value'])]+=r['count']
 byid={r['run_id']:r for r in metrics};whole=[]
 for (rid,dist,value),count in sums.items():whole.append(dict(run_id=rid,roi=byid[rid]['roi'],arm=byid[rid]['arm'],seed=byid[rid]['seed'],macs=byid[rid]['macs'],budget=byid[rid]['budget'],filter_current=byid[rid]['filter_current'],distribution=dist,value=value,count=count))
 csvout(a.out/'metrics_per_seed.csv',metrics);csvout(a.out/'full_roi_distributions.csv',whole);csvout(a.out/'phase_distributions.csv',dists)
 csvout(a.out/'request_profiles.csv',profiles);csvout(a.out/'mechanism_fields.csv',all_numeric);csvout(a.out/'per_pc_updates.csv',pc_updates);csvout(a.out/'startup_curves.csv',curves)
 csvout(a.out/'classified_requests.csv',classified_rows);csvout(a.out/'classified_request_actual_phases.csv',classified_phases);csvout(a.out/'raw_per_seed.csv',raw_rows)
 full_classes={}
 for row in classified_rows:
  key=tuple(row[k] for k in ('run_id','fill_scope','slot','current_line','list_duplicate','teacher_category'))
  if key not in full_classes:
   full_classes[key]={k:row[k] for k in ('run_id','roi','arm','seed','macs','budget','filter_current','payload','silent','execution','source_protocol','source_build','fill_scope','slot','current_line','list_duplicate','teacher_category','l2_lifecycle_fully_tracked')}
   full_classes[key].update(origin_phase='all',actual_attempt_phase_counts=Counter(),actual_allowed_phase_counts=Counter(),l2_cohort=Counter() if row['l2_lifecycle_fully_tracked'] else None)
  whole_class=full_classes[key]
  for field in ('cache_attempted','allowed','current_suppressed','page_suppressed','budget_suppressed','cache_accepted_nonmerged','cache_merged','cache_rejected','pq_enqueued'):whole_class[field]=whole_class.get(field,0)+row[field]
  for field in ('actual_attempt_phase_counts','actual_allowed_phase_counts'):whole_class[field].update(row[field])
  if row['l2_lifecycle_fully_tracked']:whole_class['l2_cohort'].update(row['l2_cohort'])
 for row in full_classes.values():
  cohort=row['l2_cohort'];useful=cohort['timely_load']+cohort['late_load'] if cohort is not None else None
  row.update(l2_load_accuracy_enqueued=div(useful,cohort['enqueued'] if cohort is not None else None),l2_load_timeliness=div(cohort['timely_load'] if cohort is not None else None,useful))
 csvout(a.out/'full_roi_request_classes.csv',list(full_classes.values()))
 pairs=[];missing_pair_controls=[]
 def pair(t,control,kind):
  if not control:missing_pair_controls.append(dict(treatment=t['run_id'],comparison=kind));return
  if len(control)!=1:check(False,t['run_id'],'nonunique '+kind+' control');return
  z=control[0];pairs.append(dict(comparison=kind,treatment=t['run_id'],control=z['run_id'],roi=t['roi'],seed=t['seed'],macs=t['macs'],budget=t['budget'],
   treatment_ipc=t['ipc'],control_ipc=z['ipc'],delta_ipc=t['ipc']-z['ipc'],relative_ipc_change_pct=100*(t['ipc']/z['ipc']-1),
   delta_requests_per_million=t['requests_per_million']-z['requests_per_million'],delta_dram_bus_bytes=t['dram_total_bus_bytes']-z['dram_total_bus_bytes']))
 for t in metrics:
  same=lambda r:all(r[k]==t[k] for k in ['roi','seed','macs','budget']) and r['silent']==0
  if t['arm']=='nn' and t['filter_current']==1:
   pair(t,[r for r in metrics if same(r) and r['arm']=='nn' and r['filter_current']==0],'filter_triggering_current_line')
  if t['arm'] in ['nn_gate_fixed_plus2','pc_gate_plus2']:
   pair(t,[r for r in metrics if same(r) and r['arm']=='nn' and r['filter_current']==1],'replace_learned_payload' if t['arm']=='nn_gate_fixed_plus2' else 'frozen_pc_gate_plus2_vs_learned')
  if t['arm']=='nn_gate_fixed_plus2':
   pair(t,[r for r in metrics if all(r[k]==t[k] for k in ['roi','macs','budget']) and r['arm']=='fixed_plus2'],'nn_gate_plus2_vs_unconditional')
  if t['arm']=='pc_gate_plus2':pair(t,[r for r in metrics if same(r) and r['arm']=='nn_gate_fixed_plus2'],'pc_gate_plus2_vs_nn_gate_plus2')
  if t['silent']:pair(t,[r for r in metrics if same(r) and r['arm']=='nn' and r['filter_current']==0],'teacher_silent_output_suppression')
 csvout(a.out/'paired_comparisons.csv',pairs)
 audit=dict(status='FAIL' if fail else 'INCOMPLETE' if pending else 'PASS',planned_rows=len(matrix),completed=len(runs),pending=pending,failures=fail,
  execution_counts=dict(Counter(r['planned']['execution'] for r in runs)),reference_compatibility=compatibility,
  no_nopf_reference_rows=[r['run_id'] for r in metrics if r['relative_ipc_vs_nopf'] is None],missing_pair_controls=missing_pair_controls,
  classified_profiles_available=sum(r['classified_request_profile_available'] for r in metrics),legacy_reference_profiles_not_imputed=sum(not r['classified_request_profile_available'] for r in metrics),
  created_utc=datetime.now(timezone.utc).isoformat())
 put(a.out/'audit.json',audit)
 put(a.out/'definitions.json',dict(ipc='Actual ROI retired instructions / cycles',requests_per_million='Actual common-credit submissions *1e6 / actual retired instructions; current-line rejected before credit; duplicates counted separately, not deduplicated',
 coverage='LOAD-only timely cohort usefulness / same-ROI NoPF LOAD misses; SPP is only tracked L2-target subset, never all-SPP coverage',accuracy='LOAD usefulness / fully tracked L2-target PQ-enqueued cohort. Pending/resident-unused remain right-censored in this denominator. Resolved-only accuracy conditions on observed resolution. LLC-only accuracy/timeliness null; old mixed-fill SPP cohorts unavailable.',submission_yield='l2_target_tracked_load_uses_per_submission = tracked L2-target LOAD uses / L2-target allowed cache calls. This is not complete-cohort accuracy: submitted calls include merge/reject and possible WQ-forward outcomes. Historical request_cache_enqueued means API accepted-nonmerged (-1), while request_cache_pq_inserted is actual PQ insertion.',timeliness='L2-target timely LOAD / (timely+late LOAD); LLC-only usefulness unavailable',dram='Instrumented completed controller bus transactions*line size; initial/end queues and bus-start/in-progress separate. No issued*64 inference.',filter_pairs='Same seed/ROI/cost/common cap and frozen checkpoint; current filter changes only the common output gate before credit. Cache feedback can change later arrivals, generated outputs and completions, so equal generated counts or equal realized requests are not assumed. Classified current-line allowed count must be zero in filtered arms; both cap arms independently obey the validation-selected ceiling.',
 distributions='All completed ROI events accumulated in disjoint5M instruction phases, last phase absorbs retirement overshoot. Full ROI is their exact sum. Raw learned forward outputs and final replacement payload distributions are separate. Generation/allowed/useful cohort-phase definitions in measurement schema.',classified_requests='Engine mechanism_request_profile counts generated payload addresses before engine output filters. Observer mechanism_classified_requests ARRAY counts actual cache-call attempts after those filters. classified_requests.csv preserves every group and nested L2 cohort; LLC-only L2 outcomes null, native SPP slot/duplicate/teacher unknown (-1). Origin phase is payload generation/completion, not input arrival. actual_attempt/allowed_phase_counts record actual gate phases separately. Redundant/timely/late/useful/censored lifecycles are credited to the originating generation phase. The older100k cohort buckets are instead actual PQ-enqueue instruction buckets. Compatible old Stride references have no new classified profiles; they are not imputed.',startup='Every saved cumulative snapshot plus exact adjacent interval IPC; prefixes are not whole-ROI distributions or minimum observation thresholds.',pc_gate='Frozen exact-PC bounded64 table fitted to full train+validation NN emit outputs, majority fixed>=0.5, top-frequency fixed selection, unseen PC silent; no test fitting. A gap is not evidence of stateful novelty.',cost='NN-gate replacement keeps full original forward and h/c; zero-cost arms are functional comparators. MAC4 is only seed7 cost reference. Table/CAM/add latency and physical energy/PPA are not synthesized.'))
 if not a.no_plots:plot(a.out,metrics,curves)
 print(json.dumps({k:audit[k] for k in ['status','planned_rows','completed','execution_counts']}))
 if fail or (pending and not a.partial):raise SystemExit(1)

def plot(out,metrics,curves):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 colors={'nn_unfiltered':'#3571a2','nn_filtered':'#e88929','nn_gate_fixed_plus2':'#239d83','fixed_plus2':'#9367bd','pc_gate_plus2':'#ad5864','stride_forward1':'#7c8e37','stride_forward2':'#5e668b','spp':'#546273','none':'#222222'}
 def label(r):return 'nn_filtered' if r['arm']=='nn' and r['filter_current'] else 'nn_unfiltered' if r['arm']=='nn' else r['arm']
 for roi in [0,1]:
  for cap in [0,2363]:
   group=[r for r in metrics if r['roi']==roi and r['budget']==cap and r['macs']==0 and not r['silent'] and r['arm']!='none']
   names=[x for x in colors if any(label(r)==x for r in group)]
   fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
   for i,name in enumerate(names):
    rr=sorted([r for r in group if label(r)==name],key=lambda r:r['seed'])
    for j,r in enumerate(rr):ax.scatter(i+(j-(len(rr)-1)/2)*.09,100*(r['relative_ipc_vs_nopf']-1),color=colors[name],marker={7:'o',17:'s',27:'^'}.get(r['seed'],'D'),s=45)
   ax.axhline(0,color='#536779',lw=1);ax.set_xticks(range(len(names)),[x.replace('_',' ') for x in names],rotation=25,ha='right');ax.set_ylabel('IPC change vs matched NoPF (%)');ax.set_title(f'602 ROI{roi} · '+('original request policy' if not cap else f'common ceiling{cap}/M')+'\nPoints retain actual independent training seeds; conventional references are deterministic')
   for ext in ['png','svg']:fig.savefig(out/f'ipc_roi{roi}_cap{cap}.{ext}',dpi=170)
   plt.close(fig)
  for cap in [0,2363]:
   selected=[r for r in metrics if r['roi']==roi and r['budget']==cap and r['macs']==0 and not r['silent'] and r['seed']==7 and r['arm'] in ['nn','nn_gate_fixed_plus2','fixed_plus2','pc_gate_plus2','stride_forward1','spp']]
   selected += [r for r in metrics if r['roi']==roi and r['arm']=='none']
   fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
   for r in selected:
    cc=[v for v in curves if v['run_id']==r['run_id']];name=label(r);color=colors.get(name,'#555555')
    for ax,field in zip(axes,['cumulative_ipc','interval_ipc']):ax.plot([v['instructions']/1e6 for v in cc],[v[field] for v in cc],label=name.replace('_',' '),color=color,alpha=.85,lw=1)
   for ax,name in zip(axes,['Cumulative','Adjacent saved interval']):ax.set_xlabel('ROI retired instructions (M)');ax.set_ylabel(name+' IPC');ax.grid(alpha=.2)
   axes[0].legend(fontsize=8);fig.suptitle(f'602 ROI{roi} · seed7 startup snapshots · cap{cap}\nPrefixes do not substitute for full-ROI distributions; no minimum-history crossing claim')
   for ext in ['png','svg']:fig.savefig(out/f'startup_roi{roi}_cap{cap}.{ext}',dpi=170)
   plt.close(fig)

if __name__=='__main__':main()
