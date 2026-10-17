#!/usr/bin/env python3
"""Cross-check actual formal results, denominators, resource and flow identities."""
import argparse
import json
from pathlib import Path


def load(p):return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--expect-complete',action='store_true')
    p.add_argument('--matrix',type=Path,help='Explicit stored supplement matrix; default is RUN_DIR/formal_matrix.json.')
    p.add_argument('--training-evidence',type=Path,help='Original immutable training/parity root when reusing those checkpoints in a separate controlled experiment.')
    a=p.parse_args();raw=a.run_dir
    matrix=load(a.matrix or raw/'formal_matrix.json');failures=[];passed=[];pending=[];origins={}
    def check(ok,run,reason):
        if not ok:failures.append({'run_id':run,'reason':reason})
    check(len(matrix)==len({s['run_id'] for s in matrix}),'matrix','duplicate run identifier')
    for expected in matrix:
        rid=expected['run_id'];directory=raw/'test'/rid;rp=directory/'run.json'
        if not rp.exists():pending.append({'run_id':rid,'status':'not_started'});continue
        record=load(rp)
        if record['status']!='complete':pending.append({'run_id':rid,'status':record['status']});continue
        before=len(failures);s=record['spec'];r=load(directory/'summary.json')
        for k,v in expected.items():check(s.get(k)==v,rid,'specification mismatch: '+k)
        n=r['instructions'];c=s['controls'];h=s.get('model_metadata',{}).get('hidden_size',0)
        check(r.get('final') is True and r.get('event')=='snapshot',rid,'no final snapshot')
        check(abs(n-s['instructions'])<=4 and r['cycles']>0,rid,'invalid ROI denominator')
        origin=r['trace_origin_records'];declared=s['skip']+s['warmup']
        check(0<=origin-declared<=4,rid,'unexpected actual warmup boundary')
        group=(s['workload'],s['roi'])
        if group in origins:check(origin==origins[group],rid,'cache warmup boundary differs between policies')
        else:origins[group]=origin
        for field,value in [('h',h),('macs_per_cycle',s['macs']),('state_capacity',c['state_capacity']),
          ('input_queue_capacity',c['input_queue']),('output_queue_capacity',c['output_queue']),
          ('nonlinear_cycles_per_operation',c['nonlinear_cycles']),('page_policy',c['page_policy']),
          ('history_instructions',s['history']),('queue_policy',s['queue']),('silent_suppress',s['silent'])]:
            check(r.get(field)==value,rid,'engine setting mismatch: '+field)
        check(r.get('numerical_failures',0)==0,rid,'numerical failure')
        check(r['request_gate_generated']==r['request_gate_allowed']+r['request_gate_budget_suppressed']+r['request_gate_page_suppressed'],rid,'request gate conservation')
        check(r['request_gate_allowed']==r['request_cache_enqueued']+r['request_cache_merged']+r['request_cache_rejected'],rid,'cache submission conservation')
        check(r['P']==r['request_gate_allowed'],rid,'prefetch submission scope differs')
        if s['budget']:check(r['request_gate_allowed']<=n*s['budget']//1000000,rid,'common request ceiling violated')
        outcomes=('cohort_timely','cohort_late','cohort_unused','cohort_redundant_cache',
                  'cohort_redundant_inflight','cohort_pending','cohort_resident_unused')
        check(r['cohort_enqueued']==sum(r[k] for k in outcomes),rid,'request lifecycle conservation')
        check(r['cohort_pending']+r['cohort_resident_unused']==r['measurement_unresolved_lifecycles'],rid,'unresolved request scope')
        check(r['shadow_teacher_calls']==r['eligible_l2_callbacks'],rid,'shadow did not observe every eligible LOAD')
        check(r['teacher_active_events']+r['teacher_silent_events']==r['eligible_l2_callbacks'],rid,'teacher event partition')
        starts=r['dram_read_bus_transfers']+r['dram_write_bus_transfers']
        completes=r['dram_read_bus_completed_transactions']+r['dram_write_bus_completed_transactions']
        check(starts==completes+r['dram_bus_transfers_in_progress'],rid,'DRAM started/completed/in-progress conservation')
        check(r['dram_total_bus_bytes']==completes*r['line_bytes'],rid,'DRAM byte accounting')
        if s['arm']=='nn':
            check(r['nn_started']==r['nn_completed']+r['inference_inflight'],rid,'serial active/completed conservation')
            check(r['nn_admitted']==r['nn_started']+r['input_queue']+r.get('queue_evicted_oldest',0),rid,'input queue conservation')
            if s['queue']=='fifo':check(r['eligible_l2_callbacks']==r['nn_admitted']+r['inference_dropped'],rid,'FIFO admission conservation')
            else:check(r['eligible_l2_callbacks']==r['nn_admitted'] and r['inference_dropped']==r.get('queue_evicted_oldest',0),rid,'drop-oldest accounting')
            check(r['nn_completed']==r['state_history_completed']==sum(r['pc_completed_updates'].values()),rid,'per-PC completed state update conservation')
            check(r['state_entries']<=c['state_capacity'] and r['peak_input_queue']<=c['input_queue'] and r['peak_output_queue']<=c['output_queue'],rid,'capacity exceeded')
            hist=r['distributions_log2_upper_bound']
            for field,total in [('scheduled_service_cycles',r['nn_started']),('completed_service_cycles',r['nn_completed']),
                 ('completion_age_cycles',r['nn_completed']),('queue_wait_cycles',r['nn_started']),('prior_completed_updates',r['nn_completed'])]:
                check(sum(hist.get(field,{}).values())==total,rid,'distribution population mismatch: '+field)
            check(r['generated_addresses']==sum(r.get(k+'_generated',0) for k in ('teacher_overlap','teacher_active_extra','teacher_silent_extra')),rid,'request attribution partition')
            check(r['weight_bytes']==7632 and r['deployment_bytes_configured']==16808 and r['nn_weight_dram_bytes']==0,rid,'model/storage/resident-weight assumption mismatch')
        if len(failures)==before:passed.append(rid)
    training=a.training_evidence or raw
    pairs=load(training/'training_pairs.json')
    check(pairs.get('status')=='PASS' and len(pairs['pairs'])==3,'models','training pair identity missing')
    for pair in pairs['pairs']:
        check(pair['paired_initialization_exact'] and pair['paired_data_and_training_budget_exact'] and pair['optimizer_steps_per_arm']==228,'models','training pairing mismatch')
    parities=list((training/'parity').glob('seed*/*/parity.json'))
    check(len(parities)==6,'models','missing actual-checkpoint Python/C++ parity')
    for path in parities:
        v=load(path);check(v.get('status')=='PASS' and v['events']==2000 and v['action_mismatch_count']==0,str(path),'actual checkpoint parity failed')
    result=dict(status='FAIL' if failures else 'INCOMPLETE' if pending else 'PASS',planned_test_rows=len(matrix),
      validated_test_rows=len(passed),pending=pending,failures=failures,validated_runs=passed,
      training_pairs_status=pairs.get('status'),checkpoint_parities=len(parities),
      training_evidence_root=str(training),
      denominator_note='Actual ROI instructions/cycles; actual common cache-warmup origins retained; DRAM bytes derive only from actual bus transactions.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','planned_test_rows','validated_test_rows','checkpoint_parities')}))
    if failures or (a.expect_complete and pending):raise SystemExit(1)


if __name__=='__main__':main()
