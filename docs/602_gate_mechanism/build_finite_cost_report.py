#!/usr/bin/env python3
"""Read-only full-ROI finite-cost report; the four-row scope is seed7 only."""
import argparse
import json
import math
from pathlib import Path


def timing(v,name,total,denominator,maximum=None):
    histogram=v['distributions_log2_upper_bound'][name]
    count=sum(histogram.values())
    assert count==v[denominator]
    result=dict(count=count,denominator_field=denominator,mean=v[total]/count,
                histogram_upper_bounds=histogram)
    if maximum:result['maximum']=v[maximum]
    for quantile in (.5,.95,.99):
        rank=math.ceil(count*quantile);cumulative=0;lower=0
        for upper,n in sorted((int(key),value) for key,value in histogram.items()):
            cumulative+=n
            if cumulative>=rank:
                result['p'+str(int(quantile*100))+'_bounds']=[lower,upper];break
            lower=upper+1
    return result


def region(raw,roi):
    rows=[];run_records=[];summaries=[]
    for filtered in (0,1):
        folder=raw/'test'/f'test_602_r{roi}_nn_s7_m4_b2363_f{filtered}_sil0'
        run=json.loads((folder/'run.json').read_text());assert run['status']=='complete'
        v=json.loads((folder/'summary.json').read_text());run_records.append(run);summaries.append(v)
        keys=['instructions','cycles','P','request_gate_generated','request_gate_allowed',
              'request_gate_current_suppressed','request_gate_budget_suppressed',
              'request_cache_enqueued','request_cache_merged','request_cache_rejected',
              'eligible_l2_callbacks','nn_admitted','nn_started','nn_completed','inference_dropped',
              'state_evictions','warmup_state_evictions','input_queue','inference_inflight',
              'state_entries','peak_state_entries','peak_input_queue','output_queue','peak_output_queue',
              'output_dropped_addresses','inference_macs','inference_weight_read_bytes',
              'generated_addresses','mechanism_raw_generated_addresses','mechanism_payload_generated_addresses',
              'cohort_timely','cohort_late','cohort_unused','cohort_redundant_cache',
              'cohort_redundant_inflight','dram_total_bus_bytes']
        row={key:v.get(key) for key in keys}
        expected_controls=dict(h=8,macs_per_cycle=4,weight_bandwidth_bytes_per_cycle=16,
            nonlinear_cycles_per_operation=4,nonlinear_units=1,state_capacity=64,
            input_queue_capacity=16,output_queue_capacity=32,history_instructions=5000000,
            warmup_instructions=5000000,weight_bytes=7632,nn_weight_dram_bytes=0,
            service_architecture='serial_II_equals_service',queue_policy='fifo',page_policy='same_page',
            engine_method='frozen',engine_payload_mode='learned',decoded_result_bytes_configured=512,
            keep_ppm=1000000,max_requests_per_event=0,silent_suppress=0,min_emit_margin=0)
        assert run['environment']['DYNAMIC602_REQUESTS_PER_MILLION_INSTRUCTIONS']=='2363'
        assert all(v.get(key)==value for key,value in expected_controls.items())
        row['actual_engine_configuration']={key:v[key] for key in expected_controls}
        row.update(roi=roi,seed=7,filtered=filtered,summary=str(folder/'summary.json'),
                   IPC=v['instructions']/v['cycles'],drop_fraction=v['inference_dropped']/v['eligible_l2_callbacks'],
                   completed_service_busy_fraction_lower_bound=v['inference_completed_service_cycles_sum']/v['cycles'])
        assert v['nn_started']==v['nn_completed']+v['inference_inflight']
        assert v['eligible_l2_callbacks']==v['nn_admitted']+v['inference_dropped']
        assert v['nn_admitted']==v['nn_started']+v['input_queue']
        row['queue_wait_cycles']=timing(v,'queue_wait_cycles','inference_wait_cycles_sum','nn_started','inference_wait_cycles_max')
        row['scheduled_service_cycles']=timing(v,'scheduled_service_cycles','inference_service_cycles','nn_started')
        row['completed_service_cycles']=timing(v,'completed_service_cycles','inference_completed_service_cycles_sum','nn_completed','inference_completed_service_cycles_max')
        row['completion_age_cycles']=timing(v,'completion_age_cycles','inference_completion_cycles_sum','nn_completed','inference_completion_cycles_max')
        rows.append(row)
    a,b=run_records
    differences={key:[a['environment'].get(key),b['environment'].get(key)]
                 for key in a['environment'].keys()|b['environment'].keys()
                 if a['environment'].get(key)!=b['environment'].get(key)}
    functional={key:value for key,value in differences.items() if key not in {'DYNAMIC602_STATS','DYNAMIC602_HISTORY_LOG'}}
    assert functional=={'DYNAMIC602_FILTER_CURRENT':['0','1']}
    assert a['command']==b['command']
    warmup_fields=['warmup_completed_updates','warmup_state_evictions','warmup_discarded_work',
                   'warmup_discarded_outputs','pc_warmup_updates']
    warmup_equal={key:summaries[0].get(key)==summaries[1].get(key) for key in warmup_fields}
    assert all(warmup_equal.values())
    path=raw/'test'/f'test_602_r{roi}_none_s7_m0_b0_f0_sil0/summary.json'
    n=json.loads(path.read_text());nopf=dict(summary=str(path),instructions=n['instructions'],
                                           cycles=n['cycles'],IPC=n['instructions']/n['cycles'])
    first,second=rows
    return dict(roi=roi,test_records=[[25000000,50000000],[75000000,100000000]][roi],
                rows=rows,NoPF=nopf,
                delta_filtered_vs_original=dict(IPC_fraction=second['IPC']/first['IPC']-1,
                    allowed_request_fraction=second['request_gate_allowed']/first['request_gate_allowed']-1,
                    allowed_requests=second['request_gate_allowed']-first['request_gate_allowed'],
                    cycles=second['cycles']-first['cycles']),
                speedup_vs_NoPF={str(row['filtered']):row['IPC']/nopf['IPC']-1 for row in rows},
                pair_control_checks=dict(commands_equal=True,only_functional_environment_difference=functional,
                                         warmup_evidence_equal=warmup_equal))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    regions=[region(args.raw.resolve(),roi) for roi in (0,1)]
    report=dict(status='verified_formal_rows',protocol='gate_mechanism_20261016_v1',
        scope='Four full25M formal rows: seed7 balanced only, two independent602 ROIs, MAC4,B16,nonlinear4,FIFOinput16/output32,state64,history5M,learned payload,validation-fixed cap2363; current filter0/1.',
        completed=4,regions=regions,
        notes=['No finite-cost seed17/27 filtering comparison is claimed by this subset.',
          'All timing aggregates include the fullROI; bounded example logs are not used. Quantile intervals are exact histogram bucket bounds, not precise percentile point estimates.',
          'Queue-wait and scheduled-service means divide by nn_started; completed-service and completion-age means divide by nn_completed.',
          'Scheduled service includes an unfinished final inference. Completed-service/ROIcycles is only a lower bound on engine busy fraction.',
          'Identical per-event cost policy does not require identical aggregate work: cache feedback changes event timing/admission/h/c paths.',
          'Dense weight reads=MAC*4 excludes bias loads by model assumption; resident weights create zero modeled DRAM traffic. RealDRAM bytes are independently measured.',
          'Source-level cost controls remain as verified in cost_invariants.json. No architecture, online training or canonical h/c initialization is introduced.'])
    (out/'finite_currentline.json').write_text(json.dumps(report,indent=2)+'\n')
    for r in regions:
        (out/f'finite_currentline_roi{r["roi"]}.json').write_text(json.dumps(r,indent=2)+'\n')
    invariants=out/'cost_invariants.json'
    if invariants.exists():
        value=json.loads(invariants.read_text())
        value['four_finite_formal_pair_controls']={str(r['roi']):r['pair_control_checks'] for r in regions}
        value['four_finite_formal_scope']='seed7 only; both independentROIs; cap2363; full25M each; currentfilter0/1'
        invariants.write_text(json.dumps(value,indent=2)+'\n')
    lines=['# Finite engine: current-line filtering','',
      'The four rows below are the seed 7 balanced-checkpoint subset. Each uses a full 25M measured-instruction ROI, 5M common cache warmup/history, MAC4, resident-weight bandwidth 16B/cycle, one nonlinear unit at 4 cycles/operation, FIFO input16/output32, state64, learned payload and the validation-fixed cap of 2363 requests/million instructions. They retain the original v2 state and service policy. Finite-cost filtering for seeds 17/27 is outside this subset.','',
      '| ROI | Arm | IPC | Cycles | Allowed requests | Current suppressed | Budget suppressed | Input dropped | State evictions |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in regions:
        for row in r['rows']:
            lines.append(f'| {r["roi"]}: {r["test_records"][0]//1000000}–{r["test_records"][1]//1000000}M | {"filtered" if row["filtered"] else "original"} | {row["IPC"]:.8f} | {row["cycles"]:,} | {row["request_gate_allowed"]:,} | {row["request_gate_current_suppressed"]:,} | {row["request_gate_budget_suppressed"]:,} | {row["drop_fraction"]:.4%} | {row["state_evictions"]} |')
    lines+=['','| ROI | Arm | Queue wait mean | Completed service mean | Completion age mean | Queue wait max | Completion age max | Completed-service busy lower bound |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in regions:
        for row in r['rows']:
            lines.append(f'| {r["roi"]} | {"filtered" if row["filtered"] else "original"} | {row["queue_wait_cycles"]["mean"]:.3f} | {row["completed_service_cycles"]["mean"]:.3f} | {row["completion_age_cycles"]["mean"]:.3f} | {row["queue_wait_cycles"]["maximum"]:,} | {row["completion_age_cycles"]["maximum"]:,} | {row["completed_service_busy_fraction_lower_bound"]:.4%} |')
    lines+=['','Times are simulated cycles. Means, maxima and complete log2 histogram counts use all started/completed events. Quantile intervals are bucket bounds, not exact percentile points. The denominator is `nn_started` for queue wait/scheduled service, and `nn_completed` for completed service/completion age. The scheduled-service sum may exceed ROI cycles because it includes the final unfinished inference; the busy fraction above excludes service already spent on that final inference.','']
    for r in regions:
        a,b=r['rows'];delta=r['delta_filtered_vs_original']
        lines.append(f'ROI {r["roi"]}: the filter changes IPC by {delta["IPC_fraction"]:+.5%} and requests by {delta["allowed_request_fraction"]:+.3%}. NoPF IPC is {r["NoPF"]["IPC"]:.8f}; original/filtered changes against NoPF are {r["speedup_vs_NoPF"]["0"]:+.5%}/{r["speedup_vs_NoPF"]["1"]:+.5%}. Eligible LOAD counts are {a["eligible_l2_callbacks"]:,}/{b["eligible_l2_callbacks"]:,}; inference starts are {a["nn_started"]:,}/{b["nn_started"]:,}, completions {a["nn_completed"]:,}/{b["nn_completed"]:,}. Both finish with {a["input_queue"]}/{b["input_queue"]} queued inputs and {a["inference_inflight"]}/{b["inference_inflight"]} active inference(s).')
        lines.append('')
    lines+=['The [source certificate](cost_invariants.json) verifies that `latency` and `begin` are byte-identical to v2. Removing only the diagnostic request-context call makes `service` identical too. The original learned decoder count after the existing cap32 determines MAC/nonlinear/scratch/service cost, before final payload selection. One active inference holds h/c and output until completion; saturated initiation interval equals service time. Unconditional outer-cycle service continues without new LOAD callbacks.','',
      'Current-line filtering runs after output-queue removal and before page/budget credits. The pop and output initiation interval still occur for suppressed calls, so filtering directly saves requests and credits, rather than original forward computation or output-drain cycles. Both within-ROI pairs have identical simulator commands and functional environments except the filter0/1 flag; warmup completion, eviction, pending-work cleanup and per-PC warmup-update evidence match. No canonical h/c initialization was transplanted. Differences in aggregate work arise from simulation feedback, not a changed service formula.','',
      'The present finite engine remains almost continuously busy, drops most eligible inputs and incurs long queue delays. These four rows show request reductions with very small IPC changes under the stated finite model. They do not establish a robust finite-cost performance benefit across training seeds. MAC×4 weight-read accounting omits bias traffic by assumption; resident weights produce no modeled DRAM transactions, while actual cache/DRAM bus transfers are measured separately. No lookup/energy/area synthesis is claimed.','',
      'The [combined JSON](finite_currentline.json) preserves all four raw summaries, control checks, timing histograms, means, maxima and denominators. The per-ROI JSON files are retained as subsets. Rebuild this report with `python build_finite_cost_report.py --raw /path/to/gate_output/raw`.']
    (out/'FINITE_CURRENTLINE.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({str(r['roi']):dict(delta=r['delta_filtered_vs_original'],values=[dict(filtered=x['filtered'],IPC=x['IPC'],P=x['P'],queuewait=x['queue_wait_cycles']['mean'],service=x['completed_service_cycles']['mean'],age=x['completion_age_cycles']['mean'],drop=x['drop_fraction']) for x in r['rows']]) for r in regions},indent=2))

if __name__=='__main__':main()
