#!/usr/bin/env python3
"""Compact tables from the complete, independently audited mechanism matrix."""
import argparse,csv,json,statistics
from collections import Counter,defaultdict
from pathlib import Path

def write(path, rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,required=True);p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    audit=json.loads((a.analysis/'audit.json').read_text())
    assert audit['status']=='PASS' and audit['completed']==71 and not audit['failures']
    a.out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((a.analysis/'metrics_per_seed.csv').open()))
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in ['roi','arm','macs','budget','filter_current','silent'])].append(r)
    aggregate=[]
    for key,group in sorted(groups.items()):
        d=dict(zip(['roi','arm','macs','budget','filter_current','silent'],key));d['seeds']=','.join(r['seed'] for r in group);d['n']=len(group)
        for field in ['ipc','relative_ipc_vs_nopf','requests_per_million','dram_total_bus_bytes','l2_load_mpki','l2_load_miss_reduction','l2_target_timely_coverage','l2_target_timeliness','l2_target_useful_accuracy_enqueued','l2_target_tracked_load_uses_per_submission','l2_target_useful_accuracy_resolved']:
            values=[float(r[field]) for r in group if r.get(field) not in ('',None)]
            if values:
                for suffix,value in [('mean',statistics.mean(values)),('min',min(values)),('max',max(values))]:d[field+'_'+suffix]=value
        aggregate.append(d)
    write(a.out/'aggregate.csv',aggregate)
    payload=[];credits=[];classified=[]
    for r in rows:
        if r['execution']!='fresh':continue
        c=json.loads((a.raw/'test'/r['run_id']/'summary.json').read_text())
        ident={k:r[k] for k in ['run_id','roi','arm','seed','macs','budget','filter_current','silent']}
        categories=defaultdict(Counter)
        for x in c.get('mechanism_classified_requests',[]):
            target=categories[(x['slot'],x['current_line'],x['list_duplicate'],x['teacher_category'],x['fill_scope'])]
            for field in ['cache_attempted','allowed','current_suppressed','page_suppressed','budget_suppressed','cache_accepted_nonmerged','cache_merged','cache_rejected','pq_enqueued']:target[field]+=x[field]
            if x['l2_cohort']:
                for field,value in x['l2_cohort'].items():
                    if isinstance(value,(int,float)):target['l2_'+field]+=value
        for key,count in categories.items():classified.append(dict(ident,slot=key[0],current_line=key[1],list_duplicate=key[2],teacher_category=key[3],fill_scope=key[4],**count))
        if r['arm']=='nn':
            current=sum(v['allowed'] for k,v in categories.items() if k[1]==1)
            credits.append(dict(ident,allowed=c['request_gate_allowed'],current_allowed=current,current_credit_fraction=current/c['request_gate_allowed'] if c['request_gate_allowed'] else None,current_suppressed=c['request_gate_current_suppressed'],raw_generated=c['mechanism_raw_generated_addresses'],raw_current=c['mechanism_raw_current_line_addresses'],raw_list_duplicates=c['mechanism_raw_list_duplicate_addresses']))
        if r['arm']=='nn' and r['macs']=='0' and r['budget']=='0' and r['filter_current']=='0' and r['silent']=='0':
            hist=defaultdict(Counter)
            for name,bins in c['mechanism_distributions_exact'].items():
                phase,dist=name.split('_',2)[1:]
                for value,count in bins.items():hist[dist][int(value)]+=count
            emit=hist['raw_emit'][1];raw=c['mechanism_raw_generated_addresses'];slot1=sum(hist['raw_delta_slot_1'].values())
            payload.append(dict(ident,completed=c['nn_completed'],emit_events=emit,emit_fraction=emit/c['nn_completed'],k2_events=hist['raw_K'][2],k2_fraction_of_emit=hist['raw_K'][2]/emit if emit else None,raw_generated=raw,current_line=c['mechanism_raw_current_line_addresses'],current_fraction=c['mechanism_raw_current_line_addresses']/raw if raw else None,list_duplicates=c['mechanism_raw_list_duplicate_addresses'],slot0_zero=hist['raw_delta_slot_0'][0],slot0_total=sum(hist['raw_delta_slot_0'].values()),slot1_plus2=hist['raw_delta_slot_1'][2],slot1_total=slot1,slot1_plus2_fraction=hist['raw_delta_slot_1'][2]/slot1 if slot1 else None,raw_K=json.dumps(dict(hist['raw_K']),sort_keys=True),slot1_delta=json.dumps(dict(hist['raw_delta_slot_1']),sort_keys=True)))
    write(a.out/'learned_payload_full_roi.csv',payload);write(a.out/'current_line_credits.csv',credits);write(a.out/'classified_whole_roi.csv',classified)
    pairs=list(csv.DictReader((a.analysis/'paired_comparisons.csv').open()));grouped=defaultdict(list)
    for r in pairs:grouped[tuple(r[k] for k in ['comparison','roi','macs','budget'])].append(r)
    pairs_out=[]
    for key,group in sorted(grouped.items()):
        d=dict(zip(['comparison','roi','macs','budget'],key));d.update(n=len(group),seeds=','.join(r['seed'] for r in group))
        for field in ['relative_ipc_change_pct','delta_requests_per_million','delta_dram_bus_bytes']:
            v=[float(r[field]) for r in group]
            d.update({field+'_mean':statistics.mean(v),field+'_min':min(v),field+'_max':max(v)})
        pairs_out.append(d)
    write(a.out/'paired_ranges.csv',pairs_out)
    table=['| ROI | cap/M | arm | MAC | filter | n | IPC mean [min, max] | gain vs NoPF (%) | requests/M | DRAM MiB |','|---|---:|---|---:|---:|---:|---|---|---|---|']
    for r in aggregate:
        def value(field,scale=1,offset=0,places=3):
            values=[r.get(field+'_'+s) for s in ('mean','min','max')]
            if any(v is None for v in values):return 'NA'
            v=[scale*x+offset for x in values]
            return f'{v[0]:.{places}f}' if v[1]==v[2] else f'{v[0]:.{places}f} [{v[1]:.{places}f}, {v[2]:.{places}f}]'
        arm=r['arm']+(' silent-suppressed' if r['silent']=='1' else '')
        table.append('| '+' | '.join([r['roi'],r['budget'],arm,r['macs'],r['filter_current'],str(r['n']),value('ipc',places=6),value('relative_ipc_vs_nopf',100,-100),value('requests_per_million',places=2),value('dram_total_bus_bytes',1/2**20,places=3)])+' |')
    (a.out/'RESULT_TABLE.md').write_text('\n'.join(table)+'\n')
    quality=['Observed LOAD usefulness only. Accuracy = (timely_LOAD + late_LOAD) / tracked L2-target PQ insertions; timely coverage = timely_LOAD / matched NoPF L2 LOAD misses; timeliness = timely_LOAD / (timely_LOAD + late_LOAD). Pending and resident-unused remain right-censored. SPP rows describe its L2-target subset; LLC-only accuracy/timeliness are NA. Means/ranges retain individual training seeds; they are not confidence intervals.','',
             '| ROI | cap/M | arm | MAC | filter | timely coverage (%) | observed LOAD accuracy (%) | LOAD timeliness (%) |',
             '|---|---:|---|---:|---:|---|---|---|']
    for r in aggregate:
        values=[]
        for field in ['l2_target_timely_coverage','l2_target_useful_accuracy_enqueued','l2_target_timeliness']:
            vv=[r.get(field+'_'+s) for s in ['mean','min','max']]
            if any(x is None for x in vv):values.append('NA');continue
            values.append(f'{100*vv[0]:.3f}' if vv[1]==vv[2] else f'{100*vv[0]:.3f} [{100*vv[1]:.3f}, {100*vv[2]:.3f}]')
        arm=r['arm']+(' silent-suppressed' if r['silent']=='1' else '')+(' (L2 subset)' if r['arm']=='spp' else '')
        quality.append('| '+' | '.join([r['roi'],r['budget'],arm,r['macs'],r['filter_current']]+values)+' |')
    (a.out/'QUALITY_TABLE.md').write_text('\n'.join(quality)+'\n')
    print(json.dumps({'status':'PASS','logical_rows':len(rows),'aggregate_rows':len(aggregate),'payload_full_roi_rows':len(payload),'classified_whole_roi_rows':len(classified)}))

if __name__=='__main__':main()
