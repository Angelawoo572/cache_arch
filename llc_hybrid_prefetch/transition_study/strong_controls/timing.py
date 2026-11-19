"""Compare same-consumer lead and service without aligning absolute replay clocks."""
import csv
import gzip
from pathlib import Path
import numpy as np
from run import HERE,ORIGINAL

def records():
    with gzip.open(ORIGINAL/'loads.csv.gz','rb') as f:
        for row in f:
            if b',LLC,ingress,' in row:
                fields=row.split(b',',3)
                yield int(fields[1]),int(fields[2])

def save(file,rows):
    with (HERE/file).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
    reference={k:i for i,k in enumerate(records())}
    with (HERE/'uncovered_targets.csv').open() as f:
        targets=[r for r in csv.DictReader(f) if r['lbm_exact_outcome']=='timely' and r['lbm_exact_origin']=='auxiliary']
    data=[]
    for r in targets:
        if r['spp_outcome']!='late':raise ValueError('unexpected witness')
        q=int(r['instr_id']),int(r['operand'])
        row=dict(instr_id=q[0],operand=q[1],line=int(r['physical_line']),baseline_consumer_ingress=reference[q])
        for policy,prefix in (('lbm_exact','exact'),('spp','spp')):
            field=lambda k:int(r[policy+'_'+k])
            a=field('anchor'),field('anchor_op')
            row[prefix+'_producer_instr']=a[0];row[prefix+'_producer_op']=a[1]
            row[prefix+'_producer_stride_ingress_rank']=reference.get(a,'')
            t={k:field(k) for k in ('generated','dispatch','filled','ingress','tag')}
            if not t['generated']<=t['dispatch']<=t['filled']:raise ValueError('impossible request chronology')
            for k,v in t.items():row[prefix+'_'+k]=v
            row[prefix+'_issue_to_demand']=t['ingress']-t['generated']
            row[prefix+'_issue_to_dispatch']=t['dispatch']-t['generated']
            row[prefix+'_service']=t['filled']-t['dispatch']
            row[prefix+'_slack_ingress']=t['ingress']-t['filled']
            row[prefix+'_slack_lookup']=t['tag']-t['filled']
            row[prefix+'_head']=field('head')
        a=(row['exact_producer_instr'],row['exact_producer_op'])
        b=(row['spp_producer_instr'],row['spp_producer_op'])
        row['exact_earlier_program_order']=(-1 if a<b else 1 if a>b else 0)
        ia,ib=reference.get(a),reference.get(b)
        row['exact_earlier_stride_ingress_order']=('' if ia is None or ib is None else -1 if ia<ib else 1 if ia>ib else 0)
        data.append(row)
    summary=[]
    for metric in ('issue_to_demand','issue_to_dispatch','service','slack_ingress','slack_lookup','head'):
        left=np.array([v['exact_'+metric] for v in data]);right=np.array([v['spp_'+metric] for v in data])
        def percentile(v):return np.percentile(v,[25,50,75]).tolist()
        d=left-right
        parts=dict(zip(('exact_p25','exact_p50','exact_p75','spp_p25','spp_p50','spp_p75','paired_diff_p25','paired_diff_p50','paired_diff_p75'),[*percentile(left),*percentile(right),*percentile(d)]))
        summary.append(dict(metric=metric,matched_operands=len(data),**parts,exact_greater=int((d>0).sum()),equal=int((d==0).sum()),exact_smaller=int((d<0).sum())))
    save('timing_targets.csv',data);save('timing_summary.csv',summary)
    from collections import Counter
    print('exact relative to SPP, -1 earlier, +1 later, blank unknown')
    for k in ('exact_earlier_program_order','exact_earlier_stride_ingress_order'):print(k,Counter(v[k] for v in data))
if __name__=='__main__':main()
