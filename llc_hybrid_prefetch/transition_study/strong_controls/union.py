"""Classify each sparse success outside the four full-rate conventional direct-first-use sets."""
import csv
import gzip
from bisect import bisect_left,bisect_right
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
from endpoints import claims,read,HERE,RAW
STRONG=('spp','multi','xpc','temporal')
SPARSE=('lbm_exact','lbm_page')

def classify(item):
    name,p,targets=item
    lines=set(targets.values())
    generation={};uses={};events=defaultdict(list);lookup={};ingress={};last={};head=Counter();stages=defaultdict(dict)
    for r in read(p/'candidates.csv.gz'):
        line=int(r['target'])
        if line not in lines:continue
        stage=r['stage'];pid=int(r['id']);cycle=int(r['cycle'])
        events[line].append((cycle,stage,r['reason']))
        stages[pid][stage]=cycle
        if stage=='proposed':generation[pid]=r
        if stage in ('timely','late') and int(r['demand_cpu'])==0:
            k=(int(r['demand_instr_id']),int(r['demand_operand']))
            if k in targets:
                if targets[k]!=line:raise ValueError('wrong first-use address')
                uses[k]=(stage,pid)
    with gzip.open(p/'loads.csv.gz','rb') as f:
        for row in f:
            if b',LLC,lookup,' not in row and b',LLC,ingress,' not in row:continue
            a=row.split(b',',10)
            line=int(a[5])
            if line not in lines:continue
            if int(a[0])!=0:raise ValueError('expected solo replay')
            k=(int(a[1]),int(a[2]));t=int(a[8])
            if a[7]==b'ingress':
                if k in targets:ingress[k]=t
                continue
            if k in targets:
                if targets[k]!=line:raise ValueError('mismatching dynamic address')
                lookup[k]=(int(a[9]),t,last.get(line,-1),int(a[3]),int(a[4]))
            last[line]=t
    for r in read(p/'head.csv.gz'):
        k=(int(r['instr_id']),int(r['operand']))
        if k in targets:head[k]+=int(r['cycles'])
    result={}
    for k,line in targets.items():
        h,t,prev,pc,va=lookup.get(k,(-1,-1,-1,'',''))
        stage,pid=uses.get(k,('',0))
        if stage=='timely' and h!=1 or stage=='late' and h!=0:raise ValueError('first-use tag mismatch')
        classification=stage or ('ordinary_hit' if h==1 else 'persistent' if h==0 else 'no_tag')
        ev=events[line];cycles=[r[0] for r in ev]
        if any(a>b for a,b in zip(cycles,cycles[1:])):raise ValueError('nonmonotonic target history')
        selected=ev[bisect_right(cycles,prev):bisect_left(cycles,t)] if t>=0 else []
        g=generation.get(pid,{})
        result[k]=dict(outcome=classification,head=head[k],tag=t,ingress=ingress.get(k,''),prior_proposal=int(any(s=='proposed' for _,s,_ in selected)),
                       last_action=selected[-1][1]+'_'+selected[-1][2] if selected else 'none',
                       origin=g.get('reason',''),anchor=g.get('instr_id',''),anchor_op=g.get('operand',''),anchor_pc=g.get('pc',''),
                       anchor_line=g.get('base',''),generated=g.get('cycle',''),dispatch=stages[pid].get('dispatch',''),filled=stages[pid].get('fill',''),
                       claim_cycle=stages[pid].get(stage,''),pc=pc,va=va)
    return name,result

def main():
    e=HERE.parent/'raw/episodes/lbm-r2/episode_ids.csv.gz'
    original={(int(r['first_miss_instr']),int(r['first_miss_op'])):int(r['target_line']) for r in read(e)}
    paths={n:RAW/'verified'/('open-'+n) for n in STRONG}
    paths.update({n:HERE.parent/'raw/replay/runs'/('lbm-r2__'+n) for n in SPARSE})
    useful={n:claims(p,original) for n,p in paths.items()}
    covered=set().union(*(useful[n] for n in STRONG))
    chosen=set().union(*(useful[n]-covered for n in SPARSE))
    targets={k:original[k] for k in sorted(chosen)}
    with ThreadPoolExecutor(4) as exe:allarms=dict(exe.map(classify,[(n,p,targets) for n,p in paths.items()]))
    summary=[]
    for n in SPARSE:
        left=useful[n]-covered
        actual=sum(allarms[n][k]['origin']=='auxiliary' for k in left)
        every_miss=sum(all(allarms[m][k]['outcome'] in ('persistent','late') for m in STRONG) for k in left)
        summary.append(dict(rule=n,original_failure_operands=len(original),strong_union_direct=len(covered),
                rule_direct=len(useful[n]),already_in_strong_union=len(useful[n]&covered),direct_outside_union=len(left),
                outside_with_auxiliary_source=actual,outside_with_stride_source=len(left)-actual,outside_miss_in_all_four_tags=every_miss))
    rows=[]
    for k,line in targets.items():
        r=dict(instr_id=k[0],operand=k[1],physical_line=line,pc=allarms['lbm_exact'][k]['pc'],virtual_line=allarms['lbm_exact'][k]['va'])
        for n in (*STRONG,*SPARSE):
            x=allarms[n][k];fields=('outcome','prior_proposal','last_action','head','anchor','anchor_op','anchor_pc','anchor_line','generated','dispatch','filled','ingress','tag','claim_cycle') if n in STRONG else ('outcome','origin','anchor','anchor_op','anchor_pc','anchor_line','generated','dispatch','filled','ingress','tag','claim_cycle','head')
            for field in fields:r[n+'_'+field]=x[field]
        rows.append(r)
    for path,data in [(HERE/'union_results.csv',summary),(HERE/'uncovered_targets.csv',rows)]:
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=data[0],lineterminator='\n');w.writeheader();w.writerows(data)
    print(summary)
if __name__=='__main__':main()
