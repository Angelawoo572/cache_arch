"""Check grant-for-grant native choices, join first uses, and measure fixed-work costs."""
import argparse
import csv
import gzip
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from run import BASE,DATA,HERE,MODES

def rows(p,name):
    file=p/(name+'.csv')
    with (file.open() if file.is_file() else gzip.open(str(file)+'.gz','rt')) as f:
        yield from csv.reader(f)

def records(p,name):
    source=rows(p,name)
    keys=next(source)
    for row in source:yield dict(zip(keys,row))

def lookups(p):
    d={}
    src=rows(p,'loads')
    next(src)
    for c,i,o,pc,vl,pl,level,stage,t,hit,*rest in src:
        if level=='LLC' and stage=='lookup':
            d[int(i)*8+int(o)]=(int(pc),int(vl),int(hit))
    return d

def check_shadow():
    shadow=DATA/'runs/shadow'
    base=BASE/'gap_bfs3_late-stride'
    for file in ('cores.csv','stats.csv'):
        if (shadow/file).read_bytes()!=(base/file).read_bytes():raise ValueError('changed baseline '+file)
    with (DATA/'shadow_reference.csv').open() as f:reference=list(csv.DictReader(f))
    observed=list(records(shadow,'decisions'))
    if len(reference)!=len(observed) or not reference:raise ValueError('grant count mismatch')
    for expect,actual in zip(reference,observed):
        if any(actual[k]!=v for k,v in expect.items()):raise ValueError((expect,actual))
    return len(reference)

def arm(mode):
    base=BASE/'gap_bfs3_late-nopf'
    stride=BASE/'gap_bfs3_late-stride'
    out=base if mode=='nopf' else (DATA/'runs'/mode)
    base_reads=lookups(base)
    stride_reads=lookups(stride)
    now=lookups(out)
    for k in now.keys()&(base_reads.keys()|stride_reads.keys()):
        ref=base_reads[k] if k in base_reads else stride_reads[k]
        if now[k][:2]!=ref[:2]:raise ValueError('unaligned load '+str(k))
    c=Counter()
    s={r['key']:int(r['value']) for r in records(out,'stats')}
    choices={} if mode=='nopf' else {(int(r['instr']),int(r['operand'])):r for r in records(out,'decisions')}
    if len(choices) not in (0,1460):raise ValueError('incomplete or repeated granted triggers')
    for a in choices.values():
        c[a['action']]+=1
        if int(a['prior_age'])>=0:c['prior_seen']+=1
    candidate={}
    stage=Counter()
    for r in records(out,'candidates'):
        pid=int(r['id'])
        if r['stage']=='proposed':
            anch=(int(r['instr_id']),int(r['operand']))
            choice=choices.get(anch)
            kind='native' if not choice or choice['action']=='none' else choice['action']
            if kind!='native' and int(r['target'])!=int(choice['chosen']):raise ValueError('wrong issued target')
            if (kind=='replace')!=(r['reason']=='intervention'):raise ValueError('wrong origin')
            candidate[pid]=kind
        if pid not in candidate:continue
        kind=candidate[pid];stage[kind,r['stage']]+=1
        if r['stage']=='rejected':stage['reject_'+kind,r['reason']]+=1
        if r['stage']=='timely':
            key=8*int(r['demand_instr_id'])+int(r['demand_operand'])
            if (kind=='replace' and key in stride_reads and stride_reads[key][-1]==0 and key in now and now[key][-1]):
                c['new_direct']+=1
    baseline=int(next(records(stride,'cores'))['cycles'])
    cycles=int(next(records(out,'cores'))['cycles'])
    common=now.keys()&stride_reads.keys()
    c['gains']=sum(now[k][-1] and not stride_reads[k][-1] for k in common)
    c['losses']=sum(not now[k][-1] and stride_reads[k][-1] for k in common)
    c['nopf_gains']=sum(now[k][-1] and not base_reads[k][-1] for k in now.keys()&base_reads.keys())
    both=now.keys()^stride_reads.keys()
    outrow=dict(mode=mode,cycles=cycles,cycles_saved_vs_stride=baseline-cycles,
                roi_time_saved_pct=round((baseline-cycles)*100/baseline,5),dram_bytes=sum(s['dram_'+k+'_bytes'] for k in ('demand','prefetch','write')),
                rob_load_head_cycles=s['head_unique_cycles_core_0'],grants=len(choices),page_prior_hits=c['prior_seen'],agreements=c['agree'],
                replacements=c['replace'],aux_accepted=stage['replace','accepted'],aux_timely=stage['replace','timely'],aux_late=stage['replace','late'],
                aux_unused_evicted=stage['replace','evicted'],agree_accepted=stage['agree','accepted'],agree_timely=stage['agree','timely'],
                aux_timely_for_stride_miss=c['new_direct'],aux_resident=stage['reject_replace','resident'],aux_mshr=stage['reject_replace','mshr'],
                total_accepted=s['prefetch_accepted'],common_stride_lookups=len(common),lookup_only_one_arm=len(both),stride_miss_to_hit=c['gains'],
                stride_hit_to_miss=c['losses'],nopf_miss_to_hit=c['nopf_gains'])
    return outrow

def main(jobs):
    n=check_shadow()
    with ProcessPoolExecutor(jobs) as ex:out=list(ex.map(arm,('nopf',*MODES)))
    with (HERE/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=out[0],lineterminator='\n');w.writeheader();w.writerows(out)
    print(n,'shadow grants agree across three rules')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',type=int,default=4)
    main(ap.parse_args().jobs)
