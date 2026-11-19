"""Recover the registered LBM witness's ingress, tag, first-use, dispatch and fill times."""
import csv
import gzip
from concurrent.futures import ProcessPoolExecutor
from run import HERE,MODES,RAW
WITNESSES=((12657217,552063,4201648,2189100332159),(10399777,451157,4201648,2189100319189))
def raw(folder,name):
    with gzip.open(folder/(name+'.csv.gz'),'rt') as f:yield from csv.reader(f)
def one(arg):
    setting,mode,identity=arg;folder=RAW/'verified'/(setting+'-'+mode)
    target,line,pc,va=identity
    levels={}
    rd=raw(folder,'loads');next(rd)
    for r in rd:
        if r[0]=='0' and r[1]==str(target) and r[2]=='0' and r[6]=='LLC':
            if (int(r[3]),int(r[4]),int(r[5]))!=(pc,va,line):raise ValueError('witness identity mismatch')
            if r[7] in ('ingress','lookup'):levels[r[7]]=(int(r[8]),int(r[9]))
    hit=levels['lookup'][1]
    rd=raw(folder,'candidates');cols=next(rd);tx={};claim=None;target_evidence=[]
    for vals in rd:
        if vals[6]!=str(line):continue
        r=dict(zip(cols,vals));t=int(r['cycle'])
        if t<levels['lookup'][0]:target_evidence.append((t,r['stage'],r['reason']))
        steps=tx.setdefault(r['id'],{})
        steps[r['stage']]=r
        if r['stage'] in ('timely','late') and (r['demand_cpu'],r['demand_instr_id'],r['demand_operand'])==('0',str(target),'0'):
            if claim is not None:raise ValueError('multiple claimants')
            claim=(r['id'],r['stage'])
    head=0;rd=raw(folder,'head');next(rd)
    for c,i,o,t in rd:
        if (c,i,o)==('0',str(target),'0'):head+=int(t)
    name=claim[1] if claim else ('ordinary_hit' if hit else 'miss_without_direct_claim')
    stages=tx.get(claim[0],{}) if claim else {}
    g=stages.get('proposed',{})
    prior=target_evidence[-1] if target_evidence else ('','','')
    return dict(setting=setting,policy=mode,target_instr=target,target_operand=0,pc=pc,physical_line=line,virtual_line=va,
                queue_ingress=levels['ingress'][0],tag_lookup=levels['lookup'][0],tag_hit=hit,class_at_target=name,retirement_head_cycles=head,
                claim_anchor_instr=g.get('instr_id',''),claim_anchor_operand=g.get('operand',''),claim_generation=g.get('cycle',''),
                claim_dispatch=stages.get('dispatch',{}).get('cycle',''),claim_fill=stages.get('fill',{}).get('cycle',''),
                last_target_action_before_lookup=prior[1],last_action_cycle=prior[0],last_action_reason=prior[2])
if __name__=='__main__':
    with ProcessPoolExecutor(5) as ex:rows=list(ex.map(one,[(s,m,v) for s in ('open','capped') for m in MODES for v in WITNESSES]))
    with (HERE/'witness.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
