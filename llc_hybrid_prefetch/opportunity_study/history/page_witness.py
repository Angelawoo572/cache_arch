"""Trace the previous-page evidence and matched gain/loss for fixed held-out decisions."""
import csv
import gzip
import numpy as np
from page_association import signals,targets
from evaluate import HERE,openz,guarded,grant,rates,table,writer

def audit(train,test,within):
    z=openz(train);t=z if train==test else openz(test)
    end=int(.6*len(z['t'])); hi=int(.8*len(z['t']))
    serv=(z['fill']-z['dispatch'])[(z['dispatch']>=0)&(z['fill']>=z['dispatch'])&(z['fill']<z['t'][end])]
    service=int(np.median(serv)) if len(serv) else 200
    tri=guarded(z,0,end); tri=tri[grant(tri,2,131)];lo=hi if within else 0
    raw=guarded(t,lo,len(t['t']));g=raw[grant(raw)]
    sk,sn=signals(z);tk,tn=signals(t)
    tt=targets(z,sn,service);q=targets(t,tn,service)[g]; safe=np.maximum(q,0)
    labels=(tt[tri,None]>=0)&(np.arange(64)==(z['pa'][np.maximum(tt[tri],0)]%64)[:,None]);prior=labels.mean(0)
    info={}
    for name in ('current','last_two_pcs','page_offset_only','page_link'):
        tab=table(sk[name][tri],labels);r,s,y=rates(tab,tk[name][g],prior if name=='current' else base)
        if name=='current':base=r
        r=np.where((t['pa'][g]%64)[:,None]!=np.arange(64),r,-1);pred=r.argmax(1)
        hits=(q>=0)&(pred==t['pa'][safe]%64)
        info[name]=(pred,s,y[np.arange(len(s)),pred].astype('int32'),hits)
    po,span,yes,hits=info['page_offset_only'];cur=info['current'];linked=info['page_link']
    prev=np.full(len(t['pc']),-1);pgs={}
    for k,line in enumerate(t['pa']):
        page=int(line//64);v=pgs.get(page,-1)
        if v>=0 and k-v<=64 and t['t'][k]-t['t'][v]<=4096:prev[k]=v
        pgs[page]=k
    olds=prev[g]
    gain=hits&~cur[3];loss=cur[3]&~hits
    offsets=t['pa'][g]%64; ground=t['pa'][safe]%64
    forward=(q>=0)&(offsets+1==ground)
    repeat=2*offsets-(t['pa'][np.maximum(olds,0)]%64)
    repeat_hit=(q>=0)&(olds>=0)&(repeat!=offsets)&(repeat==ground)
    outputs=[];witnesses=[]
    bands=[('all',np.ones(len(g),bool))]
    tercile=np.minimum(2,((g-lo)*3)//(len(t['t'])-lo))
    bands.extend(('third_'+str(i+1),tercile==i) for i in range(3))
    for band,mask in bands:
        ii=np.flatnonzero(mask&gain);e=g[ii];o=olds[ii]
        ok=o>=0;e=e[ok];o=o[ok]
        counts=[len(set(t['pc'][a+1:b])) for a,b in zip(o,e)]
        gaps=e-o
        outputs.append(dict(train=train,test=test,band=band,grants=int(mask.sum()),available=int(((q>=0)&mask).sum()),
                 current=int((cur[3]&mask).sum()),two_adjacent=int((info['last_two_pcs'][3]&mask).sum()),
                 previous_offset=int((hits&mask).sum()),previous_offset_and_pc=int((linked[3]&mask).sum()),
                 next_line=int((forward&mask).sum()),repeat_delta=int((repeat_hit&mask).sum()),
                 extra_over_next=int((hits&~forward&mask).sum()),lost_versus_next=int((~hits&forward&mask).sum()),
                 incremental_also_next=int((gain&forward&mask).sum()),
                 incremental=int((gain&mask).sum()),lost=int((loss&mask).sum()),with_prior_page_reference=len(e),
                 prior_reference_nonadjacent=int(sum(gaps>1)),prior_reference_other_pc=int(sum(t['pc'][o]!=t['pc'][e])),
                 gap_events_median=float(np.median(gaps)) if len(e) else '',gap_cycles_median=float(np.median(t['t'][e]-t['t'][o])) if len(e) else '',
                 intervening_unique_pc_median=float(np.median(counts)) if counts else '',
                 additional_persistent_misses_with_head=int(np.sum(gain&mask&(t['hit'][safe]==0)&(t['head'][safe]>0)))))
    kinds=set()
    for ii in np.flatnonzero(gain):
        a=int(g[ii]);old=int(olds[ii]);q0=int(q[ii])
        if old<0:continue
        cat=('nonadjacent' if a-old>1 else 'adjacent','head' if t['head'][q0]>0 and t['hit'][q0]==0 else 'other')
        if cat in kinds:continue
        kinds.add(cat)
        ids=t['pc'][old+1:a].tolist()
        witnesses.append(dict(train=train,test=test,cue_relation=cat[0],target_class=cat[1],
                  previous_instr=int(t['instr'][old]),previous_operand=int(t['op'][old]),previous_pc=int(t['pc'][old]),
                  previous_physical_line=int(t['pa'][old]),previous_cycle=int(t['t'][old]),
                  intervening_pc_sequence=' '.join(hex(int(ip)) for ip in ids),
                  cue_instr=int(t['instr'][a]),cue_operand=int(t['op'][a]),cue_pc=int(t['pc'][a]),
                  cue_line=int(t['pa'][a]),cue_cycle=int(t['t'][a]),previous_cue_age_ingresses=a-old,
                  previous_cue_age_cycles=int(t['t'][a]-t['t'][old]),query_only_offset=int(cur[0][ii]),
                  query_plus_previous_offset=int(po[ii]),training_context_count=int(span[ii]),training_qualifying_targets=int(yes[ii]),
                  target_instr=int(t['instr'][q0]),target_operand=int(t['op'][q0]),target_pc=int(t['pc'][q0]),target_line=int(t['pa'][q0]),
                  target_ingress=int(t['t'][q0]),target_lead_ingresses=q0-a,target_lead_cycles=int(t['t'][q0]-t['t'][a]),
                  original_tag_hit=int(t['baseline'][q0]),stride_tag_hit=int(t['hit'][q0]),stride_pf_first_use=int(t['pf_kind'][q0]),
                  head_operand_wait_cycles=int(t['head'][q0]),service_training=service,
                  target_own_dispatch_fill=int(t['fill'][q0]-t['dispatch'][q0]) if t['dispatch'][q0]>=0 else ''))
    return outputs,witnesses


def verify_live(witness):
    runs=HERE.parent/'raw/page_memory/runs'
    if not runs.is_dir():
        return []
    trigger=str(witness['cue_instr']); target=str(witness['target_instr']); target_line=str(witness['target_line'])
    tc=str(witness['cue_operand']); op=str(witness['target_operand'])
    output=[]
    for mode in ('shadow','current','offset','next'):
        d=runs/mode
        with (d/'decisions.csv').open() as f:
            decision=next(r for r in csv.DictReader(f) if (r['instr'],r['operand'])==(trigger,tc))
        lifecycle={}; lookup={};operand_wait=0
        with gzip.open(d/'candidates.csv.gz','rt') as f:
            for r in csv.DictReader(f):
                if (r['instr_id'],r['operand'],r['target'])==(trigger,tc,decision['chosen']):
                    if r['stage']=='timely' and (r['demand_instr_id'],r['demand_operand'])!=(target,op):
                        continue
                    lifecycle[r['stage']]=r['cycle']
        with gzip.open(d/'loads.csv.gz','rt') as f:
            header=f.readline().rstrip().split(',')
            token=','+target+','+op+','
            for line in f:
                if token not in line or ',LLC,' not in line:
                    continue
                r=dict(zip(header,line.rstrip().split(',')))
                if (r['instr_id'],r['operand'],r['level'])!=(target,op,'LLC'):
                    continue
                assert (r['physical_line'],r['pc'])==(target_line,str(witness['target_pc']))
                lookup[r['stage']]=r['cycle']
                if r['stage']=='lookup': lookup['hit']=r['hit']
        with gzip.open(d/'head.csv.gz','rt') as f:
            for r in csv.DictReader(f):
                if (r['instr_id'],r['operand'])==(target,op):
                    operand_wait+=int(r['cycles'])
        output.append(dict(arm=mode,cue_instr=trigger,cue_operand=tc,cue_line=decision['base'],
                           cue_cycle=decision['cycle'],chosen_line=decision['chosen'],action=decision['action'],
                           accepted_cycle=lifecycle.get('accepted',''),prefetch_fill=lifecycle.get('fill',''),
                           target_instr=target,target_operand=op,target_line=target_line,
                           target_ingress=lookup['ingress'],target_lookup=lookup['lookup'],hit=lookup['hit'],
                           direct_first_use=lifecycle.get('timely',''),head_operand_wait_cycles=operand_wait))
    return output

if __name__=='__main__':
    jobs=[('gap_bfs3_late-c0','gap_bfs3_late-c0',True),
          ('mcf-r2-c0','mcf994_adjacent-c0',False),('gap_bfs3_late-c0','gap_bfs8_late-c0',False),
          ('gap_bfs3-c0','gap_bfs8-c0',False),('gap_bfs8_late-c0','gap_bfs8_late-c0',True)]
    outputs=[];witness=[]
    for params in jobs:
        a,b=audit(*params);outputs.extend(a);witness.extend(b)
    writer(HERE/'page_memory_stability.csv',outputs);writer(HERE/'page_memory_witnesses.csv',witness)
    target=next(r for r in witness if r['test']==r['train']=='gap_bfs3_late-c0' and r['cue_relation']=='nonadjacent' and r['target_class']=='head')
    native=verify_live(target)
    if native:writer(HERE/'page_memory_native_witness.csv',native)
