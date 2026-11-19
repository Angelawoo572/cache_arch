"""Fit R1-only tables and export R2 open-loop audit schedules. Never tune on audit data."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from evaluate import HERE,CACHE,openz,build,contexts,positive,guarded,grant,table,rates,choices,LABELS

def main(out,train,test):
    out.mkdir(exist_ok=True,parents=True)
    z=openz(train); b=build(z);n=len(z['t']);end=int(.60*n)
    gs=guarded(z,0,end);idx=gs[grant(gs,2,131)]
    s=(z['fill']-z['dispatch'])[(z['dispatch']>=0)&(z['fill']>=z['dispatch'])&(z['fill']<z['t'][end])]
    lead=int(np.median(s)) if len(s) else 200
    labels=positive(z,b,lead)[idx]; prior=labels.mean(0)
    name,key=next(contexts(z,b));assert name=='recent'
    tab=table(key[idx],labels);u,n,yes=tab
    path=out/'r1_table.txt'
    with path.open('w') as f:
        def write(category,k,support,scores):
            order=np.argsort(-scores,kind='stable')
            f.write(' '.join(map(str,[category,int(k),int(support),*map(int,order)]))+'\n')
        write('global',0,len(idx),prior)
        for i in np.flatnonzero(n>=8):write('key',u[i],n[i],(yes[i]+8*prior)/(n[i]+8))
    other=openz(test);bb=build(other);_,kk=next(contexts(other,bb))
    ix=np.arange(len(kk));g=ix[grant(ix)];rr,support,y=rates(tab,kk[g],prior)
    pick=choices(rr,bb,g); eligible=set(guarded(other,0,len(ix)))
    first={int(page):j for page,j in bb['first_page'].items()}
    with (out/'r2_reference.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n')
        w.writerow(['cpu','instr_id','operand','pc','base_line','ingress_cycle','ingress_index','target_line','physical_delta',
                    'action','action_index','plus128_target','within_offline_scoring_horizon','context_key','training_support'])
        for j,c,su in zip(g,pick,support):
            base=int(other['pa'][j]);targ=int(bb['cand'][j,c]);delta=targ-base
            fixed=base+128 if first.get((base+128)//64,len(ix))<=j else 0
            w.writerow([0,int(other['instr'][j]),int(other['op'][j]),int(other['pc'][j]),base,int(other['t'][j]),j,targ,delta,
                        LABELS[c],c,fixed,int(j in eligible),int(kk[j]),int(su)])
    with (out/'r2_ingress.txt').open('w') as f:
        for p,l in zip(other['pc'],other['pa']):f.write(str(p)+' '+str(l)+'\n')
    meta=dict(train=train,test=test,training_anchors=len(idx),training_service=lead,known_contexts=int((n>=8).sum()),
              r2_ingresses=len(ix),r2_grants=len(g),r2_distinct_pages=len(first),r2_distinct_pcs=len(np.unique(other['pc'])))
    (out/'counts.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(meta)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=CACHE/'cactus_live')
    ap.add_argument('--train',default='cactus-r1-c0');ap.add_argument('--test',default='cactus-r2-c0')
    a=ap.parse_args();main(a.out,a.train,a.test)
