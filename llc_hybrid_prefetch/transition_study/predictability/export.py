"""Export the fixed 1-in-8 page rule, with training labels guarded within the prefix."""
import argparse
import csv
from pathlib import Path
import numpy as np
from predict import HERE,case,fit_all,measured_service,guarded,outcomes,fit_keys,frequencies,eligible_scores,ACTIONS,WIDE
from evaluate import grant
from controls import keys as ablation_keys

MODES=('page','exact','unrelated_older','last_global_pc')


def export(name,test,dest,mode='page',expected=None):
    z,f,g=case(name,'ingress');t,ft,_=case(test,'ingress')
    cut=int(.6*len(z['t']))
    service=measured_service(z,cut)
    tr=guarded(z,0,cut)
    tables,prior=fit_all(z,f,g,tr,service,'1_64')
    _,labels,_=outcomes(z,g,tr,service,'1_64')
    y=np.where(labels,g['code'][tr],-1)
    k=ablation_keys(z,f)[mode]
    top=fit_keys(k[tr],y)
    lo=int(.8*len(t['t'])) if name==test else 0
    eval_ids=guarded(t,lo,len(t['t']))
    with dest.open('w') as h:
        print(lo,eval_ids[-1]+1,MODES.index(mode),file=h)
        print(*prior.tolist(),file=h)
        for stage,fit in enumerate((tables['low12'],top)):
            keys,counts,positive=fit
            for i in np.flatnonzero(counts>=8):
                print(stage,int(keys[i]),counts[i],*positive[i],file=h)

    if expected is not None:
        te=eval_ids[grant(eval_ids)]
        low,_=frequencies(tables['low12'],ft['keys']['low12'][te],prior)
        raw,_=frequencies(top,ablation_keys(t,ft)[mode][te],low)
        prob=eligible_scores(raw,t,te,ft['wide_legal']);code=prob.argmax(1)
        score=prob[np.arange(len(te)),code];line=t['pa'][te]
        target=np.where(code<64,(line//64)*64+code,line+WIDE[np.maximum(0,code-64)])
        target=np.where(score>0,target,-1)
        with expected.open('w') as out:
            w=csv.writer(out,lineterminator='\n');w.writerow(('index','instr','operand','base','cycle','choice'))
            w.writerows(zip(te,t['instr'][te],t['op'][te],line,t['t'][te],target))

if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('source');cli.add_argument('test');cli.add_argument('output',type=Path)
    cli.add_argument('--mode',choices=MODES,default='page');cli.add_argument('--expected',type=Path)
    a=cli.parse_args();export(a.source,a.test,a.output,a.mode,a.expected)
