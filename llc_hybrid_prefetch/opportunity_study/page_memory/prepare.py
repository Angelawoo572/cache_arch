"""Export prefix-only offset histograms and a separate shadow QA reference."""
import csv
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
OUT = STUDY/'raw/page_memory'
sys.path.insert(0, str(STUDY/'history'))
from evaluate import openz, guarded, grant, table, rates
from page_association import signals, targets

RULES = ('current', 'page_offset_only', 'page_link')

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    z=openz('gap_bfs3_late-c0')
    n=len(z['pa'])
    end, dev = int(.6*n), int(.8*n)
    observed=(z['fill']-z['dispatch'])[(z['dispatch']>=0)&(z['fill']>=z['dispatch'])&(z['fill']<z['t'][end])]
    service=int(np.median(observed))
    train=guarded(z,0,end)
    train=train[grant(train,2,131)]
    test=guarded(z,dev,n)
    grants=test[grant(test)]
    keys,nxt=signals(z)
    dest=targets(z,nxt,service)
    offset=z['pa'][np.maximum(dest[train],0)]%64
    labels=(dest[train,None]>=0)&(np.arange(64)==offset[:,None])
    prior=labels.mean(axis=0)
    reference={}
    with (OUT/'prefix.txt').open('w') as f:
        print(dev, int(test[-1])+1, file=f)
        print(*[repr(float(x)) for x in prior],file=f)
        for mode,name in enumerate(RULES):
            fit=table(keys[name][train],labels)
            u,counts,yes=fit
            for key,count,outcomes in zip(u,counts,yes):
                if count>=8:
                    print(mode,int(key),int(count),*outcomes.astype(int).tolist(),file=f)
            score,_,_=rates(fit,keys[name][grants],prior if not mode else base)
            if not mode:
                base=score
            masked=np.where((z['pa'][grants]%64)[:,None]!=np.arange(64),score,-1)
            reference[name]=(z['pa'][grants]//64)*64+masked.argmax(axis=1)
    with (OUT/'shadow_reference.csv').open('w',newline='') as f:
        out=csv.writer(f,lineterminator='\n')
        out.writerow(('index','instr','operand','base',*RULES))
        for j,i in enumerate(grants):
            out.writerow((i,z['instr'][i],z['op'][i],z['pa'][i],*[reference[k][j] for k in RULES]))
    print(n,end,dev,int(test[-1])+1,len(train),len(grants),service)
if __name__=='__main__':prepare()
