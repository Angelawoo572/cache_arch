"""Keep the forecast frozen while varying the offline service reference and inspecting source labels."""
import csv
import gzip
import numpy as np
from predict import HERE,case,measured_service,guarded,fit_all,predictions,outcomes,writer
from evaluate import grant


def run(source,test):
    z,f,g=case(source,'ingress');t,u,h=case(test,'ingress')
    cut=int(.6*len(z['t']));lo=int(.8*len(t['t'])) if source==test else 0
    tr=guarded(z,0,cut);ids=guarded(t,lo,len(t['t']));ids=ids[grant(ids)]
    valid=(z['dispatch']>=0)&(z['fill']>=z['dispatch'])&(z['fill']<z['t'][cut]);svc=(z['fill']-z['dispatch'])[valid]
    med=measured_service(z,cut);table,prior=fit_all(z,f,g,tr,med,'1_64')
    yp,score,_=predictions('page',t,u,ids,table,prior)
    raw,good,q=outcomes(t,h,ids,med,'1_64');dt=t['t'][q]-t['t'][ids]
    match=raw&(score>0)&(yp==t['pa'][q])
    table1=[]
    for label in ('p25','p50','p75','individual'):
        if label=='individual':
            known=(t['dispatch'][q]>=0)&(t['fill'][q]>=t['dispatch'][q])
            threshold=t['fill'][q]-t['dispatch'][q]
            d=''
        else:
            known=np.ones(len(ids),bool);d=int(np.percentile(svc,int(label[1:]))) if len(svc) else 200
            threshold=d
        table1.append(dict(source=source,test=test,reference=label,cycles=d,training_reference=med,
                           grants=len(ids),named_address_targets=len(np.unique(q[match])),
                           named_with_available_reference=len(np.unique(q[match&known])),
                           named_meeting_reference=len(np.unique(q[match&known&(dt>=threshold)]))))
    ground=HERE.parents[1]/'opportunity_study/raw/traces'/(test+'.labels.csv.gz')
    table2=[]
    if ground.is_file():
        with gzip.open(ground,'rt') as f: lab={int(r['instr_id']):r['label'] for r in csv.DictReader(f) if r['measurement']=='roi'}
        targets=np.flatnonzero(h['end']&(np.arange(len(t['t']))>=lo))
        statuses={int(j):lab.get(int(t['instr'][j]),'unmatched') for j in targets}
        reached=set(q[match&good])
        for cl in sorted(set(statuses.values())):
            pop=[j for j,v in statuses.items() if v==cl]
            table2.append(dict(source=source,test=test,source_class=cl,endpoints=len(pop),timely_page_matches=len(set(pop)&reached)))
    return table1,table2

if __name__=='__main__':
    rows=[];truth=[]
    for a,b in [('mcf-r1','mcf-r1'),('lbm-r1','lbm-r2'),('gap_bfs3_late','gap_bfs3_late'),('row_period2_paced','row_period5_paced')]:
        x,y=run(a,b);rows.extend(x);truth.extend(y)
    writer(HERE/'service_sensitivity.csv',rows);writer(HERE/'row_source_truth.csv',truth)
