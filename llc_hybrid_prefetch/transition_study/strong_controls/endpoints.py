"""Compare direct timely claims for the same original failure endpoints."""
import csv
import gzip
from run import HERE,MODES,RAW

def read(p):
    with gzip.open(p,'rt') as f:yield from csv.DictReader(f)

def claims(folder,pop):
    out=set()
    for r in read(folder/'candidates.csv.gz'):
        if r['stage']!='timely' or int(r['demand_cpu'])!=0:continue
        k=int(r['demand_instr_id']),int(r['demand_operand'])
        if k in pop:
            if int(r['target'])!=pop[k]:raise ValueError('physical endpoint mismatch')
            out.add(k)
    return out
def main():
    e=HERE.parent/'raw/episodes/lbm-r2/episode_ids.csv.gz'
    pop={(int(r['first_miss_instr']),int(r['first_miss_op'])):int(r['target_line']) for r in read(e)}
    aux={key:claims(HERE.parent/'raw/replay/runs'/('lbm-r2__'+key),pop) for key in ('lbm_exact','lbm_page')}
    out=[]
    for setting in ('open','capped'):
        for mode in MODES:
            conventional=claims(RAW/'verified'/(setting+'-'+mode),pop)
            for name,observed in aux.items():
                out.append(dict(setting=setting,conventional=mode,sparse_rule=name,fixed_original_failure_operands=len(pop),
                                conventional_direct=len(conventional),sparse_rule_direct=len(observed),both_direct=len(observed&conventional),
                                conventional_only=len(conventional-observed),sparse_rule_only=len(observed-conventional)))
    with (HERE/'endpoints.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=out[0],lineterminator='\n');w.writeheader();w.writerows(out)
if __name__=='__main__':main()
