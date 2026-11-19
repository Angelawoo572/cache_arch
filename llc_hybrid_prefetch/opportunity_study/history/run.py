"""Preregistered matrix and compact publication of chronological and cross-window results."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse,csv,json,gzip,io
from evaluate import run,EVAL,HERE
from extract import CACHE


def jobs():
    old=[k+'-r'+str(r)+'-c0' for k in ('mcf','cactus','lbm','xalan') for r in (1,2)]
    mixed=['mcf-cactus-r1-c0','mcf-cactus-r1-c1','lbm-xalan-r2-c0','lbm-xalan-r2-c1']
    manifest=json.loads((HERE.parent/'workloads/cases.json').read_text())
    new=[q['case']+'-c'+str(c) for q in manifest for c in range(len(q['traces']))]
    recent=[(k,k,'within') for k in old+mixed+new]
    blind=[(k+'-r1-c0',k+'-r2-c0','blind') for k in ('mcf','cactus','xalan','lbm')]
    for a,b in (('gap_bfs3','gap_bfs8'),('mcf-r2','mcf994_adjacent'),('row_period2_paced','row_period5_paced'),
                ('row_previous_paced','row_independent_paced'),('crosspc_cue_paced','crosspc_mask_paced'),
                ('streams_onepc_paced','streams_pc_paced'),('gap_bfs3_late','gap_bfs8_late'),
                ('gap_bfs3','gap_bfs3_late'),('gap_bfs8','gap_bfs8_late'),('xor_context_paced','xor_maskfirst_paced'),
                ('streams_three_switch_paced','streams_three_identified_paced')):
        blind.append((a+'-c0',b+'-c0','blind'))
    return blind+recent


def one(item):
    a,b,mode=item;out=EVAL/('__'.join(item))
    if not all((CACHE/(z+'.npz')).is_file() for z in (a,b)):
        return 'no LLC ingress or missing extraction: '+a+','+b
    try:
        run(a,b,mode)
    except Exception as e:
        raise RuntimeError('study failed for '+str(item)) from e
    return str(item)


def combine():
    matrix=jobs()
    for table in ('models','ceiling','examples','verified','classes'):
        out=HERE/('history_'+table+'.csv')
        with out.open('w',newline='') as f:
            writer=csv.writer(f,lineterminator='\n');header=None
            for item in matrix:
                path=EVAL/('__'.join(item))/(table+'.csv')
                if not path.exists():continue
                with path.open() as g:
                    r=csv.reader(g);h=next(r)
                    if header is None:header=h;writer.writerow(h)
                    elif header!=h:raise ValueError('incompatible output schema')
                    for row in r:writer.writerow(row)

    # Publish complete per-miss provenance without thousands of duplicated
    # uncompressed lines in the human-facing report.
    target=HERE/'sampled_opportunities.csv.gz'
    with target.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb',mtime=0) as zipped, io.TextIOWrapper(zipped,newline='') as out:
        header=None
        for item in matrix:
            p=EVAL/('__'.join(item))/'sampled_misses.csv.gz'
            if not p.exists():continue
            with gzip.open(p,'rt') as f:
                h=f.readline()
                if header is None:header=h;out.write(h)
                elif h!=header:raise ValueError('incompatible per-miss records')
                for line in f:out.write(line)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',type=int,default=4);ap.add_argument('--combine-only',action='store_true')
    ap.add_argument('--only-new',action='store_true');ap.add_argument('--real',action='store_true');ap.add_argument('--generated',action='store_true')
    args=ap.parse_args()
    matrix=jobs()
    if args.only_new:
        matrix=[j for j in matrix if any(x in j[1] for x in ('_late','xor_','streams_three','row_hierarchy'))]
    if args.real or args.generated:
        real=('mcf','cactus','lbm','xalan','gap','compiled')
        matrix=[j for j in matrix if j[0].startswith(real)==args.real]
    if not args.combine_only:
        with ProcessPoolExecutor(args.jobs) as ex:
            for msg in ex.map(one,matrix):print(msg,flush=True)
    combine()
