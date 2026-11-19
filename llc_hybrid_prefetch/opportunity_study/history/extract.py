"""Materialize per-core physical LLC ingress, delayed feedback and paired offline labels."""
import argparse
import csv
import gzip
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
OLD = STUDY.parent / 'raw' / 'runs'
NEW = STUDY / 'raw' / 'runs'
CACHE = STUDY / 'raw' / 'history'
FIELDS = ('instr','op','pc','pa','va','t','tag_t','hit','baseline','dispatch','fill','state','proposed','head','pf_gen','pf_fill','pf_kind')
STATE = {'cold':1,'delta_changed':2,'zero_delta':3,'stride':4,'intervention':5}


def records(path, name):
    f = path / (name + '.csv')
    stream = f.open() if f.is_file() else gzip.open(str(f)+'.gz','rt')
    with stream:
        reader = csv.reader(stream)
        header = next(reader)
        yield {k:i for i,k in enumerate(header)}
        yield from reader


def id_of(row, h):
    return int(row[h['cpu']]), (int(row[h['instr_id']]) << 8) + int(row[h['operand']])


def parse_arm(folder, baseline=None):
    ret = []
    maps = []
    info = json.loads((folder/'run.json').read_text())
    traces = info.get('traces') or info.get('workloads')
    for _ in traces:
        ret.append([])
        maps.append({})
    itr = records(folder,'loads')
    h = next(itr)
    for v in itr:
        if v[h['level']] != 'LLC':
            continue
        s = v[h['stage']]
        if s not in ('ingress','lookup','dispatch','fill'):
            continue
        cpu,k = id_of(v,h)
        timestamp = int(v[h['cycle']])
        if baseline is None:
            if s=='lookup':
                maps[cpu][k] = (int(v[h['pc']]),int(v[h['virtual_line']]),int(v[h['hit']]))
            continue
        if s=='ingress':
            a = [k>>8,k%256,int(v[h['pc']]),int(v[h['physical_line']]),int(v[h['virtual_line']]),timestamp,-1,-1,-1,-1,-1,0,-1,0,-1,-1,0]
            match = baseline[cpu].get(k)
            if match:
                if (a[2],a[4]) != match[:2]:
                    raise ValueError('paired PC/virtual line mismatch {} {}'.format(folder,k))
                a[8] = match[2]
            if k in maps[cpu]:
                raise ValueError('repeated ingress')
            ret[cpu].append(a)
            maps[cpu][k] = a
        elif k in maps[cpu]:
            a = maps[cpu][k]
            if s=='lookup':
                a[6:8] = [timestamp,int(v[h['hit']])]
            elif s=='dispatch':
                a[9] = timestamp
            elif s=='fill':
                a[10] = timestamp
    if baseline is None:
        return maps
    it = records(folder,'head'); h = next(it)
    for v in it:
        cpu,k = id_of(v,h)
        if k in maps[cpu]:
            maps[cpu][k][13] += int(v[h['cycles']])
    it = records(folder,'candidates'); h = next(it)
    generations, fills, claims = {},{},{}
    for v in it:
        s = v[h['stage']]
        cpu,k = id_of(v,h)
        pid = int(v[h['id']])
        if s in ('silent','proposed') and k in maps[cpu]:
            maps[cpu][k][11] = STATE.get(v[h['reason']],0)
            if s=='proposed':
                maps[cpu][k][12] = int(v[h['target']])
                generations[pid] = int(v[h['cycle']])
        if s=='fill':
            fills[pid] = int(v[h['cycle']])
        if s in ('timely','late') and int(v[h['demand_cpu']])>=0:
            dc = int(v[h['demand_cpu']])
            dk = (int(v[h['demand_instr_id']])<<8) + int(v[h['demand_operand']])
            claims[pid] = (dc,dk,s)
    for pid,(cpu,k,s) in claims.items():
        if k in maps[cpu]:
            maps[cpu][k][14:] = [generations.get(pid,-1),fills.get(pid,-1),1 if s=='timely' else 2]
    return ret,traces


def extract(args):
    case, base = args
    prefix = CACHE/case
    b, s = [base/(case+'-'+p) for p in ('nopf','stride')]
    if not all((p/'complete.json').exists() for p in (b,s)):
        return
    marker=prefix.parent/(case+'.ready')
    completed=max((f/'complete.json').stat().st_mtime_ns for f in (b,s))
    if marker.exists() and marker.stat().st_mtime_ns>=completed:
        return
    data,traces = parse_arm(s,parse_arm(b))
    for c, recs in enumerate(data):
        if not recs:
            continue
        a=np.array(recs,dtype=np.int64)
        if np.any(a[1:,5]<a[:-1,5]):
            raise ValueError('LLC ingress cycles run backward')
        path=prefix.parent/(case+'-c'+str(c)+'.npz')
        np.savez_compressed(path,trace=traces[c],**{k:a[:,i] for i,k in enumerate(FIELDS)})
    (prefix.parent/(case+'.ready')).write_text('complete\n')
    print(case, [len(r) for r in data], flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--runs', type=Path, default=OLD)
    ap.add_argument('--case', nargs='*')
    ap.add_argument('--jobs',type=int,default=4)
    args=ap.parse_args()
    CACHE.mkdir(parents=True,exist_ok=True)
    names=args.case if args.case else sorted(f.name[:-5] for f in args.runs.glob('*-nopf') if not f.name.startswith('warm5-'))
    with ProcessPoolExecutor(args.jobs) as ex:
        list(ex.map(extract,[(k,args.runs) for k in names]))
