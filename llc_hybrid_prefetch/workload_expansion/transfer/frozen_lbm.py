"""Blind replay of the earlier LBM donor tables as offline probes on seven Google traces."""
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from pathlib import Path
import subprocess
from evaluate import HERE, STUDY, lookup, native_class, read_history, write

PARENT = STUDY.parent / "transition_study"
RAW = STUDY / "raw/frozen_lbm"
LABELS = ("lbm_exact", "lbm_page", "lbm_prior", "next_line", "plus5")
LIMIT = 561


def prepare():
    RAW.mkdir(parents=True, exist_ok=True)
    inc = PARENT / "predictability"
    exe = RAW / "reader"
    subprocess.run(["c++","-std=c++14","-O2","-I",str(inc),str(HERE/"frozen_reader.cc"),"-o",str(exe)], check=True)
    tables=[]
    for name in ("exact","page","prior"):
        source = PARENT / "raw/predictability" / ("lbm-"+("exact" if name=="prior" else name)+".txt")
        lines = source.read_text().splitlines()
        mode = int(lines[0].split()[2])
        table = RAW / (name+".txt")
        keep = lines[1:2] if name=="prior" else lines[1:]
        table.write_text(f"0 18446744073709551615 {mode}\n"+"\n".join(keep)+"\n")
        tables.append(table)
    buckets={int(r.split()[1]) for r in tables[0].read_text().splitlines()[2:] if r.startswith("1 ")}
    return [str(exe),*map(str,tables)], buckets


def run_case(root,window,case,engine,buckets):
    z=read_history(root,case,"stride")
    native = [lookup(read_history(root,case,p)) for p in ("stride","multi","spp")]
    positions=defaultdict(list)
    for j,p in enumerate(z["pa"]):
        positions[int(p)].append(j)
    rows="".join(f"{int(pc)} {int(line)} {int(t)}\n" for pc,line,t in zip(z["pc"],z["pa"],z["t"]))
    forecasts = subprocess.run(engine,input=rows,capture_output=True,text=True,check=True).stdout.splitlines()
    counts=[Counter() for _ in LABELS]
    gains=[set() for _ in LABELS]
    for row in forecasts:
        i,*choices=map(int,row.split())
        if i+64>=len(z["pa"]):
            continue
        base=int(z["pa"][i])
        bucket=(((int(z["pc"][i])%4096) ^ (base%64))*11400714819323198485)%(2**64)
        choices.extend([base+d if base//64==(base+d)//64 else -1 for d in (1,5)])
        for rule,target in enumerate(choices):
            c=counts[rule]
            c["slots"]+=1
            c["exact_bucket_present"]+=bucket in buckets
            if target<=0:continue
            assert base!=target
            c["offered"]+=1
            offsets=positions.get(target,[])
            rank=bisect_right(offsets,i)
            if rank==len(offsets):continue
            j=offsets[rank]
            lead=int(z["t"][j]-z["t"][i])
            if j-i>64 or lead>4096:continue
            c["address_matches"]+=1
            if int(z["baseline"][j])!=0:continue
            c["original_miss_matches"]+=1
            if lead<LIMIT:continue
            c["original_matches_with_margin"]+=1
            if all(s in ("miss","late") for s in native_class(z,j,native)):
                gains[rule].add(j)
    return [dict(window=window,case=case,rule=name,
                 **{k:counts[i][k] for k in ("slots","exact_bucket_present","offered","address_matches",
                    "original_miss_matches","original_matches_with_margin")},unique_all_miss_with_margin=len(gains[i]),
                 all_miss_gained_over_prior=len(gains[i]-gains[2]),all_miss_lost_vs_prior=len(gains[2]-gains[i]))
            for i,name in enumerate(LABELS)]


def main():
    engine,buckets=prepare()
    with (STUDY/"intake/registry.csv").open() as f:
        cases=[r["case"] for r in csv.DictReader(f) if r["split"]=="discovery" and r["collection"]=="gtrace_v2"]
    data=[]
    for label,folder in (("A","screen"),("B","replication"),("C","long_google")):
        for case in cases:
            data.extend(run_case(STUDY/"raw"/folder,label,case,engine,buckets))
    write(HERE/"frozen_lbm.csv",data)


if __name__=="__main__":
    main()
