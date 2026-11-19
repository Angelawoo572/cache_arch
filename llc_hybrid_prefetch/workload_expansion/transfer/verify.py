"""Check the later-window witnesses and the causal boundaries of the descriptive tables."""
import csv
from collections import defaultdict
from evaluate import HERE,STUDY,fit,legal,read_history,snapshots


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def verify(folder,root):
    cases={r["case"] for r in table(STUDY/"intake/registry.csv") if r["split"]=="discovery"}
    results=table(folder/"rules.csv")
    for r in results:
        assert r["case"] in cases and int(r["context_entries"])<=256
        for group in (("anchors","supported","offered","first_use_address_matches","matches_with_lead",
                       "original_matches_with_lead","gap_matches_with_lead","gap_positive_head"),
                      ("anchors","original_misses","all_three_missed","gap_legal_with_lead")):
            values=[int(r[k]) for k in group]
            assert values==sorted(values,reverse=True)
    for case in sorted(set(r["case"] for r in results)):
        a=read_history(STUDY/"raw/screen",case,"stride"); b=read_history(root,case,"stride")
        if len(a["t"]):
            end=int(.6*len(a["t"]))
            stripped={k:a[k][:end] for k in ("pc","pa","t")}
            original=snapshots(a); prefix=snapshots(stripped)
            assert prefix[0]==original[0][:end]
            assert fit(a,*original,end,4)==fit(stripped,*prefix,end,4)
        idx={(int(n),int(op)):j for j,(n,op) in enumerate(zip(b["instr"],b["op"]))}
        keys,pages=snapshots(b)
        groups=defaultdict(list)
        for row in table(folder/"paired_futures.csv"):
            if row["case"]!=case:continue
            i=idx[int(row["cue_instr"]),int(row["cue_operand"])]; j=idx[int(row["target_instr"]),int(row["target_operand"])]
            assert j-i==int(row["horizon"]) and b["pa"][i]==int(row["cue_line"]) and b["pa"][j]==int(row["target_line"])
            assert int(b["pa"][j])-int(b["pa"][i])==int(row["chosen_delta"])
            assert legal(int(b["pa"][i]),int(b["pa"][j]),pages,i) and b["pa"][j] not in b["pa"][i+1:j]
            assert b["t"][j]-b["t"][i]==int(row["lead_cycles"])>=float(row["service_reference"])
            assert keys[i][0]==tuple(int(row[k]) for k in ("local_pc","offset","same_pc_delta","preceding_same_pc_delta"))
            assert keys[i][2][-3:]==tuple(map(int,row["previous_three_pcs"].split()))
            groups[row["horizon"],keys[i][0]].append(row)
        for group in groups.values():
            assert len({r["chosen_delta"] for r in group})>=2 and len({r["previous_three_pcs"] for r in group})>=2
            assert any(int(r["baseline_miss"]) for r in group)
    return len(results)


if __name__=="__main__":
    n=verify(HERE,STUDY/"raw/replication")
    m=verify(HERE/"long_google",STUDY/"raw/long_google")
    print(f"Verified {n+m} transfer score rows, training cutoffs and every strict paired-future witness.")
