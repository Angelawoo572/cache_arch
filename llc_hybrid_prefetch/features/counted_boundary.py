"""Test a causal counted-stride explanation on the known, paced loop."""
import argparse
import csv
import sys
from collections import Counter, defaultdict
import gzip
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "intervention"))
from earlier_issue import read_llc, identity


def make_plan(source):
    events, _, _ = read_llc(source)
    prior, histograms = {}, defaultdict(Counter)
    plan, evidence = {}, []
    for a in sorted(events, key=lambda a: a.cycle):
        stream = (a.cpu, a.pc)
        last = prior.get(stream)
        delta = None
        length = 0
        if last is not None:
            old, old_delta, old_len, old_key = last
            delta = a.line - old.line
            histograms[old_key][delta] += 1
            length = min(63, old_len + 1) if old_delta == delta else 1
        key = (a.cpu, a.pc, a.line % 64, delta, length)
        votes = histograms[key]
        if votes:
            predicted, count = votes.most_common(1)[0]
            # Require all earlier observations of this context to agree.
            if (count >= 2 and len(votes) == 1 and predicted != delta and predicted != 0
                    and a.line // 64 == (a.line + predicted) // 64):
                plan[identity(a)] = predicted
                evidence.append((a.cpu,a.instr,a.operand,a.line % 64,delta,length,predicted,count))
        prior[stream] = a, delta, length, key
    return plan, evidence


def save(out, plan, evidence):
    out.mkdir(parents=True,exist_ok=True)
    with (out/"plan.csv").open("w", newline="") as f:
        writer=csv.writer(f,lineterminator="\n")
        writer.writerow(("cpu","instr_id","operand","delta"))
        for key,value in sorted(plan.items()):
            writer.writerow((*key,value))
    with (out/"prefix_evidence.csv").open("w", newline="") as f:
        writer=csv.writer(f,lineterminator="\n")
        writer.writerow(("cpu","instr_id","operand","offset","last_stride","run_length","predicted_delta","prior_observations"))
        writer.writerows(evidence)


def compare_boundaries(baseline, replay, boundaries, output):
    labels = {}
    with gzip.open(boundaries, "rt") as stream:
        for row in csv.DictReader(stream):
            if row["measurement"] == "roi":
                labels[int(row["instruction_id"])] = row["pattern"]
    result = []
    for label, folder in (("stride", baseline), ("counted_stride", replay)):
        _, outcomes, _ = read_llc(folder)
        counts = defaultdict(Counter)
        for key, hit in outcomes.items():
            counts[labels.get(key.instr, "interior")][hit] += 1
        with (folder / "cores.csv").open() as stream:
            cycles = next(csv.DictReader(stream))["cycles"]
        for group, count in sorted(counts.items()):
            result.append((label, group, count[0], count[1], cycles))
    with output.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("policy","load","llc_misses","llc_hits","fixed_work_cycles"))
        writer.writerows(result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--verify-replay",type=Path)
    p.add_argument("--boundaries",type=Path)
    p.add_argument("--summary",type=Path)
    args=p.parse_args()
    plan, evidence=make_plan(args.source)
    if args.verify_replay:
        check, history=make_plan(args.verify_replay)
        if check != plan or history != evidence:
            raise ValueError("replay histories changed the counted-stride actions")
    if args.summary:
        if not args.verify_replay or not args.boundaries:
            p.error("--summary needs --verify-replay and --boundaries")
        compare_boundaries(args.source,args.verify_replay,args.boundaries,args.summary)
    save(args.out,plan,evidence)
    print("Counted-stride overrides:", len(plan))
