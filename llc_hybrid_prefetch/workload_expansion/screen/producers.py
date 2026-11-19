"""Which observed PC and displacement generated the same physical request before a paired load?"""

from collections import defaultdict
import argparse
import csv
import gzip
from pathlib import Path
import numpy as np
from run import DEFAULT_OUTPUT, HERE, select


def requests(folder):
    proposed, consumed = {}, {}
    with gzip.open(folder / "candidates.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            pid = int(r["id"])
            stage = r["stage"]
            if stage == "proposed":
                proposed[pid] = dict(
                    instr=int(r["instr_id"]),
                    op=int(r["operand"]),
                    pc=int(r["pc"]),
                    base=int(r["base"]),
                    target=int(r["target"]),
                    generation=int(r["cycle"]),
                )
            elif stage == "dispatch" or stage == "fill":
                proposed[pid][stage] = int(r["cycle"])
            elif stage in ("timely", "late"):
                key = int(r["demand_instr_id"]), int(r["demand_operand"])
                assert key not in consumed
                consumed[key] = stage, proposed[pid]
    return consumed


def traces(root, case, pol):
    with np.load(root / "history" / case / (pol + ".npz")) as z:
        return {
            (int(i), int(op)): (int(pc), int(pa), int(t), int(tag), int(h))
            for i, op, pc, pa, t, tag, h, old in zip(
                z["instr"],
                z["op"],
                z["pc"],
                z["pa"],
                z["t"],
                z["tag_t"],
                z["head"],
                z["baseline"],
            )
            if old == 0
        }


def details(root, case):
    case_dir = root / "runs" / case
    sa, sb = (requests(case_dir / p) for p in ("stride", "spp"))
    ta, tb = (traces(root, case, p) for p in ("stride", "spp"))
    by = defaultdict(list)
    for key in sa.keys() & sb.keys() & ta.keys() & tb.keys():
        ka, a = sa[key]
        kb, b = sb[key]
        if ka == kb:
            continue
        pc, line, t1, tag1, w1 = ta[key]
        pc2, line2, t2, tag2, w2 = tb[key]
        assert pc == pc2 and line == line2 == a["target"] == b["target"]
        grp = (
            ka,
            kb,
            pc,
            a["pc"],
            b["pc"],
            a["target"] - a["base"],
            b["target"] - b["base"],
        )
        item = dict(target_instr=key[0], target_op=key[1], physical_line=line)
        for name, q, t, tag, w in (
            ("stride", a, t1, tag1, w1),
            ("spp", b, t2, tag2, w2),
        ):
            item.update(
                {
                    name + "_" + k: q[k]
                    for k in ("instr", "op", "base", "generation", "dispatch", "fill")
                }
            )
            item.update(
                {name + "_ingress": t, name + "_lookup": tag, name + "_head": w}
            )
        by[grp].append(item)
    groups = []
    examples = []
    for g, cohort in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        meta = dict(
            case=case,
            stride_outcome=g[0],
            spp_outcome=g[1],
            consumer_pc=g[2],
            stride_producer_pc=g[3],
            spp_producer_pc=g[4],
            stride_displacement=g[5],
            spp_displacement=g[6],
        )
        a = np.array([r["stride_instr"] for r in cohort])
        b = np.array([r["spp_instr"] for r in cohort])
        groups.append(
            dict(
                meta,
                requests=len(a),
                stride_earlier_original_instruction=int((a < b).sum()),
                same_original_instruction=int((a == b).sum()),
                spp_earlier_original_instruction=int((a > b).sum()),
            )
        )
        examples.append(
            dict(meta, **min(cohort, key=lambda r: (r["target_instr"], r["target_op"])))
        )
    return groups, examples


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", nargs="+", default=["gg_delta"])
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--summaries", type=Path, default=HERE)
    args = p.parse_args()
    if args.output.resolve() != DEFAULT_OUTPUT and args.summaries.resolve() == HERE:
        p.error("Use a separate summary directory for another run window.")
    results = [details(args.output, r["case"]) for r in select(args.case)]
    args.summaries.mkdir(exist_ok=True, parents=True)
    for j, name in enumerate(("producer_pairs.csv", "producer_examples.csv")):
        rows = [row for output in results for row in output[j]]
        with (args.summaries / name).open("w", newline="") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
                w.writeheader()
                w.writerows(rows)
