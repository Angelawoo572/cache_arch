#!/usr/bin/env python3
import argparse, csv, os
from collections import Counter

def to_int(x, default=0):
    try:
        if x is None or x == "":
            return default
        s = str(x).strip()
        if s.startswith("0x") or s.startswith("0X"):
            return int(s, 16)
        return int(float(s))
    except Exception:
        return default

def div(a, b):
    return float(a) / float(b) if b else 0.0
 
ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--trace", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-rows", type=int, default=0)
ap.add_argument("--append", action="store_true")
args = ap.parse_args()

n = 0
issued = useful = duplicate = good = 0
hit = miss = 0
self_cand = zero_cand_delta = nonzero_cand_delta = 0
good_nonzero = useful_nonzero = duplicate_nonzero = 0
top_cand_delta = Counter()
top_spp_delta = Counter()
top_pc = Counter()

with open(args.csv, newline="") as f:
    r = csv.DictReader(f)
    fields = r.fieldnames or []
    for row in r:
        n += 1
        if args.max_rows and n > args.max_rows:
            break

        addr = to_int(row.get("addr"), 0)
        pf_addr = to_int(row.get("pf_addr", row.get("prefetch_addr", 0)), 0)
        line = addr // 64
        pf_line = pf_addr // 64 if pf_addr else line
        cand_delta = pf_line - line

        s = to_int(row.get("spp_issued"), 1)
        u = to_int(row.get("outcome_useful"), 0)
        d = to_int(row.get("outcome_duplicate"), 0)
        h = to_int(row.get("hit", row.get("cache_hit", 0)), 0)
        spp_delta = to_int(row.get("delta", row.get("spp_delta", 0)), 0)
        pc = row.get("pc", "")

        issued += s
        useful += u
        duplicate += d
        good += int(u == 1 and d == 0 and s == 1)

        hit += int(h == 1)
        miss += int(h == 0)

        self_cand += int(pf_line == line)
        zero_cand_delta += int(cand_delta == 0)
        nonzero = int(cand_delta != 0)
        nonzero_cand_delta += nonzero

        if nonzero:
            useful_nonzero += u
            duplicate_nonzero += d
            good_nonzero += int(u == 1 and d == 0 and s == 1)

        top_cand_delta[cand_delta] += 1
        top_spp_delta[spp_delta] += 1
        top_pc[pc] += 1

summary = {
    "trace": args.trace,
    "rows": n,
    "issued": issued,
    "useful": useful,
    "useful_rate": div(useful, n),
    "duplicate": duplicate,
    "duplicate_rate": div(duplicate, n),
    "good_prefetch": good,
    "good_prefetch_rate": div(good, n),
    "hit_rate": div(hit, hit + miss),
    "self_candidate": self_cand,
    "self_candidate_rate": div(self_cand, n),
    "zero_candidate_delta": zero_cand_delta,
    "zero_candidate_delta_rate": div(zero_cand_delta, n),
    "nonzero_candidate_delta": nonzero_cand_delta,
    "nonzero_candidate_delta_rate": div(nonzero_cand_delta, n),
    "useful_nonzero": useful_nonzero,
    "good_nonzero": good_nonzero,
    "duplicate_nonzero": duplicate_nonzero,
    "good_nonzero_rate_among_all": div(good_nonzero, n),
    "good_nonzero_rate_among_nonzero": div(good_nonzero, nonzero_cand_delta),
    "top_candidate_deltas": top_cand_delta.most_common(10),
    "top_spp_deltas": top_spp_delta.most_common(10),
    "top_pcs": top_pc.most_common(5),
}

for k, v in summary.items():
    print("{} = {}".format(k, v))

os.makedirs(os.path.dirname(args.out), exist_ok=True)
need_header = (not os.path.exists(args.out)) or (not args.append)
with open(args.out, "a" if args.append else "w", newline="") as f:
    w = csv.writer(f)
    if need_header:
        w.writerow([
            "trace","rows","useful_rate","duplicate_rate","good_prefetch_rate",
            "hit_rate","self_candidate_rate","zero_candidate_delta_rate",
            "nonzero_candidate_delta_rate","good_nonzero_rate_among_all",
            "good_nonzero_rate_among_nonzero","good_nonzero","top_candidate_deltas"
        ])
    w.writerow([
        summary["trace"], summary["rows"], summary["useful_rate"],
        summary["duplicate_rate"], summary["good_prefetch_rate"],
        summary["hit_rate"], summary["self_candidate_rate"],
        summary["zero_candidate_delta_rate"], summary["nonzero_candidate_delta_rate"],
        summary["good_nonzero_rate_among_all"],
        summary["good_nonzero_rate_among_nonzero"],
        summary["good_nonzero"],
        str(summary["top_candidate_deltas"]),
    ])
