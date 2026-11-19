"""Keep per-demand, per-request and fixed-work denominators separate."""
from collections import Counter, defaultdict
import csv
import gzip
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "episodes"))
from extract import CASES, runpath


def records(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def ratio(n, d):
    return round(n / d, 7) if d else ""


def main():
    paths = defaultdict(Counter)
    for r in records(HERE / "episodes/path_pairs.csv"):
        paths[r["case"], int(r["core"])][r["nopf_path"], r["stride_path"]] += int(r["demands"])
    summary = {(r["case"], int(r["core"])): r for r in records(HERE / "episodes/summary.csv")
               if r["view"] == "ingress"}
    output = []
    for case in CASES:
        folders = [runpath(case, mode) for mode in ("nopf", "stride")]
        cores = [records(folder / "cores.csv") for folder in folders]
        assert len(cores[0]) == len(cores[1])
        requests = defaultdict(Counter)
        with gzip.open(folders[1] / "candidates.csv.gz", "rt") as f:
            for r in csv.DictReader(f):
                requests[int(r["cpu"])][r["stage"]] += 1

        for c0, c1 in zip(*cores):
            cpu = int(c0["cpu"])
            assert c0["cpu"] == c1["cpu"] and c0["instructions"] == c1["instructions"]
            instructions = int(c0["instructions"])
            row = dict(case=case, core=cpu, instructions=instructions)
            outcomes = paths[case, cpu]
            for j, mode in enumerate(("nopf", "stride")):
                hits = sum(n for pair,n in outcomes.items() if pair[j] == "llc_hit")
                misses = sum(n for pair,n in outcomes.items() if pair[j] == "llc_miss")
                cycles = int((c0, c1)[j]["cycles"])
                row.update({mode+"_llc_tag_hit": hits, mode+"_llc_tag_miss": misses,
                            mode+"_tag_hit_rate": ratio(hits, hits+misses),
                            mode+"_tag_miss_rate": ratio(misses, hits+misses),
                            mode+"_llc_mpki": ratio(misses*1000, instructions),
                            mode+"_cycles": cycles, mode+"_ipc": ratio(instructions, cycles)})
            candidate = requests[cpu]
            assert candidate["proposed"] == candidate["accepted"] + candidate["rejected"]
            assert candidate["accepted"] == sum(candidate[k] for k in ("timely","late","evicted","redundant","unresolved"))
            for k in ("proposed", "rejected", "accepted", "timely", "late", "evicted", "redundant", "unresolved"):
                row["producer_"+k] = candidate[k]
            row["first_use_accuracy_including_late"] = ratio(candidate["timely"]+candidate["late"], candidate["accepted"])
            row["timely_accuracy"] = ratio(candidate["timely"], candidate["accepted"])
            row["timely_fraction_of_used"] = ratio(candidate["timely"], candidate["timely"]+candidate["late"])
            s = summary[case, cpu]
            row["paired_timely"] = int(s["direct_timely"])
            row["paired_late"] = int(s["all_paired_late"])
            row["paired_persistent"] = int(s["all_paired_persistent"])
            row["direct_timely_coverage"] = ratio(row["paired_timely"],row["nopf_llc_tag_miss"])
            row["net_miss_reduction_fraction"] = ratio(row["nopf_llc_tag_miss"]-row["stride_llc_tag_miss"],row["nopf_llc_tag_miss"])
            output.append(row)
    with (HERE / "baseline_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(output[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(output)


if __name__ == "__main__":
    main()
