"""Select sparse legal oracle timing interventions from prior matched native runs."""

from collections import defaultdict
import csv
import gzip
from pathlib import Path
from random import Random

HERE = Path(__file__).resolve().parent
FULL = HERE.parents[1] / "full_workload"
RUNS = FULL / "raw/native/runs"
RAW = HERE.parent / "raw/trials"
CASES = ("mcf17_discovery", "bfs_discovery", "soplex06_heldout")


def read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as file:
        yield from csv.DictReader(file)


def dump(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return int(row["instr_id"]), int(row["operand"])


def select(case):
    baseline = RUNS / (case + "__nopf")
    policy = RUNS / (case + "__spp")
    original = {
        key(r): int(r["physical_line"])
        for r in read(baseline / "loads.csv.gz")
        if r["level"] == "LLC" and r["stage"] == "lookup" and r["hit"] == "0"
    }
    spp_misses = {
        key(r)
        for r in read(policy / "loads.csv.gz")
        if r["level"] == "LLC" and r["stage"] == "lookup" and r["hit"] == "0"
    }
    arrivals = [
        r
        for r in read(policy / "loads.csv.gz")
        if r["level"] == "LLC" and r["stage"] == "ingress"
    ]
    assert all(
        int(a["cycle"]) <= int(b["cycle"]) for a, b in zip(arrivals, arrivals[1:])
    )
    uses = {
        (int(r["demand_instr_id"]), int(r["demand_operand"])): r["stage"]
        for r in read(policy / "candidates.csv.gz")
        if r["stage"] in ("timely", "late")
    }
    groups = defaultdict(list)
    for i, demand in enumerate(arrivals):
        ident = key(demand)
        line = int(demand["physical_line"])
        if original.get(ident) != line or ident not in spp_misses:
            continue
        kind = uses.get(ident, "plain")
        if kind == "timely":
            continue
        quarter = (ident[0] - 50_000_001) // 50_000_000
        pos = (ident[0] - 50_000_001) % 50_000_000
        if pos < 4096 or pos >= 249000:
            continue
        early = near = None
        for old in reversed(arrivals[max(i - 64, 0) : i]):
            previous = int(old["physical_line"])
            if (key(old)[0] - 50_000_001) // 50_000_000 != quarter or previous == line:
                break
            if previous // 64 != line // 64:
                continue
            lead = int(demand["cycle"]) - int(old["cycle"])
            if 1 <= lead <= 64 and near is None:
                near = old
            if lead >= 220:
                early = old
                break
        if early is not None and near is not None:
            groups[(quarter, kind)].append((demand, early, near))
    rng = Random(202611)
    targets, plans, used = [], {"early": [], "near": []}, set()
    for (q, kind), options in sorted(groups.items()):
        rng.shuffle(options)
        for demand, early, near in options:
            keys = {key(x) for x in (demand, early, near)}
            if len(keys) != 3 or any(
                abs(a[0] - b[0]) < 1200 for a in keys for b in used
            ):
                continue
            used.update(keys)
            row = dict(
                case=case,
                quarter=q + 1,
                original_spp=kind,
                eligible_targets=len(options),
                target_instr=key(demand)[0],
                target_operand=key(demand)[1],
                target_line=int(demand["physical_line"]),
                target_cycle=int(demand["cycle"]),
            )
            for mode, anchor in (("early", early), ("near", near)):
                delta = row["target_line"] - int(anchor["physical_line"])
                row.update(
                    {
                        mode + "_" + k: v
                        for k, v in {
                            "instr": key(anchor)[0],
                            "operand": key(anchor)[1],
                            "base": int(anchor["physical_line"]),
                            "cycle": int(anchor["cycle"]),
                            "lead": row["target_cycle"] - int(anchor["cycle"]),
                            "mshr": int(anchor["mshr"]),
                            "dram_rq": int(anchor["dram_rq"]),
                        }.items()
                    }
                )
                plans[mode].append(
                    dict(
                        cpu=0,
                        instr_id=key(anchor)[0],
                        operand=key(anchor)[1],
                        delta=delta,
                    )
                )
            targets.append(row)
            break
    if not targets:
        raise ValueError(f"No matched candidates in {case}")
    for mode in plans:
        dump(HERE / (case + "_" + mode + ".csv"), plans[mode])
    return targets


if __name__ == "__main__":
    dump(HERE / "targets.csv", [r for case in CASES for r in select(case)])
