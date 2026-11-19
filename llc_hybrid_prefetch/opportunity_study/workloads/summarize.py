"""Join per-source labels and actual accesses without matching simulated cycles."""

import argparse, csv, gzip
from array import array
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from cases import RAW, ROOT, TRACES, specs

OUT = ROOT / "workloads"
FIELDS = (
    "case",
    "core",
    "slice",
    "label",
    "loads",
    "base_ingress",
    "stride_ingress",
    "base_llc_miss",
    "stride_llc_miss",
    "miss_hit",
    "miss_hit_firstuse",
    "miss_miss",
    "miss_no_lookup",
    "hit_miss",
    "stride_late_merge",
    "base_operand_head_cycles",
    "stride_operand_head_cycles",
)


def compressed(path):
    with gzip.open(path, "rt") as stream:
        read = csv.reader(stream)
        next(read)
        yield from read


def read_arm(folder, cores):
    data = [
        dict(source=array("Q"), tags={}, ingress=set(), head={}, late=set(), use=set())
        for _ in range(cores)
    ]
    for r in compressed(folder / "loads.csv.gz"):
        level, event = r[6], r[7]
        if level not in ("CORE", "LLC"):
            continue
        c = data[int(r[0])]
        key = int(r[1]) * 4 + int(r[2])
        if level == "CORE":
            c["source"].extend((key, int(r[3]), int(r[4])))
        elif event == "lookup":
            if key in c["tags"]:
                raise ValueError("duplicated completed lookup: " + str(folder))
            c["tags"][key] = int(r[9])
        elif event == "ingress":
            c["ingress"].add(key)
    for r in compressed(folder / "head.csv.gz"):
        data[int(r[0])]["head"][int(r[1]) * 4 + int(r[2])] = int(r[3])
    for r in compressed(folder / "candidates.csv.gz"):
        if r[8] in ("timely", "late") and int(r[10]) >= 0:
            data[int(r[10])]["use" if r[8] == "timely" else "late"].add(int(r[11]) * 4 + int(r[12]))
    with (folder / "cores.csv").open() as f:
        cycles = [int(r["cycles"]) for r in csv.DictReader(f)]
    with (folder / "stats.csv").open() as f:
        stats = {r["key"]: int(r["value"]) for r in csv.DictReader(f)}
    return data, cycles, stats


def one(spec):
    name = spec["case"]
    basep = RAW / "runs" / (name + "-nopf")
    pfp = RAW / "runs" / (name + "-stride")
    if not (basep / "complete.json").exists() or not (pfp / "complete.json").exists():
        raise RuntimeError("missing pair " + name)
    a, cyc0, stats0 = read_arm(basep, len(spec["traces"]))
    b, cyc1, stats1 = read_arm(pfp, len(spec["traces"]))
    labels = {}
    if spec["group"] in ("compiled", "synthetic"):
        labels = {
            int(r[1]) * 4 + int(r[2]): r[5]
            for r in compressed(TRACES / (name + ".labels.csv.gz"))
            if r[4] == "roi"
        }
    out = []
    mid = spec["skip"] + spec["warmup"] + spec["roi"] // 2
    for c, (d0, d1) in enumerate(zip(a, b)):
        if d0["source"] != d1["source"]:
            raise ValueError("unmatched original accesses for " + name)
        counts = defaultdict(lambda: [0] * (len(FIELDS) - 4))
        source = d0["source"]
        base, other = d0["tags"], d1["tags"]
        for i in range(0, len(source), 3):
            k = source[i]
            n = base.get(k)
            s = other.get(k)
            v = (
                1,
                int(k in d0["ingress"]),
                int(k in d1["ingress"]),
                int(n == 0),
                int(s == 0),
                int(n == 0 and s == 1),
                int(n == 0 and s == 1 and k in d1["use"]),
                int(n == 0 and s == 0),
                int(n == 0 and s is None),
                int(n == 1 and s == 0),
                int(k in d1["late"]),
                d0["head"].get(k, 0),
                d1["head"].get(k, 0),
            )
            # All-source rows remain available even when a group has no hardware-visible triggers.
            groups = [("full", "all")]
            if labels:
                kind = labels.get(k, "other_compiled_load")
                groups.append(("full", kind))
                groups.append(("first" if k // 4 <= mid else "second", kind))
            for key in groups:
                accum = counts[key]
                for j, value in enumerate(v):
                    accum[j] += value
        for (sl, l), cnt in sorted(counts.items()):
            out.append(dict(zip(FIELDS, (name, c, sl, l, *cnt))))
    perf = dict(
        case=name,
        group=spec["group"],
        split=spec["split"],
        nopf_cycles=max(cyc0),
        stride_cycles=max(cyc1),
        stride_speedup=max(cyc0) / max(cyc1),
        nopf_core_cycles="/".join(map(str, cyc0)),
        stride_core_cycles="/".join(map(str, cyc1)),
    )
    for tag, stats in (("nopf", stats0), ("stride", stats1)):
        perf[tag + "_dram_bytes"] = sum(
            stats["dram_" + k + "_bytes"] for k in ("demand", "write", "prefetch")
        )
        perf[tag + "_prefetch_dram_bytes"] = stats["dram_prefetch_bytes"]
        perf[tag + "_avg_mshr"] = stats["llc_mshr_cycle_sum"] / stats["pressure_cycles"]
        perf[tag + "_avg_dram_rq"] = stats["dram_rq_cycle_sum"] / stats["pressure_cycles"]
    return out, perf


def write(name, rows):
    with (OUT / name).open("w", newline="") as f:
        table = csv.DictWriter(f, fieldnames=list(rows[0]))
        table.writeheader()
        table.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=3)
    arg = p.parse_args()
    with ProcessPoolExecutor(arg.jobs) as ex:
        groups = list(ex.map(one, specs()))
    write("outcomes.csv", [r for out, _ in groups for r in out])
    write("performance.csv", [perf for _, perf in groups])
    print(
        "verified original PC/virtual-line/operand sequences and summarized",
        len(groups),
        "policy pairs",
    )
