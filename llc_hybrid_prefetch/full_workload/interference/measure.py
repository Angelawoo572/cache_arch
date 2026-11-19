"""Compare each core against its slot-matched isolated execution at identical work."""

import csv
import gzip
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "raw/native/runs"
ARMS = ("nopf", "stride", "multi", "spp")
ROI = 20_000_000


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def get(name, arm):
    path = RUNS / ("interference_" + name + "__" + arm)
    config = json.loads((path / "complete.json").read_text())["config"]
    if (config["warmup"], config["roi"], config["skip"]) != (5_000_000, ROI, 0):
        raise ValueError(f"Wrong work in {path}")
    cores = table(path / "cores.csv")
    stats = {r["key"]: int(r["value"]) for r in table(path / "stats.csv")}
    phases = table(path / "phases.csv")
    return path, config, cores, stats, phases


def page_map(folder, cpu):
    with gzip.open(folder / "pages.csv.gz", "rt") as f:
        return {
            r["virtual_page"]: (r["page_slot"], r["physical_page"])
            for r in csv.DictReader(f)
            if int(r["cpu"]) == cpu
        }


def outcomes(inputs):
    # Four words refer to the same instruction in NoPF solo, NoPF pair, policy solo, policy pair.
    names = (
        "common_original",
        "both_direct",
        "solo_only_direct",
        "pair_only_direct",
        "solo_timely_pair_late",
        "solo_late_pair_timely",
    )
    total = dict.fromkeys(names, 0)
    readers = [
        gzip.open(path / f"outcomes_core{cpu}.u16le.gz", "rb") for path, cpu in inputs
    ]
    try:
        for start in range(0, ROI, 131072):
            count = min(131072, ROI - start)
            blocks = [f.read(2 * count) for f in readers]
            if any(len(block) != 2 * count for block in blocks):
                raise ValueError("Incomplete dynamic operand alignment")
            a, b, s, p = (int.from_bytes(block, "little") for block in blocks)
            tag_miss_bits = int.from_bytes(b"\x22" * (2 * count), "little")
            common = (a & b & tag_miss_bits) >> 1
            st, pt = (s >> 3) & common, (p >> 3) & common
            late_s, late_p = (s >> 2) & common, (p >> 2) & common
            for name, mask in zip(
                names, (common, st & pt, st & ~pt, pt & ~st, st & late_p, late_s & pt)
            ):
                total[name] += mask.bit_count()
        if any(f.read(1) for f in readers):
            raise ValueError("Outcome file exceeds fixed work")
    finally:
        for f in readers:
            f.close()
    return total


def save(name, rows):
    with (HERE / name).open("w") as f:
        out = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        out.writeheader()
        out.writerows(rows)


def main():
    by_core, by_pair = [], []
    for arm in ARMS:
        pair, cfg, cores, totals, phases = get("pair", arm)
        paired_time = max(int(c["cycles"]) for c in cores)
        row = dict(policy=arm, paired_makespan_cycles=paired_time)
        weighted_speedup = 0
        for cpu, solo_name in enumerate(("mcf", "bfs")):
            solo, scfg, isolated, stats, sphases = get(solo_name, arm)
            if (cfg["traces"][cpu], cfg["slots"][cpu]) != (
                scfg["traces"][0],
                scfg["slots"][0],
            ):
                raise ValueError("Trace/placement mismatch")
            if page_map(solo, 0) != page_map(pair, cpu):
                raise ValueError("Physical mappings differ")
            solo_cycles, paired_cycles = (
                int(isolated[0]["cycles"]),
                int(cores[cpu]["cycles"]),
            )
            r = dict(
                policy=arm,
                core=cpu,
                program=solo_name,
                instructions=ROI,
                solo_cycles=solo_cycles,
                paired_cycles=paired_cycles,
                slowdown=round(paired_cycles / solo_cycles, 6),
                solo_ipc=round(ROI / solo_cycles, 6),
                paired_ipc=round(ROI / paired_cycles, 6),
                solo_head_wait=stats["head_unique_cycles_core_0"],
                paired_head_wait=totals[f"head_unique_cycles_core_{cpu}"],
            )
            weighted_speedup += solo_cycles / paired_cycles
            for col in ("llc_tag_hits", "llc_tag_misses", "timely", "late"):
                r["solo_" + col] = sum(int(q[col]) for q in sphases)
                r["paired_" + col] = sum(
                    int(q[col]) for q in phases if int(q["cpu"]) == cpu
                )
            r.update(
                outcomes(
                    [
                        (get(solo_name, "nopf")[0], 0),
                        (get("pair", "nopf")[0], cpu),
                        (solo, 0),
                        (pair, cpu),
                    ]
                )
            )
            by_core.append(r)
        row["weighted_speedup"] = round(weighted_speedup, 6)
        for key in (
            "dram_demand_bytes",
            "dram_prefetch_bytes",
            "dram_write_bytes",
            "prefetch_accepted",
            "llc_mshr_cycle_sum",
            "dram_rq_cycle_sum",
            "pressure_cycles",
        ):
            row[key] = totals[key]
        by_pair.append(row)
    save("per_core.csv", by_core)
    save("shared_system.csv", by_pair)


if __name__ == "__main__":
    main()
