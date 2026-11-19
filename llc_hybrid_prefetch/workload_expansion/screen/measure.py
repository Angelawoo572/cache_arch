"""Reconcile every emitted load, demand path and request, then report fixed-work results."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run import DEFAULT_OUTPUT, HERE, POLICIES, STUDY, select, table

LAYOUT = np.dtype(
    [
        ("pc", "<u8"),
        ("branch", "u1"),
        ("taken", "u1"),
        ("reg_dst", "u1", (2,)),
        ("reg_src", "u1", (4,)),
        ("mem_dst", "<u8", (2,)),
        ("mem_src", "<u8", (4,)),
    ]
)


def ratio(a, b):
    return round(float(a) / b, 8) if b else ""


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        w = csv.DictWriter(
            file,
            fieldnames=(
                list(rows[0]) if rows else ["case", "policy", "reason", "proposals"]
            ),
            lineterminator="\n",
        )
        w.writeheader()
        w.writerows(rows)


def counts(path):
    with gzip.open(path, "rt") as f:
        return Counter((r["stage"], r["reason"]) for r in csv.DictReader(f))


def measure(job):
    row, root = job
    case = row["case"]
    baseline = root / "runs" / case / "nopf"
    meta = json.loads((baseline / "complete.json").read_text())
    skip, warmup, roi = (meta[k] for k in ("skip", "warmup", "roi"))
    trace = np.memmap(STUDY / meta["trace_path"], mode="r", dtype=LAYOUT)
    assert LAYOUT.itemsize == 64
    records = trace[skip + warmup : skip + warmup + roi]
    assert len(records) == roi and (records["pc"] != 0).all()
    assert (
        np.isin(records["branch"], [0, 1]).all()
        and np.isin(records["taken"], [0, 1]).all()
    )
    source_i, op = np.nonzero(records["mem_src"])
    ids = source_i + skip + warmup + 1
    ips = records["pc"][source_i]
    vlines = records["mem_src"][source_i, op] // 64
    expected_maps = None
    rows = []
    audits = []
    flows = []
    rejected = []
    base_cycles = None
    base_misses = None
    for policy in POLICIES:
        folder = baseline.parent / policy
        arm = json.loads((folder / "complete.json").read_text())
        assert (
            arm["skip"] == skip
            and arm["warmup"] == warmup
            and arm["roi"] == roi
            and arm["trace_path"] == meta["trace_path"]
        )
        keys = ["cpu", "page_slot", "virtual_page", "physical_page"]
        maps = pd.read_csv(folder / "pages.csv.gz", usecols=keys)
        assert (
            (maps.cpu == 0).all()
            and (maps.page_slot == 0).all()
            and not maps.virtual_page.duplicated().any()
        )
        mapping = dict(zip(maps.virtual_page, maps.physical_page))
        if expected_maps is None:
            expected_maps = mapping
        if mapping != expected_maps:
            raise ValueError(f"Unstable page map {case}/{policy}")
        df = pd.read_csv(
            folder / "loads.csv.gz",
            usecols=[
                "cpu",
                "instr_id",
                "operand",
                "pc",
                "virtual_line",
                "physical_line",
                "level",
                "stage",
                "hit",
            ],
        )
        core = df[df.level == "CORE"]
        assert (core.stage == "trace").all() and (df.cpu == 0).all()
        assert np.array_equal(core.instr_id.to_numpy(), ids) and np.array_equal(
            core.operand.to_numpy(), op
        )
        assert np.array_equal(core.pc.to_numpy(), ips) and np.array_equal(
            core.virtual_line.to_numpy(), vlines
        )
        arrival = df[(df.level == "LLC") & (df.stage == "ingress")]
        i = arrival.instr_id.to_numpy(dtype="int64") - skip - warmup - 1
        op_a = arrival.operand.to_numpy(dtype="int64")
        assert np.all((i >= 0) & (i < roi)) and np.all((op_a >= 0) & (op_a < 4))
        va = records["mem_src"][i, op_a] // 64
        assert np.array_equal(va, arrival.virtual_line.to_numpy())
        assert np.array_equal(records["pc"][i], arrival.pc.to_numpy())
        pa = np.fromiter(
            (mapping[int(v) // 64] * 64 + int(v) % 64 for v in va),
            dtype="int64",
            count=len(va),
        )
        assert np.array_equal(pa, arrival.physical_line.to_numpy())
        zpath = root / "history" / case / (policy + ".npz")
        with np.load(zpath) as archive:
            z = {k: archive[k] for k in archive.files}
        for a, b in [
            ("instr", "instr_id"),
            ("op", "operand"),
            ("pa", "physical_line"),
            ("pc", "pc"),
        ]:
            assert np.array_equal(z[a], arrival[b].to_numpy())
        stats = {r["key"]: int(r["value"]) for r in table(folder / "stats.csv")}
        coretime = table(folder / "cores.csv")
        cycles = int(coretime[0]["cycles"])
        ins = int(coretime[0]["instructions"])
        assert ins == roi and int(coretime[0]["warmup"]) == warmup
        values = dict(
            case=case,
            collection=row["collection"],
            policy=policy,
            skip=skip,
            retired=ins,
            cycles=cycles,
            ipc=ratio(ins, cycles),
        )
        if policy == "nopf":
            base_cycles = cycles
        values["cycles_saved_vs_nopf"] = base_cycles - cycles
        values["time_saved_fraction_vs_nopf"] = ratio(base_cycles - cycles, base_cycles)
        values["traced_load_operands"] = len(core)
        values["llc_ingresses"] = len(arrival)
        for level in ("L2", "LLC"):
            lookup = df[(df.level == level) & (df.stage == "lookup")]
            n_hit = int((lookup.hit == 1).sum())
            n_miss = int((lookup.hit == 0).sum())
            assert (
                n_hit + n_miss == len(lookup)
                and not lookup[["instr_id", "operand"]].duplicated().any()
            )
            values.update(
                {
                    level.lower() + "_hit": n_hit,
                    level.lower() + "_miss": n_miss,
                    level.lower() + "_hit_rate": ratio(n_hit, len(lookup)),
                    level.lower() + "_miss_rate": ratio(n_miss, len(lookup)),
                    level.lower() + "_mpki": round(1000 * n_miss / ins, 6),
                }
            )
        if policy == "nopf":
            base_misses = values["llc_miss"]
        assert values["llc_miss"] == int((z["hit"] == 0).sum())
        cand = counts(folder / "candidates.csv.gz")
        stages = Counter()
        for (stage, reason), n in cand.items():
            stages[stage] += n
            if stage == "rejected":
                rejected.append(
                    dict(case=case, policy=policy, reason=reason, proposals=n)
                )
        accepted = stages["accepted"]
        assert (
            accepted == stats["prefetch_accepted"]
            and stages["proposed"] == stages["rejected"] + accepted
        )
        assert accepted == sum(
            stages[k] for k in ("timely", "late", "evicted", "redundant", "unresolved")
        )
        if policy == "nopf":
            assert stages["proposed"] == 0
        base = z["baseline"] == 0
        timely = base & (z["pf_kind"] == 1) & (z["hit"] == 1)
        late = base & (z["pf_kind"] == 2) & (z["hit"] == 0)
        persistent = base & (z["hit"] == 0) & ~late
        ordinary = base & (z["hit"] == 1) & ~timely
        upstream = base_misses - int(
            sum(x.sum() for x in [timely, late, persistent, ordinary])
        )
        assert upstream >= 0
        values.update(
            original_misses=base_misses,
            original_to_direct_timely=int(timely.sum()),
            original_to_late=int(late.sum()),
            original_to_persistent=int(persistent.sum()),
            original_to_ordinary_hit=int(ordinary.sum()),
            original_no_arm_llc_lookup=upstream,
            direct_timely_coverage=ratio(timely.sum(), base_misses),
            net_miss_reduction=ratio(base_misses - values["llc_miss"], base_misses),
        )
        for k in (
            "proposed",
            "accepted",
            "rejected",
            "dispatch",
            "fill",
            "timely",
            "late",
            "evicted",
            "redundant",
            "unresolved",
        ):
            values["pf_" + k] = stages[k]
        values.update(
            useful_accuracy_including_late=ratio(
                stages["timely"] + stages["late"], accepted
            ),
            timely_accuracy=ratio(stages["timely"], accepted),
            timely_fraction_of_used=ratio(
                stages["timely"], stages["timely"] + stages["late"]
            ),
        )
        for k in ("dram_demand_bytes", "dram_prefetch_bytes", "dram_write_bytes"):
            values[k] = stats[k]
        values["dram_total_bytes"] = sum(
            values[k]
            for k in ("dram_demand_bytes", "dram_prefetch_bytes", "dram_write_bytes")
        )
        values["unique_rob_head_load_wait_cycles"] = stats["head_unique_cycles_core_0"]
        values["head_wait_fraction"] = ratio(
            values["unique_rob_head_load_wait_cycles"], cycles
        )
        values["avg_llc_mshr"] = ratio(
            stats["llc_mshr_cycle_sum"], stats["pressure_cycles"]
        )
        values["avg_dram_read_queue"] = ratio(
            stats["dram_rq_cycle_sum"], stats["pressure_cycles"]
        )
        rows.append(values)
        audits.append(
            dict(
                case=case,
                policy=policy,
                original_first_id=skip + warmup + 1,
                original_last_id=skip + warmup + roi,
                raw_roi_load_operands=len(ids),
                exact_native_core_matches=len(core),
                verified_virtual_pages=len(mapping),
                exact_ingress_pc_operand_physical_line_matches=len(arrival),
                stage_rows=len(df),
                initial_mshr=stats["roi_start_llc_mshr"],
                initial_dram_read_queue=stats["roi_start_dram_rq"],
            )
        )
        for (lvl, stage, hit), n in (
            df.groupby(["level", "stage", "hit"]).size().items()
        ):
            flows.append(
                dict(
                    case=case,
                    policy=policy,
                    level=lvl,
                    stage=stage,
                    hit=hit,
                    dynamic_events=int(n),
                )
            )
    return rows, audits, flows, rejected


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--summaries", type=Path, default=HERE)
    p.add_argument("--case", nargs="+")
    p.add_argument("--jobs", type=int, default=4)
    a = p.parse_args()
    if a.output.resolve() != DEFAULT_OUTPUT and a.summaries.resolve() == HERE:
        p.error("Use a separate --summaries directory for a different run root.")
    jobs = [(r, a.output.resolve()) for r in select(a.case)]
    with ProcessPoolExecutor(a.jobs) as ex:
        result = list(ex.map(measure, jobs))
    for j, name in enumerate(
        ("performance.csv", "identity_audit.csv", "cache_events.csv", "rejections.csv")
    ):
        write(a.summaries / name, [r for item in result for r in item[j]])
    print(
        "Validated every traced ROI load, LLC ingress, placement, lifecycle and retirement interval in",
        len(jobs) * 4,
        "arms.",
    )
