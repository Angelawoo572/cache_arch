"""Join complete native outcomes from the existing conventional and new arithmetic runs."""

import csv
import gzip
import json
from collections import Counter
from contextlib import ExitStack
from run_mcf import CASES, FULL, HERE, POLICIES, RAW, ROI, WARMUP, native

CONVENTIONAL = ("stride", "multi", "spp", "temporal")
ARMS = ("nopf", *CONVENTIONAL, *POLICIES)


def folder(case, arm):
    raw = RAW if arm in POLICIES else FULL / "raw/native"
    return raw / "runs" / (case + "__" + arm)


def write_table(name, rows):
    path = HERE / name
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def load(case):
    origin = 1_750_000_000 if case == "mcf17_discovery_far" else 0
    reference = None
    maps = None
    phases, stats = {}, {}
    for arm in ARMS:
        path = folder(case, arm)
        saved = json.loads((path / "complete.json").read_text())
        config = dict(saved["config"])
        assert config.pop("policy") == arm
        if reference is None:
            reference = config
        if (
            config != reference
            or config.get("source_ordinal_base", 0) != origin
            or (config["warmup"], config["roi"], config["skip"], config["slots"])
            != (WARMUP, ROI, 0, [0])
        ):
            raise ValueError(f"{case}/{arm}: unmatched fixed-work input or page seed")
        with gzip.open(path / "pages.csv.gz", "rt") as f:
            arm_map = {
                int(r["virtual_page"]): int(r["physical_page"])
                for r in csv.DictReader(f)
            }
        if maps is not None and arm_map != maps:
            raise ValueError(f"{case}/{arm}: physical page placement differs")
        maps = arm_map
        phases[arm] = [
            {k: int(v) for k, v in row.items()}
            for row in native.rows(path / "phases.csv")
        ]
        for q, phase in enumerate(phases[arm]):
            first = origin + WARMUP + q * ROI // 4 + 1
            last = origin + WARMUP + (q + 1) * ROI // 4
            if (
                phase["cpu"],
                phase["quarter"],
                phase["first_ordinal"],
                phase["last_ordinal"],
            ) != (0, q, first, last):
                raise ValueError(f"{case}/{arm}: unexpected phase boundaries")
        stats[arm] = {
            r["key"]: int(r["value"]) for r in native.rows(path / "stats.csv")
        }
        (core,) = saved["core_results"]
        if int(core["instructions"]) != ROI or int(core["cycles"]) != sum(
            p["cycles"] for p in phases[arm]
        ):
            raise ValueError("Incomplete retirement")
    return phases, stats


def join(case):
    quarters = []
    with ExitStack() as stack:
        files = {
            a: stack.enter_context(
                gzip.open(folder(case, a) / "outcomes_core0.u16le.gz", "rb")
            )
            for a in ARMS
        }
        for _ in range(4):
            counts = Counter()
            for pos in range(0, ROI // 4, 131072):
                size = min(131072, ROI // 4 - pos) * 2
                data = {a: f.read(size) for a, f in files.items()}
                if any(len(b) != size for b in data.values()):
                    raise ValueError("Partial outcome stream")
                bits = {a: int.from_bytes(b, "little") for a, b in data.items()}
                slots = int.from_bytes(b"\x11" * size, "little")
                original = (bits["nopf"] >> 1) & slots
                counts["original"] += original.bit_count()
                timely = {}
                for a, b in bits.items():
                    direct = (b >> 3) & slots
                    timely[a] = direct & original
                    counts[(a, "original_timely")] += timely[a].bit_count()
                    counts[(a, "all_timely")] += direct.bit_count()
                    counts[(a, "tag_miss")] += ((b >> 1) & slots).bit_count()
                union = 0
                for a in CONVENTIONAL:
                    union |= timely[a]
                shared = timely["shared_increment"]
                counts["new_vs_union"] += (shared & ~union).bit_count()
                counts["lost_vs_union"] += (union & ~shared).bit_count()
                for ref in (*CONVENTIONAL, "local"):
                    gained = shared & ~timely[ref]
                    counts[(ref, "gained")] += gained.bit_count()
                    counts[(ref, "lost")] += (timely[ref] & ~shared).bit_count()
                    counts[(ref, "late_to_timely")] += (
                        gained & (bits[ref] >> 2)
                    ).bit_count()
            quarters.append(counts)
        if any(f.read(1) for f in files.values()):
            raise ValueError("Outcomes exceed the ROI")
    return quarters


def percentage(x, y):
    return round(100 * x / y, 6) if y else ""


def analyze(case):
    phases, stats = load(case)
    quarters = join(case)
    totals = sum(quarters, Counter())
    perf, comparisons = [], []
    for arm in ARMS:
        p, s = phases[arm], stats[arm]
        for q, counts in enumerate(quarters):
            if (
                counts[arm, "tag_miss"] != p[q]["llc_tag_misses"]
                or counts[arm, "all_timely"] != p[q]["timely"]
            ):
                raise ValueError(
                    f"{case}/{arm}: packed outcome does not match native counters in Q{q + 1}"
                )
        cycles = sum(q["cycles"] for q in p)
        accepted = s["prefetch_accepted"]
        timely = totals[arm, "all_timely"]
        late = sum(q["late"] for q in p)
        perf.append(
            dict(
                case=case,
                arm=arm,
                roi=ROI,
                cycles=cycles,
                ipc=round(ROI / cycles, 6),
                original_misses=totals["original"],
                original_direct_timely=totals[arm, "original_timely"],
                timely_coverage_pct=percentage(
                    totals[arm, "original_timely"], totals["original"]
                ),
                accepted=accepted,
                all_timely=timely,
                all_late=late,
                useful_accuracy_pct=percentage(timely + late, accepted),
                timely_accuracy_pct=percentage(timely, accepted),
                evicted_unused=s.get("study_pf_evicted", 0),
                l2_mpki=round(1000 * sum(q["l2_tag_misses"] for q in p) / ROI, 6),
                llc_mpki=round(1000 * sum(q["llc_tag_misses"] for q in p) / ROI, 6),
                dram_demand_bytes=s["dram_demand_bytes"],
                dram_prefetch_bytes=s["dram_prefetch_bytes"],
                mean_llc_mshrs=round(s["llc_mshr_cycle_sum"] / s["pressure_cycles"], 6),
                mean_dram_read_queue=round(
                    s["dram_rq_cycle_sum"] / s["pressure_cycles"], 6
                ),
                head_wait_cycles=s["head_unique_cycles_core_0"],
                grants=s.get("study_arithmetic_grants", 0),
                agreed=s.get("study_arithmetic_agrees", 0),
                displaced_offers=s.get("study_arithmetic_displaces", 0),
                new_offers=s.get("study_arithmetic_extra", 0),
            )
        )
    for index, counts in enumerate([*quarters, totals]):
        indices = range(4) if index == 4 else (index,)
        cycles = sum(phases["shared_increment"][q]["cycles"] for q in indices)
        for ref in (*CONVENTIONAL, "local"):
            ref_cycles = sum(phases[ref][q]["cycles"] for q in indices)
            gain, loss = counts[ref, "gained"], counts[ref, "lost"]
            assert (
                gain - loss
                == counts["shared_increment", "original_timely"]
                - counts[ref, "original_timely"]
            )
            comparisons.append(
                dict(
                    case=case,
                    interval="all" if index == 4 else f"Q{index + 1}",
                    reference=ref,
                    shared_increment_cycles=cycles,
                    reference_cycles=ref_cycles,
                    time_saved_pct=percentage(ref_cycles - cycles, ref_cycles),
                    original_misses=counts["original"],
                    timely_gained=gain,
                    timely_lost=loss,
                    net_timely=gain - loss,
                    reference_late_to_shared_timely=counts[ref, "late_to_timely"],
                )
            )
    union = dict(
        case=case,
        original_misses=totals["original"],
        shared_timely_none_of_four=totals["new_vs_union"],
        any_of_four_timely_shared_not=totals["lost_vs_union"],
    )
    return perf, comparisons, union


if __name__ == "__main__":
    results = [analyze(case) for case in CASES]
    write_table("performance.csv", [row for result in results for row in result[0]])
    write_table("paired.csv", [row for result in results for row in result[1]])
    write_table("conventional_union.csv", [result[2] for result in results])
    print(
        "Validated seven matched native arms and four full-length operand joins for every MCF window."
    )
