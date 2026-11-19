"""Compare look2 with the matched frozen structural rule and ordinary Stride."""

import csv
import gzip
import json
from collections import Counter
from contextlib import ExitStack

from run_lookahead import CASES, FULL, HERE, MEASURED, WARMUP

ARMS = ("nopf", "stride", "shape", "look2")
REFERENCES = ("stride", "shape")


def read_rows(file):
    with file.open() as stream:
        return list(csv.DictReader(stream))


def page_map(folder):
    with gzip.open(folder / "pages.csv.gz", "rt") as f:
        return {
            int(r["virtual_page"]): int(r["physical_page"]) for r in csv.DictReader(f)
        }


def run_data(case):
    paths, phases, stats = {}, {}, {}
    mapping = None
    for arm in ARMS:
        base = FULL / ("raw/discovery/full" if arm == "shape" else "raw/native")
        folder = base / "runs" / (case + "__" + arm)
        saved = json.loads((folder / "complete.json").read_text())
        config = saved["config"]
        expected = dict(
            case=case,
            policy=arm,
            warmup=WARMUP,
            roi=MEASURED,
            slots=[0],
            skip=0,
            sample_width=250000,
            traces=[
                f"llc_hybrid_prefetch/full_workload/raw/traces/{case}.champsimtrace"
            ],
        )
        if config != expected:
            raise ValueError(f"{case}/{arm}: mismatched input, work or instrumentation")
        actual = page_map(folder)
        if mapping is not None and actual != mapping:
            raise ValueError(f"{case}/{arm}: physical line map differs")
        mapping = actual
        paths[arm] = folder
        phases[arm] = [
            {key: int(value) for key, value in row.items()}
            for row in read_rows(folder / "phases.csv")
        ]
        stats[arm] = {
            r["key"]: int(r["value"]) for r in read_rows(folder / "stats.csv")
        }
        (core,) = saved["core_results"]
        if int(core["instructions"]) != MEASURED or sum(
            p["cycles"] for p in phases[arm]
        ) != int(core["cycles"]):
            raise ValueError(f"{case}/{arm}: incomplete retirement")
    return paths, phases, stats


def joins(paths):
    # Four independent four-bit operand slots per instruction, in retirement order.
    quarter_records = MEASURED // 4
    result = []
    with ExitStack() as stack:
        files = {
            a: stack.enter_context(
                gzip.open(paths[a] / "outcomes_core0.u16le.gz", "rb")
            )
            for a in ARMS
        }
        for _ in range(4):
            counts = Counter()
            for start in range(0, quarter_records, 131072):
                size = 2 * min(131072, quarter_records - start)
                blocks = {a: f.read(size) for a, f in files.items()}
                if any(len(b) != size for b in blocks.values()):
                    raise ValueError("Truncated operand stream")
                words = {a: int.from_bytes(b, "little") for a, b in blocks.items()}
                positions = int.from_bytes(b"\x11" * size, "little")
                original = (words["nopf"] >> 1) & positions
                counts["original"] += original.bit_count()
                timely = {}
                for a in ARMS:
                    timely[a] = (words[a] >> 3) & original
                    counts["direct_" + a] += timely[a].bit_count()
                    counts["all_tag_miss_" + a] += (
                        (words[a] >> 1) & positions
                    ).bit_count()
                    counts["all_timely_" + a] += (
                        (words[a] >> 3) & positions
                    ).bit_count()
                for ref in REFERENCES:
                    gain = timely["look2"] & ~timely[ref]
                    counts["gained_" + ref] += gain.bit_count()
                    counts["lost_" + ref] += (
                        timely[ref] & ~timely["look2"]
                    ).bit_count()
                    counts["kept_" + ref] += (timely[ref] & timely["look2"]).bit_count()
                    counts["ref_late_to_direct_" + ref] += (
                        gain & (words[ref] >> 2)
                    ).bit_count()
            result.append(counts)
        if any(f.read(1) for f in files.values()):
            raise ValueError("Unexpected extra outcome words")
    return result


def percent(n, d):
    return round(100 * n / d, 6) if d else ""


def analyze(case):
    paths, phases, stats = run_data(case)
    masks = joins(paths)
    totals = sum(masks, Counter())
    performance = []
    comparisons = []
    for a in ARMS:
        p, s = phases[a], stats[a]
        for q, counts in enumerate(masks):
            expected = (
                q,
                WARMUP + q * MEASURED // 4 + 1,
                WARMUP + (q + 1) * MEASURED // 4,
            )
            if (
                p[q]["quarter"],
                p[q]["first_ordinal"],
                p[q]["last_ordinal"],
            ) != expected:
                raise ValueError("Unexpected quarter boundaries")
            if (counts["all_tag_miss_" + a], counts["all_timely_" + a]) != (
                p[q]["llc_tag_misses"],
                p[q]["timely"],
            ):
                raise ValueError("Packed operand outcomes differ from native counts")
        cycles = sum(r["cycles"] for r in p)
        accepted = s["prefetch_accepted"]
        timely = sum(r["timely"] for r in p)
        late = sum(r["late"] for r in p)
        llc_misses = sum(r["llc_tag_misses"] for r in p)
        performance.append(
            dict(
                case=case,
                policy=a,
                instructions=MEASURED,
                cycles=cycles,
                ipc=round(MEASURED / cycles, 6),
                original_misses=totals["original"],
                original_direct_timely=totals["direct_" + a],
                timely_coverage_pct=percent(totals["direct_" + a], totals["original"]),
                accepted=accepted,
                all_timely_first_uses=timely,
                all_late_merges=late,
                timely_accuracy_pct=percent(timely, accepted),
                useful_accuracy_pct=percent(timely + late, accepted),
                llc_mpki=round(1000 * llc_misses / MEASURED, 6),
                l2_mpki=round(1000 * sum(r["l2_tag_misses"] for r in p) / MEASURED, 6),
                dram_read_bytes=s["dram_demand_bytes"] + s["dram_prefetch_bytes"],
                dram_prefetch_bytes=s["dram_prefetch_bytes"],
                head_wait_cycles=s["head_unique_cycles_core_0"],
            )
        )
    for ref in REFERENCES:
        for q, counts in enumerate([*masks, totals]):
            chosen = list(range(4)) if q == 4 else [q]
            target_cycles = sum(phases["look2"][i]["cycles"] for i in chosen)
            ref_cycles = sum(phases[ref][i]["cycles"] for i in chosen)
            kept, gained, lost = (
                counts[name + "_" + ref] for name in ("kept", "gained", "lost")
            )
            if (kept + gained, kept + lost) != (
                counts["direct_look2"],
                counts["direct_" + ref],
            ):
                raise ValueError("Paired original first-use counts do not partition")
            comparisons.append(
                dict(
                    case=case,
                    interval="all" if q == 4 else f"Q{q + 1}",
                    reference=ref,
                    reference_cycles=ref_cycles,
                    look2_cycles=target_cycles,
                    look2_time_saved_pct=percent(
                        ref_cycles - target_cycles, ref_cycles
                    ),
                    original_misses=counts["original"],
                    kept=kept,
                    gained=gained,
                    lost=lost,
                    net_timely=gained - lost,
                    ref_late_to_look2_timely=counts["ref_late_to_direct_" + ref],
                )
            )
    return performance, comparisons


if __name__ == "__main__":
    joined = [analyze(case) for case in CASES]
    for filename, groups in zip(("performance.csv", "paired.csv"), zip(*joined)):
        data = [r for group in groups for r in group]
        destination = HERE / filename
        tmp = destination.with_suffix(".tmp")
        with tmp.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=data[0], lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
        tmp.replace(destination)
    print(
        "Validated two fixed-work holdouts, eight matched runs and their exact per-operand joins."
    )
