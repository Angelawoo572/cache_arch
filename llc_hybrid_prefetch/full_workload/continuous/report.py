"""Verify complete native arrival exports, compare the same targets in two orders and report sampling effects."""

import argparse
import csv
import filecmp
import gzip
import struct
import subprocess

from run_arrivals import CASES, HERE, NATIVE, RAW


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write(path, data):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def detail(folder):
    with gzip.open(folder / "loads.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            if r["level"] == "LLC" and r["stage"] == "ingress":
                yield (
                    (int(r["instr_id"]) << 2) | int(r["operand"]),
                    int(r["pc"]),
                    int(r["physical_line"]),
                    int(r["cycle"]),
                )


def audit_islands(folder, start, roi):
    expected = detail(folder)
    observed = total = 0
    with gzip.open(folder / "arrival.u64le.gz", "rb") as f:
        while chunk := f.read(32 * 65536):
            for row in struct.iter_unpack("<QQQQ", chunk):
                total += 1
                pos = (row[0] >> 2) - start
                q = pos * 4 // roi
                if pos - (q * roi + 3) // 4 < 250000:
                    if next(expected, None) != row:
                        raise ValueError(
                            "Chronological ordinal/operand/PC/physical line/cycle mismatch"
                        )
                    observed += 1
    if next(expected, None) is not None:
        raise ValueError("Sampled ingress missing from complete chronology")
    return total, observed


def compile_analysis():
    subprocess.run(
        [
            "c++",
            "-std=c++11",
            "-O3",
            str(HERE / "analyze.cc"),
            "-lz",
            "-o",
            str(RAW / "analyze"),
        ],
        check=True,
    )


def process(case):
    folder = RAW / "runs" / (case + "__stride")
    ordinary = NATIVE / (case + "__stride")
    baseline = NATIVE / (case + "__nopf")
    if not all((d / "complete.json").exists() for d in (folder, ordinary, baseline)):
        raise ValueError("Continuous/native/NoPF execution has not completed: " + case)
    for name in (
        "cores.csv",
        "phases.csv",
        "stats.csv",
        "loads.csv.gz",
        "candidates.csv.gz",
        "head.csv.gz",
        "pages.csv.gz",
        "outcomes_core0.u16le.gz",
    ):
        if not filecmp.cmp(folder / name, ordinary / name, shallow=False):
            raise ValueError("Passive replay changed reference " + name)
    quarters = rows(folder / "phases.csv")
    start = int(quarters[0]["first_ordinal"])
    roi = sum(int(r["last_ordinal"]) - int(r["first_ordinal"]) + 1 for r in quarters)
    total, sample = audit_islands(folder, start, roi)
    if total != sum(int(r["llc_ingress"]) for r in quarters):
        raise ValueError("Whole order omits a native ingress")
    dest = RAW / "by_view" / case
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(RAW / "analyze"),
            case,
            str(folder / "arrival.u64le.gz"),
            str(baseline / "outcomes_core0.u16le.gz"),
            str(ordinary / "outcomes_core0.u16le.gz"),
            str(start),
            str(roi),
            str(dest),
        ],
        check=True,
    )
    data = rows(dest / "counts.csv")

    def one(pop, order, cohort, scope):
        return next(
            r
            for r in data
            if (r["window"], r["order"], r["cohort"], r["scope"])
            == (pop, order, cohort, scope)
        )

    for pop in ("full", "four_islands", *(f"sample_q{q}" for q in range(4))):
        for cohort in ("all_admitted", "nopf_miss"):
            for scope in ("global", "pc", "page", "region256k"):
                a = one(pop, "arrival", cohort, scope)
                b = one(pop, "program", cohort, scope)
                for k in (
                    "events",
                    "nopf_misses",
                    "direct_hits",
                    "late_merges",
                    "never_timely_group_instances",
                    "group_instances",
                    "events_in_never_timely_groups",
                ):
                    if a[k] != b[k]:
                        raise ValueError(
                            "Changing order changed cohort membership: " + k
                        )
    if (
        int(one("four_islands", "arrival", "all_admitted", "global")["events"])
        != sample
    ):
        raise ValueError("Sample membership disagrees with original CSV replay")
    original = sum(int(r["llc_tag_misses"]) for r in rows(baseline / "phases.csv"))
    matched = int(one("full", "arrival", "nopf_miss", "global")["events"])
    if matched > original:
        raise ValueError("Duplicate NoPF miss identity in ingress stream")
    write(
        dest / "validation.csv",
        [
            dict(
                case=case,
                retired=roi,
                total_stride_ingress=total,
                exact_sampled_ingress=sample,
                nopf_original_misses=original,
                matched_nopf_misses_at_stride_ingress=matched,
                original_misses_without_stride_ingress=original - matched,
                identical_fixed_cycles=True,
                identical_counters_and_per_operand_masks=True,
                identical_ordered_detail_records=True,
            )
        ],
    )


def render():
    tables = {}
    for key in ("counts", "gaps", "predecessors", "validation"):
        data = [
            r for case in CASES for r in rows(RAW / "by_view" / case / (key + ".csv"))
        ]
        write(HERE / (key + ".csv"), data)
        tables[key] = data

    def get(case, pop, order, cohort, scope):
        return next(
            r
            for r in tables["counts"]
            if (r["case"], r["window"], r["order"], r["cohort"], r["scope"])
            == (case, pop, order, cohort, scope)
        )

    scopes = {
        "global": "core",
        "pc": "PC",
        "page": "4 KiB physical page",
        "region256k": "256 KiB physical region",
    }
    labels = {"bfs_discovery": "GAP BFS-3", "soplex06_heldout": "SPEC06 Soplex-92B"}
    lines = [
        "# Whole LLC arrival order versus instruction order",
        "",
        "On each complete 200M-instruction ROI, record the actual sequence presented to Stride. Compare the same dynamic operands in arrival order and source instruction/operand order. A timely-to-miss adjacency is an observation to explain. A transition can change when outstanding instructions arrive in a different order.",
        "",
        "| Recording | Stride demand admissions | Admissions in four 250K-instruction islands | Original NoPF misses present / total | Original misses absent at Stride ingress |",
        "|---|---:|---:|---:|---:|",
    ]
    for v in tables["validation"]:
        lines.append(
            f"| {labels[v['case']]} | {int(v['total_stride_ingress']):,} | {int(v['exact_sampled_ingress']):,} | {int(v['matched_nopf_misses_at_stride_ingress']):,} / {int(v['nopf_original_misses']):,} | {int(v['original_misses_without_stride_ingress']):,} |"
        )
    lines += [
        "",
        "## What followed an attributed timely hit",
        "",
        "Each row conditions on adjacent targets that both entered Stride and were LLC misses in NoPF. The denominator counts eligible previous direct hits with an observed next target in that stream. The failure numerator combines the next plain tag miss and next late first merge. An ordinary tag hit has its own outcome. The island rate pools four independent slices.",
        "",
        "| Recording and scope | Arrival late / plain after direct | Program order late / plain after direct | Whole arrival failure rate | Island arrival failure rate | Arrival denominator, whole / islands |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in CASES:
        for scope in ("global", "pc", "region256k"):
            a = get(case, "full", "arrival", "nopf_miss", scope)
            b = get(case, "full", "program", "nopf_miss", scope)
            small = get(case, "four_islands", "arrival", "nopf_miss", scope)

            def rate(r):
                num = int(r["next_late"]) + int(r["next_plain_miss"])
                den = int(r["adjacent_after_direct"])
                return f"{num / den:.2%}" if den else "—"

            lines.append(
                f"| {labels[case]}, {scopes[scope]} | {int(a['next_late']):,} / {int(a['next_plain_miss']):,} | {int(b['next_late']):,} / {int(b['next_plain_miss']):,} | {rate(a)} | {rate(small)} | {int(a['adjacent_after_direct']):,} / {int(small['adjacent_after_direct']):,} |"
            )
    lines += [
        "",
        "## Streams with no direct timely conversion",
        "",
        "| Recording and scope | Whole groups with 32+ original misses | Never timely groups | Original misses in those groups | Island instances with 32+ misses | Never timely island instances |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in CASES:
        for scope in ("pc", "page", "region256k"):
            a = get(case, "full", "arrival", "nopf_miss", scope)
            b = get(case, "four_islands", "arrival", "nopf_miss", scope)
            lines.append(
                f"| {labels[case]}, {scopes[scope]} | {int(a['groups_at_least_32']):,} | {int(a['never_timely_supported_groups']):,} | {int(a['events_in_never_timely_supported_groups']):,} | {int(b['groups_at_least_32']):,} | {int(b['never_timely_supported_groups']):,} |"
            )
    lines += [
        "",
        "Ordering preserves whether a fixed group ever had a direct timely hit. Short logging windows can omit its success or exclude the group through insufficient support. In the island columns one PC or physical region can contribute a separate instance to each quarter.",
        "",
        "Gap distributions retain the elapsed cycles and the distance in the full admission sequence from the last direct hit to the first late/plain miss, allowing ordinary tag hits in between. A no-tag event censors an open run. Signed negative gaps in program order expose reordered demand arrivals. An arrival rank difference of one means adjacent admissions. These outcome-to-outcome gaps describe the sequence and do not establish when a prefetch address was predictable.",
        "",
        "See [all and matched stream counts](counts.csv), [all predecessor outcomes](predecessors.csv), [signed cycle and admission-rank gaps](gaps.csv), and [native parity and denominators](validation.csv). The separate [retirement-order study](../streams/README.md) also retains original misses absent at Stride ingress.",
    ]
    (HERE / "FINDINGS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=CASES)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    compile_analysis()
    for case in args.cases:
        process(case)
    if args.render or tuple(args.cases) == CASES:
        render()
