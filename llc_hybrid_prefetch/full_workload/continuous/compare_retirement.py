"""Put the complete original-miss retirement census alongside the common-arrival-population audit."""

import importlib.util
import subprocess
from report import rows, write
from run_arrivals import CASES, FULL, HERE, RAW

STREAMS = FULL / "streams"


def original_census(case):
    events = [
        r
        for r in (
            rows(STREAMS / "event_counts.csv")
            if (STREAMS / "event_counts.csv").exists()
            else []
        )
        if r["case"] == case and r["policy"] == "stride"
    ]
    groups = [
        r
        for r in (
            rows(STREAMS / "group_counts.csv")
            if (STREAMS / "group_counts.csv").exists()
            else []
        )
        if r["case"] == case and r["policy"] == "stride"
    ]
    if len(events) == 45 and len(groups) == 6:
        return events, groups
    events, groups = [], []
    # Use the original complete-retirement analysis directly when its combined table has not yet been exported.
    spec = importlib.util.spec_from_file_location(
        "retirement_census", STREAMS / "measure.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    exe = RAW / "episodes"
    subprocess.run(
        [
            "c++",
            "-O3",
            "-std=c++14",
            str(STREAMS / "episodes.cc"),
            "-lz",
            "-o",
            str(exe),
        ],
        check=True,
    )
    for kind, *fields in module.run(case, exe):
        r = dict(zip(module.FIELDS[kind].split(), fields))
        if r["policy"] == "stride":
            (events if kind == "event" else groups).append(r)
    return events, groups


def main():
    common = rows(HERE / "counts.csv")
    output = []
    for case in CASES:
        original, groups = original_census(case)
        for view, scope in (
            ("all", "global"),
            ("pc", "pc"),
            ("physical_256KiB_region", "region256k"),
        ):
            seq = [
                r
                for r in original
                if r["view"] == view and r["preceding"] == "after_timely"
            ]
            if len(seq) != 5:
                raise ValueError("Incomplete original retirement census")
            full = next(
                r
                for r in groups
                if r["view"] == view and int(r["minimum_original_misses"]) == 1
            )
            states = {r["current"]: int(r["events"]) for r in seq}
            output.append(
                dict(
                    case=case,
                    scope=scope,
                    order="program",
                    population="all_original_misses",
                    events=full["original_misses"],
                    groups=full["streams"],
                    never_timely_groups=full["never_direct_streams"],
                    events_in_never_timely_groups=full[
                        "misses_in_never_direct_streams"
                    ],
                    observed_next_after_direct=sum(states.values()),
                    next_late=states["late_first_use"],
                    next_plain_miss=states["other_tag_miss"],
                    next_no_tag=states["no_llc_tag"],
                )
            )
            for order in ("program", "arrival"):
                r = next(
                    r
                    for r in common
                    if r["case"] == case
                    and r["window"] == "full"
                    and r["order"] == order
                    and r["scope"] == scope
                    and r["cohort"] == "nopf_miss"
                )
                output.append(
                    dict(
                        case=case,
                        scope=scope,
                        order=order,
                        population="original_misses_admitted_by_stride",
                        events=r["events"],
                        groups=r["group_instances"],
                        never_timely_groups=r["never_timely_group_instances"],
                        events_in_never_timely_groups=r[
                            "events_in_never_timely_groups"
                        ],
                        observed_next_after_direct=r["adjacent_after_direct"],
                        next_late=r["next_late"],
                        next_plain_miss=r["next_plain_miss"],
                        next_no_tag=r["next_no_tag"],
                    )
                )
    write(HERE / "retirement_comparison.csv", output)


if __name__ == "__main__":
    main()
