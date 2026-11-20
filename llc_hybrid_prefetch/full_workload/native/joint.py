"""Compute how often degree-matched conventional arms cover the same original miss under separate replay."""

import csv
import io
import subprocess

from run import RAW


def summarize_joint(case, policies, roi, original_totals, raw=RAW):
    if tuple(policies) not in (
        ("nopf", "stride", "multi", "spp"),
        ("nopf", "stride", "multi", "spp", "temporal"),
    ):
        raise ValueError(
            "Joint masks require NoPF, Stride, Multi-delta, SPP, and optional Temporal in that order"
        )
    dirs = [raw / "runs" / (case + "__" + policy) for policy in policies]
    args = [
        str(raw / "join"),
        *(str(d / "outcomes_core0.u16le.gz") for d in dirs),
        str(roi),
    ]
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    cells = list(csv.DictReader(io.StringIO(result.stdout)))
    summaries = []
    for phase in (*range(4), "all"):
        selected = [r for r in cells if phase == "all" or int(r["quarter"]) == phase]
        original = sum(int(r["operands"]) for r in selected)
        expected = sum(original_totals) if phase == "all" else original_totals[phase]
        if original != expected:
            raise ValueError(
                "Joint dynamic target count differs from original NoPF misses"
            )
        suffix = {3: "three", 4: "four"}[len(policies) - 1]
        persistent = f"persistent_tag_miss_all_{suffix}"
        unanimous = f"timely_all_{suffix}"
        counts = dict(
            any_timely=0,
            none_timely=0,
            none_timely_any_late=0,
            none_timely_none_late=0,
        )
        counts[persistent] = counts[unanimous] = 0
        unique = {p: 0 for p in policies[1:]}
        for cell in selected:
            states = {p: int(cell[p]) for p in policies[1:]}
            timely = [p for p, status in states.items() if status == 9]
            n = int(cell["operands"])
            if timely:
                counts["any_timely"] += n
                if len(timely) == 1:
                    unique[timely[0]] += n
                if len(timely) == len(states):
                    counts[unanimous] += n
            else:
                counts["none_timely"] += n
                counts[
                    "none_timely_any_late"
                    if 6 in states.values()
                    else "none_timely_none_late"
                ] += n
                if all(v == 2 for v in states.values()):
                    counts[persistent] += n
        summaries.append(
            dict(
                case=case,
                quarter=phase,
                original_nopf_misses=original,
                **counts,
                **{"timely_only_" + p: v for p, v in unique.items()},
            )
        )
    return summaries
