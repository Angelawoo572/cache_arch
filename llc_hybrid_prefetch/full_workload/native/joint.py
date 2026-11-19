"""Compute how often degree-matched conventional arms cover the same original miss under separate replay."""

import csv
import io
import subprocess

from run import RAW


def summarize_joint(case, policies, roi, original_totals):
    dirs = [RAW / "runs" / (case + "__" + policy) for policy in policies]
    args = [
        str(RAW / "join"),
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
        counts = dict(
            any_timely=0,
            none_timely=0,
            none_timely_any_late=0,
            none_timely_none_late=0,
            persistent_tag_miss_all_four=0,
            timely_all_four=0,
        )
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
                    counts["timely_all_four"] += n
            else:
                counts["none_timely"] += n
                counts[
                    "none_timely_any_late"
                    if 6 in states.values()
                    else "none_timely_none_late"
                ] += n
                if all(v == 2 for v in states.values()):
                    counts["persistent_tag_miss_all_four"] += n
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
