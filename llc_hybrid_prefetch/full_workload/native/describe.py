"""Render a compact, per-recording reading of the validated full-workload baseline tables."""

import argparse
from pathlib import Path
from run import HERE, rows
from validate_results import validate

ARM_NAMES = {
    "stride": "Stride",
    "multi": "Multi-delta",
    "spp": "SPP",
    "temporal": "Temporal",
}
LABELS = {
    "bfs_discovery": "GAP BFS-3",
    "bfs_transfer": "GAP BFS-14",
    "bfs_discovery_far": "GAP BFS-3 far",
    "mcf17_discovery": "SPEC17 MCF-472B",
    "mcf17_transfer": "SPEC17 MCF-1644B",
    "mcf17_discovery_far": "SPEC17 MCF-472B far",
    "mcf06_discovery": "SPEC06 MCF-22B",
    "soplex06_heldout": "SPEC06 Soplex-92B",
    "pr_heldout": "GAP PageRank-3",
    "pr_heldout_entire": "GAP PageRank-3 through EOF",
}


def pct(x):
    return f"{float(x) * 100:.2f}%" if x != "" else "—"


def traffic(r):
    return sum(
        int(r["dram_" + label + "_bytes"]) for label in ("demand", "prefetch", "write")
    )


def describe(directory):
    validate(directory)
    groups = {}
    for r in rows(directory / "performance.csv"):
        groups.setdefault(r["case"], {})[r["policy"]] = r
    joint = {
        r["case"]: r
        for r in rows(directory / "joint_coverage.csv")
        if r["quarter"] == "all"
    }
    quarter_cycles = {
        (r["case"], r["policy"], int(r["quarter"])): int(r["cycles"])
        for r in rows(directory / "quarters.csv")
    }
    fastest = {}
    lines = [
        "# Conventional performance on the registered recordings",
        "",
        "Each row reports one uninterrupted execution per policy after 50M instructions of demand-only warmup. All cache and predictor state carries across the four measured quarters. The attribution table selects the lowest-cycle arm among policies with at least one accepted request on that recording. Independently choosing a policy on future workloads would require additional evidence.",
        "",
        "## Fixed-work time",
        "",
        "| Recording | Measured instructions | NoPF cycles | Stride time saved | Multi-delta time saved | SPP time saved | Temporal time saved |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c, arms in groups.items():
        base = arms["nopf"]
        savings = [pct(arms[p]["time_reduction_vs_nopf"]) for p in ARM_NAMES]
        lines.append(
            f"| {LABELS[c]} | {int(base['measured_instructions']):,} | {int(base['cycles']):,} | "
            + " | ".join(savings)
            + " |"
        )
    lines += [
        "",
        "A positive percentage means that the identical measured instruction stream retired sooner than with NoPF. The CSV retains exact cycles, IPC, L2 and LLC MPKI, tag hit and miss rates, first-use counts, pressure and DRAM bytes for every arm. [Full performance](performance.csv) · [Quarter boundaries and cycles](quarters.csv)",
        "",
        "## First-use attribution and traffic of the fastest issuing arm",
        "",
        "| Recording | Fastest | NoPF LLC misses | Direct timely coverage | Matched late merges | Useful accuracy | ROI cycles waiting at ROB head | Total DRAM bytes / NoPF |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c, arms in groups.items():
        issuing = [p for p in ARM_NAMES if int(arms[p]["accepted"]) > 0]
        b = arms["nopf"]
        if not issuing:
            lines.append(
                f"| {LABELS[c]} | No accepted request | {int(b['nopf_original_misses']):,} | — | — | — | — | — |"
            )
            continue
        best = min(issuing, key=lambda p: int(arms[p]["cycles"]))
        a = arms[best]
        fastest[c] = best
        winners = [ARM_NAMES[p] for p in issuing if arms[p]["cycles"] == a["cycles"]]
        bw = traffic(a) / traffic(b) if traffic(b) else None
        traffic_string = f"{bw:.3f}×" if bw is not None else "—"
        lines.append(
            f"| {LABELS[c]} | {' / '.join(winners)} | {int(b['nopf_original_misses']):,} | {pct(a['timely_coverage'])} | {int(a['orig_to_late']):,} | {pct(a['useful_accuracy'])} | {pct(a['head_union_fraction'])} | {traffic_string} |"
        )
    lines += [
        "",
        "Direct timely coverage uses a matched NoPF miss and the first demand use of a completed prefetch. A late merge still waits for data. Head-wait fraction is the fraction of ROI cycles with a qualifying in-flight demand at the ROB head. It is a stall proxy. A cycle reduction from one arm includes its effects on every request and cache occupant. [Matched lookup and first-use outcomes](transitions.csv) · [Sampled lead and service](sampled_timeliness.csv) · [Head-wait distributions](sampled_exposure.csv)",
        "",
        "## Execution-time changes across consecutive quarters",
        "",
        "Each row keeps the same issuing arm selected on the whole interval above. Each cell compares its retirement time to NoPF for the same consecutive dynamic instructions. All four quarters run in sequence with live processor and cache state.",
        "",
        "| Recording | Arm | Quarter 1 time saved | Quarter 2 | Quarter 3 | Quarter 4 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for c, arm in fastest.items():
        quarter_savings = [
            pct(1 - quarter_cycles[c, arm, q] / quarter_cycles[c, "nopf", q])
            for q in range(4)
        ]
        lines.append(
            f"| {LABELS[c]} | {ARM_NAMES[arm]} | " + " | ".join(quarter_savings) + " |"
        )
    lines += [
        "",
        "[All quarter counts and program ordinals](quarters.csv)",
        "",
        "## What the four predictors cover together in hindsight",
        "",
        "| Recording | Timely under any arm | Timely under none | Late under at least one among those never timely | Plain tag miss under all four |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in groups:
        j = joint[c]
        lines.append(
            f"| {LABELS[c]} | {pct(int(j['any_timely']) / int(j['original_nopf_misses'])) if int(j['original_nopf_misses']) else '—'} | {int(j['none_timely']):,} | {int(j['none_timely_any_late']):,} | {int(j['persistent_tag_miss_all_four']):,} |"
        )
    lines += [
        "",
        "A late first-use cohort is a candidate for earlier legally issuable address generation. Persistent misses require testing whether an informative cue arrives before the deadline. The four arms produce different memory-system states, so their retrospective union supplies a target census only. The table also distinguishes targets unique to each conventional arm. [Joint target coverage](joint_coverage.csv)",
        "",
    ]
    temporal = [arms["temporal"] for arms in groups.values()]
    if not any(int(r["proposed"]) for r in temporal):
        lines += [
            "## Scope of the exact-line temporal control",
            "",
            "The 256-entry temporal control proposed zero targets throughout the measured intervals in this table. Its policy requires a physical predecessor-line tag to survive and repeat the same successor before it can propose. On the separate recurring nineteen-line conflict fixture, the same native binary issued 592 cross-page requests and provided 591 timely first uses. These results apply to this finite table and trigger interface. Larger temporal metadata stores, PC-conditioned associations and other candidate generators require their own comparison.",
            "",
        ]
        support = next(
            (
                r
                for r in rows(directory / "temporal_support.csv")
                if r["case"] in ("pr_heldout", "pr_heldout_entire")
                and r["quarter"] == "all"
            ),
            None,
        )
        if support is not None:
            lines += [
                f"On PageRank's four sampled NoPF ingress sequences, {int(support['seen_exact_line']):,} of {int(support['llc_ingresses']):,} observations repeated an earlier physical line in the same sample. An unbounded exact-line association table reached confidence for {int(support['unbounded_evaluable']):,} evaluable next observations and matched {int(support['unbounded_immediate_successor_hit']):,}. The 256-entry shadow reached confidence {int(support['bounded_confident_candidate']):,} times. These per-sample shadows reset at the start of each island. [Exact-line retention and immediate-successor matches](temporal_support.csv)",
                "",
            ]
    if "mcf17_discovery" in groups:
        a = groups["mcf17_discovery"]
        lines += [
            "## Why timing and MPKI can move in different directions",
            "",
            f"On MCF-472B, LLC load MPKI was {float(a['nopf']['llc_mpki']):.3f} with NoPF and {float(a['spp']['llc_mpki']):.3f} with SPP. SPP saved {pct(a['spp']['time_reduction_vs_nopf'])} of fixed-work cycles. It directly converted {int(a['spp']['orig_to_timely']):,} original misses to timely first uses, and {int(a['spp']['orig_to_late']):,} original misses merged with an outstanding prefetch. These counts motivate measuring earlier legal cues and the cost of displaced requests. They do not assign the cycle savings to individual late loads. [Complete MCF and other controls](performance.csv)",
            "",
        ]
    (directory / "BASELINE_FINDINGS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=HERE)
    describe(parser.parse_args().dir)
