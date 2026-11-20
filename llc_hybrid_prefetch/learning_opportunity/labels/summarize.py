"""Report address reach at fixed horizons and target-anchored lead budgets."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
HORIZONS = (1, 2, 4, 8, 16, 32, 64)
TRACE_NAMES = {
    "bfs_discovery": "BFS-3",
    "bfs_discovery_far": "BFS-3, later phase",
    "bfs_transfer": "BFS-14",
    "pr_heldout": "PageRank-3",
    "mcf17_discovery": "MCF17 472B",
    "mcf17_discovery_far": "MCF17 472B, later phase",
    "mcf17_transfer": "MCF17 1644B",
    "mcf06_discovery": "MCF06 22B",
    "soplex06_heldout": "SoPlex 92B",
    "gcc734": "GCC 734B",
    "cc5": "Connected components-5",
    "cactus2421": "CactuBSSN 2421B",
    "bc3": "Betweenness centrality-3",
    "omnet874": "OMNeT++17 874B",
    "xalan700": "Xalancbmk 700B",
    "pr14": "PageRank-14",
    "astar42": "Astar 42B",
    "omnet188": "OMNeT++06 188B",
    "soplex247": "SoPlex 247B",
    "mcf51": "MCF06 51B",
    "gcc734_slot1": "GCC 734B, placement 1",
    "cc5_slot1": "CC-5, placement 1",
    "snap_stanford": "BFS, Stanford web graph",
    "snap_google": "BFS, Google web graph",
    "domain_gg_charlie": "Google Charlie",
    "domain_ai_llama": "Llama",
}


def display_name(case):
    return TRACE_NAMES.get(case, case)


def records(suffix):
    result = []
    for file in sorted((HERE / "results").glob("*__stride_" + suffix + ".csv")):
        with file.open() as f:
            result.extend(csv.DictReader(f))
    return result


def percent(row, num, den):
    total = int(row[den])
    return 100 * int(row[num]) / total if total else np.nan


def plot(horizons):
    plt.rcParams.update(
        {"svg.fonttype": "none", "svg.hashsalt": "learnability-labels", "font.size": 9}
    )
    cases = sorted({r["case"] for r in horizons})
    lookup = {(r["case"], int(r["horizon"])): r for r in horizons}
    fig, axes = plt.subplots(
        1, 3, figsize=(12, max(4.7, 0.32 * len(cases) + 2.5)), sharey=True
    )
    fig.subplots_adjust(left=0.20, right=0.99, bottom=0.16, top=0.82, wspace=0.12)
    fig.suptitle(
        "Physical-page reach and lead time at seven future LLC arrivals",
        x=0.02,
        ha="left",
        fontweight="bold",
    )
    panels = [
        ("Reach from recent pages (%)", "page_in_last64", "observed", 0, 100, "Blues"),
        ("Median lead (cycles)", "lead_p50_cycles", None, 0, 4, "Purples"),
        (
            "Eligible missed/late targets (%)",
            "original_all_gap_legal_lead_ge130",
            "original_and_all_three_miss_or_late",
            0,
            100,
            "Greens",
        ),
    ]
    for ax, (title, field, den, vmin, vmax, color) in zip(axes, panels):
        values = np.array(
            [
                [
                    percent(lookup[c, k], field, den)
                    if den
                    else int(lookup[c, k][field])
                    for k in HORIZONS
                ]
                for c in cases
            ]
        )
        shades = values if den else np.log10(1 + values)
        ax.imshow(shades, aspect="auto", vmin=vmin, vmax=vmax, cmap=color)
        ax.set_title(title, fontsize=10, pad=8)
        ax.set_xticks(range(len(HORIZONS)), list(HORIZONS))
        ax.set_yticks(range(len(cases)), [display_name(c) for c in cases])
        ax.set_xlabel("Future arrival number")
        ax.tick_params(length=0, pad=4)
        for (i, j), v in np.ndenumerate(values):
            label = "—" if np.isnan(v) else (f"{v:.0f}%" if den else f"{int(v):,}")
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if shades[i, j] > 0.62 * vmax else "#171c20",
            )
    fig.text(
        0.02,
        0.055,
        "Eligible = a page in the last 64 Stride arrivals and at least 130 cycles of lead. "
        "Missed/late = original NoPF miss and no timely or ordinary hit in all three reference arms (each records miss/late).",
        fontsize=7.5,
    )
    fig.text(
        0.02,
        0.025,
        "Each row uses up to 32,768 uniform unfiltered anchors. Median lead uses observed event time. "
        "Source: labels/results/*__stride_horizons.csv",
        fontsize=7.5,
    )
    fig.savefig(HERE / "reach_and_lead.svg", metadata={"Date": None})
    plt.close(fig)


def plot_novelty(cases, novelty):
    records = [
        (case, novelty[(case, "new_line_known_page")], novelty[(case, "seen_line")])
        for case in cases
        if (case, "new_line_known_page") in novelty
    ]
    if not records:
        return
    field = "original_all_gap_retained_page_lead_ge130"
    cases = [r[0] for r in records]
    new = np.array([int(r[1][field]) for r in records])
    seen = np.array([int(r[2][field]) for r in records])
    choices = [int(r[1]["legal_lines_p50_when_reachable"] or 0) for r in records]
    rows = np.arange(len(records))
    fig, (left, right) = plt.subplots(
        1, 2, sharey=True, figsize=(11, 2.5 + 0.3 * len(cases))
    )
    fig.subplots_adjust(left=0.21, right=0.97, top=0.88, bottom=0.15, wspace=0.15)
    fig.suptitle(
        "How many reachable targets were new lines?",
        x=0.03,
        ha="left",
        fontweight="bold",
    )
    left.barh(rows, seen, color="#596778", label="Previously observed line")
    left.barh(rows, new, left=seen, color="#197368", label="New line on a known page")
    for y, (n, old) in enumerate(zip(new, seen)):
        left.text(n + old, y, f"  {n:,} / {n + old:,}", va="center", fontsize=7)
    left.set_xlim(0, max(1, (new + seen).max()) * 1.4)
    left.set_title("New / eligible eighth-arrival targets")
    left.set_xlabel("Sampled eligible target pairs")
    left.legend(
        loc="upper center",
        bbox_to_anchor=(0.52, -0.12),
        ncol=2,
        frameon=False,
        fontsize=7,
    )
    right.barh(rows, choices, color="#806398")
    for y, count in enumerate(choices):
        right.text(
            count, y, f"  {count:,}" if count else "  —", va="center", fontsize=7
        )
    right.set_xlim(0, max(1, max(choices)) * 1.15)
    right.set_title("Median address choices on reachable new lines")
    right.set_xlabel("Physical lines legal under retained pages")
    left.set_yticks(rows, [display_name(c) for c in cases])
    left.invert_yaxis()
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(HERE / "new_lines_and_choices.svg", metadata={"Date": None})
    plt.close(fig)


def visibility_table():
    rows = []
    for path in sorted((HERE / "results").glob("*_visibility.csv")):
        with path.open() as f:
            data = list(csv.DictReader(f))
        absent = {
            r["policy"]: int(r["matched_original_operands"])
            for r in data
            if r["original_operand_present_at_arm_ingress"] == "0"
        }
        if absent:
            rows.append(
                "| "
                + data[0]["case"]
                + " | "
                + f"{int(data[0]['original_llc_misses']):,}"
                + " | "
                + " | ".join(
                    f"{absent[arm]:,}" for arm in ("nopf", "stride", "multi", "spp")
                )
                + " |"
            )
    if not rows:
        return []
    return [
        "## What the LLC never observed",
        "",
        "The denominator covers every original NoPF demand operand with a measured LLC tag miss. The remaining columns count how many of those operands had no matching admission at the LLC in each execution. A different load can issue the package-level request when accesses merge. Per-policy retirement labels and the missing-ingress count can therefore differ. [Full source-operand joins](results/)",
        "",
        "| Capture | Original NoPF misses | NoPF absent | Stride absent | Multi-delta absent | SPP absent |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
    ]


def placement_table():
    file = HERE / "placements.csv"
    if not file.is_file():
        return []
    with file.open() as f:
        records = [r for r in csv.DictReader(f) if r["horizon"] in ("1", "8", "64")]
    result = [
        "## Changing physical-page placement",
        "",
        "The same source operand anchors each paired Stride query. Changing the frame allocation can also change admission order. The last two columns condition on the same target operand occurring at that horizon in both executions. [All paired horizons](placements.csv)",
        "",
        "| Capture | Horizon | Paired horizons | Same target operand | Same signed delta among matched targets | Matched targets crossing 130-cycle threshold |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        result.append(
            "| "
            + r["case"]
            + " | "
            + " | ".join(
                f"{int(r[k]):,}"
                for k in (
                    "horizon",
                    "jointly_observed_horizon",
                    "same_target_operand",
                    "matched_operand_same_relative_delta",
                    "matched_operand_crossed_130_cycle_threshold",
                )
            )
            + " |"
        )
    if all("distinct_physical_page_shifts" in r for r in records):
        result.extend(
            [
                "",
                "The complete source mappings show the frame displacement for each virtual page.",
                "",
            ]
        )
        for r in records:
            if r["horizon"] == "1":
                result.append(
                    f"- {display_name(r['case'])} used {int(r['distinct_physical_page_shifts']):,} distinct frame {'shift' if int(r['distinct_physical_page_shifts']) == 1 else 'shifts'}. {int(r['pages_with_modal_shift']):,} of {int(r['compared_virtual_pages']):,} mapped pages moved by {int(r['modal_physical_page_shift']):+,} physical frames."
                )
        result.append("")
        result.append(
            "A uniform frame shift leaves relative deltas intact by construction. Per-page permutations test a different form of generalization."
        )
    return result + [""]


def domain_observations(horizon, novelty):
    lines = ["## What changed between workloads", ""]

    def number(row, field):
        return f"{int(row[field]):,}"

    field = "original_all_gap_legal_lead_ge130"
    stricter = "eligible_distinct_line_first_future_demand"
    mcfs = ("mcf17_discovery", "mcf17_transfer")
    if all((c, 8) in horizon and (c, "new_line_known_page") in novelty for c in mcfs):
        values = []
        for c in mcfs:
            new = novelty[c, "new_line_known_page"]
            if stricter in new:
                values.append(
                    f"{number(new, stricter)} of {number(horizon[c, 8], field)}"
                )
        if len(values) == 2:
            lines.append(
                f"- **MCF17, independent recordings.** At horizon eight, {values[0]} eligible target pairs in 472B and {values[1]} in 1644B concern a new exact line on a recently observed physical page. In each count the target is the first subsequent demand for that line. This gives an address-generation probe a concrete test across recordings."
            )
    if ("pr_heldout", 8) in horizon:
        r = horizon["pr_heldout", 8]
        lines.append(
            f"- **PageRank-3.** Median lead at horizon eight was {number(r, 'lead_p50_cycles')} cycles. Of {number(r, 'original_and_all_three_miss_or_late')} pairs missed or late in all three controls, only {number(r, field)} met the recent-page and 130-cycle tests. A different construction vocabulary could change this ceiling."
        )
    domains = ("domain_gg_charlie", "domain_ai_llama")
    if all((c, 8) in horizon for c in domains):
        google, llama = [horizon[c, 8] for c in domains]
        lines.append(
            f"- **Google Charlie and Llama.** In equally sized horizon-eight samples, {number(google, 'different_pc')} Google targets and {number(llama, 'different_pc')} Llama targets had a different PC from their trigger. The programs had {number(google, field)} and {number(llama, field)} eligible pairs respectively. Cross-PC targets describe a subset of the available opportunity. Whether previous PC context helps choose the address is an independent prediction test."
        )
    if ("xalan700", 8) in horizon:
        r = horizon["xalan700", 8]
        lines.append(
            f"- **Xalancbmk 700B.** Of {number(r, 'observed')} unfiltered horizon-eight targets, {number(r, 'original_miss')} missed under NoPF and {number(r, field)} met the joint eligibility test. Large fractions of a small gap population can have little effect on total work."
        )
    return lines + [""] if len(lines) > 2 else []


def main():
    horizons = [r for r in records("horizons") if r["scope"] == "global"]
    deadlines = [
        r
        for r in records("deadlines")
        if r["cohort"] == "original_all_three_miss_or_late"
        and r["vocabulary"] == "page_in_last64"
    ]
    cases = sorted({r["case"] for r in horizons})
    horizon = {(r["case"], int(r["horizon"])): r for r in horizons}
    window = {
        r["case"]: r for r in records("windows") if int(r["window_cycles"]) == 128
    }
    novelty = {
        (r["case"], r["novelty"]): r for r in records("novelty") if r["horizon"] == "8"
    }
    summary = [
        "# Which future targets are physically reachable in time?",
        "",
        "The tables count observed targets on real simulations. A row uses up to 32,768 uniform anchors on the selected "
        "Stride execution. Each anchor is eligible for sampling regardless of its cache outcomes. "
        "The full indexed stream can be queried at any rank. [Queries and split boundaries](README.md) · [Per-scope counts](results/)",
        "",
        *domain_observations(horizon, novelty),
        "## Fixed horizons",
        "",
        "Each count requires the target to be an original NoPF miss and a plain miss or late first use in all three "
        "conventional executions. The target page must occur among the latest 64 physical-page observations at the anchor. "
        "The target's admission must follow the anchor by at least 130 cycles. Counts at different horizons may refer to the same target. "
        "The fixed-horizon count can include a line already requested between the trigger and the target. The novelty table below reports the stricter first-demand subset. The denominator gives every anchor with an observed target at horizon eight. The 128-cycle footprint counts distinct lines "
        "after the anchor, measured only where the complete future window is observable.",
        "",
        "| Capture | Complete arrivals | h8 eligible / observed | h1 eligible | h64 eligible | h8 lead p50 (cycles) | 128-cycle footprint p50 / p90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cases:
        one, eight, last = [horizon[c, k] for k in (1, 8, 64)]
        w = window[c]
        field = "original_all_gap_legal_lead_ge130"
        summary.append(
            f"| {c} | {int(one['total_arrivals']):,} | {int(eight[field]):,} / {int(eight['observed']):,} | "
            f"{int(one[field]):,} | {int(last[field]):,} | {int(eight['lead_p50_cycles']):,} | "
            f"{w['distinct_lines_p50']} / {w['distinct_lines_p90']} |"
        )
    summary += [
        "",
        "![Per-horizon reach, lead and eligible fraction](reach_and_lead.svg)",
        "",
        "## Earliest legal anchor",
        "",
        "For each sampled target that misses under NoPF and records miss/late in all three conventional arms, search the last "
        "64 observed triggers. At each trigger the page-and-offset vocabulary can address any offset on the previous 64 observed "
        "pages including the trigger's page. Counts are retrospective address-construction ceilings. Actual prediction must choose "
        "a target, reach the issuer, obtain resources and finish in time.",
        "",
        "| Capture | Sampled miss/late targets | At least one legal anchor | ≥130 cycles | ≥146 cycles | ≥194 cycles | Median lead when legal |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in deadlines:
        fields = [
            "sampled_targets",
            "has_legal_anchor",
            "lead_ge130",
            "lead_ge146",
            "lead_ge194",
            "lead_p50_cycles",
        ]
        summary.append(
            "| "
            + r["case"]
            + " | "
            + " | ".join(f"{int(r[k]):,}" if r[k] else "—" for k in fields)
            + " |"
        )
    summary += [
        "",
        "The fixed-horizon and target-anchored tables answer different questions. The first fixes a prediction time and asks "
        "which observed successors meet the requirements. The second fixes the target and allows hindsight to select its earliest "
        "expressible trigger. A longer observed gap increases the time budget but can put the eventual target outside a small "
        "observed-page vocabulary. Transfer of address information across programs is tested by the [held-out probes](../probes/).",
        "",
    ]
    summary += [
        "## Novel lines at horizon eight",
        "",
        "For each eligible original miss at the eighth future global arrival, split the targets by whether the exact physical "
        "line had already been observed at the anchor. All eligible targets have a retained page. The last column counts "
        "the address alternatives at the anchor, using 64 offsets for every distinct retained page, among all reachable new-line "
        "targets. The stricter subset requires a distinct target line with no earlier demand for it after the trigger. Lines might still be resident or outstanding from before the trigger.",
        "",
        "| Capture | Eligible seen line | Eligible new line on known page | Distinct line and first future demand (both groups) | Median legal alternatives on reachable new lines |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in cases:
        if (case, "new_line_known_page") not in novelty:
            continue
        seen = novelty[(case, "seen_line")]
        new = novelty[(case, "new_line_known_page")]
        field = "original_all_gap_retained_page_lead_ge130"
        assert int(seen[field]) + int(new[field]) == int(
            horizon[case, 8]["original_all_gap_legal_lead_ge130"]
        )
        options = new["legal_lines_p50_when_reachable"]
        distinct = "eligible_distinct_line_first_future_demand"
        first_demand = (
            f"{int(seen[distinct]) + int(new[distinct]):,}" if distinct in seen else "—"
        )
        choices = f"{int(options):,}" if options else "—"
        summary.append(
            f"| {case} | {int(seen[field]):,} | {int(new[field]):,} | {first_demand} | {choices} |"
        )
    summary.extend(
        [
            "",
            "![Novel lines and the number of legal alternatives](new_lines_and_choices.svg)",
            "",
        ]
    )
    summary.extend(visibility_table())
    summary.extend(placement_table())
    (HERE / "RESULTS.md").write_text("\n".join(summary))
    plot(horizons)
    plot_novelty(cases, novelty)


if __name__ == "__main__":
    main()
