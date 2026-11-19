"""Render measured discovery/replay evidence; leave absent configurations out of measured tables."""

from collections import defaultdict, Counter
from pattern import HERE, table


def read(name):
    p = HERE / (name + ".csv")
    return table(p) if p.is_file() else []


def n(r, k):
    return int(r.get(k) or 0)


def pct(x):
    return f"{x:+.2f}%"


def pretty(x):
    return f"{int(x):,}"


def markdown(header, data):
    return "\n".join(
        [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]
        + ["| " + " | ".join(map(str, r)) + " |" for r in data]
    )


def separator(out, title):
    out.extend(("", f"## {title}", ""))


def sets(records):
    d = defaultdict(dict)
    for r in records:
        d[r["case"]][r["arm"]] = r
    return d


def main():
    out = [
        "# What real history predicted, and what survived native replay",
        "",
        "A repeated relationship in GAP PageRank allowed an independently generated physical target to arrive before a matching conventional request. Its full execution value depends on the opportunities lost elsewhere. The experiments below test the relationship at a fixed decision budget and count both sides of every replacement.",
    ]
    primary = sets(read("full_performance"))
    compact = sets(read("diagnostic_performance"))
    cases = [d for c, d in primary.items() if c != "pr_heldout_entire"]
    if cases:
        faster = sum(n(d["shape"], "cycles") < n(d["stride"], "cycles") for d in cases)
        equal = sum(n(d["shape"], "cycles") == n(d["stride"], "cycles") for d in cases)
        best = sum(
            n(d["shape"], "cycles")
            < min(n(d[m], "cycles") for m in ("nopf", "stride", "multi", "spp"))
            for d in cases
        )
        out.extend(
            [
                "",
                f"Across {len(cases)} primary 200M-instruction replays, the frozen rule plus Stride beat Stride on {faster}, tied on {equal}, and took longer on {len(cases) - faster - equal}. It beat the fastest of NoPF, Stride, Multi-delta and SPP on {best}. Full PageRank overlaps its primary interval and is excluded from those counts. These results establish transferable address examples and measurable costs. They leave the need for learned prediction unproven.",
            ]
        )
    align = {
        (r["case"], r["reference"]): r for r in read("full_complete_load_alignment")
    }
    separator(out, "Fixed-work intervention")
    out.append(
        "The pooled donors were the earlier MCF17-665B and BFS-3 training prefixes. Every eighth measured LLC ingress offered one opportunity to use a signed displacement from the frozen producer-equality table. This key uses current line offset, the last two clipped same-PC deltas, and the equality pattern among four consecutive producer PCs. It has no absolute PC identity, and can name a physical line absent from the logged history. A legal independent candidate has slot priority. Native Stride proceeds on other slots and whenever the auxiliary table abstains. Full runs retain warmed processor/cache/Stride state. The independent feature history starts empty at ROI entry and persists across all quarters. Added decision latency is modeled as zero."
    )
    out.append("")
    row = []
    for c, data in primary.items():
        s = data["shape"]
        win = align[(c, "stride")]
        vs = align[(c, "spp")]
        row.append(
            [
                c,
                pretty(s["instructions"]),
                pretty(s["cycles"]),
                pretty(s["independent_different_proposals"])
                if "independent_different_proposals" in s
                else "pending source count",
                pct(float(s["savings_vs_stride_pct"])),
                pretty(win["original_timely_gained"]),
                pretty(win["original_timely_lost"]),
                pretty(win["original_late_to_timely"]),
                pretty(vs["original_timely_gained"]),
                pretty(vs["original_timely_lost"]),
            ]
        )
    out.append(
        markdown(
            [
                "Recording",
                "Retired",
                "Shape + Stride cycles",
                "Independent offers differing from Stride",
                "Time saved vs Stride",
                "Gained vs Stride",
                "Lost vs Stride",
                "Previously Stride-late among gains",
                "Gained vs SPP",
                "Lost vs SPP",
            ],
            row,
        )
    )
    out += [
        "",
        "Gains and losses are attributed timely first uses on aligned original NoPF misses, joined over every retired instruction and memory operand. Other residency hits receive no direct prefetch credit. A positive time saving means faster fixed-work completion. The 200M PageRank interval overlaps its entire-recording run. Interpret them separately.",
    ]
    joint = read("full_joint_coverage")
    if joint:
        out.extend(
            [
                "",
                "Across all retired operands, the following comparison counts any timely first use observed under each of four separately executed conventional mechanisms. The union cannot be implemented by selecting the best outcome with hindsight. Complete masks capture the hybrid execution's first-use status; attribution to an independent producer requires the detailed samples.",
                "",
                markdown(
                    [
                        "Recording",
                        "Hybrid timely, none of four timely",
                        "Hybrid timely, all four explicitly miss/late",
                        "At least one conventional timely, hybrid non-timely",
                    ],
                    [
                        [
                            r["case"],
                            pretty(r["hybrid_timely_none_reference_timely"]),
                            pretty(r["hybrid_timely_all_reference_persistent_or_late"]),
                            pretty(r["any_reference_timely_hybrid_became_non_timely"]),
                        ]
                        for r in joint
                    ],
                ),
            ]
        )
    separator(out, "Separated matched diagnostics")
    out.append(
        "Each row uses original-source instructions 3,000,001–3,500,000 following a separately reset 1M warmup. The complete-source registry establishes provenance. Policy effects apply to this diagnostic window. A shadow run checks Stride equivalence; each frozen, local, ordered and +1 decision is also replayed through an independent Python prefix reader."
    )
    diag = {
        (r["case"], r["actual_arm"], r["reference"]): r
        for r in read("diagnostic_first_uses")
    }
    row = []
    for c, data in compact.items():
        s = data["shape"]
        best = min(
            (
                m
                for m in ("stride", "multi", "spp", "look2", "look4", "temporal")
                if m in data
            ),
            key=lambda m: n(data[m], "cycles"),
        )
        r = diag[(c, "shape", "spp")]
        row.append(
            [
                c,
                pct(float(s["savings_vs_stride_pct"])),
                best,
                pct(100 * (1 - n(s, "cycles") / n(data[best], "cycles"))),
                n(s, "native_was_silent"),
                n(s, "different_native_proposal_displaced"),
                n(s, "displaced_then_rejected"),
                n(r, "gained"),
                n(r, "lost"),
                n(r, "gained_direct_offered_earlier_than_ref"),
            ]
        )
    out += [
        "",
        markdown(
            [
                "Case",
                "Shape time saved vs Stride",
                "Fastest tested conventional",
                "Shape time saved vs fastest",
                "Silent native slots filled",
                "Different native offers displaced",
                "Redirect then reject",
                "Original timely gained vs SPP",
                "Lost vs SPP",
                "Direct frozen gains earlier than SPP-late",
            ],
            row,
        ),
    ]
    example = next(
        (
            r
            for r in read("diagnostic_native_examples")
            if r["case"] == "pr_heldout"
            and r["actual_arm"] == "shape"
            and r["prior_arm"] == "spp"
            and r["prior_status"] == "late"
        ),
        None,
    )
    if example:
        r = example
        out += [
            "",
            f"At GAP PageRank instruction {pretty(r['cue_instr'])}, operand {r['cue_op']}, the frozen rule observed physical line {pretty(r['cue_line'])}. It generated physical line {pretty(r['target_line'])} at cycle {pretty(r['generation'])}; the request dispatched at {pretty(r['dispatch'])} and filled at {pretty(r['fill'])}. Dynamic instruction {pretty(r['target_instr'])}, operand {r['target_op']}, reached the LLC at {pretty(r['demand_ingress'])}. Lead was {r['lead']} cycles. SPP's matched producer had {r['prior_lead']} cycles. NoPF head exposure was {r['nopf_head']} cycles. The producer PCs in observed order were {r.get('observed_pcs', '')}. Raw PCs differ from the two fitted donors. GAP PageRank was held out as an executable/recording; the project previously examined a different Ligra PageRank implementation.",
        ]
    controls = read("diagnostic_joint_coverage")
    out.extend(
        [
            "",
            "The same PageRank target was already timely under fixed two-stride lookahead, from the same source instruction as the frozen rule. The next table asks how many direct independent timely uses still missed or merged late under every cheap comparison. Each row first checks Stride, Multi-delta, SPP, temporal, two-stride, and sparse +1. The eight-control column also checks four-stride and sparse +4, both added after +4 appeared in MCF17-1644B. It is a retrospective union of achievable outcomes from separate runs.",
            "",
            markdown(
                [
                    "Case",
                    "Direct independently generated original timely uses",
                    "All six missed/late",
                    "All eight missed/late",
                    "Of those with positive NoPF head wait",
                    "Of those fresh at cue",
                ],
                [
                    [
                        r["case"],
                        n(r, "direct_independent_original_timely"),
                        n(r, "all_six_persistent_or_late"),
                        n(r, "all_eight_persistent_or_late"),
                        n(r, "all_eight_and_positive_nopf_head"),
                        n(r, "all_eight_and_fresh_line"),
                    ]
                    for r in controls
                ],
            ),
        ]
    )
    fixed3 = read("bfs_fixed3_witness")
    if fixed3:
        byarm = {r["arm"]: r for r in fixed3}
        r = byarm["sparse3"]
        out.extend(
            [
                "",
                f"The BFS-3 load at source instruction 100,101,506 first used physical line 3,500,606. Shape + Stride issued +3 from line 3,500,603; Stride, Multi-delta and SPP issued +1 and merged late. NoPF recorded 150 exposed head cycles for this operand. Stride, Multi-delta, SPP, shape and fixed +3 each recorded zero, including the three late conventional first uses. Fixed +3 on the identical every-eighth-ingress schedule produced status **{r['outcome']}** and finished the complete 200M interval in {pretty(r['fixed_work_cycles'])} cycles, compared with {pretty(byarm['shape']['fixed_work_cycles'])} for the structural table. The trigger history and exact stages are in bfs_fixed3_witness.csv.",
            ]
        )
    separator(out, "Transfer across phases and recordings")
    out.append(
        "At fixed lags of 1, 4 and 16 ingress events, each table was fit once on the appropriate donor's first quarter sample. The same current-PC/offset/delta context is compared to an ordered three-PC suffix, a PC bag, and absolute-PC-free equality patterns. A post-hoc ablation drops the older identity pattern while preserving offset and two same-PC deltas. All keep at most 256 surviving contexts under the same support and vote cutoffs. Q2 of each donor is development; Q3–Q4 are chronological tests. Regions, different binaries and held-out GAP PageRank use an unmodified donor. Counts aggregate opportunities by horizon; an operand can appear at more than one horizon. The cutoff uses completed NoPF demand service in donor Q1; it screens feasibility and does not model queueing."
    )
    phase = read("phase_transfer")
    reach = read("phase_reach")
    by = defaultdict(Counter)
    for r in phase:
        c = r["case"]
        g = by[c]
        if c == r["donor"] and n(r, "quarter") == 2:
            continue
        if r["mode"] == "local":
            g["gap"] += n(r, "gap_with_lead")
        for k in ("gap_matches", "gained_gap_vs_local", "lost_gap_vs_local"):
            g[r["mode"] + "_" + k] += n(r, k)
    for r in reach:
        if r["case"] == r["donor"] and n(r, "quarter") == 2:
            continue
        g = by[r["case"]]
        for k in (
            "original_miss_targets",
            "enough_lead",
            "cross_page_with_lead",
            "conditional_context_horizons",
        ):
            g[k] += n(r, k)
    out += [
        "",
        markdown(
            [
                "Test recording",
                "Original miss targets",
                "Reachable with service lead",
                "Of those also reachable across page",
                "Gap opportunities at fixed lags",
                "Local matches",
                "Ordered gained / lost vs local",
                "Structural gained / lost vs local",
                "PC-free local gained / lost",
                "Distinct alternate-future contexts per sample summed",
            ],
            [
                [
                    c,
                    pretty(g["original_miss_targets"]),
                    pretty(g["enough_lead"]),
                    pretty(g["cross_page_with_lead"]),
                    pretty(g["gap"]),
                    pretty(g["local_gap_matches"]),
                    f"{g['ordered_gained_gap_vs_local']} / {g['ordered_lost_gap_vs_local']}",
                    f"{g['shape_gained_gap_vs_local']} / {g['shape_lost_gap_vs_local']}",
                    f"{g['anonymous_local_gained_gap_vs_local']} / {g['anonymous_local_lost_gap_vs_local']}",
                    g["conditional_context_horizons"],
                ]
                for c, g in by.items()
            ],
        ),
    ]
    out += [
        "",
        "Each sampled original miss receives one reach count if at least one of the preceding 16 ingress events could legally form its address at sufficient lead before its first use. A legal offset can cross physical pages only if the target page was observed within the last 64 ingresses. The 512-line signed reach is a vocabulary choice. An unseen exact target means unseen since the start of the logging island. Gaps at a fixed lag require a NoPF miss and a measured miss or late first use under all three conventional arms. A load missing from any arm's ingress stream is outside that conservative subset.",
    ]
    info = defaultdict(Counter)
    for r in read("phase_address_information"):
        if n(r, "horizon") != 1 or (r["case"] == r["donor"] and n(r, "quarter") == 2):
            continue
        g = info[r["case"]]
        mode = r["mode"]
        g[mode + "_bits"] += n(r, "anchors") * float(r["mean_bits"])
        if mode == "local":
            for k in (
                "anchors",
                "ambiguous_local_contexts",
                "anchors_in_ambiguous_local_contexts",
            ):
                g[k] += n(r, k)
    out.extend(
        [
            "",
            "A separate test scores every next address with a held-out probability distribution, including an inexpressible-target class. Locally ambiguous means at least two representable displacements each observed at least twice for the identical current PC, offset and two local deltas. Lower bits per anchor means better held-out probability prediction. Quarter-specific counts may include the same context in two quarters. These fits predict addresses without first selecting baseline misses or exposed stalls.",
            "",
            markdown(
                [
                    "Recording",
                    "Anchors",
                    "Ambiguous context-quarter groups",
                    "Anchors in those contexts",
                    "Local bits/anchor",
                    "Ordered",
                    "Structural",
                    "PC-free local",
                ],
                [
                    [
                        c,
                        pretty(g["anchors"]),
                        pretty(g["ambiguous_local_contexts"]),
                        pretty(g["anchors_in_ambiguous_local_contexts"]),
                        *[
                            f"{g[m + '_bits'] / g['anchors']:.3f}"
                            for m in ("local", "ordered", "shape", "anonymous_local")
                        ],
                    ]
                    for c, g in info.items()
                ],
            ),
        ]
    )
    separator(out, "Transition evidence")
    cause = read("full_transition_causes")
    pop = read("full_populations")
    summary = defaultdict(Counter)
    for r in cause:
        if r["view"] == "pc" and r["cohort"] == "immediately_after_timely":
            summary[r["case"]][r["outcome"]] += n(r, "occurrences")
    for r in pop:
        summary[r["case"]]["interleaved"] += n(r, "interleaved_local_repeat")
        summary[r["case"]]["silent_misses"] += n(r, "pc_misses_in_silent_streams")
    out.append(
        "Counts distinguish runs of directly attributed timely uses from ordinary hits, late consumptions, and original misses on streams without a timely hit during that island. Each post-hit row considers the next original baseline miss within the same PC's filtered ingress sequence. A post-hit miss alone identifies no loop edge. The source span for unresolved misses is the preceding 16 visible ingresses or last use of that line, whichever is later."
    )
    out += [
        "",
        markdown(
            [
                "Recording",
                "Next also timely",
                "Next late",
                "Next ordinary hit",
                "Next miss with no observed prior offer in span",
                "Next miss after observed rejection",
                "Original misses on never-timely PC streams",
                "Nonadjacent repeated local stride, global step differs",
            ],
            [
                [
                    c,
                    pretty(g["attributed_timely"]),
                    pretty(g["attributed_late"]),
                    pretty(g["ordinary_hit"]),
                    pretty(g["miss_no_prior_offer_in_span"]),
                    pretty(
                        sum(
                            v
                            for k, v in g.items()
                            if k.startswith("miss_prior_offer_rejected_")
                        )
                    ),
                    pretty(g["silent_misses"]),
                    pretty(g["interleaved"]),
                ]
                for c, g in summary.items()
            ],
        ),
    ]
    separator(out, "What follows from the measurements")
    out.extend(
        [
            "1. Separate recognition of conditional structure from its execution value. The PageRank example establishes an earlier independently issuable address and a timely fill. The native turnover tables directly test the opportunity cost elsewhere; the zero-head-wait example alone establishes no speedup.",
            "2. Require accurate selection among alternate futures on held-out data. The phase experiment records instances where the *same* PC, current physical offset, and two local deltas support distinct targets at the *same* event distance and an ordered training branch identifies both. Empty files mean zero examples passed the stated support/lead filters. Local, ordered, bag and structural rows quantify failed transfer and the cases already captured by a bounded table.",
            "3. Compare a future positive result to a direct-address generator with higher measured benefit at the same resource budget. Performance improvements require completion-time gains with plausible delay and state costs. A new network or representation remains a hypothesis.",
            "",
            "A compact implementation of this structural control would need tags and displacement values for 169 rules (capacity 256), a 64-slot tagged last-line/delta store, the prior three PC identities and a 64-entry physical-page-reference ring per active core. An explicit implementation might compare associative keys or use indexed SRAM with tags and collision handling. The simulator's exact software map and its zero delay are optimistic; their host memory is not an area estimate. For example, 256 entries storing a 64-bit key, 11-bit signed displacement and valid bit require 2,432 bytes before index circuitry, and a 64-slot history with 48-bit PC tag, 48-bit line and 8-bit delta requires 832 bytes before valid bits. Structural keys could be narrowed to their finite fields, which requires a different costed implementation.",
            "",
            "The full and diagnostic streams retain source instruction IDs and operand numbers. Matched integers are checked for identical PC, virtual line and physical line wherever both LLC streams contain the operand. Candidate dispatch and fills that traverse logging islands can be unavailable; affected examples explicitly omit their timing. Head waits are diagnostic per-load values, never added as independent speedups.",
        ]
    )
    (HERE / "FINDINGS.md").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
