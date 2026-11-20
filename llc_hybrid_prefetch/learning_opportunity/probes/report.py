"""Render the scored probe experiments to a short report and annotated figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extract import HERE

MAIN = [
    ("bfs_discovery", "bfs_discovery", "BFS-3 → later BFS-3"),
    ("soplex06_heldout", "soplex06_heldout", "Soplex → later Soplex"),
    ("soplex06_heldout", "bfs_discovery", "Soplex → BFS-3"),
    ("bfs_discovery", "soplex06_heldout", "BFS-3 → Soplex"),
    ("snap_stanford", "snap_google", "Stanford BFS → Google BFS"),
    ("mcf17_layout_original", "mcf17_layout_reversed", "MCF → permuted pages"),
]
REFS = ("same_pc", "hide_other_pc", "global32")


def render(scores):
    plt.rcParams.update(
        {"svg.fonttype": "none", "svg.hashsalt": "llc-learning-probes", "font.size": 9}
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.35))
    fig.subplots_adjust(left=0.21, right=0.985, bottom=0.2, top=0.73, wspace=0.19)
    fig.suptitle(
        "Earlier context changes address probability and legal target recovery",
        x=0.015,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    colors = ["#456685", "#db974c", "#15806c"]
    for j, (scope, title) in enumerate(
        zip(REFS, ("Same PC", "Hide other-PC values", "Ordered global 32"))
    ):
        for i, (source, dest, _) in enumerate(MAIN):
            row = scores.loc[(source, dest, scope)]
            y = i + (j - 1) * 0.24
            for ax, value, label in (
                (axes[0], row.mean_bits, f"{row.mean_bits:.2f}"),
                (
                    axes[1],
                    100 * row.forced_deadline / row.anchors,
                    f"{100 * row.forced_deadline / row.anchors:.1f}%",
                ),
            ):
                ax.scatter(
                    value,
                    y,
                    marker=("o", "s", "D")[j],
                    s=23,
                    color=colors[j],
                    label=title if i == 0 else None,
                )
                ax.annotate(
                    label,
                    (value, y),
                    xytext=(5, 0),
                    textcoords="offset points",
                    fontsize=7.8,
                    va="center",
                    color=colors[j],
                )
    for j, ax in enumerate(axes):
        ax.set_ylim(len(MAIN) - 0.5, -0.5)
        ax.set_yticks(range(len(MAIN)), [x[2] for x in MAIN] if j == 0 else [])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", color="#dedede")
        ax.set_axisbelow(True)
        ax.set_xlim(-0.008 * ax.get_xlim()[1], ax.get_xlim()[1] * 1.13 + 0.1)
    axes[0].set_title("Address-action log loss ↓", loc="left")
    axes[0].set_xlabel("Bits per cue, including the other-action class")
    axes[1].set_title("One proposal per cue ↑", loc="left")
    axes[1].set_xlabel("Exact legal target with ≥130 cycles of observed lead")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.58, 0.91),
        frameon=False,
        ncol=3,
    )
    fig.text(
        0.015,
        0.025,
        "Horizon 8, signed displacement. Test uses the last fifth of each destination stream. Source: scores.csv",
        fontsize=8,
        color="#4c5a68",
    )
    fig.savefig(HERE / "heldout_history.svg", metadata={"Date": None})
    plt.close(fig)


def render_history_width(rows):
    widths = [
        "local_region",
        "global4",
        "global8",
        "global16",
        "global32",
        "global64",
        "shuffled",
    ]
    names = [
        "Local + region",
        "Global 4",
        "Global 8",
        "Global 16",
        "Global 32",
        "Global 64",
        "Shuffled 32",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.65), sharey=True)
    fig.subplots_adjust(left=0.19, right=0.97, top=0.78, bottom=0.16, wspace=0.12)
    for ax, case, title in zip(
        axes, ("bfs_discovery", "soplex06_heldout"), ("BFS-3", "Soplex-92B")
    ):
        selected = rows[
            (rows.source == case)
            & (rows.destination == case)
            & (rows.split == "test")
            & (rows.encoding == "delta")
        ]
        data = (
            selected.pivot(index="scope", columns="horizon", values="forced_deadline")
            .loc[widths, [1, 8, 32]]
            .to_numpy()
        )
        n = int(selected.anchors.iloc[0])
        values = 100 * data / n
        ax.imshow(values, cmap="Blues", aspect="auto", vmin=0, vmax=30)
        for j in range(3):
            for i in range(len(widths)):
                v = values[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if v >= 17 else "#203746",
                )
        ax.set_xticks(range(3), ["1", "8", "32"])
        ax.set_xlabel("Future LLC admission")
        ax.set_yticks(range(len(widths)), names)
        ax.set_title(f"{title} · {n:,} held-out cues", loc="left", fontsize=10)
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
    fig.suptitle(
        "How much history identifies an exact address with time to issue?",
        x=0.02,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.845,
        "Cells show the share of cues with an exact legal address and ≥130 cycles of lead at one proposal per cue.",
        fontsize=9,
    )
    fig.text(
        0.02,
        0.025,
        "Page-relative legality is checked at the cue. Models share capacity and training targets within each horizon. Source: scores.csv",
        fontsize=8,
    )
    fig.savefig(HERE / "history_widths.svg", metadata={"Date": None})
    plt.close(fig)


def transfer_summary(scores):
    phase = {
        "bfs_transfer",
        "mcf17_transfer",
        "soplex247",
        "mcf51",
        "pr14",
    }
    env = {"gcc734_slot1", "cc5_slot1", "mcf17_layout_reversed"}
    cross = []
    for (s, d), part in scores.groupby(level=[0, 1]):
        same = part.loc[(s, d, "same_pc")]
        glob = part.loc[(s, d, "global32")]
        if s == d:
            kind = "Chronological phase"
        elif d in env:
            kind = "Physical placement"
        elif d == "snap_google":
            kind = "Named graph input"
        elif d.endswith("_far"):
            kind = "Separated window, same recording"
        elif (
            d in phase
            and s != "bfs_discovery"
            or (s == "bfs_discovery" and d in ("bfs_discovery_far", "bfs_transfer"))
        ):
            kind = "Same program, independent recording"
        else:
            kind = "Cross-program"
        cross.append(
            dict(
                family=kind,
                source=s,
                destination=d,
                anchors=int(glob.anchors),
                local_bits=same.mean_bits,
                global_bits=glob.mean_bits,
                global_vocabulary_coverage=glob.expressible / glob.anchors,
                conditional_bits_local=same.conditional_mean_bits,
                conditional_bits_global=glob.conditional_mean_bits,
                legal_lead_local=int(same.deadline),
                legal_lead_global=int(glob.deadline),
                forced_legal_lead_local=int(same.forced_deadline),
                forced_legal_lead_global=int(glob.forced_deadline),
            )
        )
    table = pd.DataFrame(cross)
    table.to_csv(HERE / "transfer_summary.csv", index=False, float_format="%.7g")
    return table


def additional(rows, transfer, selected):
    rows.extend(
        [
            "",
            "## Local ambiguity and different target definitions",
            "",
            "In BFS-3, 53 training-supported PC/offset/two-distance contexts had multiple recurring displacement actions. They accounted for 469 held-out cues. Ordered global history raised loss from 1.287 to 1.303 bits and increased forced, qualified matches from 65 to 67. The stronger within-trace effect occurred among 3,493 cues with low or unseen local support, where forced matches rose from 319 to 385. The experiment finds limited disambiguation on repeating exact contexts. [Training-defined ambiguity cohorts](ambiguous_contexts.csv)",
            "",
            "Changing which future event is predicted changes the result. For the eighth subsequent demand from a different PC in BFS-3, adding global context to same-PC history reduced forced, qualified matches from 192 to 152 among 6,002 cues. For the eighth demand outside the current 256 KiB region, only 30 targets were expressible and neither probe recovered a qualified target. Soplex yielded 76 versus 68 for the other-PC target across 6,022 eligible cues. [Future relation comparisons](related_targets.csv)",
            "",
            "## Transfer, target novelty and recording coverage",
            "",
        ]
    )
    strata = pd.read_csv(HERE / "unseen_contexts.csv")
    novel = strata.query(
        "source==destination and scope=='global32' and encoding=='delta' and stratum=='unseen_train_anchor_region'"
    )
    if len(novel):
        best = novel.loc[novel.anchors.idxmax()]
        rows.append(
            f"In {best.destination}, {int(best.anchors):,} test cues fell in trigger regions absent from the sampled training anchors. Global-32 recovered {int(best.deadline):,} qualified targets with abstention and {int(best.forced_deadline):,} with one proposal per cue. This measures transfer to trigger regions; the output line may itself have previously appeared in the measured history. In Soplex → BFS-3, all 6,002 trigger PCs were unseen in donor training, 2,515 targets were in the H8 vocabulary, and 1,033 were recovered legally with lead. [Novelty denominators and outcomes](unseen_contexts.csv)"
        )
        rows.append("")
    if ("bfs_discovery", "bfs_discovery", "with_stride_feedback") in selected.index:
        tag = selected.loc[("bfs_discovery", "bfs_discovery", "with_feedback")]
        stride = selected.loc[
            ("bfs_discovery", "bfs_discovery", "with_stride_feedback")
        ]
        rows.extend(
            [
                f"The timed BFS probe recovered 1,594 qualified targets at one proposal per cue. Giving it earlier LLC tag outcomes and their ages yielded {int(tag.forced_deadline):,} and {tag.mean_bits:.3f} bits per cue. Adding earlier Stride stage/reason/age feedback yielded {int(stride.forced_deadline):,} and {stride.mean_bits:.3f} bits. All callback serials used as features strictly precede the anchor. [Feedback joins](feedback.py) · [All scopes](scores.csv)",
                "",
            ]
        )
    if ("bfs_discovery", "bfs_transfer", "global32") in selected.index:
        rows.extend(
            [
                "On the independent BFS-14 recording, adding global history to local and region context reduced loss from 2.223 to 1.747 bits. One concrete proposal per cue gained 245 and lost 38 qualified lines, spanning eight additional displacements and 51 target regions. Of the net 207, 196 were original misses that SPP already covered on time. The net change was +3 among 334 SPP-late original misses and −1 among 865 persistent ones. This recording supports transfer of predictive context, with limited complementary coverage. [Matched individual gains and losses](paired_targets.csv)",
                "",
            ]
        )
    rows.extend(
        [
            "In the 2M-instruction MCF placement replay, 4,418 of 4,426 alternate-layout cues matched an original cue operand. The same H8 target operand followed in 4,388 pairs, and its physical line changed in 3,859 pairs. In these relocated pairs, global32 recovered 101 qualified fixed-budget targets in both layouts, 41 only in the original and 32 only in the alternate. The same fit and source instructions can yield different physical displacement and timing outcomes. [Paired placements](matched_placements.csv)",
            "",
        ]
    )
    registered = pd.read_csv(HERE.parent / "expansion/sources.csv")
    ext = registered.merge(
        transfer.loc[transfer.source == "bfs_discovery"],
        left_on="case",
        right_on="destination",
    )
    if len(ext):
        ext["bits_saved"] = ext.local_bits - ext.global_bits
        ext["forced_change"] = (
            ext.forced_legal_lead_global - ext.forced_legal_lead_local
        )
        rows.extend(
            [
                f"The fixed BFS-3 training fit has been applied to {len(ext)} of {len(registered)} preregistered held-out expansion recordings. Each row compares local same-PC history with local, region and global-32 history. The alphabet, sampled cues, trees per fit and issue budget stay fixed.",
                "",
                "![BFS model on new workloads](expansion_transfer.svg)",
                "",
                "| Independent target recording | Cues | Vocabulary coverage | Bits saved | Change in qualified matches at one proposal per cue |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for r in ext.itertuples():
            rows.append(
                f"| {r.case} | {int(r.anchors):,} | {100 * r.global_vocabulary_coverage:.1f}% | {r.bits_saved:+.3f} | {int(r.forced_change):+,} |"
            )
        draw_expansion(ext)
    native = pd.read_csv(HERE / "conventional_outcomes.csv")
    cohort = native[(native.source == "pooled_three") & (native.arm == "spp")]
    if not cohort.empty:
        counts = cohort.pivot(
            index=["destination", "original_miss_outcome"],
            columns="scope",
            values="forced_deadline",
        )
        support = cohort.pivot(
            index=["destination", "original_miss_outcome"],
            columns="scope",
            values="anchors",
        )
        states = ("persistent", "late", "timely")
        cases = [
            c for c in registered.case if all((c, k) in counts.index for k in states)
        ]
        values = np.array(
            [
                [
                    counts.loc[(c, k), "global32"] - counts.loc[(c, k), "local_region"]
                    for k in states
                ]
                for c in cases
            ],
            dtype=int,
        )
        total = np.array(
            [[support.loc[(c, k), "global32"] for k in states] for c in cases],
            dtype=int,
        )
        if cases:
            rows.extend(
                [
                    "",
                    "### Does a better address add to SPP coverage?",
                    "",
                    "The pooled global-32 probe and its local-plus-region control each receive one proposal per cue. The figure gives their difference in lead-qualified exact matches, stratified by the SPP outcome for the same dynamic target. Each denominator counts only original NoPF misses. A gain on an already timely SPP target cannot become an additional timely first use for that target. [Native outcome decomposition](conventional_outcomes.csv)",
                    "",
                    "![Marginal probe target matches by SPP outcome](spp_complement.svg)",
                ]
            )
            size = max(2.65, 0.34 * len(cases) + 1.8)
            fig, ax = plt.subplots(figsize=(8, size))
            fig.subplots_adjust(
                left=0.17, right=0.96, top=1 - 0.83 / size, bottom=0.49 / size
            )
            bound = max(1, np.abs(values).max())
            ax.imshow(values, aspect="auto", cmap="RdBu", vmin=-bound, vmax=bound)
            ax.set_xticks(range(3), ["Still a miss", "Late merge", "Timely first use"])
            ax.xaxis.tick_top()
            ax.set_yticks(range(len(cases)), cases)
            ax.tick_params(length=0)
            ax.spines[:].set_visible(False)
            for a in range(len(cases)):
                for b in range(3):
                    v = values[a, b]
                    ax.text(
                        b,
                        a,
                        f"{v:+,} / {total[a, b]:,}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if abs(v) > 0.55 * bound else "#102638",
                    )
            fig.suptitle(
                "What remains after accounting for SPP?",
                x=0.025,
                ha="left",
                fontweight="bold",
                fontsize=13,
            )
            fig.text(
                0.025,
                0.04,
                "Each cell shows net matches / original misses in that SPP class. Horizon 8. Source: conventional_outcomes.csv",
                fontsize=8,
            )
            fig.savefig(HERE / "spp_complement.svg", metadata={"Date": None})
            plt.close(fig)
    pairs_path = HERE / "paired_targets.csv"
    if pairs_path.exists():
        paired = pd.read_csv(pairs_path)
        gcc = paired.query(
            "source=='pooled_three' and destination=='gcc734' and comparator=='local_region' and cohort=='all' and decision=='one_proposal'"
        )
        if len(gcc):
            r = gcc.iloc[0]
            rows.extend(
                [
                    "",
                    f"GCC-734B shows how the donor can change the apparent need for history. With the pooled training set, global32 gained {int(r.rich_only):,} qualified lines and lost {int(r.comparator_only):,} against local+region at a fixed budget. Displacement {int(r.most_frequent_gain_delta):+} accounted for {int(r.most_frequent_gain_count):,} newly qualified lines, across {int(r.distinct_gain_cue_pcs)} cue PCs and {int(r.distinct_gain_target_regions)} target regions overall. With the BFS-only donor, the same scope change reduced qualified hits from 3,761 to 3,677. [Paired address concentration](paired_targets.csv) · [Unpooled controls](scores.csv)",
                    "",
                    "One pooled H8 prediction on GCC originated at instruction 215,006,233, physical line 164,231, PC 4,257,721. The target was line 164,239 at PC 4,257,546, 2,268 observed cycles later. This target had appeared in the prior measured stream and was an original miss that SPP predicted late. The example transfers a displacement between producer PCs. It has no isolated performance replay. [Source operands](paired_examples.csv)",
                ]
            )
    feedback_path = HERE / "external_feedback.csv"
    if feedback_path.exists():
        feedback = pd.read_csv(feedback_path).set_index(["destination", "scope"])
        rows.extend(
            [
                "",
                "### Resolved cache feedback on external recordings",
                "",
                "The fixed BFS-3 fit uses identical ordered history and timing features in all three columns. Each cell gives action log loss in bits and the count of exact, legal, lead-qualified addresses at one proposal per cue. Prior tag feedback supplies resolved tag hit/miss and age. The last column adds prior Stride decisions and outcomes. [Callback counts and raw scores](external_feedback.csv)",
                "",
                "| Destination | Timing only | + Prior tag feedback | + Prior tag and Stride feedback |",
                "|---|---:|---:|---:|",
            ]
        )
        for case in registered.case:
            if (case, "with_time") not in feedback.index:
                continue
            cells = []
            for scope in ("with_time", "with_feedback", "with_stride_feedback"):
                m = feedback.loc[(case, scope)]
                cells.append(f"{m.mean_bits:.3f} / {int(m.forced_deadline):,}")
            rows.append("| " + case + " | " + " | ".join(cells) + " |")
    pooled_file = HERE / "pooled_transfer.csv"
    if pooled_file.exists():
        f = pd.read_csv(pooled_file)
        all_rows = f[f.cohort == "all"].set_index(["destination", "scope"])
        rows.extend(
            [
                "",
                "### Balanced multi-program training",
                "",
                "The pooled fit uses 12,000 training cues each from BFS-3, MCF17-472B and Soplex-92B. It learns its own alphabet from the 36,000 donor targets. The following counts hold that new alphabet fixed across feature scopes. Absolute log loss across the pooled and single-donor alphabets uses different target classes. [All pooled cohorts](pooled_transfer.csv)",
                "",
                "| Independent target | Cues | Cues with expressible H8 targets | Conditional bits, local+region → global | Fixed-budget qualifying lines, local+region → global |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for case in registered.case:
            if (case, "global32") not in all_rows.index:
                continue
            base = all_rows.loc[(case, "local_region")]
            rich = all_rows.loc[(case, "global32")]
            rows.append(
                f"| {case} | {int(rich.anchors):,} | {int(rich.expressible):,} | {base.conditional_mean_bits:.2f} → {rich.conditional_mean_bits:.2f} | {int(base.forced_deadline):,} → {int(rich.forced_deadline):,} |"
            )
    rows.extend(
        [
            "",
            "The [full transfer matrix](transfer_summary.csv) separates a later phase from a different recording of the same program, a named graph input, a different executable and a changed simulated physical placement. A model is frozen at transfer. Same-executable recording IDs alone do not identify independent graph inputs.",
        ]
    )


def draw_expansion(ext):
    height = max(2.8, 1.65 + 0.34 * len(ext))
    fig, axes = plt.subplots(1, 2, figsize=(10, height))
    fig.subplots_adjust(
        left=0.15, right=0.94, top=1 - 0.73 / height, bottom=0.55 / height, wspace=0.21
    )
    for ax, field, title in zip(
        axes,
        ("bits_saved", "forced_change"),
        (
            "Log-loss reduction (bits per cue)",
            "Change in exact legal targets with lead",
        ),
    ):
        y = np.arange(len(ext))
        v = ext[field].to_numpy()
        ax.barh(y, v, color=np.where(v >= 0, "#147567", "#a46a40"), height=0.55)
        span = max(np.max(np.abs(v)), 4 if field == "forced_change" else 0.001)
        if field == "forced_change":
            from matplotlib.ticker import MaxNLocator

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(min(v.min(), 0) - 0.23 * span, max(v.max(), 0) + 0.28 * span)
        for row, value in enumerate(v):
            label = f"{value:+.3f}" if field == "bits_saved" else f"{int(value):+,}"
            ax.annotate(
                label,
                (value, row),
                xytext=(3 if value >= 0 else -3, 0),
                textcoords="offset points",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=8,
            )
        ax.set_yticks(y, list(ext.case) if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.axvline(0, color="#566772", lw=0.6)
        ax.set_title(title, fontsize=10, loc="left")
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle(
        "Can BFS-trained history improve predictions on held-out recordings?",
        x=0.015,
        ha="left",
        fontweight="bold",
        fontsize=13,
    )
    fig.text(
        0.015,
        0.02,
        "Global 32 plus local and region histories versus same-PC history. Horizon 8, fixed one-proposal budget. Source: transfer_summary.csv",
        fontsize=8,
    )
    fig.savefig(HERE / "expansion_transfer.svg", metadata={"Date": None})
    plt.close(fig)


def main():
    scores = pd.read_csv(HERE / "scores.csv")
    selected = scores.query(
        "split=='test' and horizon==8 and encoding=='delta'"
    ).set_index(["source", "destination", "scope"])
    render(selected)
    render_history_width(scores)
    transfer = transfer_summary(selected)
    rows = [
        "# What transfers in an address-prediction probe",
        "",
        "An offline tree ensemble trained on Soplex learned when to propose a relative address in BFS-3, where every test cue PC was absent from the source training-anchor dictionary. With 32 ordered events, it recovered 1,033 exact legal targets at least 130 cycles before demand among 6,002 BFS cues. The same-PC probe recovered zero qualified targets. Each of the 1,033 additional targets was eight lines ahead of its cue. The outcome shows transfer of a simple displacement across producer PCs and 75 target regions. Evidence for more complex complementary behavior varies by workload. [Paired target-level counts](paired_targets.csv)",
        "",
        "![Address probability and legal targets](heldout_history.svg)",
        "",
        "## Equal work and equal address budgets",
        "",
        "The table fixes horizon eight, the source-trained displacement vocabulary, and the chronological test partition. Bits per cue includes the aggregate out-of-vocabulary class. Qualified counts require the exact line on a recently observed page and at least 130 cycles of lead. The last two columns force one concrete proposal per cue even when the model favors abstention. Both counts measure an address opportunity before queueing.",
        "",
        "| Train → evaluated trace | Cues | Same-PC bits | Global-32 bits | Qualified, same PC | Qualified, global | One proposal, same PC | One proposal, global |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source, dest, title in MAIN:
        local = selected.loc[(source, dest, "same_pc")]
        global_ = selected.loc[(source, dest, "global32")]
        rows.append(
            f"| {title} | {int(local.anchors):,} | {local.mean_bits:.3f} | {global_.mean_bits:.3f} | "
            f"{int(local.deadline):,} | {int(global_.deadline):,} | {int(local.forced_deadline):,} | {int(global_.forced_deadline):,} |"
        )
    rows += [
        "",
        f"The [complete transfer table](transfer_summary.csv) contains {len(transfer)} source/destination comparisons. The [per-horizon results](scores.csv) include development scores, other feature windows and a second encoding that predicts an offset on a known page.",
        "",
        "## Which part of history helps?",
        "",
        "![Horizon and history width across workloads](history_widths.svg)",
        "",
        "- On BFS-3, a same-PC H8 probe used 1.421 bits per cue. Adding region history and 32 ordered events reduced this to 1.346, with lower loss in five of eight contiguous test blocks. With other-PC global token values hidden, it used 1.414 bits. The global probe made 1,608 legal, lead-qualified matches under one proposal per cue, versus 1,565 with masking. Using both same-PC and region histories without the global window yields 1.360 bits and 1,533 forced matches. Adding the ordered global window to that control saves 0.014 bits and adds 75 matches. The other-PC mask covers global tokens only. Region histories can still contain other producers’ values. [History contrasts](history_value.csv)",
        "- On Soplex, the same comparison raised log loss from 1.446 to 1.487 bits. For the Soplex-trained transfer to BFS, hiding other-PC values yields 1,182 qualified addresses under the forced budget, versus 1,091 when retaining them. The direction depends on the workload, score and issue budget. [Scores](scores.csv)",
        "- The named Stanford-to-Google BFS test exposed a vocabulary limit. Only 145 of 7,079 Google targets could be named by the Stanford-trained H8 displacement alphabet. Mean loss fell from 0.339 to 0.291 bits, while both decision rules with abstention recovered zero lead-qualified lines. The global probe recovered seven under the forced budget. A low loss can mostly measure correct rejection of an unrepresentable target. [Trace inputs](../graph_inputs/README.md) · [Vocabulary and exact outcomes](scores.csv)",
        "- In BFS-3, the 32-event timed probe used 1.355 bits. Adding contemporaneous occupancy from six queues reduced loss to 1.344. Adding trace-derived core metadata in a separate comparison gave 1.338. Each would require extra inputs and implementation costs. [Core and pressure controls](scores.csv)",
        "",
        "## Relation to conventional prefetching",
        "",
        "In Soplex-to-BFS transfer, the global probe added a qualifying target for 123 of 241 sampled original misses that merged into a late SPP request. It added none among 1,034 plain persistent SPP misses and none among 84 cues for a newly observed line on a known page. With one concrete proposal per cue, global gained 116 and lost two against same-PC in the late group. Those outcomes tie opportunity to specific missing lead time. They do not measure cache hits or execution benefit from issuing the model's proposals. [Paired gains and losses](paired_targets.csv) · [Additional trace witnesses](paired_examples.csv)",
        "",
        "For one such case, source instruction 228,391,751 on the held-out BFS run reached physical line 36,114. The Soplex-trained global probe proposed line 36,122. Its target reached the LLC 607 cycles later. Both caller and target had PC 93,831,678,904,576; the physical line had previously appeared in the measured LLC stream. [Dynamic key and event values](paired_examples.csv)",
        "",
        "The first actual demand to a predicted line can occur at a different position from the training label. Across 6,002 Soplex-to-BFS cues, the global probe offered 1,081 legal addresses. A matching line first appeared within 8,192 cycles for 1,050 offers. Eleven arrived within 130 cycles and 1,039 had at least 130 cycles of lead. Fifteen qualifying first uses occurred at a position other than horizon eight. Of the 1,039, 782 were original NoPF misses and 123 were SPP-late. This follow-up does not require hitting the originally supervised position. [First-demand windows](first_demand_soplex06_heldout.csv) · [MCF](first_demand_mcf17_discovery.csv)",
        "",
        "## What remains open",
        "",
        "The current positive transfers concentrate on a few displacements or on targets that SPP already covers. A useful next test targets independent recordings with adequate legal reach and asks whether the richer probe consistently predicts original SPP-late or persistent misses. Only prospective replay with equal issue opportunity can establish marginal completion-time gains. The measured one-off replay sets a counterexample to inferring performance from an exact address alone. [Novelty strata](unseen_contexts.csv) · [Protocol and limits](PROTOCOL.md)",
    ]
    open_questions = rows[-4:]
    rows = rows[:-4]
    replay = HERE.parent / "one_offer/effects.csv"
    if replay.is_file():
        measured = pd.read_csv(replay).iloc[0]
        rows.extend(
            [
                "",
                "## One prospective action in the real issuer",
                "",
                f"The predeclared cue-only selector supplied one BFS-3 offer. It predicted line {int(measured.predicted_line):,} at a lead of {int(measured.actual_eighth_lead_cycles):,} cycles before the eighth demand. That operand already hit under NoPF and was already a timely first use under Stride. After replacing the ordinary Stride proposal at that one decision, the real 200M-instruction replay gained {int(measured.timely_gained)} timely attributions and displaced {int(measured.timely_displaced)}, with {int(-measured.cycles_saved):,} extra fixed-work cycles. One exact forecast had no marginal value for its intended target. [Selection protocol](PROTOCOL.md) · [Native replay and outcome definitions](../one_offer/README.md)",
            ]
        )
    additional(rows, transfer, selected)
    selection = pd.read_csv(HERE.parent / "expansion/sources.csv").merge(
        transfer[transfer.source == "bfs_discovery"],
        left_on="case",
        right_on="destination",
    )
    if len(selection):
        more_lines = (
            selection.forced_legal_lead_global > selection.forced_legal_lead_local
        )
        fewer_bits = selection.global_bits < selection.local_bits
        more_and_lower = more_lines & fewer_bits
        rows[2:2] = [
            f"The frozen BFS-3 model has now been scored on {len(selection)} independent expansion recordings. Adding region and ordered global history to same-PC history reduced aggregate action log loss in {int(fewer_bits.sum())}; it also recovered more legal, exact H8 targets with ≥130 cycles of lead at the same one-proposal budget in {int(more_and_lower.sum())}. These are per-recording directions, with denominators and sizes below. The same-workload BFS-14 recording shows the clearest separation between learning an address relationship and complementing SPP: net +207 fixed-budget qualified addresses against a local-plus-region probe, of which +196 were already timely under SPP. [Transfer results](transfer_summary.csv) · [Matched conventional outcomes](paired_targets.csv)",
            "",
        ]
    domain_path = HERE / "domain_transfer.csv"
    if domain_path.is_file():
        dom = (
            pd.read_csv(domain_path)
            .query("cohort=='all'")
            .set_index(["destination", "scope"])
        )
        rows.extend(
            [
                "",
                "## Google service and AI inference transfer",
                "",
                "The BFS-3 H8 donor, alphabet and feature scopes stay frozen. These recordings previously had a short screening interval. Their later evaluation provides a cross-domain check. The separate reserved files remain untouched. [Trace selection and dependency limits](../domains/README.md) · [Domain scores](domain_transfer.csv) · [Matched native outcomes](domain_outcomes.csv)",
                "",
                "| Recording | Cues | Expressible H8 targets | Page-bank reach | Conditional bits, local+region → global | Qualified fixed-budget lines, local+region → global |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for case in dom.index.get_level_values(0).unique():
            a = dom.loc[(case, "local_region")]
            b = dom.loc[(case, "global32")]
            rows.append(
                f"| {case} | {int(b.anchors):,} | {int(b.expressible):,} | {int(b.reachable):,} | {a.conditional_mean_bits:.2f} → {b.conditional_mean_bits:.2f} | {int(a.forced_deadline):,} → {int(b.forced_deadline):,} |"
            )
    rows.extend(open_questions)
    (HERE / "FINDINGS.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
