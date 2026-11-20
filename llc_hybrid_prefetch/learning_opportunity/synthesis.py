"""Relate held-out address information to actual conventional outcomes for all external recordings."""

from pathlib import Path
import csv
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.titlelocation": "left",
        "font.size": 9,
    }
)


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def summarize():
    register = read(HERE / "expansion/sources.csv")
    perf = read(HERE / "expansion/results/performance.csv")
    outcomes = read(HERE / "probes/conventional_outcomes.csv")
    scores = read(HERE / "probes/scores.csv")
    pooled = read(HERE / "probes/pooled_transfer.csv")
    rows = []
    for entry in register:
        case = entry["case"]
        baseline = {r["policy"]: r for r in perf if r["case"] == case}
        winner = min(baseline, key=lambda k: int(baseline[k]["cycles"]))
        sr = {
            r["scope"]: r
            for r in scores
            if r["source"] == "bfs_discovery"
            and r["destination"] == case
            and r["split"] == "test"
            and r["encoding"] == "delta"
            and int(r["horizon"]) == 8
        }
        po = {
            r["scope"]: r
            for r in pooled
            if r["destination"] == case and r["cohort"] == "all"
        }
        r0, r1 = sr["local_region"], sr["global32"]
        assert int(r0["anchors"]) == int(r1["anchors"])
        cohorts = {
            (r["source"], r["scope"], r["original_miss_outcome"]): r
            for r in outcomes
            if r["source"] in ("bfs_discovery", "pooled_three")
            and r["destination"] == case
            and r["arm"] == "spp"
            and r["split"] == "test"
            and r["encoding"] == "delta"
            and int(r["horizon"]) == 8
        }
        result = dict(
            workload=case,
            fastest_baseline=winner,
            baseline_time_saved_pct=round(
                100 * float(baseline[winner]["time_reduction_vs_nopf"]), 4
            ),
            test_cues=int(r0["anchors"]),
            expressible=int(r1["expressible"]),
            legal_page_reach=int(r1["reachable"]),
            local_bits=float(r0["mean_bits"]),
            global_bits=float(r1["mean_bits"]),
            local_qualified=int(r0["forced_deadline"]),
            global_qualified=int(r1["forced_deadline"]),
            pool_local_qualified=int(po["local_region"]["forced_deadline"]),
            pool_global_qualified=int(po["global32"]["forced_deadline"]),
        )
        for label in ("timely", "late", "persistent"):
            left, right = (
                cohorts[("bfs_discovery", scope, label)]
                for scope in ("local_region", "global32")
            )
            assert left["anchors"] == right["anchors"]
            result[label + "_targets"] = int(left["anchors"])
            result[label + "_local_qualified"] = int(left["forced_deadline"])
            result[label + "_global_qualified"] = int(right["forced_deadline"])
            pool_left, pool_right = (
                cohorts[("pooled_three", scope, label)]
                for scope in ("local_region", "global32")
            )
            assert pool_left["anchors"] == pool_right["anchors"] == left["anchors"]
            result["pool_" + label + "_local_qualified"] = int(
                pool_left["forced_deadline"]
            )
            result["pool_" + label + "_global_qualified"] = int(
                pool_right["forced_deadline"]
            )

        def net_count(row, field):
            return row[field + "_global_qualified"] - row[field + "_local_qualified"]

        result["net_all"] = result["global_qualified"] - result["local_qualified"]
        result["net_spp_late_or_persistent"] = net_count(result, "late") + net_count(
            result, "persistent"
        )
        result["net_spp_timely"] = net_count(result, "timely")
        result["pool_net_all"] = (
            result["pool_global_qualified"] - result["pool_local_qualified"]
        )
        result["pool_net_spp_timely"] = net_count(result, "pool_timely")
        result["pool_net_spp_late_or_persistent"] = net_count(
            result, "pool_late"
        ) + net_count(result, "pool_persistent")
        blocks = read(
            HERE / "raw/probes/fits" / f"matrix_bfs_discovery_to_{case}" / "blocks.csv"
        )
        block_counts = {
            (r["scope"], int(r["block"])): int(r["forced_deadline"])
            for r in blocks
            if r["split"] == "test"
            and r["encoding"] == "delta"
            and int(r["horizon"]) == 8
        }
        differences = [
            block_counts["global32", i] - block_counts["local_region", i]
            for i in range(8)
        ]
        if sum(differences) != result["net_all"]:
            raise ValueError(
                f"Chronological blocks do not reproduce the aggregate for {case}"
            )
        result.update(
            positive_blocks=sum(delta > 0 for delta in differences),
            tied_blocks=sum(delta == 0 for delta in differences),
            min_block_net=min(differences),
            max_block_net=max(differences),
        )
        rows.append(result)
    return rows


def figure(rows, donor="BFS-3", prefix="", file="complementarity.svg"):
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.5), layout="constrained")
    fig.suptitle(
        f"Where richer history helps with {donor} training",
        fontweight="bold",
        fontsize=14,
    )
    metrics = ("net_all", "net_spp_timely", "net_spp_late_or_persistent")
    titles = ("All test cues", "SPP already timely¹", "SPP late or persistent¹")
    for ax, field, title in zip(axes, metrics, titles):
        y = np.arange(len(rows))
        vals = [r[prefix + field] for r in rows]
        bars = ax.barh(
            y, vals, color=["#147d80" if n >= 0 else "#b8602a" for n in vals]
        )
        ax.axvline(0, color="#657077", lw=0.75)
        ax.bar_label(bars, fmt="{:+,.0f}", padding=3, fontsize=8)
        ax.set_yticks(
            y,
            [r["workload"] for r in rows] if field == metrics[0] else [""] * len(rows),
        )
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.margins(x=0.30)
        ax.set_xlabel("Net exact, legal, lead-qualified targets", fontsize=8)
    fig.text(
        0.01,
        -0.045,
        "¹ Aligned original NoPF misses. Differences compare ordered global history with the same-PC + region probe at one address per cue.",
        fontsize=8,
    )
    fig.savefig(
        HERE / "figures" / file,
        bbox_inches="tight",
        metadata={"Date": None},
    )
    plt.close(fig)


def main():
    rows = summarize()
    with (HERE / "external_evidence.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    doc = [
        "# Does additional history help on external recordings?",
        "",
        "One donor, one eighth-arrival displacement vocabulary and one proposed physical line per test cue. The comparator sees the current event, same-PC history and region-local history. The richer probe also sees 32 globally ordered events. Both see the same cues. A counted address must match the exact target, use an observed physical page and precede demand by at least 130 cycles. The independent recordings were chosen by source and workload before collecting their policy outcomes.",
        "",
        "![Measured overlap with SPP](figures/complementarity.svg)",
        "",
        "| External recording | Test cues | In donor vocabulary | Qualified local → global | Net on SPP-timely¹ | Net on SPP-late¹ | Net on SPP-persistent¹ | Fastest baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    arms = {"nopf": "NoPF", "stride": "Stride", "spp": "SPP", "multi": "Multi-delta"}
    for r in rows:
        doc.append(
            f"| {r['workload']} | {r['test_cues']:,} | {r['expressible']:,} | {r['local_qualified']:,} → {r['global_qualified']:,} | {r['net_spp_timely']:+,} | {r['late_global_qualified'] - r['late_local_qualified']:+,} | {r['persistent_global_qualified'] - r['persistent_local_qualified']:+,} | {arms[r['fastest_baseline']]} |"
        )
    doc += [
        "",
        "¹ These three columns require an original NoPF miss at the same dynamic operand. The overall column includes every sampled cue. Ordinary hits and no-LLC-tag outcomes have their own categories in the underlying table. A net count can hide both gains and losses. SPP outcomes come from a matched SPP replay. The 130-cycle lead is measured in the Stride history that supplied the cue. The two clocks cannot establish how many cycles earlier this proposed line would arrive in SPP. No sampled probe proposal in this table is automatically a timely first use or a saved cycle.",
        "",
        "![Three-donor sensitivity](figures/pooled_complementarity.svg)",
        "",
        "The full table includes the number of gains, ties and extrema across eight contiguous test blocks, donor vocabulary support, legal page reach, log loss, subgroup denominators and an equal-weight pool of three training programs. Coarse log loss includes the out-of-vocabulary class. When vocabulary coverage is low, better abstention can dominate the score. The comparison among conventional policies uses all 200M measured instructions. The single prospective replay separates the effect of one probe action from these offline counts. [Joined evidence](external_evidence.csv) · [Held-out scores](probes/scores.csv) · [Matched conventional outcomes](probes/conventional_outcomes.csv) · [Pooled donor comparison](probes/pooled_transfer.csv) · [Whole-run policy results](expansion/results/performance.csv) · [Actual offered address](one_offer/README.md)",
        "",
    ]
    (HERE / "WORKLOADS.md").write_text("\n".join(doc))
    figure(rows)
    figure(rows, "pooled", "pool_", "pooled_complementarity.svg")


if __name__ == "__main__":
    main()
