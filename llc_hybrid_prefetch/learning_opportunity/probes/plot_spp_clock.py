"""Plot request-service-qualified forecasts and one native MCF lifetime."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from extract import HERE

plt.rcParams.update(
    {"svg.fonttype": "none", "svg.hashsalt": "spp-forecast-clock", "font.size": 9}
)
rows = pd.read_csv(HERE / "spp_clock_claims.csv")
ex = (
    pd.read_csv(HERE / "spp_clock_examples.csv")
    .query("case=='mcf17_transfer' and scope=='global32' and claim=='late'")
    .iloc[0]
)
fig, (ax0, ax) = plt.subplots(
    2, 1, figsize=(8.9, 6.0), gridspec_kw={"height_ratios": [1.25, 1]}
)
fig.subplots_adjust(left=0.15, right=0.96, top=0.83, bottom=0.15, hspace=0.7)
colors = ["#426a8e", "#027a65"]
for k, (case, recording) in enumerate(
    [
        ("bfs_transfer", "BFS-14"),
        ("mcf17_transfer", "MCF17-1644B"),
        ("soplex06_heldout", "Soplex-92B"),
    ]
):
    for j, scope in enumerate(("local_region", "global32")):
        s = rows[
            (rows.case == case) & (rows.scope == scope) & (rows.claim == "late")
        ].set_index("policy")
        count = s.loc["one_proposal"].native_claimed_targets
        y = k + (j - 0.5) * 0.31
        pct = 100 * s.loc["one_proposal"].before_and_enough / count
        ax0.barh(
            y,
            pct,
            color=colors[j],
            height=0.25,
            label=("Local + region", "Add global 32")[j] if k == 0 else None,
        )
        ax0.text(
            pct + 0.5,
            y,
            f"{int(s.loc['one_proposal'].before_and_enough)}/{int(count):,}",
            va="center",
            fontsize=8,
        )
ax0.set_yticks(range(3), ["BFS-14", "MCF17-1644B", "Soplex-92B"])
ax0.invert_yaxis()
ax0.set_xlim(0, 54)
ax0.set_xlabel("Percent of sampled original misses that SPP claimed late")
ax0.set_title(
    "Correct line before SPP, with more lead than the observed proposal-to-fill service",
    loc="left",
    fontsize=10,
)
ax0.legend(ncol=2, frameon=False, loc="lower right")
ax0.spines[["right", "top", "left"]].set_visible(False)
ax0.tick_params(axis="y", length=0)
base = int(ex.cycle)
events = [
    ("Forecast", 0, 0.25),
    ("SPP proposal / accept", int(ex.proposed) - base, 0.37),
    ("Dispatch", int(ex.dispatch) - base, -0.31),
    ("Demand arrives", int(ex.y8_cycle) - base, 0.78),
    ("Late merge", int(ex.claim_cycle) - base, -0.71),
    ("Fill", int(ex["fill"]) - base, 0.35),
]
ax.hlines(0, 0, events[-1][1], color="#68747f", lw=1.5)
for name, value, y in events:
    ax.vlines(value, 0, y * 0.77, color="#65788b", lw=0.9)
    ax.plot(value, 0, "o", ms=4, color="#027a65" if name == "Forecast" else "#375a77")
    ha = (
        "right"
        if name == "SPP proposal / accept"
        else ("left" if name == "Fill" else "center")
    )
    ax.text(
        value,
        y,
        f"{name}\n+{value:,}",
        ha=ha,
        va="bottom" if y > 0 else "top",
        fontsize=8,
    )
ax.set_xlim(-65, 1250)
ax.set_ylim(-1.14, 1.23)
ax.set_yticks([])
ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_title("One MCF target, request 6,869,190", loc="left", fontsize=10)
ax.set_xlabel("Cycles since the forecast on the same SPP clock")
fig.suptitle(
    "When did the probe name the same line as SPP?",
    fontweight="bold",
    fontsize=15,
    x=0.025,
    ha="left",
)
fig.text(
    0.025,
    0.035,
    "Horizon 8; one concrete proposal per cue before legality filtering. Sources: spp_clock_claims.csv and spp_clock_examples.csv",
    fontsize=8,
)
fig.savefig(HERE / "spp_clock.svg", metadata={"Date": None})
plt.close(fig)
