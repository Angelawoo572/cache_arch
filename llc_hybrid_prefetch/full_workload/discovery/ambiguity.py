"""Quantify local ambiguity and held-out log loss separately from timely-miss intervention success."""

from collections import Counter, defaultdict
from math import log2
from phase import MODES, H, phase_keys, readcase
from characterize import publish
from replay import CASES
from pattern import snapshots, legal

OTHER = 1025
VOCAB = 1026  # 1025 signed displacements including zero, plus the inexpressible-address outcome.


def outcome(z, i, h, pages):
    source = int(z["pa"][i])
    dest = int(z["pa"][i + h])
    d = dest - source
    if d == 0:
        return 512
    return d + 512 if legal(source, dest, pages) else OTHER


def fit(z):
    obs, _ = snapshots(z)
    g = defaultdict(Counter)
    fits = defaultdict(lambda: defaultdict(Counter))
    for i, (ks, pages) in enumerate(obs[:-16]):
        for h in H:
            y = outcome(z, i, h, pages)
            g[h][y] += 1
            for mode, key in zip(MODES, phase_keys(ks)):
                fits[h, mode][key][y] += 1
    kept = {}
    for ix, rows in fits.items():
        eligible = ((key, sum(a.values()), a) for key, a in rows.items())
        kept[ix] = {
            key: (n, a)
            for key, n, a in sorted(eligible, key=lambda r: (-r[1], r[0]))[:256]
            if n >= 4
        }
    totals = {h: sum(v.values()) for h, v in g.items()}
    return g, totals, kept


def eval_part(case, donor, q, z, trained):
    obs, _ = snapshots(z)
    g, totals, kept = trained
    scored = defaultdict(Counter)
    observed = defaultdict(lambda: defaultdict(Counter))
    for i, (ks, pages) in enumerate(obs[:-16]):
        for h in H:
            y = outcome(z, i, h, pages)
            prior = (g[h][y] + 0.5) / (totals[h] + 0.5 * VOCAB)
            observed[h][ks[0]][y] += 1
            for mode, key in zip(MODES, phase_keys(ks)):
                counts = kept.get((h, mode), {}).get(key)
                prob = (
                    prior
                    if counts is None
                    else (counts[1][y] + 8 * prior) / (counts[0] + 8)
                )
                c = scored[h, mode]
                c["anchors"] += 1
                c["supported"] += counts is not None
                c["code_bits"] -= log2(prob)
                c["prior_bits"] -= log2(prior)
    rows = []
    for (h, mode), c in scored.items():
        local_ambig = {
            k
            for k, actions in observed[h].items()
            if sum(v >= 2 for a, v in actions.items() if a != OTHER) >= 2
        }
        n = c["anchors"]
        local = scored[h, "local"]["code_bits"] / n
        rows.append(
            dict(
                case=case,
                donor=donor,
                quarter=q,
                horizon=h,
                mode=mode,
                anchors=n,
                supported=c["supported"],
                mean_bits=c["code_bits"] / n,
                prior_mean_bits=c["prior_bits"] / n,
                bits_saved_vs_local=local - c["code_bits"] / n,
                distinct_local_contexts=len(observed[h]),
                ambiguous_local_contexts=len(local_ambig),
                anchors_in_ambiguous_local_contexts=sum(
                    sum(observed[h][k].values()) for k in local_ambig
                ),
            )
        )
    return rows


if __name__ == "__main__":
    donors = {}
    for donor in ("mcf17_discovery", "bfs_discovery"):
        a = readcase(donor)
        if a is not None:
            donors[donor] = fit(a["stride"][1])
    out = []
    for case in (
        *CASES,
        "mcf17_discovery_far",
        "bfs_discovery_far",
        "pr_heldout_entire",
    ):
        a = readcase(case)
        donor = (
            "bfs_discovery" if case.startswith(("bfs", "pr_")) else "mcf17_discovery"
        )
        if a is None or donor not in donors:
            continue
        for q, z in a["stride"].items():
            if q != 1 or donor != case:
                out.extend(eval_part(case, donor, q, z, donors[donor]))
    publish("phase_address_information", out)
