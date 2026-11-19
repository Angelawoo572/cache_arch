"""Describe what surrounds matched load outcomes and what preceded the original issue deadline."""

import csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "raw/history"
CASES = (
    "mcf-cactus-r1-c0",
    "cactus-r2-c0",
    "lbm-r1-c0",
    "gap_bfs8-c0",
    "gap_bfs8_lbm-c0",
)


def fraction(x):
    return "" if not len(x) else round(float(np.mean(x)), 6)


def main():
    rows = []
    for name in CASES:
        with np.load(CACHE / (name + ".npz")) as f:
            z = {
                k: f[k]
                for k in (
                    "baseline",
                    "hit",
                    "pf_kind",
                    "tag_t",
                    "t",
                    "pc",
                    "pa",
                    "pf_gen",
                    "pf_fill",
                )
            }
        for group, mask in (
            (
                "attributed_timely",
                (z["baseline"] == 0) & (z["hit"] == 1) & (z["pf_kind"] == 1),
            ),
            (
                "attributed_late",
                (z["baseline"] == 0) & (z["hit"] == 0) & (z["pf_kind"] == 2),
            ),
            (
                "other_persistent",
                (z["baseline"] == 0) & (z["hit"] == 0) & (z["pf_kind"] != 2),
            ),
            (
                "ordinary_hit_to_hit",
                (z["baseline"] == 1) & (z["hit"] == 1) & (z["pf_kind"] != 1),
            ),
        ):
            for lag in (-4, -1, 1, 4):
                targets = np.flatnonzero(mask)
                targets = targets[(targets + lag >= 0) & (targets + lag < len(mask))]
                neighbors = targets + lag
                ok = z["hit"][neighbors] >= 0
                before = (z["tag_t"][neighbors] >= 0) & (
                    z["tag_t"][neighbors] < z["t"][targets]
                )
                issue = z["pf_gen"][targets] >= 0
                rows.append(
                    dict(
                        case=name,
                        cohort=group,
                        nearby_ingress=lag,
                        count=len(targets),
                        same_pc_fraction=fraction(
                            z["pc"][targets] == z["pc"][neighbors]
                        ),
                        same_physical_page_fraction=fraction(
                            z["pa"][targets] // 64 == z["pa"][neighbors] // 64
                        ),
                        neighbor_hit_fraction=fraction(z["hit"][neighbors][ok]),
                        neighbor_outcome_known_at_target_ingress=fraction(before),
                        neighbor_outcome_known_at_original_proposal=fraction(
                            (
                                (z["tag_t"][neighbors] >= 0)
                                & (z["tag_t"][neighbors] < z["pf_gen"][targets])
                            )[issue]
                        ),
                        median_signed_ingress_gap_cycles=(
                            ""
                            if not len(targets)
                            else float(np.median(z["t"][neighbors] - z["t"][targets]))
                        ),
                    )
                )
    with (HERE / "hit_neighborhoods.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
