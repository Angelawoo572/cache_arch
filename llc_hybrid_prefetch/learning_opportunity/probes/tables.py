"""Assemble compact, inspectable results from completed probe pairs."""

import json

import numpy as np
import pandas as pd

from extract import HERE, RAW, publish_text

KEY = ["source", "destination", "split", "encoding", "horizon", "scope"]


def collect(folder, name, keys):
    frames = [pd.read_csv(f) for f in sorted(folder.glob("*/" + name + ".csv"))]
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    # The pressure supplement repeats one core-side check. Prefer the richest record.
    result["complete_fields"] = result.notna().sum(axis=1)
    result = result.sort_values("complete_fields").drop_duplicates(keys, keep="last")
    return result.drop(columns="complete_fields").sort_values(keys)


def save(name, frame):
    with publish_text(HERE / (name + ".csv")) as f:
        frame.to_csv(f, index=False, float_format="%.7g", lineterminator="\n")


def main():
    scores = collect(RAW / "fits", "scores", KEY)
    if scores.empty:
        raise RuntimeError("No completed fits")
    save("scores", scores)
    test = scores.loc[scores.split == "test"].copy()
    blocks = collect(RAW / "fits", "blocks", [*KEY, "block"])
    base_keys = ["source", "destination", "encoding", "horizon"]
    compare_scopes = ["same_pc", "local_region", "hide_other_pc"]
    columns = [
        *base_keys,
        "scope",
        "anchors",
        "mean_bits",
        "conditional_mean_bits",
        "correct",
        "deadline",
        "forced_deadline",
    ]
    base = (
        test[columns]
        .loc[test.scope.isin(compare_scopes)]
        .rename(columns={"scope": "reference"})
    )
    comparisons = (
        test[columns]
        .loc[test.scope == "global32"]
        .merge(base, on=base_keys, suffixes=("_rich", "_ref"))
    )
    per_block = blocks.pivot(
        index=[*base_keys, "block"], columns="scope", values="mean_bits"
    ).reset_index()
    for i, row in comparisons.iterrows():
        sel = per_block
        for k in base_keys:
            sel = sel.loc[sel[k] == row[k]]
        paired = sel[row.reference] - sel["global32"]
        comparisons.loc[i, "blocks_scored"] = int(paired.notna().sum())
        comparisons.loc[i, "blocks_with_lower_loss"] = int((paired > 0).sum())
    comparisons["bits_saved"] = comparisons.mean_bits_ref - comparisons.mean_bits_rich
    comparisons["address_bits_saved_given_expressible"] = (
        comparisons.conditional_mean_bits_ref - comparisons.conditional_mean_bits_rich
    )
    comparisons["net_lead_hits"] = comparisons.deadline_rich - comparisons.deadline_ref
    comparisons["net_forced_lead_hits"] = (
        comparisons.forced_deadline_rich - comparisons.forced_deadline_ref
    )
    save(
        "history_value",
        comparisons[
            [
                *base_keys,
                "reference",
                "anchors_rich",
                "bits_saved",
                "address_bits_saved_given_expressible",
                "net_lead_hits",
                "net_forced_lead_hits",
                "blocks_scored",
                "blocks_with_lower_loss",
            ]
        ],
    )
    strata = collect(RAW / "fits", "strata", [*KEY, "stratum"])
    keep = strata.loc[
        (strata.horizon == 8) & (strata.scope.isin(["same_pc", "global32"]))
    ].copy()
    save("unseen_contexts", keep)
    native = collect(
        RAW / "fits", "baseline_outcomes", [*KEY, "arm", "original_miss_outcome"]
    )
    keep = native.loc[
        (native.horizon == 8)
        & (native.encoding == "delta")
        & (native.scope.isin(["same_pc", "local_region", "global32"]))
    ].copy()
    save("conventional_outcomes", keep)
    witnesses = collect(RAW / "fits", "examples", [*KEY, "stratum"])
    save("examples", witnesses)
    pairs = collect(
        RAW / "paired",
        "paired",
        ["source", "destination", "cohort", "comparator", "decision"],
    )
    save("paired_targets", pairs)
    paired_examples = collect(
        RAW / "paired", "paired_examples", ["source", "destination", "comparator"]
    )
    save("paired_examples", paired_examples)
    summary = []
    for f in sorted(RAW.glob("*__stride.json")):
        data = json.loads(f.read_text())
        n = data["arrival_count"]
        with np.load(f.with_suffix(".npz")) as a:
            idx = a["index"]
            first, second = 3 * n // 5, 4 * n // 5
            n_train = int((idx + 32 < first).sum())
            n_dev = int(((idx >= first) & (idx + 32 < second)).sum())
            n_test = int(((idx >= second) & (idx + 32 < n)).sum())
            summary.append(
                {
                    **data,
                    "train": n_train,
                    "dev": n_dev,
                    "test": n_test,
                    "purged": len(idx) - n_train - n_dev - n_test,
                }
            )
    save("sampling", pd.DataFrame(summary))
    print(
        len(scores),
        "population scores;",
        len(comparisons),
        "paired scope comparisons;",
        len(summary),
        "streams",
    )


if __name__ == "__main__":
    main()
