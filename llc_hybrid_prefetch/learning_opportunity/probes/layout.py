"""Pair frozen predictions by measured source operand across real physical layouts."""

import numpy as np

from extract import HERE, RAW
from fit import load, predict, splits, train, write


def run(source, dest):
    original, n = load(source)
    train_slice = original.loc[splits(original, n)["train"]]
    reference = np.unique(train_slice.pc)
    a = original.loc[splits(original, n)["test"]].set_index("key", drop=False)
    b, m = load(dest)
    b = b.loc[splits(b, m)["test"]].set_index("key", drop=False)
    destination_size = len(b)
    both = np.intersect1d(a.key, b.key)
    a, b = a.loc[both], b.loc[both]
    rows = []
    for scope in ("same_pc", "global32"):
        model = train(train_slice, scope, reference, 8, "delta")
        p = predict(model, a, scope, reference, 8, "delta")
        q = predict(model, b, scope, reference, 8, "delta")
        same_target = a.y8_key.to_numpy() == b.y8_key.to_numpy()
        moved = a.y8_line.to_numpy() != b.y8_line.to_numpy()
        for cohort, mask in (
            ("same_cue", np.ones(len(a), bool)),
            ("same_cue_and_target", same_target),
            ("same_cue_target_and_line_moved", same_target & moved),
        ):
            rows.append(
                dict(
                    source=source,
                    destination=dest,
                    scope=scope,
                    cohort=cohort,
                    destination_test_anchors=destination_size,
                    paired=int(mask.sum()),
                    original_bits=float(p["loss"][mask].mean()) if mask.any() else "",
                    alternate_bits=float(q["loss"][mask].mean()) if mask.any() else "",
                    both_forced_lead=int(
                        (mask & p["forced_deadline"] & q["forced_deadline"]).sum()
                    ),
                    original_only_forced_lead=int(
                        (mask & p["forced_deadline"] & ~q["forced_deadline"]).sum()
                    ),
                    alternate_only_forced_lead=int(
                        (mask & ~p["forced_deadline"] & q["forced_deadline"]).sum()
                    ),
                    original_only_abstention_allowed=int(
                        (mask & p["deadline"] & ~q["deadline"]).sum()
                    ),
                    alternate_only_abstention_allowed=int(
                        (mask & ~p["deadline"] & q["deadline"]).sum()
                    ),
                )
            )
    return rows


if __name__ == "__main__":
    all_rows = []
    for source, dest in (
        ("mcf17_layout_original", "mcf17_layout_reversed"),
        ("gcc734", "gcc734_slot1"),
        ("cc5", "cc5_slot1"),
    ):
        if all((RAW / f"{c}__stride.npz").exists() for c in (source, dest)):
            all_rows.extend(run(source, dest))
    write(HERE / "matched_placements.csv", all_rows)
