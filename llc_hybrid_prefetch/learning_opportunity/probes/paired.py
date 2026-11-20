"""Join alternate probe predictions on the same held-out cue and future operand."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from fit import HORIZONS, load, masks_for, predict, splits, train, write
from pooled import training_data


def run_pairs(source, destinations, out):
    if source == "pooled_three":
        training = training_data()
    else:
        frame, n = load(source)
        training = frame.loc[splits(frame, n)["train"]].copy()
    reference = np.unique(training.pc)
    h = 8
    models = {
        mode: train(training, mode, reference, h, "delta")
        for mode in ("same_pc", "local_region", "global32", "hide_other_pc")
    }
    rows = []
    examples = []
    for dest in destinations:
        frame, n = load(dest)
        ev = frame.loc[splits(frame, n)["test"]].copy()
        arms = masks_for(dest, ev)
        preds = {
            mode: predict(model, ev, mode, reference, h, "delta")
            for mode, model in models.items()
        }
        target = ev.y8_line.to_numpy()
        displacement = target - ev.line.to_numpy()
        for comparator in ("same_pc", "local_region", "hide_other_pc"):
            base = preds[comparator]
            rich = preds["global32"]
            groups = {
                "all": np.ones(len(ev), bool),
                "unseen_pc_vs_sampled_train": ~np.isin(ev.pc, reference),
                "new_line_in_known_page": (ev.y8_seen == 0).to_numpy()
                & (ev.y8_seenpage != 0).to_numpy(),
            }
            k = HORIZONS.index(h)
            if "nopf" in arms:
                for arm in ("stride", "multi", "spp"):
                    if arm in arms:
                        for kind, code in (
                            ("miss", 2),
                            ("late", 6),
                            ("timely", 9),
                            ("ordinary_hit", 1),
                            ("no_tag", 0),
                        ):
                            groups["original_" + arm + "_" + kind] = (
                                arms["nopf"][:, k] == 2
                            ) & (arms[arm][:, k] == code)
            for group, mask in groups.items():
                for policy, field in (
                    ("abstention_allowed", "deadline"),
                    ("one_proposal", "forced_deadline"),
                ):
                    gain = mask & rich[field] & ~base[field]
                    loss = mask & ~rich[field] & base[field]
                    counts = Counter(map(int, displacement[gain]))
                    common = counts.most_common(1)
                    rows.append(
                        dict(
                            source=source,
                            destination=dest,
                            comparator=comparator,
                            horizon=h,
                            encoding="delta",
                            cohort=group,
                            decision=policy,
                            anchors=int(mask.sum()),
                            rich_only=int(gain.sum()),
                            comparator_only=int(loss.sum()),
                            both=int((mask & rich[field] & base[field]).sum()),
                            distinct_gain_deltas=len(counts),
                            most_frequent_gain_delta=common[0][0] if common else "",
                            most_frequent_gain_count=common[0][1] if common else 0,
                            distinct_gain_cue_pcs=len(
                                np.unique(ev.pc.to_numpy()[gain])
                            ),
                            distinct_gain_target_regions=len(
                                np.unique(target[gain] // 4096)
                            ),
                        )
                    )
                    if (
                        policy == "abstention_allowed"
                        and group == "original_spp_late"
                        and gain.any()
                    ):
                        j = np.flatnonzero(gain)[0]
                        e = ev.iloc[j]
                        examples.append(
                            dict(
                                source=source,
                                destination=dest,
                                comparator=comparator,
                                cue_key=int(e.key),
                                cue_pc=int(e.pc),
                                cue_line=int(e.line),
                                target_key=int(e.y8_key),
                                target_pc=int(e.y8_pc),
                                target_line=int(e.y8_line),
                                inferred_delta=int(displacement[j]),
                                lead=int(rich["lead"][j]),
                                target_seen=int(e.y8_seen),
                                target_page_seen=int(e.y8_seenpage),
                            )
                        )
    out.mkdir(exist_ok=True, parents=True)
    write(out / "paired.csv", rows)
    write(out / "paired_examples.csv", examples)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--dest", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run_pairs(a.source, a.dest, a.out)
