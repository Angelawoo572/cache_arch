"""Join completed conventional arms that finished after a destination was first scored."""

import csv

import numpy as np

from extract import RAW, STUDY
from fit import HORIZONS, counters, load, masks_for, predict, splits, train, write
from paired import run_pairs
from pooled import training_data
from suite import pairs


def main():
    models = {}
    with (STUDY / "expansion/sources.csv").open() as f:
        heldout = [r["case"] for r in csv.DictReader(f)]
    plan = pairs() + [("pooled_three", x) for x in heldout]
    scopes = ("same_pc", "local_region", "global32")
    for source, dest in plan:
        run = RAW / "fits" / f"matrix_{source}_to_{dest}"
        reference_score = (
            run
            if source != "pooled_three"
            else RAW / "fits" / f"matrix_bfs_discovery_to_{dest}"
        )
        if not (reference_score / "scores.csv").is_file():
            continue
        if not (STUDY / "raw/expansion/runs" / f"{dest}__nopf/complete.json").is_file():
            continue
        out = RAW / "fits" / f"joined_{source}_to_{dest}"
        known = set()
        for f in (run / "baseline_outcomes.csv", out / "baseline_outcomes.csv"):
            if f.is_file():
                with f.open() as stream:
                    known.update(
                        (r["arm"], r["scope"])
                        for r in csv.DictReader(stream)
                        if r["encoding"] == "delta" and int(r["horizon"]) == 8
                    )
        ready = {
            name
            for name in ("stride", "multi", "spp")
            if (
                STUDY / "raw/expansion/runs" / f"{dest}__{name}/complete.json"
            ).is_file()
        }
        pair_out = RAW / "paired" / f"external_{source}_to_{dest}"
        if (
            dest in heldout
            and ready == {"stride", "multi", "spp"}
            and not (pair_out / "paired.csv").is_file()
        ):
            run_pairs(source, [dest], pair_out)
        if not ({(arm, scope) for arm in ready for scope in scopes} - known):
            continue
        if source not in models:
            if source == "pooled_three":
                training = training_data()
            else:
                d, n = load(source)
                training = d.loc[splits(d, n)["train"]]
            ref = np.unique(training.pc)
            models[source] = (
                ref,
                {s: train(training, s, ref, 8, "delta") for s in scopes},
            )
        ref, ensemble = models[source]
        d, n = load(dest)
        cues = d.loc[splits(d, n)["test"]]
        masks = masks_for(dest, cues)
        rows = []
        for s, model in ensemble.items():
            pred = predict(model, cues, s, ref, 8, "delta")
            for name in sorted(ready):
                k = HORIZONS.index(8)
                for status, code in (
                    ("no_tag", 0),
                    ("ordinary", 1),
                    ("persistent", 2),
                    ("late", 6),
                    ("timely", 9),
                ):
                    eligible = (masks["nopf"][:, k] == 2) & (masks[name][:, k] == code)
                    rows.append(
                        dict(
                            source=source,
                            destination=dest,
                            split="test",
                            encoding="delta",
                            horizon=8,
                            scope=s,
                            arm=name,
                            original_miss_outcome=status,
                            **counters(pred, eligible),
                        )
                    )
        out.mkdir(parents=True, exist_ok=True)
        write(out / "baseline_outcomes.csv", rows)


if __name__ == "__main__":
    main()
