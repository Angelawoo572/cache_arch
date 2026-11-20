"""Measure address loss inside locally ambiguous contexts identified on training only."""

from collections import Counter, defaultdict

import numpy as np

from extract import HERE
from fit import action_codes, counters, load, predict, splits, train, write


def contexts(frame, include_id):
    components = [
        frame.line.to_numpy() % 64,
        frame.p1_delta.to_numpy(),
        frame.p2_delta.to_numpy(),
    ]
    if include_id:
        components.insert(0, frame.pc.to_numpy())
    return zip(*components)


def study(source, destinations):
    donor, n = load(source)
    training = donor.loc[splits(donor, n)["train"]]
    h, enc = 8, "delta"
    reference = np.unique(training.pc)
    models = {
        scope: train(training, scope, reference, h, enc)
        for scope in ("same_pc", "global32")
    }
    vocab = set(models["same_pc"][0])
    actions = action_codes(training, h, enc)
    out = []
    for identity in (False, True):
        history = defaultdict(Counter)
        for key, action in zip(contexts(training, identity), actions):
            history[key][int(action) if action in vocab else None] += 1
        categories = {}
        for key, counts in history.items():
            if sum(counts.values()) < 6:
                continue
            distinct = sum(value is not None and n >= 2 for value, n in counts.items())
            categories[key] = (
                "multiple_supported_actions" if distinct >= 2 else "other_supported"
            )
        for dest in destinations:
            frame, count = load(dest)
            test = frame.loc[splits(frame, count)["test"]]
            groups = np.array(
                [
                    categories.get(key, "low_or_unseen_support")
                    for key in contexts(test, identity)
                ]
            )
            for mode, model in models.items():
                result = predict(model, test, mode, reference, h, enc)
                for group in (
                    "multiple_supported_actions",
                    "other_supported",
                    "low_or_unseen_support",
                ):
                    out.append(
                        dict(
                            source=source,
                            destination=dest,
                            scope=mode,
                            include_pc_id=int(identity),
                            training_contexts=sum(
                                v == group for v in categories.values()
                            )
                            if group != "low_or_unseen_support"
                            else "",
                            group=group,
                            **counters(result, groups == group),
                        )
                    )
    return out


if __name__ == "__main__":
    rows = []
    for source in ("bfs_discovery", "soplex06_heldout"):
        rows.extend(study(source, ("bfs_discovery", "soplex06_heldout")))
    write(HERE / "ambiguous_contexts.csv", rows)
