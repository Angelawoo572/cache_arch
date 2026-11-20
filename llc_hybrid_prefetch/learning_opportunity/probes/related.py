"""Compare causal probes on different definitions of a future LLC demand."""

import argparse
import sys

import numpy as np

from extract import HERE, STUDY
from fit import counters, load, predict, splits, train, write

sys.path.insert(0, str(STUDY / "labels"))
from sequence import ROOT, Sequence


def run(case):
    frame, count = load(case)
    seq = Sequence(ROOT / f"{case}__stride")
    if seq.size != count:
        raise ValueError("Raw and indexed sequences disagree")
    anchors = frame["index"].to_numpy()
    if not np.array_equal(seq.observations(anchors)[:, 0], frame.key):
        raise ValueError("Anchor key mismatch")
    parts = splits(frame, count)
    out = []
    for selector in ("same_pc", "other_pc", "other_region"):
        labels = np.array([seq.relative(int(i), [1, 8], selector) for i in anchors])
        for j, h in enumerate((1, 8)):
            ranks = labels[:, j]
            valid = ranks >= 0
            target = np.zeros((len(frame), 4), dtype=np.int64)
            target[valid] = seq.observations(ranks[valid])
            view = frame.assign(
                **{f"y{h}_line": target[:, 2], f"y{h}_cycle": target[:, 3]}
            )
            training = view.loc[parts["train"] & valid & (ranks < 3 * count // 5)]
            reference = np.unique(training.pc)
            testing = view.loc[parts["test"] & valid]
            for enc in ("delta", "page_offset"):
                for mode in ("current", "same_pc", "global32", "hide_other_pc"):
                    model = train(training, mode, reference, h, enc)
                    pred = predict(model, testing, mode, reference, h, enc)
                    out.append(
                        dict(
                            case=case,
                            selector=selector,
                            horizon=h,
                            encoding=enc,
                            scope=mode,
                            train=len(training),
                            test_eligible=int(parts["test"].sum()),
                            **counters(pred, np.ones(len(testing), bool)),
                        )
                    )
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cases", nargs="+")
    args = p.parse_args()
    results = []
    for case in args.cases:
        results.extend(run(case))
    write(HERE / "related_targets.csv", results)
