"""Exhaustively verify label and prefix queries on a small, tied-cycle sequence."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from build import write_postings
from measure import grouped_offsets
from sequence import FIELDS, HORIZONS, ROOT, SCOPES, Sequence


def fixture(folder):
    folder.mkdir()
    ranks = np.arange(144)
    pcs = np.array([8, 8, 9, 10, 8, 11])[ranks % 6]
    # Late first observation exercises page, region and line novelty.
    lines = np.array([1, 65, 1, 67, 7009, 66, 8300])[ranks % 7]
    lines[95:] += 2**15
    cycles = ranks // 3 * 7
    events = np.column_stack([4 * ranks + 1, pcs, lines, cycles]).astype("<u8")
    # Deliberate source-ordinal reversal in the arrival stream.
    events[[30, 31], 0] = events[[31, 30], 0]
    step = 17
    chunks = []
    for i, lo in enumerate(range(0, len(events), step)):
        batch = events[lo : lo + step]
        masks = np.array([int(v) for v in (cycles[lo : lo + step] % 2)], dtype="<u2")
        np.savez_compressed(
            folder / f"events_{i:04d}.npz", **dict(zip(FIELDS, batch.T))
        )
        np.savez_compressed(folder / f"outcomes_{i:04d}.npz", masks=masks)
        chunks.append((lo, lo + len(batch), int(batch[0, 3]), int(batch[-1, 3])))
    for key in ("pc", "line", "page", "region"):
        write_postings(folder, events, key)
    (folder / "index.json").write_text(
        json.dumps(dict(observations=len(events), chunk_rows=step, chunks=chunks))
    )
    return events


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=ROOT, prefix="test-") as tmp:
        folder = Path(tmp) / "fixture"
        events = fixture(folder)
        seq = Sequence(folder)
        anchors = np.arange(len(events))
        steps = np.array([-64, -3, -1, 1, 2, 4, 8, 16, 32, 64])
        for scope in SCOPES:
            if scope == "global":
                valid = np.ones((len(events), len(events)), dtype=bool)
            else:
                kind, key = scope.split("_")
                values = (
                    events[:, 1]
                    if key == "pc"
                    else events[:, 2] >> (6 if key == "page" else 12)
                )
                same = values[:, None] == values[None, :]
                valid = same if kind == "same" else ~same
                a, b = grouped_offsets(seq, events, anchors, key)
                for i in anchors:
                    np.testing.assert_array_equal(
                        (a if kind == "same" else b)[i],
                        seq.relative(i, HORIZONS, scope),
                    )
            for i in anchors:
                before = anchors[valid[i] & (anchors < i)]
                after = anchors[valid[i] & (anchors > i)]
                truth = [
                    before[s]
                    if s < 0 and -s <= len(before)
                    else after[s - 1]
                    if 0 < s <= len(after)
                    else -1
                    for s in steps
                ]
                np.testing.assert_array_equal(seq.relative(i, steps, scope), truth)
                stop = min(i + 7, len(events))
                np.testing.assert_array_equal(
                    seq.relative(i, steps, scope, stop=stop),
                    [v if v < stop else -1 for v in truth],
                )
                context = seq.context(i, 8, scope)
                np.testing.assert_array_equal(context["ranks"][:-1], before[-8:])
                assert context["ranks"][-1] == i and "outcomes" not in context
        batches = list(
            seq.batches(20, 32, history=5, horizons=[1, 4, 16], batch_size=4)
        )
        assert len(batches) == 3
        for batch in batches:
            inp, lab = batch["observations"], batch["labels"]
            for ix, rank in enumerate(inp["anchor_ranks"]):
                ok = inp["history_valid"][ix]
                np.testing.assert_array_equal(
                    inp["history"][ix, ok], events[max(20, rank - 4) : rank + 1]
                )
                dest = rank + np.array([1, 4, 16])
                valid = dest < 32
                np.testing.assert_array_equal(lab["target_valid"][ix], valid)
                np.testing.assert_array_equal(
                    lab["future"][ix, valid], events[dest[valid]]
                )
                assert np.all(lab["future"][ix, ~valid] == 0)
        for horizons in ([], [0], [-1, 1]):
            with np.testing.assert_raises(ValueError):
                seq.future(10, horizons)
        for i in anchors:
            for width in (0, 7, 19, 100):
                result = seq.time_window(i, width)
                ranks = anchors[(anchors > i) & (events[:, 3] <= events[i, 3] + width)]
                np.testing.assert_array_equal(result["ranks"], ranks)
                np.testing.assert_array_equal(
                    result["lines"], np.unique(events[ranks, 2])
                )
                assert result["complete"] == bool(events[-1, 3] > events[i, 3] + width)
            future = seq.future(i)
            r = future["ranks"][future["valid"]]
            np.testing.assert_array_equal(
                future["line_seen"], np.isin(events[r, 2], events[: i + 1, 2])
            )
            np.testing.assert_array_equal(
                future["region_seen"],
                np.isin(events[r, 2] >> 12, events[: i + 1, 2] >> 12),
            )
            assert future["outcomes"].shape == (len(r), 4)
    print(
        "Passed exhaustive scope, first-observation, cutoff, chunk, timing-tie and causal-history tests."
    )


if __name__ == "__main__":
    main()
