"""Directed checks for chronology, distinct future labels and exact address decoding."""

import gzip
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np
import pandas as pd

from extract import HERE, RAW
from feedback import scan
from fit import action_codes, encode, features, OTHER, predict, splits


def export(exe, root, name, rows):
    source = root / (name + ".gz")
    with gzip.open(source, "wb") as f:
        for event in rows:
            f.write(struct.pack("<QQQQ", *event))
    out = root / (name + ".tsv")
    with out.open("wb") as f:
        subprocess.run(
            [str(exe), str(source), str(len(rows)), "1"], stdout=f, check=True
        )
    return pd.read_csv(out, sep="\t")


def main():
    RAW.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=RAW) as tmp:
        root = Path(tmp)
        exe = root / "sample"
        subprocess.run(
            [
                "c++",
                "-O2",
                "-std=c++17",
                str(HERE / "sample.cc"),
                "-lz",
                "-o",
                str(exe),
            ],
            check=True,
        )
        # Strict serial comparison must also hold across decompression block boundaries.
        serials = np.arange(65545, dtype="<u8") * 3
        rows_for_feedback = np.column_stack([serials, serials + 9])
        feedback_file = root / "feedback.gz"
        with gzip.open(feedback_file, "wb") as f:
            f.write(rows_for_feedback.tobytes())
        queries = np.array([0, 3, 4, serials[65536], serials[-1] + 2], dtype=np.int64)
        observed = scan(feedback_file, 2, lambda x: x, queries, 1)[:, :, 0]
        for i, serial in enumerate(queries):
            past = rows_for_feedback[serials < serial, 1]
            assert list(observed[i]) == [
                past[-lag] if len(past) >= lag else -1 for lag in (1, 2, 4, 8)
            ]
        rows = [
            (4 * (t + 100), 0x4000 + 4 * (t % 5), (t % 13) * 64 + t % 4, t * 19)
            for t in range(400)
        ]
        a = export(exe, root, "original", rows)
        cut = 185
        changed = rows[: cut + 1] + [
            (k, 0x9100 + i, 850004 + i, cy)
            for i, (k, pc, line, cy) in enumerate(rows[cut + 1 :])
        ]
        b = export(exe, root, "changed", changed)
        assert list(a["index"]) == list(range(len(rows) - 32))
        hist = [c for c in a if not c.startswith("y")]
        assert a.loc[cut, hist].equals(b.loc[cut, hist])
        assert a.loc[cut, "y1_line"] != b.loc[cut, "y1_line"]
        for h in (1, 8, 32):
            for i in (0, cut, len(a) - 1):
                future = rows[i + h]
                assert a.loc[i, f"y{h}_key"] == future[0]
                assert a.loc[i, f"y{h}_seen"] == (
                    future[2] in {r[2] for r in rows[: i + 1]}
                )
        frame = a.iloc[[50]].copy()
        reference = np.unique(a.pc)
        for mode in (
            "current",
            "same_pc",
            "region",
            "global32",
            "no_pc_ids",
            "with_time",
            "shuffled",
        ):
            assert np.array_equal(
                features(a.iloc[[cut]], mode, reference),
                features(b.iloc[[cut]], mode, reference),
            )
        for encoding in ("delta", "page_offset"):
            frame["y1_line"] = frame["bank1"] * 64 + 5
            code = action_codes(frame, 1, encoding)
            assert code[0] != OTHER
            vocab = np.array([code[0]])
            pred = predict(
                (vocab, np.array([0.01, 0.99]), None),
                frame,
                "prior",
                reference,
                1,
                encoding,
            )
            assert pred["hit"][0] and pred["correct"][0] and pred["legal"][0]
            assert encode(np.array([code[0] + 999]), vocab)[0] == 0
        split = splits(a, len(rows))
        assert (a.loc[split["train"], "index"] + 32).max() < 240
        assert (a.loc[split["dev"], "index"] + 32).max() < 320
        assert a.loc[split["test"], "index"].min() >= 320
    print(
        "Passed prefix invariance, 64-event sampler, exact labels, novelty, "
        "shifted target reconstruction, strict callback order and split purging."
    )


if __name__ == "__main__":
    main()
