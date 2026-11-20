"""Construct optional core-side features for admitted LLC anchors from recorded operands."""

import argparse
import json

import numpy as np

from extract import RAW, STUDY, locate

RECORD = np.dtype(
    [
        ("pc", "<u8"),
        ("branch", "u1"),
        ("taken", "u1"),
        ("dst", "u1", (2,)),
        ("src", "u1", (4,)),
        ("memory", "u1", (48,)),
    ]
)
assert RECORD.itemsize == 64


def prepare(case):
    config = json.loads((locate(case, "stride") / "complete.json").read_text())[
        "config"
    ]
    with np.load(RAW / f"{case}__stride.npz", allow_pickle=False) as a:
        keys, pcs = a["key"], a["pc"]
    trace = np.memmap(STUDY.parents[1] / config["traces"][0], mode="r", dtype=RECORD)
    indices = (keys >> 2) - 1 - config.get("source_ordinal_base", 0)
    assert indices.min() >= 320 and indices.max() < len(trace)
    anchors = trace[indices]
    if not np.array_equal(anchors["pc"], pcs):
        raise ValueError("Instruction PC in source differs from LLC trigger")
    values = {
        "core_sources": (anchors["src"] != 0).sum(axis=1),
        "core_dests": (anchors["dst"] != 0).sum(axis=1),
        "core_operand": keys % 4,
    }
    # Nearest recorded definition of each nonzero architectural source register.
    earlier = trace[indices[:, None] - np.arange(1, 33)[None, :]]
    for r in range(4):
        register = anchors["src"][:, r]
        match = (earlier["dst"] == register[:, None, None]).any(axis=2) & (
            register[:, None] != 0
        )
        values[f"core_dep_{r}"] = np.where(
            match.any(axis=1), match.argmax(axis=1) + 1, 0
        )
    # The oldest end of the ROB has passed this band whenever the cue is resident.
    older = trace[indices[:, None] - np.arange(257, 321)[None, :]]
    flag = older["branch"] != 0
    order = np.argsort(~flag, axis=1, kind="stable")[:, :4]
    for j in range(4):
        ix = order[:, j]
        valid = flag[np.arange(len(ix)), ix]
        values[f"core_branch_age_{j}"] = np.where(valid, 257 + ix, 0)
        values[f"core_branch_taken_{j}"] = np.where(
            valid, older["taken"][np.arange(len(ix)), ix].astype(np.int64), -1
        )
    np.savez_compressed(RAW / f"{case}__sideband.npz", key=keys, **values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+")
    for case in parser.parse_args().cases:
        prepare(case)
