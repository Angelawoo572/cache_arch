"""Check that measuring PageRank through EOF preserves the shared prefix's dynamic LLC outcomes."""

import gzip
import json
import struct
from run import HERE, RAW
from summarize import POLICIES, write

CHUNK = 2 * 1024 * 1024


def check(policy):
    a = RAW / "runs" / ("pr_heldout__" + policy)
    b = RAW / "runs" / ("pr_heldout_entire__" + policy)
    aa, bb = [json.loads((p / "complete.json").read_text())["config"] for p in (a, b)]
    if (
        aa["warmup"] != bb["warmup"]
        or aa["slots"] != bb["slots"]
        or aa["roi"] >= bb["roi"]
    ):
        raise ValueError("The two views do not share warmup, slot and a strict prefix")
    short_roi = aa["roi"]
    offset = changed = 0
    first = None
    with (
        gzip.open(a / "outcomes_core0.u16le.gz", "rb") as x,
        gzip.open(b / "outcomes_core0.u16le.gz", "rb") as y,
    ):
        while data := x.read(CHUNK):
            other = y.read(len(data))
            if len(other) != len(data) or len(data) % 2:
                raise ValueError("Incomplete dynamic outcomes")
            if data != other:
                left = struct.iter_unpack("<H", data)
                right = struct.iter_unpack("<H", other)
                for i, (p, q) in enumerate(zip(left, right)):
                    if p != q:
                        changed += 1
                        if first is None:
                            first = offset + i
            offset += len(data) // 2
    if offset != short_roi or (first is not None and first < short_roi - 4096):
        raise ValueError("Overlap diverged before conservative end-of-window guard")
    return dict(
        policy=policy,
        common_warmup=aa["warmup"],
        overlapping_retired_instructions=offset,
        identical_outcome_words=offset - changed,
        different_outcome_words=changed,
        first_different_source_ordinal=first + aa["warmup"] + 1
        if first is not None
        else "",
        end_guard_instructions=4096,
    )


if __name__ == "__main__":
    out = HERE / "entire"
    out.mkdir(exist_ok=True)
    write(out / "pagerank_overlap.csv", [check(p) for p in POLICIES])
