"""Bound the incremental timely population over every dynamic operand and separate reference replays."""

from contextlib import ExitStack
from collections import Counter
import gzip
import numpy as np
from replay import CASES
from streams import location
from characterize import publish


def compare(case, refs):
    names = ("nopf", "shape", *refs)
    folders = [location(case, p, "full") for p in names]
    if any(not (f / "complete.json").is_file() for f in folders):
        return None
    result = Counter()
    with ExitStack() as stack:
        inputs = [
            stack.enter_context(gzip.open(f / "outcomes_core0.u16le.gz", "rb"))
            for f in folders
        ]
        while True:
            buf = [f.read(2 * 1024 * 1024) for f in inputs]
            if len(set(map(len, buf))) != 1:
                raise ValueError("length mismatch in full joint comparison")
            if not buf[0]:
                break
            words = [np.frombuffer(b, dtype="<u2") for b in buf]
            result["instructions"] += len(words[0])
            for shift in (0, 4, 8, 12):
                original = (words[0] & (2 << shift)) != 0
                if not original.any():
                    continue
                hybrid = (words[1] & (8 << shift)) != 0
                timely = np.zeros_like(original)
                all_failed = original.copy()
                for w in words[2:]:
                    timely |= (w & (8 << shift)) != 0
                    all_failed &= (w & (6 << shift)) != 0
                joint = original & hybrid & ~timely
                result["original_misses"] += int(np.sum(original))
                result["hybrid_original_timely"] += int(np.sum(hybrid & original))
                result["any_reference_original_timely"] += int(
                    np.sum(timely & original)
                )
                result["hybrid_timely_none_reference_timely"] += int(np.sum(joint))
                result["hybrid_timely_all_reference_persistent_or_late"] += int(
                    np.sum(joint & all_failed)
                )
                result["any_reference_timely_hybrid_became_non_timely"] += int(
                    np.sum(timely & original & ~hybrid)
                )
    expected = 450000001 if case == "pr_heldout_entire" else 200000000
    if result["instructions"] != expected:
        raise ValueError("unaligned fixed work")
    return dict(case=case, reference_arms=" ".join(refs), **result)


if __name__ == "__main__":
    rows = []
    for case in (*CASES, "pr_heldout_entire"):
        row = compare(case, ("stride", "multi", "spp", "temporal"))
        if row:
            rows.append(row)
    publish("full_joint_coverage", rows)
    with_lookahead = []
    for case in ("pr_heldout", "soplex06_heldout"):
        row = compare(case, ("stride", "multi", "spp", "temporal", "look2"))
        if row:
            with_lookahead.append(row)
    publish("heldout_lookahead_union", with_lookahead)
