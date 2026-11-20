"""Count full-retirement NoPF misses present and absent at each policy's physical ingress."""

import argparse
from contextlib import ExitStack
import gzip

import numpy as np

from build import ARMS, CACHE
from measure import HERE, save
from sequence import ROOT, Sequence


def admission_keys(seq):
    keys = np.empty(seq.size, dtype="<u8")
    for i, (lo, hi, _, _) in enumerate(seq.meta["chunks"]):
        keys[lo:hi] = seq._events(i)[:, 0]
    keys.sort()
    return keys


def measure(case):
    seqs = [Sequence(ROOT / f"{case}__{arm}") for arm in ARMS]
    refs = seqs[0].meta["outcomes"]
    first, last = seqs[0].meta["first_ordinal"], seqs[0].meta["last_ordinal"]
    for seq in seqs[1:]:
        if (
            seq.meta["outcomes"] != refs
            or seq.meta["first_ordinal"] != first
            or seq.meta["last_ordinal"] != last
        ):
            raise ValueError("Aligned domains differ")
    keys = [admission_keys(seq) for seq in seqs]
    counts = np.zeros((4, 2, 16), dtype=np.int64)
    original = 0
    with ExitStack() as stack:
        files = [
            stack.enter_context(
                gzip.open(CACHE / path / "outcomes_core0.u16le.gz", "rb")
            )
            for path in refs
        ]
        block = 262144
        for lo in range(first, last + 1, block):
            n = min(block, last + 1 - lo)
            raw = [np.frombuffer(f.read(2 * n), dtype="<u2") for f in files]
            if any(len(r) != n for r in raw):
                raise ValueError("Truncated reference")
            for operand in range(4):
                pos = np.flatnonzero(((raw[0] >> (4 * operand)) & 15) == 2)
                original += len(pos)
                wanted = ((pos.astype("<u8") + lo) << 2) | operand
                for arm, k in enumerate(keys):
                    # Limit binary searches to keys in this block of source ordinals.
                    ix = k.searchsorted(np.array([lo << 2, (lo + n) << 2], dtype="<u8"))
                    subset = k[ix[0] : ix[1]]
                    if len(subset):
                        slots = subset.searchsorted(wanted)
                        admitted = (slots < len(subset)) & (
                            subset[np.minimum(slots, len(subset) - 1)] == wanted
                        )
                    else:
                        admitted = np.zeros(len(pos), dtype=bool)
                    mask = (raw[arm][pos] >> (4 * operand)) & 15
                    counts[arm] += np.bincount(
                        16 * admitted + mask, minlength=32
                    ).reshape(2, 16)
        if any(f.read(1) for f in files):
            raise ValueError("ROI longer than registered")
    rows = []
    for arm, seq in enumerate(seqs):
        if counts[arm].sum() != original:
            raise ValueError("Reference denominator changed")
        for admitted in (0, 1):
            cell = counts[arm, admitted]
            if sum(cell[[0, 1, 2, 6, 9]]) != sum(cell):
                raise ValueError("Unknown matched outcome")
            rows.append(
                dict(
                    case=case,
                    policy=ARMS[arm],
                    measured_instructions=last - first + 1,
                    original_llc_misses=original,
                    total_arm_admissions=seq.size,
                    original_operand_present_at_arm_ingress=admitted,
                    matched_original_operands=int(cell.sum()),
                    no_llc_tag_or_claim=cell[0],
                    ordinary_tag_hit=cell[1],
                    plain_miss=cell[2],
                    late_first_use=cell[6],
                    timely_first_use=cell[9],
                )
            )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+")
    args = parser.parse_args()
    for case in args.cases:
        save(HERE / "results" / f"{case}_visibility.csv", measure(case))
