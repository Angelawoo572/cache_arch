"""Pair actual Stride histories across simulated address-placement slots."""

import argparse

import numpy as np

from measure import HERE, full_arrays, sample, save
from sequence import ROOT, Sequence


def paired_indices(keys, query):
    """Return an index only for keys with exactly one occurrence."""
    order = keys.argsort(kind="stable")
    val = keys[order]
    first = val.searchsorted(query)
    end = val.searchsorted(query, side="right")
    unique = end - first == 1
    return np.where(unique, order[np.minimum(first, len(keys) - 1)], -1)


def compare(case):
    left = Sequence(ROOT / (case + "__stride"))
    right = Sequence(ROOT / (case + "_slot1__stride"))
    aobs, _ = full_arrays(left)
    bobs, _ = full_arrays(right)
    a = sample(left)
    q = aobs[a, 0]
    pa = paired_indices(aobs[:, 0], q)
    pb = paired_indices(bobs[:, 0], q)
    matched = (pa == a) & (pb >= 0)
    a, b = a[matched], pb[matched]
    np.testing.assert_array_equal(aobs[a, 1], bobs[b, 1])
    np.testing.assert_array_equal(aobs[a, 2] & 63, bobs[b, 2] & 63)
    rows = []
    for k in (1, 2, 4, 8, 16, 32, 64):
        valid = (a + k < len(aobs)) & (b + k < len(bobs))
        aa, bb = a[valid], b[valid]
        x, y = aobs[aa + k], bobs[bb + k]
        dx = x[:, 2].astype(np.int64) - aobs[aa, 2].astype(np.int64)
        dy = y[:, 2].astype(np.int64) - bobs[bb, 2].astype(np.int64)
        same_operand = x[:, 0] == y[:, 0]
        lead_left = x[:, 3] - aobs[aa, 3]
        lead_right = y[:, 3] - bobs[bb, 3]
        rows.append(
            dict(
                case=case,
                horizon=k,
                sampled_placement0_anchors=len(q),
                common_unique_anchors=len(a),
                jointly_observed_horizon=len(aa),
                anchor_exact_line_changed=int((aobs[aa, 2] != bobs[bb, 2]).sum()),
                same_target_operand=int(same_operand.sum()),
                same_target_pc=int((x[:, 1] == y[:, 1]).sum()),
                same_target_offset=int(((x[:, 2] & 63) == (y[:, 2] & 63)).sum()),
                same_signed_line_delta=int((dx == dy).sum()),
                matched_operand_changed_physical_line=int(
                    (same_operand & (x[:, 2] != y[:, 2])).sum()
                ),
                matched_operand_same_relative_delta=int(
                    (same_operand & (dx == dy)).sum()
                ),
                matched_operand_crossed_130_cycle_threshold=int(
                    (same_operand & ((lead_left >= 130) != (lead_right >= 130))).sum()
                ),
            )
        )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+")
    args = parser.parse_args()
    rows = [row for case in args.cases for row in compare(case)]
    save(HERE / "placements.csv", rows)
