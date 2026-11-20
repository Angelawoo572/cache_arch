"""Compare every logical column from RAM-backed and registered disk-backed indexes."""

import argparse
import csv
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

from build import ARMS, ReferenceCache, build, read_into
from sequence import ROOT


def compare(reference, product):
    names = sorted(p.name for p in reference.iterdir())
    assert names == sorted(p.name for p in product.iterdir())
    columns = 0
    for name in names:
        left, right = reference / name, product / name
        if name.endswith(".npy"):
            a, b = (np.load(f, mmap_mode="r") for f in (left, right))
            assert a.dtype == b.dtype
            np.testing.assert_array_equal(a, b)
            columns += 1
        elif name.endswith(".npz"):
            with np.load(left) as a, np.load(right) as b:
                assert a.files == b.files
                for key in a.files:
                    assert a[key].dtype == b[key].dtype
                    np.testing.assert_array_equal(a[key], b[key])
                    columns += 1
        else:
            assert json.loads(left.read_text()) == json.loads(right.read_text())
    return columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    parser.add_argument("--mask-cache-mib", type=int, default=2048)
    args = parser.parse_args()
    cache = ReferenceCache(args.mask_cache_mib)
    assert ReferenceCache(0).begin(["a"], 1) is None
    with np.testing.assert_raises(ValueError):
        read_into(io.BytesIO(b"10"), np.empty(3, dtype="u1"))
    rows = []
    for case in args.cases:
        for arm in args.arms:
            with TemporaryDirectory(prefix="parity-", dir=ROOT) as tmp:
                path = build(case, arm, memory=True, mask_cache=cache, output=Path(tmp))
                meta = json.loads((path / "index.json").read_text())
                rows.append(
                    dict(
                        case=case,
                        policy=arm,
                        arrivals=meta["observations"],
                        compared_arrays=compare(ROOT / path.name, path),
                        cache_bound_mib=args.mask_cache_mib,
                        retained_mask_bytes=sum(
                            a.nbytes for a in cache.arrays.values()
                        ),
                    )
                )
    writer = csv.DictWriter(sys.stdout, fieldnames=rows[0], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
