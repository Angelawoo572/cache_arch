"""Exercise a known three-operand reordering, late/timely attribution and independent island boundaries."""

import csv
import gzip
import struct
import subprocess

from report import compile_analysis
from run_arrivals import RAW


def main():
    compile_analysis()
    out = RAW / "fixture"
    out.mkdir(exist_ok=True)
    ids = [1, 3, 2, 4, 5, 6, 7, 8]
    baseline = [2] * 8
    policy = [9, 6, 9, 2, 0, 1, 1, 2]
    with gzip.open(out / "arrival.gz", "wb") as f:
        for rank, ordinal in enumerate(ids):
            f.write(
                struct.pack(
                    "<QQQQ",
                    ordinal << 2,
                    7 if ordinal < 5 else 9,
                    600 + ordinal,
                    100 + rank * 10,
                )
            )
    for name, x in (("nopf", baseline), ("stride", policy)):
        (out / (name + ".gz")).write_bytes(gzip.compress(struct.pack("<8H", *x)))
    subprocess.run(
        [
            str(RAW / "analyze"),
            "fixture",
            str(out / "arrival.gz"),
            str(out / "nopf.gz"),
            str(out / "stride.gz"),
            "1",
            "8",
            str(out),
        ],
        check=True,
    )
    with (out / "counts.csv").open() as f:
        data = list(csv.DictReader(f))

    def group(order, scope="global", window="full"):
        return next(
            r
            for r in data
            if r["order"] == order
            and r["scope"] == scope
            and r["window"] == window
            and r["cohort"] == "nopf_miss"
        )

    a, b = group("arrival"), group("program")
    if (
        a["next_direct"],
        a["next_late"],
        a["next_plain_miss"],
        a["adjacent_idop_reversals"],
    ) != ("1", "1", "0", "1"):
        raise ValueError(a)
    if (
        b["next_direct"],
        b["next_late"],
        b["next_plain_miss"],
        b["adjacent_clock_reversals"],
    ) != ("0", "1", "1", "1"):
        raise ValueError(b)
    if group("arrival", "pc")["events_in_never_timely_groups"] != "4":
        raise ValueError("lost never-timely stream")
    if group("arrival", window="four_islands")["group_instances"] != "4":
        raise ValueError("sample edges joined")
    print(
        "Verified arrival/retirement inversion, direct-to-late and direct-to-plain adjacency, never-timely PC and separate sample edges"
    )


if __name__ == "__main__":
    main()
