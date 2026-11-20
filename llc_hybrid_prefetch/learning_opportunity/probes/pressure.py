"""Align captured ingress pressure to sampled raw arrivals."""

import argparse
import gzip

import numpy as np

from extract import RAW, STUDY


def prepare(case):
    folder = STUDY / "raw/capture/runs" / f"{case}__stride"
    if not (folder / "complete.json").is_file():
        raise ValueError("Incomplete rich capture")
    with np.load(RAW / f"{case}__stride.npz") as data:
        anchors = data["index"]
        values = np.column_stack([data[k] for k in ("key", "pc", "line", "cycle")])
        count = int(data["total_arrivals"][0])
    state = np.zeros(len(anchors), dtype=np.int64)
    pos = 0
    with (
        gzip.open(folder / "ingress_state.u64le.gz", "rb") as sf,
        gzip.open(folder / "arrival.u64le.gz", "rb") as af,
    ):
        while chunk := sf.read(16 * 65536):
            arr = np.frombuffer(chunk, "<u8").reshape(-1, 2)
            observed = np.frombuffer(af.read(len(arr) * 32), "<u8").reshape(-1, 4)
            lo, hi = np.searchsorted(anchors, [pos, pos + len(arr)])
            offsets = anchors[lo:hi] - pos
            if not np.array_equal(values[lo:hi], observed[offsets]):
                raise ValueError("Sampled pressure has a different native anchor")
            state[lo:hi] = arr[offsets, 1]
            pos += len(arr)
        if af.read(1) or pos != count:
            raise ValueError("Full state and arrival counts differ")
    np.savez_compressed(
        RAW / f"{case}__pressure.npz", key=values[:, 0], system_state=state
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+")
    for case in parser.parse_args().cases:
        prepare(case)
