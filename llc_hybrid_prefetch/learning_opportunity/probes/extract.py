"""Read native LLC arrivals and retain reproducibly sampled anchor features and labels."""

import argparse
from contextlib import contextmanager
import csv
import json
import gzip
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
PREVIOUS = STUDY.parent / "full_workload"
RAW = STUDY / "raw/probes"
TARGET_COUNT = 30_000


@contextmanager
def publish_text(path):
    """Allow readers to see either the previous complete result or the new one."""
    with tempfile.TemporaryDirectory(dir=path.parent) as scratch:
        pending = Path(scratch) / path.name
        with pending.open("w", newline="") as f:
            yield f
        pending.replace(path)


def locate(case, policy):
    name = f"{case}__{policy}"
    for parent in (
        PREVIOUS / "raw/continuous/runs",
        STUDY / "raw/capture/runs",
        STUDY / "raw/expansion/runs",
    ):
        run = parent / name
        if (run / "complete.json").is_file() and (run / "arrival.u64le.gz").is_file():
            return run
    raise FileNotFoundError(f"Missing complete {policy} arrivals for {case}")


def extract(case, policy):
    RAW.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=RAW) as tmp:
        exe = Path(tmp) / "sample"
        subprocess.run(
            [
                "c++",
                "-O3",
                "-std=c++17",
                str(HERE / "sample.cc"),
                "-lz",
                "-o",
                str(exe),
            ],
            check=True,
        )
        run = locate(case, policy)
        with (run / "phases.csv").open() as f:
            phases = list(csv.DictReader(f))
        if {r["cpu"] for r in phases} != {"0"}:
            raise ValueError("Probe requires a single active core")
        count = sum(int(r["llc_ingress"]) for r in phases)
        block = max(1, count // TARGET_COUNT)
        path = Path(tmp) / "rows.tsv"
        with path.open("wb") as f:
            subprocess.run(
                [str(exe), str(run / "arrival.u64le.gz"), str(count), str(block)],
                stdout=f,
                check=True,
            )
        frame = pd.read_csv(path, sep="\t", dtype=np.int64)
        columns = {k: frame[k].to_numpy() for k in frame}
        state = run / "ingress_state.u64le.gz"
        if state.is_file():
            anchors = columns["index"]
            values = np.zeros(len(anchors), np.uint64)
            pos = 0
            with gzip.open(state, "rb") as stream:
                while block_bytes := stream.read(16 * 65536):
                    fields = np.frombuffer(block_bytes, "<u8").reshape(-1, 2)
                    lo, hi = np.searchsorted(anchors, [pos, pos + len(fields)])
                    values[lo:hi] = fields[anchors[lo:hi] - pos, 1]
                    pos += len(fields)
            if pos != count:
                raise ValueError("State/arrival row count differs")
            columns["system_state"] = values
        dest = RAW / f"{case}__{policy}.npz"
        temp = Path(tmp) / "export.npz"
        np.savez_compressed(temp, total_arrivals=[count], **columns)
        temp.replace(dest)
        meta = dict(
            case=case,
            policy=policy,
            arrival_count=count,
            anchor_count=len(frame),
            sample_block=block,
            raw_directory=run.relative_to(STUDY.parent).as_posix(),
        )
        (RAW / (dest.stem + ".json")).write_text(json.dumps(meta, indent=2) + "\n")
        print(case, count, len(frame), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+")
    parser.add_argument("--policy", default="stride")
    args = parser.parse_args()
    for case in args.cases:
        extract(case, args.policy)
