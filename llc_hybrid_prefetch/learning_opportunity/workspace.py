"""Compare reconstructed datasets across cache workspaces using native counters and ZIP member checks."""

import argparse
import csv
import json
from pathlib import Path
import struct
from zipfile import ZipFile
import zlib
import time

HERE = Path(__file__).resolve().parent
CACHE = HERE.parents[1]


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def fingerprint(path, acc=0):
    with path.open("rb") as f:
        while data := f.read(1 << 20):
            acc = zlib.crc32(data, acc)
    return acc


def inspect(row):
    run = CACHE / row["raw_run"]
    ix = (CACHE / row["aligned_index"]).parent
    if not ix.is_dir() or not (run / "complete.json").is_file():
        raise FileNotFoundError(f"{row['case']} {row['policy']} is incomplete")
    idx = json.loads((ix / "index.json").read_text())
    stats = table(run / "cores.csv")[0]
    source = CACHE / row["source_archive"]
    view = CACHE / row["native_input"]
    if not source.is_file() or not view.is_file():
        raise FileNotFoundError(f"Source or native view absent for {row['case']}")
    paths = sorted(ix.glob("events_*.npz")) + sorted(ix.glob("outcomes_*.npz"))
    if len(paths) != 2 * len(idx["chunks"]):
        raise ValueError(f"Incomplete label chunks for {ix.name}")
    words = 0
    for p in paths:
        with ZipFile(p) as archive:
            for obj in sorted(archive.infolist(), key=lambda x: x.filename):
                words = zlib.crc32(
                    obj.filename.encode() + struct.pack("<IQ", obj.CRC, obj.file_size),
                    words,
                )
    posts = 0
    for p in sorted(ix.glob("*.npy")):
        posts = fingerprint(p, zlib.crc32(p.name.encode(), posts))
    with (run / "outcomes_core0.u16le.gz").open("rb") as f:
        f.seek(-8, 2)
        mask_check, mask_bytes = struct.unpack("<II", f.read(8))
    assert mask_bytes == 2 * int(stats["instructions"])
    rec = dict(
        case=row["case"],
        policy=row["policy"],
        measured=int(stats["instructions"]),
        cycles=int(stats["cycles"]),
        arrivals=idx["observations"],
        label_chunks=len(paths),
        native_input_bytes=view.stat().st_size,
        source_archive_bytes=source.stat().st_size,
        mask_check=mask_check,
        event_column_check=words,
        postings_check=posts,
    )
    return rec


def inventory(wait=False):
    registry = table(HERE / "data_registry.csv")
    complete = {}
    while len(complete) < len(registry):
        for row in registry:
            key = row["case"], row["policy"]
            if key in complete:
                continue
            try:
                complete[key] = inspect(row)
            except FileNotFoundError:
                if not wait:
                    raise
                continue
            if wait:
                print("Verified", *key, len(complete), "of", len(registry), flush=True)
        if len(complete) < len(registry):
            time.sleep(30)
    return [complete[row["case"], row["policy"]] for row in registry]


def save(path, data):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=data[0])
        w.writeheader()
        w.writerows(data)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--compare", type=Path)
    cli.add_argument(
        "--wait",
        action="store_true",
        help="Check each immutable dataset as it becomes available",
    )
    args = cli.parse_args()
    local = inventory(wait=args.wait)
    if args.compare:
        other = {(r["case"], r["policy"]): r for r in table(args.compare)}
        if len(local) != len(other):
            raise ValueError("Different dataset counts")
        for r in local:
            each = other[r["case"], r["policy"]]
            for k, v in r.items():
                if str(v) != each[k]:
                    raise ValueError(f"{r['case']}/{r['policy']}: {k} differs")
            for k in ("mask_check", "event_column_check", "postings_check"):
                r.pop(k)
            r["fixed_work_masks_event_columns_postings"] = "match"
    save(args.out, local)


if __name__ == "__main__":
    main()
