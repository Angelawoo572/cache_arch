"""Copy an already certified native view from its complete source archive on a second host."""

import argparse
import csv
import fcntl
from pathlib import Path
import subprocess
import zlib
from run import TRACES, fetch

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
GROUPS = ("expansion", "domains")
BUFF = 4 << 20


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def stream_check(path):
    n = 0
    check = 0
    with path.open("rb") as f:
        while block := f.read(BUFF):
            n += len(block)
            check = zlib.crc32(block, check)
    return n, check


def load(case):
    for group in GROUPS:
        for r in read(STUDY / group / "sources.csv"):
            if r["case"] == case:
                return group, r
    raise ValueError("Unknown source")


def certify(case):
    group, r = load(case)
    certified = read(STUDY / group / "validation" / (case + ".validation.csv"))[0]
    archive_size, archive_check = stream_check(TRACES / r["source"])
    view_size, view_check = stream_check(TRACES / (case + ".champsimtrace"))
    if archive_size != int(certified["compressed_bytes"]) or view_size != 64 * int(
        certified["materialized_records"]
    ):
        raise ValueError("Source length differs from EOF validation")
    return dict(
        case=case,
        archive_size=archive_size,
        archive_check=archive_check,
        view_size=view_size,
        view_check=view_check,
    )


def replicate(cert):
    case = cert["case"]
    _, r = load(case)
    archive = TRACES / r["source"]
    view = TRACES / (case + ".champsimtrace")
    fetch.RAW = TRACES
    source = dict(
        case=case, manifest_path=r["source"], compressed_bytes=r["bytes"], url=r["url"]
    )
    for attempt in range(4):
        try:
            fetch.fetch(source)
            break
        except subprocess.CalledProcessError as error:
            if error.returncode not in (18, 28, 52, 55, 56, 92) or attempt == 3:
                raise
    if stream_check(archive) != (int(cert["archive_size"]), int(cert["archive_check"])):
        raise ValueError(f"Different complete compressed object for {case}")
    expected = (int(cert["view_size"]), int(cert["view_check"]))
    if view.exists() and stream_check(view) == expected:
        return
    partial = view.with_suffix(".replica")
    cmd = ["gzip" if archive.suffix == ".gz" else "xz", "-dc", str(archive)]
    check = 0
    n = 0
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    ) as decoder:
        try:
            with partial.open("wb") as f:
                while n < expected[0]:
                    block = decoder.stdout.read(min(BUFF, expected[0] - n))
                    if not block:
                        raise ValueError(f"Premature source EOF for {case}")
                    f.write(block)
                    check = zlib.crc32(block, check)
                    n += len(block)
        finally:
            decoder.stdout.close()
            decoder.wait()
    if (n, check) != expected:
        raise ValueError(f"Decoded native view differs for {case}")
    partial.replace(view)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certify", nargs="+")
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()
    if args.certify:
        records = [certify(case) for case in args.certify]
        with args.table.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=records[0])
            w.writeheader()
            w.writerows(records)
    else:
        for cert in read(args.table):
            with (TRACES / (cert["case"] + ".prepare.lock")).open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                replicate(cert)
            print("Verified complete archive and native view", cert["case"], flush=True)


if __name__ == "__main__":
    main()
