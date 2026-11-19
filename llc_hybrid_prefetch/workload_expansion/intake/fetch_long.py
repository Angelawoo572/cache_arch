"""Extend all original Google discovery inputs for a disjoint deeper interval."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import json

from fetch import CHUNK, RAW, TARGET, remote_prefix
from register import HERE, STUDY, write_csv

LONG = RAW / "long"
RECORDS = 5_001_024


def verify_prefix(long_file, base_file):
    remaining = TARGET
    with long_file.open("rb") as a, base_file.open("rb") as b:
        while remaining:
            n = min(CHUNK, remaining)
            block = b.read(n)
            if len(block) != n or a.read(n) != block:
                raise ValueError("The long input disagrees with its registered short prefix")
            remaining -= n


def get(row):
    assert (row["collection"], row["split"]) == ("gtrace_v2", "discovery")
    short = STUDY / row["prefix_path"]
    dest = LONG / short.name
    meta = dest.with_suffix(".json")
    with dest.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if dest.exists():
            saved = json.loads(meta.read_text())
            assert dest.stat().st_size == RECORDS * 64
            assert saved["source_url"] == row["source_url"] and saved["records"] == RECORDS
            verify_prefix(dest, short)
            return saved
        tmp = dest.with_suffix(".partial")
        received = remote_prefix(row, tmp, target=RECORDS * 64, scratch=LONG)
        assert tmp.stat().st_size == RECORDS * 64
        verify_prefix(tmp, short)
        result = dict(case=row["case"], collection=row["collection"], source_url=row["source_url"],
                      prefix_path=str(dest.relative_to(STUDY)), records=RECORDS, record_bytes=64,
                      compressed_bytes_received=received, identical_short_prefix_records=TARGET//64)
        meta.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(dest)
        print(row["case"], received, flush=True)
        return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args()
    with (HERE / "registry.csv").open() as f:
        inputs = [r for r in csv.DictReader(f) if r["collection"] == "gtrace_v2" and r["split"] == "discovery"]
    assert len(inputs) == 7
    LONG.mkdir(exist_ok=True, parents=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(get, inputs))
    write_csv("long_acquisition.csv", rows)
