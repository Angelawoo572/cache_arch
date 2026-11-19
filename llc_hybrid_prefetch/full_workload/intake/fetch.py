"""Fetch complete frozen objects. Prefixes are resumable but never counted as acquired."""
import argparse
import csv
import fcntl
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw" / "traces"

def fetch(row):
    name = row["manifest_path"].split("/")[-1]
    destination = RAW / name
    with (RAW / (name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        fetch_object(row, destination)


def fetch_object(row, destination):
    name = destination.name
    size = int(row["compressed_bytes"])
    if destination.exists():
        if destination.stat().st_size != size:
            raise ValueError(f"{name}: existing file size differs from published size")
        return
    part = RAW / (name + ".part")
    if not part.exists() or part.stat().st_size < size:
        subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                        "--retry", "5", "--connect-timeout", "25", "--continue-at", "-",
                        "--output", str(part), row["url"]], check=True)
    if part.stat().st_size != size:
        raise ValueError(f"{name}: {part.stat().st_size} of {size} bytes; incomplete download")
    part.replace(destination)
    print(row["case"], size, "compressed bytes complete", flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*")
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    with (HERE / "selection.csv").open() as f:
        rows = list(csv.DictReader(f))
    if args.cases:
        unknown = set(args.cases) - {r["case"] for r in rows}
        if unknown:
            parser.error(f"unknown cases {sorted(unknown)}")
        rows = [r for r in rows if r["case"] in args.cases]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(fetch, rows))

if __name__ == "__main__":
    main()
