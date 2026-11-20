"""Check indexed paths on a cache host and emit a compact availability report."""

import argparse
import csv
from pathlib import Path
import sys

from inventory import DATA, read_csv


def check(root, relative, expected_bytes=None):
    path = root / relative
    if not path.is_file():
        return "missing"
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        return "size mismatch"
    return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--sacramento", action="store_true")
    args = parser.parse_args()

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("kind", "key", "folder_or_trace", "files_checked", "exceptions"))
    for row in read_csv(DATA / "trace_files.csv"):
        items = {
            row["compressed_archive"]: int(row["compressed_bytes"]),
            row["native_trace"]: 64 * int(row["native_records"]),
        }
        if row["physical_placement_seed"]:
            items[row["physical_placement_seed"]] = None
        errors = []
        for path, size in items.items():
            if error := check(args.cache, path, size):
                errors.append(f"{path}: {error}")
        writer.writerow(
            ("trace", row["view"], row["native_trace"], len(items), "|".join(errors))
        )

    for row in read_csv(DATA / "run_files.csv"):
        folder = Path(row["raw_directory"])
        if args.sacramento and row["group"] == "independent Linux copy":
            folder = folder.parent.parent / "native/runs" / folder.name
        names = row["artifacts"].split("|")
        errors = []
        for name in names:
            if error := check(args.cache, folder / name):
                errors.append(f"{name}: {error}")
        key = f'{row["group"]}/{row["case"]}/{row["policy"]}'
        writer.writerow(("run", key, folder, len(names), "|".join(errors)))


if __name__ == "__main__":
    main()
