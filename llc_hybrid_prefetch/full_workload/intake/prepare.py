"""Validate a complete compressed input through EOF; write only the requested native interval."""
import argparse
import csv
import fcntl
import io
import subprocess
import tempfile
from fetch import HERE, RAW

COUNTERS = ["source_records", "materialized_records", "branches", "taken", "load_instructions",
            *(f"load_slot_{i}" for i in range(4)), *(f"store_slot_{i}" for i in range(2)),
            *(f"source_reg_slot_{i}" for i in range(4)), *(f"dest_reg_slot_{i}" for i in range(2))]

def prepare(row, view):
    start, count = int(view["start"]), int(view["count"])
    name = row["manifest_path"].split("/")[-1]
    source = RAW / name
    if source.stat().st_size != int(row["compressed_bytes"]):
        raise ValueError(f"incomplete object {source}")
    stem = view["view"]
    output = RAW / f"{stem}.champsimtrace"
    partial = output.with_suffix(".building")
    prefix = RAW / f"{stem}.prefix_vpns.u64le"
    seed = prefix.with_suffix(".building")
    validator = [str(RAW / "verify_records"), str(partial), str(start), str(count)]
    if start:
        validator.append(str(seed))
    decoder = ["gzip", "-dc"] if name.endswith(".gz") else ["xz", "-dc"]
    with tempfile.TemporaryFile(dir=RAW) as messages:
        with subprocess.Popen([*decoder, str(source)], stdout=subprocess.PIPE, stderr=messages) as stream:
            try:
                result = subprocess.run(validator, stdin=stream.stdout, capture_output=True, text=True)
            finally:
                stream.stdout.close()
            decode_status = stream.wait()
        if decode_status or result.returncode:
            messages.seek(0)
            diagnostic = messages.read(4096).decode(errors="replace")
            raise RuntimeError(f"{name}: decode={decode_status} audit={result.returncode}: {diagnostic}{result.stderr}")
    values = [int(x) for x in next(csv.reader(io.StringIO(result.stdout)))]
    if len(values) != len(COUNTERS):
        raise ValueError("unexpected validation counters")
    record = dict(case=row["case"], compressed_bytes=source.stat().st_size, materialized_start=start,
                  **dict(zip(COUNTERS, values)))
    assert partial.stat().st_size == 64 * record["materialized_records"]
    if record["materialized_records"] != count:
        raise ValueError(f"{stem}: full source has too few records for the frozen view")
    assert record["load_instructions"] and record["source_reg_slot_0"] and record["dest_reg_slot_0"]
    if start:
        assert seed.stat().st_size % 8 == 0
        record["prefix_pages"] = seed.stat().st_size // 8
        seed.replace(prefix)
    partial.replace(output)
    validation = HERE / f"{stem}.validation.csv"
    temp_summary = validation.with_suffix(".tmp")
    with temp_summary.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=record)
        writer.writeheader()
        writer.writerow(record)
    temp_summary.replace(validation)
    print(f"{stem}: {values[0]:,} full records verified; {values[1]:,} raw records available", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("view")
    args = parser.parse_args()
    with (HERE / "selection.csv").open() as f:
        sources = {r["case"]: r for r in csv.DictReader(f)}
    with (HERE / "views.csv").open() as f:
        views = {r["view"]: r for r in csv.DictReader(f)}
    view = views[args.view]
    with (RAW / (args.view + ".prepare.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prepare(sources[view["case"]], view)
