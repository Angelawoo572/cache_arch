"""Validate complete source recordings, then collect aligned continuous conventional runs."""

import argparse
import csv
from subprocess import CalledProcessError, run
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import fcntl
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
FULL = BASE / "full_workload"
RAW = HERE.parent / "raw/expansion"
TRACES = RAW / "traces"
VALID = HERE / "validation"
sys.path.insert(0, str(FULL / "intake"))
import fetch
import prepare


def native_module():
    spec = importlib.util.spec_from_file_location("native_run", FULL / "native/run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BIN = FULL / "raw/continuous/build/pythia/bin/champsim"
    module.RAW = RAW
    return module


def acquire(row):
    with (TRACES / (row["case"] + ".prepare.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return prepare_recording(row)


def prepare_recording(row):
    case = row["case"]
    archive = TRACES / row["source"]
    prepared = VALID / (case + ".validation.csv")
    view = TRACES / (case + ".champsimtrace")
    if (
        prepared.exists()
        and view.is_file()
        and archive.is_file()
        and view.stat().st_size == 64 * 250001024
        and archive.stat().st_size == int(row["bytes"])
    ):
        return case
    source = dict(
        case=case,
        manifest_path=row["source"],
        compressed_bytes=row["bytes"],
        url=row["url"],
    )
    cached = BASE.parent / "traces" / archive.name
    if (
        cached.is_file()
        and cached.stat().st_size == int(row["bytes"])
        and not archive.exists()
    ):
        try:
            archive.hardlink_to(cached)
        except OSError:
            shutil.copyfile(cached, archive)
    for attempt in range(4):
        try:
            fetch.fetch(source)
            break
        except CalledProcessError as error:
            if error.returncode not in (18, 28, 52, 55, 56, 92) or attempt == 3:
                raise
            print(f"{case}: resuming interrupted download", flush=True)
    prepare.prepare(source, dict(view=case, start=0, count=250001024))
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--decoders", type=int, default=2)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate archives and materialize native views without replay",
    )
    args = parser.parse_args()
    rows = list(csv.DictReader((HERE / "sources.csv").open()))
    known = {r["case"] for r in rows}
    if args.cases and not set(args.cases) <= known:
        parser.error("unknown case")
    rows = [r for r in rows if not args.cases or r["case"] in args.cases]
    TRACES.mkdir(exist_ok=True, parents=True)
    VALID.mkdir(exist_ok=True)
    fetch.RAW = prepare.RAW = TRACES
    prepare.HERE = VALID
    validator = TRACES / "verify_records"
    code = FULL / "intake/verify_records.cc"
    if not validator.is_file() or validator.stat().st_mtime < code.stat().st_mtime:
        candidate = validator.with_suffix(f".{os.getpid()}.building")
        run(["c++", "-std=c++17", "-O3", str(code), "-o", str(candidate)], check=True)
        candidate.replace(validator)
    native = native_module()
    if not args.prepare_only and not native.BIN.is_file():
        run([sys.executable, "-B", str(FULL / "continuous/build.py")], check=True)
    with (
        ThreadPoolExecutor(args.decoders) as decoders,
        ThreadPoolExecutor(args.jobs) as sims,
    ):
        inputs = {decoders.submit(acquire, row): row["case"] for row in rows}
        jobs = []
        errors = []
        for future in as_completed(inputs):
            try:
                case = future.result()
            except (CalledProcessError, OSError, ValueError) as error:
                errors.append((inputs[future], str(error)))
                continue
            if args.prepare_only:
                continue
            slots = (0, 1) if case in ("gcc734", "cc5") else (0,)
            for slot in slots:
                label = case if slot == 0 else case + "_slot1"
                for arm in ("stride", "nopf", "multi", "spp"):
                    job = sims.submit(
                        native.run,
                        label,
                        arm,
                        [TRACES / (case + ".champsimtrace")],
                        [slot],
                        50_000_000,
                        200_000_000,
                    )
                    jobs.append(job)
        for job in as_completed(jobs):
            print("completed", job.result().name, flush=True)
        if errors:
            raise RuntimeError(f"Trace preparation failed: {errors}")


if __name__ == "__main__":
    main()
