"""Run the frozen complete-trace matrix, starting arms as full-source validation records become available."""

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from run import FULL, run

POLICIES = ("nopf", "stride", "multi", "spp")


def execute(case, policy):
    cert = FULL / "intake" / (case + ".validation.csv")
    with cert.open() as f:
        row = next(csv.DictReader(f))
    length = int(row["materialized_records"])
    if length < 250_000_000:
        raise ValueError(f"{case}: register a shortened span explicitly before running")
    origin = int(row["materialized_start"])
    prefix = FULL / "raw/traces" / (case + ".prefix_vpns.u64le") if origin else None
    return run(
        case,
        policy,
        [FULL / "raw/traces" / (case + ".champsimtrace")],
        [0],
        50_000_000,
        200_000_000,
        origin=origin,
        prefix_pages=prefix,
    )


def main(jobs, selected=None, policies=POLICIES):
    with (FULL / "intake/selection.csv").open() as f:
        cases = [r["case"] for r in csv.DictReader(f)]
    remaining = [(c, p) for c in (selected or cases) for p in policies]
    failures = []
    active = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        while remaining or active:
            for job in list(remaining):
                if len(active) == jobs:
                    break
                case, policy = job
                if not (FULL / "intake" / (case + ".validation.csv")).exists():
                    continue
                remaining.remove(job)
                future = pool.submit(execute, case, policy)
                active[future] = job
                print("starting", case, policy, flush=True)
            if active:
                done, _ = wait(active, timeout=2, return_when=FIRST_COMPLETED)
                for future in done:
                    job = active.pop(future)
                    try:
                        future.result()
                    except Exception as e:
                        failures.append((job, str(e)))
                        print("failed", job, str(e), flush=True)
            else:
                time.sleep(1)
    if failures:
        raise SystemExit(failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument(
        "--policies", nargs="+", choices=(*POLICIES, "temporal"), default=POLICIES
    )
    args = parser.parse_args()
    main(args.jobs, args.cases, args.policies)
