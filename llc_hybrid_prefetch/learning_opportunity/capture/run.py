"""Replay registered 50M-warmup/200M-measurement policy arms with the continuous writer."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import sys

from build import FULL, RAW, BUILD, STUDY

sys.path.insert(0, str(FULL / "native"))
import importlib.util

spec = importlib.util.spec_from_file_location("baseline_runner", FULL / "native/run.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)

CASES = (
    "mcf17_discovery",
    "bfs_discovery",
    "soplex06_heldout",
    "pr_heldout",
    "bfs_transfer",
    "mcf17_transfer",
    "mcf06_discovery",
    "bfs_discovery_far",
    "mcf17_discovery_far",
)
POLICIES = ("nopf", "stride", "multi", "spp")
EXPANSION = STUDY / "raw/expansion"


def external_cases():
    with (STUDY / "expansion/sources.csv").open(newline="") as file:
        return tuple(r["case"] for r in csv.DictReader(file))


def external(case):
    view = EXPANSION / "traces" / (case + ".champsimtrace")
    validation = STUDY / "expansion/validation" / (case + ".validation.csv")
    if not validation.is_file() or view.stat().st_size != 64 * 250001024:
        raise ValueError(
            f"Prepare and validate the complete external recording first: {case}"
        )
    baseline.RAW = RAW
    baseline.BIN = BUILD / "bin/champsim"
    return baseline.run(case, "stride", [view], [0], 50000000, 200000000, sample=250000)


def run(job):
    case, policy, fixture = job
    inputs = FULL / "raw/traces"
    baseline.RAW = RAW
    baseline.BIN = BUILD / "bin/champsim"
    seed = inputs / (case + ".prefix_vpns.u64le")
    origin = {"bfs_discovery_far": 250000001, "mcf17_discovery_far": 1750000000}.get(
        case, 0
    )
    return baseline.run(
        case + "_fixture" if fixture else case,
        policy,
        [inputs / (case + ".champsimtrace")],
        [0],
        100000 if fixture else 50000000,
        250000 if fixture else 200000000,
        sample=250000,
        origin=origin,
        prefix_pages=seed if origin else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument(
        "--external",
        action="store_true",
        help="Capture Stride for each long external recording at placement slot zero",
    )
    parser.add_argument("--policies", choices=POLICIES, nargs="+")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    # Schedule expensive PageRank immediately and distribute early complete arms across programs.
    available = external_cases() if args.external else CASES
    cases = args.cases or list(available)
    if not set(cases) <= set(available):
        parser.error("Case outside the selected primary or external registry")
    with ThreadPoolExecutor(args.jobs) as pool:
        if args.external:
            if args.fixture or args.policies not in (None, ["stride"]):
                parser.error(
                    "External continuous replay uses the registered full Stride arm"
                )
            list(pool.map(external, cases))
        else:
            jobs = product(args.policies or POLICIES, cases)
            list(pool.map(run, ((c, p, args.fixture) for p, c in jobs)))
