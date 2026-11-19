"""Run one matched native arm. All paths are resolved against the existing project."""

import argparse
import csv
import fcntl
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FULL = HERE.parent
BASE = FULL.parent
CACHE = BASE.parent
RAW = FULL / "raw/native"
BIN = RAW / "build/pythia/bin/champsim"
POLICIES = ("nopf", "stride", "multi", "spp", "look2", "look4", "temporal", "xpc")


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def run(
    case,
    policy,
    traces,
    slots,
    warmup,
    roi,
    skip=0,
    sample=250000,
    origin=0,
    prefix_pages=None,
):
    if not case.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Need a simple case name")
    if (
        len(traces) not in (1, 2)
        or len(slots) != len(traces)
        or len(set(slots)) != len(traces)
        or not set(slots) <= {0, 1}
    ):
        raise ValueError("One distinct page slot per active core required")
    if warmup < 1 or roi < 4 or min(skip, sample, origin) < 0:
        raise ValueError("Invalid instruction span")
    paths = [Path(p).resolve() for p in traces]
    for p in paths:
        if not p.is_relative_to(CACHE) or p.suffix != ".champsimtrace":
            raise ValueError(f"Need a validated native-format trace inside cache: {p}")
        if p.stat().st_size % 64 or p.stat().st_size // 64 < skip + warmup + roi:
            raise ValueError(f"Insufficient or misaligned fixed 64-byte records: {p}")
    config = dict(
        case=case,
        policy=policy,
        traces=[str(p.relative_to(CACHE)) for p in paths],
        slots=slots,
        warmup=warmup,
        roi=roi,
        skip=skip,
        sample_width=sample,
    )
    if origin:
        if not prefix_pages or not Path(prefix_pages).is_file():
            raise ValueError(
                "A far source view requires the first-touch VPN seed from the skipped prefix"
            )
        config.update(
            source_ordinal_base=origin,
            prefix_pages=str(Path(prefix_pages).resolve().relative_to(CACHE)),
        )
    folder = RAW / "runs" / (case + "__" + policy)
    folder.mkdir(exist_ok=True, parents=True)
    with (folder / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        complete = folder / "complete.json"
        if complete.exists():
            result = json.loads(complete.read_text())
            if result["config"] != config:
                raise ValueError(
                    f"Immutable completed run uses another configuration: {folder}"
                )
            return folder
        settings = folder / "config.json"
        if settings.exists() and json.loads(settings.read_text()) != config:
            raise ValueError(
                f"Existing incomplete run uses another configuration: {folder}"
            )
        settings.write_text(json.dumps(config, indent=2) + "\n")
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("STUDY_", "FULL_", "HYBRID_", "LLC_PF_"))
        }
        controls = dict(
            LLC_PF_POLICY="nopf" if policy == "nopf" else "stride",
            STUDY_POLICY="stride" if policy == "nopf" else policy,
            HYBRID_ACTIVE_CORES=str(len(paths)),
            HYBRID_DETAIL="1",
            HYBRID_COMPACT="1",
            HYBRID_COMPRESS="1",
            HYBRID_OUT=str(folder),
            HYBRID_PAGE_SLOTS=",".join(map(str, slots)),
            HYBRID_SKIPS=",".join([str(skip)] * len(paths)),
            FULL_SAMPLE_WIDTH=str(sample),
        )
        if origin:
            controls.update(
                FULL_SOURCE_ORIGIN=str(origin),
                FULL_PREFIX_PAGES=str(Path(prefix_pages).resolve()),
            )
        cmd = [
            str(BIN),
            f"--warmup_instructions={warmup}",
            f"--simulation_instructions={roi}",
            "-traces",
            *map(str, paths),
        ]
        start = time.monotonic()
        with (folder / "run.log").open("w") as log:
            subprocess.run(
                cmd,
                env={**env, **controls},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        cores = rows(folder / "cores.csv")
        phases = rows(folder / "phases.csv")
        stats = {r["key"]: int(r["value"]) for r in rows(folder / "stats.csv")}
        if len(cores) != len(paths) or len(phases) != 4 * len(paths):
            raise ValueError("Missing native core or quarter")
        for c, data in enumerate(cores):
            qs = [r for r in phases if int(r["cpu"]) == c]
            if (int(data["cpu"]), int(data["warmup"]), int(data["instructions"])) != (
                c,
                warmup,
                roi,
            ):
                raise ValueError("Fixed-work retirement mismatch")
            if sum(int(r["cycles"]) for r in qs) != int(data["cycles"]):
                raise ValueError("Quarter cycles do not add up to retirement time")
        for stage in ("timely", "late"):
            if sum(int(r[stage]) for r in phases) != stats.get("study_pf_" + stage, 0):
                raise ValueError("First-use attribution mismatch")
        if (
            sum(
                stats.get("study_pf_" + s, 0)
                for s in ("timely", "late", "evicted", "unresolved", "redundant")
            )
            != stats["prefetch_accepted"]
        ):
            raise ValueError(
                "Accepted requests do not partition into lifecycle outcomes"
            )
        result = dict(
            config=config,
            wall_seconds=round(time.monotonic() - start, 2),
            command=shlex.join(cmd),
            environment=controls,
            core_results=cores,
        )
        with (folder / "complete.tmp").open("w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        (folder / "complete.tmp").replace(complete)
        print("completed", case, policy, result["wall_seconds"], flush=True)
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--policy", required=True, choices=POLICIES)
    parser.add_argument("--traces", nargs="+", type=Path)
    parser.add_argument("--slots", nargs="+", type=int)
    parser.add_argument("--warmup", type=int, default=50_000_000)
    parser.add_argument("--roi", type=int, default=200_000_000)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--sample", type=int, default=250000)
    parser.add_argument(
        "--origin",
        type=int,
        default=0,
        help="source ordinal offset before a far materialization",
    )
    parser.add_argument(
        "--prefix-pages", type=Path, help="ordered source-prefix VPN seed for --origin"
    )
    args = parser.parse_args()
    traces = args.traces or [FULL / "raw/traces" / (args.case + ".champsimtrace")]
    run(
        args.case,
        args.policy,
        traces,
        args.slots or list(range(len(traces))),
        args.warmup,
        args.roi,
        args.skip,
        args.sample,
        args.origin,
        args.prefix_pages,
    )
