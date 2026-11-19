"""Regenerate the complete discovery artifacts once every registered simulator arm has retired."""

import argparse
import os
import subprocess
import sys
from replay import CASES, POLICIES
from streams import location
from pattern import HERE


def missing_runs():
    cases = (*CASES, "pr_heldout_entire")
    pending = []
    for scope, pairs in (
        ("diagnostic", ((c, p) for c in CASES for p in POLICIES)),
        (
            "full",
            (
                (c, p)
                for c in (*cases, "bfs_discovery_far", "mcf17_discovery_far")
                for p in ("nopf", "stride", "multi", "spp")
            ),
        ),
        ("full", ((c, p) for c in cases for p in ("shape", "temporal"))),
        ("full", [("bfs_discovery", "sparse3")]),
    ):
        for c, p in pairs:
            if not (location(c, p, scope) / "complete.json").is_file():
                pending.append(f"{scope}/{c}/{p}")
    return pending


def run(script, *args):
    subprocess.run(
        [sys.executable, str(HERE / script), *args],
        cwd=HERE,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="list pending simulator arms without rebuilding",
    )
    args = parser.parse_args()
    pending = missing_runs()
    if pending:
        print("\n".join(pending))
        raise SystemExit(1)
    if args.check:
        raise SystemExit(0)
    for scope in ("source", "diagnostic", "full"):
        run("characterize.py", "--scope", scope)
    for scope in ("diagnostic", "full"):
        run("failures.py", "--scope", scope)
        run("opportunities.py", "--scope", scope)
        run("score_replay.py", "--scope", scope)
        run("alternatives.py", "--scope", scope)
    for script in (
        "phase.py",
        "ambiguity.py",
        "joint_targets.py",
        "whole_union.py",
        "registry.py",
        "fixed3.py",
        "report.py",
        "verify.py",
    ):
        run(script)
