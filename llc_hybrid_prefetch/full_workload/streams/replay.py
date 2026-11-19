"""Run sparse, independent arithmetic-address alternatives in four 1M windows after 50M processor warmup."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
FULL = HERE.parent
sys.path.insert(0, str(FULL / "native"))
import run as native

RAW = FULL / "raw/arithmetic"
BASE = FULL / "raw/native/build/pythia"
TREE = RAW / "build/pythia"
CASES = ("bfs_discovery", "bfs_transfer", "pr_heldout", "soplex06_heldout")
NATIVE = ("nopf", "stride", "spp", "multi")
AUX = ("local", "shared_increment", "other_continuation", "plus1", "plus3")


def build():
    if not TREE.exists():
        shutil.copytree(
            BASE, TREE, symlinks=True, ignore=shutil.ignore_patterns(".git")
        )
    source = (BASE / "src/controls.inc").read_text()
    insert = "void study_init() {"
    call = '    if (log) {\n        ++diagnostics["anchors"];'
    pick = 'variant = env_or("STUDY_POLICY", "stride");'
    for marker in (insert, call, pick):
        if source.count(marker) != 1:
            raise ValueError(f"Unexpected native control source near {marker}")
    source = source.replace(
        insert, (HERE / "arithmetic.inc").read_text() + "\n" + insert
    )
    names = " || ".join(f'variant=="{arm}"' for arm in AUX)
    source = source.replace(
        pick,
        pick
        + "\n    if ("
        + names
        + ') { arithmetic_mode=variant; variant="stride"; }',
    )
    source = source.replace(
        call,
        "    if (log && !arithmetic_mode.empty()) result=arithmetic_offer(p,result);\n"
        + call,
    )
    (TREE / "src/controls.inc").write_text(source)
    cc = os.getenv("CXX") or shutil.which("g++-15") or "g++"
    subprocess.run(["make", "-j2", "CXX=" + cc], cwd=TREE, check=True)


def execute(job):
    case, policy = job
    if not (FULL / "intake" / (case + ".validation.csv")).is_file():
        raise ValueError(f"Unvalidated trace {case}")
    native.run(
        case,
        policy,
        [FULL / "raw/traces" / (case + ".champsimtrace")],
        [0],
        50_000_000,
        1_000_000,
        sample=10000,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--cases", choices=CASES, nargs="+")
    parser.add_argument("--jobs", type=int, default=2)
    a = parser.parse_args()
    if a.build:
        build()
    if a.cases:
        native.RAW = RAW
        native.BIN = TREE / "bin/champsim"
        with ThreadPoolExecutor(a.jobs) as pool:
            list(pool.map(execute, product(a.cases, (*NATIVE, *AUX))))
