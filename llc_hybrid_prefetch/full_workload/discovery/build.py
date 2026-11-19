"""Apply the frozen-address candidate to an isolated copy of the validated conventional build."""

import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RAW = HERE.parent / "raw/discovery"
BASE = PROJECT / "opportunity_study/raw/conventional/build/pythia"
TREE = RAW / "pythia"
BIN = TREE / "bin/champsim"


def replace(s, before, after):
    if s.count(before) != 1:
        raise ValueError(f"conventional instrumentation changed near {before}")
    return s.replace(before, after)


def build():
    if not (BASE / "bin/champsim").exists():
        subprocess.run(
            ["python3", str(PROJECT / "opportunity_study/conventional/build.py")],
            check=True,
        )
    if not TREE.exists():
        shutil.copytree(BASE, TREE, symlinks=True)
    source = (PROJECT / "opportunity_study/conventional/controls.inc").read_text()
    source = replace(
        source,
        "void study_init() {",
        (HERE / "frozen.inc").read_text() + "\nvoid study_init() {\n    frozen_init();",
    )
    source = replace(
        source,
        '    if (log) {\n        ++diagnostics["anchors"];',
        '    if (log && !frozen_mode.empty()) result=frozen_apply(p,result);\n    if (log) {\n        ++diagnostics["anchors"];',
    )
    control = TREE / "src/controls.inc"
    if control.read_text() != source:
        control.write_text(source)
    cc = os.getenv("CXX") or shutil.which("g++-15") or "g++"
    subprocess.run(["make", "-j2", "CXX=" + cc], cwd=TREE, check=True)
    return BIN


if __name__ == "__main__":
    print(build())
