"""Run the registered undelayed frozen shape rule on full native samples and counters."""

import argparse
import importlib.util
import os
import shutil
import subprocess
from build import HERE, RAW, replace


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_full(control=False):
    base = HERE.parent / "raw/native/build/pythia"
    tree = RAW / ("full_control_build/pythia" if control else "full_build/pythia")
    if not tree.exists():
        shutil.copytree(
            base, tree, symlinks=True, ignore=shutil.ignore_patterns(".git")
        )
    for filename in ("hybrid.cc", "observations.inc"):
        shutil.copyfile(base / "src" / filename, tree / "src" / filename)
    source = (base / "src/controls.inc").read_text()
    source = replace(
        source,
        "void study_init() {",
        (HERE / "frozen.inc").read_text() + "\nvoid study_init() {\n    frozen_init();",
    )
    source = replace(
        source,
        'variant = env_or("STUDY_POLICY", "stride");',
        'variant = env_or("STUDY_POLICY", "stride");\n    if ((variant=="shape" || variant=="sparse3") && frozen_mode==variant) variant="stride";',
    )
    source = replace(
        source,
        '    if (log) {\n        ++diagnostics["anchors"];',
        '    if (log && !frozen_mode.empty()) result=frozen_apply(p,result);\n    if (log) {\n        ++diagnostics["anchors"];',
    )
    (tree / "src/controls.inc").write_text(source)
    cc = os.getenv("CXX") or shutil.which("g++-15") or "g++"
    subprocess.run(["make", "-j2", "CXX=" + cc], cwd=tree, check=True)
    return tree / "bin/champsim"


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--build", action="store_true")
    cli.add_argument("--policy", choices=("shape", "sparse3"), default="shape")
    cli.add_argument(
        "cases",
        nargs="*",
        choices=(
            "bfs_discovery",
            "bfs_transfer",
            "pr_heldout",
            "pr_heldout_entire",
            "mcf06_discovery",
            "mcf17_discovery",
            "mcf17_transfer",
            "soplex06_heldout",
        ),
    )
    a = cli.parse_args()
    control = a.policy == "sparse3"
    if control and a.cases != ["bfs_discovery"]:
        cli.error("the fixed +3 follow-up is registered only on BFS-3")
    binary = (
        build_full(control)
        if a.build
        else RAW
        / ("full_control_build" if control else "full_build")
        / "pythia/bin/champsim"
    )
    native = load("full_native", HERE.parent / "native/run.py")
    native.RAW = RAW / "full"
    native.BIN = binary
    os.environ["DISCOVERY_RULE"] = a.policy
    os.environ["DISCOVERY_FIT"] = str(HERE / "shape.rules")
    for case in a.cases:
        roi = 450_000_001 if case == "pr_heldout_entire" else 200_000_000
        native.run(
            case,
            a.policy,
            [HERE.parent / "raw/traces" / (case + ".champsimtrace")],
            [0],
            50_000_000,
            roi,
        )
