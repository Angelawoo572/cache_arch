"""Build a separate passive export against the retained full-workload native simulator."""

import os
import platform
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
FULL = HERE.parent
SOURCE = FULL / "raw/native/build/pythia"
BUILD = FULL / "raw/continuous/build/pythia"


def main():
    if not (SOURCE / "bin/champsim").is_file():
        raise ValueError("First build full_workload/native/build.py")
    if not BUILD.exists():
        shutil.copytree(SOURCE, BUILD, ignore=shutil.ignore_patterns(".git"))
    text = (SOURCE / "src/hybrid.cc").read_text()
    anchors = [
        (
            '#include "observations.inc"',
            '#include "observations.inc"\n#include "ingress.inc"',
        ),
        (
            '        } else if (!std::strcmp(stage, "ingress"))\n            ++r.ingress;',
            '        } else if (!std::strcmp(stage, "ingress")) {\n            ++r.ingress;\n            arrival_record(p);\n        }',
        ),
        (
            "    start_observations();",
            "    start_observations();\n    arrival_start();",
        ),
        (
            "    finish_observations();",
            "    arrival_finish();\n    finish_observations();",
        ),
    ]
    for old, new in anchors:
        if text.count(old) != 1:
            raise ValueError(old)
        text = text.replace(old, new)
    (BUILD / "src/hybrid.cc").write_text(text)
    shutil.copyfile(HERE / "ingress.inc", BUILD / "src/ingress.inc")
    cc = (
        os.getenv("CXX")
        or (shutil.which("g++-15") if platform.system() == "Darwin" else None)
        or "g++"
    )
    subprocess.run(["make", "-j2", "CXX=" + cc], cwd=BUILD, check=True)


if __name__ == "__main__":
    main()
