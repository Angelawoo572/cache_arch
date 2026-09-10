#!/usr/bin/env python3
"""Run small C++ fixtures using the inspected simulator's actual source/types."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

EXP = Path(__file__).resolve().parents[1]


def simulator_source():
    explicit = os.environ.get("DYNAMIC602_SIMULATOR")
    if explicit:
        candidate = Path(explicit).resolve()
        if not (candidate / "inc/instruction.h").is_file():
            raise RuntimeError("DYNAMIC602_SIMULATOR is not a simulator source: " + str(candidate))
        return candidate
    config = json.loads((EXP / "config.json").read_text())
    built = Path(config["raw"]) / "build.json"
    candidates = []
    if built.is_file():
        candidates.append(Path(json.loads(built.read_text())["simulator"]))
    candidates.append(Path(config["simulator"]))
    # Local terminal development: discover only the named, inspected workspace
    # snapshots, rather than substituting a made-up instruction layout.
    for parent in EXP.parents:
        candidates.extend((parent / "sacramento_sim", parent / "sim_test"))
    for candidate in candidates:
        if (candidate / "inc/instruction.h").is_file():
            return candidate
    raise RuntimeError("No actual ChampSim source found; set DYNAMIC602_SIMULATOR")


class MeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = simulator_source()
        cls.temporary = tempfile.TemporaryDirectory(prefix="dynamic602_measurement_")
        cls.stage = Path(cls.temporary.name)
        shutil.copytree(cls.source / "inc", cls.stage / "inc")
        stride = (cls.source / "prefetcher/stride.cc").read_text()
        old = "StridePrefetcher::StridePrefetcher(string type) : Prefetcher(type)\n{\n\n}"
        if old in stride:
            stride = stride.replace(old, "StridePrefetcher::StridePrefetcher(string type) : Prefetcher(type)\n{\n   init_stats();\n}", 1)
        (cls.stage / "stride.cc").write_text(stride)
        header = cls.stage / "inc/stride.h"
        text = header.read_text()
        if "dynamic602_tracker_count" not in text:
            anchor = "   ~StridePrefetcher();"
            if text.count(anchor) != 1:
                raise RuntimeError("actual Stride header anchor changed")
            text = text.replace(anchor, anchor + "\n   size_t dynamic602_tracker_count() const { return trackers.size(); }", 1)
        header.write_text(text)
        cls.compiler = os.environ.get("CXX") or shutil.which("c++") or "g++"

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def compile_and_run(self, name, extra=()):
        binary = self.stage / name
        command = [self.compiler, "-std=c++11", "-O2",
                   "-I" + str(self.stage / "inc"),
                   "-I" + str(EXP / "measurement"),
                   str(EXP / "measurement" / (name + ".cc"))]
        command.extend(map(str, extra))
        command.extend(("-o", str(binary)))
        compiled = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
        result = subprocess.run([str(binary)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS", result.stdout)
        print(result.stdout.strip(), flush=True)

    def test_occupancy_lifecycles_and_actual_trace_record_skipping(self):
        self.compile_and_run("test_measurement")

    def test_preserved_conventional_stride(self):
        self.compile_and_run("test_stride", (self.stage / "stride.cc",))


if __name__ == "__main__":
    unittest.main(verbosity=2)
