#!/usr/bin/env python3
"""Check malformed model rejection and count overflow, without simulation."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "python"))
from live_model_format import expected_tensor_shapes, write_model


class ArrayTensor:
    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


def run_case(runner, directory, name, mutate, error):
    tensors = {key: np.zeros(shape, dtype=np.float32)
               for key, shape in expected_tensor_shapes(8).items()}
    mutate(tensors)
    model = directory / (name + ".bin")
    write_model(model, {key: ArrayTensor(value) for key, value in tensors.items()}, 8, 128)
    events = directory / "events.csv"
    events.write_text("pc,line\n7,1024\n")
    result = subprocess.run([str(runner), "--model", str(model), "--events", str(events),
                             "--output", str(directory / (name + ".jsonl"))],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if error is None:
        if result.returncode:
            raise RuntimeError(result.stderr)
    elif result.returncode == 0 or error not in result.stderr:
        raise RuntimeError("{}: expected rejection {}; got {} {}".format(
            name, error, result.returncode, result.stderr))
    return {"case": name, "status": "PASS", "expected_error": error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    with tempfile.TemporaryDirectory(prefix="stride-formal-numeric-") as value:
        directory = Path(value)
        rows.append(run_case(args.runner, directory, "finite_valid", lambda t: None, None))
        for name, value in [("nan", np.nan), ("positive_inf", np.inf), ("negative_inf", -np.inf)]:
            def mutate(t, value=value):
                t["input_projection.weight"][0, 0] = value
            rows.append(run_case(args.runner, directory, name, mutate, "non-finite model tensor"))
        def overflow(t):
            t["emit_head.bias"][1] = 1
            t["log_count_mean.bias"][0] = 1000
        rows.append(run_case(args.runner, directory, "count_overflow", overflow,
                             "positive-count prediction exceeds host domain"))
    rendered = json.dumps({"status": "PASS", "cases": rows}, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
