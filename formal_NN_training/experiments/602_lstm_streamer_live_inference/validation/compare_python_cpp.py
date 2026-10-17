#!/usr/bin/env python3
"""Real-checkpoint Python/C++ callback parity, including global-state reset/edges."""
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import subprocess

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "python"))
from streamer_reference import (ROOT, StreamerReference, checked_output, configure_runtime,
                                coordinate_delta, count_mode, exact_integer, load_checkpoint)


def source_lines(path, limit):
    opener = gzip.open if str(path).endswith(".gz") else open
    result = []
    with opener(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        if "line" not in (reader.fieldnames or []):
            raise RuntimeError("Real sequence needs the canonical line column: " + str(path))
        for row in reader:
            line = exact_integer(row["line"])
            if not 0 <= line < 2 ** 58:
                raise RuntimeError("Real input line is outside aligned uint64 domain")
            result.append(line)
            if len(result) >= limit:
                break
    if not result:
        raise RuntimeError("Empty correctness input sequence")
    return result


def expected_cost(hidden, actions):
    h, k = hidden, actions
    return (4*h*(64+h)+h+(h*h if k else 0)+k*(3*h*(1+h)+12*h),
            5*h+1+(h if k else 0)+k*(3*h+4+1))


def math_boundaries():
    assert count_mode(0) == 0
    assert count_mode(20) == 20
    assert coordinate_delta(0) == 0
    assert coordinate_delta(math.log1p(.5)) == 0
    assert coordinate_delta(math.log1p(1.5)) == 2
    assert coordinate_delta(-math.log1p(1.5)) == -2
    for function, values in [(count_mode, [math.nan, math.inf, -math.inf, float(2**63)]),
                             (coordinate_delta, [math.nan, math.inf, -math.inf, 10000])]:
        for value in values:
            try:
                function(value)
            except RuntimeError:
                continue
            raise AssertionError("Invalid numeric boundary accepted")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--run-metadata", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--allow-legacy-metadata", action="store_true")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--sequence", type=Path, action="append", required=True)
    p.add_argument("--max-events", type=int, default=1024)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--runtime-dir", type=Path)
    p.add_argument("--abs-tol", type=float, default=2e-5)
    p.add_argument("--rel-tol", type=float, default=2e-5)
    a = p.parse_args()
    if a.max_events < 1 or a.abs_tol <= 0 or a.rel_tol < 0:
        raise ValueError("Invalid correctness fixture limits/tolerances")
    configure_runtime(a.runtime_dir)
    output = checked_output(a.out_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Preserve previous correctness output; use a fresh directory")
    model, checkpoint, metadata, legacy = load_checkpoint(a.checkpoint, a.run_metadata, a.seed, a.allow_legacy_metadata)
    sys.path.insert(0, str(ROOT))
    from formal_NN_training.common.live602_runtime.model_format import canonicalize_poisson_lstm, read_model
    config, tensors = read_model(a.model)
    canonical = canonicalize_poisson_lstm(model.state_dict())
    if config["feature_count"] != 64 or config["count_mode"] or config["fill_classes"] or config["kind_feature"]:
        raise RuntimeError("Not the Streamer Poisson address-only configuration")
    for name, value in canonical.items():
        if value.reshape(-1).tolist() != tensors[name]["data"]:
            raise RuntimeError("Binary does not contain the actual Streamer checkpoint tensors: " + name)
    reference = StreamerReference(model)
    boundaries = [0, 1, 62, 63, 64, 65, (1<<58)-2, (1<<58)-1, 1<<57, 1<<56, 0, 63, 64, 0]
    commands, expected, segments = [], [], []
    for path in a.sequence:
        lines = source_lines(path, a.max_events)
        commands.append("R")
        expected.append(dict(reset=True))
        reference.reset()
        for line in lines + boundaries:
            commands.append("D " + str(line << 6))
            expected.append(reference.step(line))
        segments.append(dict(source=str(path.resolve()), actual_contiguous_source_rows=len(lines), boundary_rows=len(boundaries),
                             initial_state="independent_zero_then_continuous_global_h_c"))
    # Explicit deterministic reset: replay the first contiguous callbacks twice.
    first = source_lines(a.sequence[0], min(8, a.max_events))
    for _ in range(2):
        commands.append("R")
        expected.append(dict(reset=True))
        reference.reset()
        for line in first:
            commands.append("D " + str(line << 6))
            expected.append(reference.step(line))
    text = "\n".join(commands) + "\n"
    result = subprocess.run([str(a.probe), str(a.model)], input=text, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError("Shared C++ probe failed: " + result.stderr)
    actual = [json.loads(line) for line in result.stdout.splitlines()]
    if len(actual) != len(expected):
        raise RuntimeError("Probe callback/reset record count differs")
    max_errors = dict(hidden=0.0, cell=0.0, count_raw=0.0, coordinate=0.0)
    numeric_checks = 0
    actions = 0
    callbacks = 0
    failures = []
    def numbers(key, left, right, index):
        nonlocal numeric_checks
        if len(left) != len(right):
            failures.append(dict(index=index, field=key, reason="shape"))
            return
        for x, y in zip(left, right):
            numeric_checks += 1
            error = abs(float(x)-float(y))
            max_errors[key] = max(max_errors[key], error)
            if not math.isfinite(float(y)) or not math.isclose(float(x), float(y), rel_tol=a.rel_tol, abs_tol=a.abs_tol):
                failures.append(dict(index=index, field=key, python=x, cpp=y))
    for index, (py, cpp) in enumerate(zip(expected, actual)):
        if py.get("reset"):
            if cpp != dict(reset=True):
                failures.append(dict(index=index, reason="reset record"))
            continue
        callbacks += 1
        for key in ("count", "emit"):
            if py[key] != cpp[key]:
                failures.append(dict(index=index, field=key, python=py[key], cpp=cpp[key]))
        numbers("hidden", py["hidden"], cpp["hidden"], index)
        numbers("cell", py["cell"], cpp["cell"], index)
        numbers("count_raw", [py["count_raw"]], [cpp["count_raw"]], index)
        if len(py["actions"]) != len(cpp["actions"]) or cpp.get("truncated") is not False:
            failures.append(dict(index=index, reason="action length or unreported truncation"))
            continue
        for pa, ca in zip(py["actions"], cpp["actions"]):
            actions += 1
            for key in ("address", "delta", "fill", "fill_index"):
                if pa[key] != ca[key]:
                    failures.append(dict(index=index, field="action." + key, python=pa[key], cpp=ca[key]))
            numbers("coordinate", [pa["coordinate"]], [ca["coordinate"]], index)
        macs, nonlinear = expected_cost(model.model_size, py["count"])
        if cpp["macs"] != macs or cpp["nonlinears"] != nonlinear:
            failures.append(dict(index=index, reason="executable operation count", expected=[macs, nonlinear], actual=[cpp["macs"], cpp["nonlinears"]]))
    math_boundaries()
    rejected = []
    for command in ("D 1", "F 0"):
        edge = subprocess.run([str(a.probe), str(a.model)], input=command+"\n", text=True, capture_output=True)
        if edge.returncode == 0:
            failures.append(dict(reason="invalid input unexpectedly accepted", command=command))
        else:
            rejected.append(dict(command=command, diagnostic=edge.stderr.strip()))
    output.mkdir(parents=True, exist_ok=True)
    (output / "commands.txt").write_text(text)
    (output / "python.jsonl").write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in expected))
    (output / "cpp.jsonl").write_text(result.stdout)
    report = dict(status="FAIL" if failures else "PASS", policy="streamer", seed=a.seed,
                  checkpoint=str(a.checkpoint.resolve()), model=str(a.model.resolve()), probe=str(a.probe.resolve()),
                  actual_global_callbacks=callbacks, compared_all_actions=actions, numeric_checks=numeric_checks,
                  state_scope="one_global_h_c", source_segments=segments, reset_records=len(expected)-callbacks,
                  max_absolute_errors=max_errors, abs_tol=a.abs_tol, rel_tol=a.rel_tol, failures=failures[:100],
                  failure_count=len(failures), invalid_inputs_rejected=rejected, scalar_boundary_tests="PASS",
                  count_and_all_action_fields_required_exact=True, weight_roundtrip="exact FP32", executable_cost_checked=True,
                  qualification="Numerical/semantic correctness fixtures using real contiguous inputs plus boundaries; no ChampSim ROI, pilot IPC or benchmark result.")
    (output / "parity_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
