#!/usr/bin/env python3
"""Compare Python and standalone C++ frozen inference event by event."""

import argparse
import csv
import gzip
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "formal_NN_training/experiments/602_lstm_stride_live_inference"
sys.path.insert(0, str(EXP / "python"))

from live_model_format import TENSOR_ORDER, read_model, write_model


LINE_MASK = (1 << 58) - 1


class ArrayTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def shapes(hidden):
    return {
        "input_projection.weight": (hidden, 128),
        "input_projection.bias": (hidden,),
        "encoder_lstm.weight_ih_l0": (4 * hidden, hidden),
        "encoder_lstm.weight_hh_l0": (4 * hidden, hidden),
        "encoder_lstm.bias_ih_l0": (4 * hidden,),
        "encoder_lstm.bias_hh_l0": (4 * hidden,),
        "emit_head.weight": (2, hidden),
        "emit_head.bias": (2,),
        "log_count_mean.weight": (1, hidden),
        "log_count_mean.bias": (1,),
        "action_decoder.action_cell.weight_ih": (3 * hidden, 1),
        "action_decoder.action_cell.weight_hh": (3 * hidden, hidden),
        "action_decoder.action_cell.bias_ih": (3 * hidden,),
        "action_decoder.action_cell.bias_hh": (3 * hidden,),
        "action_decoder.delta_head.weight": (1, hidden),
        "action_decoder.delta_head.bias": (1,),
    }


def empty_tensors(hidden):
    return {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in shapes(hidden).items()
    }


def scalar_dot(matrix, row, values):
    total = np.float32(0.0)
    for weight, value in zip(matrix[row], values):
        total = np.float32(total + np.float32(weight * value))
    return total


def linear(weight, bias, values):
    result = np.empty(weight.shape[0], dtype=np.float32)
    for row in range(weight.shape[0]):
        result[row] = np.float32(
            bias[row] + scalar_dot(weight, row, values)
        )
    return result


def sigmoid(value):
    value = np.float32(value)
    if value >= 0:
        inverse = np.float32(np.exp(np.float32(-value)))
        return np.float32(1.0) / np.float32(1.0 + inverse)
    exponential = np.float32(np.exp(value))
    return exponential / np.float32(1.0 + exponential)


def coordinate_to_delta(coordinate):
    magnitude = math.expm1(abs(float(coordinate)))
    if not math.isfinite(magnitude):
        raise RuntimeError("synthetic delta exceeds domain")
    rounded = int(round(magnitude))
    return -rounded if coordinate < 0 else rounded


class NumpyFrozenRuntime:
    def __init__(self, binary):
        self.t = binary["tensors"]
        self.h = int(binary["hidden_size"])
        self.states = {}

    def infer(self, pc, line):
        features = np.zeros(128, dtype=np.float32)
        address = (int(line) << 6) & ((1 << 64) - 1)
        for bit in range(64):
            features[bit] = (int(pc) >> bit) & 1
            features[64 + bit] = (address >> bit) & 1
        projected = np.tanh(linear(
            self.t["input_projection.weight"],
            self.t["input_projection.bias"],
            features,
        )).astype(np.float32)
        previous_h, previous_c = self.states.get(
            int(pc),
            (
                np.zeros(self.h, dtype=np.float32),
                np.zeros(self.h, dtype=np.float32),
            ),
        )
        w_ih = self.t["encoder_lstm.weight_ih_l0"]
        w_hh = self.t["encoder_lstm.weight_hh_l0"]
        b_ih = self.t["encoder_lstm.bias_ih_l0"]
        b_hh = self.t["encoder_lstm.bias_hh_l0"]
        gates = np.empty(4 * self.h, dtype=np.float32)
        for row in range(4 * self.h):
            value = np.float32(b_ih[row] + b_hh[row])
            value = np.float32(value + scalar_dot(w_ih, row, projected))
            value = np.float32(value + scalar_dot(
                w_hh, row, previous_h
            ))
            gates[row] = value
        hidden = np.empty(self.h, dtype=np.float32)
        cell = np.empty(self.h, dtype=np.float32)
        for index in range(self.h):
            input_gate = sigmoid(gates[index])
            forget_gate = sigmoid(gates[self.h + index])
            candidate = np.float32(np.tanh(
                np.float32(gates[2 * self.h + index])
            ))
            output_gate = sigmoid(gates[3 * self.h + index])
            cell[index] = np.float32(
                np.float32(forget_gate * previous_c[index])
                + np.float32(input_gate * candidate)
            )
            hidden[index] = np.float32(
                output_gate * np.float32(np.tanh(cell[index]))
            )
        self.states[int(pc)] = (hidden.copy(), cell.copy())
        logits = linear(
            self.t["emit_head.weight"],
            self.t["emit_head.bias"],
            hidden,
        )
        emit = 1 if logits[1] > logits[0] else 0
        if emit:
            log_mean = linear(
                self.t["log_count_mean.weight"],
                self.t["log_count_mean.bias"],
                hidden,
            )[0]
            count = max(1, int(np.rint(math.exp(float(log_mean)))))
        else:
            count = 0
        decoder = hidden.copy()
        deltas = []
        addresses = []
        for _ in range(count):
            coordinate = linear(
                self.t["action_decoder.delta_head.weight"],
                self.t["action_decoder.delta_head.bias"],
                decoder,
            )[0]
            delta = coordinate_to_delta(coordinate)
            deltas.append(delta)
            target_line = (int(line) + delta) & LINE_MASK
            addresses.append(target_line << 6)
            next_state = np.empty(self.h, dtype=np.float32)
            wi = self.t["action_decoder.action_cell.weight_ih"]
            wh = self.t["action_decoder.action_cell.weight_hh"]
            bi = self.t["action_decoder.action_cell.bias_ih"]
            bh = self.t["action_decoder.action_cell.bias_hh"]
            for index in range(self.h):
                input_r = np.float32(
                    bi[index] + np.float32(wi[index, 0] * coordinate)
                )
                input_z = np.float32(
                    bi[self.h + index]
                    + np.float32(wi[self.h + index, 0] * coordinate)
                )
                input_n = np.float32(
                    bi[2 * self.h + index]
                    + np.float32(wi[2 * self.h + index, 0] * coordinate)
                )
                hidden_r = np.float32(
                    bh[index] + scalar_dot(wh, index, decoder)
                )
                hidden_z = np.float32(
                    bh[self.h + index]
                    + scalar_dot(wh, self.h + index, decoder)
                )
                hidden_n = np.float32(
                    bh[2 * self.h + index]
                    + scalar_dot(wh, 2 * self.h + index, decoder)
                )
                reset = sigmoid(np.float32(input_r + hidden_r))
                update = sigmoid(np.float32(input_z + hidden_z))
                candidate = np.float32(np.tanh(np.float32(
                    input_n + np.float32(reset * hidden_n)
                )))
                next_state[index] = np.float32(
                    np.float32((np.float32(1.0) - update) * candidate)
                    + np.float32(update * decoder[index])
                )
            decoder = next_state
        return {
            "pc": int(pc),
            "line": int(line),
            "emit": emit,
            "k": count,
            "deltas": deltas,
            "addresses": addresses,
            "hidden": hidden.tolist(),
            "cell": cell.tolist(),
        }


def write_model_from_arrays(path, tensors, hidden):
    wrapped = {name: ArrayTensor(tensors[name]) for name in TENSOR_ORDER}
    write_model(path, wrapped, hidden, 128)


def write_events(path, events):
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pc", "line"])
        writer.writerows(events)


def run_cpp(cpp_runner, model, events, output, stats=None):
    command = [
        str(cpp_runner), "--model", str(model), "--events", str(events),
        "--output", str(output),
    ]
    if stats:
        command.extend(["--stats", str(stats)])
    subprocess.run(command, check=True)
    return [
        json.loads(line) for line in Path(output).read_text().splitlines()
        if line.strip()
    ]


def compare_results(expected, observed, state_tolerance, fixture):
    if len(expected) != len(observed):
        raise RuntimeError(
            "{} event count differs: {} != {}".format(
                fixture, len(expected), len(observed)
            )
        )
    max_hidden = 0.0
    max_cell = 0.0
    for index, (left, right) in enumerate(zip(expected, observed)):
        action_fields = (
            "pc", "line", "emit", "k", "deltas", "addresses"
        )
        differences = {
            field: (left[field], right.get(field))
            for field in action_fields
            if left[field] != right.get(field)
        }
        if differences:
            raise RuntimeError(
                "exact action mismatch fixture={} event={} details={}".format(
                    fixture, index, differences
                )
            )
        hidden_error = max(
            [abs(a - b) for a, b in zip(
                left["hidden"], right["hidden"]
            )] or [0.0]
        )
        cell_error = max(
            [abs(a - b) for a, b in zip(
                left["cell"], right["cell"]
            )] or [0.0]
        )
        max_hidden = max(max_hidden, hidden_error)
        max_cell = max(max_cell, cell_error)
        if hidden_error > state_tolerance or cell_error > state_tolerance:
            raise RuntimeError(
                "state mismatch fixture={} event={} hidden={} cell={}".format(
                    fixture, index, hidden_error, cell_error
                )
            )
    return max_hidden, max_cell


def synthetic_cases():
    hidden = 8
    events = [
        (0x10, 0),
        (0x10, 1),
        (0x20, LINE_MASK),
        (0x10, LINE_MASK - 1),
        (0x20, 2),
        (0x30, 1 << 57),
    ]
    cases = []
    definitions = [
        ("silent_output", 0, 1, 0),
        ("k1_positive_delta", 1, 1, 3),
        ("k2_negative_delta", 1, 2, -2),
        ("k4_zero_delta", 1, 4, 0),
    ]
    for name, emit, count, delta in definitions:
        tensors = empty_tensors(hidden)
        tensors["emit_head.bias"][:] = (
            [-1.0, 1.0] if emit else [1.0, -1.0]
        )
        tensors["log_count_mean.bias"][0] = np.float32(math.log(count))
        tensors["action_decoder.delta_head.bias"][0] = np.float32(
            math.copysign(math.log1p(abs(delta)), delta)
            if delta else 0.0
        )
        cases.append((name, tensors, events))
    rng = np.random.default_rng(602)
    tensors = empty_tensors(hidden)
    for name in TENSOR_ORDER:
        tensors[name][...] = rng.uniform(
            -0.035, 0.035, size=tensors[name].shape
        ).astype(np.float32)
    tensors["emit_head.bias"][:] = [-1.0, 1.0]
    tensors["log_count_mean.bias"][0] = np.float32(math.log(2.0))
    tensors["action_decoder.delta_head.bias"][0] = np.float32(0.8)
    cases.append(("stateful_interleaved_address_edges", tensors, events))
    return hidden, cases


def run_synthetic(args):
    args.work_dir.mkdir(parents=True, exist_ok=True)
    hidden, cases = synthetic_cases()
    result = {
        "mode": "synthetic",
        "fixtures": [],
        "action_mismatch_count": 0,
        "max_hidden_state_error": 0.0,
        "max_cell_state_error": 0.0,
    }
    for name, tensors, events in cases:
        case_dir = args.work_dir / name
        case_dir.mkdir(parents=True, exist_ok=True)
        model_path = case_dir / "model.bin"
        events_path = case_dir / "events.csv"
        output_path = case_dir / "cpp_outputs.jsonl"
        stats_path = case_dir / "runtime_stats.json"
        write_model_from_arrays(model_path, tensors, hidden)
        write_events(events_path, events)
        reference = NumpyFrozenRuntime(read_model(model_path))
        expected = [
            reference.infer(pc, line) for pc, line in events
        ]
        observed = run_cpp(
            args.cpp_runner, model_path, events_path, output_path, stats_path
        )
        max_hidden, max_cell = compare_results(
            expected, observed, args.state_tolerance, name
        )
        result["max_hidden_state_error"] = max(
            result["max_hidden_state_error"], max_hidden
        )
        result["max_cell_state_error"] = max(
            result["max_cell_state_error"], max_cell
        )
        result["fixtures"].append({
            "name": name,
            "events": len(events),
            "status": "PASS",
        })
    result["status"] = "PASS"
    return result


def load_gzip_rows(path, max_events):
    rows = []
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((int(row["pc"]), int(row["line"])))
            if max_events and len(rows) >= max_events:
                break
    return rows


def run_recorded(args):
    if not args.model_bin or not args.checkpoint or not args.stream:
        raise RuntimeError(
            "recorded mode requires --model-bin, --checkpoint, and --stream"
        )
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from formal_NN_training.common.stride_direct_action_model import (
        FrozenStrideLiveModel,
        load_checkpoint,
    )
    args.work_dir.mkdir(parents=True, exist_ok=True)
    events = load_gzip_rows(args.stream, args.max_events)
    event_path = args.work_dir / "recorded_events.csv"
    output_path = args.work_dir / "recorded_cpp_outputs.jsonl"
    stats_path = args.work_dir / "recorded_runtime_stats.json"
    write_events(event_path, events)
    model, _ = load_checkpoint(args.checkpoint)
    reference = FrozenStrideLiveModel(model)
    expected = [reference.infer(pc, line) for pc, line in events]
    observed = run_cpp(
        args.cpp_runner, args.model_bin, event_path, output_path, stats_path
    )
    max_hidden, max_cell = compare_results(
        expected, observed, args.state_tolerance, "recorded_stream"
    )
    return {
        "mode": "recorded_stream",
        "events": len(events),
        "action_mismatch_count": 0,
        "action_mismatch_rate": 0.0,
        "first_mismatch": None,
        "max_hidden_state_error": max_hidden,
        "max_cell_state_error": max_cell,
        "status": "PASS",
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp-runner", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--model-bin", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stream", type=Path)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--state-tolerance", type=float, default=2e-5)
    parser.add_argument("--output", type=Path)
    return parser


def main():
    args = build_parser().parse_args()
    if not args.cpp_runner.is_file():
        raise RuntimeError("C++ runner missing: {}".format(args.cpp_runner))
    result = run_synthetic(args) if args.synthetic else run_recorded(args)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
