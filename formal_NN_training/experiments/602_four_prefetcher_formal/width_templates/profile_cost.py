#!/usr/bin/env python3
"""H16 requested-allocation/operation evidence; no ChampSim execution.

Compiles the pre-existing storage probe against the actual archived simulator
runtime and native headers. Inputs are exactly the already-passed correctness
fixtures. No critical protocol, model, core or historical output is modified.
"""
import argparse
import csv
import gzip
import importlib.util
import itertools
import json
import os
import platform
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("width_cost_driver", HERE / "driver.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    D.write(path, value)


def run(command, destination, environment, stdin=None):
    process = subprocess.run(
        [str(v) for v in command], input=stdin, text=True,
        capture_output=True, cwd=D.ROOT, env=environment)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(process.stdout + process.stderr)
    if process.returncode:
        raise RuntimeError("cost-only command failed; see " + str(destination))
    return process.stdout


def fixture(policy, raw, destination):
    parity = read(raw / "correctness" / (policy + "_h16") / "parity.json")
    if parity.get("status") not in ("PASS", "verified"):
        raise RuntimeError("unverified source correctness fixture")
    bindings = [D.bind(
        raw / "correctness" / (policy + "_h16") / "parity.json",
        destination.parent / "source" / (policy + "_h16_parity.json"))]
    if policy == "ampm":
        source = raw / "correctness/ampm_h16/commands.txt"
        text = source.read_text()
        bindings.append(D.bind(source, destination))
        expected_callbacks = parity["callbacks"]
        expected_actions = parity["actions"]
        expected_demands, expected_fills = expected_callbacks, 0
        source_stream = parity["source_stream"]
    else:
        source_stream = parity["sequence_source"]
        sequence = []
        with gzip.open(source_stream, "rt", newline="") as handle:
            for row in itertools.islice(csv.DictReader(handle), 4096):
                word = row["event_address"]
                if not word or word[0] in "+-":
                    raise RuntimeError("invalid unsigned source address")
                address = int(word, 16 if word.lower().startswith("0x") else 10)
                sequence.append(("D" if row["event_kind"] == "DEMAND" else "F", address))
        if len(sequence) != 4096:
            raise RuntimeError("SPP correctness stream shorter than expected")
        boundary = [0, 64, 4032, 4096, 1 << 63, (1 << 64) - 64, 64, 0]
        sequence += [(kind, address) for address in boundary for kind in ("F", "D")]
        text = "".join("{} {}\n".format(kind, address) for kind, address in sequence)
        if destination.exists() and destination.read_text() != text:
            raise RuntimeError("saved SPP fixture changed")
        destination.write_text(text)
        expected_callbacks = parity["callback_count"]
        expected_actions = parity["actions_compared"]
        expected_demands = parity["demand_callbacks"]
        expected_fills = parity["fill_callbacks"]
    records = text.splitlines()
    if len(records) != expected_callbacks:
        raise RuntimeError("correctness fixture callback count changed")
    for line in records:
        words = line.split()
        if len(words) != 2 or words[0] not in ("D", "F") or not words[1].isdigit():
            raise RuntimeError("cost fixture must use unsigned decimal D/F records")
        address = int(words[1], 10)
        if address >= (1 << 64) or address & 63:
            raise RuntimeError("cost fixture invalid aligned uint64 address")
    return text, dict(callbacks=expected_callbacks, actions=expected_actions,
                      demand_callbacks=expected_demands, fill_callbacks=expected_fills,
                      source_stream=source_stream, commands=str(destination),
                      real_contiguous_callbacks=4096,
                      boundary_callbacks=expected_callbacks - 4096), bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        help="default: configured independent RAW/cost")
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()
    state, build = D.frozen()
    raw = D.RAW
    gate = read(raw / "correctness_gate.json")
    if gate.get("status") != "PASS" or gate.get("build_id") != build["build_id"]:
        raise RuntimeError("actual-build H16 correctness gate not passed")
    cost = args.out or raw / "cost"
    cost.mkdir(parents=True, exist_ok=True)
    runtime = cost / "runtime"
    runtime.mkdir(exist_ok=True)
    source = cost / "source"
    source.mkdir(exist_ok=True)
    sim = Path(build["simulator"])
    probe_source = D.PARENT / "validation/storage_probe.cc"
    core = sim / "prefetcher/live602_runtime.cc"
    core_header = sim / "inc/live602_runtime.h"
    bindings = [
        D.bind(Path(__file__), source / "profile_cost.py"),
        D.bind(probe_source, source / "storage_probe.cc"),
        D.bind(core, source / "live602_runtime.cc"),
        D.bind(core_header, source / "native_headers/live602_runtime.h"),
        D.bind(Path(D.C["reference_raw"]) / "build.json", source / "actual_build.json"),
        D.bind(raw / "correctness_gate.json", source / "correctness_gate.json")]
    for name in ("streamer.h", "ampm.h", "bitmap.h", "prefetcher.h",
                 "spp_dev2.h", "spp_dev2_helper.h"):
        bindings.append(D.bind(sim / "inc" / name, source / "native_headers" / name))
    # Confirm this is the exact shared scalar implementation, not a H16 fork.
    for name, actual in (("live602_runtime.cc", core), ("live602_runtime.h", core_header)):
        if actual.read_bytes() != (D.ROOT / "formal_NN_training/common/live602_runtime" / name).read_bytes():
            raise RuntimeError("canonical core differs from actual archived core: " + name)
    environment = D.env()
    compiler_version = run([args.compiler, "--version"], cost / "compiler.txt", environment)
    command = [args.compiler, "-std=c++11", "-O3", "-ffp-contract=off",
               "-I" + str(sim / "inc"), core, probe_source,
               "-o", runtime / "storage_probe"]
    run(command, cost / "compile.log", environment)
    models = {}
    common_sizes = None
    for policy in ("ampm", "spp"):
        key = policy + "_h16"
        model_path = D.model_path(policy)
        immutable = model_path.with_name("immutable_model.bin")
        if model_path.read_bytes() != immutable.read_bytes():
            raise RuntimeError("frozen H16 model changed: " + policy)
        bindings.append(D.bind(model_path, source / "models" / key / "model.bin"))
        bindings.append(D.bind(immutable, source / "models" / key / "immutable_model.bin"))
        commands, expected, fixture_bindings = fixture(
            policy, raw, cost / (key + "_commands.txt"))
        bindings.extend(fixture_bindings)
        output = run([runtime / "storage_probe", model_path],
                     cost / (key + "_probe.log"), environment, stdin=commands)
        data = json.loads(output)
        header = D.header(model_path)
        actual = data["model"]
        for field in ("hidden_size", "parameter_count", "feature_count", "fill_classes"):
            target = {"parameter_count": "parameters", "feature_count": "features"}.get(field, field)
            if actual[target] != header[field]:
                raise RuntimeError("probe/model-header mismatch " + key + "/" + field)
        if actual["hidden_size"] != 16 or actual["global_hc_bytes"] != 128:
            raise RuntimeError("incorrect H16 state storage")
        if actual["fp32_weight_bytes"] != actual["parameters"] * 4:
            raise RuntimeError("incorrect FP32 weight storage")
        for field in ("callbacks", "actions", "demand_callbacks", "fill_callbacks"):
            if data["fixture"][field] != expected[field]:
                raise RuntimeError("cost probe differs from correctness fixture: " + key + "/" + field)
        if data["host_allocations"]["residual_after_runtime_destruction"] != 0:
            raise RuntimeError("tracked runtime allocation leak")
        if common_sizes is not None and common_sizes != data["sizeof"]:
            raise RuntimeError("inconsistent actual C++ type layouts")
        common_sizes = data["sizeof"]
        data["model_path"] = str(model_path)
        data["model_header"] = header
        data["model"]["weight_plus_global_hc_payload_bytes"] = (
            actual["fp32_weight_bytes"] + actual["global_hc_bytes"])
        data["model"]["payload_qualifier"] = (
            "FP32 weights plus one global h/c pair only; excludes scratch/output/control; "
            "not a full architectural storage or hardware PPA total")
        data["fixture"].update(expected)
        data["fixture"]["bound_to_passed_correctness_sequence"] = True
        save(cost / (key + "_host_memory.json"), data)
        models[key] = data
    D.verify(bindings)
    D.frozen()
    result = dict(
        schema="four602_width_actual_source_memory_cost_v1",
        status="PASS", protocol=D.C["protocol"], actual_build=build,
        host=platform.node(), compiler=compiler_version.splitlines()[0],
        compile_command=[str(v) for v in command],
        binding_method="source/model/fixture byte comparison and preserved copies; no new hash ledger",
        bindings=bindings, nn_models=models,
        native_cpp_sizeof={key: value for key, value in common_sizes.items()
                           if not key.startswith("live602::")},
        native_sizeof_scope=(
            "actual archived native class/helper layouts, including descriptors and padding; "
            "not complete native constructor heap allocation or synthesized storage. "
            "Native histograms/diagnostic descriptors are present in class sizeof."),
        cost_model=dict(
            encoder_macs="4*H*(F+H) per callback",
            encoder_nonlinears="5*H per callback",
            ampm_h16_demand_macs="5136 + 256*(K>0) + 1008*K",
            ampm_h16_demand_nonlinears="81 + 16*(K>0) + 53*K",
            spp_h16_fill_macs=5184, spp_h16_fill_nonlinears=80,
            spp_h16_demand_macs="5216 + 16*emit + 1136*K",
            spp_h16_demand_nonlinears="80 + emit + 53*K",
            mac_definition="dense FP32 multiply-accumulate pairs; bias adds, argmax, bit extraction and control excluded",
            nonlinear_definition="primitive sigmoid/tanh/softplus/exp/expm1 counts, not calibrated cycles",
            full_forward="raw learned K/delta/fill decode completes before external gates or payload replacement",
            measurement="synchronous MAC0; no modeled resource queue/service/bandwidth/weight transfer/energy",
            mac4_ratio="ceil(counted_MACs/4)/measured_MAC0_ROI_cycles is only an aggregate throughput necessary condition on that MAC0 callback timeline/window, not finite-resource IPC",
            weights="resident host vectors; no simulated NN-weight DRAM traffic; 4*MAC is not measured weight traffic"),
        byte_interpretation=(
            "scalar runtime object sizeof plus ordinary new/new[] requested persistent payload "
            "includes its weights/global h/c/container overhead; do not add weight+h/c again. "
            "Automatic stack frames, allocator metadata/alignment/tracking headers, C-library "
            "malloc internals, Four602 outer adapter, measurement observers, logs and fixture "
            "buffers excluded; neither RSS nor synthesized hardware/PPA."),
        scratch_scope=(
            "temporary requested-heap high-water mark observed only on the exact passed "
            "4096-real-callback-plus-boundary fixtures. Global decoder has no model degree cap; "
            "observed K/peak is not a universal worst-case resource bound."),
        no_simulator_executed=True)
    save(cost / "bindings.json", dict(status="PASS", bindings=bindings))
    save(cost / "cost_fields_h16.json", result)
    print(json.dumps(dict(status="PASS", output=str(cost / "cost_fields_h16.json"),
                         nn_models={k: dict(model=v["model"],
                                           host_allocations=v["host_allocations"],
                                           fixture=v["fixture"])
                                    for k, v in models.items()}), indent=2))


if __name__ == "__main__":
    main()
