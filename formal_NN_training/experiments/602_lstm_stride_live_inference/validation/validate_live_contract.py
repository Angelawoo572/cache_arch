#!/usr/bin/env python3
"""Static contract and artifact audit for 602 frozen live inference."""

import argparse
import ast
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "formal_NN_training/experiments/602_lstm_stride_live_inference"
OFFLINE = ROOT / "formal_NN_training/experiments/602_offline_lstm_stride"


def fail(message):
    raise RuntimeError(message)


def read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        fail("{} must be a JSON object".format(path))
    return value


def validate_notebook(path):
    notebook = read_json(path)
    if notebook.get("nbformat") != 4:
        fail("notebook must use nbformat 4")
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if any(
            line.lstrip().startswith(("!", "%"))
            for line in source.splitlines()
        ):
            continue
        ast.parse(source, filename="{}:{}".format(path, index))


def validate_config():
    sweep = read_json(EXP / "config/training_prefix_sweep.json")
    if sweep.get("hidden_sizes") != [8, 16] or sweep.get("seeds") != [7]:
        fail("exploratory matrix must be exactly h8/h16 and seed 7")
    expected = [
        (100, "i100"), (250, "i250"), (500, "i500"),
        (1000, "i1k"), (2000, "i2k"), (5000, "i5k"),
        (10000, "i10k"), (20000, "i20k"), (50000, "i50k"),
        (100000, "i100k"), (250000, "i250k"),
        (500000, "i500k"), (1000000, "i1m"),
        (2000000, "i2m"), (5000000, "i5m"),
        (10000000, "i10m"), (20000000, "i20m"),
    ]
    observed = [
        (item["instructions"], item["tag"])
        for item in sweep.get("instruction_budgets", [])
    ]
    if observed != expected:
        fail("instruction budgets/tags changed")
    controls = sweep["fixed_controls"]
    fixed = {
        "model_revision": "compact_shared_pc_hurdle_delta_v9",
        "epochs": 12,
        "chunk_length": 256,
        "optimizer": "Adam",
        "learning_rate": 0.002,
        "seed": 7,
        "training_warmup_instructions": 0,
        "evaluation_warmup_instructions": 25000000,
        "evaluation_simulation_instructions": 25000000,
        "fresh_initialization_per_point": True,
    }
    for key, expected_value in fixed.items():
        if controls.get(key) != expected_value:
            fail("fixed control changed: {}".format(key))

    runtime = read_json(EXP / "config/runtime_contract.json")
    required_false = (
        "online_learning", "backpropagation_in_live_runtime",
        "optimizer_in_live_runtime", "teacher_in_live_runtime",
        "replay_list_in_live_runtime", "same_page_rule", "quantization",
    )
    for key in required_false:
        if runtime.get(key) is not False:
            fail("{} must be false".format(key))
    if runtime.get("weights_frozen_in_champsim") is not True:
        fail("live weights are not declared frozen")
    if runtime.get("external_model_inputs") != [
        "pc", "cache_line_address"
    ]:
        fail("live external input boundary changed")
    if runtime.get("parameter_counts") != {"h8": 1908, "h16": 5220}:
        fail("h8/h16 parameter counts changed")
    if runtime.get("format_version") != 1:
        fail("model export format is not versioned")
    if runtime.get("primary_state_mode") != "parity":
        fail("primary state semantics changed")


def validate_sources():
    required = (
        EXP / "README.md",
        EXP / "COMMANDS.md",
        EXP / "python/export_live_model.py",
        EXP / "python/run_training_prefix_sweep.py",
        EXP / "runtime/stride_lstm_runtime.cc",
        EXP / "runtime/stride_lstm_model_loader.cc",
        EXP / "runtime/champsim/stride_lstm_live.cc",
        EXP / "runtime/champsim/install_live_prefetcher.sh",
        EXP / "validation/compare_python_cpp_outputs.py",
        OFFLINE / "python/train_and_offline_infer.py",
        ROOT / "formal_NN_training/common/stride_direct_action_model.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        fail("required source missing: {}".format(missing))
    for path in EXP.rglob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    notebook = EXP / "colab/train_prefix_sweep.ipynb"
    if not notebook.is_file():
        fail("Colab notebook missing")
    validate_notebook(notebook)

    wrapper = (
        ROOT / "formal_NN_training/common/stride_direct_action_model.py"
    ).read_text()
    if "class CompactPCKeyedHurdleStrideLSTM" in wrapper:
        fail("shared wrapper copied the authoritative model definition")
    if "train_and_offline_infer.py" not in wrapper:
        fail("shared wrapper does not import the offline authority")

    runtime_paths = (
        EXP / "runtime/stride_lstm_runtime.cc",
        EXP / "runtime/stride_lstm_runtime.h",
        EXP / "runtime/champsim/stride_lstm_live.cc",
        EXP / "runtime/champsim/stride_lstm_live.h",
    )
    runtime_text = "\n".join(path.read_text() for path in runtime_paths)
    for forbidden in (
        "PFETCH_LIST_PATH", "replay.csv", "optimizer", "backprop",
        "normal_policy", "probability_threshold", "degree_cap",
    ):
        if forbidden.lower() in runtime_text.lower():
            fail("forbidden live runtime token: {}".format(forbidden))
    if (
        "Infer(std::uint64_t pc, std::uint64_t cache_line)"
        not in runtime_text
    ):
        fail("C++ runtime input boundary is not PC/cache-line only")
    for required_token in (
        "STRIDE_LSTM_MODEL_BIN", "weights frozen", "runtime_->Reset()",
        "realistic_state_warmup", "warmup_complete",
    ):
        if required_token not in runtime_text:
            fail("live state/weight token missing: {}".format(required_token))

    exporter = (EXP / "python/live_model_format.py").read_text()
    if "FORMAT_VERSION = 1" not in exporter or "TENSOR_ORDER" not in exporter:
        fail("versioned complete export definition missing")
    if "action_decoder.delta_head.weight" not in exporter:
        fail("decoder tensors omitted from export")
    cpp_loader = (EXP / "runtime/stride_lstm_model_loader.cc").read_text()
    if "11ULL * hidden_size_ * hidden_size_" not in cpp_loader:
        fail("C++ parameter-count formula missing")


def validate_shell():
    scripts = sorted(EXP.rglob("*.sh"))
    for path in scripts:
        subprocess.run(["bash", "-n", str(path)], check=True)
        text = path.read_text()
        if "Usage:" not in text and path.name != "install_live_prefetcher.sh":
            fail("shell entrypoint has no usage: {}".format(path))
        if "git clean -" in text:
            fail("destructive git clean token in {}".format(path))


def validate_sacramento_python_compatibility():
    """Keep host-side control/validation code usable on Sacramento Python 3.6."""
    sources = list(EXP.rglob("*.py")) + list(EXP.rglob("*.sh"))
    sources.append(EXP / "colab/train_prefix_sweep.ipynb")
    forbidden = {
        "Python 3.7 subprocess text keyword": "text" + "=True",
        "Python 3.7 subprocess capture_output keyword": (
            "capture_" + "output="
        ),
        "NumPy 1.17 random generator": "default_" + "rng",
        "Python 3.8 pathlib missing_ok": "missing_" + "ok=",
        "Python 3.9 string prefix helper": "remove" + "prefix(",
        "Python 3.9 pathlib relative helper": "is_" + "relative_to(",
    }
    bad = []
    for path in sources:
        source = path.read_text()
        for description, token in forbidden.items():
            if token in source:
                bad.append("{}: {}".format(path, description))
        if "import " + "pandas" in source or "from " + "pandas" in source:
            bad.append("{}: pandas is not a Sacramento dependency".format(path))
    if bad:
        fail("Sacramento Python 3.6 compatibility failures: {}".format(bad))


def validate_artifacts():
    probe = EXP / "runs/contract_probe/model.bin"
    ignored = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "-q", str(probe)],
        check=False,
    )
    if ignored.returncode != 0:
        fail("generated runs/model.bin is not ignored")
    tracked = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files"], universal_newlines=True
    ).splitlines()
    prefix = str(EXP.relative_to(ROOT)) + "/"
    bad = []
    for name in tracked:
        if not name.startswith(prefix):
            continue
        relative = name[len(prefix):]
        if relative.startswith("runs/") or relative.endswith((
            ".pt", ".pth", ".ckpt", ".bin", ".log", ".csv.gz",
            ".tar.gz",
        )):
            bad.append(name)
    if bad:
        fail("tracked generated artifacts: {}".format(bad))


def build_parser():
    return argparse.ArgumentParser(description=__doc__)


def main():
    build_parser().parse_args()
    if not OFFLINE.is_dir():
        fail("existing offline Stride experiment was removed")
    validate_config()
    validate_sources()
    validate_shell()
    validate_sacramento_python_compatibility()
    validate_artifacts()
    print("[PASS] frozen live runtime has no learning/teacher/list dependency")
    print("[PASS] external model inputs are PC and cache-line address only")
    print("[PASS] h8=1908 and h16=5220; no other hidden size configured")
    print("[PASS] versioned float32 export contains all recurrent/action heads")
    print("[PASS] parity reset and optional realistic warmup are separated")
    print("[PASS] Sacramento Python 3.6/legacy NumPy path needs no pandas")
    print("[PASS] generated data, logs, checkpoints, and weights are untracked")
    print("[PASS] completed offline 602 Stride reference remains present")


if __name__ == "__main__":
    main()
