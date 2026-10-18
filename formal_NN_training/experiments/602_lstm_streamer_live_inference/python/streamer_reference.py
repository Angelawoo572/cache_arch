"""Faithful frozen address-only Streamer v7 reference; one global LSTM state."""
import sys
sys.dont_write_bytecode = True
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REVISION = "source_input_variable_delta_free_running_v7"
TRACE = "602.gcc_s-734B"
LINE_MASK = (1 << 58) - 1


def cache_root():
    for parent in Path(__file__).resolve().parents:
        if parent.name == "cache" and (parent / "formal_NN_training").is_dir():
            return parent
    raise RuntimeError("This study must run inside the authorized cache directory")


def configure_runtime(path=None):
    runtime = Path(path) if path else cache_root() / ".runtime/streamer_live"
    runtime = runtime.resolve()
    if runtime != cache_root() and cache_root() not in runtime.parents:
        raise RuntimeError("Runtime artifacts must remain inside cache")
    for name in ("tmp", "pycache", "mpl", "torch"):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(runtime / "tmp")
    os.environ["PYTHONPYCACHEPREFIX"] = str(runtime / "pycache")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["MPLCONFIGDIR"] = str(runtime / "mpl")
    os.environ["TORCH_HOME"] = str(runtime / "torch")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.pycache_prefix = str(runtime / "pycache")
    import tempfile
    tempfile.tempdir = str(runtime / "tmp")
    return runtime


def exact_integer(value):
    text = str(value).strip()
    return int(text, 16) if text.lower().startswith("0x") else int(text, 10)


def checked_output(path):
    path = Path(path).resolve()
    if path != cache_root() and cache_root() not in path.parents:
        raise RuntimeError("Output must remain inside cache")
    return path


def load_checkpoint(checkpoint, metadata_path, expected_seed, allow_legacy_metadata=False):
    import torch
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from formal_NN_training.common.threshold_free_policy import build_model
    checkpoint, metadata_path = Path(checkpoint), Path(metadata_path)
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = json.loads(metadata_path.read_text())
    for obj, label in ((data, "checkpoint"), (metadata, "training metadata")):
        if obj.get("matched_normal_prefetcher") != "streamer" or obj.get("experiment_revision") != REVISION:
            raise RuntimeError(label + " is not the actual Streamer v7 model")
        if obj.get("trace") != TRACE:
            raise RuntimeError(label + " has the wrong training trace")
    if metadata.get("seed") != expected_seed:
        raise RuntimeError("Requested seed differs from original training metadata")
    legacy = "seed" not in data
    if legacy and not allow_legacy_metadata:
        raise RuntimeError("Legacy checkpoint has no seed field; explicitly allow original external metadata")
    if not legacy and data["seed"] != expected_seed:
        raise RuntimeError("Requested seed differs from checkpoint")
    if metadata.get("training_runtime_fields") != ["cache_line_address"] or metadata.get("inference_runtime_fields") != ["cache_line_address"]:
        raise RuntimeError("Streamer must remain address-only")
    for field in ("normal_policy_outputs_used_as_model_inputs", "normal_policy_candidates_used_as_model_inputs", "normal_policy_private_state_used_as_model_inputs"):
        if metadata.get(field) is not False:
            raise RuntimeError("Unsupported source input contract: " + field)
    hidden, features = int(data["hidden_size"]), int(data["feature_count"])
    if features != 64 or metadata.get("hidden_size") != hidden or metadata.get("runtime_feature_count") != features:
        raise RuntimeError("Checkpoint and source dimensions differ")
    model, parameters = build_model("lstm", features, hidden, fill_classes=0, mixture_components=4)
    if parameters != data["parameters"] or parameters != metadata["parameter_count"]:
        raise RuntimeError("Parameter inventory mismatch")
    if set(model.state_dict()) != set(data["state_dict"]):
        raise RuntimeError("Not the source 14-tensor Poisson/mixture LSTM")
    for name, value in data["state_dict"].items():
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise RuntimeError("Tensor is not finite FP32: " + name)
        if value.shape != model.state_dict()[name].shape:
            raise RuntimeError("Tensor shape mismatch: " + name)
    model.load_state_dict(data["state_dict"], strict=True)
    torch.set_num_threads(1)
    model.eval()
    return model, data, metadata, legacy


def count_mode(raw):
    raw = float(raw)
    if not math.isfinite(raw):
        raise RuntimeError("Nonfinite Poisson log-rate")
    # Matches numpy.logaddexp(float64(0), float64(raw)); positive rates >=2^63
    # are outside the shared signed host count domain.
    import numpy as np
    rate = float(np.logaddexp(0.0, raw))
    if not math.isfinite(rate) or rate >= 2 ** 63:
        raise RuntimeError("Poisson count outside signed int64 domain")
    return math.floor(rate)


def coordinate_delta(coordinate):
    coordinate = float(coordinate)
    if not math.isfinite(coordinate):
        raise RuntimeError("Nonfinite signed-log delta")
    try:
        magnitude = math.expm1(abs(coordinate))
    except OverflowError as error:
        raise RuntimeError("Delta outside aligned uint64 domain") from error
    if not math.isfinite(magnitude) or magnitude > 2 ** 57:
        raise RuntimeError("Delta outside aligned uint64 domain")
    value = int(round(magnitude))  # Python nearest / ties-to-even.
    return -value if coordinate < 0 else value


class StreamerReference:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.state = None
        self.completed_updates = 0

    def step(self, line):
        import torch
        line = int(line)
        if not 0 <= line <= LINE_MASK:
            raise ValueError("Input must be a 58-bit cache-line address")
        address = line << 6
        runtime = torch.tensor([[(address >> bit) & 1 for bit in range(64)]], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            encoded, self.state = self.model.encode(runtime, self.state)
            self.state = tuple(t.detach() for t in self.state)
            context = encoded[0]
            raw_count = self.model.decoder.count_raw(context)[0].item()
            count = count_mode(raw_count)
            state = self.model.decoder.begin(context)
            actions = []
            for _ in range(count):
                mix, mean, _, fill = self.model.decoder.distribution(state)
                if fill is not None:
                    raise RuntimeError("Streamer has no learned fill head")
                component = mix.argmax(dim=-1, keepdim=True)
                coordinate = mean.gather(1, component).squeeze(1)
                delta = coordinate_delta(coordinate.item())
                target = ((line + delta) & LINE_MASK) << 6
                actions.append(dict(address=target, delta=delta, fill=2, fill_index=-1, coordinate=coordinate.item()))
                state = self.model.decoder.advance(state, coordinate)
        self.completed_updates += 1
        return dict(hidden=self.state[0].reshape(-1).tolist(), cell=self.state[1].reshape(-1).tolist(),
                    count=count, count_raw=raw_count, emit=count > 0, actions=actions)


def literal_graph_cost(hidden, actions):
    h, k = int(hidden), int(actions)
    return dict(lstm_dense_macs=4*h*(64+h), count_dense_macs=h,
                decoder_initial_dense_macs=h*h, decoder_per_action_dense_macs=3*h*(1+h)+12*h,
                literal_dense_macs=4*h*(64+h)+h+h*h+k*(3*h*(1+h)+12*h),
                weights_bytes=(8*h*h+287*h+13)*4,
                global_recurrent_state_bytes=8*h,
                scope="Literal source graph. Backend-returned macs/nonlinears are authoritative for executable cost; no physical PPA claim.")
