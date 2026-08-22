#!/usr/bin/env python3
"""Stable access to the authoritative 602 Stride direct-action model.

The class definitions intentionally remain in the completed offline experiment
so old state_dict keys and checkpoints retain their original import semantics.
New training, export, and parity tools all import this wrapper instead of
copying model or decoder code.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
AUTHORITATIVE_SOURCE = (
    ROOT
    / "formal_NN_training"
    / "experiments"
    / "602_offline_lstm_stride"
    / "python"
    / "train_and_offline_infer.py"
)
_MODULE_NAME = "formal_nn_training_602_stride_authoritative"


def _load_authoritative_module():
    module = sys.modules.get(_MODULE_NAME)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(
        _MODULE_NAME, AUTHORITATIVE_SOURCE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "cannot load authoritative model source: {}".format(
                AUTHORITATIVE_SOURCE
            )
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


authoritative = _load_authoritative_module()

MODEL_REVISION = authoritative.MODEL_REVISION
EXPERIMENT_REVISION = authoritative.EXPERIMENT_REVISION
TRACE = authoritative.TRACE
CompactDirectDeltaDecoder = authoritative.CompactDirectDeltaDecoder
CompactPCKeyedHurdleStrideLSTM = (
    authoritative.CompactPCKeyedHurdleStrideLSTM
)
expected_parameter_count = authoritative.expected_parameter_count
runtime_features = authoritative.runtime_features
runtime_encoder_sha256 = authoritative.runtime_encoder_sha256
state_router_sha256 = authoritative.state_router_sha256
load_stream = authoritative.load_stream


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authoritative_source_sha256():
    return sha256(AUTHORITATIVE_SOURCE)


def _torch_load(path, map_location="cpu"):
    try:
        return torch.load(
            path, map_location=map_location, weights_only=False
        )
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(path, expected_hidden_size=None):
    payload = _torch_load(path)
    required = {
        "state_dict", "parameter_count", "hidden_size", "feature_count",
        "model_revision",
    }
    missing = required.difference(payload)
    if missing:
        raise RuntimeError(
            "checkpoint missing fields: {}".format(sorted(missing))
        )
    if payload["model_revision"] != MODEL_REVISION:
        raise RuntimeError(
            "model revision mismatch: {} != {}".format(
                payload["model_revision"], MODEL_REVISION
            )
        )
    hidden_size = int(payload["hidden_size"])
    feature_count = int(payload["feature_count"])
    if expected_hidden_size is not None and hidden_size != expected_hidden_size:
        raise RuntimeError(
            "hidden size mismatch: {} != {}".format(
                hidden_size, expected_hidden_size
            )
        )
    model = CompactPCKeyedHurdleStrideLSTM(feature_count, hidden_size)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    observed = sum(parameter.numel() for parameter in model.parameters())
    expected = expected_parameter_count(feature_count, hidden_size)
    if observed != expected or int(payload["parameter_count"]) != expected:
        raise RuntimeError(
            "parameter mismatch: checkpoint={} model={} expected={}".format(
                payload["parameter_count"], observed, expected
            )
        )
    return model, payload


class FrozenStrideLiveModel:
    """Causal, dynamic exact-PC inference using the authoritative PyTorch model."""

    def __init__(self, model):
        self.model = model.cpu().eval()
        self.hidden_size = int(model.hidden_size)
        self.states = {}

    def reset(self):
        self.states.clear()

    def infer(self, pc, cache_line):
        pc = int(pc)
        cache_line = int(cache_line)
        row = [(pc, cache_line, 0)]
        features = torch.from_numpy(runtime_features(row)).reshape(1, 1, -1)
        previous = self.states.get(pc)
        if previous is None:
            h0 = torch.zeros(1, 1, self.hidden_size)
            c0 = torch.zeros(1, 1, self.hidden_size)
        else:
            h0 = previous[0].reshape(1, 1, -1)
            c0 = previous[1].reshape(1, 1, -1)
        with torch.no_grad():
            projected = torch.tanh(self.model.input_projection(features))
            encoded, (h1, c1) = self.model.encoder_lstm(
                projected, (h0, c0)
            )
            context = encoded[:, 0, :]
            emit = int(self.model.emit_head(context).argmax(dim=-1).item())
            if emit:
                log_mean = (
                    self.model.log_count_mean(context).squeeze(-1).numpy()
                )
                count = int(
                    authoritative._positive_counts_from_log_mean(log_mean)[0]
                )
            else:
                count = 0
            decoder_state = self.model.action_decoder.begin(context)
            deltas = []
            lines = []
            for _ in range(count):
                coordinate = self.model.action_decoder.coordinate(
                    decoder_state
                )
                delta = authoritative._delta_to_integer(
                    float(coordinate.item())
                )
                deltas.append(delta)
                lines.append(
                    authoritative.apply_signed_line_delta(cache_line, delta)
                )
                decoder_state = self.model.action_decoder.advance(
                    decoder_state, coordinate
                )
        h_value = h1[0, 0].detach().clone()
        c_value = c1[0, 0].detach().clone()
        self.states[pc] = (h_value, c_value)
        return {
            "pc": pc,
            "line": cache_line,
            "emit": emit,
            "k": count,
            "deltas": deltas,
            "lines": lines,
            "addresses": [line * 64 for line in lines],
            "hidden": h_value.numpy().astype(np.float32).tolist(),
            "cell": c_value.numpy().astype(np.float32).tolist(),
        }

