#!/usr/bin/env python3
"""Persistent CPU optimizer for the Stride simulator co-simulation.

Requests: little-endian u32 JSON byte length, then UTF-8 JSON. Replies:
little-endian u32 status, u32 payload length. Successful payloads contain
<QQQQddd (version, available decisions, positive decisions, exposures, total
loss, mean gate loss, gradient L2), then little-endian FP32 TENSOR_ORDER weights.
Error payloads contain UTF-8 text. The simulator supplies only already-observed
events and their predecision h/c; teacher actions are loss-only line addresses.
"""
import argparse
from collections import OrderedDict
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import resource
import socket
import struct
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "formal_NN_training/experiments/602_lstm_stride_live_inference/python"))
from formal_NN_training.common.stride_direct_action_model import (
    CompactPCKeyedHurdleStrideLSTM, authoritative, load_checkpoint,
    runtime_features,
)
from formal_NN_training.common.threshold_free_policy import targets_from_actions
from live_model_format import TENSOR_ORDER, write_model

REQUEST_LENGTH = struct.Struct("<I")
REPLY_FRAME = struct.Struct("<II")
REPLY_HEADER = struct.Struct("<QQQQddd")
MAX_EVENTS = 64
MAX_REQUEST_BYTES = 1024 * 1024


def _storage(tensor):
    storage = tensor.untyped_storage() if hasattr(tensor, "untyped_storage") else tensor.storage()
    size = storage.nbytes() if hasattr(storage, "nbytes") else storage.size() * tensor.element_size()
    return (str(tensor.device), storage.data_ptr(), size), size


class SavedTensorMeter:
    """Count unique saved storage, excluding shared parameters and inputs."""

    def __init__(self, exclude, byte_budget):
        self.excluded = {_storage(tensor)[0] for tensor in exclude}
        self.storages = {}
        self.byte_budget = byte_budget
        self.total_bytes = 0

    def pack(self, tensor):
        key, size = _storage(tensor)
        if key not in self.excluded and key not in self.storages:
            self.storages[key] = size
            self.total_bytes += size
        if self.bytes > self.byte_budget:
            raise RuntimeError("online saved activation storage exceeds configured {} byte budget; update not published".format(self.byte_budget))
        # Retain the storage/value needed by backward, never its grad_fn.
        # Returning the original tensor creates reference cycles, notably in
        # the authoritative decoder's unused final recurrent advance.
        return tensor.detach()

    def unpack(self, tensor):
        return tensor

    @property
    def bytes(self):
        return self.total_bytes


def _uint(value, name, bits=64):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("{} must be an integer".format(name))
    if value < 0 or value >= 1 << bits:
        raise ValueError("{} outside uint{}".format(name, bits))
    return value


class OnlineTrainer:
    """One pass per bounded chronological batch, with no future trace access."""

    def __init__(self, hidden_size=8, seed=7, checkpoint=None, learning_rate=0.002):
        self.started_at = time.monotonic()
        if hidden_size not in (8, 16):
            raise ValueError("hidden_size must be 8 or 16")
        torch.set_num_threads(1)
        # This may already have been set by another trainer in software tests.
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.model = (
            load_checkpoint(checkpoint, expected_hidden_size=hidden_size)[0]
            if checkpoint else CompactPCKeyedHurdleStrideLSTM(128, hidden_size)
        ).cpu()
        self.model.train()
        self.hidden_size = hidden_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.version = 0
        self.decisions = 0
        self.positives = 0
        self.exposures = 0
        self.actions = 0
        self.total_padded_positions = 0
        self.peak_padded_positions = 0
        self.peak_tbptt_span = 0
        self.peak_examples = 0
        self.peak_example_payload_bytes = 0
        self.peak_gradient_bytes = 0
        self.peak_optimizer_bytes = 0
        self.peak_saved_activation_bytes = 0
        self.peak_input_target_bytes = 0
        self.saved_tensor_meter_available = hasattr(getattr(torch.autograd, "graph", None), "saved_tensors_hooks")
        if not self.saved_tensor_meter_available:
            raise RuntimeError("PyTorch saved_tensors_hooks required to enforce activation storage limit")
        self.activation_byte_budget = hidden_size // 8 * 1024 * 1024
        self.stats_path = None
        self.last_loss = 0.0
        self.last_gate_loss = 0.0
        self.last_gradient_l2 = 0.0
        self.last_class_weights = [1.0, 1.0]
        self.last_components = {}
        self.last_group_lengths = []

    def weight_array(self):
        state = self.model.state_dict()
        return np.concatenate([
            state[name].detach().cpu().numpy().reshape(-1)
            for name in TENSOR_ORDER
        ]).astype("<f4", copy=False)

    def response(self):
        return REPLY_HEADER.pack(
            self.version, self.decisions, self.positives, self.exposures,
            self.last_loss, self.last_gate_loss, self.last_gradient_l2,
        ) + self.weight_array().tobytes()

    def _prepare(self, events):
        if not isinstance(events, list) or len(events) > MAX_EVENTS:
            raise ValueError("training batch must contain at most 64 events")
        groups = OrderedDict()
        starts = {}
        lifetime_pcs = {}
        rows, actions = [], []
        for index, event in enumerate(events):
            pc = _uint(event["pc"], "pc")
            line = _uint(event["line"], "line", 58)
            life = _uint(event["life"], "life")
            if life in lifetime_pcs and lifetime_pcs[life] != pc:
                raise ValueError("one lifetime cannot contain different PCs")
            lifetime_pcs[life] = pc
            # Every entry carries its actual predecision state. Grouping uses
            # the first one, then the authoritative recurrent encoder carries
            # differentiable h/c through only this already-observed batch.
            h = np.asarray(event["h"], dtype=np.float32)
            c = np.asarray(event["c"], dtype=np.float32)
            if h.shape != (self.hidden_size,) or c.shape != (self.hidden_size,):
                raise ValueError("predecision h/c shape mismatch")
            if not np.isfinite(h).all() or not np.isfinite(c).all():
                raise ValueError("non-finite predecision h/c")
            if life not in starts:
                starts[life] = (torch.from_numpy(h).clone(), torch.from_numpy(c).clone())
            groups.setdefault(life, []).append(index)
            rows.append((pc, line, 0))
            values = event["actions"]
            if not isinstance(values, list):
                raise ValueError("teacher actions must be a list")
            # Stride labels have at most two requests; this validates the
            # teacher contract and is never a neural decoder degree limit.
            if len(values) > 2:
                raise ValueError("Stride teacher emitted more than two labels")
            actions.append([_uint(value, "teacher line", 58) for value in values])
        batch = list(groups.items())
        batch.sort(key=lambda item: (-len(item[1]), item[1][0]))
        return rows, actions, batch, starts

    def train_batch(self, events):
        rows, actions, batch, starts = self._prepare(events)
        if not rows:
            return self.response()
        counts, deltas, _ = targets_from_actions([row[1] for row in rows], actions)
        # The original loss reshapes with -1; a zero-width all-silent tensor
        # cannot be inferred by PyTorch. One unused zero column preserves the
        # exact loss because all counts are zero and no action is active.
        if deltas.shape[1] == 0:
            deltas = np.zeros((len(rows), 1), dtype=np.float32)
        available = self.decisions + len(rows)
        positive = self.positives + int(np.count_nonzero(counts))
        negative = available - positive
        class_weights = (
            np.asarray([available / (2.0 * negative), available / (2.0 * positive)], dtype=np.float32)
            if positive and negative else np.ones(2, dtype=np.float32)
        )
        padded, count_batch, delta_batch, lengths = authoritative._make_padded_batch(
            runtime_features(rows), counts, deltas, batch, torch.device("cpu")
        )
        self.optimizer.zero_grad(set_to_none=True)
        meter = SavedTensorMeter(list(self.model.parameters()) + [padded, count_batch, delta_batch], self.activation_byte_budget)
        hooks = (
            torch.autograd.graph.saved_tensors_hooks(meter.pack, meter.unpack)
            if self.saved_tensor_meter_available else nullcontext()
        )
        with hooks:
            context = authoritative._encode_shared(
                self.model, padded, lengths, starts, [life for life, _ in batch]
            )
            loss, components = authoritative._compact_hurdle_direct_delta_loss(
                self.model, context, count_batch, delta_batch, torch.from_numpy(class_weights)
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite online loss; update not published")
            loss.backward()
        squared_gradient = 0.0
        gradient_bytes = 0
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("non-finite gradient; update not published")
                squared_gradient += float(parameter.grad.detach().double().square().sum())
                gradient_bytes += parameter.grad.numel() * parameter.grad.element_size()
        self.optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in self.model.parameters()):
            raise RuntimeError("non-finite updated weights; worker must terminate")
        self.version += 1
        self.decisions = available
        self.positives = positive
        self.exposures += len(rows)
        self.actions += int(counts.sum())
        padded_positions = len(batch) * max(lengths)
        self.total_padded_positions += padded_positions
        self.peak_padded_positions = max(self.peak_padded_positions, padded_positions)
        self.peak_tbptt_span = max(self.peak_tbptt_span, max(lengths))
        self.peak_examples = max(self.peak_examples, len(rows))
        # Logical binary example payload: PC,line,lifetime, count (u64 each),
        # saved h/c FP32 and actual teacher line addresses. JSON transport and
        # Python allocator overhead are separate host overhead.
        self.peak_example_payload_bytes = max(
            self.peak_example_payload_bytes,
            len(rows) * (32 + 8 * self.hidden_size) + int(counts.sum()) * 8,
        )
        self.peak_gradient_bytes = max(self.peak_gradient_bytes, gradient_bytes)
        optimizer_bytes = sum(
            value.numel() * value.element_size()
            for state in self.optimizer.state.values()
            for value in state.values() if torch.is_tensor(value)
        )
        self.peak_optimizer_bytes = max(self.peak_optimizer_bytes, optimizer_bytes)
        self.peak_saved_activation_bytes = max(self.peak_saved_activation_bytes, meter.bytes)
        self.peak_input_target_bytes = max(self.peak_input_target_bytes, sum(
            tensor.numel() * tensor.element_size() for tensor in (padded, count_batch, delta_batch)
        ))
        self.last_loss = float(loss.detach())
        self.last_gate_loss = components["gate_loss_sum"] / len(rows)
        self.last_gradient_l2 = math.sqrt(squared_gradient)
        self.last_class_weights = class_weights.tolist()
        self.last_components = components
        self.last_group_lengths = lengths
        self.write_stats()
        return self.response()

    def stats(self):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {
            "worker_host_elapsed_seconds_since_initialization": time.monotonic() - self.started_at,
            "worker_host_cpu_seconds": usage.ru_utime + usage.ru_stime,
            "worker_peak_rss_bytes": int(usage.ru_maxrss) * (1 if sys.platform == "darwin" else 1024),
            "worker_rss_scope": "entire Python worker including libraries, allocator, IPC, optimizer and measurement; separate from required predictor storage",
            "versions_completed": self.version,
            "trainer_available_decisions": self.decisions,
            "trainer_available_positive_decisions": self.positives,
            "training_sample_exposures": self.exposures,
            "training_action_exposures": self.actions,
            "total_padded_projection_positions": self.total_padded_positions,
            "peak_padded_projection_positions": self.peak_padded_positions,
            "peak_effective_same_lifetime_tbptt_span": self.peak_tbptt_span,
            "peak_examples": self.peak_examples,
            "peak_logical_example_payload_bytes": self.peak_example_payload_bytes,
            "peak_gradient_tensor_bytes": self.peak_gradient_bytes,
            "peak_optimizer_tensor_bytes_including_steps": self.peak_optimizer_bytes,
            "peak_unique_saved_activation_storage_bytes_excluding_parameters_and_padded_inputs": (
                self.peak_saved_activation_bytes if self.saved_tensor_meter_available else None
            ),
            "saved_activation_measurement": "unique autograd-saved underlying storages; aliases counted once; parameter and padded input/target storages excluded",
            "peak_padded_input_target_tensor_bytes": self.peak_input_target_bytes,
            "model_parameter_bytes": self.weight_array().nbytes,
            "configured_max_decisions_per_batch": MAX_EVENTS,
            "configured_max_padded_positions": 1056,
            "configured_max_logical_example_payload_bytes": MAX_EVENTS * (48 + 8 * self.hidden_size),
            "configured_max_padded_input_target_tensor_bytes": 1056 * (128 * 4 + 8 + 2 * 4),
            "configured_max_gradient_tensor_bytes": self.weight_array().nbytes,
            "configured_max_adam_moment_tensor_bytes": 2 * self.weight_array().nbytes,
            "configured_max_adam_step_tensor_bytes": len(TENSOR_ORDER) * 4,
            "configured_max_same_lifetime_tbptt_span": MAX_EVENTS,
            "configured_saved_activation_storage_byte_budget": self.activation_byte_budget,
            "last_loss": self.last_loss,
            "last_gradient_l2": self.last_gradient_l2,
            "last_gate_class_weights": self.last_class_weights,
            "class_balance_history": "only labels received in already-observed admitted training batches; unit weights until both classes",
            "state_initialization": "saved predecision h/c for first event of each lifetime in each batch; no trainer PC history table",
            "optimizer": "Adam", "learning_rate": self.optimizer.param_groups[0]["lr"],
            "library_threads": 1,
        }

    def write_stats(self):
        if self.stats_path:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.stats_path.with_name(self.stats_path.name + ".tmp")
            temporary.write_text(json.dumps(self.stats(), indent=2) + "\n")
            temporary.replace(self.stats_path)


def receive_exact(connection, size):
    parts = []
    remaining = size
    while remaining:
        value = connection.recv(remaining)
        if not value:
            if remaining == size:
                return None
            raise EOFError("truncated IPC message")
        parts.append(value)
        remaining -= len(value)
    return b"".join(parts)


def serve(connection, trainer):
    while True:
        raw = receive_exact(connection, REQUEST_LENGTH.size)
        if raw is None:
            return
        size, = REQUEST_LENGTH.unpack(raw)
        if not 0 < size <= MAX_REQUEST_BYTES:
            raise ValueError("IPC request length out of bounds")
        request_bytes = receive_exact(connection, size)
        if request_bytes is None:
            raise EOFError("missing IPC request payload")
        try:
            request = json.loads(request_bytes.decode("utf-8"))
            if request.get("op") == "init":
                payload = trainer.response()
            elif request.get("op") == "train":
                payload = trainer.train_batch(request["events"])
            else:
                raise ValueError("unknown worker operation")
        except Exception as exc:
            payload = "{}: {}".format(type(exc).__name__, exc).encode("utf-8")
            connection.sendall(REPLY_FRAME.pack(1, len(payload)) + payload)
            raise
        connection.sendall(REPLY_FRAME.pack(0, len(payload)) + payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--hidden-size", type=int, choices=(8, 16), default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--initial-model-bin", type=Path)
    parser.add_argument("--stats", type=Path)
    args = parser.parse_args()
    trainer = OnlineTrainer(args.hidden_size, args.seed, args.checkpoint, args.learning_rate)
    trainer.stats_path = args.stats
    if args.socket.exists():
        raise RuntimeError("socket path exists; refusing to unlink another worker")
    if args.initial_model_bin:
        if args.initial_model_bin.exists():
            raise RuntimeError("initial model exists; choose a fresh run directory")
        args.initial_model_bin.parent.mkdir(parents=True, exist_ok=True)
        write_model(args.initial_model_bin, trainer.model.state_dict(), args.hidden_size, 128)
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(args.socket))
        os.chmod(str(args.socket), 0o600)
        listener.listen(1)
        print("online_worker_ready socket={} hidden={} parameters={}".format(
            args.socket, args.hidden_size, len(trainer.weight_array())), flush=True)
        connection, _ = listener.accept()
        try:
            serve(connection, trainer)
        finally:
            connection.close()
    finally:
        listener.close()
        if args.socket.exists():
            args.socket.unlink()
        if args.stats:
            trainer.write_stats()


if __name__ == "__main__":
    main()
