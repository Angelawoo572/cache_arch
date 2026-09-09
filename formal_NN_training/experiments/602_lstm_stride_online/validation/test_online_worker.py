#!/usr/bin/env python3
"""Essential CPU worker correctness tests; these are not research results."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "python"))
from online_worker import (
    MAX_EVENTS, OnlineTrainer, REPLY_FRAME, REPLY_HEADER, REQUEST_LENGTH, SavedTensorMeter,
    authoritative, receive_exact, runtime_features,
)
from formal_NN_training.common.normal_policy_reference import stride_actions
from formal_NN_training.common.threshold_free_policy import targets_from_actions
from live_model_format import TENSOR_ORDER, read_model


def events(count=64, hidden=8, positive=None, offset=0):
    rows = [(11 + i % 3, 1024 + (i // 3) % 40 + offset, 0) for i in range(count)]
    actions, _ = stride_actions(rows)
    if positive is not None:
        actions = [[line, line + 1] if positive else [] for _, line, _ in rows]
    return [{
        "pc": pc, "line": line, "life": pc,
        "h": [0.0] * hidden, "c": [0.0] * hidden, "actions": action,
    } for (pc, line, _), action in zip(rows, actions)]


class OnlineWorkerTests(unittest.TestCase):
    def test_real_gradients_and_parameters_change(self):
        trainer = OnlineTrainer(8, 7)
        before = trainer.weight_array().copy()
        trainer.train_batch(events())
        self.assertEqual(trainer.version, 1)
        self.assertEqual(trainer.exposures, 64)
        self.assertGreater(trainer.last_gradient_l2, 0)
        self.assertTrue(np.isfinite(trainer.weight_array()).all())
        self.assertFalse(np.array_equal(before, trainer.weight_array()))
        self.assertGreater(trainer.peak_gradient_bytes, 0)
        self.assertGreater(trainer.peak_optimizer_bytes, 2 * before.nbytes)
        self.assertEqual(trainer.peak_tbptt_span, 22)

    def test_single_classes_short_and_empty(self):
        for positive in (False, True):
            trainer = OnlineTrainer(8, 17)
            initial = trainer.response()
            self.assertEqual(trainer.train_batch([]), initial)
            trainer.train_batch(events(7, positive=positive))
            self.assertEqual(trainer.version, 1)
            self.assertEqual(trainer.exposures, 7)
            self.assertEqual(trainer.last_class_weights, [1.0, 1.0])
            self.assertTrue(np.isfinite(trainer.last_loss))
            self.assertEqual(trainer.last_components["action_atoms"], 14 if positive else 0)
            if not positive:
                self.assertEqual(trainer.last_components["positive_count_atoms"], 0)
                self.assertEqual(trainer.last_components["action_delta_loss_sum"], 0)

    def test_causal_class_balance(self):
        trainer = OnlineTrainer(8, 7)
        trainer.train_batch(events(9, positive=False))
        self.assertEqual(trainer.last_class_weights, [1.0, 1.0])
        trainer.train_batch(events(3, positive=True))
        np.testing.assert_allclose(trainer.last_class_weights, [12 / 18, 12 / 6])
        self.assertEqual(trainer.decisions, 12)
        self.assertEqual(trainer.positives, 3)

    def test_future_prefix_independence_of_updates_and_decisions(self):
        prefix = [events(64), events(13, offset=64)]
        outputs = []
        for future in (events(64, positive=False), events(64, positive=True, offset=128)):
            trainer = OnlineTrainer(8, 27)
            snapshots = []
            frozen = authoritative.CompactPCKeyedHurdleStrideLSTM(128, 8)
            for batch in prefix:
                snapshots.append(trainer.train_batch(batch))
                # The prediction modules see current/past weights and inputs;
                # future labels cannot change these saved earlier decisions.
                frozen.load_state_dict(trainer.model.state_dict())
                encoded = torch.tanh(frozen.input_projection(
                    torch.from_numpy(runtime_features([(91, 1088, 0)]))
                )).reshape(1, 1, -1)
                with torch.no_grad():
                    context, _ = frozen.encoder_lstm(encoded)
                    snapshots.append(frozen.emit_head(context).numpy().tobytes())
            trainer.train_batch(future)
            outputs.append(snapshots)
        self.assertEqual(outputs[0], outputs[1])

    def test_lifetime_reallocation_does_not_reconnect(self):
        trainer = OnlineTrainer(8, 7)
        batch_events = events(3, positive=False)
        for index, event in enumerate(batch_events):
            event["pc"] = 11
            event["life"] = 100 + index
            event["h"] = [float(index)] * 8
            event["c"] = [float(index) / 2] * 8
        rows, actions, groups, states = trainer._prepare(batch_events)
        self.assertEqual([len(indices) for _, indices in groups], [1, 1, 1])
        self.assertEqual(set(states), {100, 101, 102})
        for index in range(3):
            np.testing.assert_array_equal(states[100 + index][0], [float(index)] * 8)
        trainer.train_batch(batch_events)
        self.assertEqual(trainer.peak_tbptt_span, 1)
        malformed = events(2)
        malformed[1]["life"] = malformed[0]["life"]
        with self.assertRaisesRegex(ValueError, "different PCs"):
            trainer.train_batch(malformed)

    def test_teacher_deltas_remain_loss_only(self):
        first = events(8, positive=True)
        second = events(8, positive=True)
        for event in second:
            event["actions"] = [event["line"] - 3, event["line"] + 7]
        recorded = []
        losses = []
        for batch_events in (first, second):
            trainer = OnlineTrainer(8, 7)
            inputs = []
            original = trainer.model.action_decoder.action_cell.forward
            def record(value, state):
                inputs.append((value.detach().numpy().copy(), state.detach().numpy().copy()))
                return original(value, state)
            with patch.object(trainer.model.action_decoder.action_cell, "forward", record):
                trainer.train_batch(batch_events)
            recorded.append(inputs)
            losses.append(trainer.last_loss)
        self.assertNotEqual(losses[0], losses[1])
        self.assertEqual(len(recorded[0]), len(recorded[1]))
        for first_input, second_input in zip(recorded[0], recorded[1]):
            for left, right in zip(first_input, second_input):
                np.testing.assert_array_equal(left, right)

    def test_bounds_and_header(self):
        trainer = OnlineTrainer(16, 7)
        with self.assertRaisesRegex(ValueError, "at most 64"):
            trainer.train_batch(events(MAX_EVENTS + 1, hidden=16))
        self.assertEqual(trainer.version, 0)
        header = REPLY_HEADER.unpack(trainer.response()[:REPLY_HEADER.size])
        self.assertEqual(header, (0, 0, 0, 0, 0.0, 0.0, 0.0))
        self.assertEqual(REPLY_HEADER.size, 56)
        self.assertEqual(len(trainer.response()), 56 + 5220 * 4)

    def test_activation_budget_overflow_is_explicit_and_unpublished(self):
        trainer = OnlineTrainer(8, 7)
        self.assertEqual(trainer.activation_byte_budget, 1024 * 1024)
        before = trainer.weight_array().copy()
        trainer.activation_byte_budget = 1
        with self.assertRaisesRegex(RuntimeError, "exceeds configured"):
            trainer.train_batch(events(8))
        self.assertEqual(trainer.version, 0)
        np.testing.assert_array_equal(before, trainer.weight_array())

    def test_saved_tensor_hook_breaks_graph_cycles_without_changing_updates(self):
        value = torch.ones(8, requires_grad=True) * 2
        meter = SavedTensorMeter([], 1024)
        packed = meter.pack(value)
        self.assertIsNone(packed.grad_fn, "saved values must not retain their own computation graph")
        self.assertFalse(packed.requires_grad)
        self.assertEqual(packed.data_ptr(), value.data_ptr(), "detach should share storage, not add a copy")
        np.testing.assert_array_equal(meter.unpack(packed).numpy(), value.detach().numpy())
        fixed = OnlineTrainer(8, 7)
        expected = [fixed.train_batch(events(64, offset=index)) for index in range(8)]
        original_hook = SavedTensorMeter.pack
        def previous_pack(self, tensor):
            original_hook(self, tensor)
            return tensor
        previous = OnlineTrainer(8, 7)
        with patch.object(SavedTensorMeter, "pack", previous_pack):
            observed = [previous.train_batch(events(64, offset=index)) for index in range(8)]
        self.assertEqual(expected, observed, "measurement correction must preserve every update and weight exactly")

    def test_persistent_socket_roundtrip_and_initial_export(self):
        with tempfile.TemporaryDirectory(prefix="s602_", dir="/tmp") as temporary:
            directory = Path(temporary)
            socket_path = directory / "worker.sock"
            stats = directory / "stats.json"
            initial = directory / "model.bin"
            process = subprocess.Popen([
                sys.executable, str(EXP / "python/online_worker.py"),
                "--socket", str(socket_path), "--initial-model-bin", str(initial),
                "--stats", str(stats), "--seed", "7",
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                deadline = time.monotonic() + 20
                while not socket_path.exists():
                    if process.poll() is not None or time.monotonic() >= deadline:
                        self.fail("worker startup failed: {}".format(process.communicate(timeout=5)))
                    time.sleep(0.01)
                connection.connect(str(socket_path))
                def exchange(value):
                    raw = json.dumps(value, separators=(",", ":")).encode()
                    connection.sendall(REQUEST_LENGTH.pack(len(raw)) + raw)
                    status, size = REPLY_FRAME.unpack(receive_exact(connection, REPLY_FRAME.size))
                    self.assertEqual(status, 0)
                    return receive_exact(connection, size)
                reply = exchange({"op": "init"})
                model = read_model(initial)
                weights = np.concatenate([model["tensors"][key].reshape(-1) for key in TENSOR_ORDER])
                np.testing.assert_array_equal(np.frombuffer(reply[56:], dtype="<f4"), weights)
                updated = exchange({"op": "train", "events": events()})
                self.assertEqual(REPLY_HEADER.unpack(updated[:56])[:4], (1, 64, 58, 64))
                self.assertNotEqual(reply[56:], updated[56:])
                self.assertEqual(exchange({"op": "train", "events": []}), updated)
                connection.close()
                output, error = process.communicate(timeout=20)
                self.assertEqual(process.returncode, 0, output + error)
                self.assertEqual(json.loads(stats.read_text())["training_sample_exposures"], 64)
                self.assertFalse(socket_path.exists())
            finally:
                connection.close()
                if process.poll() is None:
                    process.terminate()
                    process.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
