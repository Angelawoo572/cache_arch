#!/usr/bin/env python3
"""Focused correctness checks; these do not run a simulator or training pilot."""
import ast
import copy
import importlib.util
import math
import sys
import unittest
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[2]
sys.path.insert(0, str(EXP / "python"))
from export_live_model import validate_training_identity

TRAINER = ROOT / "formal_NN_training/experiments/602_offline_lstm_stride/python/train_and_offline_infer.py"


def numerical_functions():
    tree = ast.parse(TRAINER.read_text())
    wanted = {"as_int", "_positive_counts_from_log_mean"}
    selected = ast.Module(body=[node for node in tree.body
                               if isinstance(node, ast.FunctionDef)
                               and node.name in wanted], type_ignores=[])
    namespace = dict(np=np, math=math, Decimal=Decimal, InvalidOperation=InvalidOperation)
    exec(compile(selected, str(TRAINER), "exec"), namespace)
    return namespace


class TrainingIdentity(unittest.TestCase):
    def setUp(self):
        self.config = {"seed": 11, "emit_loss": "unweighted", "optimizer_steps": 228}
        self.checkpoint = {"seed": 11, "emit_loss": "unweighted", "trace": "602",
                           "training_config": self.config}
        self.metadata = copy.deepcopy(self.checkpoint)

    def test_new_checkpoint_identity(self):
        value = validate_training_identity(self.checkpoint, self.metadata, 11)
        self.assertEqual(value["emit_loss"], "unweighted")
        self.assertFalse(value["legacy_checkpoint"])

    def test_metadata_cannot_relabel_seed(self):
        with self.assertRaisesRegex(RuntimeError, "original training metadata"):
            validate_training_identity(self.checkpoint, self.metadata, 7)

    def test_checkpoint_cannot_relabel_seed(self):
        self.metadata["seed"] = 7
        with self.assertRaisesRegex(RuntimeError, "checkpoint training seed"):
            validate_training_identity(self.checkpoint, self.metadata, 7)

    def test_loss_mismatch_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "requested emit loss"):
            validate_training_identity(self.checkpoint, self.metadata, 11, "balanced")

    def test_config_mismatch_is_rejected(self):
        self.metadata["training_config"]["optimizer_steps"] = 1
        with self.assertRaisesRegex(RuntimeError, "training configurations"):
            validate_training_identity(self.checkpoint, self.metadata, 11)

    def test_legacy_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(RuntimeError, "legacy checkpoint"):
            validate_training_identity({}, {"seed": 7}, 7)

    def test_legacy_requires_original_loss_evidence(self):
        with self.assertRaisesRegex(RuntimeError, "cannot be established"):
            validate_training_identity({}, {"seed": 7}, 7, allow_legacy=True)
        value = validate_training_identity(
            {}, {"seed": 7, "data_derived_class_balancing_used": True},
            7, allow_legacy=True,
        )
        self.assertTrue(value["legacy_checkpoint"])
        self.assertEqual(value["emit_loss"], "balanced")


class NumericBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fn = numerical_functions()

    def test_integer_csv_preserves_uint64(self):
        parse = self.fn["as_int"]
        self.assertEqual(parse("18446744073709551615"), (1 << 64) - 1)
        self.assertEqual(parse("1.8446744073709551615e19"), (1 << 64) - 1)
        with self.assertRaises(ValueError):
            parse("1.25")

    def test_count_decode_and_domain(self):
        decode = self.fn["_positive_counts_from_log_mean"]
        self.assertEqual(decode(np.log([1., 2., 9., 257.])).tolist(), [1, 2, 9, 257])
        self.assertEqual(decode([-1000.]).tolist(), [1])
        for value in (1000., float("nan"), float("inf")):
            with self.assertRaises(RuntimeError):
                decode([value])


if __name__ == "__main__":
    unittest.main()
