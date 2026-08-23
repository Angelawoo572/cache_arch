#!/usr/bin/env python3
"""Synthetic regression tests for same-H and provenance audit behavior."""

import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
PYTHON = HERE.parent / "python"
for directory in (HERE, PYTHON):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from offline_sufficiency import build_analysis  # noqa: E402
from validate_20m_regression import compare_artifact  # noqa: E402
from validate_offline_fairness import equal_number  # noqa: E402


class OfflineFairnessAnalysisTest(unittest.TestCase):
    @staticmethod
    def row(hidden, budget, tag, ipc, coverage, pressure, act):
        return {
            "hidden_size": hidden,
            "instruction_budget": budget,
            "budget_tag": tag,
            "seed": 7,
            "ipc": ipc,
            "coverage": coverage,
            "l2_load_miss_rate": 0.20 - coverage / 10,
            "requests_per_l2_load": pressure,
            "student_heldout_act_rate": act,
            "timeliness": 0.8,
            "offline_replay_entry_count": int(1000 * pressure),
        }

    def test_same_hidden_reference_and_stable_plateau(self):
        rows = [
            self.row(8, 250000, "i250k", 1.08, 0.30, 0.50, 0.40),
            self.row(8, 1000000, "i1m", 1.002, 0.20, 0.30, 0.25),
            self.row(8, 20000000, "i20m", 1.0, 0.20, 0.30, 0.25),
            self.row(16, 100000, "i100k", 2.10, 0.40, 0.60, 0.50),
            self.row(16, 1000000, "i1m", 2.004, 0.22, 0.31, 0.26),
            self.row(16, 20000000, "i20m", 2.0, 0.22, 0.31, 0.26),
        ]
        analysis = build_analysis(rows, [])
        points = analysis["offline_points"]
        h8_candidate = next(
            row for row in points
            if row["hidden_size"] == 8 and row["budget_tag"] == "i250k"
        )
        h16_candidate = next(
            row for row in points
            if row["hidden_size"] == 16 and row["budget_tag"] == "i100k"
        )
        self.assertAlmostEqual(
            h8_candidate["relative_ipc_difference_vs_same_h_20m"], 0.08
        )
        self.assertAlmostEqual(
            h16_candidate["relative_ipc_difference_vs_same_h_20m"], 0.05
        )
        self.assertFalse(
            h8_candidate["stable_0_5_percent_ipc_plateau_member"]
        )
        self.assertFalse(
            h16_candidate["stable_0_5_percent_ipc_plateau_member"]
        )
        for hidden in (8, 16):
            plateau = next(
                row for row in points
                if row["hidden_size"] == hidden
                and row["budget_tag"] == "i1m"
            )
            self.assertTrue(
                plateau["stable_0_5_percent_ipc_plateau_start"]
            )

    def test_missing_artifact_never_becomes_exact_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comparison = compare_artifact(
                "stream",
                root / "old.csv.gz",
                root / "new.csv.gz",
                content=True,
                exact_required=True,
            )
            self.assertEqual(comparison["status"], "UNAVAILABLE")
            self.assertIsNone(comparison["same_identity"])

    def test_prefix_local_class_weight_rounding_is_tolerant(self):
        expected = 1000.0 / (2.0 * 157.0)
        self.assertTrue(equal_number(expected, expected + 1e-9))
        self.assertFalse(equal_number(expected, expected + 0.1))


if __name__ == "__main__":
    unittest.main()
