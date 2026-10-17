#!/usr/bin/env python3
"""Regression checks for 20M equivalence and stable-plateau terminology."""

import sys
import unittest
from pathlib import Path


PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(PYTHON_DIR))

from analysis_policy import (  # noqa: E402
    reference_20m,
    smallest_reaching_at_least,
    smallest_within,
    stable_plateau,
)
from compare_offline_live import conclusions_for_hidden  # noqa: E402
from offline_sufficiency import build_analysis, point_differences  # noqa: E402


def row(tag, budget, ipc):
    return {"budget_tag": tag, "instruction_budget": budget, "ipc": ipc}


class AnalysisPolicyTest(unittest.TestCase):
    def test_high_point_reaches_but_is_not_within(self):
        rows = [row("i100k", 100000, 1.03), row("i20m", 20000000, 1.0)]
        reference = reference_20m(rows)
        self.assertEqual(
            smallest_reaching_at_least(rows, reference, 0.999)["budget_tag"],
            "i100k",
        )
        self.assertEqual(
            smallest_within(rows, reference, 0.001)["budget_tag"], "i20m"
        )

    def test_spike_cannot_start_plateau(self):
        rows = [
            row("i100k", 100000, 1.03),
            row("i250k", 250000, 1.004),
            row("i500k", 500000, 1.003),
            row("i20m", 20000000, 1.0),
        ]
        reference = reference_20m(rows)
        candidate, suffix = stable_plateau(rows, reference, 0.005)
        self.assertEqual(candidate["budget_tag"], "i250k")
        self.assertEqual(
            [item["budget_tag"] for item in suffix],
            ["i250k", "i500k", "i20m"],
        )

    def test_later_instability_rejects_early_candidate(self):
        rows = [
            row("i100k", 100000, 1.001),
            row("i250k", 250000, 1.02),
            row("i20m", 20000000, 1.0),
        ]
        candidate, suffix = stable_plateau(
            rows, reference_20m(rows), 0.005
        )
        self.assertIsNone(candidate)
        self.assertEqual(suffix, [])

    def test_reference_alone_is_not_a_plateau(self):
        rows = [row("i20m", 20000000, 1.0)]
        candidate, suffix = stable_plateau(
            rows, reference_20m(rows), 0.005
        )
        self.assertIsNone(candidate)
        self.assertEqual(suffix, [])

    def test_generated_conclusions_keep_concepts_separate(self):
        rows = [
            dict(row("i100k", 100000, 1.03), decision_rows=10,
                 silent_rows=5, positive_count_rows=5,
                 offline_replay_entry_count=1, optimizer_steps=1),
            dict(row("i250k", 250000, 0.9995), decision_rows=20,
                 silent_rows=10, positive_count_rows=10,
                 offline_replay_entry_count=1, optimizer_steps=2),
            dict(row("i500k", 500000, 1.02), decision_rows=30,
                 silent_rows=15, positive_count_rows=15,
                 offline_replay_entry_count=1, optimizer_steps=3),
            dict(row("i1m", 1000000, 1.0005), decision_rows=40,
                 silent_rows=20, positive_count_rows=20,
                 offline_replay_entry_count=1, optimizer_steps=4),
            dict(row("i20m", 20000000, 1.0), decision_rows=50,
                 silent_rows=25, positive_count_rows=25,
                 offline_replay_entry_count=1, optimizer_steps=5),
        ]
        result = conclusions_for_hidden(rows, rows)
        self.assertEqual(
            result[
                "smallest_budget_reaching_at_least_99_9_percent_of_20m_offline_ipc"
            ],
            "i100k",
        )
        self.assertEqual(
            result["smallest_within_0.1_percent_20m_offline_ipc"],
            "i250k",
        )
        self.assertEqual(
            result["offline_stable_plateau"]["budget_tag"], "i1m"
        )
        self.assertEqual(
            result["offline_high_performing_small_budget_candidates"][0][
                "budget_tag"
            ],
            "i100k",
        )

    def test_point_differences_use_same_hidden_reference(self):
        rows = [
            dict(row("i1m", 1000000, 0.401), hidden_size=8, seed=7),
            dict(row("i20m", 20000000, 0.400), hidden_size=8, seed=7),
        ]
        result = point_differences(rows)
        self.assertAlmostEqual(
            result[0]["relative_ipc_difference_vs_same_h_20m"], 0.0025
        )
        self.assertEqual(
            result[0]["same_hidden_size_20m_budget_tag"], "i20m"
        )

    def test_h8_and_h16_references_never_cross(self):
        offline = [
            dict(row("i1m", 1000000, 0.401), hidden_size=8, seed=7),
            dict(row("i20m", 20000000, 0.400), hidden_size=8, seed=7),
            dict(row("i1m", 1000000, 0.503), hidden_size=16, seed=7),
            dict(row("i20m", 20000000, 0.500), hidden_size=16, seed=7),
        ]
        analysis = build_analysis(offline, [])
        h8_i1m = next(
            item for item in analysis["offline_points"]
            if item["hidden_size"] == 8 and item["budget_tag"] == "i1m"
        )
        h16_i1m = next(
            item for item in analysis["offline_points"]
            if item["hidden_size"] == 16 and item["budget_tag"] == "i1m"
        )
        self.assertAlmostEqual(
            h8_i1m["relative_ipc_difference_vs_same_h_20m"], 0.0025
        )
        self.assertAlmostEqual(
            h16_i1m["relative_ipc_difference_vs_same_h_20m"], 0.006
        )
        self.assertEqual(
            analysis["comparison_rule"],
            "h8-N versus h8-20M; h16-N versus h16-20M",
        )


if __name__ == "__main__":
    unittest.main()
