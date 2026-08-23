#!/usr/bin/env python3
"""Synthetic regression tests for same-H analysis and 20M comparison mode."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
PYTHON = HERE.parent / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


analysis = module("compare_offline_live", PYTHON / "compare_offline_live.py")
regression = module("validate_20m_backward_regression", HERE / "validate_20m_backward_regression.py")
fairness = module("validate_offline_fairness", HERE / "validate_offline_fairness.py")


class OfflineFairnessAnalysisTest(unittest.TestCase):
    def row(self, hidden, budget, tag, ipc, coverage, pressure, act):
        return {
            "hidden_size": hidden, "instruction_budget": budget,
            "budget_tag": tag, "seed": 7, "ipc": ipc,
            "coverage": coverage, "l2_load_miss_rate": 0.20 - coverage / 10,
            "requests_per_l2_load": pressure,
            "student_heldout_act_rate": act, "timeliness": 0.8,
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
        result = analysis.offline_equivalence_rows(rows)
        h8_candidate = next(row for row in result if row["hidden_size"] == 8 and row["budget_tag"] == "i250k")
        h16_candidate = next(row for row in result if row["hidden_size"] == 16 and row["budget_tag"] == "i100k")
        self.assertAlmostEqual(h8_candidate["relative_ipc_difference_vs_same_h_20m"], 0.08)
        self.assertAlmostEqual(h16_candidate["relative_ipc_difference_vs_same_h_20m"], 0.05)
        self.assertFalse(h8_candidate["within_plus_minus_0_5_percent_20m_ipc"])
        self.assertFalse(h16_candidate["stable_0_5_percent_ipc_plateau_member"])
        for hidden in (8, 16):
            plateau = next(row for row in result if row["hidden_size"] == hidden and row["budget_tag"] == "i1m")
            self.assertTrue(plateau["stable_0_5_percent_ipc_plateau_start"])

    def test_missing_old_artifacts_is_not_exact_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "old"
            new = root / "new"
            old.mkdir()
            (new / "points/h8/i20m/seed7").mkdir(parents=True)
            result = regression.compare_hidden(old, new, 8, 0.005)
            self.assertEqual(result["mode"], "historical_metric_regression_only")
            self.assertEqual(result["status"], "INCOMPLETE")

    def test_fairness_audit_reports_missing_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            point = run / "points/h8/i20m/seed7"
            prefix = run / "training_prefixes/i20m"
            offline = point / "offline"
            offline.mkdir(parents=True)
            prefix.mkdir(parents=True)
            stream = prefix / "train.csv"
            stream.write_text("trace,demand_idx,pc,line,pc_line_occ\n")
            (prefix / "training_manifest.json").write_text(json.dumps({
                "trace": "602.gcc_s-734B", "instruction_budget": 20000000,
                "budget_tag": "i20m", "training_warmup_instructions": 0,
                "training_simulation_instructions": 20000000,
                "collection_semantics": "trace_start_to_exact_retired_instruction_budget",
                "training_stream": str(stream),
            }))
            metadata = {
                "trace": "602.gcc_s-734B", "hidden_size": 8,
                "instruction_budget": 20000000, "budget_tag": "i20m",
                "seed": 7, "model_revision": "compact_shared_pc_hurdle_delta_v9",
                "epochs": 12, "chunk_length": 256, "optimizer": "Adam",
                "learning_rate": 0.002,
            }
            path = point / "point_metadata.json"
            path.write_text(json.dumps(metadata))
            encoder, router = fairness.source_identities()
            result = fairness.audit_point(
                path, fairness.load(fairness.CONTRACT_PATH), encoder, router
            )
            self.assertEqual(result["status"], "INCOMPLETE")
            self.assertNotIn("FAIL", [item["status"] for item in result["checks"].values()])


if __name__ == "__main__":
    unittest.main()
