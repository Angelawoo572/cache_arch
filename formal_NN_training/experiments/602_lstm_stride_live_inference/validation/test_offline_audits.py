#!/usr/bin/env python3
"""Source-level tests for fairness and backward-regression helpers."""

import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


VALIDATION_DIR = Path(__file__).resolve().parent
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from provenance_utils import (  # noqa: E402
    artifact,
    authoritative_source_hashes,
)
from validate_20m_regression import (  # noqa: E402
    compare_artifact,
    compare_mapping,
    metrics_from_log,
)
from validate_offline_fairness import (  # noqa: E402
    add_instruction_counter_check,
    equal_number,
)


ROOT = Path(__file__).resolve().parents[4]


class OfflineAuditTest(unittest.TestCase):
    def test_authoritative_hashes_are_stable_sha256_values(self):
        first = authoritative_source_hashes(ROOT)
        second = authoritative_source_hashes(ROOT)
        self.assertEqual(first, second)
        for value in first.values():
            self.assertEqual(len(value), 64)
            int(value, 16)

    def test_prefix_local_class_weight_rounding_is_tolerant(self):
        expected = 1000.0 / (2.0 * 157.0)
        self.assertTrue(equal_number(expected, expected + 1e-9))
        self.assertFalse(equal_number(expected, expected + 0.1))

    def test_gzip_content_identity_ignores_container_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.csv.gz"
            right = root / "right.csv.gz"
            payload = b"pc,line\n1,2\n"
            with gzip.GzipFile(str(left), "wb", mtime=0) as handle:
                handle.write(payload)
            with gzip.GzipFile(str(right), "wb", mtime=9) as handle:
                handle.write(payload)
            left_identity = artifact(left, gzip_content=True)
            right_identity = artifact(right, gzip_content=True)
            self.assertNotEqual(
                left_identity["sha256"], right_identity["sha256"]
            )
            self.assertEqual(
                left_identity["content_sha256"],
                right_identity["content_sha256"],
            )
            comparison = compare_artifact(
                "stream", left, right, content=True, exact_required=True
            )
            self.assertEqual(comparison["status"], "PASS")

    def test_champsim_roi_and_cumulative_instruction_counters_are_distinct(self):
        for observed, scope in (
            (25000000, "measurement_window_only"),
            (25000003, "measurement_window_only"),
            (50000000, "cumulative_warmup_plus_measurement"),
        ):
            checks = []
            add_instruction_counter_check(
                checks, observed, 25000000, 25000000, "replay.log"
            )
            self.assertEqual(checks[0]["status"], "PASS")
            self.assertEqual(checks[0]["observed"]["counter_scope"], scope)
        checks = []
        add_instruction_counter_check(
            checks, 25000003, 25000000, 25000000, "replay.log"
        )
        self.assertEqual(
            checks[0]["observed"]["retirement_boundary_overshoot"], 3
        )
        checks = []
        add_instruction_counter_check(
            checks, 25000005, 25000000, 25000000, "replay.log"
        )
        self.assertEqual(checks[0]["status"], "FAIL")

    def test_reference_coverage_uses_no_prefetch_l2_miss_denominator(self):
        parsed = {
            "ipc": 0.4,
            "cycles": 62500000,
            "l2_loads": 1000,
            "l2_load_miss": 200,
            "l2_load_miss_rate": 0.2,
            "request_per_l2_load": 0.25,
            "accuracy": 0.5,
            "selected_accuracy": 0.5,
            "timeliness": 0.75,
            "pf_requested": 250,
            "pf_issued": 240,
            "pf_useful": 50,
            "pf_late": 5,
        }
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "run.log"
            log.write_text("synthetic\n")
            with patch("validate_20m_regression.parse_log", return_value=parsed):
                baseline = metrics_from_log(log)
                stride = metrics_from_log(log, baseline)
        self.assertEqual(baseline["l2_load_miss"], 200)
        self.assertAlmostEqual(stride["coverage"], 0.25)

    def test_reference_metric_tolerance_does_not_imply_artifact_identity(self):
        rows = compare_mapping(
            {"coverage": 0.40}, {"coverage": 0.401},
            {"coverage": 0.02},
        )
        self.assertEqual(rows[0]["status"], "PASS")
        self.assertAlmostEqual(rows[0]["delta_new_minus_old"], 0.001)

    def test_all_valid_live_selection_can_split_disjoint_hidden_sizes(self):
        script = (
            ROOT / "formal_NN_training/experiments/"
            "602_lstm_stride_live_inference/linux/run_live_sweep.sh"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "fake_champsim"
            binary.write_text("#!/usr/bin/env bash\nexit 0\n")
            binary.chmod(0o755)
            trace = root / "trace.xz"
            trace.write_bytes(b"synthetic")
            run_dir = root / "run"
            for hidden in (8, 16):
                point = run_dir / "points/h{}/i1m/seed7".format(hidden)
                export = point / "export"
                export.mkdir(parents=True)
                (point / "point_metadata.json").write_text(json.dumps({
                    "hidden_size": hidden,
                    "budget_tag": "i1m",
                    "seed": 7,
                    "status": "offline_replay_complete",
                }))
                (export / "model.bin").write_bytes(b"frozen")
                (export / "model_metadata.json").write_text(json.dumps({
                    "weights_frozen": True,
                    "format_version": 1,
                }))
            env = os.environ.copy()
            env.update({
                "RUN_DIR": str(run_dir),
                "BIN": str(binary),
                "TRACE_FILE": str(trace),
                "LIVE_ALL_VALID": "1",
                "HIDDEN_SIZES": "8",
                "DRY_RUN": "1",
                "STATE_MODE": "parity",
                "RUN_MODE": "live",
            })
            completed = subprocess.run(
                ["bash", str(script), "run"], env=env, check=True,
                text=True, capture_output=True,
            )
        self.assertIn("points/h8/i1m/seed7", completed.stdout)
        self.assertNotIn("points/h16/i1m/seed7", completed.stdout)


if __name__ == "__main__":
    unittest.main()
