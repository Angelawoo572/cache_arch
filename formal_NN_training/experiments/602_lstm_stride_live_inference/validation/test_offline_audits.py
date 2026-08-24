#!/usr/bin/env python3
"""Source-level tests for fairness and backward-regression helpers."""

import gzip
import sys
import tempfile
import unittest
from pathlib import Path


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
            checks, 49000000, 25000000, 25000000, "replay.log"
        )
        self.assertEqual(checks[0]["status"], "FAIL")

    def test_reference_metric_tolerance_does_not_imply_artifact_identity(self):
        rows = compare_mapping(
            {"coverage": 0.40}, {"coverage": 0.401},
            {"coverage": 0.02},
        )
        self.assertEqual(rows[0]["status"], "PASS")
        self.assertAlmostEqual(rows[0]["delta_new_minus_old"], 0.001)


if __name__ == "__main__":
    unittest.main()
