'''Correctness checks for anchored forward generation; no simulator ROI runs.'''
from pathlib import Path
import importlib.util
import json
import os
import shutil
import tempfile
import unittest

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[2]
SPEC = importlib.util.spec_from_file_location('checked_prepare_forward', EXP/'runtime/prepare_forward.py')
FORWARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FORWARD)


class PrepareForwardTests(unittest.TestCase):
    def setUp(self):
        parent = os.environ.get('TMPDIR')
        if not parent:
            self.fail('Set TMPDIR to a cache-owned runtime directory; do not use system /tmp')
        self.temporary = tempfile.TemporaryDirectory(prefix='forward_prepare_', dir=parent)
        self.stage = Path(self.temporary.name)
        self.repo = self.stage/'source'
        target = self.repo/FORWARD.SOURCE
        target.mkdir(parents=True)
        for name in FORWARD.NAMES:
            shutil.copy2(ROOT/FORWARD.SOURCE/name, target/name)
        self.sim = self.stage/'simulator'
        (self.sim/'inc').mkdir(parents=True)
        (self.sim/'prefetcher').mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def test_generate_and_verify_all_four_bytes_and_marker(self):
        result = FORWARD.prepare(self.repo, self.sim)
        self.assertEqual(result['status'], 'PASS')
        self.assertTrue(result['marker_present'])
        self.assertEqual(len(result['artifacts']), 4)
        self.assertEqual(FORWARD.verify_prepared(self.repo, self.sim, True), result)

    def test_missing_and_duplicate_anchors_fail_before_any_write(self):
        header = self.repo/FORWARD.SOURCE/'stride_lstm_runtime.h'
        original = header.read_text()
        for changed in [original.replace('  void Reset();',''), original+'\n  void Reset();\n']:
            header.write_text(changed)
            with self.assertRaisesRegex(RuntimeError, 'exactly one source anchor'):
                FORWARD.prepare(self.repo, self.sim)
            self.assertFalse(any(p.is_file() for p in self.sim.rglob('*')))

    def test_source_already_extended_is_rejected(self):
        header = self.repo/FORWARD.SOURCE/'stride_lstm_runtime.h'
        header.write_text(header.read_text()+'\n// InferState( already generated\n')
        with self.assertRaisesRegex(RuntimeError, 'already extended'):
            FORWARD.prepare(self.repo, self.sim)
        self.assertFalse(any(p.is_file() for p in self.sim.rglob('*')))

    def test_explicit_count_guard_variants_and_rejection(self):
        source = (self.repo/FORWARD.SOURCE/'stride_lstm_runtime.cc').read_text()
        self.assertEqual(FORWARD.count_domain(source), source)
        legacy='raw_count > static_cast<double>(\n            std::numeric_limits<std::uint64_t>::max())'
        inline='std::nearbyint(raw_count) >= 9223372036854775808.0'
        self.assertEqual(FORWARD.count_domain(legacy), inline)
        self.assertEqual(FORWARD.count_domain(inline), inline)
        for bad in ['unrecognized count cast', inline+' '+inline, legacy+' '+inline]:
            with self.assertRaisesRegex(RuntimeError, 'supported representation'):
                FORWARD.count_domain(bad)

    def test_tampered_artifact_is_rejected_without_rewriting(self):
        FORWARD.prepare(self.repo, self.sim)
        target=self.sim/'prefetcher/stride_lstm_runtime.cc'
        changed=target.read_bytes()+b'\n// tampered\n'
        target.write_bytes(changed)
        with self.assertRaisesRegex(RuntimeError, 'byte mismatch'):
            FORWARD.verify_prepared(self.repo, self.sim)
        self.assertEqual(target.read_bytes(), changed)

    def test_legacy_marker_absence_is_explicit_and_bad_marker_rejected(self):
        FORWARD.prepare(self.repo, self.sim)
        marker=self.sim/FORWARD.MARKER
        marker.unlink()
        self.assertFalse(FORWARD.verify_prepared(self.repo, self.sim)['marker_present'])
        with self.assertRaisesRegex(RuntimeError, 'marker is missing'):
            FORWARD.verify_prepared(self.repo, self.sim, True)
        marker.write_text(json.dumps({'protocol':'unexpected'}))
        with self.assertRaisesRegex(RuntimeError, 'marker does not match'):
            FORWARD.verify_prepared(self.repo, self.sim)


if __name__ == '__main__':
    unittest.main(verbosity=2)
