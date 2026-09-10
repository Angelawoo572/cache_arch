#!/usr/bin/env python3
"""Scoped driver regression: frozen selection and safe completed-result reuse."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

EXP = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location('dynamic602_driver', EXP/'run.py')
driver = importlib.util.module_from_spec(loader)
loader.loader.exec_module(driver)


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='dynamic602_driver_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = dict(driver.CONFIG, raw=str(self.root/'new'), reference_runs=str(self.root/'references'))
        self.config_patch = patch.object(driver, 'CONFIG', self.config)
        self.raw_patch = patch.object(driver, 'RAW', Path(self.config['raw']))
        self.config_patch.start(); self.raw_patch.start()
        self.addCleanup(self.config_patch.stop); self.addCleanup(self.raw_patch.stop)

    def completed(self, spec, root=None):
        expected = driver.execution_spec(spec)
        path = (root or Path(self.config['reference_runs']))/spec['run_id']
        path.mkdir(parents=True)
        # This represents the older frozen run's metadata, with no new runtime fields.
        record = dict(expected, status='complete', exit_code=0, worker_pid=None)
        record.pop('model_binary', None); record.pop('output_limit')
        driver.write(path/'run.json', record)
        method = 'frozen' if spec['arm'].startswith('frozen_') else spec['arm']
        counts = {'h':spec['h'], 'macs_per_cycle':spec['macs'], 'state_capacity':spec['state_capacity'],
                  'completed_updates':0, 'published_versions':0, 'teacher_calls':0}
        log = ('trace_0 {trace}\nonline602_trace_skip_records {skip_records} reader_record_bytes 64 cpu 0\n'
               'online602 method '+method+' macs {macs} state_capacity {state_capacity} output_limit {output_limit}\n'
               'warmup_instructions {warmup_instructions}\nsimulation_instructions {simulation_instructions}\n'
               'Finished CPU 0 instructions: {simulation_instructions} cycles: 69000000\n'
               'ChampSim completed all CPUs\nonline602_final ').format(**expected)+json.dumps(counts)+'\n'
        (path/'run.log').write_text(log)
        return path, expected

    def test_thirteen_specs_contain_only_frozen_and_matched_conventional_arms(self):
        specs = driver.specs()
        self.assertEqual(len(specs), 13)
        self.assertEqual(len({spec['run_id'] for spec in specs}), 13)
        self.assertEqual({spec['arm'] for spec in specs}, {'none','stride','frozen_i1m','frozen_i20m'})
        self.assertEqual(sum(spec['protocol']=='historical' for spec in specs), 4)
        self.assertEqual({spec['macs'] for spec in specs if spec['state_capacity']==64}, {0,4,16})
        self.assertTrue(all(spec['seed']==7 for spec in specs))

    def test_all_compatible_references_skip_without_any_process_or_new_output(self):
        specs = driver.specs()
        for spec in specs:
            path, expected = self.completed(spec)
            self.assertTrue(driver.completed_compatible(path, expected), spec['run_id'])
        with patch.object(driver.subprocess, 'Popen', side_effect=AssertionError('must not launch')), redirect_stdout(io.StringIO()):
            for spec in specs:driver.run_one(spec)
        self.assertFalse(driver.RAW.exists())

    def test_metadata_and_actual_log_mismatches_and_incomplete_runs_are_rejected(self):
        spec = next(s for s in driver.specs() if s['h']==8 and s['macs']==4)
        path, expected = self.completed(spec)
        metadata = json.loads((path/'run.json').read_text())
        for field in ('protocol','skip_records','warmup_instructions','simulation_instructions',
                      'h','hidden_size','seed','state_capacity','macs','trace','checkpoint'):
            mutated = dict(metadata); mutated[field] = str(mutated[field])+'-mismatch'
            driver.write(path/'run.json', mutated)
            self.assertFalse(driver.completed_compatible(path, expected), field)
        driver.write(path/'run.json', metadata)
        original = (path/'run.log').read_text()
        for old,new in [('skip_records 25000000','skip_records 0'),('method frozen','method online'),
                        ('macs 4','macs 16'),('output_limit 32','output_limit 2'),
                        ('instructions: 25000000','instructions: 24999999'),
                        ('ChampSim completed all CPUs','incomplete'),
                        ('"completed_updates": 0','"completed_updates": 1')]:
            (path/'run.log').write_text(original.replace(old,new))
            self.assertFalse(driver.completed_compatible(path, expected), (old,new))
        (path/'run.log').write_text(original)
        driver.write(path/'run.json', dict(metadata,status='failed'))
        self.assertFalse(driver.completed_compatible(path, expected))

    def test_existing_failed_output_has_priority_over_old_reference_and_is_preserved(self):
        spec = driver.specs()[0]
        self.completed(spec)
        path, _ = self.completed(spec, driver.RAW/'final')
        record = json.loads((path/'run.json').read_text());record['status']='failed'
        driver.write(path/'run.json',record)
        original = (path/'run.log').read_text()
        with self.assertRaises(FileNotFoundError):
            # No build exists in this fixture. Reaching its lookup proves this is
            # a fresh attempt, not a silent substitution of the older reference.
            driver.run_one(spec)
        saved = list((driver.RAW/'failed').glob(spec['run_id']+'-*'))
        self.assertEqual(len(saved),1)
        self.assertEqual((saved[0]/'run.log').read_text(),original)
        self.assertEqual(json.loads((saved[0]/'run.json').read_text())['status'],'failed')
        self.assertTrue((Path(self.config['reference_runs'])/spec['run_id']/'run.log').is_file())


if __name__=='__main__':
    unittest.main(verbosity=2)
