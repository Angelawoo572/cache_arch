"""Focused protocol/table checks; no simulated performance workload."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

EXP=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('gate_runner_test',EXP/'run_gate.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)

class GateRunnerTests(unittest.TestCase):
    def test_matrix_has_exact_formal_scope(self):
        rows=gate.matrix()
        self.assertEqual(len(rows),71)
        self.assertEqual(sum(r['execution']=='fresh' for r in rows),64)
        self.assertEqual(sum(r['execution']=='compatible_reference' for r in rows),7)
        self.assertEqual(len({r['run_id'] for r in rows}),71)
        self.assertTrue(all(r['instructions']==25000000 and r['warmup']==5000000 and r['history']==5000000 for r in rows))
        self.assertEqual({r['skip'] for r in rows},{20000000,70000000})

    def test_learned_and_payload_controls_are_paired(self):
        rows=gate.matrix()
        learned=[r for r in rows if r['arm']=='nn' and not r['macs'] and not r['silent']]
        payload=[r for r in rows if r['arm']=='nn_gate_fixed_plus2']
        self.assertEqual(len(learned),24);self.assertEqual(len(payload),12)
        for row in payload:
            peer=[r for r in learned if all(r[k]==row[k] for k in ('roi','seed','budget','filter_current'))]
            self.assertEqual(len(peer),1)
            self.assertEqual(row['method'],'frozen');self.assertEqual(row['payload'],'fixed_plus2')
        finite=[r for r in rows if r['macs']]
        self.assertEqual(len(finite),4)
        self.assertTrue(all(r['seed']==7 and r['budget']==2363 and r['arm']=='nn' for r in finite))

    def test_top64_frequency_tie_is_unsigned_pc_ascending(self):
        counts={pc:(2,pc%3,0) for pc in range(1,67)}
        counts[1000]=(3,1,0)
        text,gates=gate.make_table(counts)
        self.assertEqual(list(gates),[1000]+list(range(1,64)))
        self.assertNotIn(64,gates)
        self.assertEqual(gates[2],1)  # 2 positives / 2 events
        self.assertEqual(gates[1],1)  # exactly 1/2: fixed >=.5 rule
        self.assertEqual(gates[3],0)

    def test_table_roundtrip_and_unseen_silent(self):
        counts={18446744073709551615:(10,5,0),4:(10,4,0)}
        text,gates=gate.make_table(counts)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'gate.tsv';path.write_text(text)
            self.assertEqual(gate.read_gate_table(path),{4:(0,10,4),18446744073709551615:(1,10,5)})
        self.assertEqual(gates.get(99,0),0)

    def test_reject_incorrect_majority_or_duplicate_table(self):
        for text in ('pc gate count positives\n1 1 10 4\n',
                     'pc gate count positives\n1 1 2 1\n1 1 2 1\n',
                     'pc gate count positives\n1 0 0 0\n'):
            with tempfile.TemporaryDirectory() as td:
                path=Path(td)/'gate.tsv';path.write_text(text)
                with self.assertRaises(RuntimeError):gate.read_gate_table(path)

    def test_postprocessing_is_outside_source_freeze(self):
        own=[p.name for p in gate.source_files() if p.parent==EXP]
        self.assertEqual(set(own),{'run_gate.py','score_pc.cc','gate_config.json'})

if __name__=='__main__':unittest.main()
