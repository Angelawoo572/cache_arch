"""A paired load can arrive as another core's prefetch, merge, or bypass LLC."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from characterize import compare


def write(path, fields, data):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields.split())
        writer.writerows(data)


class CharacterizationTest(unittest.TestCase):
    def populate(self, root, policy, mismatched=False):
        run = root / ("case-" + policy)
        run.mkdir()
        (run/"run.json").write_text(json.dumps(dict(run_id=run.name, workloads=["micro"], traces=["micro"],
              warmup=0,roi=5,skip=0,group="synthetic")))
        data = []
        for i in range(1,6):
            va = i + int(mismatched and i == 2)
            for level,stage,hit,t in (("CORE","trace",-1,0), ("L1","lookup",0,1), ("L2","lookup",0,2)):
                data.append([0,i,0,8,va,va,level,stage,10*i+t,hit,0,0,0])
            if policy=="stride" and i==5:
                data.append([0,i,0,8,va,va,"LLC","merged",10*i+3,-1,0,0,0])
                continue
            data.append([0,i,0,8,va,va,"LLC","lookup",10*i+4,int(policy=="stride" and i==3),1,2,3])
        write(run/"loads.csv", "cpu instr_id operand pc virtual_line physical_line level stage cycle hit mshr rq pq", data)
        events=[]
        if policy=="stride":
            for stage,t in (("proposed",13),("accepted",13),("dispatch",15),("fill",30),("timely",34)):
                events.append([1,0,1,0,8,1,3,t,stage,"",0 if stage=="timely" else -1,
                               3 if stage=="timely" else -1,0 if stage=="timely" else -1])
        write(run/"candidates.csv","id cpu instr_id operand pc base target cycle stage reason demand_cpu demand_instr_id demand_operand",events)
        write(run/"cores.csv","cpu warmup instructions cycles",[(0,0,5,56)])
        write(run/"stats.csv","key value",[("dram_demand_bytes",192)])
        write(run/"head.csv","cpu instr_id operand cycles",[(0,4,0,3)])
        return run

    def test_separate_lifecycle_and_joined_outcomes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = compare(self.populate(root, "nopf"), self.populate(root, "stride"))
            core = data["core_performance"][0]
            self.assertEqual((core["miss_to_hit"],core["miss_to_miss"],core["miss_to_first_pf_hit"]),(1,3,1))
            self.assertEqual(core["paired_coverage"], 0.2)
            self.assertEqual(core["prefetch_accuracy"], 1)
            self.assertTrue(any(r["after"]=="LLC_merged" and r["count"]==1 for r in data["outcomes"]))
            self.assertTrue(any(r["visibility"]=="prediction_instant" for r in data["examples"]))
            self.assertTrue(any(r["matched_pairs"]==1 for r in data["hit_context"]))

    def test_virtual_line_mismatch_is_fatal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                compare(self.populate(root,"nopf"),self.populate(root,"stride",mismatched=True))


if __name__=="__main__":
    unittest.main()
