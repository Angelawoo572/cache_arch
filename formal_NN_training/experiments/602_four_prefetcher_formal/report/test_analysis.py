"""Focused scientific regression checks; fixtures are metadata, not simulator pilots."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
import json
import struct

MODULE=Path(__file__).resolve().parents[1]/"analyze.py"
loader=importlib.util.spec_from_file_location("four602_analysis",MODULE)
A=importlib.util.module_from_spec(loader);loader.loader.exec_module(A)


def cohort(enqueued=0,timely_load=0,late_load=0,unused=0,**extra):
    c={k:0 for k in A.COHORT_FIELDS}
    c.update(enqueued=enqueued,timely=timely_load,late=late_load,timely_load=timely_load,late_load=late_load,unused=unused,**extra)
    return c


def spec(policy="spp",method="student"):
    return dict(run_id="fixture",protocol="new",stage="test",roi=1,policy=policy,method=method,mode="full",budget=0,seed=7,macs=0,
                skip=70000000,warmup=5000000,instructions=100,measurement_records=[75000000,75000100],page_policy="cross_page",build_id="same")


def snapshot(llc_available=True):
    x=dict(instructions=100,cycles=50,NL=30,M=8,LLC_LOAD_misses=4,
           request_gate_allowed=10,request_gate_generated=12,
           four602_counters=dict(demand_callbacks=20,fill_callbacks=5,completed_updates=25,inference_macs=10000,inference_nonlinears=500),
           llc_only_lifecycle_accuracy_available=llc_available,llc_target_cohort=cohort(3,1,0,2),
           dram_total_bus_bytes=128)
    x.update({"cohort_"+k:v for k,v in cohort(5,1,1,3).items()})
    return x


class ScientificSemantics(unittest.TestCase):
    def test_independent_target_denominators_and_incomplete_LLC(self):
        s=spec()
        r=dict(spec=s,summary=snapshot(False),dir=Path("/not-a-real-run"))
        b=dict(summary=dict(instructions=100,cycles=100,M=20,LLC_LOAD_misses=10),spec=dict(run_id="NoPF"))
        row=A.derive(r,b,Path("/nonexistent"))
        self.assertAlmostEqual(row["l2_target_LOAD_accuracy_enqueued"],2/5)
        self.assertAlmostEqual(row["l2_target_LOAD_timeliness"],1/2)
        self.assertAlmostEqual(row["l2_target_LOAD_useful_over_NoPF_misses"],2/20)
        self.assertAlmostEqual(row["net_l2_load_miss_reduction"],12/20)
        self.assertIsNone(row["llc_target_LOAD_accuracy_enqueued"])
        r["summary"]=snapshot(True)
        row=A.derive(r,b,Path("/nonexistent"))
        self.assertAlmostEqual(row["llc_target_LOAD_accuracy_enqueued"],1/3)
        self.assertAlmostEqual(row["llc_target_LOAD_useful_over_NoPF_misses"],1/10)
        self.assertEqual(row["inference_macs"],10000)
        self.assertEqual(row["completed_updates"],25)

    def test_NoPF_matches_common_machine_not_model_state(self):
        a=spec()
        b=dict(a,policy="none",method="none",budget=0,model="different",state_capacity=0)
        self.assertEqual(A.protocol_key(a),A.protocol_key(b))
        for key in ("build_id","skip","warmup","instructions","page_policy"):
            changed=dict(b);changed[key]="different"
            self.assertNotEqual(A.protocol_key(a),A.protocol_key(changed))

    def test_LLC_classifier_full_group_identity(self):
        c=cohort(2,1,0,1)
        key="phase_0/slot_0/current_0/duplicate_0/teacher_3"
        item=dict(arm="fixture",fill_scope="llc_only",key=key,origin_phase=0,slot=0,current_line=0,list_duplicate=0,teacher_category=3,
                  cache_attempted=2,allowed=2,current_suppressed=0,page_suppressed=0,budget_suppressed=0,
                  cache_accepted_nonmerged=2,cache_merged=0,cache_rejected=0,pq_enqueued=2,
                  actual_attempt_phase_counts={"0":2},actual_allowed_phase_counts={"0":2},l2_cohort=None)
        x=dict(mechanism_classified_requests=[item],llc_target_classified_cohorts=[dict(group="fixture/llc_only/"+key,cohort=c)],
               llc_target_cohort=c,request_gate_generated=2,request_gate_allowed=2,request_cache_enqueued=2,
               request_cache_merged=0,request_cache_rejected=0,request_cache_pq_inserted=2,
               request_gate_current_suppressed=0,request_gate_page_suppressed=0,request_gate_budget_suppressed=0)
        x.update({"cohort_"+k:v for k,v in cohort().items()})
        audit=A.Audit()
        rows=A.classification_rows(x,spec(),audit)
        self.assertEqual(rows[0]["target_cohort"]["enqueued"],2)
        self.assertEqual(audit.failures,[])
        x["llc_target_cohort"]["unused"]+=1
        audit=A.Audit();A.classification_rows(x,spec(),audit)
        self.assertTrue(audit.failures)

    def test_snapshot_boundary_keeps_actual_overshoot_not_interpolation(self):
        rows=[]
        for i,cy in ((0,0),(6,3),(11,8)):
            rows.append(dict(instructions=i,cycles=cy,demand_callbacks=i,fill_callbacks=0,completed_updates=i,
                             inference_macs=i*10,inference_nonlinears=i,l2_load_misses=i,dram_total_bus_bytes=i*64))
        s=dict(spec(),instructions=10)
        phases=A.phase_progress(rows,s,dict(phase_instructions=5))
        self.assertEqual([p["actual_end_instructions"] for p in phases],[6,11])
        self.assertEqual([p["interval_instructions"] for p in phases],[6,5])
        self.assertEqual(phases[1]["interval_ipc"],1.0)

    def test_paired_intervention_sign_and_request_gap(self):
        base=dict(spec(),ipc=2.0,raw__request_gate_allowed=10,raw__dram_total_bus_bytes=640)
        changed=dict(base,run_id="fixed",mode="fixed_payload",ipc=1.5,raw__request_gate_allowed=12,raw__dram_total_bus_bytes=704)
        pair=A.pair_rows([base,changed])[0]
        self.assertEqual(pair["delta_ipc"],-0.5)
        self.assertEqual(pair["candidate_over_control_ipc"],0.75)
        self.assertEqual(pair["delta_allowed"],2)
        self.assertEqual(pair["delta_dram_bus_bytes"],64)

    def test_Stride_head_K_preserves_existing_decode_cap(self):
        s=spec("stride","stride_student")
        x=dict(nn_completed=1,decoder_dropped_addresses=1,mechanism_raw_generated_addresses=1,mechanism_payload_generated_addresses=1,
               mechanism_distributions_exact={"phase_0_raw_K":{"2":1},"phase_0_raw_emit":{"1":1},
                   "phase_0_raw_delta_slot_0":{"7":1},"phase_0_payload_K":{"1":1},"phase_0_payload_emit":{"1":1},
                   "phase_0_payload_delta_slot_0":{"7":1}})
        audit=A.Audit();rows=A.profile_rows(x,s,audit)
        self.assertEqual(audit.failures,[])
        self.assertEqual(next(r["value"] for r in rows if r["stream"]=="raw" and r["statistic"]=="K"),2)
        self.assertEqual(sum(r["count"] for r in rows if r["stream"]=="raw" and r["statistic"]=="delta"),1)
        self.assertEqual(next(r["value"] for r in rows if r["stream"]=="raw" and r["statistic"]=="fill"),2)

    def test_binary_storage_header_uses_hidden_size_not_feature_count(self):
        cache=next(p for p in MODULE.parents if p.name=="cache")
        folder=cache/"formal_study_20261017/runtime_v2"
        folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder,prefix="model_header_fixture_") as temp:
            root=Path(temp)
            fields=[1,0x01020304,1,14,8,64,4,0,0,0,0,0,57,2,4,0]
            (root/"model.bin").write_bytes(struct.pack("<8s16IQ",b"LV602R1\0",*fields,2821))
            s=spec("streamer","student")
            storage=A.model_storage(dict(spec=s,summary={},dir=root),root)
            self.assertEqual(storage["hidden_size"],8)
            self.assertEqual(storage["recurrent_hc_bytes"],64)
            self.assertEqual(storage["model_weight_bytes"],11284)

    def test_callback_tail_is_strict_failure_with_exact_missing_count(self):
        cache=next(p for p in MODULE.parents if p.name=="cache")
        folder=cache/"formal_study_20261017/runtime_v2"
        folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder,prefix="callback_tail_fixture_") as temp:
            root=Path(temp)
            (root/"callbacks.csv").write_text("sequence,instructions,cycle,kind,address,completed_updates\n0,1,10,D,64,0\n1,2,20,D,128,1\n2,3")
            run=dict(spec=spec("streamer","student"),dir=root,summary={"four602_counters":{"completed_updates":3,"demand_callbacks":3}})
            audit=A.Audit();result=A.verify_callbacks(run,audit)
            self.assertTrue(result["recorded_prefix_valid"])
            self.assertFalse(result["complete_log"])
            self.assertEqual(result["missing_complete_rows"],1)
            self.assertEqual([e["check"] for e in audit.failures],["actual_callback_log_complete"])
        run=dict(spec=spec("streamer","student"),dir=Path("/nonexistent"),summary={})
        audit=A.Audit();A.verify_callbacks(run,audit)
        self.assertEqual([e["check"] for e in audit.failures],["required_actual_callback_log_present"])

    def test_training_budgets_do_not_appear_as_one_shared_budget(self):
        definition=A.definitions()["splits"]
        self.assertIn("Stride",definition)
        self.assertIn("0..1M",definition)
        self.assertIn("0..20M",definition)
        self.assertIn("not a matched",definition)


if __name__=="__main__":
    unittest.main(verbosity=2)
