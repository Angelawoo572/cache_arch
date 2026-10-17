#!/usr/bin/env python3
"""Analysis correctness fixtures; never launch training or simulation."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

EXP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXP/'python'))
import formal_analysis as analysis


class StorageBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence=analysis.load_storage_audit()
        if cls.evidence['status']!='available':
            raise unittest.SkipTest('Full storage audit JSON unavailable in this source checkout.')

    def run_fixture(self,arm,budget=0):
        audit=self.evidence['audit']
        return {'spec':{'arm':arm,'budget':budget,'protocol':audit['protocol'],
                        'build_id':audit['build_id'],'controls':{'decoder_limit':32}},
                'summary':{'h':8 if arm=='nn' else 0,'state_capacity':64,
                           'input_queue_capacity':16,'output_queue_capacity':32,
                           'deployment_bytes_configured':16808 if arm=='nn' else 0},
                'directory':None}

    def test_complete_budgets_include_control_and_cap(self):
        for arm,fixed,capped in [('nn',16920,16936),('stride_forward2',2136,2152),('spp',5938,5954),('none',0,0)]:
            for cap,expected in [(0,fixed),(2363,capped)]:
                with self.subTest(arm=arm,cap=cap):
                    r=analysis.source_storage_fields(self.run_fixture(arm,cap),self.evidence)
                    self.assertEqual(r['source_field_for_this_run_bytes'],expected)
                    self.assertEqual(sum(r['source_field_components_bytes'].values()),expected)
                    self.assertFalse(r['source_field_budget_is_rtl_or_silicon_area'])

    def test_stale_resource_or_build_audit_is_not_applied(self):
        r=self.run_fixture('nn');r['summary']['state_capacity']=32
        self.assertEqual(analysis.source_storage_fields(r,self.evidence)['source_field_budget_status'],'audited_nn_resource_shape_mismatch')
        r=self.run_fixture('nn');r['spec']['build_id']='another-build'
        self.assertEqual(analysis.source_storage_fields(r,self.evidence)['source_field_budget_status'],'audit_build_mismatch')

    def test_host_rss_is_separate_from_logical_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'host_time.txt').write_text('Maximum resident set size (kbytes): 9876\n')
            r=self.run_fixture('nn');r['directory']=tmp
            fields=analysis.source_storage_fields(r,self.evidence)
            self.assertEqual(fields['host_peak_rss_kib'],9876)
            self.assertEqual(fields['source_field_for_this_run_bytes'],16920)
            self.assertNotEqual(fields['host_cpp_sizeof_object_bytes'],16920)

    def test_raw_core_counter_not_replaced_by_complete_budget(self):
        r=self.run_fixture('nn');r['spec'].update(run_id='fixture',stage='test',workload='602')
        row=analysis.ancillary([r],self.evidence)['storage'][0]
        self.assertEqual(row['deployment_bytes_configured'],16808)
        self.assertEqual(row['corrected_nn_core_arrays_events_scratch_bytes'],16808)
        self.assertEqual(row['source_field_for_this_run_bytes'],16920)


class BudgetAndDenominators(unittest.TestCase):
    def budget_fixture(self,path):
        source_id='validation-source';p=path/'validation'/source_id;p.mkdir(parents=True)
        spec={'stage':'validation','protocol':'fixture','workload':'602','arm':'stride_forward1'}
        (p/'run.json').write_text(json.dumps({'spec':spec,'status':'complete'}))
        snapshot={'final':True,'P':123,'instructions':10000}
        (p/'summary.json').write_text(json.dumps(snapshot))
        selection={'protocol':'fixture','requests_per_million_instructions':{'602':12300},
                   'evidence':{'602':{'source_run':source_id,'requests':123,'instructions':10000}}}
        (path/'budget_selection.json').write_text(json.dumps(selection))
        return p

    def test_validation_cap_and_zero_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);p=self.budget_fixture(path)
            self.assertEqual(analysis.audit_budget(path,'fixture')['status'],'valid')
            summary=json.loads((p/'summary.json').read_text());summary['instructions']=0
            (p/'summary.json').write_text(json.dumps(summary))
            self.assertEqual(analysis.audit_budget(path,'fixture')['status'],'invalid_evidence')

    def test_test_source_cannot_select_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);p=self.budget_fixture(path)
            record=json.loads((p/'run.json').read_text());record['spec']['stage']='test'
            (p/'run.json').write_text(json.dumps(record))
            self.assertEqual(analysis.audit_budget(path,'fixture')['status'],'invalid_evidence')

    def test_load_outcomes_exclude_rfo_and_keep_undefined_rates(self):
        baseline={'spec':{'run_id':'nopf'},'summary':{'M':100,'instructions':1000,'cycles':2000}}
        run={'spec':{'arm':'nn'},'errors':[],'summary':{'instructions':1000,'cycles':1000,'M':50,'NL':200,
             'cohort_timely_load':20,'cohort_late_load':5,'cohort_timely_rfo':70,
             'cohort_enqueued':100,'cohort_pending':10,'cohort_resident_unused':10,
             'U':90,'Ipf':100,'L':5}}
        m=analysis.derive(run,baseline)
        self.assertEqual(m['timely_load_coverage'],.2)
        self.assertEqual(m['load_useful_accuracy_enqueued'],.25)
        self.assertEqual(m['load_useful_timeliness'],.8)
        self.assertIsNone(analysis.ratio(0,0))

    def test_sparse_log_bins_do_not_stretch_missing_bins(self):
        r={'spec':{},'summary':{'distributions_log2_upper_bound':{'wait':{'0':1,'7':2,'31':4}}}}
        bins=analysis.ancillary([r])['service_distributions']
        self.assertEqual([(v['lower_bound'],v['upper_bound']) for v in bins],[(0,0),(4,7),(16,31)])
        self.assertEqual(sum(v['count'] for v in bins),7)
        self.assertEqual(bins[-1]['cumulative_fraction'],1)


if __name__=='__main__':unittest.main()
