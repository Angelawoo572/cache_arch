"""Strict quality comparisons and exact retained-window fixtures."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
import compare_quality as q

class QualityFixtures(unittest.TestCase):
    def metric(self,*values):return dict(zip(q.QUALITY,values))
    def test_strict_joint_requires_all_three(self):
        self.assertEqual(q.compare_metrics(self.metric(.5,.6,.7),self.metric(.4,.5,.6))[0],'EXCEEDS')
        self.assertEqual(q.compare_metrics(self.metric(.5,.6,.6),self.metric(.4,.5,.6))[0],'NONLOWER_WITH_TIE')
        self.assertEqual(q.compare_metrics(self.metric(.5,.4,.7),self.metric(.4,.5,.6))[0],'NOT_EXCEEDS')
    def test_exact_counter_fraction_ties(self):
        c=dict(pf_useful=1,pf_issued=3,pq_merged_duplicate_proxy=0,pf_late=2)
        d=dict(pf_useful=2,pf_issued=6,pq_merged_duplicate_proxy=0,pf_late=4)
        m=q.exact_quality(c,{'l2_load_miss':3});n=q.exact_quality(d,{'l2_load_miss':6})
        status,differences=q.compare_metrics(m,n)
        self.assertEqual(status,'NONLOWER_WITH_TIE')
        self.assertTrue(all(v==0 for v in differences.values()))
    def test_zero_denominator_is_na(self):
        c={'pf_useful':0,'pf_issued':0,'pq_merged_duplicate_proxy':0,'pf_late':0}
        m=q.a.metrics(c,{'l2_load_miss':0})
        self.assertEqual(q.compare_metrics(m,self.metric(.4,.5,.6))[0],'NA')
        self.assertIsNone(m['selected_accuracy'])
    def test_maximal_sampled_intervals_and_first_exit(self):
        states=['NOT_EXCEEDS','EXCEEDS','EXCEEDS','NONLOWER_WITH_TIE','EXCEEDS']
        rows=[dict(quality_status=s,start_instructions=0,end_instructions=100*(i+1),total_completed_decisions=i+1,limiting_metrics='accuracy' if s!='EXCEEDS' else '') for i,s in enumerate(states)]
        spans=q.segments(rows)
        self.assertEqual(len(spans),4)
        self.assertEqual((spans[1]['first_recorded_instructions'],spans[1]['last_recorded_instructions'],spans[1]['exit_at_instructions']),(200,300,400))
        self.assertEqual(spans[1]['preceding_recorded_instructions'],100)
        self.assertIsNone(spans[-1]['exit_at_instructions'])
    def test_both_comparisons_require_same_nn_run_and_scope(self):
        x=dict(run_id='a',scale='cumulative',start_instructions=0,end_instructions=100,quality_status='EXCEEDS',limiting_metrics='')
        with self.assertRaises(ValueError):q.joint_series([x],[dict(x,run_id='b')])
        with self.assertRaises(ValueError):q.joint_series([x],[dict(x,end_instructions=101)])
        self.assertEqual(q.joint_series([x],[x])[0]['quality_status'],'EXCEEDS')
        self.assertEqual(q.joint_series([x],[dict(x,quality_status='MISSING_REFERENCE')])[0]['quality_status'],'MISSING_OR_UNDEFINED_REFERENCE')
    def test_protocol_and_same_checkpoint_reference_selection(self):
        def run(name,arm='frozen',protocol='cold',h=8,budget='i1m'):
            return {'complete':True,'identity':dict(run_id=name,arm=arm,protocol=protocol,phase='final',slice_origin_instructions=25000000,simulation_instructions=25000000,hidden_size=h,checkpoint=budget)}
        nn=run('nn');other=run('summer','summer_replay','historical');same=run('same','summer_replay',budget='i20m');wrong=run('wrong','summer_replay',h=16)
        self.assertIsNone(q.select_reference(nn,[other,wrong],'summer'))
        nn['identity']['observed_trace_origin_records']=25000000
        wrong_origin=run('origin','summer_replay',budget='i20m')
        wrong_origin['identity']['observed_trace_origin_records']=25000004
        self.assertIsNone(q.select_reference(nn,[wrong_origin],'summer'))
        self.assertIs(q.select_reference(nn,[other,same,wrong],'summer'),same)
        original=run('i1m','summer_replay')
        self.assertIs(q.select_reference(nn,[original,same],'offline_same_checkpoint'),original)
    def test_cross_window_censoring_is_separate(self):
        run={'snapshots':[{'instructions':200000}],'cohorts':[dict(start_instructions=0,enqueued=3,timely=1,late=0,unused=0,redundant_cache=0,redundant_inflight=0,pending=1,resident_unused=1)]}
        c=q.cohort_fields(run,0,100000)
        self.assertEqual(c['issue_cohort_censored'],2)
        self.assertEqual(c['issue_cohort_unused'],0)
        self.assertIn('UNAVAILABLE',q.cohort_fields(run,0,1000)['cohort_followup'])
    def test_reference_window_savings_do_not_replace_cumulative_savings(self):
        def run(name,cycles):
            ident=dict(run_id=name,arm='frozen',protocol='cold',hidden_size=8,checkpoint='i1m',state_capacity=0,service_case=0,slice_origin_instructions=25000000)
            rows=[]
            for i,cy in enumerate(cycles):
                r={k:0 for k in q.a.COUNTERS+q.a.PROGRESS}
                r.update(instructions=i*100000,cycles=cy,l2_load_miss=i*10,l2_loads=i*20,pf_useful=i,pf_issued=i*2)
                rows.append(r)
            return dict(identity=ident,snapshots=rows,engine={},cohorts=[])
        nn=run('nn',[0,100,180]);ref=run('ref',[0,110,220])
        rows=q.compare_series(nn,ref,ref,ref,'interval_100k','summer')
        self.assertEqual(rows[-1]['interval_cycles_saved_vs_reference'],30)
        self.assertEqual(rows[-1]['cumulative_cycles_saved_vs_reference'],40)
        self.assertEqual(rows[-1]['reference_cumulative_cycles'],220)
    def test_retirement_and_trace_axes_do_not_count_skipped_records_as_simulation(self):
        cold=q.retirement_axes(dict(observed_trace_origin_records=25000000,skip_records=25000000),2600000)
        self.assertEqual(cold['total_simulated_retired_instructions'],2600000)
        self.assertEqual(cold['trace_record_ordinal'],27600000)
        historical=q.retirement_axes(dict(observed_trace_origin_records=25000004,skip_records=0),20500000)
        self.assertEqual(historical['total_simulated_retired_instructions'],45500004)
        self.assertEqual(historical['trace_record_ordinal'],45500004)
        self.assertEqual(historical['roi_retired_instructions'],20500000)
    def test_no_interpolation_or_reference_total_substitution(self):
        run={'snapshots':[{'instructions':100000,'cycles':20},{'instructions':200003,'cycles':40}]}
        s=q.series(run,'cumulative')
        self.assertIn((0,200003),s)
        self.assertNotIn((0,200000),s)

if __name__=='__main__':unittest.main()
