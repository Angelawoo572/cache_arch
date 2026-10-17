#!/usr/bin/env python3
"""Independent check of actual full-run native-SPP scope; never edits raw data."""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--prior-raw', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    with (args.analysis / 'metrics_per_seed.csv').open() as handle:
        metrics = {r['run_id']: r for r in csv.DictReader(handle)}
    analysis_audit = read(args.analysis / 'audit.json')
    matrix = read(args.raw / 'formal_matrix.json')
    planned = [s for s in matrix if s['arm'] == 'spp']
    assert len(planned) == 4 and {(s['roi'], s['budget']) for s in planned} == {(roi, cap) for roi in (0, 1) for cap in (0, 2363)}
    completed, pending, failures = [], [], []
    fields = ('enqueued', 'timely', 'late', 'timely_load', 'late_load', 'timely_rfo', 'late_rfo', 'unused', 'redundant_cache', 'redundant_inflight', 'pending', 'resident_unused')
    for spec in planned:
        rid = spec['run_id']
        directory = args.raw / 'test' / rid
        if not (directory / 'run.json').exists() or read(directory / 'run.json').get('status') != 'complete':
            pending.append(rid)
            continue
        try:
            record, c = read(directory / 'run.json'), read(directory / 'summary.json')
            assert c['event'] == 'snapshot' and c['final'] and record['exit_code'] == 0
            assert record['spec']['protocol'] == 'gate_mechanism_20261016_v1'
            assert '--l2c_prefetcher_types=spp_dev2' in record['command']
            assert c['engine_method'] == 'none' and c['nn_completed'] == c['h'] == 0
            assert c['measurement_cohort_scope'] == 'l2_target_fully_tracked_only'
            assert c['llc_only_lifecycle_accuracy_available'] is False
            assert c['cohort_all_enqueued_requests_fully_tracked'] == (c['cohort_llc_only_excluded'] == 0)
            assert c['request_l2_target_generated'] + c['request_llc_only_generated'] == c['request_gate_generated']
            assert c['request_l2_target_allowed'] + c['request_llc_only_allowed'] == c['request_gate_allowed']
            assert c['request_gate_allowed'] == c['request_cache_enqueued'] + c['request_cache_merged'] + c['request_cache_rejected']
            assert c['measurement_unconsumed_context_labels'] == 0
            assert c['request_cache_pq_inserted'] == c['cohort_enqueued'] + c['cohort_llc_only_excluded']
            phases, scopes, l2, llc_count = {}, {'l2_target': Counter(), 'llc_only': Counter()}, Counter(), 0
            groups = c['mechanism_classified_requests']
            assert isinstance(groups, list) and groups
            for group in groups:
                scope = group['fill_scope']
                assert group['arm'] == 'spp' and scope in scopes
                assert group['slot'] == group['list_duplicate'] == group['teacher_category'] == -1
                assert group['current_line'] in (0, 1) and group['origin_phase'] in range(5)
                expected_key = f'phase_{group["origin_phase"]}/slot_unknown/current_{group["current_line"]}/duplicate_unknown/teacher_unknown'
                assert group['key'] == expected_key
                for phase_field, count in [('actual_attempt_phase_counts', 'cache_attempted'), ('actual_allowed_phase_counts', 'allowed')]:
                    assert group[phase_field] == ({str(group['origin_phase']): group[count]} if group[count] else {})
                count_names = ('cache_attempted', 'allowed', 'current_suppressed', 'page_suppressed', 'budget_suppressed', 'cache_accepted_nonmerged', 'cache_merged', 'cache_rejected', 'pq_enqueued')
                assert all(isinstance(group[name], int) and group[name] >= 0 for name in count_names)
                scopes[scope].update({name: group[name] for name in count_names})
                assert group['cache_attempted'] == sum(group[k] for k in ('allowed', 'current_suppressed', 'page_suppressed', 'budget_suppressed'))
                assert group['allowed'] == sum(group[k] for k in ('cache_accepted_nonmerged', 'cache_merged', 'cache_rejected'))
                key = (group['origin_phase'], scope)
                phases.setdefault(key, Counter()).update({name: group[name] for name in count_names})
                if scope == 'llc_only':
                    llc_count += 1
                    assert group['l2_lifecycle_fully_tracked'] is False and group['l2_cohort'] is None
                    assert group['l2_load_accuracy_enqueued'] is None and group['l2_load_timeliness'] is None
                    continue
                v = group['l2_cohort']
                assert group['l2_lifecycle_fully_tracked'] is True
                assert v['enqueued'] == group['pq_enqueued'] == sum(v[k] for k in ('timely', 'late', 'unused', 'redundant_cache', 'redundant_inflight', 'pending', 'resident_unused'))
                assert v['timely'] == v['timely_load'] + v['timely_rfo']
                assert v['late'] == v['late_load'] + v['late_rfo']
                assert v['useful'] == v['timely'] + v['late'] and v['useful_load'] == v['timely_load'] + v['late_load']
                l2.update({name: v[name] for name in fields})
            assert llc_count > 0
            for name in fields:
                assert l2[name] == c['cohort_' + name], name
            assert {phase for phase, scope in phases} == set(range(5))
            assert scopes['l2_target']['pq_enqueued'] == c['cohort_enqueued']
            assert scopes['llc_only']['pq_enqueued'] == c['cohort_llc_only_excluded']
            for scope, prefix in [('l2_target', 'request_l2_target_'), ('llc_only', 'request_llc_only_')]:
                for field, suffix in [('cache_attempted', 'generated'), ('allowed', 'allowed'), ('cache_merged', 'cache_merged'), ('cache_rejected', 'cache_rejected')]:
                    assert scopes[scope][field] == c[prefix + suffix]
            for field, target in [('cache_attempted', 'request_gate_generated'), ('allowed', 'request_gate_allowed'), ('cache_accepted_nonmerged', 'request_cache_enqueued'), ('cache_merged', 'request_cache_merged'), ('cache_rejected', 'request_cache_rejected'), ('pq_enqueued', 'request_cache_pq_inserted')]:
                assert sum(count[field] for count in scopes.values()) == c[target]
            old_path = args.prior_raw / 'test' / f'test_602_r{spec["roi"]}_spp_s7_balanced_m0_b{spec["budget"]}_sil0_qfifo_hist5000000' / 'summary.json'
            old = read(old_path)
            stable = ['instructions', 'cycles', 'M', 'NL', 'P', 'request_gate_allowed'] + sorted(k for k in old.keys() & c.keys() if k.startswith('dram_'))
            differences = {k: [old[k], c[k]] for k in stable if old[k] != c[k]}
            assert not differences, differences
            assert old['cohort_enqueued'] == c['request_cache_pq_inserted']
            assert old['cohort_enqueued'] - c['cohort_enqueued'] == c['cohort_llc_only_excluded']
            metric = metrics[rid]
            useful = c['cohort_timely_load'] + c['cohort_late_load']
            accuracy = useful / c['cohort_enqueued']
            timeliness = c['cohort_timely_load'] / useful if useful else None
            assert math.isclose(float(metric['l2_target_useful_accuracy_enqueued']), accuracy, rel_tol=1e-12)
            if timeliness is None:
                assert metric['l2_target_timeliness'] == ''
            else:
                assert math.isclose(float(metric['l2_target_timeliness']), timeliness, rel_tol=1e-12)
            assert metric['llc_only_accuracy'] == metric['llc_only_timeliness'] == ''
            assert 'subset' in metric['coverage_scope']
            completed.append(dict(run_id=rid, roi=spec['roi'], budget=spec['budget'],
                source_summary=str((directory / 'summary.json').resolve()), source_protocol=record['spec']['protocol'], build_id=record['spec']['build_id'],
                prior_summary=str(old_path.resolve()), status='PASS', functional_fields_exact=len(stable), functional_differences=differences,
                ipc=c['instructions'] / c['cycles'], cycles=c['cycles'], l2_load_misses=c['M'], actual_dram_completed_bytes=c['dram_total_bus_bytes'],
                actual_all_target_submissions=c['request_gate_allowed'], l2_target_submissions=c['request_l2_target_allowed'], llc_only_submissions=c['request_llc_only_allowed'],
                actual_all_target_pq_insertions=c['request_cache_pq_inserted'], complete_l2_target_cohort_enqueued=c['cohort_enqueued'], llc_only_pq_insertions_excluded=c['cohort_llc_only_excluded'],
                complete_l2_target_cohort=dict(l2), l2_target_load_accuracy_enqueued=accuracy, l2_target_load_timeliness=timeliness,
                llc_only_lifecycle_accuracy=None, llc_only_lifecycle_timeliness=None, llc_only_null_profile_groups=llc_count,
                old_mixed_fill_cohort_enqueued=old['cohort_enqueued'], old_mixed_fill_pending=old['cohort_pending'],
                old_mixed_fill_timely_load=old['cohort_timely_load'], new_l2_target_timely_load=c['cohort_timely_load'],
                phase_scope_rows=[dict(origin_phase=key[0], fill_scope=key[1], **count) for key, count in sorted(phases.items())]))
        except Exception as error:
            failures.append(dict(run_id=rid, error=str(error), error_type=type(error).__name__))
    complete_scope = len(completed) == 4 and not pending and not failures
    report = dict(status='FAIL' if failures else 'PASS' if complete_scope else 'INCOMPLETE',
        status_scope='Independent native-SPP L2/LLC cohort audit; full matrix has its own analysis audit',
        expected_spp_cases=4, completed_spp_cases=len(completed), pending_spp_cases=pending, failures=failures,
        full_matrix_status=analysis_audit['status'], full_matrix_completed=analysis_audit['completed'], full_matrix_expected=71,
        full_matrix_verified=complete_scope and analysis_audit['status'] == 'PASS' and analysis_audit['completed'] == 71,
        runs=completed, definitions=dict(cohort='Only original L2-target PQ insertions receive L2 lifecycle IDs; LLC-only lifecycle is unavailable.',
        accuracy='LOAD useful outcomes / the fully tracked L2-target PQ-enqueued cohort; right-censored pending/resident-unused retained.',
        coverage='L2-target useful/M0 is a partial target-level subset, not whole-SPP coverage.',
        functional='Native fill policy, overall cycles/IPC/LOAD misses/requests and all common actual DRAM counters compare exactly to the same original full formal run.',
        old_scope='Old mixed-fill cohort metrics are not whole-SPP accuracy/timeliness. Old LLC-origin requests could be promoted by later merges; strict original-target exclusion can change both numerator and denominator. This does not change functional performance.',
        qualification='No complete LLC lifecycle or universal runtime proof is claimed. Evidence covers only the listed completed full25M cases; no prefixes or pilots.'))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('runs', 'definitions')}, indent=2))
    if failures or (args.require_complete and not complete_scope):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
