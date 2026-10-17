#!/usr/bin/env python3
"""Summarize completed measurement-only recurrent histories from real frozen runs."""
import argparse
import json
from pathlib import Path
from analyze import inspect_run, load_oracle, storage_categories, write_csv, ratio


def summarize(run_dir, out_dir, reference_dir=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    oracle = load_oracle()
    summaries, examples, comparisons = [], [], []
    for directory in sorted(run_dir.glob('cold_history*')):
        run = inspect_run(directory, oracle)
        if not run['complete']:
            raise RuntimeError('History run is unfinished: ' + str(directory))
        s = run['snapshots'][-1]
        storage = storage_categories(run)
        configured = [r['configured_max_bytes'] for r in storage]
        peaks = [r['observed_peak_bytes'] for r in storage]
        record = {
            'run_id': directory.name, 'instructions': s['instructions'],
            'eligible': s['eligible_callbacks'], 'admitted': s['admitted_decisions'],
            'started': s['nn_started'], 'completed': s['completed_decisions'],
            'state_history_completed': s['state_history_completed'],
            'mean_prior': ratio(s['state_history_prior_sum'], s['state_history_completed']),
            'max_prior': s['state_history_prior_max'], 'zero_prior': s['state_history_zero_prior'],
            'mean_wait_cycles': ratio(s['inference_wait_cycles_sum'], s['nn_started']),
            'max_wait_cycles': s['inference_wait_cycles_max'],
            'mean_service_cycles': ratio(s['inference_completed_service_cycles_sum'], s['completed_decisions']),
            'max_service_cycles': s['inference_completed_service_cycles_max'],
            'mean_completion_cycles': ratio(s['inference_completion_cycles_sum'], s['completed_decisions']),
            'max_completion_cycles': s['inference_completion_cycles_max'],
            'input_drops': s['input_drops'], 'input_queue': s['input_queue'],
            'inference_inflight': s['inference_inflight'],
            'deployment_configured_kib': sum(configured)/1024 if all(v is not None for v in configured) else None,
            'deployment_component_peaks_kib': sum(peaks)/1024 if all(v is not None for v in peaks) else None,
            'weight_kib': s['weight_bytes']/1024, 'state_peak_kib': s['state_bytes_peak']/1024,
            'occupancy_lines': s['occupancy_lines'], 'capacity_lines': s['l2_capacity_lines'],
            'cohort_redundant_cache': s['cohort_redundant_cache'],
            'cohort_redundant_inflight': s['cohort_redundant_inflight'],
            'cohort_pending': s['cohort_pending'], 'cohort_resident_unused': s['cohort_resident_unused'],
            'measurement_overhead': 'history counters/timestamps and bounded JSONL excluded; component peaks need not be simultaneous',
        }
        summaries.append(record)
        history = []
        lines = (directory/'history.jsonl').read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines)-1:
                    raise
                record['truncated_final_log_record'] = 1
        record['complete_logged_records'] = len(history)
        # Pick a consecutive excerpt containing an interleaved repeated PC. No
        # filtering of model observations or reconstruction of synthetic events.
        excerpt = None
        for index in range(max(0, len(history)-7)):
            candidate = history[index:index+8]
            pcs = [r['pc'] for r in candidate]
            if len(set(pcs)) > 1 and any(pcs[i] == pcs[k] != pcs[j] for i in range(8) for j in range(i+1,8) for k in range(j+1,8)):
                excerpt = candidate
                break
        if excerpt is None:
            raise RuntimeError('No actual interleaved/reused-PC excerpt within bounded history: '+directory.name)
        for value in excerpt:
            row = {'run_id': directory.name, **value}
            row['k'] = row.pop('K')
            examples.append(row)
        if reference_dir:
            refname = ('cold_frozen_i1m_h8_s7_state0_mac0' if not s['macs_per_cycle'] else
                       'cold_frozen_i1m_h8_s7_state64_mac'+str(s['macs_per_cycle']))
            ref = reference_dir/refname/'run.log'
            old = oracle.parse_log(ref)
            new = run['counters']
            for field in sorted(set(old)|set(new)):
                comparisons.append({'run_id': directory.name, 'reference': refname,
                                    'field': field, 'reference_value': old.get(field),
                                    'instrumented_value': new.get(field),
                                    'equal': old.get(field) == new.get(field)})
            if old != new:
                raise RuntimeError('Instrumentation changed final counters: '+directory.name)
    write_csv(out_dir/'history_summary.csv', summaries)
    write_csv(out_dir/'history_example.csv', examples)
    if comparisons:
        write_csv(out_dir/'history_parity.csv', comparisons)
    print(json.dumps({'completed_runs': len(summaries), 'example_rows': len(examples),
                      'parity_fields': len(comparisons), 'summary': summaries}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path)
    args = parser.parse_args()
    summarize(args.run_dir, args.out_dir, args.reference_dir)
