#!/usr/bin/env python3
"""Combine matched ChampSim arms and report direct LLC miss coverage."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'conventional'))
from coverage import read_events


def run_config(directory):
    settings = json.loads((directory / 'config.json').read_text())
    return tuple(settings[field] for field in ('traces', 'warmup', 'instructions', 'skip', 'degree'))


def start_signature(result):
    warmup = tuple((core['warmup_retired_instructions'], core['warmup_cycles']) for core in result['cores'])
    return result['trace_seed'], result['cpu_mhz'], warmup


def rows(directories):
    config = run_config(directories[0])
    results = {}
    nopf = None
    nopf_events = None
    for directory in directories:
        if run_config(directory) != config:
            raise ValueError('trace, warmup, ROI, or degree mismatch in {}'.format(directory))
        for row in json.loads((directory / 'summary.json').read_text()):
            policy = row['policy']
            event_file = (directory / row['log']).with_name('demand.csv')
            if policy == 'nopf':
                events = read_events(event_file, row)
                if nopf is not None:
                    if nopf_events != events or nopf['roi_cycles'] != row['roi_cycles']:
                        raise ValueError('the NoPF anchor differs between builds')
                    continue
                nopf, nopf_events = row, events
            if policy in results:
                raise ValueError('duplicate policy {}'.format(policy))
            results[policy] = row, event_file
    if nopf is None:
        raise ValueError('NoPF is required for coverage and speedup')
    misses = {key for key, (_, hit, _) in nopf_events.items() if not hit}
    for policy, (row, event_file) in results.items():
        if start_signature(row) != start_signature(nopf):
            raise ValueError('warmup, seed, or frequency mismatch for {}'.format(policy))
        events = nopf_events if policy == 'nopf' else read_events(event_file, row)
        matched = misses & events.keys()
        if any(nopf_events[key][0] != events[key][0] for key in nopf_events.keys() & events.keys()):
            raise ValueError('PC or virtual line mismatch for {}'.format(policy))
        timely = sum(events[key][2] == 'timely' for key in matched)
        late = sum(events[key][2] == 'late' for key in matched)
        yield {
            'policy': policy,
            'accuracy': row['prefetch_usefulness_per_issued'],
            'timeliness': row['prefetch_timeliness'],
            'coverage': (timely + late) / len(misses) if misses else None,
            'timely_coverage': timely / len(misses) if misses else None,
            'nopf_misses': len(misses),
            'matched_nopf_misses': len(matched),
            'llc_hit_rate': row['llc_hit_rate'],
            'llc_miss_rate': row['llc_miss_rate'],
            'llc_mpki': row['llc_mpki'],
            'aggregate_ipc': row['aggregate_ipc'],
            'roi_cycles': row['roi_cycles'],
            'roi_seconds': row['roi_seconds'],
            'speedup': nopf['roi_cycles'] / row['roi_cycles'],
            'prefetch_issued': row['prefetch_issued'],
            'prefetch_timely': row['prefetch_timely'],
            'prefetch_late': row['prefetch_late'],
            'dram_total_bytes': row['dram_total_bytes'],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compared = list(rows(args.directories))
    with args.output.open('w', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=compared[0])
        writer.writeheader()
        writer.writerows(compared)


if __name__ == '__main__':
    main()
