#!/usr/bin/env python3
"""Compare SPP and LSTM coverage on the same dynamic NoPF misses."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

from compare import read_events, run_config, start_signature


POLICIES = ('nopf', 'spp', 'lstm')
USES = frozenset(('timely', 'late'))
REGION = re.compile(r'frozen_(\d+)m')


def read_outcomes(directory):
    results = json.loads((directory / 'summary.json').read_text())
    by_policy = {row['policy']: row for row in results}
    if len(by_policy) != len(results) or not set(POLICIES) <= by_policy.keys():
        raise ValueError(f'{directory} needs one NoPF, SPP, and LSTM run')
    nopf = by_policy['nopf']
    if nopf['prefetch_issued']:
        raise ValueError(f'{directory} issued prefetches under NoPF')
    events = {}
    for policy in POLICIES:
        result = by_policy[policy]
        if start_signature(result) != start_signature(nopf):
            raise ValueError(f'{directory} changed warmup, seed, or frequency for {policy}')
        demand = (directory / result['log']).with_name('demand.csv')
        events[policy] = read_events(demand, result)
    for left, right in (('nopf', 'spp'), ('nopf', 'lstm'), ('spp', 'lstm')):
        same_requests = events[left].keys() & events[right].keys()
        for request in same_requests:
            if events[left][request][0] != events[right][request][0]:
                raise ValueError(f'{directory} changed PC or virtual line for {request} in {left}/{right}')
    absent = (None, None, 'absent')
    counts = Counter((events['spp'].get(request, absent)[2], events['lstm'].get(request, absent)[2])
                     for request, (_, hit, _) in events['nopf'].items() if not hit)
    if counts.total() != nopf['llc_demand_misses']:
        raise ValueError(f'{directory} has inconsistent NoPF miss counts')
    return counts


def summarize(region, counts):
    total = counts.total()
    lstm_matched = spp_matched = both_covered = lstm_only = spp_only = 0
    lstm_only_timely = spp_only_timely = 0
    for (spp, lstm), count in counts.items():
        lstm_matched += count * (lstm != 'absent')
        spp_matched += count * (spp != 'absent')
        both_covered += count * (spp in USES and lstm in USES)
        lstm_only += count * (spp not in USES and lstm in USES)
        spp_only += count * (spp in USES and lstm not in USES)
        lstm_only_timely += count * (spp not in USES and lstm == 'timely')
        spp_only_timely += count * (spp == 'timely' and lstm not in USES)
    lstm_covered = both_covered + lstm_only
    spp_covered = both_covered + spp_only
    both_timely = counts['timely', 'timely']
    late_to_timely = counts['late', 'timely']
    timely_to_late = counts['timely', 'late']
    lstm_timely = both_timely + late_to_timely + lstm_only_timely
    spp_timely = both_timely + timely_to_late + spp_only_timely
    return {
        'region': region,
        'nopf_misses': total,
        'lstm_matched': lstm_matched,
        'spp_matched': spp_matched,
        'lstm_covered': lstm_covered,
        'spp_covered': spp_covered,
        'both_covered': both_covered,
        'lstm_only': lstm_only,
        'spp_only': spp_only,
        'neither_covered': total - both_covered - lstm_only - spp_only,
        'both_timely': both_timely,
        'both_late': counts['late', 'late'],
        'spp_late_lstm_timely': late_to_timely,
        'spp_timely_lstm_late': timely_to_late,
        'lstm_only_timely': lstm_only_timely,
        'spp_only_timely': spp_only_timely,
        'lstm_timely': lstm_timely,
        'spp_timely': spp_timely,
        'lstm_coverage': lstm_covered / total if total else None,
        'spp_coverage': spp_covered / total if total else None,
        'lstm_timely_coverage': lstm_timely / total if total else None,
        'spp_timely_coverage': spp_timely / total if total else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+', help='frozen_<region>m replay directories')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    regions = [REGION.fullmatch(run.name) for run in args.runs]
    if not all(regions):
        parser.error('run directory names must match frozen_<region>m')
    region_ids = [int(match.group(1)) for match in regions]
    if len(set(region_ids)) != len(region_ids):
        parser.error('each region must appear once')
    settings = run_config(args.runs[0])[1:]
    all_counts = Counter()
    output = []
    for region, run in sorted(zip(region_ids, args.runs)):
        if run_config(run)[1:] != settings:
            raise ValueError(f'{run} changed warmup, ROI, trace skip, or issue degree')
        counts = read_outcomes(run)
        output.append(summarize(region, counts))
        all_counts.update(counts)
    output.append(summarize('all', all_counts))
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=output[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(output)


if __name__ == '__main__':
    main()
