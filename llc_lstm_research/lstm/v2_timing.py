#!/usr/bin/env python3
"""Calibrate NoPF lead labels with measured trigger-to-fill latency."""

import argparse
from bisect import bisect_right
from collections import Counter
import csv
import json
import math
from pathlib import Path


def samples(files):
    service = []
    counts = {}
    for path in files:
        total, completed = 0, 0
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                total += 1
                if not row['fill_cycle']:
                    continue
                ingress, start, ready, accepted, filled = (int(row[name]) for name in (
                    'ingress_cycle', 'predict_start_cycle', 'ready_cycle', 'accepted_cycle', 'fill_cycle'))
                if not ingress <= start <= ready <= accepted <= filled:
                    raise ValueError('invalid trigger-to-fill cycle order in {}'.format(path))
                service.append(filled - ingress)
                completed += 1
        counts[str(path)] = {'sampled_requests': total, 'completed_fills': completed}
    if not service:
        raise ValueError('no completed prefetch fills')
    return sorted(service), counts


def used_by_nopf_lead(baseline, files):
    """Compare first-use outcomes at the same dynamic loads in the NoPF trace."""
    def identity(row, prefix=''):
        return tuple(row[prefix + field] for field in ('cpu', 'instr_id', 'operand'))

    with baseline.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    events = {identity(row): row for row in rows}
    if len(events) != len(rows):
        raise ValueError('baseline demand identities are not unique')

    bands = ((0, 149), (150, 299), (300, 500), (501, 2000), (2001, 4000), (4001, None))
    cuts = [high + 1 for _, high in bands if high is not None]
    outcomes = Counter()
    for path in files:
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                outcome = row['outcome']
                if outcome not in ('timely', 'late') or row['cpu'] != row['demand_cpu']:
                    continue
                trigger = events.get((row['cpu'], row['trigger_instr_id'], row['trigger_operand']))
                target = events.get(identity(row, 'demand_'))
                if trigger is None or target is None or target['hit'] != '0':
                    continue
                lead = int(target['ingress_cycle']) - int(trigger['ingress_cycle'])
                if lead >= 0:
                    outcomes[bisect_right(cuts, lead), outcome] += 1
    result = []
    for band, (low, high) in enumerate(bands):
        timely, late = outcomes[band, 'timely'], outcomes[band, 'late']
        used = timely + late
        result.append(dict(min_cycles=low, max_cycles=high, timely=timely, late=late, used=used,
                           timely_fraction=timely / used if used else None))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--baseline-demand', type=Path,
                        help='report live timeliness by matching same-core demand to this NoPF log')
    args = parser.parse_args()
    if args.baseline_demand is not None:
        result = used_by_nopf_lead(args.baseline_demand, args.logs)
        with args.output.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=result[0])
            writer.writeheader()
            writer.writerows(result)
        return
    service, counts = samples(args.logs)
    n = len(service)
    distribution = {str(q): service[math.ceil(n * q / 100) - 1] for q in (10, 25, 50, 75, 90, 95, 99)}
    by_lead = {str(t): sum(value <= t for value in service) / n for t in (150, 300, 400, 500, 700, 1000, 2000)}
    report = {'source': counts, 'service_cycles': service, 'percentile_cycles': distribution,
              'fraction_filled_by_cycles': by_lead}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'service_cycles'}, indent=2))


if __name__ == '__main__':
    main()
