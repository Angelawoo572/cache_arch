#!/usr/bin/env python3
"""Collect unadjusted LLC demand service samples from verified NoPF replays."""

import argparse
import csv
import json
from pathlib import Path


def collect_service(paths):
    counts = {}
    service = []
    for path in paths:
        command = json.loads(path.with_name('command.json').read_text())
        environment = command['environment']
        if (environment.get('LLC_PF_POLICY') != 'nopf'
                or environment.get('LLC_PF_DEMAND_SERVICE') != path.name):
            raise ValueError(f'service log is not from a documented NoPF replay: {path}')
        samples = []
        with path.open(newline='') as source:
            for row in csv.DictReader(source):
                ingress, dispatch, fill = (int(row[f'{stage}_cycle'])
                                           for stage in ('ingress', 'dispatch', 'fill'))
                if not 0 <= ingress <= dispatch <= fill:
                    raise ValueError(f'invalid demand service clock order in {path}')
                samples.append(fill - ingress)
        if not samples:
            raise ValueError(f'no completed demand services in {path}')
        counts[str(path)] = len(samples)
        service.extend(samples)
    service.sort()
    return {
        'definition': 'Completed NoPF LLC demand ingress-to-fill cycles, without modeled prediction latency',
        'sources': counts,
        'samples': len(service),
        'nearest_rank_cycles': {q: service[(q * len(service) + 99) // 100 - 1]
                                for q in (10, 50, 90, 95, 99)},
        'service_cycles': service,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len({path.resolve() for path in args.logs}) != len(args.logs):
        parser.error('each source log must be distinct')
    data = collect_service(args.logs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, separators=(',', ':')) + '\n')
    print(json.dumps({key: value for key, value in data.items() if key != 'service_cycles'}, indent=2))


if __name__ == '__main__':
    main()
