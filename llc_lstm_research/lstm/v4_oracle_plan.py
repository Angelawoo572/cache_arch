#!/usr/bin/env python3
"""Build a closed-loop reach diagnostic from development NoPF data."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np

from v3_oracle import matching


def load_arrays(path, low, high):
    names = ('event_id', 'event_instr', 'cores', 'event_operand', 'valid', 'lines',
             'future_targets', 'lead_cycles')
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in names}
    lead = arrays['lead_cycles']
    arrays['eligible'] = arrays['valid'] & (arrays['future_targets'] > 0) & (lead >= low) & (lead <= high)
    return arrays


def virtual_targets(path):
    with path.open(newline='') as stream:
        return {int(row['event_id']): (int(row['physical_line']), int(row['virtual_line']), int(row['cpu']))
                for row in csv.DictReader(stream)}


def make_plan(data, degree, preferred, virtual=None):
    graph = {}
    columns = {}
    eligible_rows = np.flatnonzero(data['eligible'].any(axis=1)).tolist()
    for row in eligible_rows:
        slots = np.flatnonzero(data['eligible'][row])
        leads = data['lead_cycles'][row, slots].astype(np.int32)
        slots = slots[np.argsort(np.abs(leads - preferred), kind='stable')]
        targets = data['future_targets'][row, slots].tolist()
        columns[row] = dict(zip(targets, slots.tolist()))
        graph[(row, 0)] = targets
    for slot in range(1, degree):
        for row in eligible_rows:
            if len(graph[(row, 0)]) > slot:
                graph[(row, slot)] = graph[(row, 0)]

    picks = defaultdict(list)
    for (row, _), target in matching(graph).items():
        slot = columns[row][target]
        picks[row].append((abs(int(data['lead_cycles'][row, slot]) - preferred), slot))
    for row in sorted(picks, key=lambda index: int(data['event_id'][index])):
        for score_drop, slot in sorted(picks[row]):
            core = int(data['cores'][row])
            line = int(data['lines'][row, slot])
            if virtual is not None:
                physical, mapped, owner = virtual[int(data['future_targets'][row, slot])]
                if (physical, owner) != (line, core) or (mapped ^ physical) & 63:
                    raise ValueError('demand observation does not match the NoPF candidate target')
                line = mapped
            yield (core, int(data['event_instr'][row]), int(data['event_operand'][row]), line,
                   16384 - min(score_drop, 16000))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--lead', nargs=2, type=int, default=(280, 4000))
    parser.add_argument('--preferred', type=int, default=600)
    parser.add_argument('--degree', type=int, default=1)
    parser.add_argument('--virtual-demand', type=Path, help='original matched NoPF demand.csv for the virtual-key mode')
    args = parser.parse_args()
    low, high = args.lead
    if not 1 <= args.degree <= 16 or not 0 <= low <= args.preferred <= high:
        parser.error('invalid degree or lead window')
    virtual = virtual_targets(args.virtual_demand) if args.virtual_demand else None
    plan = list(make_plan(load_arrays(args.dataset, low, high), args.degree, args.preferred, virtual))
    with args.output.open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerow(('core', 'instr_id', 'operand', 'target_line', 'score'))
        writer.writerows(plan)
    print(f'{len(plan)} distinct development targets')


if __name__ == '__main__':
    main()
