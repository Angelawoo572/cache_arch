#!/usr/bin/env python3
"""Find the label-based coverage ceiling by enumerating every NoPF candidate."""

import argparse
from collections import defaultdict, deque
import csv
from pathlib import Path

import numpy as np


def matching(graph):
    """Assign each trigger at most one distinct future miss."""
    owner = {}
    match = {}
    for trigger in graph:
        waiting = deque([trigger])
        seen = {trigger}
        parent = {}
        free = None
        while waiting and free is None:
            current = waiting.popleft()
            for target in graph[current]:
                if target in parent:
                    continue
                parent[target] = current
                if target not in owner:
                    free = target
                    break
                next_trigger = owner[target]
                if next_trigger not in seen:
                    seen.add(next_trigger)
                    waiting.append(next_trigger)
        while free is not None:
            current = parent[free]
            previous = match.get(current)
            owner[free] = current
            match[current] = free
            free = previous
    return match


def matching_size(graph):
    return len(matching(graph))


def ceiling(path, lead):
    with np.load(path, allow_pickle=False) as archive:
        targets = archive['future_targets']
        delay = archive['lead_cycles']
        valid = archive['valid']
    low, high = lead
    eligible = valid & (targets > 0) & (delay >= low) & (delay <= high)
    rows, slots = np.nonzero(eligible)
    graph = defaultdict(list)
    for row, target in zip(rows.tolist(), targets[rows, slots].tolist()):
        graph[row].append(target)
    distinct = np.unique(targets[eligible]).size
    return {
        'split': path.stem,
        'lead_min': low,
        'lead_max': high,
        'eligible_triggers': len(graph),
        'eligible_candidates': len(rows),
        'distinct_targets': distinct,
        'max_one_candidate_per_trigger': matching_size(graph),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets', nargs='+', type=Path)
    parser.add_argument('--lead', nargs=2, type=int, action='append',
                        help='inclusive lead window in cycles, repeatable')
    parser.add_argument('--budgets', nargs='+', type=int, default=(2000, 4000))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    windows = args.lead or [(300, 2000)]
    if any(low < 0 or low > high for low, high in windows) or min(args.budgets) < 1:
        parser.error('lead windows must be nonnegative and ordered, and budgets positive')
    results = []
    for path in args.datasets:
        for window in windows:
            row = ceiling(path, window)
            for budget in args.budgets:
                row['max_at_{}'.format(budget)] = min(budget, row['max_one_candidate_per_trigger'])
            results.append(row)
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)


if __name__ == '__main__':
    main()
