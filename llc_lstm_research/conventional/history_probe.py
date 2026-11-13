#!/usr/bin/env python3
"""Compare fixed count lookups on two disjoint LLC demand logs."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from random import Random
from statistics import mean
from typing import NamedTuple


WINDOWS = (1, 2, 3, 4, 8, 16, 32)
OFFSETS = (1, 2, 3, 4, 6)
POLICIES = ('streamer', 'spp')
PRIOR = 20
SHUFFLE_SEEDS = (11, 23, 37, 47, 59)


class Demand(NamedTuple):
    event: int
    cycle: int
    core: int
    pc: int
    line: int
    identity: tuple
    virtual_line: int
    miss: bool
    late: bool


def read_demands(directory, policy):
    with (directory / 'summary.json').open() as source:
        summary = next(row for row in json.load(source) if row['policy'] == policy)
    events_path = (directory / summary['log']).with_name('demand.csv')
    events = []
    with events_path.open() as source:
        for row in csv.DictReader(source):
            core = int(row['cpu'])
            events.append(Demand(
                int(row['event_id']), int(row['ingress_cycle']), core, int(row['pc']),
                int(row['physical_line']), (core, int(row['instr_id']), int(row['operand'])),
                int(row['virtual_line']), row['hit'] == '0', row['prefetch_use'] == 'late',
            ))
    if (len({row.identity for row in events}) != len(events)
            or len({row.event for row in events}) != len(events)
            or len(events) != summary['llc_demand_accesses']
            or sum(row.miss for row in events) != summary['llc_demand_misses']
            or sum(row.late for row in events) != summary['prefetch_late']):
        raise ValueError('{} disagrees with its summary or has duplicate IDs'.format(events_path))
    return sorted(events, key=lambda row: row.event)


def anchors(events, offsets, low, high, deltas_only):
    by_line = defaultdict(list)
    end = {}
    for row in events:
        by_line[row.line].append(row)
        end[row.core] = row.cycle
    line_ids = {line: [row.event for row in rows] for line, rows in by_line.items()}
    previous_line = {}
    history = defaultdict(list)
    for row in events:
        previous = previous_line.get(row.core)
        previous_line[row.core] = row.line
        if previous is None:
            continue
        delta = max(-8, min(8, row.line - previous)) if row.line // 64 == previous // 64 else 9
        token = (delta,) if deltas_only else (row.pc, delta)
        sequence = history[row.core]
        sequence.append(token)
        if len(sequence) < max(WINDOWS) or row.cycle + high > end[row.core]:
            continue
        candidates = []
        for offset in offsets:
            line = row.line + offset
            if line // 64 != row.line // 64:
                continue
            index = bisect_right(line_ids.get(line, ()), row.event)
            matches = by_line.get(line, ())
            target = None
            if index < len(matches):
                next_row = matches[index]
                if next_row.miss and low <= next_row.cycle - row.cycle <= high:
                    target = next_row
            candidates.append((offset, target))
        if candidates:
            yield row, (row.pc, row.line % 64, delta), tuple(sequence[-max(WINDOWS):]), candidates


def history_keys(rows, mode, seed):
    rng = Random(seed)
    for row, current, sequence, candidates in rows:
        keys = [(), (sequence[-2],)]
        priorities = [rng.random() for _ in sequence] if mode == 'shuffled' else None
        for width in WINDOWS[2:]:
            older = list(sequence[-width:-2])
            if mode == 'shuffled':
                older = [token for _, token in sorted(zip(priorities[-width:-2], older))]
            if mode == 'sorted':
                older.sort()
            keys.append(tuple(older) + (sequence[-2],))
        yield row, current, keys, candidates


def count_rates(rows, late_targets):
    global_pos, global_total = Counter(), Counter()
    positive = [Counter() for _ in WINDOWS]
    total = [Counter() for _ in WINDOWS]
    for _, current, keys, candidates in rows:
        for offset, target in candidates:
            is_positive = target is not None and (late_targets is None or target.identity in late_targets)
            global_pos[offset] += is_positive
            global_total[offset] += 1
            for index, key in enumerate(keys):
                key = (offset, current, key)
                positive[index][key] += is_positive
                total[index][key] += 1
    return global_pos, global_total, positive, total


def ranked_decisions(rows, rates):
    global_pos, global_total, positive, total = rates
    rankings = [[] for _ in WINDOWS]
    known = [0] * len(WINDOWS)
    ties = Random(19)
    for row, current, keys, candidates in rows:
        tie_break = ties.random()
        winners = [None] * len(WINDOWS)
        for offset, target in candidates:
            score = (global_pos[offset] + 1) / (global_total[offset] + 2)
            for index, key in enumerate(keys):
                key = (offset, current, key)
                support = total[index][key]
                score = (positive[index][key] + PRIOR * score) / (support + PRIOR)
                decision = (-score, tie_break, offset, row, target, support)
                if winners[index] is None or decision[:3] < winners[index][:3]:
                    winners[index] = decision
        for index, decision in enumerate(winners):
            rankings[index].append(decision)
            known[index] += decision[-1] >= 5
    return [sorted(choices, key=lambda decision: decision[:3]) for choices in rankings], known


def matched_late(reference, comparison):
    reference_by_id = {row.identity: row for row in reference}
    by_id = {row.identity: row for row in comparison}
    for key in reference_by_id.keys() & by_id.keys():
        expected, actual = reference_by_id[key], by_id[key]
        if (expected.pc, expected.virtual_line) != (actual.pc, actual.virtual_line):
            raise ValueError('demand {} changed PC or virtual line'.format(key))
    return {row.identity for row in comparison
            if row.late and row.identity in reference_by_id and reference_by_id[row.identity].miss}, by_id


def lead_in_policy(decision, comparison, low, high):
    _, _, offset, anchor, target, _ = decision
    by_id = comparison[1]
    start = by_id.get(anchor.identity)
    end = by_id[target.identity]
    return (start is not None and start.event < end.event and start.line + offset == end.line
            and low <= end.cycle - start.cycle <= high)


def summarize(decisions, comparisons, low, high):
    successful = [decision for decision in decisions if decision[4] is not None]
    targets = {decision[4].identity for decision in successful}
    result = {'all_misses': len(targets)}
    union = set()
    for policy, comparison in comparisons.items():
        matched = targets & comparison[0]
        union |= matched
        result[policy + '_late'] = len(matched)
        result[policy + '_rechecked'] = len({decision[4].identity for decision in successful
                                            if decision[4].identity in matched
                                            and lead_in_policy(decision, comparison, low, high)})
    result['either_late'] = len(union)
    return result


def run(development, evaluation, budgets, offsets, low, high, deltas_only, late_only):
    dev_events = read_demands(development, 'nopf')
    dev = list(anchors(dev_events, offsets, low, high, deltas_only))
    eval_events = read_demands(evaluation, 'nopf')
    check = list(anchors(eval_events, offsets, low, high, deltas_only))
    if not dev or not budgets or max(budgets) > len(check):
        raise ValueError('the logs do not have enough eligible decisions')
    comparisons = {policy: matched_late(eval_events, read_demands(evaluation, policy)) for policy in POLICIES}
    for policy, (matched, _) in comparisons.items():
        print('{} matched late targets: {}'.format(policy, len(matched)))
    late_targets = None
    if late_only:
        late_targets = set()
        for policy in POLICIES:
            matched, _ = matched_late(dev_events, read_demands(development, policy))
            late_targets.update(matched)
    oracle = [(-1, row.event, offset, row, target, 0)
              for row, _, _, choices in check for offset, target in choices]
    print('Candidate reach without an issue cap: {}'.format(summarize(oracle, comparisons, low, high)))
    print('Eligible development/evaluation events: {}/{}'.format(len(dev), len(check)))
    output = defaultdict(list)
    for mode, seeds in (('ordered', (0,)), ('sorted', (0,)), ('shuffled', SHUFFLE_SEEDS)):
        for seed in seeds:
            rates = count_rates(history_keys(dev, mode, seed), late_targets)
            rankings, known = ranked_decisions(history_keys(check, mode, seed + 1000), rates)
            for width, choices, support in zip(WINDOWS, rankings, known):
                if mode != 'ordered' and width <= 3:
                    continue
                for budget in budgets:
                    metrics = summarize(choices[:budget], comparisons, low, high)
                    metrics['known_choices'] = support
                    output[(budget, width, mode)].append(metrics)
    for (budget, width, mode), results in sorted(output.items()):
        aggregate = {key: mean(row[key] for row in results) for key in results[0]}
        yield {'budget': budget, 'window': width, 'history': mode, **aggregate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('development', type=Path)
    parser.add_argument('evaluation', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget', type=int, nargs='+', default=[1000, 2000])
    parser.add_argument('--offsets', type=int, nargs='+', default=list(OFFSETS))
    parser.add_argument('--lead', type=int, nargs=2, default=[300, 2000], metavar=('MIN', 'MAX'))
    parser.add_argument('--deltas-only', action='store_true', help='omit PCs from older event tokens')
    parser.add_argument('--late-only', action='store_true', help='count only the 10M baseline-late opportunities')
    args = parser.parse_args()
    low, high = args.lead
    if not 0 <= low <= high or min(args.budget) < 1 or any(not 0 < abs(d) < 64 for d in args.offsets):
        parser.error('invalid lead bounds, decision budget, or same-page offsets')
    rows = list(run(args.development, args.evaluation, args.budget, args.offsets, low, high,
                    args.deltas_only, args.late_only))
    with args.output.open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
