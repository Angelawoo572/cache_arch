#!/usr/bin/env python3
"""Test history beyond the same page-PC fields on a separate NoPF region."""

import argparse
from collections import Counter, deque
import csv
from random import Random
from statistics import mean
from pathlib import Path

from long_memory_probe import LINES_PER_PAGE, OFFSETS, PAGE_HISTORY, WARMUP_EVENTS, read_demands


WINDOWS = (8, 16, 32)
SEEDS = (10, 20, 30)
BASE = 'page_pc'
MODES = (BASE,) + tuple((order, window, seed) for window in WINDOWS
                       for order, seeds in (('ordered', (0,)), ('shuffled', SEEDS))
                       for seed in seeds)


def summaries(tokens, permutations):
    # Keep the most recent token. Reordering never changes the token counts.
    result = []
    for order, seed in permutations:
        prior = tokens[:-1]
        if order == 'shuffled':
            prior = [token for _, token in sorted(zip(seed[:len(prior)], prior))]
        sequence = prior + tokens[-1:]
        if len(sequence) < 2:
            result.append(None)
            continue
        switches = sum(left != right for left, right in zip(sequence, sequence[1:]))
        middle = sequence[len(sequence) // 2]
        result.append((sequence[0], middle, sequence[-1], min(3, switches * 4 // len(sequence))))
    return result


def contexts(interior, source, current, seeds):
    # Distinguish page revisits and recurring PCs without using an address as a label.
    tokens = [(page == current[0], pc == source[1], pc == current[1]) for page, pc in interior]
    result = {}
    for window in WINDOWS:
        suffix = tokens[-(window - 1):]
        permutations = [('ordered', ())] + [('shuffled', priorities) for priorities in seeds]
        values = summaries(suffix, permutations)
        result['ordered', window, 0] = values[0]
        for seed, value in zip(SEEDS, values[1:]):
            result['shuffled', window, seed] = value
    return result


def examples(directory, stride, lead):
    randomizers = [Random(seed) for seed in SEEDS]
    for demands in read_demands(directory):
        first = {}
        for demand in demands:
            first.setdefault(demand.line, demand)
        seen = set()
        prior = deque(maxlen=PAGE_HISTORY)
        for i, demand in enumerate(demands):
            page, pos = divmod(demand.line, LINES_PER_PAGE)
            if i >= WARMUP_EVENTS and i % stride == 0 and demand.cycle + lead[1] <= demands[-1].cycle:
                directory_entries = {}
                preceding = list(prior)
                for j, previous in enumerate(reversed(preceding), 1):
                    old_page = previous.line // LINES_PER_PAGE
                    if old_page != page:
                        directory_entries.setdefault(old_page, (j, previous))
                candidates = []
                for old_page, (age, previous) in directory_entries.items():
                    position = previous.line % LINES_PER_PAGE
                    choices = []
                    for delta in OFFSETS:
                        line = previous.line + delta
                        if line // LINES_PER_PAGE != old_page or line in seen:
                            continue
                        future = first.get(line)
                        positive = (future is not None and future.miss
                                    and lead[0] <= future.cycle - demand.cycle <= lead[1])
                        base = (delta, (age - 1).bit_length(), demand.pc, pos // 16,
                                position // 16, previous.pc)
                        choices.append((base, line, future.event if positive else 0))
                    if choices:
                        interior = [(entry.line // LINES_PER_PAGE, entry.pc)
                                    for entry in preceding[len(preceding) - age + 1:]]
                        seeds = [[rng.random() for _ in range(len(interior))] for rng in randomizers]
                        features = contexts(interior, (old_page, previous.pc), (page, demand.pc), seeds)
                        candidates.extend((base, features, target, line, age)
                                          for base, line, target in choices)
                if candidates:
                    yield candidates
            seen.add(demand.line)
            prior.append(demand)


def increment(total, positive, key, target):
    total[key] += 1
    if target:
        positive[key] += 1


def count(directory, stride, lead):
    total, positive = Counter(), Counter()
    for candidates in examples(directory, stride, lead):
        for base, features, target, _, _ in candidates:
            for n in (2, 5, 6):
                increment(total, positive, base[:n], target)
            for mode, feature in features.items():
                if feature is not None:
                    increment(total, positive, (mode, base, feature), target)
    return total, positive


def smoothed(key, parent, total, positive):
    return (positive[key] + 20 * parent) / (total[key] + 20)


def rank(directory, stride, lead, total, positive):
    rng = Random(37)
    results = {mode: [] for mode in MODES}
    for candidates in examples(directory, stride, lead):
        tie = rng.random()
        best = {}
        for base, features, target, line, age in candidates:
            score = 0.005
            for n in (2, 5, 6):
                score = smoothed(base[:n], score, total, positive)
            scores = [(BASE, score, total[base])]
            for mode, feature in features.items():
                key = (mode, base, feature)
                adjusted = smoothed(key, score, total, positive) if feature is not None else score
                scores.append((mode, adjusted, total[key]))
            for mode, value, support in scores:
                order = value, -age, -abs(base[0]), -line
                if mode not in best or order > best[mode][0]:
                    best[mode] = order, target, support
        for mode, (order, target, support) in best.items():
            results[mode].append((order[0], tie, target, support))
    return results


def report(results, budgets):
    for rows in results.values():
        rows.sort(reverse=True)
    if max(budgets) > len(results[BASE]):
        raise ValueError('not enough eligible triggers for the decision budget')
    for budget in budgets:
        baseline = {row[2] for row in results[BASE][:budget] if row[2]}
        for order, windows, seeds in ((BASE, (0,), (0,)), ('ordered', WINDOWS, (0,)),
                                       ('shuffled', WINDOWS, SEEDS)):
            for window in windows:
                rows = [results[BASE if order == BASE else (order, window, seed)][:budget] for seed in seeds]
                hits = [{row[2] for row in choices if row[2]} for choices in rows]
                yield {
                    'decisions': budget,
                    'method': order,
                    'window': window,
                    'unique_misses': mean(len(items) for items in hits),
                    'min_misses': min(map(len, hits)),
                    'max_misses': max(map(len, hits)),
                    'new_vs_table': mean(len(items - baseline) for items in hits),
                    'lost_vs_table': mean(len(baseline - items) for items in hits),
                    'supported_decisions': mean(sum(row[3] >= 5 for row in choices) for choices in rows),
                    'eligible_anchors': len(results[BASE]),
                }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('development', type=Path)
    parser.add_argument('evaluation', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dev-stride', default=4, type=int)
    parser.add_argument('--eval-stride', default=4, type=int)
    parser.add_argument('--lead', nargs=2, default=(300, 2000), type=int)
    parser.add_argument('--budgets', nargs='+', default=(1000, 2000), type=int)
    args = parser.parse_args()
    if min(args.dev_stride, args.eval_stride, min(args.budgets), args.lead[0]) < 1 or args.lead[0] > args.lead[1]:
        parser.error('invalid stride, decision budget, or timing window')
    rates = count(args.development, args.dev_stride, args.lead)
    results = rank(args.evaluation, args.eval_stride, args.lead, *rates)
    rows = list(report(results, args.budgets))
    with args.output.open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
