#!/usr/bin/env python3
"""Measure exploratory cache-line offset associations in logged NoPF demand."""

from bisect import bisect_right
from collections import Counter, defaultdict, deque
import csv
from pathlib import Path
import sys

LSTM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LSTM))
from dataset import FIRST_ANCHOR, LINES_PER_PAGE, PAGE_HISTORY, read_demands
from v2_data import HORIZON

RUNS = LSTM.parent / 'conventional' / 'runs'
TRAIN = (10, 30, 50)
TEST = (410, 430, 450)
EXAMPLE_PC = 0x40245C
CONTEXTS = ('all_eligible', 'example_pc', 'example_pc_advance_1', 'example_pc_other_context')
PAIRED_CONTEXT = 'example_pc_advance_1_both_legal'


def is_legal(target, line, recent):
    return target // LINES_PER_PAGE == line // LINES_PER_PAGE and target not in recent


def is_first_use_miss(events, positions, index, target):
    visits = positions.get(target, ())
    next_index = bisect_right(visits, index)
    if next_index == len(visits):
        return False
    future = events[visits[next_index]]
    return future.miss and future.cycle - events[index].cycle <= HORIZON


def count_offsets(run_paths):
    eligible = Counter()
    misses = Counter()
    for run_path in run_paths:
        cores, _ = read_demands(run_path)
        for events in cores.values():
            positions = defaultdict(list)
            recent = deque(maxlen=PAGE_HISTORY)
            for index, event in enumerate(events):
                positions[event.line].append(index)
            for index, event in enumerate(events):
                target = event.line + 1
                if (index >= FIRST_ANCHOR and event.cycle + HORIZON <= events[-1].cycle
                        and is_legal(target, event.line, recent)):
                    contexts = [('all_eligible', 1)]
                    if event.pc == EXAMPLE_PC:
                        previous = events[index - 1]
                        continued = (previous.pc == event.pc and previous.line + 1 == event.line
                                     and previous.line // LINES_PER_PAGE == event.line // LINES_PER_PAGE)
                        name = 'example_pc_advance_1' if continued else 'example_pc_other_context'
                        contexts.extend((('example_pc', 1), (name, 1)))
                        if continued and is_legal(event.line + 2, event.line, recent):
                            contexts.append((PAIRED_CONTEXT, 1))
                            eligible[PAIRED_CONTEXT, 2] += 1
                            misses[PAIRED_CONTEXT, 2] += is_first_use_miss(events, positions, index, event.line + 2)
                    eligible.update(contexts)
                    if is_first_use_miss(events, positions, index, target):
                        misses.update(contexts)
                recent.append(event.line)
    return eligible, misses


def main():
    # The PC was chosen from a worked example. This is an observation, not a
    # model-selection test or an estimate of realized prefetch timeliness.
    cohorts = (
        ('training', TRAIN, [RUNS / 'v4_comparison' / f'rob_full_{region}m' for region in TRAIN]),
        ('test', TEST, [RUNS / 'v5_clean' / 'label32' / 'fresh' / f'{region}m' for region in TEST]),
    )
    output = Path(__file__).with_name('offset_correlation.csv')
    with output.open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(('cohort', 'regions_m', 'context', 'example_pc', 'target_delta', 'lookahead_cycles',
                         'eligible_anchors', 'first_use_nopf_llc_misses', 'miss_fraction'))
        for cohort, regions, paths in cohorts:
            eligible, misses = count_offsets(paths)
            candidates = [(context, 1) for context in CONTEXTS]
            candidates.extend((PAIRED_CONTEXT, delta) for delta in (1, 2))
            for context, delta in candidates:
                n, k = eligible[context, delta], misses[context, delta]
                writer.writerow((cohort, ' '.join(map(str, regions)), context,
                                 '' if context == 'all_eligible' else hex(EXAMPLE_PC), delta, HORIZON,
                                 n, k, f'{k / n:.8f}'))


if __name__ == '__main__':
    main()
