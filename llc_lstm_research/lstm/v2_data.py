#!/usr/bin/env python3
"""Fit a bounded NoPF correlation table and export candidates with lead labels."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict, deque
import json
from pathlib import Path

import numpy as np

from dataset import (CANDIDATE_CAP, FIRST_ANCHOR, LINES_PER_PAGE, PAGE_HISTORY,
                     encode_events, read_demands)


TABLE_HEADER = 'LLC_V2_CORRELATION_V1'
MAX_OFFSETS = 4
LEAD_CUTS = (150, 300, 2001, 4001)
HORIZON = LEAD_CUTS[-1] - 1
EVENT_FEATURES = ('pc_low_byte', 'pc_second_byte', 'position', 'same_page_delta',
                  'page_change', 'same_pc', 'log_cycle_gap')
CANDIDATE_FEATURES = ('source_pc_low_byte', 'source_pc_second_byte', 'source_position',
                      'delta_div_63', 'log_source_cycle_age', 'same_pc')
CLASSES = ('no_future_miss', '0_to_149', '150_to_299', '300_to_2000', '2001_to_4000')


def lookup_keys(current, source, age):
    relation = int(age != 0)
    current_pc = current.pc & 65535
    source_pc = source.pc & 65535
    current_bucket = current.line % LINES_PER_PAGE // 16
    source_bucket = source.line % LINES_PER_PAGE // 16
    lag_bucket = (age - 1).bit_length() if age else 0
    return ((2, current_pc, current_bucket, source_pc, source_bucket, lag_bucket, relation),
            (1, 0, 0, source_pc, source_bucket, 0, relation),
            (0, 0, 0, 0, 0, 0, relation))


def recent_sources(current, history):
    sources = {current.line // LINES_PER_PAGE: (0, current)}
    recent_lines = {current.line}
    for age, source in enumerate(reversed(history), 1):
        recent_lines.add(source.line)
        sources.setdefault(source.line // LINES_PER_PAGE, (age, source))
    return sources, recent_lines


def fit_counts(runs):
    counts = defaultdict(Counter)
    for run in runs:
        cores, _ = read_demands(run)
        for demands in cores.values():
            cycles = [demand.cycle for demand in demands]
            history = deque(maxlen=PAGE_HISTORY)
            for index, demand in enumerate(demands):
                deadline = demand.cycle + HORIZON
                if index >= FIRST_ANCHOR and deadline <= cycles[-1]:
                    sources, recent_lines = recent_sources(demand, history)
                    cutoff = bisect_right(cycles, deadline, index + 1)
                    seen = set()
                    for target in demands[index + 1:cutoff]:
                        if target.line not in seen and target.miss and target.line not in recent_lines:
                            origin = sources.get(target.line // LINES_PER_PAGE)
                            if origin is not None:
                                age, source = origin
                                delta = target.line - source.line
                                for key in lookup_keys(demand, source, age):
                                    counts[key][delta] += 1
                        seen.add(target.line)
                history.append(demand)
    return counts


def compact_table(counts, max_contexts):
    by_level = [[], [], []]
    for key, offsets in counts.items():
        support = sum(offsets.values())
        if key[0] == 0 or support >= 4:
            by_level[key[0]].append((-support, key))
    for rows in by_level:
        rows.sort()
    backoff, source, exact = by_level
    remaining = max_contexts - len(backoff)
    source_count = min(len(source), max((remaining + 2) // 3, remaining - len(exact)))
    chosen = backoff + source[:source_count] + exact[:remaining - source_count]
    table = {}
    for negative_support, key in sorted(chosen, key=lambda row: row[1]):
        support = -negative_support
        ranked = sorted(counts[key].items(), key=lambda row: (-row[1], row[0]))[:MAX_OFFSETS]
        table[key] = tuple((delta, max(1, (255 * count + support // 2) // support))
                           for delta, count in ranked)
    return table


def fit(runs, output, max_contexts):
    table = compact_table(fit_counts(runs), max_contexts)
    if not table:
        raise ValueError('no qualifying address correlations found in the training runs')
    with output.with_suffix('.tmp').open('w') as stream:
        stream.write(TABLE_HEADER + '\n')
        for key, proposals in table.items():
            for delta, strength in proposals:
                stream.write(' '.join(map(str, (*key, delta, strength))) + '\n')
    output.with_suffix('.tmp').replace(output)
    # Fixed-width, low-16 PC keys and four (signed delta, strength) slots.
    # Level partitioning avoids a tag byte; zero-strength slots are unused.
    key_bytes = (1, 3, 5)
    estimate = sum(key_bytes[key[0]] + MAX_OFFSETS * 2 for key in table)
    report = {'inputs': [run.name for run in runs], 'context_limit': max_contexts,
              'contexts': len(table), 'contexts_by_level': [sum(key[0] == level for key in table)
                                                            for level in range(3)],
              'stored_offsets': sum(map(len, table.values())), 'packed_data_bytes': estimate,
              'max_four_offset_width': MAX_OFFSETS, 'correlation_horizon': HORIZON}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def read_table(path):
    table = defaultdict(list)
    with path.open() as stream:
        if stream.readline().strip() != TABLE_HEADER:
            raise ValueError('unexpected correlation table header')
        for row in stream:
            level, current_pc, current_pos, source_pc, source_pos, lag, relation, delta, strength = map(int, row.split())
            key = (level, current_pc, current_pos, source_pc, source_pos, lag, relation)
            if (level not in (0, 1, 2) or not 0 <= current_pc <= 65535 or not 0 <= source_pc <= 65535
                    or not 0 <= current_pos <= 3 or not 0 <= source_pos <= 3 or not 0 <= lag <= 5
                    or relation not in (0, 1) or not -63 <= delta <= 63 or delta == 0
                    or not 1 <= strength <= 255 or len(table[key]) == MAX_OFFSETS
                    or any(previous == delta for previous, _ in table[key])):
                raise ValueError('invalid correlation table row')
            table[key].append((delta, strength))
    if not table:
        raise ValueError('empty correlation table')
    return table


def correlated_candidates(current, history, table):
    sources, recent_lines = recent_sources(current, history)
    proposals = []
    for page, (age, source) in sources.items():
        offsets = ()
        for key in lookup_keys(current, source, age):
            if key in table:
                offsets = table[key]
                break
        for rank, (delta, strength) in enumerate(offsets):
            line = source.line + delta
            if line > 0 and line not in recent_lines and line // LINES_PER_PAGE == page:
                proposals.append((rank, -strength, age, delta, line, source))
    proposals.sort()
    for _, _, age, delta, line, source in proposals[:CANDIDATE_CAP]:
        yield line, source, age, delta


def cycle_feature(gap):
    return (min(gap + 1, 4096).bit_length() - 1) / 12


def export(run, output, table, table_name):
    cores, summary = read_demands(run)
    num_events = sum(map(len, cores.values()))
    shape = (num_events, CANDIDATE_CAP)
    events = np.zeros((num_events, len(EVENT_FEATURES)), np.float32)
    candidates = np.zeros((*shape, len(CANDIDATE_FEATURES)), np.float32)
    keys = np.zeros((*shape, 6), np.int64)
    valid = np.zeros(shape, np.bool_)
    targets = np.zeros(shape, np.int32)
    future_targets = np.zeros(shape, np.int32)
    lines = np.zeros(shape, np.uint64)
    lead_cycles = np.zeros(shape, np.uint16)
    lead_class = np.zeros(shape, np.uint8)
    age = np.zeros(shape, np.uint8)
    source_pos = np.zeros(shape, np.uint8)
    event_id = np.zeros(num_events, np.int32)
    event_pc = np.zeros(num_events, np.int64)
    event_line = np.zeros(num_events, np.uint64)
    event_cycle = np.zeros(num_events, np.uint64)
    event_instr = np.zeros(num_events, np.int64)
    event_operand = np.zeros(num_events, np.uint8)
    anchors = np.zeros(num_events, np.bool_)
    core = np.zeros(num_events, np.int16)
    sequence = np.zeros(num_events, np.int32)
    base = 0
    for cpu, demands in cores.items():
        by_line = defaultdict(list)
        for demand in demands:
            by_line[demand.line].append(demand)
        event_ids = {line: [demand.event for demand in visits] for line, visits in by_line.items()}
        history = deque(maxlen=PAGE_HISTORY)
        for index, demand in enumerate(demands):
            row = base + index
            event_id[row] = demand.event
            event_pc[row] = demand.pc
            event_line[row] = demand.line
            event_cycle[row] = demand.cycle
            event_instr[row] = demand.instruction
            event_operand[row] = demand.operand
            core[row] = cpu
            sequence[row] = index
            anchors[row] = index >= FIRST_ANCHOR and demand.cycle + HORIZON <= demands[-1].cycle
            if anchors[row]:
                proposals = correlated_candidates(demand, history, table)
                for slot, (line, source, lag, delta) in enumerate(proposals):
                    valid[row, slot] = True
                    lines[row, slot] = line
                    age[row, slot] = lag
                    position = source.line % LINES_PER_PAGE
                    source_pos[row, slot] = position
                    trigger_pc = demand.pc & 0xffffffff
                    source_pc = source.pc & 0xffffffff
                    keys[row, slot] = (delta, (lag - 1).bit_length() if lag else 0, trigger_pc,
                                       demand.line % LINES_PER_PAGE // 16, position // 16, source_pc)
                    candidates[row, slot] = ((source.pc & 255) / 255, ((source.pc >> 8) & 255) / 255,
                                             position / 63, delta / 63,
                                             cycle_feature(demand.cycle - source.cycle), source_pc == trigger_pc)
                    visits = event_ids.get(line, ())
                    next_index = bisect_right(visits, demand.event)
                    if next_index < len(visits):
                        target = by_line[line][next_index]
                        gap = target.cycle - demand.cycle
                        if gap <= HORIZON:
                            lead_cycles[row, slot] = gap
                            if target.miss:
                                future_targets[row, slot] = target.event
                                lead_class[row, slot] = 1 + bisect_right(LEAD_CUTS, gap)
                                if 300 <= gap <= 2000:
                                    targets[row, slot] = target.event
            history.append(demand)
        base += len(demands)
    encode_events(events, event_pc & 0xffffffff, event_line, core)
    events[:, 6] = 0
    for row in range(1, num_events):
        if core[row] == core[row - 1]:
            events[row, 6] = cycle_feature(int(event_cycle[row]) - int(event_cycle[row - 1]))
    labels = targets != 0
    future_miss = future_targets != 0
    counts = np.bincount(lead_class[valid], minlength=len(CLASSES))
    metadata = {'version': 2, 'no_prefetch': True, 'cores': sorted(cores), 'source': run.name,
                'events': num_events, 'anchors': int(anchors.sum()),
                'candidate_anchors': int(valid.any(1).sum()), 'candidates': int(valid.sum()),
                'positives': int(labels.sum()), 'unique_targets': len(np.unique(targets[labels])),
                'future_misses': int(future_miss.sum()),
                'unique_future_misses': len(np.unique(future_targets[future_miss])),
                'unique_future_misses_by_class': [len(np.unique(future_targets[lead_class == cls]))
                                                  for cls in range(1, len(CLASSES))],
                'lead_cycles': [300, 2000], 'lead_cuts': list(LEAD_CUTS),
                'classes': list(CLASSES), 'class_counts': counts.tolist(),
                'first_anchor': FIRST_ANCHOR, 'candidate_cap': CANDIDATE_CAP, 'page_history': PAGE_HISTORY,
                'event_features': list(EVENT_FEATURES), 'candidate_features': list(CANDIDATE_FEATURES),
                'target_definition': 'First later same-core demand to a candidate line misses LLC under NoPF.',
                'candidate_source': 'NoPF-fitted per-context offset table', 'correlation_table': table_name,
                'trace_seed': summary['trace_seed'], 'llc_misses': summary['llc_demand_misses'],
                'same_page_candidates': int(np.count_nonzero(valid & (age == 0)))}
    with output.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream, events=events, candidates=candidates, valid=valid, labels=labels, targets=targets,
                            lines=lines, lead_cycles=lead_cycles, keys=keys, lead_class=lead_class,
                            future_miss=future_miss, future_targets=future_targets, age=age,
                            offset=keys[:, :, 0].astype(np.int8), source_pos=source_pos, source_pc=keys[:, :, 5],
                            cores=core, sequence=sequence, anchors=anchors, event_pc=event_pc, event_id=event_id,
                            event_line=event_line, event_cycle=event_cycle, event_instr=event_instr,
                            event_operand=event_operand, metadata=json.dumps(metadata, sort_keys=True))
    output.with_suffix('.tmp').replace(output)
    output.with_suffix('.json').write_text(json.dumps(metadata, sort_keys=True, indent=2) + '\n')
    print(json.dumps(metadata, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit_parser = commands.add_parser('fit')
    fit_parser.add_argument('output', type=Path)
    fit_parser.add_argument('train_runs', nargs='+', type=Path)
    fit_parser.add_argument('--contexts', type=int, default=64)
    export_parser = commands.add_parser('export')
    export_parser.add_argument('run', type=Path)
    export_parser.add_argument('output', type=Path)
    export_parser.add_argument('--table', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == 'fit':
        if not 2 <= args.contexts <= 2048:
            parser.error('contexts must be between 2 and 2,048')
        fit(args.train_runs, args.output, args.contexts)
    else:
        export(args.run, args.output, read_table(args.table), args.table.name)


if __name__ == '__main__':
    main()
