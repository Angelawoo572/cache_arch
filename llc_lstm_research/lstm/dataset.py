#!/usr/bin/env python3
"""Export causal NoPF features, bounded page candidates and binary labels."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict, deque
import csv
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np


LINES_PER_PAGE = 64
PAGE_HISTORY = 32
FIRST_ANCHOR = 128
CANDIDATE_CAP = 16
AGE_BAND_ORDER = (3, 2, 1, 0, 4, 5, 6, 7)
OFFSETS = (1, 2, -1, 3, -2, 4, -3, 6, -4, -6)
LEAD_CYCLES = (300, 2000)
EVENT_FEATURES = ('pc_low_byte', 'pc_second_byte', 'position', 'same_page_delta',
                  'page_change', 'same_pc', 'same_line')
CANDIDATE_FEATURES = ('source_pc_low_byte', 'source_pc_second_byte', 'source_position',
                      'delta', 'age', 'same_pc')


class Demand(NamedTuple):
    event: int
    cycle: int
    pc: int
    line: int
    miss: bool
    instruction: int
    operand: int


def read_demands(run):
    with (run / 'summary.json').open() as source:
        summary = next(row for row in json.load(source) if row['policy'] == 'nopf')
    if summary['prefetch_issued']:
        raise ValueError('{} does not provide a clean NoPF stream'.format(run))
    path = (run / summary['log']).with_name('demand.csv')
    cores = defaultdict(list)
    with path.open() as source:
        for row in csv.DictReader(source):
            if row['hit'] not in ('0', '1') or row['prefetch_use'] != 'none':
                raise ValueError('invalid NoPF outcome in {}'.format(path))
            cores[int(row['cpu'])].append(Demand(
                int(row['event_id']), int(row['ingress_cycle']), int(row['pc']),
                int(row['physical_line']), row['hit'] == '0', int(row['instr_id']), int(row['operand']),
            ))
    events = [row for rows in cores.values() for row in rows]
    if (len(events) != summary['llc_demand_accesses']
            or sum(row.miss for row in events) != summary['llc_demand_misses']
            or len({row.event for row in events}) != len(events)):
        raise ValueError('{} does not match its NoPF summary'.format(path))
    for cpu, rows in cores.items():
        rows.sort(key=lambda row: row.event)
        if any(left.cycle > right.cycle for left, right in zip(rows, rows[1:])):
            raise ValueError('core {} has out-of-order ingress times'.format(cpu))
    return dict(sorted(cores.items())), summary


def recent_candidates(history, current_page, limit=CANDIDATE_CAP):
    """Return page-diverse candidates in the same order as the simulator."""
    pages = {current_page}
    lines = set()
    bands = [[] for _ in AGE_BAND_ORDER]
    for age, source in enumerate(reversed(history), 1):
        page = source.line // LINES_PER_PAGE
        lines.add(source.line)
        if page in pages:
            continue
        pages.add(page)
        bands[(age - 1) // 4].append((age, source))
    sources = [bands[band][rank] for rank in range(4) for band in AGE_BAND_ORDER
               if rank < len(bands[band])]
    count = 0
    for delta in OFFSETS:
        for age, source in sources:
            line = source.line + delta
            if (line <= 0 or line in lines
                    or line // LINES_PER_PAGE != source.line // LINES_PER_PAGE):
                continue
            yield line, source, age, delta
            count += 1
            if count == limit:
                return


def encode_events(events, pc, line, core):
    events[:, 0] = (pc & 255) / 255
    events[:, 1] = ((pc >> 8) & 255) / 255
    events[:, 2] = (line % LINES_PER_PAGE) / 63
    events[:, 4] = 1
    same_core = core[1:] == core[:-1]
    same_page = (line[1:] // LINES_PER_PAGE == line[:-1] // LINES_PER_PAGE) & same_core
    delta = (line[1:] % LINES_PER_PAGE).astype(np.int8) - (line[:-1] % LINES_PER_PAGE).astype(np.int8)
    events[1:, 3] = np.where(same_page, np.clip(delta, -8, 8) / 8, 0)
    events[1:, 4] = ~same_page
    events[1:, 5] = (pc[1:] == pc[:-1]) & same_core
    events[1:, 6] = same_page & (delta == 0)


def encode_candidates(candidates, keys, position, age):
    pc = keys[:, :, 5]
    candidates[:, :, 0] = (pc & 255) / 255
    candidates[:, :, 1] = ((pc >> 8) & 255) / 255
    candidates[:, :, 2] = position / 63
    candidates[:, :, 3] = keys[:, :, 0] / 6
    candidates[:, :, 4] = age / PAGE_HISTORY
    candidates[:, :, 5] = pc == keys[:, :, 2]


def export(run, output, lead):
    cores, summary = read_demands(run)
    num_events = sum(map(len, cores.values()))
    events = np.zeros((num_events, len(EVENT_FEATURES)), np.float32)
    shape = num_events, CANDIDATE_CAP
    candidates = np.zeros((*shape, len(CANDIDATE_FEATURES)), np.float32)
    keys = np.zeros((*shape, 6), np.int64)
    valid = np.zeros(shape, np.bool_)
    targets = np.zeros(shape, np.int32)
    lines = np.zeros(shape, np.uint64)
    lead_cycles = np.zeros(shape, np.uint16)
    age = np.zeros(shape, np.uint8)
    source_pos = np.zeros(shape, np.uint8)
    event_id = np.zeros(num_events, np.int32)
    event_pc = np.zeros(num_events, np.int64)
    event_line = np.zeros(num_events, np.uint64)
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
            event_instr[row] = demand.instruction
            event_operand[row] = demand.operand
            core[row] = cpu
            sequence[row] = index
            anchors[row] = index >= FIRST_ANCHOR and demand.cycle + lead[1] <= demands[-1].cycle
            if anchors[row]:
                page, position = divmod(demand.line, LINES_PER_PAGE)
                for slot, (line, source, lag, delta) in enumerate(recent_candidates(history, page)):
                    valid[row, slot] = True
                    lines[row, slot] = line
                    age[row, slot] = lag
                    source_pos[row, slot] = source.line % LINES_PER_PAGE
                    keys[row, slot] = (delta, (lag - 1).bit_length(), demand.pc,
                                       position // 16, source_pos[row, slot] // 16, source.pc)
                    visits = by_line.get(line, ())
                    next_index = bisect_right(event_ids.get(line, ()), demand.event)
                    if next_index < len(visits):
                        target = visits[next_index]
                        cycles = target.cycle - demand.cycle
                        if cycles <= lead[1]:
                            lead_cycles[row, slot] = cycles
                            if target.miss and lead[0] <= cycles:
                                targets[row, slot] = target.event
            history.append(demand)
        base += len(demands)
    encode_events(events, event_pc, event_line, core)
    encode_candidates(candidates, keys, source_pos, age)
    candidates[~valid] = 0
    labels = targets != 0
    metadata = {
        'version': 1,
        'source': run.name,
        'no_prefetch': True,
        'cores': sorted(cores),
        'events': num_events,
        'anchors': int(anchors.sum()),
        'candidate_anchors': int(valid.any(1).sum()),
        'candidates': int(valid.sum()),
        'positives': int(labels.sum()),
        'unique_targets': len(np.unique(targets[labels])),
        'lead_cycles': list(lead),
        'first_anchor': FIRST_ANCHOR,
        'candidate_cap': CANDIDATE_CAP,
        'page_history': PAGE_HISTORY,
        'offset_order': list(OFFSETS),
        'age_band_order': list(AGE_BAND_ORDER),
        'event_features': list(EVENT_FEATURES),
        'candidate_features': list(CANDIDATE_FEATURES),
        'target_definition': 'First later same-core demand to candidate line LLC-misses within the lead window.',
        'trace_seed': summary['trace_seed'],
    }
    with output.with_suffix('.tmp').open('wb') as target:
        np.savez_compressed(target, events=events, candidates=candidates, valid=valid, labels=labels,
                            targets=targets, lines=lines, lead_cycles=lead_cycles, keys=keys,
                            age=age, offset=keys[:, :, 0].astype(np.int8), source_pos=source_pos,
                            source_pc=keys[:, :, 5], cores=core, sequence=sequence, anchors=anchors,
                            event_pc=event_pc, event_id=event_id, event_line=event_line,
                            event_instr=event_instr, event_operand=event_operand,
                            metadata=json.dumps(metadata, sort_keys=True))
    output.with_suffix('.tmp').replace(output)
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    print(json.dumps(metadata, sort_keys=True), flush=True)


def table(output, datasets):
    counts = [Counter() for _ in (2, 5, 6)]
    for file in datasets:
        with np.load(file, allow_pickle=False) as data:
            keys, valid, labels = data['keys'], data['valid'], data['labels']
        for index in range(0, len(keys), 8192):
            sl = slice(index, index + 8192)
            rows = keys[sl][valid[sl]]
            positive = labels[sl][valid[sl]]
            unique, inverse, total = np.unique(rows, axis=0, return_inverse=True, return_counts=True)
            successes = np.bincount(inverse, weights=positive, minlength=len(total)).astype(np.int64)
            for row, successes_count, total_count in zip(unique.tolist(), successes.tolist(), total.tolist()):
                for width, counter in zip((2, 5, 6), counts):
                    key = tuple(row[:width])
                    p, n = counter.get(key, (0, 0))
                    counter[key] = p + successes_count, n + total_count
    with output.with_suffix('.tmp').open('w') as target:
        target.write('LLC_PAGE_PC_V1\n')
        for width, counter in zip((2, 5, 6), counts):
            for key, (p, n) in sorted(counter.items()):
                target.write(' '.join(map(str, (width, *key, p, n))) + '\n')
    output.with_suffix('.tmp').replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    create = subparsers.add_parser('export')
    create.add_argument('run', type=Path)
    create.add_argument('output', type=Path)
    create.add_argument('--lead', nargs=2, type=int, default=LEAD_CYCLES)
    lookup = subparsers.add_parser('table')
    lookup.add_argument('output', type=Path)
    lookup.add_argument('datasets', nargs='+', type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == 'export':
        if not 0 < args.lead[0] <= args.lead[1] <= np.iinfo(np.uint16).max:
            parser.error('lead window must be positive, ordered, and fit uint16')
        export(args.run, args.output, args.lead)
    else:
        table(args.output, args.datasets)


if __name__ == '__main__':
    main()
