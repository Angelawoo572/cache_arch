#!/usr/bin/env python3
"""Measure physical-line reach on NoPF before choosing a wider candidate generator."""

import argparse
from bisect import bisect_left, bisect_right
from heapq import nsmallest
from collections import Counter, OrderedDict, defaultdict, deque
import csv
import json
from pathlib import Path

import numpy as np

from compare import read_events, start_signature
from dataset import CANDIDATE_CAP, FIRST_ANCHOR, LINES_PER_PAGE, read_demands
from v2_data import (CLASSES, HORIZON, LEAD_CUTS, correlated_candidates, cycle_feature, lookup_keys,
                     read_table, recent_sources)
from v3_oracle import matching_size


PAGE_WINDOWS = (32, 128, 512)
RETRIEVAL_ENTRIES = (128, 512, 2048)
FUTURE_LAGS = (2, 4, 8, 16, 32)
RETRIEVAL_OUTPUTS = 4
POSITION_HEADER = 'LLC_V4_POSITION_V1'
POS_PER_CONTEXT = 8


def source_keys(demand):
    return demand.pc & 65535, demand.line


def causal_retrieval(demands, entries, low, high):
    # One private bounded dictionary per core. Every write uses an actual
    # observed demand as its destination and older observed demands as keys.
    table = OrderedDict()
    recent = deque(maxlen=PAGE_WINDOWS[0])
    for row, event in enumerate(demands):
        key = source_keys(event)
        forbidden = {event.line}
        forbidden.update(observed.line for observed in recent)
        predictions = table.get(key, ())
        if predictions:
            table.move_to_end(key)
        yield tuple(line for line in predictions if line not in forbidden)

        for lag in FUTURE_LAGS:
            if lag > row:
                break
            anchor = demands[row - lag]
            lead = event.cycle - anchor.cycle
            if not low <= lead <= high or anchor.line == event.line:
                continue
            old = source_keys(anchor)
            destinations = table.setdefault(old, [])
            if event.line in destinations:
                destinations.remove(event.line)
            destinations.insert(0, event.line)
            del destinations[RETRIEVAL_OUTPUTS:]
            table.move_to_end(old)
            if len(table) > entries:
                table.popitem(last=False)
        recent.append(event)


def dense_reach(demands, low, high):
    """Oracle pool relaxation: any unobserved line in a recent physical page."""
    cycles = [item.cycle for item in demands]
    end_anchor = bisect_right(cycles, cycles[-1] - high)
    last_line = {}
    by_page = defaultdict(list)
    for row, demand in enumerate(demands):
        page = demand.line // LINES_PER_PAGE
        page_visits = by_page[page]
        previous = last_line.get(demand.line, -1)
        if demand.miss:
            lead_first = bisect_left(cycles, demand.cycle - high, FIRST_ANCHOR, max(FIRST_ANCHOR, row))
            lead_last = min(end_anchor, bisect_right(cycles, demand.cycle - low, 0, row))
            for window in PAGE_WINDOWS:
                # Enforce v2's recent-line restriction when the window expands
                # the page source, while keeping the same 32-line exclusion.
                first = max(lead_first, previous + (PAGE_WINDOWS[0] + 1 if previous >= 0 else 0))
                if first >= lead_last:
                    continue
                source_index = bisect_left(page_visits, max(0, first - window))
                if source_index < len(page_visits) and page_visits[source_index] < lead_last:
                    yield window, demand.event
        page_visits.append(row)
        last_line[demand.line] = row


def uncapped_offsets(current, history, table):
    sources, recent_lines = recent_sources(current, history)
    ranked = []
    for page, (age, source) in sources.items():
        offsets = next((table[key] for key in lookup_keys(current, source, age) if key in table), ())
        for rank, (delta, prior) in enumerate(offsets):
            line = source.line + delta
            if line > 0 and line not in recent_lines and line // LINES_PER_PAGE == page:
                ranked.append((rank, -prior, age, delta, line, source))
    ranked.sort()
    return tuple((line, source, age, delta) for _, _, age, delta, line, source in ranked)


class ProposalBuffer:
    """Track fresh shortlist entries, regardless of which request is sent."""

    def __init__(self, cooldown, capacity):
        self.cooldown = cooldown
        self.capacity = capacity
        self.last_proposed = OrderedDict()

    def choose(self, proposals, now):
        fresh, repeat = [], []
        cutoff = now - self.cooldown
        for proposal in proposals:
            if self.last_proposed.get(proposal[0], cutoff) <= cutoff:
                fresh.append(proposal)
                if len(fresh) == CANDIDATE_CAP:
                    break
            else:
                repeat.append(proposal)
        for line, _, _, _ in fresh:
            self.last_proposed[line] = now
            self.last_proposed.move_to_end(line)
        while len(self.last_proposed) > self.capacity:
            self.last_proposed.popitem(last=False)
        return tuple(fresh + repeat[:CANDIDATE_CAP - len(fresh)])


def candidates_by_event(demands, table, entries, low, high, positions, rotations):
    history = deque(maxlen=PAGE_WINDOWS[0])
    retrievers = {size: causal_retrieval(demands, size, low, high) for size in entries}
    longer_history = deque(maxlen=PAGE_WINDOWS[1])
    buffers = {f'rotate_{cooldown}c_{entries}e': ProposalBuffer(cooldown, entries)
               for cooldown, entries in rotations}
    for row, demand in enumerate(demands):
        predictions = {size: tuple(next(generator)) for size, generator in retrievers.items()}
        if row >= FIRST_ANCHOR and demand.cycle + high <= demands[-1].cycle:
            if buffers:
                all_offsets = uncapped_offsets(demand, history, table)
                original = tuple(proposal[0] for proposal in all_offsets[:CANDIDATE_CAP])
            else:
                original = tuple(line for line, _, _, _ in correlated_candidates(demand, history, table))
            pools = {'v2': original,
                     **{f'retrieval_{size}': predictions[size] for size in entries},
                     **{f'blend_{size}': tuple(dict.fromkeys((*predictions[size], *original)))[:CANDIDATE_CAP]
                        for size in entries}}
            if buffers:
                pools['v2_uncapped'] = tuple(proposal[0] for proposal in all_offsets)
                pools.update((name, tuple(proposal[0] for proposal in buffer.choose(all_offsets, demand.cycle)))
                             for name, buffer in buffers.items())
            if positions is not None:
                for source_history in (history, longer_history):
                    proposed = tuple(option[0] for option in position_candidates(demand, source_history, positions))
                    for slots in (4, 8, 12, 16):
                        kept = original[:CANDIDATE_CAP - slots]
                        pool = tuple(dict.fromkeys((*kept, *proposed, *original)))[:CANDIDATE_CAP]
                        pools[f'pos{source_history.maxlen}_k{slots}'] = pool
            yield row, pools
        history.append(demand)
        longer_history.append(demand)


def target_edges(demands, table, entries, low, high, positions, rotations):
    visits = defaultdict(list)
    for demand in demands:
        visits[demand.line].append(demand)
    event_ids = {line: [target.event for target in line_visits] for line, line_visits in visits.items()}
    graph = defaultdict(dict)
    candidates_seen = defaultdict(int)
    for row, pools in candidates_by_event(demands, table, entries, low, high, positions, rotations):
        candidate_union = set().union(*pools.values())
        demand = demands[row]
        outcomes = {}
        for line in candidate_union:
            ids = event_ids.get(line, ())
            index = bisect_right(ids, demand.event)
            if index < len(ids):
                target = visits[line][index]
                if target.miss and low <= target.cycle - demand.cycle <= high:
                    outcomes[line] = target.event
        for kind, candidates in pools.items():
            candidates_seen[kind] += len(candidates)
            edges = [outcomes[line] for line in candidates if line in outcomes]
            if edges:
                graph[kind][demand.event] = edges
    return graph, candidates_seen


def read_spp(run, nopf_run, nopf_result):
    with (run / 'summary.json').open() as source:
        results = {row['policy']: row for row in json.load(source)}
    with (run / 'config.json').open() as source:
        spp_config = json.load(source)
    with (nopf_run / 'config.json').open() as source:
        nopf_config = json.load(source)
    spp = results['spp']
    if (any(spp_config[key] != nopf_config[key] for key in ('warmup', 'instructions'))
            or start_signature(spp) != start_signature(nopf_result)):
        raise ValueError(f'NoPF and SPP replay configuration do not match for {run}')
    spp_events = read_events((run / spp['log']).with_name('demand.csv'), spp)
    nopf_events = read_events((nopf_run / nopf_result['log']).with_name('demand.csv'), nopf_result)
    if any(nopf_events[key][0] != spp_events[key][0] for key in nopf_events.keys() & spp_events.keys()):
        raise ValueError(f'NoPF and SPP demand identities do not match for {run}')
    by_use = {
        label: {key for key, (_, _, use) in spp_events.items()
                if use == label and key in nopf_events and not nopf_events[key][1]}
        for label in ('timely', 'late')
    }
    return spp, by_use


def summarize(run, table, spp_run, entries, low, high, positions, rotations):
    cores, nopf = read_demands(run)
    identity = {demand.event: (cpu, demand.instruction, demand.operand)
                for cpu, stream in cores.items() for demand in stream}
    reference, buckets = read_spp(spp_run, run, nopf) if spp_run else ({}, {})
    target_graph = defaultdict(dict)
    candidates_seen = defaultdict(int)
    dense = defaultdict(set)
    for demands in cores.values():
        if len(demands) <= FIRST_ANCHOR:
            continue
        graph, candidates = target_edges(demands, table, entries, low, high, positions, rotations)
        for key, count in candidates.items():
            candidates_seen[key] += count
            target_graph[key].update(graph[key])
        for window, target in dense_reach(demands, low, high):
            dense[f'dense_page_{window}'].add(target)
    all_rows = []
    baseline = {target for targets in target_graph['v2'].values() for target in targets}
    for mode in (*target_graph, *dense):
        if mode in dense:
            target_ids = dense[mode]
            matched = None
        else:
            candidates = target_graph[mode]
            target_ids = {target for targets in candidates.values() for target in targets}
            matched = matching_size(candidates)
        target_keys = {identity[event] for event in target_ids}
        all_rows.append({'run': run.name, 'pool': mode, 'low': low, 'high': high,
                         'candidate_edges': candidates_seen.get(mode), 'nopf_misses': nopf['llc_demand_misses'],
                         'unique_misses': len(target_ids), 'max_one_per_trigger': matched,
                         'misses_outside_v2': len(target_ids - baseline), 'spp_requests': reference.get('prefetch_issued'),
                         'spp_matched_timely': len(buckets['timely']) if 'timely' in buckets else None,
                         'spp_matched_late': len(buckets['late']) if 'late' in buckets else None,
                         'reach_spp_timely': len(target_keys & buckets['timely']) if 'timely' in buckets else None,
                         'reach_spp_late': len(target_keys & buckets['late']) if 'late' in buckets else None,
                         'spp_ipc': reference.get('aggregate_ipc'), 'nopf_ipc': nopf['aggregate_ipc']})
    return all_rows


def position_keys(current, source, age):
    current_pc, source_pc = current.pc & 65535, source.pc & 65535
    current_position = current.line % LINES_PER_PAGE
    source_position = source.line % LINES_PER_PAGE
    older = int(age != 0)
    return (
        (2, current_pc, current_position // 16, source_pc, source_position, older),
        (1, 0, 0, source_pc, source_position, older),
        (0, 0, 0, 0, source_position // 16, older),
    )


def fit_positions(runs, output, contexts, window, lead):
    counts = defaultdict(Counter)
    for run in runs:
        cores, _ = read_demands(run)
        for demands in cores.values():
            cycles = [event.cycle for event in demands]
            last_page = {}
            last_line = {}
            for at, current in enumerate(demands):
                current_page = current.line // LINES_PER_PAGE
                last_page[current_page] = at
                last_line[current.line] = at
                if at < FIRST_ANCHOR or current.cycle + HORIZON > demands[-1].cycle:
                    continue
                end = bisect_right(cycles, current.cycle + HORIZON, at + 1)
                later_lines = set()
                for future in demands[at + 1:end]:
                    if future.line in later_lines:
                        continue
                    later_lines.add(future.line)
                    if not future.miss or future.cycle - current.cycle < lead:
                        continue
                    if last_line.get(future.line, -PAGE_WINDOWS[0] - 1) >= at - PAGE_WINDOWS[0]:
                        continue
                    page, position = divmod(future.line, LINES_PER_PAGE)
                    source_at = last_page.get(page)
                    if source_at is None or at - source_at > window:
                        continue
                    source = demands[source_at]
                    for key in position_keys(current, source, at - source_at):
                        counts[key][position] += 1
    backoff = [key for key in counts if key[0] == 0]
    source = sorted((key for key in counts if key[0] == 1),
                    key=lambda key: (-sum(counts[key].values()), key))
    exact = sorted((key for key in counts if key[0] == 2),
                   key=lambda key: (-sum(counts[key].values()), key))
    remaining = contexts - len(backoff)
    source_count = min(len(source), max((remaining + 1) // 2, remaining - len(exact)))
    chosen = backoff + source[:source_count] + exact[:remaining - source_count]
    if not chosen:
        raise ValueError('NoPF training runs contain no qualifying position transitions')
    rows = []
    for key in sorted(chosen):
        distribution = counts[key]
        support = sum(distribution.values())
        top = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))[:POS_PER_CONTEXT]
        rows.extend((*key, position, max(1, (255 * mass + support // 2) // support)) for position, mass in top)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.tmp').open('w') as stream:
        stream.write(POSITION_HEADER + '\n')
        for row in rows:
            stream.write(' '.join(map(str, row)) + '\n')
    output.with_suffix('.tmp').replace(output)
    # Relation is one bit and fits beside the 6-bit source position.
    tag_width = (2, 3, 6)
    fixed_bytes = sum(tag_width[key[0]] + 2 * POS_PER_CONTEXT for key in chosen)
    report = dict(source=[run.name for run in runs], contexts=len(chosen), windows=window, lead=lead,
                  position_slots=POS_PER_CONTEXT, packed_table_bytes=fixed_bytes)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def read_positions(path):
    table = defaultdict(list)
    with path.open() as stream:
        if stream.readline().strip() != POSITION_HEADER:
            raise ValueError('unexpected LLC position table header')
        for row in stream:
            level, current_pc, current_pos, source_pc, source_pos, older, position, prior = map(int, row.split())
            key = level, current_pc, current_pos, source_pc, source_pos, older
            values = table[key]
            if (not 0 <= level <= 2 or not 0 <= current_pc <= 65535 or not 0 <= current_pos <= 3
                    or not 0 <= source_pc <= 65535 or not 0 <= source_pos < LINES_PER_PAGE
                    or not 0 <= older <= 1 or not 0 <= position < LINES_PER_PAGE or not 1 <= prior <= 255
                    or len(values) >= POS_PER_CONTEXT or any(previous == position for previous, _ in values)):
                raise ValueError('invalid LLC position table row')
            values.append((position, prior))
    if not table:
        raise ValueError('empty LLC position table')
    return table


def position_candidates(current, history, table):
    current_page = current.line // LINES_PER_PAGE
    sources = {current_page: (0, current)}
    recently_observed = {current.line}
    for age, source in enumerate(reversed(history), 1):
        sources.setdefault(source.line // LINES_PER_PAGE, (age, source))
        if age <= PAGE_WINDOWS[0]:
            recently_observed.add(source.line)
    proposals = []
    for page, (age, source) in sources.items():
        options = next((table[key] for key in position_keys(current, source, age) if key in table), ())
        for rank, (position, prior) in enumerate(options):
            line = LINES_PER_PAGE * page + position
            if line > 0 and line not in recently_observed:
                proposals.append((rank, -prior, age, line, source))
    for _, negative_prior, age, line, source in nsmallest(CANDIDATE_CAP, proposals):
        yield line, source, age, line - source.line, -negative_prior, 1


def combined_candidates(current, history, recent, table, position_table, position_slots):
    positions = position_candidates(current, history, position_table)
    existing = correlated_candidates(current, recent, table)
    seen = set()
    for line, source, age, offset in existing:
        if len(seen) == CANDIDATE_CAP - position_slots:
            break
        seen.add(line)
        yield line, source, age, offset, 0, 0
    if len(seen) == CANDIDATE_CAP:
        return
    for proposal in positions:
        line = proposal[0]
        if line not in seen:
            seen.add(line)
            yield proposal
            if len(seen) == CANDIDATE_CAP:
                return
    for line, source, age, offset in correlated_candidates(current, recent, table):
        if line not in seen:
            yield line, source, age, offset, 0, 0
            seen.add(line)
            if len(seen) == CANDIDATE_CAP:
                return


def export_candidates(run, v2_data, output, table_path, position_path, window, slots, rotation):
    with np.load(v2_data, allow_pickle=False) as source:
        data = dict(source.items())
    shape = data['valid'].shape
    if shape[1] != CANDIDATE_CAP:
        raise ValueError('baseline candidate cap differs from the export protocol')
    metadata = json.loads(data['metadata'].item())
    table = read_table(table_path)
    position_table = read_positions(position_path) if position_path else None
    for name in ('candidates', 'keys', 'valid', 'targets', 'future_targets', 'lines', 'lead_cycles', 'lead_class',
                 'age', 'source_pos', 'source_pc', 'offset'):
        data[name].fill(0)
    data['proposal_prior_q8'] = np.zeros(shape, np.uint8)
    data['proposal_kind'] = np.zeros(shape, np.uint8)
    if window > np.iinfo(data['age'].dtype).max:
        data['age'] = data['age'].astype(np.uint16)
    cores, summary = read_demands(run)
    if metadata['trace_seed'] != summary['trace_seed'] or sum(map(len, cores.values())) != shape[0]:
        raise ValueError('baseline NPZ and replay differ in trace seed or demand count')
    first = 0
    for cpu, demands in cores.items():
        requests = defaultdict(list)
        for demand in demands:
            requests[demand.line].append(demand)
        ids = {line: [demand.event for demand in visits] for line, visits in requests.items()}
        history = deque(maxlen=window)
        recent = deque(maxlen=PAGE_WINDOWS[0])
        buffer = ProposalBuffer(*rotation) if rotation else None
        for index, demand in enumerate(demands):
            row = first + index
            actual = (cpu, demand.event, demand.pc, demand.line, demand.cycle, demand.instruction, demand.operand)
            stored = tuple(data[key][row] for key in ('cores', 'event_id', 'event_pc', 'event_line', 'event_cycle',
                                                       'event_instr', 'event_operand'))
            if actual != stored:
                raise ValueError('baseline NPZ and replay do not contain the same demands')
            if data['anchors'][row]:
                if buffer is None:
                    proposals = combined_candidates(demand, history, recent, table, position_table, slots)
                else:
                    ranked = buffer.choose(uncapped_offsets(demand, recent, table), demand.cycle)
                    proposals = ((*proposal, 0, 2) for proposal in ranked)
                for slot, (line, source, age, offset, prior, kind) in enumerate(proposals):
                    data['valid'][row, slot] = True
                    data['lines'][row, slot] = line
                    trigger_pc, source_pc = demand.pc & 0xffffffff, source.pc & 0xffffffff
                    position = source.line % LINES_PER_PAGE
                    lag_band = (age - 1).bit_length() if age else 0
                    data['keys'][row, slot] = (offset, lag_band, trigger_pc, demand.line % LINES_PER_PAGE // 16,
                                                position // 16, source_pc)
                    data['candidates'][row, slot] = ((source.pc & 255) / 255, ((source.pc >> 8) & 255) / 255,
                                                    position / 63, offset / 63,
                                                    cycle_feature(demand.cycle - source.cycle), source_pc == trigger_pc)
                    for key, value in (('age', age), ('offset', offset), ('source_pos', position),
                                       ('source_pc', source_pc), ('proposal_prior_q8', prior), ('proposal_kind', kind)):
                        data[key][row, slot] = value
                    later = ids.get(line, ())
                    after = bisect_right(later, demand.event)
                    if after == len(later):
                        continue
                    target = requests[line][after]
                    gap = target.cycle - demand.cycle
                    if gap <= HORIZON:
                        data['lead_cycles'][row, slot] = gap
                        if target.miss:
                            data['future_targets'][row, slot] = target.event
                            data['lead_class'][row, slot] = 1 + bisect_right(LEAD_CUTS, gap)
                            if metadata['lead_cycles'][0] <= gap <= metadata['lead_cycles'][1]:
                                data['targets'][row, slot] = target.event
            history.append(demand)
            recent.append(demand)
        first += len(demands)
    if first != shape[0]:
        raise ValueError('baseline NPZ and replay differ in demand count')
    labels = data['targets'] > 0
    future = data['future_targets'] > 0
    data['labels'], data['future_miss'] = labels, future
    valid = data['valid']
    same_page = valid & (data['age'] == 0)
    classes = data['lead_class']
    metadata.update({
        'version': 4, 'correlation_table': table_path.name,
        'proposal_kind_values': ['v2', 'position', 'rotated_v2'], 'candidate_replay': run.name,
        'candidate_anchors': int(valid.any(1).sum()), 'candidates': int(valid.sum()),
        'positives': int(labels.sum()), 'unique_targets': len(np.unique(data['targets'][labels])),
        'future_misses': int(future.sum()), 'unique_future_misses': len(np.unique(data['future_targets'][future])),
        'unique_future_misses_by_class': [
            len(np.unique(data['future_targets'][classes == klass])) for klass in range(1, len(CLASSES))],
        'class_counts': np.bincount(classes[valid], minlength=len(CLASSES)).tolist(),
        'same_page_candidates': int(same_page.sum()),
    })
    if rotation:
        metadata.update({
            'candidate_source': 'NoPF-fitted offset table with proposal cooldown',
            'proposal_cooldown_cycles': rotation[0],
            'proposal_buffer_entries': rotation[1],
        })
    else:
        metadata.update({
            'candidate_source': 'NoPF-fitted offset and normalized page-position tables',
            'position_table': position_path.name, 'position_history': window, 'position_slot_budget': slots,
        })
    data['metadata'] = json.dumps(metadata, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream, **data)
    output.with_suffix('.tmp').replace(output)
    output.with_suffix('.json').write_text(json.dumps(metadata, sort_keys=True, indent=2) + '\n')
    print(json.dumps(metadata, sort_keys=True), flush=True)


def probe(args):
    if args.spp_run and len(args.spp_run) != len(args.runs):
        raise ValueError('provide one matching --spp-run per NoPF run')
    if (not 1 <= args.lead_min <= HORIZON
            or any(size < 1 for size in args.entries)
            or len(set(args.entries)) != len(args.entries)):
        raise ValueError('lead must be positive and within the evaluation horizon, and entry limits distinct and positive')
    if any(min(pair) < 1 for pair in args.rotation):
        raise ValueError('rotation cycles and entries must both be positive')
    table = read_table(args.table)
    positions = read_positions(args.positions) if args.positions else None
    results = []
    for index, run in enumerate(args.runs):
        spp_run = args.spp_run[index] if args.spp_run else None
        rows = summarize(run, table, spp_run, args.entries, args.lead_min, HORIZON, positions, args.rotation)
        results.extend(rows)
        print(json.dumps(rows), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    diagnostic = commands.add_parser('probe')
    diagnostic.add_argument('runs', type=Path, nargs='+')
    diagnostic.add_argument('--table', type=Path, required=True)
    diagnostic.add_argument('--output', type=Path, required=True)
    diagnostic.add_argument('--positions', type=Path)
    diagnostic.add_argument('--rotation', metavar=('CYCLES', 'ENTRIES'), nargs=2,
                            type=int, action='append', default=[])
    diagnostic.add_argument('--lead-min', type=int, default=280)
    diagnostic.add_argument('--entries', nargs='*', type=int, default=RETRIEVAL_ENTRIES)
    diagnostic.add_argument('--spp-run', type=Path, action='append', default=[])

    fit = commands.add_parser('fit-positions')
    fit.add_argument('output', type=Path)
    fit.add_argument('runs', type=Path, nargs='+')
    fit.add_argument('--contexts', type=int, default=64)
    fit.add_argument('--window', type=int, choices=PAGE_WINDOWS, default=128)
    fit.add_argument('--lead-min', type=int, default=280)

    export = commands.add_parser('export')
    export.add_argument('run', type=Path)
    export.add_argument('baseline_npz', type=Path)
    export.add_argument('output', type=Path)
    export.add_argument('--table', type=Path, required=True)
    generator = export.add_mutually_exclusive_group(required=True)
    generator.add_argument('--positions', type=Path)
    generator.add_argument('--rotation', metavar=('CYCLES', 'ENTRIES'), nargs=2, type=int)
    export.add_argument('--window', type=int, choices=PAGE_WINDOWS, default=128)
    export.add_argument('--position-slots', type=int, default=8)

    args = parser.parse_args()
    if args.command == 'probe':
        probe(args)
    elif args.command == 'fit-positions':
        if args.contexts < 8 or not 1 <= args.lead_min <= HORIZON:
            parser.error('position fit requires at least 8 contexts and a lead in the observation horizon')
        fit_positions(args.runs, args.output, args.contexts, args.window, args.lead_min)
    else:
        if not 0 <= args.position_slots <= CANDIDATE_CAP:
            parser.error('position budget must be between 0 and the fixed candidate cap')
        if args.rotation and min(args.rotation) < 1:
            parser.error('rotation cycles and entries must both be positive')
        export_candidates(args.run, args.baseline_npz, args.output, args.table, args.positions,
                          args.window, args.position_slots, args.rotation)


if __name__ == '__main__':
    main()
