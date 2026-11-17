#!/usr/bin/env python3
"""Fit an LLC LSTM that scores categorical line positions in observed pages."""

import argparse
from bisect import bisect_right
from collections import deque
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dataset import FIRST_ANCHOR, LINES_PER_PAGE, encode_events, read_demands
from train import HIDDEN, INPUTS, q10, training_blocks
from v2_data import HORIZON, cycle_feature
from v2_train import calibrated_hits, load_calibration
from v3_train import pooled_thresholds


SOURCE_FEATURES = 6
SOURCE_WINDOW = 32
SLOTS = 2
BUDGETS = (2000, 4000)
PC_ENTRIES, PC_WIDTH = 64, 4


def observed_sources(current, history, count):
    sources = [current]
    pages = {current.line // LINES_PER_PAGE}
    for earlier in reversed(history):
        if len(sources) == count:
            break
        page = earlier.line // LINES_PER_PAGE
        if page not in pages:
            pages.add(page)
            sources.append(earlier)
    return sources


def export(run, output, count):
    cores, _ = read_demands(run)
    rows = sum(map(len, cores.values()))
    events = np.zeros((rows, INPUTS), np.float32)
    sources = np.zeros((rows, count, SOURCE_FEATURES), np.float32)
    pages = np.zeros((rows, count), np.uint64)
    valid = np.zeros((rows, count, LINES_PER_PAGE), bool)
    target = np.zeros(valid.shape, np.int32)
    lead = np.zeros(valid.shape, np.uint16)
    event_ids = np.zeros(rows, np.int32)
    event_pcs = np.zeros(rows, np.int64)
    event_lines = np.zeros(rows, np.uint64)
    core_ids = np.zeros(rows, np.int16)
    instructions = np.zeros(rows, np.int64)
    operands = np.zeros(rows, np.uint8)
    streams = []
    base = 0
    for core, demands in cores.items():
        streams.append((base, base + len(demands)))
        recent = deque(maxlen=SOURCE_WINDOW)
        cycles = [demand.cycle for demand in demands]
        for index, current in enumerate(demands):
            row = base + index
            event_ids[row] = current.event
            event_pcs[row] = current.pc
            event_lines[row] = current.line
            core_ids[row] = core
            instructions[row] = current.instruction
            operands[row] = current.operand
            if index:
                events[row, 6] = cycle_feature(current.cycle - demands[index - 1].cycle)
            if index >= FIRST_ANCHOR and current.cycle + HORIZON <= cycles[-1]:
                forbidden = {event.line for event in recent}
                forbidden.add(current.line)
                page_slots = {}
                for slot, source in enumerate(observed_sources(current, recent, count)):
                    page, position = divmod(source.line, LINES_PER_PAGE)
                    pages[row, slot] = page
                    page_slots[page] = slot
                    sources[row, slot] = (position / 63, (source.pc & 255) / 255,
                                          ((source.pc >> 8) & 255) / 255, slot == 0,
                                          source.pc == current.pc,
                                          cycle_feature(current.cycle - source.cycle))
                    valid[row, slot] = True
                for line in forbidden:
                    page, position = divmod(line, LINES_PER_PAGE)
                    if page in page_slots:
                        valid[row, page_slots[page], position] = False
                last = bisect_right(cycles, current.cycle + HORIZON, index + 1)
                seen = set()
                for future in demands[index + 1:last]:
                    page, position = divmod(future.line, LINES_PER_PAGE)
                    slot = page_slots.get(page)
                    if slot is not None and valid[row, slot, position] and future.line not in seen:
                        seen.add(future.line)
                        if future.miss:
                            target[row, slot, position] = future.event
                            lead[row, slot, position] = future.cycle - current.cycle
            recent.append(current)
        base += len(demands)
    gaps = events[:, 6].copy()
    encode_events(events, event_pcs & 0xffffffff, event_lines, core_ids)
    events[:, 6] = gaps
    np.savez_compressed(output, events=events, sources=sources, pages=pages, valid=valid, lead=lead,
                        target=target, event_ids=event_ids, streams=np.asarray(streams),
                        cores=core_ids, instructions=instructions, operands=operands)
    print(run.name, int(valid.sum()), int(np.unique(target[target > 0]).size), flush=True)


class PositionModel(nn.Module):
    def __init__(self, recurrent, relative, discrete_pc):
        super().__init__()
        self.recurrent = recurrent
        self.pc = nn.Embedding(PC_ENTRIES, PC_WIDTH) if discrete_pc else None
        extra = PC_WIDTH - 2 if discrete_pc else 0
        self.cell = nn.LSTM(INPUTS + extra, HIDDEN, batch_first=True) if recurrent else None
        self.query = nn.Linear(INPUTS + extra + (HIDDEN if recurrent else 0), HIDDEN)
        self.source = nn.Linear(SOURCE_FEATURES + extra, HIDDEN, bias=False)
        self.relative = relative
        self.position = nn.Linear(HIDDEN, 2 * LINES_PER_PAGE - 1 if relative else LINES_PER_PAGE)
        self.register_buffer('positions', torch.arange(LINES_PER_PAGE))

    def forward(self, events, sources, state=None):
        origins = sources[..., 0]
        if self.pc is not None:
            events = self.encode_pc(events, 0)
            sources = self.encode_pc(sources, 1)
        if self.cell is not None:
            hidden, state = self.cell(events, state)
            events = torch.cat((events, hidden), dim=-1)
        features = torch.relu(self.query(events).unsqueeze(-2) + self.source(sources))
        logits = self.position(features)
        if self.relative:
            origin = (origins * (LINES_PER_PAGE - 1)).round().long()
            choose = self.positions + LINES_PER_PAGE - 1 - origin.unsqueeze(-1)
            logits = logits.gather(-1, choose)
        return logits, state

    def encode_pc(self, values, start):
        bytes_ = (values[..., start:start + 2] * 255).round().long()
        pc = bytes_[..., 0] | (bytes_[..., 1] << 8)
        index = ((pc >> 2) ^ (pc >> 9)) & (PC_ENTRIES - 1)
        return torch.cat((values[..., :start], self.pc(index), values[..., start + 2:]), dim=-1)


class Split:
    def __init__(self, path, calibration, count, latency):
        self.name = path.stem
        with np.load(path, allow_pickle=False) as data:
            self.events = torch.from_numpy(data['events'])
            self.sources = torch.from_numpy(data['sources'][:, :count].copy())
            self.valid = torch.from_numpy(data['valid'][:, :count].copy())
            self.target = data['target'][:, :count].copy()
            self.lead = np.maximum(data['lead'][:, :count].astype(np.int32) - latency, 0)
            self.event_ids = data['event_ids']
            self.streams = data['streams'].tolist()
        scores = np.searchsorted(calibration, self.lead, side='right') / calibration.size
        self.labels = torch.from_numpy(np.where(self.target > 0, scores, 0).astype(np.float32))

    def rob_utility(self, data_path, log_path):
        with log_path.open(newline='') as stream:
            waits = {(int(row['cpu']), int(row['instr_id']), int(row['operand'])):
                     int(row['head_wait_cycles']) for row in csv.DictReader(stream)}
        with np.load(data_path, allow_pickle=False) as data:
            identities = zip(data['cores'].tolist(), data['instructions'].tolist(), data['operands'].tolist())
        indexed = np.zeros(self.event_ids.max() + 1, np.int32)
        indexed[self.event_ids] = np.fromiter((waits.get(key, 0) for key in identities), np.int32)
        return np.minimum(indexed[self.target], self.lead)


def pack(blocks):
    size = max(end - begin for _, begin, _, end in blocks)
    example = blocks[0][0]
    xs = torch.zeros(len(blocks), size, INPUTS)
    zs = torch.zeros(len(blocks), size, *example.sources.shape[1:])
    ys = torch.zeros(len(blocks), size, *example.labels.shape[1:])
    mask = torch.zeros_like(ys, dtype=torch.bool)
    for row, (split, begin, start, end) in enumerate(blocks):
        length = end - begin
        xs[row, :length] = split.events[begin:end]
        zs[row, :length] = split.sources[begin:end]
        ys[row, :length] = split.labels[begin:end]
        mask[row, start - begin:length] = split.valid[start:end]
    return xs, zs, ys, mask


def portable(model, slots, directory):
    path = directory / 'model.txt'
    tensors = [model.pc.weight] if model.pc is not None else []
    if model.cell is not None:
        tensors.extend((model.cell.weight_ih_l0, model.cell.weight_hh_l0,
                        model.cell.bias_ih_l0 + model.cell.bias_hh_l0))
    tensors.extend((model.query.weight, model.query.bias, model.source.weight,
                    model.position.weight, model.position.bias))
    with path.open('w') as stream:
        kind = 'lstm' if model.recurrent else 'event'
        stream.write(f'LLC_POSITION_V1 {kind} {slots} {int(model.relative)} {int(model.pc is not None)}\n')
        for tensor in tensors:
            stream.write(' '.join(format(value, '.9g') for value in tensor.detach().flatten().tolist()) + '\n')


def start_from_event(model, path, freeze):
    weights = torch.load(path, map_location='cpu', weights_only=True)
    query = weights['query.weight']
    width = query.shape[1]
    if model.cell is None or width != model.query.in_features - HIDDEN:
        raise ValueError('the warm start requires a compatible event-only scorer')
    state = model.state_dict()
    weights['query.weight'] = torch.cat((query, query.new_zeros(HIDDEN, HIDDEN)), dim=-1)
    state.update(weights)
    model.load_state_dict(state)
    if freeze:
        frozen = {name for name in weights if name != 'query.weight'}
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name not in frozen)
        model.query.weight.register_hook(
            lambda gradient: torch.cat((torch.zeros_like(gradient[:, :width]), gradient[:, width:]), dim=1))


@torch.no_grad()
def evaluate(model, split, calibration):
    winners, targets, leads = [], [], []
    for begin, end in split.streams:
        state = None
        for start in range(begin, end, 4096):
            stop = min(start + 4096, end)
            slab = slice(start, stop)
            logits, state = model(split.events[None, slab], split.sources[None, slab], state)
            live = split.valid[slab].flatten(1).numpy()
            ranked = np.where(live, q10(logits[0].flatten(1).numpy()), -32769)
            columns = ranked.argmax(1)
            rows = np.arange(stop - start)
            winners.append(ranked[rows, columns])
            targets.append(split.target[slab].reshape(rows.size, -1)[rows, columns])
            leads.append(split.lead[slab].reshape(rows.size, -1)[rows, columns])
    scores = np.concatenate(winners)
    ties = split.event_ids.astype(np.uint64) * np.uint64(11400714819323198485)
    order = np.lexsort((ties, -scores))
    target, lead = np.concatenate(targets), np.concatenate(leads)
    return [calibrated_hits(target[order[:budget]], lead[order[:budget]], calibration)
            for budget in BUDGETS], scores[order]


def fit(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    calibration = load_calibration(args.calibration)
    calibration_metadata = json.loads(args.calibration.read_text())
    if not args.historical_prefetch_calibration:
        sources = calibration_metadata.get('sources')
        if (not isinstance(sources, dict) or not sources
                or any('nopf' not in Path(path).parts or type(count) is not int or count < 1
                       for path, count in sources.items())
                or sum(sources.values()) != calibration.size
                or calibration_metadata.get('samples') != calibration.size):
            raise ValueError('fit requires NoPF demand-service calibration from lstm/nopf_service.py')
    train = [Split(path, calibration, args.slots, args.latency) for path in args.train]
    dev = [Split(path, calibration, args.slots, args.latency) for path in args.dev]
    stall_scale = None
    if args.rob_stalls:
        gains = [split.rob_utility(path, log)
                 for split, path, log in zip(train, args.train, args.rob_stalls)]
        stall_scale = float(np.percentile(np.concatenate([gain[gain > 0] for gain in gains]), 90))
        for split, gain in zip(train, gains):
            split.labels = torch.from_numpy(np.minimum(gain / stall_scale, 1).astype(np.float32))
    blocks = training_blocks(train, 128, 32)
    valid = sum(int(split.valid.sum()) for split in train)
    positives = sum(float(split.labels.sum()) for split in train)
    weight = min(32., np.sqrt((valid - positives) / positives))
    model = PositionModel(not args.event_only, args.relative, args.discrete_pc)
    if args.start_event:
        start_from_event(model, args.start_event, args.freeze_event)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0 if args.freeze_event else .01)
    args.output.mkdir(parents=True, exist_ok=False)
    best = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        ordering = rng.permutation(len(blocks))
        for first in range(0, len(blocks), 32):
            batch = [blocks[index] for index in ordering[first:first + 32]]
            events, sources, labels, mask = pack(batch)
            reviews = int(mask.sum())
            if not reviews:
                continue
            optimizer.zero_grad(set_to_none=True)
            state = None
            for start in range(0, events.shape[1], args.unroll):
                slab = slice(start, start + args.unroll)
                scores, state = model(events[:, slab], sources[:, slab], state)
                selects = mask[:, slab]
                if bool(selects.any()):
                    loss = F.binary_cross_entropy_with_logits(scores[selects], labels[:, slab][selects],
                                                             pos_weight=scores.new_tensor(weight), reduction='sum')
                    (loss / reviews).backward()
                if state is not None:
                    state = tuple(value.detach() for value in state)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
        model.eval()
        results = [evaluate(model, split, calibration) for split in dev]
        hits = [result[0] for result in results]
        score = sum(map(sum, hits))
        progress = {'epoch': epoch, 'expected_on_time': hits, 'total': score, 'slots': args.slots,
                    'event_only': args.event_only, 'relative': args.relative, 'discrete_pc': args.discrete_pc,
                    'stall_p90': stall_scale, 'train': [split.name for split in train],
                    'dev': [split.name for split in dev], 'horizon': HORIZON,
                    'unroll': args.unroll, 'latency': args.latency, 'learning_rate': args.lr,
                    'calibration_samples': int(calibration.size),
                    'historical_prefetch_calibration': args.historical_prefetch_calibration,
                    'start_event': str(args.start_event) if args.start_event else None, 'freeze_event': args.freeze_event}
        print(json.dumps(progress), flush=True)
        if score > best:
            best = score
            torch.save(model.state_dict(), args.output / 'model.pt')
            portable(model, args.slots, args.output)
            progress['thresholds'] = pooled_thresholds([item[1] for item in results], BUDGETS)
            (args.output / 'best.json').write_text(json.dumps(progress, indent=2) + '\n')


def convert(args):
    info = json.loads((args.experiment / 'best.json').read_text())
    model = PositionModel(not info['event_only'], info.get('relative', False), info.get('discrete_pc', False))
    weights = torch.load(args.experiment / 'model.pt', map_location='cpu', weights_only=True)
    weights.setdefault('positions', model.positions)
    model.load_state_dict(weights)
    model.eval()
    torch.set_num_threads(2)
    calibration = load_calibration(args.calibration)
    datasets = [Split(path, calibration, info['slots'], info['latency']) for path in args.dev]
    results = [evaluate(model, dataset, calibration) for dataset in datasets]
    info['thresholds'] = pooled_thresholds([item[1] for item in results], BUDGETS)
    portable(model, info['slots'], args.experiment)
    (args.experiment / 'best.json').write_text(json.dumps(info, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    derive = commands.add_parser('export')
    derive.add_argument('run', type=Path)
    derive.add_argument('output', type=Path)
    derive.add_argument('--slots', type=int, choices=(1, 2, 4), default=SLOTS)
    train = commands.add_parser('fit')
    train.add_argument('--train', nargs='+', type=Path, required=True)
    train.add_argument('--dev', nargs='+', type=Path, required=True)
    train.add_argument('--calibration', type=Path, required=True)
    train.add_argument('--historical-prefetch-calibration', action='store_true',
                       help='reproduce the archived fit which used an earlier model’s prefetch fills')
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--slots', type=int, choices=(1, 2, 4), default=SLOTS)
    train.add_argument('--event-only', action='store_true')
    train.add_argument('--relative', action='store_true', help='predict signed relative line positions')
    train.add_argument('--discrete-pc', action='store_true', help='embed a small index derived from the observed PC')
    train.add_argument('--rob-stalls', nargs='+', type=Path, help='NoPF ROB logs corresponding to training NPZs')
    train.add_argument('--unroll', type=int, default=16)
    train.add_argument('--latency', type=int, default=16)
    train.add_argument('--epochs', type=int, default=12)
    train.add_argument('--threads', type=int, default=2)
    train.add_argument('--seed', type=int, default=605)
    train.add_argument('--lr', type=float, default=.003)
    train.add_argument('--start-event', type=Path)
    train.add_argument('--freeze-event', action='store_true')
    convert_cmd = commands.add_parser('portable')
    convert_cmd.add_argument('experiment', type=Path)
    convert_cmd.add_argument('--dev', nargs='+', type=Path, required=True)
    convert_cmd.add_argument('--calibration', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'export':
        export(args.run, args.output, args.slots)
    elif args.command == 'portable':
        convert(args)
    else:
        if args.rob_stalls and len(args.rob_stalls) != len(args.train):
            parser.error('provide one ROB log per training split')
        if (args.start_event and args.event_only) or (args.freeze_event and not args.start_event):
            parser.error('a trained event scorer can initialize only a recurrent model')
        fit(args)


if __name__ == '__main__':
    main()
