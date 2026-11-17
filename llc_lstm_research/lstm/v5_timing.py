#!/usr/bin/env python3
"""Compare timing-aware LLC scorers using matched NoPF training and development data."""
import argparse
from dataclasses import dataclass
from pathlib import Path
import csv
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train import ranked, training_blocks
from v2_train import OutcomeSplit, calibrated_hits, unique_count
from v3_train import pooled_thresholds


TRAIN_REGIONS = ('10', '30', '50')
DEV_REGIONS = ('70', '130')
BASE_FEATURES = 7
CANDIDATE_FEATURES = 6
TIMING_FEATURES = 4
HEAD_WIDTH = 8
CANDIDATE_CAP = 16
PC_CATEGORY_CAP = 16
TIMING_MODEL_TAG = 'LLC_LSTM_TIMING_V1'
RECURRENT_ARMS = tuple(f'h{hidden}w{window}' for hidden in (8, 16) for window in (8, 16, 32))
NEGATIVE_RETENTION = .14
POSITIVE_WEIGHT = 15.
ARMS = {
    'numeric_pc': (0, ()),
    'candidate_category': (1, ()),
    'both_pc_category': (2, ()),
    'arrival_history': (0, (0, 1, 2)),
    'ingress_mshr': (0, (3,)),
    'arrival_and_mshr': (0, (0, 1, 2, 3)),
    'combined': (2, (0, 1, 2, 3)),
}


@dataclass
class Data:
    split: OutcomeSplit
    trigger_pc: np.ndarray
    candidate_pc: np.ndarray
    extra: np.ndarray
    occupancy: np.ndarray
    span: np.ndarray
    trigger_index: np.ndarray
    candidate_index: np.ndarray


def previous_window_gap(cycles, width):
    history = np.arange(len(cycles), dtype=np.int64)
    earliest = np.maximum(history - width, 0)
    return (cycles.astype(np.int64) - cycles[earliest]) / np.maximum(history - earliest, 1)


def raw_features(path, split, split_cycles):
    event_id = split.event_id
    by_id = np.full(int(event_id.max()) + 1, -1, np.int64)
    by_id[event_id] = np.arange(len(event_id))
    mshr = np.full(len(event_id), -1, np.int16)
    with path.open() as source:
        for row in csv.DictReader(source):
            i = by_id[int(row['event_id'])]
            if (i < 0 or row['prefetch_use'] != 'none' or split.cores[i] != int(row['cpu'])
                    or split_cycles[i] != int(row['ingress_cycle'])):
                raise ValueError(f'raw replay does not match exported NoPF demand {row["event_id"]}')
            mshr[i] = int(row['mshr_occupancy'])
    if (mshr < 0).any():
        raise ValueError('missing occupancy')
    event_gap = np.zeros((len(event_id), 3), np.float32)
    for first, last in split.streams:
        for feature, width in enumerate((8, 32)):
            event_gap[first:last, feature] = previous_window_gap(split_cycles[first:last], width)
    order = np.lexsort((event_id, split_cycles))
    event_gap[order, 2] = previous_window_gap(split_cycles[order], 8)
    span = event_gap[:, 0].copy()
    features = np.column_stack((np.clip(np.log2(1 + event_gap) / 9, 0, 1), np.minimum(mshr / 32, 1)))
    return features.astype(np.float32), mshr, span


def service_cdf(paths):
    cycles, mshr, fill = [], [], []
    for path in paths:
        with path.open() as source:
            for row in csv.DictReader(source):
                observed = int(row['fill_cycle']) - int(row['ingress_cycle']) + 16
                if observed < 16:
                    raise ValueError('training demand service has an invalid clock order')
                cycles.append(observed)
                mshr.append(int(row['mshr_occupancy']))
                fill.append(int(row['fill_cycle']) - int(row['dispatch_cycle']))
    return np.sort(np.asarray(cycles)), np.asarray(cycles), np.asarray(mshr), np.asarray(fill)


def read_split(path, demands):
    split = OutcomeSplit(path)
    if split.width != CANDIDATE_CAP or split.metadata['version'] != 2:
        raise ValueError('timing comparisons require the fixed v2 candidate pool')
    with np.load(path, allow_pickle=False) as source:
        trigger_pc, candidate_pc, cycles = source['event_pc'], source['source_pc'], source['event_cycle']
    extra, mshr, span = raw_features(demands, split, cycles)
    return Data(split, trigger_pc, candidate_pc, extra, mshr, span, None, None)


def map_categories(streams):
    pcs = np.concatenate([data.trigger_pc for data in streams])
    tags, counts = np.unique(pcs, return_counts=True)
    # Training-only vocabulary. Unseen and rare categories share entry zero.
    sorted_tags = tags[np.lexsort((tags, -counts))[:PC_CATEGORY_CAP]]
    return np.sort(sorted_tags)


def index_categories(data, tags):
    def encode(values):
        tentative = np.searchsorted(tags, values)
        return np.where(tags[np.minimum(tentative, len(tags) - 1)] == values, tentative + 1, 0).astype(np.int32)
    data.trigger_index = encode(data.trigger_pc)
    data.candidate_index = encode(data.candidate_pc)


def candidate_matrix(data, rows, cols):
    event, candidate = data.split.events.numpy()[rows], data.split.candidates.numpy()[rows, cols]
    numeric = np.concatenate((event, candidate, data.extra[rows]), axis=1)
    tags = np.column_stack((data.trigger_index[rows], data.candidate_index[rows, cols]))
    return torch.from_numpy(numeric), torch.from_numpy(tags)


def training_matrix(train, calibration):
    rng = np.random.default_rng(605)
    vectors, tags, labels, weights = [], [], [], []
    for data in train:
        valid = data.split.valid.numpy()
        chance = np.searchsorted(calibration, data.split.future_lead, side='right') / len(calibration)
        chance[data.split.future_targets == 0] = 0
        retention = NEGATIVE_RETENTION
        keep = valid & ((chance > 0) | (rng.random(valid.shape, dtype=np.float32) < retention))
        row, col = np.nonzero(keep)
        number, pc = candidate_matrix(data, row, col)
        target = chance[row, col].astype(np.float32)
        vectors.append(number)
        tags.append(pc)
        labels.append(torch.from_numpy(target))
        weights.append(torch.from_numpy(np.where(target > 0, 1., 1 / retention).astype(np.float32)))
    return tuple(torch.cat(items) for items in (vectors, tags, labels, weights))


class Scorer(nn.Module):
    def __init__(self, categories, mode, use_extra):
        super().__init__()
        self.hidden = nn.Linear(17, 8)
        self.score = nn.Linear(8, 1)
        self.pc_embedding = nn.Embedding(categories, 2)
        with torch.no_grad():
            self.pc_embedding.weight.mul_(.15)
        self.mode = mode
        self.unused_features = [13 + i for i in range(4) if i not in use_extra]

    def forward(self, numeric, tags):
        values = numeric.clone()
        values[:, self.unused_features] = 0
        if self.mode:
            values[:, 7:9] = self.pc_embedding(tags[:, 1])
        if self.mode == 2:
            values[:, :2] = self.pc_embedding(tags[:, 0])
        return self.score(torch.relu(self.hidden(values)))[:, 0]


def train_arm(cfg, categories, seed, training):
    torch.manual_seed(seed)
    model = Scorer(categories, *cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005)
    numeric, tags, labels, weights = training
    for epoch in range(6):
        for selected in torch.randperm(numeric.shape[0]).split(4096):
            optimizer.zero_grad(set_to_none=True)
            scores = model(numeric[selected], tags[selected])
            loss = F.binary_cross_entropy_with_logits(scores, labels[selected], weight=weights[selected],
                                                     pos_weight=scores.new_tensor(POSITIVE_WEIGHT))
            loss.backward()
            optimizer.step()
    return model.eval()


@torch.no_grad()
def measure(model, dataset, calibration, arm, region, seed):
    split = dataset.split
    eligible = split.valid.numpy()
    rows, columns = np.nonzero(eligible)
    scores = np.zeros(eligible.shape, dtype=np.float32)
    for start in range(0, len(rows), 65536):
        s = slice(start, start + 65536)
        numeric, tags = candidate_matrix(dataset, rows[s], columns[s])
        scores[rows[s], columns[s]] = model(numeric, tags).numpy()
    first, targets, _, _, _ = ranked(split, scores, True)
    ids = split.future_targets[first, targets]
    leads = split.future_lead[first, targets]
    for budget in (2000, 4000):
        yield {'arm': arm, 'region': region, 'seed': seed, 'budget': budget,
               'expected_timely_unique': calibrated_hits(ids[:budget], leads[:budget], calibration),
               'future_unique': unique_count(ids[:budget]),
               'future_positive': int(np.sum(ids[:budget] > 0)),
               'positive_at_least_280c': int(np.sum((ids[:budget] > 0) & (leads[:budget] >= 280))),
               'selected_ingress_mshr_mean': float(dataset.occupancy[first[:budget]].mean())}


def associations(data, p90, region):
    target = data.split.future_targets > 0
    valid = data.split.valid.numpy()
    lead = data.split.future_lead
    occupancy = np.digitize(data.occupancy, [4, 8, 16])
    arrivals = np.digitize(data.span, [16, 32, 64])
    for feature, groups in [('mshr', occupancy), ('cycles_per_prev8', arrivals)]:
        for group in range(4):
            rows = groups == group
            positive = target[rows]
            count = int(positive.sum())
            yield {'region': region, 'feature': feature, 'group': group, 'events': int(rows.sum()),
                   'valid_candidates': int(valid[rows].sum()), 'future_misses': count,
                   'lead_below_train_p90': float((lead[rows][positive] < p90).mean()) if count else None,
                   'lead_below_150': float((lead[rows][positive] < 150).mean()) if count else None}


def save(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


class TimingScorer(nn.Module):
    def __init__(self, recurrent, hidden, categories):
        super().__init__()
        self.recurrent, self.hidden = recurrent, hidden
        self.cell = nn.LSTM(BASE_FEATURES, hidden, batch_first=True)
        self.event_head = nn.Linear(BASE_FEATURES + TIMING_FEATURES + hidden, HEAD_WIDTH)
        self.candidate_head = nn.Linear(CANDIDATE_FEATURES, HEAD_WIDTH, bias=False)
        self.output = nn.Linear(HEAD_WIDTH, 1)
        self.embedding = nn.Embedding(categories, 2)
        with torch.no_grad():
            self.embedding.weight.mul_(.15)

    def replace_pc(self, features, pc):
        return torch.cat((self.embedding(pc), features[..., 2:]), dim=-1)

    def score(self, events, extra, candidates, source, hidden, selection=None):
        context = self.event_head(torch.cat((events, extra, hidden), dim=-1))
        if selection is not None:
            batch, time, slot = selection.nonzero(as_tuple=True)
            context = context[batch, time]
            candidates = candidates[batch, time, slot]
            source = source[batch, time, slot]
        else:
            context = context.unsqueeze(-2)
        projected = self.candidate_head(self.replace_pc(candidates, source))
        return self.output(torch.relu(context + projected)).squeeze(-1)

    def forward(self, events, extra, candidates, pc, source, state=None, selection=None, use_hidden=True):
        events = self.replace_pc(events, pc)
        if self.recurrent:
            hidden, state = self.cell(events, state)
        if not self.recurrent or not use_hidden:
            hidden = events.new_zeros(events.shape[:-1] + (self.hidden,))
        return self.score(events, extra, candidates, source, hidden, selection), state


def export_portable(model, category_tags, path):
    if not 0 < len(category_tags) <= PC_CATEGORY_CAP or not np.all((0 <= category_tags) & (category_tags < 2**32)):
        raise ValueError('the portable scorer needs 1 to 16 unsigned 32-bit PC tags')
    cell = model.cell
    tensors = (
        cell.weight_ih_l0, cell.weight_hh_l0, cell.bias_ih_l0 + cell.bias_hh_l0,
        model.event_head.weight, model.event_head.bias, model.candidate_head.weight,
        model.output.weight, model.output.bias,
    )
    fmt = lambda value: format(float(value), '.9g')
    with path.with_suffix('.tmp').open('w') as stream:
        kind = 'lstm' if model.recurrent else 'event'
        print(TIMING_MODEL_TAG, kind, BASE_FEATURES, model.hidden, HEAD_WIDTH, CANDIDATE_FEATURES,
              TIMING_FEATURES, CANDIDATE_CAP, 0, len(category_tags), file=stream)
        mapping = model.embedding.weight.detach().cpu().tolist()
        print(*map(fmt, mapping[0]), file=stream)
        for tag, embedding in zip(category_tags, mapping[1:]):
            print(int(tag), *map(fmt, embedding), file=stream)
        for array in tensors:
            flat = array.detach().cpu().reshape(-1)
            if not bool(torch.isfinite(flat).all()):
                raise ValueError('model contains a nonfinite coefficient')
            print(' '.join(map(fmt, flat.tolist())), file=stream)
    path.with_suffix('.tmp').replace(path)


def probe_as_portable(model, tags):
    if model.mode != 2 or model.unused_features:
        raise ValueError('portable timing probe requires categorical PCs and all four timing features')
    timing = TimingScorer(False, 8, len(tags) + 1)
    with torch.no_grad():
        timing.event_head.weight.zero_()
        timing.event_head.weight[:, :BASE_FEATURES] = model.hidden.weight[:, :BASE_FEATURES]
        timing.event_head.weight[:, BASE_FEATURES:BASE_FEATURES + TIMING_FEATURES] = model.hidden.weight[:, 13:]
        timing.event_head.bias.copy_(model.hidden.bias)
        timing.candidate_head.weight.copy_(model.hidden.weight[:, 7:13])
        timing.output.load_state_dict(model.score.state_dict())
        timing.embedding.load_state_dict(model.pc_embedding.state_dict())
    return timing.eval()


def training_targets(train, calibration):
    rng = np.random.default_rng(605)
    output = {}
    for data in train:
        split = data.split
        likelihood = (np.searchsorted(calibration, split.future_lead, side='right')
                      / len(calibration)).astype(np.float32)
        likelihood[split.future_targets == 0] = 0
        selected = split.valid.numpy() & ((likelihood > 0) |
                                         (rng.random(split.valid.shape, dtype=np.float32) < NEGATIVE_RETENTION))
        output[split.name] = torch.from_numpy(likelihood), torch.from_numpy(selected)
    return output


def tensor_views(data):
    return (data.split.events, torch.from_numpy(data.extra), data.split.candidates,
            torch.from_numpy(data.trigger_index), torch.from_numpy(data.candidate_index))


def batch_rollouts(blocks, views, outcomes):
    size, length = len(blocks), max(end - first for _, first, _, end in blocks)
    base = next(iter(views.values()))
    batch = [field.new_zeros((size, length, *field.shape[1:])) for field in base]
    labels = torch.zeros(size, length, CANDIDATE_CAP, dtype=torch.float32)
    selected = torch.zeros_like(labels, dtype=torch.bool)
    for row, (split, first, train_first, end) in enumerate(blocks):
        count = end - first
        for target, source in zip(batch, views[split.name]):
            target[row, :count] = source[first:end]
        y, keep = outcomes[split.name]
        labels[row, :count] = y[first:end]
        selected[row, train_first - first:count] = keep[train_first:end]
    return batch, labels, selected


def train_rollouts(model, blocks, views, outcomes, unroll, optimizer, args, generator):
    model.train()
    epoch_loss, weight = 0., 0
    permutation = generator.permutation(len(blocks))
    for start in range(0, len(blocks), args.batch_size):
        sample = [blocks[k] for k in permutation[start:start + args.batch_size]]
        fields, labels, selected = batch_rollouts(sample, views, outcomes)
        slices = selected.shape[1]
        count = int(selected.sum())
        if not count:
            continue
        optimizer.zero_grad(set_to_none=True)
        state = None
        step = unroll if model.recurrent else slices
        for first in range(0, slices, step):
            end = min(first + step, slices)
            inputs = [field[:, first:end] for field in fields]
            mask = selected[:, first:end]
            if not bool(mask.any()):
                if model.recurrent:
                    with torch.no_grad():
                        _, state = model.cell(model.replace_pc(inputs[0], inputs[3]), state)
                continue
            targets = labels[:, first:end][mask]
            loss_weight = torch.where(targets > 0, 1., 1 / NEGATIVE_RETENTION)
            scores, state = model(*inputs, state=state, selection=mask)
            loss = F.binary_cross_entropy_with_logits(scores, targets, weight=loss_weight,
                                                     pos_weight=scores.new_tensor(POSITIVE_WEIGHT), reduction='sum')
            (loss / count).backward()
            epoch_loss += float(loss.detach())
            if state is not None:
                state = tuple(tensor.detach() for tensor in state)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optimizer.step()
        weight += count
    return epoch_loss / weight


@torch.no_grad()
def rollout_scores(model, data, use_hidden=True):
    model.eval()
    split = data.split
    fields = tensor_views(data)
    scores = np.empty(split.valid.shape, dtype=np.float32)
    for first, last in split.streams:
        state = None
        for start in range(first, last, 4096):
            end = min(start + 4096, last)
            window = [field[start:end].unsqueeze(0) for field in fields]
            logits, state = model(*window, state=state, use_hidden=use_hidden)
            scores[start:end] = logits[0].numpy()
    return scores


@torch.no_grad()
def window_scores(model, data, width, seed=None):
    # Build a fresh hidden/cell state per window. Keep the current event last
    # and its candidate set and timing features fixed when testing order.
    events, timing, candidates, pc, source = tensor_views(data)
    split = data.split
    scores = np.zeros(split.valid.shape, dtype=np.float32)
    prefixes = np.arange(1 - width, 0)
    rng = np.random.default_rng(seed)
    for first, last in split.streams:
        if split.valid[first:first + width - 1].any():
            raise ValueError('the split contains eligible triggers before the fixed window starts')
        for start in range(first + width - 1, last, 1024):
            end = min(start + 1024, last)
            rows = np.arange(start, end)
            offsets = prefixes
            if seed is not None:
                offsets = prefixes[np.argsort(rng.random((len(rows), width - 1)), axis=1)]
            history = rows[:, None] + offsets
            indices = np.column_stack((history, rows))
            inputs = model.replace_pc(events[indices], pc[indices])
            hidden, _ = model.cell(inputs)
            values = model.score(inputs[:, -1], timing[rows], candidates[rows], source[rows], hidden[:, -1])
            scores[start:end] = values.numpy()
    return scores


def analyze_checkpoint(path, dev, tags, calibration, budgets, output):
    meta = torch.load(path, map_location='cpu', weights_only=True)
    if not meta['recurrent'] or meta['tags'] != tags.tolist():
        raise ValueError('analysis needs a recurrent checkpoint with the matching PC vocabulary')
    model = TimingScorer(True, meta['hidden'], len(tags) + 1)
    model.load_state_dict(meta['state_dict'])
    model.eval()
    arms = [(name, width, seed) for width in (8, 16, 32)
            for name, seed in [('ordered', None), ('shuffled', 605), ('shuffled', 606)]]
    arms = [('carry', 0, None), ('zero_state', 0, None), *arms]
    records = []
    for data in dev:
        split = data.split
        for name, width, seed in arms:
            scores = (window_scores(model, data, width, seed) if width
                      else rollout_scores(model, data, use_hidden=name == 'carry'))
            rows, slots, _, _, _ = ranked(split, scores, True)
            target, lead = split.future_targets[rows, slots], split.future_lead[rows, slots]
            if len(rows) < max(budgets):
                raise ValueError('not enough eligible triggers for the fixed analysis budget')
            for budget in budgets:
                records.append(dict(checkpoint=path.name, split=split.name, order=name, window=width,
                                    seed=seed, budget=budget, future_unique=unique_count(target[:budget]),
                                    expected_timely_unique=calibrated_hits(target[:budget], lead[:budget], calibration)))
    save(output, records)


def validation_report(model, dev, calibration, budgets):
    metrics, cutoffs = [], []
    for data in dev:
        split = data.split
        rows, slots, logits, _, _ = ranked(split, rollout_scores(model, data), True)
        if len(rows) < max(budgets):
            raise ValueError('not enough eligible triggers for the fixed validation budgets')
        targets, lead = split.future_targets[rows, slots], split.future_lead[rows, slots]
        cutoffs.append(logits)
        metrics.append([dict(split=split.name, budget=budget, future_unique=unique_count(targets[:budget]),
                             expected_timely_unique=calibrated_hits(targets[:budget], lead[:budget], calibration))
                        for budget in budgets])
    return [row for group in metrics for row in group], pooled_thresholds(cutoffs, budgets)


def timing_cost(model, tags):
    hidden = model.hidden if model.recurrent else 0
    cell_macs = 4 * hidden * (BASE_FEATURES + hidden)
    matrix_macs = HEAD_WIDTH * (BASE_FEATURES + TIMING_FEATURES + hidden)
    candidate_macs = HEAD_WIDTH * (CANDIDATE_FEATURES + 1)
    head_bytes = 4 * (matrix_macs + HEAD_WIDTH + candidate_macs + 1)
    cell_bytes = 4 * (cell_macs + 4 * hidden)
    pc_bytes = 4 * (len(tags) + 2 * (len(tags) + 1))
    return dict(max_linear_macs=cell_macs + matrix_macs + CANDIDATE_CAP * candidate_macs,
                logical_weight_and_tag_bytes=head_bytes + cell_bytes + pc_bytes,
                recurrent_state_bytes_per_core=8 * hidden,
                raw_arrival_clock_bytes_two_cores_uint32=4 * (2 * 32 + 8))


def checkpoint(model, tags, output, details):
    details = dict(details, cost=timing_cost(model, tags))
    export_portable(model, tags, output.with_suffix('.txt'))
    torch.save(dict(details, tags=tags.tolist(), hidden=model.hidden, recurrent=model.recurrent,
                    state_dict=model.state_dict()), output.with_suffix('.pt'))
    output.with_suffix('.json').write_text(json.dumps(details, indent=2) + '\n')


def warm_start(model, tags, path):
    base = torch.load(path, map_location='cpu', weights_only=True)
    if base['recurrent'] or base['tags'] != tags.tolist():
        raise ValueError('warm start must be a portable event checkpoint with the same PC vocabulary')
    weights = base['state_dict']
    context = BASE_FEATURES + TIMING_FEATURES
    with torch.no_grad():
        model.event_head.weight.zero_()
        model.event_head.weight[:, :context].copy_(weights['event_head.weight'][:, :context])
        model.event_head.bias.copy_(weights['event_head.bias'])
        for name, layer in (('candidate_head', model.candidate_head), ('output', model.output),
                            ('embedding', model.embedding)):
            layer.load_state_dict({key: weights[f'{name}.{key}'] for key in layer.state_dict()})


def residual_parameters(model):
    # A frozen event score and zero-initialized hidden projection yield the
    # same initial predictions. The learned difference must come from state.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.cell.parameters():
        parameter.requires_grad_(True)
    weight = model.event_head.weight
    weight.requires_grad_(True)
    frozen_columns = BASE_FEATURES + TIMING_FEATURES
    gradient_mask = torch.ones_like(weight)
    gradient_mask[:, :frozen_columns] = 0
    weight.register_hook(lambda gradient: gradient * gradient_mask)
    return [*model.cell.parameters(), weight]


def fit_rollout_arm(arm, train, dev, calibration, tags, args):
    recurrent = arm != 'event'
    if recurrent:
        hidden, width = map(int, arm[1:].split('w'))
    else:
        hidden, width = 8, 0
    torch.manual_seed(args.seed)
    generator = np.random.default_rng(args.seed)
    model = TimingScorer(recurrent, hidden, len(tags) + 1)
    if args.initialize_event:
        warm_start(model, tags, args.initialize_event)
    if args.residual_only:
        parameters = residual_parameters(model)
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    blocks = training_blocks([data.split for data in train], args.block, args.burn)
    views = {data.split.name: tensor_views(data) for data in train}
    outcomes = training_targets(train, calibration)
    best, best_epoch, history = -np.inf, 0, []
    start = 0 if args.initialize_event else 1
    for epoch in range(start, args.epochs + 1):
        loss = train_rollouts(model, blocks, views, outcomes, width, optimizer, args, generator) if epoch else None
        records, thresholds = validation_report(model, dev, calibration, args.budgets)
        utility = sum(row['expected_timely_unique'] for row in records)
        history.extend(dict(arm=arm, epoch=epoch, loss=loss, **row) for row in records)
        print(arm, epoch, round(loss, 5) if loss is not None else 'initial',
              [round(row['expected_timely_unique'], 2) for row in records], flush=True)
        if utility > best:
            best, best_epoch = utility, epoch
            details = dict(arm=arm, epoch=epoch, seed=args.seed, unroll=width, training_burn=args.burn,
                           validation_metrics=records, thresholds=thresholds,
                           initialize_event=str(args.initialize_event) if args.initialize_event else None,
                           residual_only=args.residual_only)
            checkpoint(model, tags, args.output / arm, details)
        if epoch >= args.min_epochs and epoch - best_epoch >= args.patience:
            break
    save(args.output / (arm + '_epochs.csv'), history)


def write_associations(output, train, dev, calibration, service, mshr, fill, tags):
    bins = np.digitize(mshr, [4, 8, 16])
    metadata = {'training_service_samples': len(calibration), 'pc_categories': tags.tolist(),
                'training_service_p50_p90_p95': np.quantile(calibration, [.50, .90, .95]).tolist(),
                'service_by_mshr': [
                    dict(group=index, n=int((bins == index).sum()),
                         service_p50_p90=np.quantile(service[bins == index], [.5, .9]).tolist(),
                         dispatch_to_fill_p50=np.quantile(fill[bins == index], .5))
                    for index in range(4) if (bins == index).any()
                ]}
    (output / 'training_service.json').write_text(json.dumps(metadata, indent=2) + '\n')
    save(output / 'associations.csv', [row for data in (*train, *dev)
                                      for row in associations(data, np.quantile(calibration, .90), data.split.name)])


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--train', nargs=3, type=Path, required=True)
    cli.add_argument('--train-demand', nargs=3, type=Path, required=True)
    cli.add_argument('--train-service', nargs=3, type=Path, required=True)
    cli.add_argument('--validation', nargs=2, type=Path, required=True)
    cli.add_argument('--validation-demand', nargs=2, type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--arms', nargs='+', choices=('event', *RECURRENT_ARMS), default=['event', 'h8w8'])
    cli.add_argument('--probe', action='store_true', help='reproduce the feed-forward timing ablations')
    cli.add_argument('--portable-probe', action='store_true', help='export the combined feed-forward probe only')
    cli.add_argument('--initialize-event', type=Path, help='start with weights of a frozen portable event scorer')
    cli.add_argument('--residual-only', action='store_true', help='freeze the event score and update only recurrent state')
    cli.add_argument('--analyze-checkpoint', type=Path, help='compare carried, zeroed and finite shuffled state')
    cli.add_argument('--budgets', type=int, nargs='+', default=[2000, 4000])
    cli.add_argument('--epochs', type=int, default=8)
    cli.add_argument('--min-epochs', type=int, default=4)
    cli.add_argument('--patience', type=int, default=3)
    cli.add_argument('--burn', type=int, default=32)
    cli.add_argument('--block', type=int, default=256)
    cli.add_argument('--batch-size', type=int, default=16)
    cli.add_argument('--seed', type=int, default=606)
    cli.add_argument('--lr', type=float, default=.005)
    cli.add_argument('--threads', type=int, default=2)
    return cli


def main():
    cli = parser()
    args = cli.parse_args()
    if args.residual_only and (not args.initialize_event or 'event' in args.arms):
        cli.error('residual training requires an event checkpoint and recurrent arms')
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    train = [read_split(path, demands) for path, demands in zip(args.train, args.train_demand)]
    dev = [read_split(path, demands) for path, demands in zip(args.validation, args.validation_demand)]
    sources = [data.split.metadata['source'] for data in (*train, *dev)]
    if sources[:3] != [f'rob_{region}m' for region in TRAIN_REGIONS] or sources[3:] != [
            'dev_70m', '605_nopf_130m']:
        raise ValueError('use the local 10/30/50 training and 70/130 development regions in that order')
    tags = map_categories(train)
    for data in (*train, *dev):
        index_categories(data, tags)
    calibration, service, mshr, fill = service_cdf(args.train_service)
    write_associations(args.output, train, dev, calibration, service, mshr, fill, tags)
    if args.analyze_checkpoint:
        analyze_checkpoint(args.analyze_checkpoint, dev, tags, calibration, args.budgets,
                           args.output / 'state_analysis.csv')
    elif args.probe or args.portable_probe:
        training = training_matrix(train, calibration)
        metrics = []
        for seed in (605, 606):
            for name, cfg in ARMS.items():
                if args.portable_probe and name != 'combined':
                    continue
                model = train_arm(cfg, len(tags) + 1, seed, training)
                rows = [record for data in dev for record in measure(model, data, calibration, name, data.split.name, seed)]
                metrics.extend(rows)
                print(name, seed, [round(row['expected_timely_unique'], 2) for row in rows], flush=True)
                if name == 'combined':
                    portable = probe_as_portable(model, tags)
                    records, thresholds = validation_report(portable, dev, calibration, args.budgets)
                    checkpoint(portable, tags, args.output / f'event_probe_{seed}',
                               dict(arm='event_probe', epoch=6, seed=seed, unroll=0,
                                    validation_metrics=records, thresholds=thresholds))
        save(args.output / 'predictions.csv', metrics)
    else:
        for arm in args.arms:
            fit_rollout_arm(arm, train, dev, calibration, tags, args)


if __name__ == '__main__':
    main()
