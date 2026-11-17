#!/usr/bin/env python3
"""Compare NoPF objectives while keeping the LLC scorer and address pool fixed."""

import argparse
from copy import copy
import csv
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from evaluate_state import window_scores, zero_state_scores
from train import (CANDIDATE_FEATURES, HIDDEN, INPUTS, RANK, Scorer, Split, batch,
                   evaluate, export, predict, ranked, read_table, table_scores,
                   training_blocks, train_epoch)


LABELS = ('timed', 'future', 'soft', 'timed_unique', 'future_unique', 'soft_unique')
PROTOCOL = ('event_features', 'candidate_features', 'offset_order', 'age_band_order',
            'page_history', 'first_anchor', 'lead_cycles', 'target_definition',
            'lead_cuts', 'candidate_source')


class OutcomeSplit(Split):
    def __init__(self, path):
        super().__init__(path)
        with np.load(path, allow_pickle=False) as archive:
            self.future_targets = archive['future_targets']
            self.future_lead = archive['lead_cycles']
        valid = self.valid.numpy()
        future = self.future_targets > 0
        timed = self.targets > 0
        if (self.future_targets.shape != self.targets.shape
                or self.future_lead.shape != self.targets.shape
                or not np.issubdtype(self.future_targets.dtype, np.integer)
                or not np.issubdtype(self.future_lead.dtype, np.integer)
                or np.any(self.future_targets < 0) or np.any(future & ~valid)
                or np.any(future & ((self.future_lead < 0) | (self.future_lead >= self.metadata['lead_cuts'][-1])))
                or np.any(timed & (self.targets != self.future_targets))):
            raise ValueError('invalid future outcomes in {}'.format(path))
        low, high = self.metadata['lead_cycles']
        if not np.array_equal(timed, future & (self.future_lead >= low) & (self.future_lead <= high)):
            raise ValueError('timed and future outcomes disagree in {}'.format(path))
        self.future = copy(self)
        self.future.targets = self.future_targets
        self.future.opportunities = int(np.unique(self.future_targets[future]).size)


def load_calibration(path):
    if path is None:
        return None
    with path.open() as source:
        data = json.load(source)
    service = np.asarray(data['service_cycles'], dtype=np.float64)
    if service.ndim != 1 or not service.size or not np.isfinite(service).all() or np.any(service < 0):
        raise ValueError('service_cycles must contain finite nonnegative development measurements')
    return np.sort(service)


def label_view(split, label, calibration):
    view = copy(split)
    targets = split.targets if label.startswith('timed') else split.future_targets
    positives = targets > 0
    view.labels = torch.from_numpy(positives)
    view.sample_weights = None
    if label.startswith('soft'):
        probability = np.zeros(targets.shape, dtype=np.float32)
        probability[positives] = (np.searchsorted(calibration, split.future_lead[positives], side='right')
                                  / calibration.size)
        view.labels = torch.from_numpy(probability)
    if label.endswith('_unique'):
        effective = view.labels.numpy() > 0
        _, ids, counts = np.unique(targets[effective], return_inverse=True, return_counts=True)
        weights = np.ones(targets.shape, dtype=np.float32)
        if ids.size:
            # Rebalance repeated target IDs while holding total positive mass fixed.
            mass = view.labels.numpy()[effective]
            inverse = 1 / counts[ids]
            weights[effective] = inverse * (mass.sum() / np.sum(mass * inverse))
        view.sample_weights = torch.from_numpy(weights)
    return view


def weighted_batch(blocks, device):
    events, candidates, _, valid = batch(blocks, 'cpu')
    labels = torch.zeros(valid.shape, dtype=torch.float32)
    weights = torch.ones_like(labels)
    for row, (split, begin, _, stop) in enumerate(blocks):
        length = stop - begin
        labels[row, :length] = split.labels[begin:stop]
        if split.sample_weights is not None:
            weights[row, :length] = split.sample_weights[begin:stop]
    return tuple(value.to(device) for value in (events, candidates, labels, valid, weights))


def weighted_epoch(model, blocks, args, optimizer, positive_weight, device, rng):
    model.train()
    indices = rng.permutation(len(blocks))
    total_loss, total_count = 0.0, 0
    for offset in range(0, len(indices), args.batch_size):
        selected = [blocks[i] for i in indices[offset:offset + args.batch_size]]
        events, candidates, labels, valid, weights = weighted_batch(selected, device)
        count = int(valid.sum())
        if not count:
            continue
        state = None
        step = args.unroll if model.recurrent else events.shape[1]
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, events.shape[1], step):
            stop = start + step
            mask = valid[:, start:stop]
            if not bool(mask.any()):
                if model.recurrent:
                    with torch.no_grad():
                        _, state = model.cell(events[:, start:stop], state)
                continue
            scores, state = model(events[:, start:stop], candidates[:, start:stop], state)
            loss = F.binary_cross_entropy_with_logits(
                scores[mask], labels[:, start:stop][mask],
                pos_weight=positive_weight * weights[:, start:stop][mask], reduction='sum',
            )
            (loss / count).backward()
            total_loss += float(loss.detach())
            if state is not None:
                state = tuple(value.detach() for value in state)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        total_count += count
    return total_loss / total_count


def write_checkpoint(output, stem, model, state, epoch, architecture, label, args, width):
    model.load_state_dict(state)
    export(model, width, output / (stem + '.txt'))
    torch.save({'state_dict': state, 'recurrent': model.recurrent, 'architecture': architecture,
                'unroll': args.unroll if model.recurrent else 0, 'epoch': epoch,
                'candidate_cap': width, 'label': label}, output / (stem + '.pt'))


def calibrated_hits(targets, lead, calibration):
    valid = targets > 0
    unique, inverse = np.unique(targets[valid], return_inverse=True)
    chance = np.searchsorted(calibration, lead[valid], side='right') / calibration.size
    best = np.zeros(unique.size, dtype=np.float64)
    np.maximum.at(best, inverse, chance)
    return float(best.sum())


def selection_key(split, scores, args, calibration):
    if args.selection == 'calibrated':
        rows, columns, _, _, _ = ranked(split, scores, True)
        if rows.size < max(args.budgets):
            raise ValueError('{} has only {} eligible events'.format(split.name, rows.size))
        future, lead = split.future_targets[rows, columns], split.future_lead[rows, columns]
        hits = [calibrated_hits(future[:budget], lead[:budget], calibration) for budget in args.budgets]
    else:
        view = split.future if args.selection == 'future' else split
        hits = [row['unique_targets'] for row in evaluate(view, scores, args.budgets)['budgets']]
    return sum(hits), hits[0]


def unique_count(targets):
    return int(np.unique(targets[targets > 0]).size)


def summarize(split, scores, budgets, fixed_thresholds, calibration):
    rows, columns, logits, winner_changes, _ = ranked(split, scores, True)
    float_rows, float_columns, _, _, _ = ranked(split, scores, False)
    if rows.size < max(budgets):
        raise ValueError('{} has only {} eligible events'.format(split.name, rows.size))
    timed_targets = split.targets[rows, columns]
    future_targets = split.future_targets[rows, columns]
    future_lead = split.future_lead[rows, columns]
    float_timed = split.targets[float_rows, float_columns]
    float_future = split.future_targets[float_rows, float_columns]
    valid_scores = scores[split.valid.numpy()]
    saturated = int(np.count_nonzero((valid_scores < -32) | (valid_scores >= 32)))
    result = []
    for budget in budgets:
        cutoff = fixed_thresholds[str(budget)]['cutoff_q10']
        accepted = logits >= cutoff
        timed_accepted = timed_targets[accepted]
        future_accepted = future_targets[accepted]
        result.append({
            'budget': budget,
            'timed_unique': unique_count(timed_targets[:budget]),
            'future_unique': unique_count(future_targets[:budget]),
            'timed_positive_decisions': int(np.count_nonzero(timed_targets[:budget])),
            'future_positive_decisions': int(np.count_nonzero(future_targets[:budget])),
            'timed_float_unique': unique_count(float_timed[:budget]),
            'future_float_unique': unique_count(float_future[:budget]),
            'timed_reachable': split.opportunities,
            'future_reachable': split.future.opportunities,
            'validation_cutoff_q10': cutoff,
            'cutoff_decisions': int(accepted.sum()),
            'cutoff_timed_unique': unique_count(timed_accepted),
            'cutoff_future_unique': unique_count(future_accepted),
            'calibrated_expected_unique': (calibrated_hits(future_targets[:budget], future_lead[:budget], calibration)
                                           if calibration is not None else None),
            'cutoff_calibrated_expected_unique': (calibrated_hits(future_accepted, future_lead[accepted], calibration)
                                                  if calibration is not None else None),
            'q10_winner_changes': winner_changes,
            'q10_saturated_candidates': saturated,
        })
    return result


def train_arm(architecture, label, training, validation, calibration, positive_weight, args, log):
    name = '{}_{}'.format(architecture, label)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    model = Scorer(architecture == 'lstm').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    views = [label_view(split, label, calibration) for split in training]
    blocks = training_blocks(views, args.block, args.burn)
    positive_mass = sum(float(split.labels.sum()) for split in views)
    log('{} positive training mass {:.2f}, score parameter count {}'.format(
        name, positive_mass, sum(parameter.numel() for parameter in model.parameters())))
    key, best_epoch, best_state = None, 0, None
    history = []
    early = None
    for epoch in range(1, args.epochs + 1):
        start = time.monotonic()
        if label in ('timed', 'future'):
            loss = train_epoch(model, blocks, args.unroll, args.batch_size, optimizer, positive_weight, device, rng)
        else:
            loss = weighted_epoch(model, blocks, args, optimizer, positive_weight, device, rng)
        scores = predict(model, validation, device)
        epoch_key = selection_key(validation, scores, args, calibration)
        history.append({'epoch': epoch, 'loss': loss, 'validation_key': list(epoch_key),
                        'seconds': time.monotonic() - start})
        log('{} epoch {} loss {:.5f}, common validation {} ({:.1f}s)'.format(
            name, epoch, loss, epoch_key, history[-1]['seconds']))
        if key is None or epoch_key > key:
            key, best_epoch = epoch_key, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == args.early_epoch:
            early = epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch >= args.early_epoch and epoch - best_epoch >= args.patience:
            break
    checkpoints = []
    for desc, (epoch, state) in (('epoch{}'.format(args.early_epoch), early), ('best', (best_epoch, best_state))):
        stem = '{}_{}'.format(name, desc)
        write_checkpoint(args.output, stem, model, state, epoch, architecture, label, args, validation.width)
        scores = predict(model.to(device), validation, device)
        thresholds = evaluate(validation, scores, args.budgets)['thresholds']
        checkpoints.append((stem, architecture, label, epoch, thresholds))
    return history, checkpoints


def protocol_matches(split, reference):
    return split.width == reference.width and all(split.metadata.get(field) == reference.metadata.get(field)
                                                    for field in PROTOCOL)


def state_ablation(model, split, args, thresholds, calibration):
    model.cpu()
    yield 'zero_state', summarize(split, zero_state_scores(model, split), args.budgets, thresholds, calibration)
    for width in args.state_windows:
        scores = window_scores(model, split, width)
        yield 'ordered_{}'.format(width), summarize(split, scores, args.budgets, thresholds, calibration)
        for seed in args.state_seeds:
            scores = window_scores(model, split, width, seed)
            yield 'shuffled_{}_{}'.format(width, seed), summarize(split, scores, args.budgets, thresholds, calibration)


def write_metrics(path, records):
    with path.open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)


def load_scorer(directory, stem, device):
    state = torch.load(directory / (stem + '.pt'), map_location='cpu', weights_only=True)
    model = Scorer(state['recurrent']).to(device)
    model.load_state_dict(state['state_dict'])
    return model, state


def recalibrate(directory, validation, args, calibration, log):
    """Score frozen preselected checkpoints against a new timing proxy."""
    with (directory / 'metrics.json').open() as source:
        saved = json.load(source)
    if saved['validation_split'] != validation.name:
        raise ValueError('the validation split does not match the original experiment')
    scores = []
    checkpoints = saved['validation_thresholds']
    for path in [args.validation] + args.assessment + args.test:
        split = validation if path == args.validation else OutcomeSplit(path)
        if not protocol_matches(split, validation):
            raise ValueError('assessment data uses a different candidate or label protocol')
        for stem, thresholds in checkpoints.items():
            model, state = load_scorer(directory, stem, args.device)
            rows = summarize(split, predict(model, split, args.device), args.budgets, thresholds, calibration)
            scores.extend(dict(arm=stem, architecture=state['architecture'], label=state['label'],
                               epoch=state['epoch'], split=split.name, **row) for row in rows)
            log('{} {} calibrated unique {}'.format(
                split.name, stem, [round(row['calibrated_expected_unique'], 2) for row in rows],
            ))
    write_metrics(args.output / 'metrics.csv', scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, nargs='+', default=[])
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--assessment', type=Path, nargs='*', default=[])
    parser.add_argument('--test', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--zip', type=Path)
    parser.add_argument('--page-pc', type=Path)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--recalibrate', type=Path,
                        help='rescore a frozen training run without changing its checkpoints or cutoffs')
    parser.add_argument('--selection', choices=('future', 'timed', 'calibrated'), default='future')
    parser.add_argument('--labels', choices=LABELS, nargs='+', default=['timed', 'future'])
    parser.add_argument('--architectures', choices=('event', 'lstm'), nargs='+', default=['event', 'lstm'])
    parser.add_argument('--budgets', type=int, nargs='+', default=[2000, 4000])
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--early-epoch', type=int, default=4)
    parser.add_argument('--unroll', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--block', type=int, default=256)
    parser.add_argument('--burn', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--seed', type=int, default=605)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--state-ablation', action='store_true')
    parser.add_argument('--state-windows', type=int, nargs='+', default=[8, 16, 32])
    parser.add_argument('--state-seeds', type=int, nargs='+', default=[605, 606, 607])
    args = parser.parse_args()
    paths = args.train + [args.validation] + args.assessment + args.test
    if len({path.resolve() for path in paths}) != len(paths) or len({path.stem for path in paths}) != len(paths):
        parser.error('development, validation, and held-out inputs must be distinct')
    if (min(args.budgets + args.state_windows + [args.epochs, args.patience, args.early_epoch, args.unroll, args.batch_size,
                            args.block, args.threads]) < 1 or args.early_epoch > args.epochs or args.burn < 0
            or not math.isfinite(args.lr) or args.lr <= 0 or len(set(args.labels)) != len(args.labels)
            or len(set(args.architectures)) != len(args.architectures)):
        parser.error('invalid or duplicate training settings')
    if (any(label.startswith('soft') for label in args.labels) or args.selection == 'calibrated') and args.calibration is None:
        parser.error('soft labels and calibrated selection require --calibration with development service_cycles')
    if (args.recalibrate is None and not args.train) or (args.recalibrate is not None and args.calibration is None):
        parser.error('pass --train for a new run or --recalibrate with --calibration for a fixed run')
    args.budgets = sorted(set(args.budgets))

    calibration = load_calibration(args.calibration)
    torch.set_num_threads(args.threads)
    validation = OutcomeSplit(args.validation)
    args.output.mkdir(parents=True, exist_ok=False)

    def log(message):
        print(message, flush=True)
        with (args.output / 'training.log').open('a') as stream:
            stream.write(message + '\n')

    if args.recalibrate is not None:
        recalibrate(args.recalibrate, validation, args, calibration, log)
        return

    training = [OutcomeSplit(path) for path in args.train]
    if any(not protocol_matches(split, validation) for split in training):
        raise ValueError('development data uses a different candidate or label protocol')
    candidates = sum(int(split.valid.sum()) for split in training)
    timed_positives = sum(int(split.labels.sum()) for split in training)
    future_positives = sum(int(np.count_nonzero(split.future_targets)) for split in training)
    if not 0 < timed_positives <= future_positives < candidates:
        raise ValueError('training data must contain positive and negative outcomes')
    # Equal class weight isolates the target choice in the timed and future comparison.
    weight = min(32.0, math.sqrt((candidates - timed_positives) / timed_positives))
    positive_weight = torch.tensor(weight, device=args.device)
    log('Training label counts: {} timed, {} future of {} candidates, shared positive weight {:.6f}'.format(
        timed_positives, future_positives, candidates, weight))
    history, checkpoints = {}, []
    for architecture in args.architectures:
        for label in args.labels:
            history[architecture + '_' + label], selected = train_arm(
                architecture, label, training, validation, calibration, positive_weight, args, log,
            )
            checkpoints.extend(selected)

    del training
    # The held-out splits are opened after every checkpoint and threshold has been chosen.
    tables = read_table(args.page_pc) if args.page_pc else None
    table_thresholds = (evaluate(validation, table_scores(validation, tables), args.budgets)['thresholds']
                        if tables else None)
    records, state_records = [], []
    for path in [args.validation] + args.assessment + args.test:
        split = validation if path == args.validation else OutcomeSplit(path)
        if not protocol_matches(split, validation):
            raise ValueError('held-out data uses a different candidate or label protocol')
        methods = checkpoints
        if tables:
            methods = [('page_pc', 'table', 'timed', 0, table_thresholds)] + methods
        for stem, architecture, label, epoch, thresholds in methods:
            if architecture == 'table':
                scores = table_scores(split, tables)
            else:
                model, _ = load_scorer(args.output, stem, args.device)
                scores = predict(model, split, args.device)
            rows = summarize(split, scores, args.budgets, thresholds, calibration)
            records.extend(dict(arm=stem, architecture=architecture, label=label, epoch=epoch,
                                split=split.name, **row) for row in rows)
            log('{} {} epoch {} timed/future {}'.format(
                split.name, stem, epoch, [(row['timed_unique'], row['future_unique']) for row in rows],
            ))
            if args.state_ablation and path != args.validation and architecture == 'lstm' and stem.endswith('_best'):
                state_records.extend(dict(arm=stem, epoch=epoch, split=split.name, state='carried', **row)
                                     for row in rows)
                for mode, ablated in state_ablation(model, split, args, thresholds, calibration):
                    state_records.extend(dict(arm=stem, epoch=epoch, split=split.name, state=mode, **row)
                                         for row in ablated)

    description = {'training_splits': [path.stem for path in args.train],
                   'validation_split': args.validation.stem, 'test_splits': [path.stem for path in args.test],
                   'assessment_splits': [path.stem for path in args.assessment],
                   'dimensions': [INPUTS, HIDDEN, CANDIDATE_FEATURES, RANK],
                   'training_label_counts': {'timed': timed_positives, 'future': future_positives,
                                             'valid': candidates},
                   'positive_weight': weight,
                   'seed': args.seed, 'max_epochs': args.epochs, 'patience': args.patience,
                   'unroll': args.unroll, 'block': args.block, 'burn': args.burn,
                   'selection': args.selection,
                   'selection_rule': 'Maximize sum of unique target scores across validation budgets, then smallest budget',
                   'calibration_samples': None if calibration is None else int(calibration.size),
                   'calibration_file': None if args.calibration is None else str(args.calibration),
                   'history': history,
                   'validation_thresholds': {stem: thresholds for stem, _, _, _, thresholds in checkpoints},
                   'runs': records}
    with (args.output / 'metrics.json').open('w') as target:
        json.dump(description, target, indent=2, allow_nan=False)
        target.write('\n')
    write_metrics(args.output / 'metrics.csv', records)
    if state_records:
        write_metrics(args.output / 'state_ablation.csv', state_records)
    if args.zip:
        with zipfile.ZipFile(args.zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(args.output.iterdir()):
                if path.resolve() != args.zip.resolve():
                    archive.write(path, path.name)


if __name__ == '__main__':
    main()
