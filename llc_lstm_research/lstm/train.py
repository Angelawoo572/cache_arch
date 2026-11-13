#!/usr/bin/env python3
"""Train and compare compact LLC scorers using only NoPF replay data."""

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


INPUTS, HIDDEN, RANK, CANDIDATE_FEATURES = 7, 8, 4, 6
SCORE_SCALE = 1024


class Split:
    def __init__(self, path):
        self.name = path.stem
        with np.load(path, allow_pickle=False) as data:
            self.events = torch.from_numpy(data['events'].astype(np.float32, copy=False))
            self.candidates = torch.from_numpy(data['candidates'].astype(np.float32, copy=False))
            self.labels = torch.from_numpy(data['labels'].astype(np.bool_, copy=False))
            self.valid = torch.from_numpy(data['valid'].astype(np.bool_, copy=False))
            self.targets = data['targets']
            self.cores = data['cores']
            self.sequence = data['sequence']
            self.event_id = data['event_id']
            self.metadata = json.loads(data['metadata'].item())
        rows, width, features = self.candidates.shape
        self.width = width
        if (rows == 0 or self.events.shape != (rows, INPUTS) or features != CANDIDATE_FEATURES
                or self.labels.shape != (rows, width) or self.valid.shape != (rows, width)
                or self.targets.shape != (rows, width) or self.cores.shape != (rows,)
                or self.sequence.shape != (rows,) or self.event_id.shape != (rows,)):
            raise ValueError('invalid input shape in {}'.format(path))
        if (self.metadata['no_prefetch'] is not True or self.metadata['candidate_cap'] != width
                or self.metadata['events'] != rows):
            raise ValueError('NoPF metadata does not match {}'.format(path))
        if (torch.any(self.labels & ~self.valid) or np.any((self.targets != 0) != self.labels.numpy())
                or np.unique(self.event_id).size != rows):
            raise ValueError('invalid label or event ID in {}'.format(path))
        boundaries = np.flatnonzero(self.cores[1:] != self.cores[:-1]) + 1
        points = np.concatenate(([0], boundaries, [rows]))
        self.streams = list(zip(points[:-1], points[1:]))
        if (len(self.streams) != np.unique(self.cores).size
                or any(np.any(np.diff(self.sequence[a:b]) != 1) for a, b in self.streams)):
            raise ValueError('split must contain one contiguous, chronological sequence per core')
        self.opportunities = int(np.unique(self.targets[self.targets > 0]).size)
        self.rows = rows
        self.path = path


class Scorer(nn.Module):
    def __init__(self, recurrent):
        super().__init__()
        self.recurrent = recurrent
        self.cell = nn.LSTM(INPUTS, HIDDEN, batch_first=True)
        self.query = nn.Linear(INPUTS + HIDDEN, RANK)
        self.key = nn.Linear(CANDIDATE_FEATURES, RANK)
        self.bias = nn.Linear(CANDIDATE_FEATURES, 1)

    def forward(self, events, candidates, state=None):
        if self.recurrent:
            hidden, state = self.cell(events, state)
        else:
            hidden = events.new_zeros(events.shape[:-1] + (HIDDEN,))
        query = self.query(torch.cat((events, hidden), dim=-1))
        scores = (query.unsqueeze(-2) * self.key(candidates)).sum(dim=-1)
        return scores + self.bias(candidates).squeeze(-1), state


def read_table(path):
    tables = {2: {}, 5: {}, 6: {}}
    with path.open() as source:
        if source.readline().strip() != 'LLC_PAGE_PC_V1':
            raise ValueError('unexpected page-PC file header')
        for line in source:
            width, *fields = map(int, line.split())
            if width not in tables or len(fields) != width + 2:
                raise ValueError('invalid page-PC file row')
            positive, total = fields[-2:]
            if not 0 <= positive <= total:
                raise ValueError('invalid page-PC counts')
            tables[width][tuple(fields[:-2])] = positive, total
    return tables


def table_scores(split, tables):
    with np.load(split.path, allow_pickle=False) as source:
        keys = source['keys']
    scores = np.zeros(split.targets.shape, dtype=np.float32)
    rows, columns = np.nonzero(split.valid.numpy())
    memo = {}
    for row, column, values in zip(rows, columns, keys[rows, columns]):
        key = tuple(int(value) for value in values)
        logit = memo.get(key)
        if logit is None:
            probability = 0.005
            for width in (2, 5, 6):
                positive, total = tables[width].get(key[:width], (0, 0))
                probability = (positive + 20 * probability) / (total + 20)
            logit = math.log(probability / (1.0 - probability))
            memo[key] = logit
        scores[row, column] = logit
    return scores


def q10(scores):
    return np.clip(np.floor(scores * SCORE_SCALE + 0.5), -32768, 32767).astype(np.int32)


def ranked(split, scores, use_integer):
    valid = split.valid.numpy()
    eligible = np.flatnonzero(valid.any(axis=1))
    raw = scores[eligible]
    mask = valid[eligible]
    quantized = q10(raw)
    masked_float = np.where(mask, raw, -np.inf)
    masked_int = np.where(mask, quantized, -32769)
    float_winners = masked_float.argmax(axis=1)
    int_winners = masked_int.argmax(axis=1)
    winners = int_winners if use_integer else float_winners
    indices = np.arange(eligible.size)
    chosen = (masked_int if use_integer else masked_float)[indices, winners]
    # Target-independent event ID mixing avoids favoring an early region on tied scores.
    ties = split.event_id[eligible].astype(np.uint64) * np.uint64(11400714819323198485)
    order = np.lexsort((ties, -chosen))
    return (eligible[order], winners[order], chosen[order],
            int(np.count_nonzero(float_winners != int_winners)),
            int(np.count_nonzero((masked_int == masked_int.max(axis=1, keepdims=True)).sum(axis=1) > 1)))


def evaluate(split, scores, budgets):
    rows, columns, chosen, changes, ties = ranked(split, scores, True)
    float_rows, float_columns, _, _, _ = ranked(split, scores, False)
    if rows.size < max(budgets):
        raise ValueError('{} has only {} eligible events'.format(split.name, rows.size))
    quantized = chosen
    target = split.targets[rows, columns]
    float_target = split.targets[float_rows, float_columns]
    outcomes = []
    thresholds = {}
    for budget in budgets:
        selected = target[:budget]
        hits = int(np.count_nonzero(selected))
        unique = int(np.unique(selected[selected > 0]).size)
        float_unique = int(np.unique(float_target[:budget][float_target[:budget] > 0]).size)
        cutoff = int(quantized[budget - 1])
        above = int(np.count_nonzero(quantized > cutoff))
        inclusive = int(np.count_nonzero(quantized >= cutoff))
        if abs(above - budget) < abs(inclusive - budget):
            cutoff += 1
            cutoff_sends = above
        else:
            cutoff_sends = inclusive
        outcomes.append({'budget': budget, 'positive_decisions': hits, 'unique_targets': unique,
                         'float_unique_targets': float_unique, 'unique_fraction_of_reachable': unique / split.opportunities
                         if split.opportunities else 0.0})
        thresholds[str(budget)] = {'cutoff_q10': cutoff, 'predicted_sends': cutoff_sends,
                                   'event_rate': cutoff_sends / split.rows,
                                   'tie_bucket_size': inclusive - above}
    valid_scores = scores[split.valid.numpy()]
    return {'events': split.rows, 'eligible_anchors': int(rows.size), 'reachable_targets': split.opportunities,
            'valid_candidates': int(valid_scores.size),
            'score_quantiles': np.quantile(valid_scores, [0, .01, .1, .5, .9, .99, 1]).tolist(),
            'q10_ties': ties, 'q10_candidate_winner_changes': changes,
            'q10_saturated_candidates': int(np.count_nonzero((valid_scores < -32) | (valid_scores >= 32))),
            'budgets': outcomes, 'thresholds': thresholds}


@torch.no_grad()
def predict(model, split, device):
    model.eval()
    output = np.empty(split.targets.shape, dtype=np.float32)
    for first, last in split.streams:
        state = None
        for start in range(first, last, 4096):
            stop = min(start + 4096, last)
            x = split.events[start:stop].to(device).unsqueeze(0)
            z = split.candidates[start:stop].to(device).unsqueeze(0)
            scores, state = model(x, z, state)
            output[start:stop] = scores.squeeze(0).cpu().numpy()
    return output


def training_blocks(splits, block, burn):
    # Each sampled block has the recent causal events needed to warm its state.
    # Loss is masked for these events so every anchor is supervised once per epoch.
    output = []
    for split in splits:
        for first, last in split.streams:
            for start in range(first, last, block):
                output.append((split, max(first, start - burn), start, min(last, start + block)))
    return output


def batch(blocks, device):
    length = max(stop - begin for _, begin, _, stop in blocks)
    width = blocks[0][0].width
    events = torch.zeros((len(blocks), length, INPUTS))
    candidates = torch.zeros((len(blocks), length, width, CANDIDATE_FEATURES))
    labels = torch.zeros((len(blocks), length, width), dtype=torch.bool)
    mask = torch.zeros_like(labels)
    for row, (split, begin, start, stop) in enumerate(blocks):
        n = stop - begin
        events[row, :n] = split.events[begin:stop]
        candidates[row, :n] = split.candidates[begin:stop]
        labels[row, :n] = split.labels[begin:stop]
        mask[row, start - begin:n] = split.valid[start:stop]
    return (tensor.to(device) for tensor in (events, candidates, labels, mask))


def train_epoch(model, blocks, unroll, batch_size, optimizer, positive_weight, device, rng):
    model.train()
    indices = rng.permutation(len(blocks))
    total_loss, total_count = 0.0, 0
    # Detach at unroll boundaries while carrying state forward inside each block.
    for offset in range(0, len(indices), batch_size):
        x, z, y, valid = batch([blocks[i] for i in indices[offset:offset + batch_size]], device)
        state = None
        step = unroll if model.recurrent else x.shape[1]
        optimizer.zero_grad(set_to_none=True)
        batch_count = int(valid.sum())
        if not batch_count:
            continue
        for start in range(0, x.shape[1], step):
            stop = start + step
            mask = valid[:, start:stop]
            if not bool(mask.any()):
                if model.recurrent:
                    with torch.no_grad():
                        _, state = model.cell(x[:, start:stop], state)
                continue
            scores, state = model(x[:, start:stop], z[:, start:stop], state)
            targets = y[:, start:stop][mask].to(scores.dtype)
            loss = F.binary_cross_entropy_with_logits(scores[mask], targets, pos_weight=positive_weight,
                                                       reduction='sum')
            (loss / batch_count).backward()
            total_loss += float(loss.detach())
            if state is not None:
                state = tuple(value.detach() for value in state)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        total_count += batch_count
    return total_loss / total_count


def export(model, width, path):
    mode = 'lstm' if model.recurrent else 'event'
    model = model.cpu()
    cell = model.cell
    parts = (cell.weight_ih_l0, cell.weight_hh_l0, cell.bias_ih_l0 + cell.bias_hh_l0,
             model.query.weight, model.query.bias, model.key.weight, model.key.bias,
             model.bias.weight, model.bias.bias)
    with path.open('w') as stream:
        stream.write('LLC_LSTM_V1 {} {} {} {} {} {} 0\n'.format(
            mode, INPUTS, HIDDEN, RANK, CANDIDATE_FEATURES, width))
        for parameter in parts:
            values = parameter.detach().reshape(-1).numpy()
            stream.write(' '.join('{:.9g}'.format(float(value)) for value in values) + '\n')


def train(args):
    torch.set_num_threads(args.threads)
    train_splits = [Split(path) for path in args.train]
    validation = Split(args.validation)
    widths = {split.width for split in train_splits + [validation]}
    if len(widths) != 1:
        raise ValueError('candidate cap changed between splits')
    protocol = ('event_features', 'candidate_features', 'offset_order', 'age_band_order',
                'page_history', 'first_anchor', 'lead_cycles', 'target_definition')
    for split in train_splits:
        if any(split.metadata[field] != validation.metadata[field] for field in protocol):
            raise ValueError('candidate or feature protocol changed in {}'.format(split.name))
    width = validation.width
    device = torch.device(args.device)
    blocks = training_blocks(train_splits, args.block, args.burn)
    positives = sum(int(split.labels.sum()) for split in train_splits)
    candidates = sum(int(split.valid.sum()) for split in train_splits)
    if not 0 < positives < candidates:
        raise ValueError('training data must contain positive and negative candidates')
    weight = min(32.0, math.sqrt((candidates - positives) / positives))
    weight_tensor = torch.tensor(weight, device=device)
    args.output.mkdir(parents=True, exist_ok=False)
    log_path = args.output / 'training.log'
    report = {'score_scale': SCORE_SCALE, 'seed': args.seed, 'training_splits': [s.name for s in train_splits],
              'validation_split': validation.name, 'positive_candidates': positives,
              'valid_candidates': candidates, 'positive_weight': weight, 'candidate_cap': width,
              'burn_in_events': args.burn, 'training_block_events': args.block,
              'protocol': {field: validation.metadata[field] for field in protocol},
              'arms': {}}
    trained = {}
    tables = read_table(args.page_pc)

    def log(message):
        print(message, flush=True)
        with log_path.open('a') as stream:
            stream.write(message + '\n')

    log('Training on {} NoPF events, {} positive of {} valid candidates'.format(
        sum(split.rows for split in train_splits), positives, candidates))
    report['arms']['page_pc'] = {'validation': evaluate(validation, table_scores(validation, tables), args.budgets)}
    for unroll in [0] + args.windows:
        name = 'lstm_w{}'.format(unroll) if unroll else 'event'
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        model = Scorer(unroll > 0).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        best_key, best_epoch, best_state, best_metrics = None, None, None, None
        epochs = []
        for epoch in range(1, args.epochs + 1):
            start = time.monotonic()
            loss = train_epoch(model, blocks, unroll, args.batch_size, optimizer, weight_tensor, device, rng)
            scores = predict(model, validation, device)
            metrics = evaluate(validation, scores, args.budgets)
            counts = [row['unique_targets'] for row in metrics['budgets']]
            key = sum(counts), counts[0]
            epochs.append({'epoch': epoch, 'loss': loss, 'validation_unique': counts,
                           'seconds': time.monotonic() - start})
            log('{} epoch {} loss {:.5f} validation unique {} ({:.1f}s)'.format(
                name, epoch, loss, counts, epochs[-1]['seconds']))
            if best_key is None or key > best_key:
                best_key, best_epoch, best_metrics = key, epoch, metrics
                best_state = {k: value.detach().cpu().clone() for k, value in model.state_dict().items()}
        model.load_state_dict(best_state)
        export(model, width, args.output / '{}.txt'.format(name))
        torch.save({'state_dict': best_state, 'recurrent': unroll > 0, 'unroll': unroll,
                    'epoch': best_epoch, 'candidate_cap': width}, args.output / '{}.pt'.format(name))
        trained[name] = model
        report['arms'][name] = {'validation': best_metrics, 'epochs': epochs, 'best_epoch': best_epoch}
    selection = max((name for name in trained if name.startswith('lstm_')),
                    key=lambda name: (sum(r['unique_targets'] for r in report['arms'][name]['validation']['budgets']),
                                      report['arms'][name]['validation']['budgets'][0]['unique_targets']))
    report['selected_lstm'] = selection
    report['validation_thresholds'] = {name: arm['validation']['thresholds'] for name, arm in report['arms'].items()}
    log('Validation selected {}'.format(selection))
    shutil.copyfile(args.output / '{}.txt'.format(selection), args.output / 'lstm.txt')
    shutil.copyfile(args.output / '{}.pt'.format(selection), args.output / 'lstm.pt')
    for path in args.test:
        split = Split(path)
        if split.width != width:
            raise ValueError('candidate cap changed in held-out split')
        if any(split.metadata[field] != validation.metadata[field] for field in protocol):
            raise ValueError('candidate or feature protocol changed in {}'.format(split.name))
        if split.name in report['training_splits'] + [validation.name]:
            raise ValueError('held-out split name duplicates a development split')
        all_scores = [('page_pc', table_scores(split, tables))]
        all_scores.extend((name, predict(model.to(device), split, device)) for name, model in trained.items())
        for name, scores in all_scores:
            metrics = evaluate(split, scores, args.budgets)
            report['arms'][name].setdefault('test', {})[split.name] = metrics
            log('{} {} held-out unique {}'.format(name, split.name,
                [row['unique_targets'] for row in metrics['budgets']]))
    with (args.output / 'metrics.json').open('w') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    with (args.output / 'metrics.csv').open('w', newline='') as stream:
        fields = ('arm', 'split', 'budget', 'positive_decisions', 'unique_targets', 'float_unique_targets',
                  'unique_fraction_of_reachable')
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, arm in report['arms'].items():
            splits = [(validation.name, arm['validation'])] + list(arm.get('test', {}).items())
            for split_name, metrics in splits:
                for row in metrics['budgets']:
                    writer.writerow(dict(arm=name, split=split_name, **row))
    if args.zip:
        artifacts = sorted(args.output.iterdir())
        with zipfile.ZipFile(args.zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in artifacts:
                archive.write(path, path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, nargs='+', required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--test', type=Path, nargs='*', default=[])
    parser.add_argument('--page-pc', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--zip', type=Path)
    parser.add_argument('--windows', type=int, nargs='+', default=[8, 16, 32])
    parser.add_argument('--budgets', type=int, nargs='+', default=[2000, 4000])
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--block', type=int, default=256)
    parser.add_argument('--burn', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--seed', type=int, default=605)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if min(args.windows + args.budgets + [args.epochs, args.batch_size, args.block, args.threads]) < 1:
        parser.error('windows, budgets, epochs, batch size, block and thread count must be positive')
    if args.burn < 0 or args.lr <= 0 or len(set(args.windows)) != len(args.windows):
        parser.error('use nonnegative burn-in, positive learning rate, and distinct windows')
    splits = args.train + [args.validation] + args.test
    if len({path.resolve() for path in splits}) != len(splits) or len({path.stem for path in splits}) != len(splits):
        parser.error('each split path and filename stem must be distinct')
    train(args)


if __name__ == '__main__':
    main()
