#!/usr/bin/env python3
"""Test fixed weights at held-out budgets and validation-derived thresholds."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from train import HIDDEN, Scorer, Split, evaluate, predict, ranked, read_table, table_scores


def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = Scorer(checkpoint['recurrent'])
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()


@torch.no_grad()
def window_scores(model, split, width, seed=None):
    output = np.zeros(split.targets.shape, dtype=np.float32)
    rng = np.random.default_rng(seed)
    valid = split.valid.numpy().any(axis=1)
    for first, last in split.streams:
        # Eligible rows are past the simulator's warmup. Keep the same recent
        # two events when disturbing older order, including the trigger itself.
        rows = np.flatnonzero(valid[first:last]) + first
        if rows.size and rows[0] - first < width - 1:
            raise ValueError('window exceeds available context for the first eligible row')
        for start in range(0, rows.size, 2048):
            anchors = rows[start:start + 2048]
            indices = anchors[:, None] - np.arange(width - 1, -1, -1)
            if seed is not None and width > 3:
                shuffled = rng.random((anchors.size, width - 2)).argsort(axis=1)
                indices[:, :-2] = np.take_along_axis(indices[:, :-2], shuffled, axis=1)
            recent = split.events[indices]
            hidden, _ = model.cell(recent)
            current = split.events[anchors]
            query = model.query(torch.cat((current, hidden[:, -1]), dim=-1))
            candidates = split.candidates[anchors]
            scores = (query[:, None, :] * model.key(candidates)).sum(dim=-1)
            output[anchors] = (scores + model.bias(candidates).squeeze(-1)).numpy()
    return output


@torch.no_grad()
def zero_state_scores(model, split):
    output = np.zeros(split.targets.shape, dtype=np.float32)
    for start in range(0, split.rows, 4096):
        events = split.events[start:start + 4096]
        candidates = split.candidates[start:start + 4096]
        query = model.query(torch.cat((events, events.new_zeros((events.shape[0], HIDDEN))), dim=-1))
        output[start:start + len(events)] = (
            (query[:, None, :] * model.key(candidates)).sum(dim=-1) + model.bias(candidates).squeeze(-1)
        ).numpy()
    return output


def evaluate_thresholds(split, scores, thresholds):
    rows, columns, values, _, _ = ranked(split, scores, True)
    targets = split.targets[rows, columns]
    metrics = {}
    for budget, threshold in thresholds.items():
        accepted = values >= threshold['cutoff_q10']
        actual = targets[accepted]
        unique = np.unique(actual[actual > 0]).size
        metrics[budget] = {'validation_cutoff_q10': threshold['cutoff_q10'],
                           'cutoff_decisions': int(accepted.sum()),
                           'cutoff_positive_decisions': int(np.count_nonzero(actual)),
                           'cutoff_unique_targets': int(unique)}
    winners = np.full(split.rows, -1, dtype=np.int16)
    winners[rows] = columns
    return metrics, winners


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--page-pc', type=Path, required=True)
    parser.add_argument('--test', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--windows', type=int, nargs='+', default=[8, 16, 32])
    parser.add_argument('--seeds', type=int, nargs='+', default=[605, 606, 607])
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.windows + [args.threads]) < 1:
        parser.error('windows and thread count must be positive')
    torch.set_num_threads(args.threads)
    with (args.experiment / 'metrics.json').open() as stream:
        training = json.load(stream)
    validation_thresholds = training['validation_thresholds']
    selected = training['selected_lstm']
    budgets = sorted(map(int, validation_thresholds[selected]))
    event = load_model(args.experiment / 'event.pt')
    model = load_model(args.experiment / 'lstm.pt')
    tables = read_table(args.page_pc)
    results = []
    for path in args.test:
        split = Split(path)
        if split.name in training['training_splits'] + [training['validation_split']]:
            raise ValueError('this evaluation expects an independent held-out split')
        continuous = predict(model, split, 'cpu')
        _, normal_winners = evaluate_thresholds(split, continuous, validation_thresholds[selected])

        def record(name, scores, calibration):
            metrics = evaluate(split, scores, budgets)
            applied, winners = evaluate_thresholds(split, scores, validation_thresholds[calibration])
            changed = int(np.count_nonzero(winners != normal_winners))
            print(split.name, name, [row['unique_targets'] for row in metrics['budgets']], changed, flush=True)
            for row in metrics['budgets']:
                results.append(dict(split=split.name, arm=name, changed_winners=changed,
                                    **row, **applied[str(row['budget'])]))

        record('page_pc', table_scores(split, tables), 'page_pc')
        record('event', predict(event, split, 'cpu'), 'event')
        record(selected, continuous, selected)
        record('same_weights_zero_state', zero_state_scores(model, split), selected)
        for width in args.windows:
            record('same_weights_ordered_{}'.format(width), window_scores(model, split, width), selected)
            for seed in args.seeds:
                name = 'same_weights_shuffled_{}_{}'.format(width, seed)
                record(name, window_scores(model, split, width, seed), selected)
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)


if __name__ == '__main__':
    main()
