#!/usr/bin/env python3
"""Train matched LLC scorer arms, then evaluate frozen weights on NoPF data."""

import argparse
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from train import HIDDEN, predict, ranked, training_blocks
from v2_train import (PROTOCOL, OutcomeSplit, calibrated_hits, label_view, load_calibration,
                      protocol_matches, summarize, weighted_batch, write_metrics)
from v3_model import Model, complexity, export, load_checkpoint, save_checkpoint


DEFAULT_ARMS = ('event_v2', 'lstm_v2', 'event_mlp8', 'lstm_mlp4', 'lstm_mlp8', 'lstm_mlp16')
TRAIN_SOURCES = {'605_nopf_10m', '605_nopf_30m', '605_nopf_50m'}
VALIDATION_SOURCES = {'605_nopf_70m', '605_nopf_130m'}
OBJECTIVES = ('timed', 'future', 'soft')


def make_model(architecture):
    kind, head = architecture.split('_')
    if head == 'v2':
        return Model(kind == 'lstm')
    return Model(kind == 'lstm', 'mlp', int(head[3:]))


def nopf_calibration(path):
    if path is not None:
        metadata = json.loads(path.read_text())
        if 'nopf' not in Path(metadata.get('source', '')).parts:
            raise ValueError('service calibration must identify a NoPF demand source')
    return load_calibration(path)


def update_state(model, events, candidates, state):
    if model.recurrent:
        with torch.no_grad():
            _, state = model(events, candidates, state)
    return state


def ranking_loss(scores, targets, valid):
    best_label = targets.masked_fill(~valid, -1).amax(-1, keepdim=True)
    top = valid & (targets == best_label)
    lower = valid & (targets < best_label)
    anchors = lower.any(-1)
    if not bool(anchors.any()):
        return scores.new_zeros(())
    top_score = scores.masked_fill(~top, -torch.inf)[anchors].amax(-1)
    lower_score, lower_index = scores.masked_fill(~lower, -torch.inf)[anchors].max(-1)
    label_difference = best_label.squeeze(-1)[anchors] - targets[anchors].gather(1, lower_index[:, None])[:, 0]
    return (label_difference * F.softplus(lower_score - top_score)).sum()


def training_scores(model, events, candidates, state, dropout):
    if not dropout:
        return model(events, candidates, state)
    hidden, state = model.cell(events, state)
    keep = torch.rand((*hidden.shape[:-1], 1), device=hidden.device) >= dropout
    return model.score_events(events, candidates, hidden * keep / (1 - dropout)), state


def train_epoch(model, blocks, unroll, optimizer, class_weight, args, rng, with_pairwise):
    model.train()
    indices = rng.permutation(len(blocks))
    sum_bce, sum_rank, count_valid, count_anchor = 0.0, 0.0, 0, 0
    for offset in range(0, len(indices), args.batch_size):
        batch_blocks = [blocks[i] for i in indices[offset:offset + args.batch_size]]
        events, candidates, labels, valid, _ = weighted_batch(batch_blocks, args.device)
        num_valid = int(valid.sum())
        if num_valid == 0:
            continue
        num_anchor = int(valid.any(-1).sum())
        state, step = None, unroll if model.recurrent else events.shape[1]
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, events.shape[1], step):
            stop = start + step
            event_slice, candidate_slice = events[:, start:stop], candidates[:, start:stop]
            valid_slice, label_slice = valid[:, start:stop], labels[:, start:stop]
            if not bool(valid_slice.any()):
                state = update_state(model, event_slice, candidate_slice, state)
                continue
            scores, state = training_scores(model, event_slice, candidate_slice, state, args.state_dropout)
            bce = F.binary_cross_entropy_with_logits(scores[valid_slice], label_slice[valid_slice],
                                                     pos_weight=class_weight, reduction='sum')
            loss = bce / num_valid
            if with_pairwise:
                # Keep the scalar calibration loss. Add one order comparison per
                # supervised trigger, using its strongest lower-label contender.
                rank = ranking_loss(scores, label_slice, valid_slice)
                (loss + args.pairwise_weight * rank / num_anchor).backward()
                sum_rank += float(rank.detach())
            else:
                loss.backward()
            sum_bce += float(bce.detach())
            if state is not None:
                state = tuple(value.detach() for value in state)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        count_valid += num_valid
        count_anchor += num_anchor
    return sum_bce / count_valid, sum_rank / count_anchor


def validation_summary(model, validation, budgets, calibration, device):
    counts, wins = [], []
    for split in validation:
        scores = predict(model, split, device)
        rows, columns, logits, _, _ = ranked(split, scores, True)
        if rows.size < max(budgets):
            raise ValueError('insufficient eligible triggers in {}'.format(split.name))
        targets = split.future_targets[rows, columns]
        lead = split.future_lead[rows, columns]
        if calibration is not None:
            counts.append([calibrated_hits(targets[:size], lead[:size], calibration) for size in budgets])
        else:
            timed = split.targets[rows, columns]
            counts.append([int(np.unique(timed[:size][timed[:size] > 0]).size) for size in budgets])
        wins.append(logits)
    key = sum(map(sum, counts)), sum(row[0] for row in counts)
    return key, counts, wins


def pooled_thresholds(wins, budgets):
    ranked_scores = np.sort(np.concatenate(wins))[::-1]
    cutoffs = {}
    for size in budgets:
        pooled_budget = size * len(wins)
        cutoff = int(ranked_scores[pooled_budget - 1])
        above = int(np.count_nonzero(ranked_scores > cutoff))
        including = int(np.count_nonzero(ranked_scores >= cutoff))
        if pooled_budget - above < including - pooled_budget:
            cutoff += 1
        cutoffs[str(size)] = {'cutoff_q10': cutoff,
                              'desired_pooled_decisions': pooled_budget,
                              'validation_decisions': [int(np.count_nonzero(logits >= cutoff)) for logits in wins]}
    return cutoffs


def read_splits(paths):
    splits = [OutcomeSplit(path) for path in paths]
    if len({split.metadata['source'] for split in splits}) != len(splits):
        raise ValueError('data paths must use distinct NoPF regions')
    return splits


def check_protocol(splits, reference):
    for split in splits:
        if (not protocol_matches(split, reference)
                or split.metadata.get('correlation_table') != reference.metadata.get('correlation_table')):
            raise ValueError('{} uses a different candidate protocol'.format(split.name))


def describe(args):
    values = vars(args).copy()
    for key, value in values.items():
        if isinstance(value, Path):
            values[key] = str(value)
        elif isinstance(value, list):
            values[key] = [str(item) if isinstance(item, Path) else item for item in value]
    return values


def log_line(path, line):
    print(line, flush=True)
    with path.open('a') as stream:
        stream.write(line + '\n')


def log_json(path, content):
    with path.open('w') as stream:
        json.dump(content, stream, indent=2, allow_nan=False)
        stream.write('\n')


def train_arm(architecture, objective, unroll, pairwise, train, validation, calibration, class_weight, args):
    seed = args.seed
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(architecture).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    blocks = training_blocks([label_view(split, objective, calibration) for split in train], args.block, args.burn)
    arm = architecture + '_' + objective
    arm += '_w{}'.format(unroll) if model.recurrent else ''
    arm += '_pairwise' if pairwise else ''
    if args.state_dropout:
        arm += '_drop{}'.format(round(100 * args.state_dropout))
    best_key, best_state, best_epoch, thresholds = None, None, 0, {}
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        bce, pair = train_epoch(model, blocks, unroll, optimizer, class_weight, args, rng, pairwise)
        key, each, scores = validation_summary(model, validation, args.budgets, calibration, args.device)
        history.append({'epoch': epoch, 'bce': bce, 'pairwise': pair, 'dev': each,
                        'validation_key': key, 'seconds': time.monotonic() - started})
        log_line(args.output / 'training.log', '{} epoch {} BCE {:.6f} pairwise {:.6f} dev {} ({:.1f}s)'.format(
            arm, epoch, bce, pair, [[round(value, 2) for value in row] for row in each], history[-1]['seconds']))
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            thresholds = pooled_thresholds(scores, args.budgets)
        if epoch >= args.min_epochs and epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    width = validation[0].width
    export(model.cpu(), width, args.output / (arm + '.txt'))
    checkpoint = {'architecture': architecture, 'objective': objective, 'pairwise': pairwise,
                  'unroll': unroll if model.recurrent else 0, 'epoch': best_epoch,
                  'thresholds': thresholds, 'candidate_cap': width, 'state_dropout': args.state_dropout}
    save_checkpoint(model, args.output / (arm + '.pt'), **checkpoint)
    return {'arm': arm, 'architecture': architecture, 'objective': objective, 'pairwise': pairwise,
            'unroll': checkpoint['unroll'], 'epoch': best_epoch, 'thresholds': thresholds,
            'state_dropout': args.state_dropout,
            'complexity': complexity(model, width), 'history': history,
            'validation_key': best_key}


def evaluate_checkpoints(directory, splits, arms, budgets, calibration, args):
    for arm in arms:
        stem = arm['arm']
        model, checkpoint = load_checkpoint(directory / (stem + '.pt'), args.device)
        for split in splits:
            if split.width != checkpoint['candidate_cap']:
                raise ValueError('candidate cap changed for {}'.format(split.name))
            scores = predict(model, split, args.device)
            rows = summarize(split, scores, budgets, checkpoint['thresholds'], calibration)
            for row in rows:
                yield dict(arm=stem, architecture=checkpoint['architecture'], objective=checkpoint['objective'],
                           pairwise=checkpoint['pairwise'], unroll=checkpoint['unroll'], epoch=checkpoint['epoch'],
                           state_dropout=checkpoint.get('state_dropout', 0.),
                           split=split.name, source=split.metadata['source'], **row)
            log_line(args.output / 'evaluation.log', '{} {} {}'.format(
                stem, split.name, [(row['timed_unique'], row['future_unique'], row['calibrated_expected_unique'])
                                   for row in rows]))


@torch.no_grad()
def ablated_scores(model, split, device, width, seed):
    result = np.zeros(split.targets.shape, dtype=np.float32)
    rng = np.random.default_rng(seed)
    valid = split.valid.numpy().any(axis=1)
    for first, last in split.streams:
        rows = np.flatnonzero(valid[first:last]) + first
        if rows.size and rows[0] - first + 1 < width:
            raise ValueError('ablation window exceeds available event prefix')
        for start in range(0, rows.size, 2048):
            anchors = rows[start:start + 2048]
            current = split.events[anchors].to(device)
            candidates = split.candidates[anchors].to(device)
            if width:
                positions = anchors[:, None] - np.arange(width - 1, -1, -1)
                if seed is not None:
                    # Keep the trigger and immediately previous observation fixed.
                    order = rng.random((anchors.size, width - 2)).argsort(axis=1)
                    positions[:, :-2] = np.take_along_axis(positions[:, :-2], order, axis=1)
                recent = split.events[positions].to(device)
                states, _ = model.cell(recent)
                hidden = states[:, -1]
            else:
                hidden = current.new_zeros((len(anchors), HIDDEN))
            result[anchors] = model.score_events(current, candidates, hidden).cpu().numpy()
    return result


def evaluate_state(directory, split, stem, budgets, calibration, args):
    model, checkpoint = load_checkpoint(directory / (stem + '.pt'), args.device)
    if not model.recurrent:
        raise ValueError('state ablation requires a recurrent checkpoint')
    carried = predict(model, split, args.device)
    rows, cols, _, _, _ = ranked(split, carried, True)
    winners = np.full(split.rows, -1, dtype=np.int16)
    winners[rows] = cols
    conditions = [('carried', None, None), ('zero_h', 0, None)]
    conditions += [('ordered', width, None) for width in args.state_windows]
    conditions += [('shuffled', width, seed) for width in args.state_windows for seed in args.state_seeds]
    for mode, width, seed in conditions:
        scores = carried if width is None else ablated_scores(model, split, args.device, width, seed)
        rows, cols, _, _, _ = ranked(split, scores, True)
        changed = int(np.count_nonzero(cols != winners[rows]))
        summary = summarize(split, scores, budgets, checkpoint['thresholds'], calibration)
        for row in summary:
            yield dict(arm=stem, split=split.name, state=mode, window=width, shuffle_seed=seed,
                       changed_winners=changed, **row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', nargs='+', type=Path, default=[])
    parser.add_argument('--validation', nargs='+', type=Path, default=[])
    parser.add_argument('--test', nargs='+', type=Path, default=[])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--frozen', type=Path, help='read checkpoints from a completed development run')
    parser.add_argument('--ablate-arms', nargs='+', default=[], help='apply reset/shuffle to these frozen recurrent arms')
    parser.add_argument('--state-windows', nargs='+', type=int, default=[8, 16, 32])
    parser.add_argument('--state-seeds', nargs='+', type=int, default=[605, 606, 607])
    parser.add_argument('--calibration', type=Path, help='NoPF demand-service cycles plus modeled prediction delay')
    supported = DEFAULT_ARMS + ('event_mlp4', 'event_mlp16')
    parser.add_argument('--arms', nargs='+', choices=supported, default=list(DEFAULT_ARMS))
    parser.add_argument('--pairwise-arms', nargs='+', choices=supported, default=[])
    parser.add_argument('--unroll32-arms', nargs='+', choices=[arm for arm in supported if arm.startswith('lstm')],
                        default=[])
    parser.add_argument('--controls-only', action='store_true', help='add controls for baselines already trained')
    parser.add_argument('--objective', default='soft', choices=OBJECTIVES)
    parser.add_argument('--budgets', nargs='+', type=int, default=[2000, 4000])
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--min-epochs', type=int, default=4)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--unroll', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--block', type=int, default=256)
    parser.add_argument('--burn', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=605)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--pairwise-weight', type=float, default=.25)
    parser.add_argument('--state-dropout', type=float, default=0.,
                        help='training probability of dropping the entire hidden vector from the scorer')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    paths = args.train + args.validation + args.test
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error('training, validation and test paths must be distinct')
    if (min(args.budgets + [args.epochs, args.min_epochs, args.patience, args.unroll, args.batch_size,
                            args.block, args.threads]) < 1 or args.burn < 0 or args.min_epochs > args.epochs
            or not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.pairwise_weight)
            or args.pairwise_weight <= 0 or len(set(args.arms)) != len(args.arms)
            or min(args.state_windows) < 4 or not 0 <= args.state_dropout < 1):
        parser.error('invalid training settings')
    if args.state_dropout and any(not arm.startswith('lstm') for arm in args.arms):
        parser.error('hidden-vector dropout requires recurrent scorers')
    if args.ablate_arms and not args.frozen:
        parser.error('state ablations use frozen models')
    if args.frozen:
        if not args.test or args.train or args.validation or args.calibration:
            parser.error('frozen evaluation needs only new test splits and saved calibration')
    elif not args.train or not args.validation or (args.objective == 'soft' and not args.calibration):
        parser.error('training needs development splits, and soft objectives need NoPF calibration')
    elif (set(args.pairwise_arms + args.unroll32_arms) - set(args.arms)
          or (args.pairwise_arms and args.objective != 'soft') or (args.unroll32_arms and args.unroll != 8)
          or (args.controls_only and not (args.pairwise_arms or args.unroll32_arms))):
        parser.error('pairwise and unroll controls need the corresponding baseline arms')

    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    args.budgets = sorted(set(args.budgets))
    if args.frozen:
        directory = args.frozen
        frozen = json.loads((directory / 'frozen.json').read_text())
        calibration = nopf_calibration(directory / 'service_calibration.json') if frozen['calibration_samples'] else None
        evaluation = read_splits(args.test)
        if any(split.metadata['source'] in set(frozen['training'] + frozen['validation']) for split in evaluation):
            parser.error('held-out data cannot reuse training or validation sources')
        for split in evaluation:
            if any(split.metadata.get(key) != value for key, value in frozen['protocol'].items()):
                parser.error('test split {} uses a different candidate protocol'.format(split.name))
    else:
        directory = args.output
        train, validation = read_splits(args.train), read_splits(args.validation)
        if ({s.metadata['source'] for s in train} != TRAIN_SOURCES or len(train) != len(TRAIN_SOURCES)
                or {s.metadata['source'] for s in validation} != VALIDATION_SOURCES
                or len(validation) != len(VALIDATION_SOURCES)):
            parser.error('use 605 NoPF 10M/30M/50M for training and 70M/130M for validation')
        check_protocol(train + validation, validation[0])
        calibration = nopf_calibration(args.calibration)
        if calibration is not None:
            shutil.copyfile(args.calibration, directory / 'service_calibration.json')
        candidates = sum(int(split.valid.sum()) for split in train)
        positives = sum(int(split.labels.sum()) for split in train)
        if not 0 < positives < candidates:
            parser.error('training requires positive and negative timed candidates')
        class_weight = min(32., math.sqrt((candidates - positives) / positives))
        class_weight_tensor = torch.tensor(class_weight, device=args.device)
        schedule = [] if args.controls_only else [(arm, args.unroll, False) for arm in args.arms]
        schedule += [(arm, args.unroll, True) for arm in args.pairwise_arms]
        schedule += [(arm, 32, False) for arm in args.unroll32_arms]
        arms = [train_arm(architecture, args.objective, unroll, pairwise, train, validation,
                          calibration, class_weight_tensor, args) for architecture, unroll, pairwise in schedule]
        del train
        frozen = {'training': sorted(TRAIN_SOURCES), 'validation': sorted(VALIDATION_SOURCES),
                  'calibration_samples': None if calibration is None else int(calibration.size),
                  'budgets': args.budgets, 'protocol': {key: validation[0].metadata.get(key)
                                                       for key in PROTOCOL + ('correlation_table',)},
                  'args': describe(args), 'positive_weight': class_weight,
                  'best_arm': max(arms, key=lambda arm: arm['validation_key'])['arm'], 'arms': arms}
        log_json(directory / 'frozen.json', frozen)
        # Only open optional held-out splits after all weights and cutoffs are frozen.
        evaluation = read_splits(args.test)
        for split in evaluation:
            if split.metadata['source'] in TRAIN_SOURCES | VALIDATION_SOURCES:
                parser.error('held-out data cannot reuse training or validation sources')
        check_protocol(evaluation, validation[0])
        evaluation = validation + evaluation
    if args.ablate_arms:
        known = {arm['arm'] for arm in frozen['arms']}
        if set(args.ablate_arms) - known:
            parser.error('state ablations require checkpoint names listed in frozen.json')
    records = list(evaluate_checkpoints(directory, evaluation, frozen['arms'], frozen['budgets'], calibration, args))
    write_metrics(args.output / 'metrics.csv', records)
    if args.ablate_arms:
        state_records = [row for stem in args.ablate_arms for split in evaluation
                         for row in evaluate_state(directory, split, stem, frozen['budgets'], calibration, args)]
        write_metrics(args.output / 'state_ablation.csv', state_records)


if __name__ == '__main__':
    main()
