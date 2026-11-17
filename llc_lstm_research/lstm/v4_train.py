#!/usr/bin/env python3
"""Train a scorer to resolve opposite directions from the same NoPF source line."""

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from train import predict, q10, ranked, training_blocks
from v2_train import PROTOCOL, calibrated_hits, label_view, summarize, unique_count, weighted_batch, write_metrics
from v3_model import complexity, export, load_checkpoint, save_checkpoint
from v3_train import (TRAIN_SOURCES, VALIDATION_SOURCES, ablated_scores, check_protocol, describe,
                      evaluate_checkpoints, log_json, log_line, make_model,
                      nopf_calibration, pooled_thresholds, read_splits,
                      update_state, validation_summary)


MATCHED_TRAIN_SOURCES = {'rob_10m', 'rob_30m', 'rob_50m'}
ROTATION_VALIDATION_SOURCES = {'dev_70m', '605_nopf_130m'}
MATCHED_PROTOCOL = PROTOCOL + ('correlation_table', 'candidate_cap', 'proposal_cooldown_cycles',
                               'proposal_buffer_entries')
GATES = ('carry', 'zero', 'opposite', 'without_opposite', 'any_opposite', 'pc_change')


def demand_overlap(split, log_root):
    directory = log_root / split.metadata['source']
    summary = json.loads((directory / 'summary.json').read_text())
    policy = next(row for row in summary if row['policy'] == 'nopf')
    if policy['prefetch_issued']:
        raise ValueError('overlap weights require NoPF data')
    path = (directory / policy['log']).with_name('demand.csv')
    occupancy = np.full(int(split.event_id.max()) + 1, -1, dtype=np.int16)
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if row['prefetch_use'] != 'none':
                raise ValueError('prefetch outcome in NoPF overlap source')
            occupancy[int(row['event_id'])] = int(row['mshr_occupancy'])
    if not np.all(occupancy[split.event_id] >= 0):
        raise ValueError('NoPF overlap source does not cover the exported events')
    return occupancy


def rob_waits(split, path):
    logged = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            identity = tuple(int(row[key]) for key in ('cpu', 'instr_id', 'operand'))
            wait = int(row['head_wait_cycles'])
            if identity in logged or wait < 0:
                raise ValueError('invalid ROB wait entry in {}'.format(path))
            logged[identity] = wait
    with np.load(split.path, allow_pickle=False) as archive:
        identities = zip(split.cores.tolist(), archive['event_instr'].tolist(), archive['event_operand'].tolist())
        waits = np.asarray([logged.get(identity, 0) for identity in identities], dtype=np.uint32)
    indexed = np.zeros(int(split.event_id.max()) + 1, dtype=np.uint32)
    indexed[split.event_id] = waits
    return indexed, len(logged)


def pair_data(split, calibration, target_weight, min_gap):
    with np.load(split.path, allow_pickle=False) as archive:
        delta = archive['offset']
        source = archive['lines'].astype(np.int64) - delta
    valid = split.valid.numpy()
    future = split.future_targets > 0
    utility = np.zeros(valid.shape, dtype=np.float32)
    utility[future] = (np.searchsorted(calibration, split.future_lead[future], side='right')
                       / calibration.size)
    if target_weight is not None:
        utility[future] *= target_weight[split.future_targets[future]]
    best = np.zeros(split.rows, dtype=np.int64)
    hard = np.zeros(valid.shape, dtype=np.bool_)
    row_index = np.arange(split.rows)
    has_pair = np.zeros(split.rows, dtype=np.bool_)
    for start in range(0, split.rows, 8192):
        stop = min(start + 8192, split.rows)
        section = slice(start, stop)
        origin, direction, legal, value = source[section], np.sign(delta[section]), valid[section], utility[section]
        partners = ((origin[:, :, None] == origin[:, None, :])
                    & (direction[:, :, None] * direction[:, None, :] == -1)
                    & legal[:, :, None] & legal[:, None, :]
                    & (value[:, :, None] - value[:, None, :] >= min_gap))
        eligible = partners.any(2)
        choice = np.where(eligible, value, -1).argmax(1)
        has_pair[section] = eligible.any(1)
        best[section] = choice
        hard[section] = partners[row_index[:stop - start], choice]
    return torch.from_numpy(utility), torch.from_numpy(best), torch.from_numpy(hard), has_pair


def target_mass_weights(split, labels, calibration, preference):
    positive = labels > 0
    targets = split.future_targets[positive]
    if targets.size == 0:
        raise ValueError('target mass needs positive NoPF labels')
    _, groups, repeats = np.unique(targets, return_inverse=True, return_counts=True)
    priority = np.ones(groups.size, dtype=np.float64)
    if preference == 'earliest':
        lead = split.future_lead[positive].astype(np.int32)
        reliable = int(np.quantile(calibration, .90, method='higher'))
        feasible = (lead >= reliable) & (lead <= 2000)
        distance = np.where(feasible, -lead, np.abs(lead - reliable))
        order = np.lexsort((np.arange(lead.size), distance, ~feasible, groups))
        first = np.r_[True, groups[order][1:] != groups[order][:-1]]
        selected = order[first]
        # Allocate 75% to the preferred opportunity and leave 25% across alternatives.
        priority[selected] += 3 * repeats[groups[selected]]
    mass = np.bincount(groups, weights=priority * labels[positive])
    mean_mass = labels[positive].sum(dtype=np.float64) / len(repeats)
    weights = np.ones(labels.shape, dtype=np.float32)
    weights[positive] = mean_mass * priority / mass[groups]
    if not np.allclose(np.bincount(groups, weights=labels[positive] * weights[positive]), mean_mass):
        raise ValueError('per-target positive mass was not conserved')
    return torch.from_numpy(weights), len(repeats), int(np.count_nonzero(repeats > 1))


def training_view(split, args, calibration, rob_weight):
    view = label_view(split, 'soft' if args.bce == 'rob_soft' else args.bce, calibration)
    if args.bce == 'rob_soft':
        future = split.future_targets > 0
        view.labels.numpy()[future] *= rob_weight[split.future_targets[future]]
    if args.target_mass != 'none':
        view.sample_weights, targets, duplicates = target_mass_weights(
            split, view.labels.numpy(), calibration, args.target_mass,
        )
        log_line(args.output / 'training.log', '{}: {} target IDs, {} repeat under {} mass allocation'.format(
            split.name, targets, duplicates, args.target_mass,
        ))
    target_weight = .5 + .5 * rob_weight if rob_weight is not None else None
    if args.criticality:
        occupancy = demand_overlap(split, args.nopf_log_root)
        target_weight = 1 / np.sqrt(1 + np.maximum(0, occupancy))
    view.utility, view.reference, view.rivals, paired = pair_data(
        split, calibration, target_weight, args.min_gap,
    )
    log_line(args.output / 'training.log', '{} opposite-direction groups: {} / {} legal triggers'.format(
        split.name, int(paired.sum()), int(split.valid.any(-1).sum()),
    ))
    return view


def rob_weights(splits, logs, output):
    observed = []
    delays = []
    for split, path in zip(splits, logs):
        waits, log_rows = rob_waits(split, path)
        targets = np.unique(split.future_targets[split.future_targets > 0])
        positive = waits[targets][waits[targets] > 0]
        log_line(output / 'training.log', '{} ROB wait positive {} of {} targets, {} log rows'.format(
            split.name, positive.size, targets.size, log_rows,
        ))
        observed.append(waits)
        delays.append(positive)
    if not any(delay.size for delay in delays):
        raise ValueError('no observed training targets had ROB head stall cycles')
    scale = float(np.quantile(np.concatenate(delays), .90))
    log_line(output / 'training.log', 'NoPF positive ROB wait p90 {:.2f} cycles'.format(scale))
    return [np.minimum(waits.astype(np.float32) / scale, 1.) for waits in observed], scale


def pair_batch(blocks, valid, device):
    utility = torch.zeros(valid.shape, dtype=torch.float32)
    reference = torch.zeros(valid.shape[:-1], dtype=torch.int64)
    rivals = torch.zeros(valid.shape, dtype=torch.bool)
    for row, (split, begin, start, stop) in enumerate(blocks):
        left, right = start - begin, stop - begin
        utility[row, left:right] = split.utility[start:stop]
        reference[row, left:right] = split.reference[start:stop]
        rivals[row, left:right] = split.rivals[start:stop]
    return tuple(tensor.to(device) for tensor in (utility, reference, rivals))


def opposite_loss(scores, utility, reference, rivals, temperature):
    present = rivals.any(-1)
    if not bool(present.any()):
        return scores.new_zeros(())
    opportunities = scores[present]
    best = reference[present].unsqueeze(1)
    high = opportunities.gather(1, best).squeeze(1)
    low, alternative = opportunities.masked_fill(~rivals[present], -torch.inf).max(1)
    selected = utility[present]
    gap = (selected.gather(1, best) - selected.gather(1, alternative[:, None])).squeeze(1)
    return (gap * F.softplus((low - high) / temperature)).sum()


def train_epoch(model, blocks, optimizer, weight, args, rng):
    model.train()
    bce_sum = rank_sum = 0.0
    valid_total = pair_total = 0
    indices = rng.permutation(len(blocks))
    for start in range(0, len(indices), args.batch_size):
        group = [blocks[index] for index in indices[start:start + args.batch_size]]
        events, candidates, labels, valid, positive_weights = weighted_batch(group, args.device)
        utility, reference, rivals = pair_batch(group, valid, args.device)
        count = int(valid.sum())
        if not count:
            continue
        pair_count = int(rivals.any(-1).sum())
        state = None
        step = args.unroll if model.recurrent else events.shape[1]
        optimizer.zero_grad(set_to_none=True)
        for begin in range(0, events.shape[1], step):
            section = slice(begin, begin + step)
            x, z, mask = events[:, section], candidates[:, section], valid[:, section]
            if not bool(mask.any()):
                state = update_state(model, x, z, state)
                continue
            scores, state = model(x, z, state)
            positive = weight if args.target_mass == 'none' else weight * positive_weights[:, section][mask]
            bce = F.binary_cross_entropy_with_logits(scores[mask], labels[:, section][mask],
                                                     pos_weight=positive, reduction='sum')
            loss = bce / count
            if pair_count and args.pairwise_weight:
                rank = opposite_loss(scores, utility[:, section], reference[:, section], rivals[:, section],
                                     args.pair_temperature)
                loss = loss + args.pairwise_weight * rank / pair_count
                rank_sum += float(rank.detach())
            loss.backward()
            bce_sum += float(bce.detach())
            if state is not None:
                state = tuple(tensor.detach() for tensor in state)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        valid_total += count
        pair_total += pair_count
    return bce_sum / valid_total, rank_sum / max(1, pair_total)


def fit_model(architecture, blocks, dev, calibration, weight, args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = make_model(architecture).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    name = '{}_{}_{}'.format(architecture, args.bce, 'opposite' if args.pairwise_weight else 'pointwise')
    name += '_overlap' if args.criticality else ''
    name += '_rob' if args.rob_stalls and args.bce != 'rob_soft' else ''
    name += '_' + args.target_mass if args.target_mass != 'none' else ''
    max_key, best_epoch, state, cutoffs = None, 0, None, {}
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        bce, pair = train_epoch(model, blocks, optimizer, weight, args, rng)
        key, scores, logits = validation_summary(model, dev, args.budgets, calibration, args.device)
        seconds = time.monotonic() - started
        history.append({'epoch': epoch, 'bce': bce, 'pairwise': pair, 'dev': scores,
                        'validation_key': key, 'seconds': seconds})
        log_line(args.output / 'training.log', '{} epoch {} BCE {:.6f} pair {:.6f} dev {} ({:.1f}s)'.format(
            name, epoch, bce, pair, [[round(x, 2) for x in row] for row in scores], seconds,
        ))
        if max_key is None or key > max_key:
            max_key, best_epoch = key, epoch
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            cutoffs = pooled_thresholds(logits, args.budgets)
        if epoch >= args.min_epochs and epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(state)
    cap = dev[0].width
    export(model.cpu(), cap, args.output / (name + '.txt'))
    info = {'architecture': architecture, 'objective': args.bce, 'pairwise': bool(args.pairwise_weight),
            'pairwise_weight': args.pairwise_weight, 'criticality': args.criticality,
            'rob_stalls': bool(args.rob_stalls), 'pair_temperature': args.pair_temperature,
            'target_mass': args.target_mass,
            'unroll': args.unroll if model.recurrent else 0, 'epoch': best_epoch,
            'thresholds': cutoffs, 'candidate_cap': cap}
    save_checkpoint(model, args.output / (name + '.pt'), **info)
    return dict(arm=name, complexity=complexity(model, cap), history=history, validation_key=max_key, **info)


def direction_metrics(directory, arms, dev, calibration, args):
    for split in dev:
        utility, reference, rivals, present = pair_data(split, calibration, None, args.min_gap)
        refs = reference.numpy()[present]
        negatives = rivals.numpy()[present]
        values = utility.numpy()[present]
        rows = np.arange(len(refs))
        for arm in arms:
            stem = arm['arm']
            model, _ = load_checkpoint(directory / (stem + '.pt'), args.device)
            scores = predict(model, split, args.device)[present]
            hard = np.where(negatives, scores, -np.inf).argmax(1)
            delta = values[rows, refs] - values[rows, hard]
            wins = scores[rows, refs] > scores[rows, hard]
            yield {'arm': stem, 'split': split.name, 'paired_anchors': len(refs),
                   'correct_directions': int(wins.sum()), 'accuracy': float(wins.mean()),
                   'utility_weighted_accuracy': float(delta[wins].sum() / delta.sum())}


def gate_inputs(split):
    with np.load(split.path, allow_pickle=False) as archive:
        delta = archive['offset']
        line = archive['lines']
    valid = split.valid.numpy()
    if delta.shape != valid.shape or line.shape != valid.shape or np.any(valid & ((delta == 0) | (line == 0))):
        raise ValueError('missing address or slope for a legal candidate in {}'.format(split.name))
    source = line.astype(np.int64) - delta
    forward, backward = valid & (delta > 0), valid & (delta < 0)
    opposite = np.empty(split.rows, dtype=np.bool_)
    for start in range(0, split.rows, 8192):
        section = slice(start, start + 8192)
        origin = source[section]
        opposite[section] = ((origin[:, :, None] == origin[:, None, :])
                             & forward[section, :, None] & backward[section, None, :]).any(axis=(1, 2))
    return {'opposite': opposite, 'without_opposite': ~opposite,
            'any_opposite': forward.any(1) & backward.any(1),
            'pc_change': split.events[:, 5].numpy() == 0}


def gated_outputs(model, split, gates, modes, device):
    full = predict(model, split, device)
    zero = ablated_scores(model, split, device, 0, None) if any(mode != 'carry' for mode in modes) else None
    for mode in modes:
        if mode == 'carry':
            scores = full
        elif mode == 'zero':
            scores = zero
        else:
            scores = np.where(gates[mode][:, None], full, zero)
        if not np.isfinite(scores[split.valid.numpy()]).all():
            raise ValueError('missing scores for a legal candidate')
        yield mode, scores


def gate_counts(split, gates, mode):
    eligible = split.valid.numpy().any(1)
    eligible_rows = int(eligible.sum())
    if mode == 'carry':
        active = eligible_rows
    elif mode == 'zero':
        active = 0
    else:
        active = int(np.count_nonzero(eligible & gates[mode]))
    return {'eligible_triggers': eligible_rows,
            'opposite_triggers': int(np.count_nonzero(eligible & gates['opposite'])),
            'hidden_enabled_triggers': active, 'missing_valid_candidates': 0}


def read_gate_models(directory, stems, device):
    result = {}
    for stem in stems:
        model, _ = load_checkpoint(directory / (stem + '.pt'), device)
        if not model.recurrent:
            raise ValueError('selective state requires a recurrent checkpoint')
        result[stem] = model
    return result


def frozen_development(args):
    directory = args.gate_frozen
    frozen = json.loads((directory / 'frozen.json').read_text())
    dev = read_splits(args.validation)
    if {split.metadata['source'] for split in dev} != set(frozen['validation']):
        raise ValueError('gate cutoffs must use the original validation regions')
    if any(split.metadata.get(key) != value for split in dev for key, value in frozen['protocol'].items()):
        raise ValueError('gate validation data uses a different candidate protocol')
    calibration = nopf_calibration(directory / 'service_calibration.json')
    return directory, frozen, dev, calibration


def calibrate_gates(args):
    directory, frozen, dev, calibration = frozen_development(args)
    known = {arm['arm']: arm for arm in frozen['arms'] if arm['architecture'].startswith('lstm')}
    stems = args.gate_arms or sorted(known)
    if not stems or set(stems) - known.keys():
        raise ValueError('gate arms must be recurrent checkpoints listed in frozen.json')
    models = read_gate_models(directory, stems, args.device)
    budgets = frozen['budgets']
    thresholds, records, calibration_scores = {}, [], {}
    for stem, model in models.items():
        pooled = {mode: [] for mode in args.gate_modes}
        for split in dev:
            gates = gate_inputs(split)
            for mode, scores in gated_outputs(model, split, gates, args.gate_modes, args.device):
                _, _, logits, _, _ = ranked(split, scores, True)
                if len(logits) < max(budgets):
                    raise ValueError('insufficient decisions for gate calibration in {}'.format(split.name))
                pooled[mode].append(logits)
                calibration_scores[stem, split.name, mode] = scores, gate_counts(split, gates, mode)
        for mode, wins in pooled.items():
            thresholds[stem, mode] = pooled_thresholds(wins, budgets)
    for (stem, region, mode), (scores, counts) in calibration_scores.items():
        split = next(split for split in dev if split.name == region)
        results = summarize(split, scores, budgets, thresholds[stem, mode], calibration)
        records.extend(dict(arm=stem, split=region, gate=mode, role='validation', **counts, **row) for row in results)

    arms = [{'arm': stem, 'gate': mode, 'cutoffs': thresholds[stem, mode],
             'portable_model': str(directory / (stem + '.txt'))} for stem, mode in thresholds]
    args.output.mkdir(parents=True, exist_ok=False)
    log_json(args.output / 'gate_frozen.json', {'models': str(directory), 'validation': frozen['validation'],
                                               'arms': arms, 'budgets': budgets, 'max_issues_per_trigger': 1})
    # Open held-out data only after all operating points have been saved.
    for split in read_splits(args.test):
        excluded = (set(frozen['training'] + frozen['validation']) | TRAIN_SOURCES | MATCHED_TRAIN_SOURCES
                    | VALIDATION_SOURCES | ROTATION_VALIDATION_SOURCES)
        if split.metadata['source'] in excluded:
            raise ValueError('held-out data overlaps gate development regions')
        if any(split.metadata.get(key) != value for key, value in frozen['protocol'].items()):
            raise ValueError('held-out gate data uses a different candidate protocol')
        gates = gate_inputs(split)
        for stem, model in models.items():
            for mode, scores in gated_outputs(model, split, gates, args.gate_modes, args.device):
                results = summarize(split, scores, budgets, thresholds[stem, mode], calibration)
                counts = gate_counts(split, gates, mode)
                records.extend(dict(arm=stem, split=split.name, gate=mode, role='test', **counts, **row)
                               for row in results)
    write_metrics(args.output / 'gate_metrics.csv', records)
    for row in records:
        if row['budget'] == max(budgets):
            log_line(args.output / 'gate.log', '{} {} {}: {:.2f} expected unique, {:,}/{:,} h on'.format(
                row['arm'], row['split'], row['gate'], row['calibrated_expected_unique'],
                row['hidden_enabled_triggers'], row['eligible_triggers'],
            ))


def approximate_epochs(split, eligible, window):
    with np.load(split.path, allow_pickle=False) as archive:
        instructions = archive['event_instr']
    progress = np.empty(split.rows, dtype=np.int64)
    for first, last in split.streams:
        # The largest instruction ID seen so far stays causal when older loads
        # enter LLC behind their younger peers. Retired position is approximate.
        progress[first:last] = np.maximum(0, np.maximum.accumulate(instructions[first:last]) - 200000)
    step = progress[eligible]
    _, epochs = np.unique(np.stack((split.cores[eligible], step // window), axis=1), axis=0, return_inverse=True)
    return epochs.reshape(-1), step % window < window // 2


def take_per_epoch(eligible, epochs, limit):
    grouped = np.argsort(epochs[eligible], kind='stable')
    sorted_groups = epochs[eligible][grouped]
    starts = np.r_[True, sorted_groups[1:] != sorted_groups[:-1]] if grouped.size else np.zeros(0, dtype=np.bool_)
    ranks = np.arange(grouped.size)
    keep = ranks - np.maximum.accumulate(np.where(starts, ranks, 0)) < limit
    return eligible[np.sort(grouped[keep])]


def exact_epoch_offers(scores, early, epochs, tie, cap, expected, penalty):
    adjusted = scores - penalty * early
    priority = np.lexsort((tie, -adjusted))
    # Count the first cap passing triggers per epoch for every threshold prefix.
    enabled = take_per_epoch(priority, epochs, cap)
    enters = np.isin(priority, enabled, assume_unique=True)
    prefix_sends = np.cumsum(enters)
    if expected < 1 or expected > prefix_sends[-1]:
        raise ValueError('requested exact decision count is outside the pacing capacity')
    boundary = int(np.searchsorted(prefix_sends, expected, side='left'))
    passes = priority[:boundary + 1]
    accepted = take_per_epoch(np.sort(passes), epochs, cap)
    if len(accepted) != expected or np.any(np.diff(prefix_sends) < 0):
        raise ValueError('pacing count is not exact and monotone')
    cutoff = int(adjusted[priority[boundary]])
    ties = adjusted == cutoff
    return accepted, {'early_cutoff_q10': cutoff + penalty, 'late_cutoff_q10': cutoff,
                      'offers': len(passes), 'accepted': len(accepted), 'early_accepted': int(early[accepted].sum()),
                      'equal_cutoff_offered': int(ties[passes].sum()), 'equal_cutoff_total': int(ties.sum()),
                      'max_accepted': int(prefix_sends[-1]), 'monotone_cutoffs': True}


def chronology(args):
    directory, frozen, dev, calibration = frozen_development(args)
    if args.test:
        raise ValueError('the chronology approximation uses development only')
    checkpoints = {arm['arm']: arm for arm in frozen['arms']}
    stems = args.gate_arms or sorted(checkpoints)
    if not stems or set(stems) - checkpoints.keys():
        raise ValueError('chronology arms must be checkpoints listed in frozen.json')
    high_budget = str(max(frozen['budgets']))
    cap = args.pace_requests
    records = []
    for split in dev:
        valid = split.valid.numpy()
        eligible = np.flatnonzero(valid.any(1))
        gates = gate_inputs(split)
        epochs, early = approximate_epochs(split, eligible, args.pace_instructions)
        tie = split.event_id[eligible].astype(np.uint64) * np.uint64(11400714819323198485)
        scored = {}
        for stem in stems:
            model, _ = load_checkpoint(directory / (stem + '.pt'), args.device)
            modes = args.gate_modes if model.recurrent else ['carry']
            for mode, raw in gated_outputs(model, split, gates, modes, args.device):
                actual = np.where(valid[eligible], q10(raw[eligible]), -32769)
                winners = actual.argmax(1)
                chosen = actual[np.arange(len(eligible)), winners]
                scored[stem, mode] = chosen, winners
        if args.chronology_decisions:
            budget = args.chronology_decisions
        else:
            allowances = []
            for (stem, mode), (scores, _) in scored.items():
                if mode == 'carry':
                    threshold = checkpoints[stem]['thresholds'][high_budget]['cutoff_q10']
                    above = np.flatnonzero(scores >= threshold)
                    allowances.append(len(take_per_epoch(above, epochs, cap)))
            if not allowances:
                raise ValueError('include carry to derive the common chronology count')
            budget = min(allowances)
        for (stem, mode), (scores, winners) in scored.items():
            quartiles = np.quantile(scores, (.25, .75))
            iqr = int(round(quartiles[1] - quartiles[0]))
            for scale in args.chronology_iqr:
                chosen, stats = exact_epoch_offers(scores, early, epochs, tie, cap, budget, int(round(iqr * scale)))
                rows, cols = eligible[chosen], winners[chosen]
                future = split.future_targets[rows, cols]
                leads = split.future_lead[rows, cols]
                records.append(dict(arm=stem, split=split.name, gate=mode, early_penalty_iqr=scale,
                                    epoch_instructions=args.pace_instructions, issues_per_epoch=cap,
                                    eligible_anchors=len(eligible), q10_iqr=iqr,
                                    timed_unique=unique_count(split.targets[rows, cols]), future_unique=unique_count(future),
                                    calibrated_expected_unique=calibrated_hits(future, leads, calibration), **stats))
    args.output.mkdir(parents=True, exist_ok=False)
    write_metrics(args.output / 'chronology.csv', records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', nargs=3, type=Path, default=[])
    parser.add_argument('--validation', nargs=2, type=Path, required=True)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gate-frozen', type=Path, help='calibrate causal top-one gates for frozen recurrent weights')
    parser.add_argument('--gate-arms', nargs='+')
    parser.add_argument('--gate-modes', nargs='+', choices=GATES, default=list(GATES))
    parser.add_argument('--test', nargs='+', type=Path, default=[], help='optional frozen gate evaluation regions')
    parser.add_argument('--chronology', action='store_true', help='test pacing with a causal instruction-ID proxy')
    parser.add_argument('--chronology-decisions', type=int, default=0,
                        help='exact accepted decision count per region, or zero for the common baseline count')
    parser.add_argument('--chronology-iqr', nargs='+', type=float, default=[0., .5, 1.],
                        help='range of first-half cutoff penalties as multiples of the score IQR')
    parser.add_argument('--pace-instructions', type=int, default=20000)
    parser.add_argument('--pace-requests', type=int, default=10)
    parser.add_argument('--architectures', nargs='+', choices=('event_mlp8', 'lstm_mlp8'), default=['lstm_mlp8'])
    parser.add_argument('--bce', choices=('future', 'soft', 'rob_soft'), default='future')
    parser.add_argument('--target-mass', choices=('none', 'equal', 'earliest'), default='none',
                        help='allocate equal total positive weight to each future miss target')
    parser.add_argument('--pairwise-weight', type=float, default=.25)
    parser.add_argument('--pair-temperature', type=float, default=1.,
                        help='scale score differences for the training-only pairwise loss')
    parser.add_argument('--min-gap', type=float, default=.10)
    parser.add_argument('--criticality', action='store_true', help='downweight targets with high NoPF LLC MSHR occupancy')
    parser.add_argument('--nopf-log-root', type=Path)
    parser.add_argument('--rob-stalls', nargs=3, type=Path,
                        help='NoPF ROB head-blocking CSVs for these training splits, in the same order')
    parser.add_argument('--budgets', nargs='+', type=int, default=[2000, 4000])
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--min-epochs', type=int, default=4)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--unroll', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--block', type=int, default=256)
    parser.add_argument('--burn', type=int, default=32)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=605)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if (min(args.budgets + [args.epochs, args.min_epochs, args.patience, args.unroll, args.block,
                            args.batch_size, args.threads]) < 1 or args.burn < 0 or args.min_epochs > args.epochs
            or not 0 < args.min_gap <= 1 or not math.isfinite(args.pairwise_weight) or args.pairwise_weight < 0
            or not math.isfinite(args.pair_temperature) or args.pair_temperature <= 0
            or args.chronology_decisions < 0 or args.pace_instructions < 2 or args.pace_requests < 1
            or any(not math.isfinite(scale) or scale < 0 for scale in args.chronology_iqr)
            or not math.isfinite(args.lr) or args.lr <= 0 or len(set(args.architectures)) != len(args.architectures)):
        parser.error('invalid training settings')
    if args.criticality and args.nopf_log_root is None:
        parser.error('criticality requires --nopf-log-root')
    if args.bce == 'rob_soft' and not args.rob_stalls:
        parser.error('ROB soft BCE needs matched --rob-stalls')
    if args.rob_stalls and (args.criticality or (not args.pairwise_weight and args.bce != 'rob_soft')):
        parser.error('ROB stalls need a pairwise or ROB soft objective and cannot also use MSHR overlap weights')
    args.budgets = sorted(set(args.budgets))
    torch.set_num_threads(args.threads)
    if args.gate_frozen:
        if args.train or args.calibration or args.rob_stalls or args.criticality or len(set(args.gate_modes)) != len(args.gate_modes):
            parser.error('frozen gate calibration uses saved models and NoPF service data')
        if args.chronology:
            chronology(args)
        else:
            calibrate_gates(args)
        return
    if not args.train or not args.calibration or args.test or args.gate_arms or args.chronology:
        parser.error('training needs NoPF development and calibration inputs, and no held-out regions')
    train, dev = read_splits(args.train), read_splits(args.validation)
    train_names = {split.metadata['source'] for split in train}
    dev_names = {split.metadata['source'] for split in dev}
    if train_names not in (TRAIN_SOURCES, MATCHED_TRAIN_SOURCES) or dev_names not in (VALIDATION_SOURCES, ROTATION_VALIDATION_SOURCES):
        parser.error('use 10M/30M/50M NoPF train and 70M/130M validation')
    if args.rob_stalls and train_names != MATCHED_TRAIN_SOURCES:
        parser.error('use matched ROB-instrumented NoPF training NPZ files with --rob-stalls')
    check_protocol(train + dev, dev[0])
    if any(split.metadata.get(key) != dev[0].metadata.get(key) for split in train + dev for key in MATCHED_PROTOCOL):
        parser.error('proposal buffer and cooldown must match within each experiment')
    calibration = nopf_calibration(args.calibration)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.calibration, args.output / 'service_calibration.json')
    weights, rob_scale = (rob_weights(train, args.rob_stalls, args.output) if args.rob_stalls
                          else ([None] * len(train), None))
    views = [training_view(split, args, calibration, weight) for split, weight in zip(train, weights)]
    blocks = training_blocks(views, args.block, args.burn)
    valid = sum(int(split.valid.sum()) for split in train)
    positives = sum(int(split.labels.sum()) for split in train)
    if not 0 < positives < valid:
        parser.error('training requires positive and negative timed examples')
    class_weight = min(32., math.sqrt((valid - positives) / positives))
    weight = torch.tensor(class_weight, device=args.device)
    arms = [fit_model(name, blocks, dev, calibration, weight, args) for name in args.architectures]
    frozen = {'training': sorted(train_names), 'validation': sorted(dev_names),
              'calibration_samples': int(calibration.size), 'budgets': args.budgets,
              'protocol': {key: dev[0].metadata.get(key) for key in MATCHED_PROTOCOL},
              'args': describe(args), 'positive_weight': class_weight, 'rob_scale_p90': rob_scale, 'arms': arms,
              'best_arm': max(arms, key=lambda arm: arm['validation_key'])['arm']}
    log_json(args.output / 'frozen.json', frozen)
    write_metrics(args.output / 'metrics.csv', list(evaluate_checkpoints(args.output, dev, arms, args.budgets,
                                                                         calibration, args)))
    write_metrics(args.output / 'direction_metrics.csv', list(direction_metrics(args.output, arms, dev,
                                                                                calibration, args)))


if __name__ == '__main__':
    main()
