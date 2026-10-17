#!/usr/bin/env python3
"""Publication figures from a complete 71-row audited mechanism matrix only."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

SEEDS = (7, 17, 27)
COLORS = {7: '#2468a2', 17: '#d27d20', 27: '#24885e'}
MARKERS = {7: 'o', 17: 's', 27: '^'}
PLOT_STEMS = ('learned_payload_phase_mechanism', 'current_line_budget_fraction')


def rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def same_float(actual, expected, label):
    if expected is None:
        require(actual in ('', None), label + ': expected undefined ratio')
    else:
        require(actual not in ('', None) and math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-15), label + ': ratio mismatch')


def write_csv(path, data):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def select_primary(metrics, budget, filtered):
    selected = [r for r in metrics if r['arm'] == 'nn' and int(r['macs']) == 0
                and int(r['budget']) == budget and int(r['filter_current']) == filtered
                and int(r['silent']) == 0 and r['payload'] == 'learned']
    require(len(selected) == 6 and {(int(r['roi']), int(r['seed'])) for r in selected}
            == {(roi, seed) for roi in (0, 1) for seed in SEEDS}, 'Expected exactly two ROIs x three independent seeds')
    return sorted(selected, key=lambda r: (int(r['roi']), int(r['seed'])))


def phase_values(analysis, summary, selected):
    ids = {r['run_id']: r for r in selected}
    hist = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    for row in rows(analysis / 'phase_distributions.csv'):
        if row['run_id'] in ids:
            hist[row['run_id']][int(row['phase'])][row['distribution']][int(row['value'])] += int(row['count'])
    full = defaultdict(lambda: defaultdict(Counter))
    for row in rows(analysis / 'full_roi_distributions.csv'):
        if row['run_id'] in ids:
            full[row['run_id']][row['distribution']][int(row['value'])] += int(row['count'])
    # These engine-generated payload classifications are before output filters.
    # In the selected learned, unfiltered arms, payload == raw decoded addresses.
    profile = defaultdict(lambda: defaultdict(Counter))
    for row in rows(analysis / 'request_profiles.csv'):
        if row['run_id'] not in ids or row['profile'] != 'mechanism_request_profile':
            continue
        labels = dict(part.split('_', 1) for part in row['key'].lstrip('/').split('/'))
        phase, slot, current = (int(labels[k]) for k in ('phase', 'slot', 'current'))
        profile[row['run_id']][phase][('slot', slot)] += int(row['value'])
        if current:
            profile[row['run_id']][phase][('current', slot)] += int(row['value'])
    summaries = {r['run_id']: r for r in rows(summary / 'learned_payload_full_roi.csv')}
    require(set(summaries) == set(ids), 'Full-ROI learned payload summary is not the selected six actual runs')
    output = []
    names = ('raw_emit', 'raw_K', 'raw_delta_slot_0', 'raw_delta_slot_1')
    for rid, record in ids.items():
        require(set(hist[rid]) == set(range(5)), rid + ': expected all five disjoint 5M phases')
        totals = defaultdict(Counter)
        for phase in range(5):
            bins = hist[rid][phase]
            emitted = bins['raw_emit'][1]
            completed = sum(bins['raw_emit'].values())
            require(completed > 0 and set(bins['raw_emit']) <= {0, 1}, rid + ': raw emit domain or missing phase')
            require(sum(bins['raw_K'].values()) == completed, rid + ': raw count events differ from completed updates')
            require(bins['raw_K'][2] <= emitted, rid + ': K=2 events exceed emit-positive updates')
            slot0, slot1 = (sum(bins[n].values()) for n in names[2:])
            require(profile[rid][phase][('slot', 0)] == slot0 and profile[rid][phase][('slot', 1)] == slot1,
                    rid + ': raw/payload slot counts disagree')
            current0 = profile[rid][phase][('current', 0)]
            require(0 <= current0 <= slot0, rid + ': slot0 current-line count outside domain')
            output.append(dict(run_id=rid, roi=int(record['roi']), seed=int(record['seed']), phase=phase,
                               phase_start_retired=phase * 5_000_000, phase_nominal_end_retired=(phase + 1) * 5_000_000,
                               completed_updates=completed, raw_emit_positive=emitted,
                               raw_emit_fraction=ratio(emitted, completed), raw_k2= bins['raw_K'][2],
                               k2_fraction_of_emit=ratio(bins['raw_K'][2], emitted),
                               slot0_current=current0, slot0_total=slot0, slot0_current_fraction=ratio(current0, slot0),
                               slot1_delta_plus2=bins['raw_delta_slot_1'][2], slot1_total=slot1,
                               slot1_plus2_fraction=ratio(bins['raw_delta_slot_1'][2], slot1)))
            for name in names:
                totals[name].update(bins[name])
        for name in names:
            require(totals[name] == full[rid][name], rid + ': phase sum differs from full-ROI ' + name)
        aggregate = summaries[rid]
        checks = {'completed': sum(totals['raw_emit'].values()), 'emit_events': totals['raw_emit'][1],
                  'k2_events': totals['raw_K'][2], 'slot0_total': sum(totals['raw_delta_slot_0'].values()),
                  'slot1_plus2': totals['raw_delta_slot_1'][2], 'slot1_total': sum(totals['raw_delta_slot_1'].values())}
        for key, expected in checks.items():
            require(int(aggregate[key]) == expected, rid + ': full-ROI summary mismatch ' + key)
        require(checks['completed'] == int(record['nn_completed']), rid + ': missing completed NN updates')
        same_float(aggregate['emit_fraction'], ratio(checks['emit_events'], checks['completed']), rid + ' emit')
        same_float(aggregate['k2_fraction_of_emit'], ratio(checks['k2_events'], checks['emit_events']), rid + ' K2')
        same_float(aggregate['slot1_plus2_fraction'], ratio(checks['slot1_plus2'], checks['slot1_total']), rid + ' slot1')
    return output


def credit_values(analysis, summary, metrics, cap):
    selected = select_primary(metrics, cap, 0) + select_primary(metrics, cap, 1)
    ids = {r['run_id']: r for r in selected}
    credits = {r['run_id']: r for r in rows(summary / 'current_line_credits.csv') if r['run_id'] in ids}
    require(set(credits) == set(ids), 'Missing capped original/filtered current-line credit rows')
    current = Counter()
    all_allowed = Counter()
    for row in rows(analysis / 'classified_requests.csv'):
        if row['run_id'] in ids:
            all_allowed[row['run_id']] += int(row['allowed'])
            if int(row['current_line']) == 1:
                current[row['run_id']] += int(row['allowed'])
    output = []
    for rid, record in sorted(ids.items(), key=lambda item: (int(item[1]['roi']), int(item[1]['seed']), int(item[1]['filter_current']))):
        row = credits[rid]
        for field in ('roi', 'seed', 'macs', 'budget', 'filter_current', 'silent'):
            require(row[field] == record[field], rid + ': credit summary identity differs')
        allowed, current_allowed = int(row['allowed']), int(row['current_allowed'])
        require(allowed == all_allowed[rid] == int(record['request_gate_allowed']), rid + ': all-allowed denominator differs')
        require(current_allowed == current[rid] and 0 <= current_allowed <= allowed, rid + ': current credit count differs')
        require(allowed <= int(record['instructions']) * cap // 1_000_000, rid + ': request cap exceeded')
        same_float(row['current_credit_fraction'], ratio(current_allowed, allowed), rid + ' current credits')
        if int(record['filter_current']):
            require(current_allowed == 0, rid + ': filtered arm spends current-line credits')
        output.append(dict(run_id=rid, roi=int(row['roi']), seed=int(row['seed']), macs=0, budget=cap,
                           filter_current=int(row['filter_current']), allowed=allowed, current_allowed=current_allowed,
                           current_credit_fraction=ratio(current_allowed, allowed)))
    return output


def plot(out, phase_data, credit_data, cap):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
    fields = [('raw_emit_fraction', 'Raw emit / completed updates'),
              ('k2_fraction_of_emit', 'K = 2 / emit-positive updates'),
              ('slot0_current_fraction', 'Slot 0 current / slot 0 addresses'),
              ('slot1_plus2_fraction', 'Slot 1 delta +2 / slot 1 addresses')]
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 6.3), sharex=True, sharey=True, layout='constrained')
    for roi in (0, 1):
        for col, (field, title) in enumerate(fields):
            ax = axes[roi, col]
            for offset, seed in zip((-.055, 0, .055), SEEDS):
                points = sorted([r for r in phase_data if r['roi'] == roi and r['seed'] == seed], key=lambda r: r['phase'])
                ax.plot([r['phase'] + offset for r in points], [r[field] if r[field] is not None else math.nan for r in points],
                        color=COLORS[seed], marker=MARKERS[seed], markersize=4.5, linewidth=1.15,
                        label='seed ' + str(seed), alpha=.9)
            ax.set_ylim(-.025, 1.055)
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.set_xticks(range(5), ['0–5', '5–10', '10–15', '15–20', '20–25'])
            ax.grid(axis='y', alpha=.2)
            if roi == 0:
                ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel('ROI ' + str(roi) + ' · fraction')
            if roi == 1:
                ax.set_xlabel('Retired ROI instruction interval (M)', fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False)
    fig.suptitle('Frozen learned payload · uncapped, unfiltered, zero-cost NN\n'
                 'Disjoint 5M phases cover each full 25M ROI; the final phase includes retirement overshoot\n'
                 'Per-seed markers are offset slightly for visibility; exact fractions are in the companion CSV', fontsize=11)
    for extension in ('png', 'svg'):
        fig.savefig(out / (PLOT_STEMS[0] + '.' + extension), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.5), sharey=True, layout='constrained')
    policy = [(0, '#2468a2', 'o', -.09, 'Original'), (1, '#d27d20', 's', .09, 'Current-line filtered')]
    for roi, ax in enumerate(axes):
        for filtered, color, marker, offset, label in policy:
            points = [next(r for r in credit_data if r['roi'] == roi and r['seed'] == seed and r['filter_current'] == filtered) for seed in SEEDS]
            ax.scatter([i + offset for i in range(3)], [r['current_credit_fraction'] if r['current_credit_fraction'] is not None else math.nan for r in points],
                       marker=marker, s=48, color=color, label=label, zorder=3)
        ax.set_xticks(range(3), ['seed ' + str(s) for s in SEEDS])
        ax.set_title('602 · ROI ' + str(roi))
        ax.set_ylim(-.025, 1.025)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.axhline(0, color='#697985', lw=.8)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Current-line allowed / all allowed requests')
    axes[1].legend(loc='center right', fontsize=9, frameon=False)
    fig.suptitle('Current-line share of consumed request credits · ceiling ' + str(cap) + '/M\n'
                 'Full 25M ROI, zero-cost NN; each independent seed retained', fontsize=12)
    for extension in ('png', 'svg'):
        fig.savefig(out / (PLOT_STEMS[1] + '.' + extension), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cap', type=int, default=2363)
    args = parser.parse_args()
    audit = json.loads((args.analysis / 'audit.json').read_text())
    require(audit.get('status') == 'PASS' and audit.get('completed') == audit.get('planned_rows') == 71
            and not audit.get('pending') and not audit.get('failures'), 'Figures require complete 71/71 PASS analysis; partial data are refused')
    metrics = rows(args.analysis / 'metrics_per_seed.csv')
    require(len(metrics) == len({r['run_id'] for r in metrics}) == 71, 'Expected 71 unique audited run rows')
    phase_data = phase_values(args.analysis, args.summary, select_primary(metrics, 0, 0))
    credits = credit_values(args.analysis, args.summary, metrics, args.cap)
    expected = [stem + '.' + extension for stem in PLOT_STEMS for extension in ('png', 'svg')]
    expected += ['phase_mechanism_values.csv', 'current_line_budget_values.csv', 'distribution_plot_manifest.json']
    require(not any((args.out / name).exists() for name in expected), 'Preserve prior output: use a fresh figure output path')
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / 'phase_mechanism_values.csv', phase_data)
    write_csv(args.out / 'current_line_budget_values.csv', credits)
    plot(args.out, phase_data, credits, args.cap)
    manifest = dict(status='PASS', analysis=str(args.analysis.resolve()), summary=str(args.summary.resolve()),
                    audited_matrix_rows=71, learned_runs=6, independent_seeds=list(SEEDS), phase_rows=len(phase_data),
                    capped_credit_rows=len(credits), files=expected[:-1],
                    definitions=dict(raw_emit_fraction='raw emit-positive completions / all completed NN updates in each disjoint retirement phase',
                                     k2_fraction_of_emit='raw K=2 completions / raw emit-positive completions; positive K is produced only on emit by this runtime',
                                     slot0_current_fraction='engine-generated slot0 current-line addresses / all decoded slot0 addresses; selected raw and payload outputs are identical',
                                     slot1_plus2_fraction='raw slot1 predicted cache-line delta +2 / all decoded slot1 addresses',
                                     current_credit_fraction='actual allowed current-line cache calls / all actual allowed calls, at the common ceiling; distinct from generated fraction',
                                     scope='Frozen balanced checkpoints, zero-cost NN; two full future ROIs. No prefix extrapolation, averaged seed replacement, causal per-request claim, or minimum-observation threshold. Undefined denominators leave gaps.'))
    (args.out / 'distribution_plot_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: manifest[k] for k in ('status', 'audited_matrix_rows', 'learned_runs', 'phase_rows', 'capped_credit_rows')}))


if __name__ == '__main__':
    main()
